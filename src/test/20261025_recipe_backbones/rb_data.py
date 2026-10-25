"""P, the 29,232 fresh development paintings, with their features and labels; the old 5,000 and fit/stop roles.

P = the full split's train and validation names minus the frozen sample's 19,889 paintings, in sorted() order
(rule §3.1). Features come from the claim test's own loaders (`ct2_data.default_images`, `default_captions`,
`instructed_images`) called on `split2('full')`, the same call stage 2's D2 training made (`ct2_d2.py`), so the
rows come from the same stores; P's rows are then picked by painting name (C6). All feature arrays are float32
with unit norm (|norm - 1| < 1e-3), rows in P order.

Labels: `pbp_graphs.label_vectors`, the helper `ab_score.py` calls (`label_vectors(s, nodes)`): emotion = plurality of
the painting's caption emotion codes ("something else" is code 8; ties to the smallest code; -1 if none), style and
genre = the first caption's label (`apc_data.paint_first`), -1 masked. P's paintings are outside the frozen sample,
so the captions are the full split's (`split2('full')`: ascending extraction ID within a painting) and not the
sample's order. The order cannot matter: style comes from the painting's own annotation field and genre from the
WikiArt table by painting name, both constant over a painting's captions (`check_constant_per_painting` asserts
it for every painting), and the plurality does not depend on order. `p_labels` equals `ct_held.painting_labels`,
which uses the annotation order (a slow test checks it).

Old 5,000 and fit/stop: roles of the frozen sample (`itp_data.split`: 13,400 fit, 1,489 stop; the 5,000 in
`subset_pid` order), loaded with the same CT loaders on `split2('sample')`.
"""
import argparse
import functools
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import rb_paths
import ct2_data
import itp_data
import pbp_graphs
from apc_data import paint_first

INSTRUCTED_KEYS = ('emotion_focus', 'style_word', 'genre_word')
P_COUNTS = {'train': 23080, 'val': 6152}
N_P = 29232
N_SAMPLE = 19889
WIDTH = 2048
NORM_TOL = 1e-3
ROLES = ('old5000', 'fit', 'stop')


# ------------------------------------------------------------------ the split
def names_sha256(names):
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


def p_split(train, val, sample_names):
    """Sorted train and validation names outside the sample, with the per-role counts."""
    train, val, sample = list(train), list(val), list(sample_names)
    for group in (train, val, sample):
        assert len(set(group)) == len(group), 'duplicate painting names'
    assert not set(train) & set(val), 'train and validation overlap'
    assert set(sample) <= set(train) | set(val), 'sample outside train and validation'
    drop = set(sample)
    train_p = [n for n in train if n not in drop]
    val_p = [n for n in val if n not in drop]
    return sorted(train_p + val_p), {'train': len(train_p), 'val': len(val_p)}


def select_by_name(store_names, x, wanted):
    """Rows of x for the wanted names, in the wanted order; the join is the name, never a position."""
    store_names, wanted = list(store_names), list(wanted)
    assert x.shape[0] == len(store_names) and x.dtype == np.float32
    assert len(set(store_names)) == len(store_names), 'duplicate store names'
    assert len(set(wanted)) == len(wanted), 'duplicate wanted names'
    lookup = {name: row for row, name in enumerate(store_names)}
    assert all(name in lookup for name in wanted), 'wanted name missing from the store'
    return x[np.asarray([lookup[name] for name in wanted], dtype=np.int64)]


def check_unit(x, n):
    assert x.dtype == np.float32 and x.ndim == 2 and x.shape[0] == n, (x.dtype, x.shape, n)
    assert np.isfinite(x).all(), 'nonfinite features'
    assert np.abs(np.linalg.norm(x, axis=1) - 1).max() < NORM_TOL, 'rows not unit norm'


def write_p_json(outputs, names, counts):
    """Write part1/P.json; an existing file must be identical (P is fixed before any export)."""
    names = list(names)
    assert names == sorted(names) and len(set(names)) == len(names)
    assert sum(counts.values()) == len(names)
    value = {'names': names, 'sha256': names_sha256(names), 'counts': dict(counts)}
    path = Path(outputs) / 'part1' / 'P.json'
    if path.exists():
        assert json.loads(path.read_text()) == value, f'{path} differs from the P being written'
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('P.pending.json')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)
    return path


def read_p_json(outputs, expect=None):
    """Read part1/P.json and verify keys, order, count and hash; `expect` is the names it must hold."""
    value = json.loads((Path(outputs) / 'part1' / 'P.json').read_text())
    assert set(value) == {'names', 'sha256', 'counts'}
    names = value['names']
    assert names == sorted(names) and len(set(names)) == len(names)
    assert sum(value['counts'].values()) == len(names)
    assert names_sha256(names) == value['sha256'], 'P.json hash mismatch'
    if expect is not None:
        assert names == list(expect), 'P.json differs from the expected names'
    return names


# ------------------------------------------------------------------ labels
def check_constant_per_painting(s):
    """Style and genre are one value per painting: every caption equals the painting's first caption's."""
    count = len(s['first'])
    for key in ('sty', 'gen'):
        first = paint_first(s[key], s['pid'], count)
        assert (first != -2).all(), f'{key}: a painting without captions'
        assert np.array_equal(s[key], first[s['pid']]), f'{key} varies within a painting'


def labels_by_name(s, names):
    """The three labels, in the order of `names`, by `pbp_graphs.label_vectors` (the helper `ab_score.py` calls)."""
    lookup = {name: row for row, name in enumerate(s['all_painting_ids'])}
    assert all(name in lookup for name in names), 'painting missing from the split'
    nodes = np.asarray([lookup[name] for name in names], dtype=np.int64)
    out = pbp_graphs.label_vectors(s, nodes)
    out = {a: np.asarray(out[a], dtype=np.int64) for a in rb_paths.RB_ASPECTS}
    assert all(v.shape == (len(names),) and v.min() >= -1 for v in out.values())
    return out


def count_labelled(labels, aspect):
    return int((labels[aspect] >= 0).sum())


# ------------------------------------------------------------------ the real data
@functools.lru_cache(maxsize=1)
def _full():
    """The CT's full split (`ct2_data.split2('full')` asserts the CT's train, val and held name hashes)."""
    s = ct2_data.split2('full')
    assert s['mode'] == 'full' and len(s['all_painting_ids']) == 42969 + 6152
    return s


@functools.lru_cache(maxsize=1)
def _sample():
    assert ct2_data.SAMPLE == rb_paths.SAMPLE
    s = ct2_data.split2('sample')
    assert len(s['first']) == N_SAMPLE and len(s['subset_pid']) == 5000
    return s


def sample_names():
    return [str(n) for n in _sample()['all_painting_ids']]


@functools.lru_cache(maxsize=1)
def _p():
    full = _full()
    names_all = [str(n) for n in full['all_painting_ids']]
    train, val = names_all[:ct2_data.FULL_COUNTS[0][0]], names_all[ct2_data.FULL_COUNTS[0][0]:]
    assert len(val) == ct2_data.FULL_COUNTS[1][0]
    assert ct2_data._names_sha256(train) == ct2_data.FULL_NAMES_SHA256[0]
    assert ct2_data._names_sha256(val) == ct2_data.FULL_NAMES_SHA256[1]
    names, counts = p_split(train, val, sample_names())
    assert len(names) == N_P and counts == P_COUNTS, (len(names), counts)
    return names, counts


def p_names():
    return list(_p()[0])


def p_counts():
    return dict(_p()[1])


def p_sha256():
    return names_sha256(_p()[0])


def _p_rows(loader, *args):
    """A CT loader's full-split rows (its `all_painting_ids` order), picked for P by name."""
    full = _full()
    x = select_by_name(full['all_painting_ids'], loader(full, *args), _p()[0])
    check_unit(x, N_P)
    return x


def p_default_images():
    return _p_rows(ct2_data.default_images)


def p_default_captions():
    """Default caption mean per painting (float64 accumulation, then float32 and L2: the CT's)."""
    return _p_rows(ct2_data.default_captions)


def p_instructed_images(key):
    assert key in INSTRUCTED_KEYS, key
    return _p_rows(ct2_data.instructed_images, key)


def p_labels():
    """{emotion, style, genre}: int64 [29232], -1 masked, in P order."""
    full = _full()
    check_constant_per_painting(full)
    return labels_by_name(full, _p()[0])


def genre_label_count():
    """Number of P paintings with a WikiArt genre label (stage 0 reports it)."""
    return count_labelled(p_labels(), 'genre')


# ------------------------------------------------------------------ old 5,000 and fit/stop
def sample_roles():
    """Node indices into the sample's painting order: old5000 (`subset_pid` order), fit (13,400), stop (1,489)."""
    s = _sample()
    roles = {'old5000': np.asarray(s['subset_pid'], dtype=np.int64),
             'fit': np.asarray(s['fit'], dtype=np.int64), 'stop': np.asarray(s['dev'], dtype=np.int64)}
    assert [len(roles[r]) for r in ROLES] == [5000, 13400, 1489]
    union = np.concatenate([roles[r] for r in ROLES])
    assert len(np.unique(union)) == len(union) == N_SAMPLE, 'roles do not partition the sample'
    return roles


def role_names(role):
    assert role in ROLES, role
    names = sample_names()
    return [names[i] for i in sample_roles()[role]]


def old5000_names():
    return role_names('old5000')


def fit_names():
    return role_names('fit')


def stop_names():
    return role_names('stop')


def sample_default_images():
    return ct2_data.default_images(_sample())


def sample_default_captions():
    return ct2_data.default_captions(_sample())


def sample_instructed_images(key):
    return ct2_data.instructed_images(_sample(), key)


def sample_instructed_captions(key):
    return ct2_data.instructed_captions(_sample(), key)


def select_role(role, x):
    """Rows of a `sample_*` array (sample painting order) for a role, picked by name."""
    return select_by_name(sample_names(), x, role_names(role))


def old5000_default_images():
    return select_role('old5000', sample_default_images())


def old5000_default_captions():
    return select_role('old5000', sample_default_captions())


def old5000_instructed_images(key):
    return select_role('old5000', sample_instructed_images(key))


def old5000_instructed_captions(key):
    return select_role('old5000', sample_instructed_captions(key))


def old5000_labels():
    return labels_by_name(_sample(), old5000_names())


# ------------------------------------------------------------------ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description='Write part1/P.json and report the label counts.')
    ap.add_argument('--outputs', type=Path, default=rb_paths.OUTPUTS)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args(argv)
    if args.smoke:
        assert str(args.outputs.resolve()).startswith('/tmp/claude-0/'), 'smoke needs an outputs path under /tmp/claude-0/'
    rb_paths.check_imports()
    path = write_p_json(args.outputs, p_names(), p_counts())
    read_p_json(args.outputs, expect=p_names())
    labels = p_labels()
    print(f'{path}: {N_P} paintings, sha256 {p_sha256()}')
    print({a: count_labelled(labels, a) for a in rb_paths.RB_ASPECTS}, 'genre_label_count', genre_label_count())


if __name__ == '__main__':
    sys.exit(main())
