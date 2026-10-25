"""Part 1 reported readings (ticket 04; spec §3.4 "Beside", rule §4.1 last line): they decide nothing.

Per recipe and aspect, beside the pick: image AMI on the old 5,000; caption-side AMI on P (none for R7, whose
trainer scores no caption side); a Leiden reading of the image side on P; the epoch-4 image exports on P; best
epochs. Per reference (the default and the three instructed Qwen images): AMI on P. All AMI are 12-22's statistic
through `rb_ami`: point estimates, the mean over the k-means states 0 to 2, labelled positions only (clustering
first covers every row, as in the pick). Leiden is 12-22's (`ct_diag.union_knn` 20-NN cosine graph, ordinary
modularity at resolution 1, seeds 0 to 2, AMI on labelled positions). Writes `part1/reported.json` and `.md`.

Needs `part1/pick.json` (read only, never written) and the exports the pick read; refuses to run without the pick.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami

import rb_paths
import rb_ami
import rb_data
import rb_pick
import rb_rows
import ct_diag

ASPECTS = rb_paths.RB_ASPECTS
SEED = rb_pick.SEED
N_P = rb_pick.N_P
N_OLD = 5000
REFERENCE_KEYS = ('default',) + rb_data.INSTRUCTED_KEYS
_CODE = ('rb_ami.py', 'rb_part1_report.py', 'rb_pick.py', 'rb_rows.py')
DECIMALS = 4


# ------------------------------------------------------------------ inputs
def read_pick(outputs, smoke=False):
    """part1/pick.json, read only; a real report needs a real pick."""
    path = Path(outputs) / 'part1' / 'pick.json'
    assert path.is_file(), f'{path} missing: run rb_pick.py first; the reported readings come after the pick'
    pick = json.loads(path.read_text())
    assert smoke or pick['smoke'] is False, f'{path} is a smoke pick'
    assert set(pick['recipes']) == set(rb_rows.ORDER), sorted(pick['recipes'])
    return pick


def record_of(outputs, recipe, aspect):
    path = Path(outputs) / 'part1' / 'train' / recipe / f'{aspect}__s{SEED}.json'
    record = json.loads(path.read_text())
    assert record['recipe'] == recipe and record['aspect'] == aspect and record['seed'] == SEED, path
    return record


def load_export(outputs, recipe, aspect, label, tag, record, n_rows, p_sha):
    """{img, cap} float16 of `<aspect>__s0__<label><tag>.npz` after the trainer record's shape, dtype and hash checks."""
    assert label in ('old5000', 'P') and tag in ('', '__e4')
    name = f'{aspect}__s{SEED}__{label}{tag}.npz'
    if label == 'P':
        assert record['p_exports'] is True and record['p_sha256'] == p_sha, f'{recipe}/{aspect}: P sha256 differs'
    with np.load(Path(outputs) / 'part1' / 'train' / recipe / name, allow_pickle=False) as z:
        assert sorted(z.files) == ['cap', 'img'], (name, z.files)
        arrays = {k: z[k] for k in z.files}
    listed = record['files'][name]
    for key, x in arrays.items():
        want = listed[key]
        assert list(x.shape) == want['shape'] == [n_rows, rb_pick.WIDTH] and str(x.dtype) == want['dtype'] == 'float16', \
            f'{recipe}/{name}: {key} shape/dtype {x.shape} {x.dtype}'
        assert rb_pick.array_sha256(x) == want['sha256'], f'{recipe}/{name}: {key} sha256 differs from the trainer record'
    return arrays


# ------------------------------------------------------------------ readings
def ami_point(y, parts):
    """The mean over the three partitions of AMI against `y` on the labelled rows (rb_ami.score)."""
    assert y.shape == (parts.shape[1],)
    return rb_ami.score(y, parts, np.flatnonzero(y >= 0))


def kmeans_ami(x, y):
    """A for vectors `x` against labels `y` (-1 masked); k-means covers every row first."""
    return ami_point(y, rb_ami.partitions(x))


def leiden_ami(x, y):
    """12-22's Leiden reading: union 20-NN cosine graph, resolution 1, seeds 0 to 2, AMI on labelled positions."""
    x = rb_ami.unit_float32(x)
    parts = ct_diag.leiden_partitions(ct_diag.union_knn(x, 20))
    assert parts.shape == (3, len(y))
    mask = y >= 0
    seeds = [float(ami(y[mask], p[mask])) for p in parts]
    return dict(mean=float(np.mean(seeds)), seeds=seeds)


def check_labels(labels, n, name):
    assert set(labels) == set(ASPECTS), name
    for a in ASPECTS:
        assert labels[a].dtype == np.int64 and labels[a].shape == (n,) and labels[a].min() >= -1, (name, a)
        assert (labels[a] >= 0).any(), (name, a)


def code_sha256():
    return rb_pick.code_sha256(_CODE)


def run(outputs, p_labels, old_labels, references, *, smoke=False, n_rows=N_P, n_old=N_OLD, expect_names=None,
        recipes=None):
    """Compute the readings and write `part1/reported.json` and `.md`; returns the record.

    `p_labels` (P order) and `old_labels` (the old 5,000 in `subset_pid` order) are {aspect: int64 [n]}, -1 masked;
    `references` = {key: callable returning float32 [n_rows, 2048] in P order}; `recipes` limits the recipes (smoke).
    """
    outputs = Path(outputs)
    pick = read_pick(outputs, smoke)                  # first: nothing is read or computed without the pick
    names = rb_data.read_p_json(outputs, expect=expect_names)
    assert len(names) == n_rows
    p_sha = rb_data.names_sha256(names)
    assert pick['p_sha256'] == p_sha and pick['n_rows'] == n_rows, 'pick.json is for another P'
    check_labels(p_labels, n_rows, 'P')
    check_labels(old_labels, n_old, 'old 5,000')
    for a in ASPECTS:
        assert rb_pick.array_sha256(p_labels[a]) == pick['label_sha256'][a], f'{a}: P labels differ from the pick\'s'
    assert set(references) == set(REFERENCE_KEYS) or smoke, sorted(references)
    scored = [n for n in rb_rows.ORDER if pick['recipes'][n] is not None and (recipes is None or n in recipes)]
    assert rb_rows.REFERENCE in scored
    out = dict(old5000={}, p_caption={}, p_leiden={}, p_e4={}, best_epochs={}, p_image_pick={})
    for name in scored:
        for key in out:
            out[key][name] = {}
        for a in ASPECTS:
            record = record_of(outputs, name, a)
            assert (not record['caption_scored']) or name not in rb_rows.IMAGE_ONLY, f'{name}: image-only with a caption side'
            out['best_epochs'][name][a] = dict(best_epoch=record['best_epoch'], epochs_run=record['epochs_run'],
                                               snapshot_epoch=record['snapshot_epoch'])
            old = load_export(outputs, name, a, 'old5000', '', record, n_old, p_sha)
            out['old5000'][name][a] = kmeans_ami(old['img'], old_labels[a])
            best = load_export(outputs, name, a, 'P', '', record, n_rows, p_sha)
            # the pick's own A, recomputed from the saved fixed partitions: a cross-check of the files read here
            parts = rb_pick.fixed_partitions(outputs, name, a, best['img'], p_sha)
            out['p_image_pick'][name][a] = ami_point(p_labels[a], parts)
            assert abs(out['p_image_pick'][name][a] - pick['recipes'][name][a]['A']) < 1e-9, f'{name}/{a}: A differs from the pick'
            out['p_caption'][name][a] = kmeans_ami(best['cap'], p_labels[a]) if record['caption_scored'] else 'not scored'
            out['p_leiden'][name][a] = leiden_ami(best['img'], p_labels[a])
            e4 = load_export(outputs, name, a, 'P', '__e4', record, n_rows, p_sha)
            out['p_e4'][name][a] = kmeans_ami(e4['img'], p_labels[a])
            print(f'{name}/{a}: done', flush=True)
    out['references_p'] = {}
    for key, load in references.items():
        x = load()
        assert x.shape == (n_rows, rb_pick.WIDTH), (key, x.shape)
        out['references_p'][key] = {a: kmeans_ami(x, p_labels[a]) for a in ASPECTS}
        print(f'reference {key}: done', flush=True)
    record = dict(p_sha256=p_sha, n_rows=n_rows, n_old=n_old, smoke=bool(smoke), versions=rb_ami.versions(),
                  pick=dict(outcome=pick['outcome'], recipe=pick['recipe'], sha256=hashlib.sha256((outputs / 'part1' / 'pick.json').read_bytes()).hexdigest()),
                  recipes=scored, label_counts_p={a: int((p_labels[a] >= 0).sum()) for a in ASPECTS},
                  label_counts_old5000={a: int((old_labels[a] >= 0).sum()) for a in ASPECTS},
                  leiden=dict(graph='union 20-NN cosine', resolution=1.0, seeds=[0, 1, 2]),
                  code_sha256=code_sha256(), **out)
    record = json.loads(json.dumps(record, allow_nan=False))
    write_outputs(outputs, record)
    return record


# ------------------------------------------------------------------ files
def _num(v):
    return v if isinstance(v, str) else f'{v:.{DECIMALS}f}'


def _table(title, data, note=''):
    lines = [f'## {title}', '', note, '' if note else None, '| row | ' + ' | '.join(ASPECTS) + ' |',
             '|---|' + '---|' * len(ASPECTS)]
    for row, by_aspect in data.items():
        lines.append(f'| {row} | ' + ' | '.join(_num(by_aspect[a]) for a in ASPECTS) + ' |')
    return [x for x in lines if x is not None] + ['']


def markdown(r):
    leiden = {n: {a: v['mean'] for a, v in by.items()} for n, by in r['p_leiden'].items()}
    epochs = {n: {a: f"{v['best_epoch']} of {v['epochs_run']}" for a, v in by.items()} for n, by in r['best_epochs'].items()}
    lines = ['# Part 1: reported readings (decide nothing)', '',
             f"P sha256 {r['p_sha256']}; pick {r['pick']['outcome']} (recipe {r['pick']['recipe']})"
             f"{'; SMOKE RUN' if r['smoke'] else ''}. Point estimates, seed 0, mean over k-means states 0 to 2; "
             f'labelled paintings only. P labelled: {r["label_counts_p"]}; old 5,000 labelled: {r["label_counts_old5000"]}.', '']
    lines += _table('Image AMI on P, best epoch (the pick\'s A, recomputed)', r['p_image_pick'])
    lines += _table('Image AMI on the old 5,000', r['old5000'])
    lines += _table('Caption-side AMI on P', r['p_caption'], 'R7 recipes have no caption network: not scored.')
    lines += _table('Leiden AMI on P, image side', leiden, 'Union 20-NN cosine graph, resolution 1.0, mean of seeds 0 to 2.')
    lines += _table('Image AMI on P, epoch-4 export', r['p_e4'])
    lines += _table('Best epoch (of epochs run)', epochs, 'Epoch-4 exports are the fixed snapshot of the trainer.')
    lines += _table('References on P (Qwen images, no training)', r['references_p'])
    return '\n'.join(lines)


def write_outputs(outputs, record):
    """Write reported.json and .md; existing files must hold the same content (nothing is overwritten)."""
    folder = Path(outputs) / 'part1'
    text_json = json.dumps(record, indent=2, allow_nan=False) + '\n'
    text_md = markdown(record) + '\n'
    for name, text in (('reported.json', text_json), ('reported.md', text_md)):
        path = folder / name
        if path.exists():
            assert path.read_text() == text, f'{path} exists with other results; move it aside'
        else:
            pending = path.with_name('pending_' + name)
            pending.write_text(text)
            pending.replace(path)


# ------------------------------------------------------------------ command line
def main(argv=None):
    parser = argparse.ArgumentParser(description='Part 1 reported readings (after the pick; decide nothing).')
    parser.add_argument('--outputs', type=Path, default=rb_paths.OUTPUTS)
    parser.add_argument('--smoke', action='store_true',
                        help='two recipes (R0-5 and R0-3), the default reference only; outputs under /tmp/claude-0/')
    args = parser.parse_args(argv)
    if args.smoke:
        assert str(args.outputs.resolve()).startswith('/tmp/claude-0/'), 'smoke needs --outputs under /tmp/claude-0/'
    rb_paths.check_imports()
    read_pick(args.outputs, args.smoke)               # refuse early, before the slow loaders
    loaders = dict(default=rb_data.p_default_images,
                   **{k: (lambda k=k: rb_data.p_instructed_images(k)) for k in rb_data.INSTRUCTED_KEYS})
    if args.smoke:
        loaders = {'default': loaders['default']}
    names = rb_data.p_names()
    record = run(args.outputs, rb_data.p_labels(), rb_data.old5000_labels(), loaders, smoke=args.smoke,
                 expect_names=names, recipes=rb_rows.ORDER[:2] if args.smoke else None)
    print(f"{args.outputs / 'part1' / 'reported.json'}: {len(record['recipes'])} recipes", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
