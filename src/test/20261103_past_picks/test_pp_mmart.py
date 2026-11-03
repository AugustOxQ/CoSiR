"""Tests of pp_mmart (r19 ticket 08): MMArt stage 1 and stage 2 scores of the new three, gates (iv) and (vii), r15 part.

A small fake MMArt world in the layout of the index: the c* exports are written through r15's own `mt_export.write_export`
(OUT/mmart/exports/{val,new}/<enc>/cstar/<aspect>__s<seed>.{npz,json}) with a fake checkpoint file per head; the MMArt
index, labels and feature stores are replaced by dictionaries; the matched probabilities come from a fake `pp_matched`
that honours ticket 05's contract. The statistics are r14's and r15's own functions (the replays in the gate commands
prove them on r15's stored files).
"""
import json
import subprocess
import zlib
import sys
import types

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score

import pp_paths as PP  # first: sets the older rounds' output roots
import pp_register  # noqa: F401  (registers the new three in r14's and r15's tables)
import pp_mmart as M

import ct_score
import mt_data
import mt_export
import mt_paths

ENC = 'clip_b32'
DIM = 16
N_TEST, N_VAL, N_EXTRA = 160, 120, 24
DRAWS = {'clu': 30, 'ret': 60}


def crc(x):
    return zlib.crc32(repr(x).encode())


# ------------------------------------------------------------------------------------------------ the fake world
def _keys(prefix, n):
    return PP.order([f'Style{i % 3}/{prefix}{i:04d}.jpg' for i in range(n)])


class World:
    """Fake MMArt: val (stage 1), test (the cut set), `new` = test plus extra keys (the export and matched sets)."""

    def __init__(self, tmp_path, rng):
        self.val, self.test = _keys('v', N_VAL), _keys('t', N_TEST)
        self.new = PP.order(self.test + _keys('x', N_EXTRA))
        every = self.val + self.new
        self.label = {a: {k: int(rng.integers(-1, 4)) if a == 'emotion' else int(rng.integers(0, 4)) for k in every}
                      for a in PP.ASPECTS}
        # class means: rows carry their label (clusters exist); unlabelled rows are pure noise
        self.mean = {a: rng.normal(size=(4, DIM)) for a in PP.ASPECTS}
        self.feat = {}
        self.x = {}                                                  # (view, key) -> raw row
        self.rng = rng
        self.tmp = tmp_path
        self.checkpoints = {}

    def rows(self, aspect, keys, strength, noise_seed):
        r = np.random.default_rng(noise_seed)
        out = []
        for k in keys:
            y = self.label[aspect][k]
            m = self.mean[aspect][y] * strength if y >= 0 else 0.0
            out.append(m + r.normal(size=DIM))
        x = np.asarray(out, dtype=np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def view(self, view, keys):
        """Frozen image / teacher rows: a fixed random draw per (view, key), carrying the label a little."""
        a = {'emotion_focus': 'emotion', 'style_word': 'style', 'genre_word': 'genre'}.get(view, 'genre')
        strength = 2.0 if view in PP.TEACHER_VIEW.values() else 0.8
        x = np.stack([self._cache(view, k, a, strength) for k in keys])
        return x.astype(np.float32)

    def _cache(self, view, k, a, strength):
        if (view, k) not in self.x:
            r = np.random.default_rng(crc((view, k)))
            y = self.label[a][k]
            m = self.mean[a][y] * strength if y >= 0 else 0.0
            v = m + r.normal(size=DIM)
            self.x[(view, k)] = v / np.linalg.norm(v)
        return self.x[(view, k)]

    def write_exports(self, enc, root):
        for keyset, keys in (('val', self.val), ('new', self.new)):
            for a in PP.ASPECTS:
                for s in PP.SEEDS:
                    ckpt = self.checkpoint(enc, a, s)
                    emb = self.rows(a, keys, 1.5, 1000 * s + crc(a) % 997 + len(keys)).astype(np.float16)
                    mt_export.write_export(root, keyset, enc, a, s, keys, emb, checkpoint=ckpt, view=enc)

    def checkpoint(self, enc, aspect, seed):
        path = mt_export.checkpoint_path(enc, aspect, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(f'fake {enc} {aspect} {seed}'.encode())
        return path

    def probs(self, enc, seed, keys):
        """Fake matched-arm probabilities [n, 40], a fixed draw per (seed, key)."""
        out = []
        for k in keys:
            r = np.random.default_rng(crc((enc, seed, k)))
            y = self.label['genre'][k]
            p = r.uniform(0, 0.2, size=40)
            if y >= 0:
                p[y * 7] += 0.7
            out.append(p)
        return np.asarray(out, dtype=np.float32)


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path, np.random.default_rng(7))
    monkeypatch.setattr(mt_paths, 'OUT', tmp_path / 'mmart')
    monkeypatch.setitem(mt_paths.CSTAR_ROOT, ENC, tmp_path / 'cstar' / ENC)
    monkeypatch.setattr(mt_data, 'keys', lambda name: list({'val': w.val, 'test': w.test, 'new': w.new}[name]))
    monkeypatch.setattr(mt_data, 'labels', lambda keys, a: np.asarray([w.label[a][k] for k in keys], dtype=np.int64))
    monkeypatch.setattr(mt_data, 'features', lambda view, keys: w.view(view, keys))
    w.write_exports(ENC, tmp_path / 'mmart')
    return w


class FakeMatched:
    """Ticket 05's contract: `partitions(set, enc, keys)` -> {seed: {'own': int [1, n], 'r20': int [3, n]}}, k-means
    recomputed on exactly the rows asked for. Records its calls."""

    def __init__(self, world):
        self.world, self.calls = world, []

    def partitions(self, dataset, enc, keys):
        self.calls.append((dataset, enc, list(keys)))
        out = {}
        for s in PP.SEEDS:
            got = ct_score.output_partitions(self.world.probs(enc, s, keys), 'percept')
            out[s] = {'own': np.asarray(got['own'], dtype=np.int32), 'r20': np.asarray(got['r20'], dtype=np.int32)}
        return out


@pytest.fixture
def matched(world, monkeypatch):
    fake = FakeMatched(world)
    mod = types.ModuleType('pp_matched')
    mod.partitions = fake.partitions
    monkeypatch.setitem(sys.modules, 'pp_matched', mod)
    return fake


def ranges_doc(enc, value):
    return {'encoder': enc, 'kind': 'x', 'ranges': {a: value for a in PP.ASPECTS}}


def picks_doc(enc, pick='center+k10'):
    arms = {arm: {'pick': pick, 'table': {}, 'ranges': {}, 'dropped': {}} for arm in ('cstar', 'frozen')}
    return {'encoder': enc, 'kind': 'mmart_retrieval_picks', 'n_val': N_VAL, 'arms': arms}


@pytest.fixture
def block():
    """The `mmart` block of a stage 1 record; the AMI range 0.01 and the mAP range 0.002 differ so a swap shows."""
    return dict(encoder=ENC, clu_ranges=ranges_doc(ENC, 0.01), ret_ranges=ranges_doc(ENC, 0.002),
                picks=picks_doc(ENC))


@pytest.fixture
def teacher():
    arms = {'teacher': {'pick': 'none+k0', 'table': {}, 'ranges': {}, 'dropped': {}}}
    return {'kind': 'mmart_retrieval_picks_teacher', 'ranges_encoder': 'qwen', 'n_val': N_VAL, 'arms': arms}


# ------------------------------------------------------------------------------------------------ SC7, SC9, SC10 (AMI)
def score_clu(enc=ENC, **kw):
    kw.setdefault('n_draws', DRAWS['clu'])
    kw.setdefault('expect_n', None)
    return M.score_clu(enc, **kw)


def test_clustering_cells_carry_r19_labels(world, matched, block, teacher):
    cells = score_clu(block=block)
    assert [(c['sc'], c['aspect']) for c in cells] == [(sc, a) for a in PP.ASPECTS for sc in ('SC7', 'SC9', 'SC10')]
    for c in cells:
        assert c['data'] == 'mmart_test' and c['task'] == 'clu' and c['encoder'] == ENC
        assert c['n_rows'] == N_TEST and c['n_draws'] == DRAWS['clu'] and c['seeds'] == list(PP.SEEDS)
        assert (c['r'] is None) == (c['sc'] != 'SC10')               # LEVEL unit on the teacher cell only
        assert c['lo'] <= c['hi']
    sc10 = [c for c in cells if c['sc'] == 'SC10']
    assert all(c['r'] == 0.01 for c in sc10), 'SC10 uses X-c*\'s own MMArt validation AMI range'
    assert {c['comparator'] for c in cells if c['sc'] == 'SC9'} == {f'matched_{PP.MATCHED[ENC]}'}
    assert {c['comparator'] for c in cells if c['sc'] == 'SC7'} == {f'frozen_{ENC}'}


def test_clustering_scores_mean_ami_of_the_test_rows(world, matched, block, teacher):
    """The frozen image's value is the mean over k-means states of sklearn's AMI on the labelled test rows, rows in sha
    order (an independent computation; r14's array-level statistics are replayed by the gate commands)."""
    cells = score_clu(block=block)
    cell = next(c for c in cells if c['sc'] == 'SC7' and c['aspect'] == 'genre')
    y = np.asarray([world.label['genre'][k] for k in world.test])
    parts = ct_score.output_partitions(world.view(ENC, world.test), 'k1')['r20']
    want = np.mean([adjusted_mutual_info_score(y[y >= 0], p[y >= 0]) for p in parts])
    assert cell['comparator_value'] == pytest.approx(want, abs=1e-9)
    assert cell['n_labelled'] == int((y >= 0).sum())


# ------------------------------------------------------------------------------------------------ the matched arm
def test_matched_partitions_are_recomputed_on_exactly_the_test_rows(world, matched, block, teacher):
    cells = score_clu(block=block)
    assert matched.calls, 'SC9 never asked pp_matched'
    for dataset, enc, keys in matched.calls:
        assert (dataset, enc) == ('mmart_new', ENC)
        assert keys == world.test, 'partitions must be computed on the test keys, in sha256 order, not on `new`'
    cell = next(c for c in cells if c['sc'] == 'SC9' and c['aspect'] == 'genre')
    y = np.asarray([world.label['genre'][k] for k in world.test])
    own, r20 = [], []
    for s in PP.SEEDS:                                   # k-means on the test rows alone, not a slice of the `new` fit
        got = ct_score.output_partitions(world.probs(ENC, s, world.test), 'percept')
        ok = y >= 0
        own.append(adjusted_mutual_info_score(y[ok], got['own'][0][ok]))
        r20.append(np.mean([adjusted_mutual_info_score(y[ok], p[ok]) for p in got['r20']]))
    assert cell['comparator_reading'] in ('M3-own', 'M3-r20')
    want = max(np.mean(own), np.mean(r20))
    assert cell['comparator_value'] == pytest.approx(want, abs=1e-9)


def test_a_matched_slice_of_the_wrong_length_is_refused(world, matched, block, teacher, monkeypatch):
    real = matched.partitions

    def cut_short(dataset, enc, keys):
        got = real(dataset, enc, keys)
        return {s: {'own': v['own'][:, :-1], 'r20': v['r20'][:, :-1]} for s, v in got.items()}
    sys.modules['pp_matched'].partitions = cut_short
    with pytest.raises(AssertionError):
        score_clu(block=block)


# ------------------------------------------------------------------------------------------------ committed records
@pytest.fixture
def git_records(tmp_path, monkeypatch):
    """A scratch git repo as the records' home: PP.read_committed_record looks at its HEAD."""
    repo = tmp_path / 'repo'
    (repo / 'records').mkdir(parents=True)
    subprocess.run(['git', '-C', str(repo), 'init', '-q'], check=True)
    for kv in (('user.email', 'a@b'), ('user.name', 'n')):
        subprocess.run(['git', '-C', str(repo), 'config', *kv], check=True)
    monkeypatch.setattr(PP, 'REPO', repo)
    monkeypatch.setattr(PP, 'RECORDS', repo / 'records')
    monkeypatch.setattr(PP, 'SMOKE', 0)
    return repo


def forbid_reads(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('a test row was read before the stage 1 record was checked')
    monkeypatch.setattr(mt_export, 'load', boom)
    monkeypatch.setattr(mt_data, 'features', boom)
    monkeypatch.setattr(mt_data, 'labels', boom)


def test_scoring_refuses_a_record_that_is_not_committed(world, matched, block, teacher, git_records, monkeypatch):
    record = git_records / 'records' / f'stage1_{ENC}.json'
    record.write_text(json.dumps({'mmart': block}))
    forbid_reads(monkeypatch)
    with pytest.raises(RuntimeError, match='not committed'):
        M.score_clu(ENC, n_draws=10, expect_n=None)
    with pytest.raises(RuntimeError, match='not committed'):
        M.score_ret(ENC, teacher=teacher, n_draws=10, expect_n=None)


def test_scoring_reads_a_committed_record_and_refuses_an_edited_one(world, matched, block, teacher, git_records):
    record = git_records / 'records' / f'stage1_{ENC}.json'
    record.write_text(json.dumps({'mmart': block}))
    subprocess.run(['git', '-C', str(git_records), 'add', '-f', 'records'], check=True)
    subprocess.run(['git', '-C', str(git_records), 'commit', '-qm', 'stage 1'], check=True)
    cells = score_clu()
    assert len(cells) == 9 and all(c['r'] == 0.01 for c in cells if c['sc'] == 'SC10')
    edited = dict(block, clu_ranges=ranges_doc(ENC, 0.5))
    record.write_text(json.dumps({'mmart': edited}))
    with pytest.raises(RuntimeError, match='not committed'):
        score_clu()


def test_a_record_of_another_encoder_is_refused(world, matched, block, teacher):
    wrong = dict(block, clu_ranges=ranges_doc('clip_b16', 0.01))
    with pytest.raises(AssertionError):
        score_clu(block=wrong)
