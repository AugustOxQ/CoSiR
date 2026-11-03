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
import mt_ret

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


def picks_doc(enc):
    # Different non-identity maps/expansions make arm swaps and fitting on test rows observable.
    arms = {arm: {'pick': pick, 'table': {}, 'ranges': {}, 'dropped': {}}
            for arm, pick in (('cstar', 'center+k10'), ('frozen', 'whiten16+k5'))}
    return {'encoder': enc, 'kind': 'mmart_retrieval_picks', 'n_val': N_VAL, 'arms': arms}


@pytest.fixture
def block():
    """The `mmart` block of a stage 1 record; the AMI range 0.01 and the mAP range 0.002 differ so a swap shows."""
    return dict(encoder=ENC, clu_ranges=ranges_doc(ENC, 0.01), ret_ranges=ranges_doc(ENC, 0.002),
                picks=picks_doc(ENC))


@pytest.fixture
def teacher():
    arms = {'teacher': {'pick': 'center+k3', 'table': {}, 'ranges': {}, 'dropped': {}}}
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
    """Independently score every aspect and every c* seed, using sklearn AMI on labelled test rows."""
    cells = score_clu(block=block)
    frozen = ct_score.output_partitions(world.view(ENC, world.test), 'k1')['r20']
    for a in PP.ASPECTS:
        y = np.asarray([world.label[a][k] for k in world.test])
        ok = y >= 0

        def mean_ami(parts):
            return np.mean([adjusted_mutual_info_score(y[ok], p[ok]) for p in parts])

        cstar = np.mean([mean_ami(ct_score.output_partitions(
            mt_export.load('new', ENC, a, s, world.test).astype(np.float32), 'k1')['r20']) for s in PP.SEEDS])
        teach = mean_ami(ct_score.output_partitions(world.view(PP.TEACHER_VIEW[a], world.test), 'k1')['r20'])
        for cell in (c for c in cells if c['aspect'] == a):
            assert cell['candidate_value'] == pytest.approx(cstar, rel=0, abs=1e-9)
            assert cell['n_labelled'] == int(ok.sum())
            if cell['sc'] in ('SC7', 'SC10'):
                want = mean_ami(frozen) if cell['sc'] == 'SC7' else teach
                assert cell['comparator_value'] == pytest.approx(want, rel=0, abs=1e-9)


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


# ------------------------------------------------------------------------------------------------ SC8, SC10 (mAP)
import pp_outcome  # noqa: E402


def unit(x):
    x = np.asarray(x).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture
def ret_block(block):
    """Each arm has its own non-identity pick; these are deliberately different from r15's common pick."""
    return block


def test_retrieval_cells_carry_r19_labels_and_the_mAP_range(world, ret_block, teacher):
    cells = M.score_ret(ENC, block=ret_block, teacher=teacher, n_draws=DRAWS['ret'], expect_n=None)
    assert [(c['sc'], c['aspect']) for c in cells] == [(sc, a) for a in PP.ASPECTS for sc in ('SC8', 'SC10')]
    for c in cells:
        pp_outcome.check_cell(c, smoke=True)                          # ticket 11 reads cells strictly
        assert c['data'] == 'mmart_test' and c['task'] == 'ret' and c['n_rows'] == N_TEST
        assert c['n_draws'] == DRAWS['ret']
    assert all(c['r'] == 0.002 for c in cells if c['sc'] == 'SC10'), 'the mAP range, not the AMI range'
    assert {c['comparator'] for c in cells if c['sc'] == 'SC8'} == {f'frozen_{ENC}'}


def test_retrieval_values_are_the_test_rows_alone(world, ret_block, teacher):
    """Each picked map fits all validation rows, then scores only the labelled test queries/gallery."""
    cells = M.score_ret(ENC, block=ret_block, teacher=teacher, n_draws=DRAWS['ret'], expect_n=None)
    picks = {arm: doc['pick'] for arm, doc in ret_block['picks']['arms'].items()}
    picks['teacher'] = teacher['arms']['teacher']['pick']
    assert len(set(picks.values())) == 3 and all(not p.startswith('none+') for p in picks.values())
    for a in PP.ASPECTS:
        y = np.asarray([world.label[a][k] for k in world.test], dtype=np.int64)
        frozen = mt_ret.arm_ap(lambda s: unit(world.view(ENC, world.val)),
                               lambda s: unit(world.view(ENC, world.test)), picks['frozen'], y, False, 2)
        tview = PP.TEACHER_VIEW[a]
        teach = mt_ret.arm_ap(lambda s: unit(world.view(tview, world.val)),
                              lambda s: unit(world.view(tview, world.test)), picks['teacher'], y, False, 2)
        cstar = mt_ret.arm_ap(lambda s: unit(mt_export.load('val', ENC, a, s, world.val)),
                              lambda s: unit(mt_export.load('new', ENC, a, s, world.test)),
                              picks['cstar'], y, True, 2)
        sc8 = next(c for c in cells if c['sc'] == 'SC8' and c['aspect'] == a)
        sc10 = next(c for c in cells if c['sc'] == 'SC10' and c['aspect'] == a)
        for cell, comparator, arm in ((sc8, frozen, 'frozen'), (sc10, teach, 'teacher')):
            assert cell['candidate_value'] == pytest.approx(cstar['point'], rel=0, abs=1e-9)
            assert cell['comparator_value'] == pytest.approx(comparator['point'], rel=0, abs=1e-9)
            assert cell['candidate_readout'] == picks['cstar']
            assert cell['comparator_readout'] == picks[arm]
            assert cell['n_queries'] == len(cstar['rows']) == len(comparator['rows'])


def test_the_teacher_must_be_picked_with_qwens_ranges(world, ret_block, teacher):
    with pytest.raises(AssertionError):
        M.score_ret(ENC, block=ret_block, teacher=dict(teacher, ranges_encoder=ENC), n_draws=10, expect_n=None)


def test_clustering_cells_pass_ticket_11s_cell_check(world, matched, block):
    for c in score_clu(block=block):
        pp_outcome.check_cell(c, smoke=True)


def test_the_test_set_size_is_asserted(world, matched, block):
    with pytest.raises(AssertionError):
        M.score_clu(ENC, block=block, n_draws=10)                     # default expect_n = 16,384


# ------------------------------------------------------------------------------------------------ stage 1
@pytest.fixture
def spy(world, monkeypatch):
    seen = {'keysets': set(), 'keys': set()}
    real_load, real_features, real_labels = mt_export.load, mt_data.features, mt_data.labels

    def load(keyset, *a, **k):
        seen['keysets'].add(keyset)
        return real_load(keyset, *a, **k)

    def features(view, keys):
        seen['keys'] |= set(keys)
        return real_features(view, keys)

    def labels(keys, a):
        seen['keys'] |= set(keys)
        return real_labels(keys, a)
    monkeypatch.setattr(mt_export, 'load', load)
    monkeypatch.setattr(mt_data, 'features', features)
    monkeypatch.setattr(mt_data, 'labels', labels)
    return seen


def test_stage1_reads_validation_only_and_has_r15s_record_form(world, spy, monkeypatch, tmp_path):
    monkeypatch.setattr(PP, 'OUT', tmp_path / 'out')
    block = M.stage1_block(ENC, expect_n=None)
    assert spy['keysets'] == {'val'}, 'stage 1 may read the validation exports only'
    assert spy['keys'] <= set(world.val), 'stage 1 touched a key outside MMArt validation'
    assert set(block) == {'encoder', 'clu_ranges', 'ret_ranges', 'picks'} and block['encoder'] == ENC
    assert block['clu_ranges']['kind'] == 'mmart_clustering_validation_ranges'
    assert block['ret_ranges']['kind'] == 'mmart_retrieval_validation_ranges'
    assert set(block['picks']['arms']) == {'cstar', 'frozen'}
    for part in ('clu_ranges', 'ret_ranges'):
        doc = block[part]
        for a in PP.ASPECTS:                                         # range = max - min over the five seeds, unrounded
            per = [doc['per_seed'][a][str(s)] for s in PP.SEEDS]
            assert doc['ranges'][a] == pytest.approx(max(per) - min(per), abs=1e-12) and doc['ranges'][a] > 0
    for arm in ('cstar', 'frozen'):                                  # the retrieval ranges feed both picks (D12, D21)
        assert block['picks']['arms'][arm]['ranges'] == block['ret_ranges']['ranges']
        assert mt_ret.readout_of(block['picks']['arms'][arm]['pick'], DIM)
    assert M.jsonable(block) == block
    path = M.write_stage1(ENC, block)
    assert path == tmp_path / 'out' / 'stage1' / f'mmart_{ENC}.json' and json.loads(path.read_text()) == block


def test_stage1_files_are_written_for_the_new_three_only(world, monkeypatch, tmp_path):
    monkeypatch.setattr(PP, 'OUT', tmp_path / 'out')
    with pytest.raises(AssertionError):
        M.write_stage1('clip_l14', {'encoder': 'clip_l14'})


_STAGE1 = {}


@pytest.fixture
def cached_stage1(world, monkeypatch):
    """`stage1_block` is deterministic on the fake world and slow: compute it once per session (the test above runs it
    for real, under a spy) and hand copies to the tests that only consume it."""
    real = M.stage1_block

    def cached(enc, workers=8, expect_n=None):
        if enc not in _STAGE1:
            _STAGE1[enc] = real(enc, workers, None)
        return json.loads(json.dumps(_STAGE1[enc]))
    monkeypatch.setattr(M, 'stage1_block', cached)


def test_a_stage1_block_scores_straight_away(world, matched, cached_stage1, teacher):
    block = M.stage1_block(ENC, expect_n=None)
    assert len(score_clu(block=block)) == 9
    assert len(M.score_ret(ENC, block=block, teacher=teacher, n_draws=DRAWS['ret'], expect_n=None)) == 6


def test_cells_merge_by_cell_identity(world, matched, block, monkeypatch, tmp_path):
    monkeypatch.setattr(PP, 'OUT', tmp_path / 'out')
    cells = score_clu(block=block)
    M.write_cells('clu', cells)
    M.write_cells('clu', cells)
    stored = json.loads(M.cells_path('clu').read_text())
    assert len(stored) == 9 and M.cells_path('clu').name == 'cells_mmart_clu.json'
    with pytest.raises(AssertionError):
        M.write_cells('ret', cells)                                  # clustering cells do not go in the retrieval file


# ------------------------------------------------------------------------------------------------ the comparison
def test_compare_docs_holds_numbers_to_1e_9_and_the_rest_exactly():
    import copy
    base = {'a': 1.0, 'b': {'c': [1.0, 2.0], 'd': 'x', 'e': None, 'f': True}}

    def bad(other):
        return [i['name'] for i in M.compare_docs(other, base) if not i['ok']]
    assert bad(copy.deepcopy(base)) == []
    near = copy.deepcopy(base)
    near['a'] += 5e-10
    assert bad(near) == []
    far = copy.deepcopy(base)
    far['a'] += 2e-9
    assert bad(far) == ['/a']
    for mutate in (lambda d: d['b']['c'].append(3.0), lambda d: d['b'].update(d='y'), lambda d: d.update(z=1),
                   lambda d: d['b'].update(f=1), lambda d: d['b'].update(e=0.0), lambda d: d.pop('a')):
        other = copy.deepcopy(base)
        mutate(other)
        assert bad(other), mutate


def r15_form(cells):
    inverse = {v: k for k, v in M.SC_NEW.items()}
    return [dict(c, sc=inverse[c['sc']]) for c in cells]


def test_compare_cells_maps_labels_and_skips_intervals_on_reduced_draws(world, matched, block):
    cells = score_clu(block=block)
    stored = r15_form(cells)
    assert not [i for i in M.compare_cells(cells, stored, 'x') if not i['ok']]
    off = [dict(c) for c in stored]
    off[0]['d'] += 2e-9
    off[1]['lo'] += 1e-6
    bad = {i['name'] for i in M.compare_cells(cells, off, 'x') if not i['ok']}
    assert bad == {f'x/{cells[0]["sc"]}/emotion/clu/d', f'x/{cells[1]["sc"]}/emotion/clu/lo'}
    assert {i['name'] for i in M.compare_cells(cells, off, 'x', intervals=False) if not i['ok']} == set()
    with pytest.raises(AssertionError):
        M.compare_cells(cells, stored[1:], 'x')                      # a replayed cell with no stored cell
    with pytest.raises(AssertionError):
        M.compare_cells(cells, stored[:-1] + [dict(stored[0])], 'x')  # or stored cells that repeat


# ------------------------------------------------------------------------------------------------ gates (iv), (vii)
@pytest.fixture
def r15_files(world, matched, block, teacher, monkeypatch, tmp_path):
    """r15's files in their layout, made by the same scorers: stored cells, records; the replay must reproduce them."""
    monkeypatch.setattr(M, 'REPLAY_ENC', ENC)
    monkeypatch.setattr(M, 'matched_r15', lambda enc, keys: M.check_matched(matched.partitions('r15', enc, keys), len(keys)))
    monkeypatch.setattr(PP, 'R15_OUT', tmp_path / 'mmart')
    monkeypatch.setattr(PP, 'N_DRAWS', dict(DRAWS))
    ret_block = block
    stored = {'clu': r15_form(score_clu(block=ret_block)),
              'ret': r15_form(M.score_ret(ENC, block=ret_block, teacher=teacher, n_draws=DRAWS['ret'], expect_n=None))}
    for part, cells in stored.items():
        path = tmp_path / 'mmart' / 'stage2' / f'cells_{part}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cells))
    records = {f'stage1_{ENC}.json': ret_block, 'stage1_teacher.json': teacher}
    monkeypatch.setattr(mt_paths, 'read_committed_record', lambda name: json.loads(json.dumps(records[name])))
    return dict(stored=stored, records=records, tmp=tmp_path)


def rewrite(path, fn):
    cells = json.loads(path.read_text())
    fn(cells)
    path.write_text(json.dumps(cells))


def replay_iv(**kw):
    kw.setdefault('n_draws_clu', DRAWS['clu'])
    kw.setdefault('n_draws_ret', DRAWS['ret'])
    return M.replay_iv(workers=2, expect_n=None, **kw)


def test_gate_iv_passes_when_the_replay_equals_the_stored_cells(r15_files):
    res = replay_iv()
    assert res['passed'] and res['complete'] and res['n_not_ok'] == 0 and res['max_abs_diff'] <= 1e-9
    assert res['n_items'] > 60


def test_gate_iv_fails_on_one_value_off_by_1e_6(r15_files):
    path = r15_files['tmp'] / 'mmart' / 'stage2' / 'cells_ret.json'
    rewrite(path, lambda cells: cells[3].update(comparator_value=cells[3]['comparator_value'] + 1e-6))
    res = replay_iv()
    assert not res['passed'] and res['n_not_ok'] == 1


def test_gate_iv_cannot_pass_a_reduced_run(r15_files):
    res = replay_iv(parts=('clu',), aspects=('genre',))
    assert res['values_ok'] and not res['complete'] and not res['passed']
    res = replay_iv(n_draws_clu=DRAWS['clu'] // 2)                  # fewer draws than stored: no intervals compared
    assert res['values_ok'] and not res['passed'] and not res['draws']['clu']['intervals_compared']


def test_gate_vii_reproduces_the_stored_record_and_catches_a_changed_pick(world, cached_stage1, monkeypatch):
    stored = M.stage1_block(ENC, expect_n=None)
    monkeypatch.setattr(M, 'REPLAY_ENC', ENC)
    monkeypatch.setattr(PP, 'R15_OUT', mt_paths.OUT)
    box = {'rec': stored}
    monkeypatch.setattr(mt_paths, 'read_committed_record', lambda name: json.loads(json.dumps(box['rec'])))
    res = M.replay_vii(ENC, workers=2, expect_n=None)
    assert res['passed'] and res['max_abs_diff'] <= 1e-9
    wrong = json.loads(json.dumps(stored))
    wrong['picks']['arms']['frozen']['pick'] = 'whiten64+k5'
    box['rec'] = wrong
    assert not M.replay_vii(ENC, workers=2, expect_n=None)['passed']
    wrong = json.loads(json.dumps(stored))
    wrong['clu_ranges']['ranges']['genre'] += 1e-6
    box['rec'] = wrong
    res = M.replay_vii(ENC, workers=2, expect_n=None)
    assert not res['passed'] and res['n_not_ok'] == 1
