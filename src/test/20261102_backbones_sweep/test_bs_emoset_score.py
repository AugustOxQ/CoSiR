"""Ticket 08 (verdict code): EmoSet stage 2 and part c, on synthetic data (CPU, no real store, no EmoSet data).

Each test guards one piece of logic and fails when that piece is broken. The real stores and exports do not exist when
this is written; `FakeSource` stands in for `bs_emoset_score.RealSource` (the same four methods).
"""
import json
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score

import bs_emoset as E
import bs_emoset_score as S
import bs_paths as BP
import ac_cluster
import ac_select
import rs_vh
import rt_readout

D = 16
N_VAL, N_TEST = 240, 320
WORKERS = 1


# ------------------------------------------------------------------------------------------------ synthetic world
def _unit(x):
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class FakeSource:
    """The source interface of `bs_emoset_score`: keys(split), labels(keys), stored(arm, split, seed, keys) (float16),
    frozen(arm, keys) (unit float32), export_sha(arm), descriptive_available(). Features are class centres plus noise;
    each arm has its own noise level, each seed its own noise."""
    NOISE = {'photo_qwen': 0.9, 'cstar_qwen': 1.1, 'photo_siglip2': 1.3, 'photo_siglip1': 1.2, 'photo_clip_l14': 1.4,
             'plain_qwen': 1.6, 'teacher_emotion_focus': 1.0, 'teacher_emotion_focus_photo': 1.05,
             'frozen_siglip2': 1.5, 'frozen_siglip1': 1.5, 'frozen_clip_l14': 1.5, 'crl_clip_l14': 1.4}

    def __init__(self, seed=0, shuffle_keys=False):
        rng = np.random.default_rng(seed)
        self.centres = rng.standard_normal((8, D))
        self.key = {}
        self.label = {}
        self.cls = {}
        for split, n in (('val', N_VAL), ('test', N_TEST)):
            names = [f'image/{split}/{i:05d}.jpg' for i in range(n)]
            self.cls[split] = rng.permutation(np.arange(n) % 8)
            for k, c in zip(names, self.cls[split]):
                self.label[k] = int(c)
            self.key[split] = names
        self.shuffle_keys = shuffle_keys
        self.calls = []

    def keys(self, split):
        self.calls.append(('keys', split))
        ks = list(self.key[split])
        return sorted(ks) if self.shuffle_keys else E.order(ks)

    def labels(self, keys):
        self.calls.append(('labels', len(keys)))
        return np.asarray([self.label[k] for k in keys], dtype=np.int64)

    def _x(self, arm, keys, seed):
        rs = np.random.default_rng(zlib.crc32(f'{arm}/{seed}'.encode()))
        cen = self.centres[[self.label[k] for k in keys]] * 1.0
        return _unit(cen + self.NOISE[arm] * rs.standard_normal((len(keys), D)))

    def stored(self, arm, split, seed, keys):
        self.calls.append(('stored', arm, split, seed))
        return self._x(arm, keys, seed).astype(np.float16)

    def frozen(self, arm, keys):
        self.calls.append(('frozen', arm, len(keys)))
        return self._x(arm, keys, None)

    def export_sha(self, arm):
        return {f'{sp}__s{s}': f'{arm}-{sp}-{s}'.ljust(64, '0') for sp in ('val', 'test') for s in range(5)}

    def descriptive_available(self):
        return list(S.DESCRIPTIVE_ARMS), {}


@pytest.fixture(scope='module')
def src():
    return FakeSource()


@pytest.fixture(scope='module')
def record(src):
    return S.stage2_record(src, workers=WORKERS)


def brute_map(x, labels):
    """Mean AP of cosine retrieval, queries = gallery = all rows, self removed, stable ties; written plainly."""
    x = x.astype(np.float64)
    sim = x @ x.T
    aps = []
    for q in range(len(x)):
        s = sim[q].copy()
        s[q] = -np.inf
        order = np.argsort(-s, kind='stable')[:-1]
        rel = labels[order] == labels[q]
        if rel.sum() == 0:
            continue
        prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
        aps.append((prec * rel).sum() / rel.sum())
    return float(np.mean(aps))


# ------------------------------------------------------------------------------------------------ order and labels
def test_order_check_refuses_sorted_keys_and_accepts_the_hashed_order(src):
    ks = src.key['val']
    S.check_order(E.order(ks))
    with pytest.raises(AssertionError):
        S.check_order(sorted(ks))                       # r12's order; it would group classes in the halves
    with pytest.raises(AssertionError):
        S.check_order(E.order(ks)[:-1] + [E.order(ks)[0]])      # duplicate key


def test_stage2_refuses_rows_not_in_the_hashed_order():
    with pytest.raises(AssertionError):
        S.stage2_record(FakeSource(shuffle_keys=True), workers=WORKERS)


def test_labels_are_codes_of_the_class_names_and_unknown_names_are_refused():
    got = S.encode_labels(['sadness', 'amusement', 'awe'])
    assert got.dtype == np.int64 and got.tolist() == [BP.EMOSET_CLASSES.index(c) for c in ('sadness', 'amusement', 'awe')]
    with pytest.raises(AssertionError):
        S.encode_labels(['amusement', 'something else'])


# ------------------------------------------------------------------------------------------------ step 1: r_E
def test_seed_ranges_are_max_minus_min_over_five_runs_of_ami_and_r0_map(src):
    keys = src.keys('val')
    lab = src.labels(keys)
    stored = {s: src.stored('photo_qwen', 'val', s, keys) for s in range(5)}
    got = S.seed_ranges(stored, lab, workers=WORKERS)
    ami = [float(np.mean([adjusted_mutual_info_score(lab, p) for p in rs_vh.r20_partition(stored[s])])) for s in range(5)]
    mp = [brute_map(stored[s].astype(np.float32) / np.linalg.norm(stored[s].astype(np.float32), axis=1, keepdims=True), lab)
          for s in range(5)]
    assert got['clu']['per_seed'] == pytest.approx(ami, abs=1e-12) and got['map']['per_seed'] == pytest.approx(mp, abs=1e-12)
    assert got['clu']['r'] == max(got['clu']['per_seed']) - min(got['clu']['per_seed']) > 0
    assert got['map']['r'] == max(got['map']['per_seed']) - min(got['map']['per_seed']) > 0
    assert abs(got['clu']['r'] - (max(ami) - min(ami))) < 1e-12 and abs(got['map']['r'] - (max(mp) - min(mp))) < 1e-12


def test_seed_ranges_need_five_runs_and_a_positive_range(src):
    keys = src.keys('val')
    lab = src.labels(keys)
    four = {s: src.stored('photo_qwen', 'val', s, keys) for s in range(4)}
    with pytest.raises(AssertionError):
        S.seed_ranges(four, lab, workers=WORKERS)
    same = {s: four[0] for s in range(5)}                   # five identical runs: range 0, LEVEL would be impossible
    with pytest.raises(AssertionError):
        S.seed_ranges(same, lab, workers=WORKERS)


# ------------------------------------------------------------------------------------------------ step 3: wording
def _doc(pick, maps, r):
    """A read-out document with R0 and one other read-out: maps {name: mAP}."""
    base = maps['none+k0']
    scores = {n: {'map': {'emotion': m}, 'score': (m - base) / r, 'kept': True} for n, m in maps.items()}
    return dict(pick=pick, scores=scores)


R = 0.5
ART = _doc('center+k10', {'none+k0': 0.30, 'center+k10': 0.3500}, R)         # own pick center+k10: 0.35
PHO = _doc('none+k0', {'none+k0': 0.3504, 'center+k10': 0.2000}, R)          # own pick none+k0: 0.3504


def test_retrieval_wording_compares_each_wordings_own_pick():
    got = S.wording_picks({'artwork': 0.2, 'photo': 0.2}, {'artwork': ART, 'photo': PHO}, 0.01, R)
    # own picks: 0.3504 - 0.3500 = 0.0004 = 0.0008 r < 0.001 r -> artwork.  At the artwork's pick (center+k10) photo is
    # 0.2000: far below. At photo's pick for the artwork wording (none+k0) 0.30: photo would win by 0.0504. Only the
    # own-pick rule gives 'artwork' here AND the shared-pick rule 'none+k0' gives 'photo', so this separates them.
    assert got['retrieval'] == 'artwork'
    assert got['map_at_own_pick'] == {'artwork': 0.35, 'photo': 0.3504}
    assert got['readout'] == {'artwork': 'center+k10', 'photo': 'none+k0'}
    better = _doc('none+k0', {'none+k0': 0.3511, 'center+k10': 0.2}, R)         # 0.0011 = 0.0022 r > 0.001 r
    assert S.wording_picks({'artwork': 0.2, 'photo': 0.2}, {'artwork': ART, 'photo': better}, 0.01, R)['retrieval'] == 'photo'


def test_photo_wording_needs_more_than_0_001_r_exactly_the_margin_is_not_enough():
    r_clu = 0.0125
    at = {'artwork': 0.1, 'photo': 0.1 + 0.001 * r_clu}          # exactly 0.001 r_E,clu
    assert S.wording_picks(at, {'artwork': ART, 'photo': ART}, r_clu, R)['clustering'] == 'artwork'
    over = {'artwork': 0.1, 'photo': 0.1 + 0.001 * r_clu + 1e-6}
    assert S.wording_picks(over, {'artwork': ART, 'photo': ART}, r_clu, R)['clustering'] == 'photo'
    under = {'artwork': 0.1, 'photo': 0.09}
    assert S.wording_picks(under, {'artwork': ART, 'photo': ART}, r_clu, R)['clustering'] == 'artwork'


def test_wording_uses_r_clu_for_clustering_and_r_map_for_retrieval():
    ami = {'artwork': 0.10, 'photo': 0.1004}             # diff 0.0004
    docs = {'artwork': ART, 'photo': _doc('center+k10', {'none+k0': 0.3, 'center+k10': 0.3504}, R)}    # diff 0.0004
    # r = 0.2: 0.0004 = 0.002 r > 0.001 r (photo). r = 0.8: 0.0005 r (artwork). Swapping the two r values flips both.
    a = S.wording_picks(ami, docs, 0.2, 0.8)
    b = S.wording_picks(ami, docs, 0.8, 0.2)
    assert (a['clustering'], a['retrieval']) == ('photo', 'artwork')
    assert (b['clustering'], b['retrieval']) == ('artwork', 'photo')


# ------------------------------------------------------------------------------------------------ stage 2 as a whole
def test_stage2_runs_r_e_before_the_picks_and_every_pick_uses_r_e_map(src, monkeypatch):
    seen = []
    real_table, real_ranges = ac_select.arm_table, S.seed_ranges

    def spy_table(xs, labels, ranges, **kw):
        seen.append(('table', dict(ranges)['emotion'], tuple(sorted((s for (_, s) in xs), key=str))))
        return real_table(xs, labels, ranges, **kw)

    def spy_ranges(*a, **kw):
        out = real_ranges(*a, **kw)
        seen.append(('ranges', out['map']['r']))
        return out

    monkeypatch.setattr(ac_select, 'arm_table', spy_table)
    monkeypatch.setattr(S, 'seed_ranges', spy_ranges)
    rec = S.stage2_record(src, workers=WORKERS)
    assert seen[0][0] == 'ranges'                                    # step 1 first
    r_map = rec['ranges']['map']['r']
    tables = [s for s in seen if s[0] == 'table']
    assert len(tables) == len(S.BAR_ARMS) + len(S.DESCRIPTIVE_ARMS)  # steps 2 and 4 (step 3 scores no table)
    assert all(t[1] == r_map for t in tables)                         # r_E,mAP for every arm, bar and descriptive
    # bar arms (step 2) are picked before the descriptive arms (step 4)
    order = [seen[i] for i in range(len(seen)) if seen[i][0] == 'table']
    assert [len(t[2]) for t in order[:4]] == [5, 1, 1, 1]            # photo model (5 seeds), plain, two wordings
    assert rec['wording']['retrieval'] in ('artwork', 'photo')


def test_stage2_record_content_and_independent_checks(src, record):
    S.check_record(record)
    assert set(record['readout']) == set(S.BAR_ARMS) and set(record['descriptive']) == set(S.DESCRIPTIVE_ARMS)
    assert record['sets']['val']['n'] == N_VAL and record['sets']['test']['n'] == N_TEST
    assert record['sets']['val']['keys_sha256'] == S.keys_sha(src.keys('val'))
    r_map = record['ranges']['map']['r']
    for arm, doc in {**record['readout'], **record['descriptive']}.items():
        table = doc['scores']
        base = table['none+k0']['map']['emotion']
        assert abs(table[doc['pick']]['score'] - (table[doc['pick']]['map']['emotion'] - base) / r_map) < 1e-12, arm
    # plain Qwen's R0 cross-fit mAP, brute force: mean over the two parity halves of the sha256 order
    keys = src.keys('val')
    lab = src.labels(keys)
    x = src.frozen('plain_qwen', keys)
    halves = [brute_map(x[h::2], lab[h::2]) for h in (0, 1)]
    assert abs(record['readout']['plain_qwen']['scores']['none+k0']['map']['emotion'] - np.mean(halves)) < 1e-12
    ami = float(np.mean([adjusted_mutual_info_score(lab, p)
                         for p in rs_vh.r20_partition(src.frozen('teacher_emotion_focus', keys))]))
    assert abs(record['teacher']['ami']['artwork'] - ami) < 1e-12
    assert record['exports']['photo_qwen'] == src.export_sha('photo_qwen')
    assert 'cstar_qwen' in record['descriptive']


def test_check_record_refuses_a_hand_edited_pick_range_or_wording(record):
    import copy
    bad = copy.deepcopy(record)
    bad['readout']['plain_qwen']['pick'] = 'none+k0' if record['readout']['plain_qwen']['pick'] != 'none+k0' else 'center+k10'
    with pytest.raises(AssertionError):
        S.check_record(bad)
    bad = copy.deepcopy(record)
    bad['ranges']['map']['r'] *= 1.5
    with pytest.raises(AssertionError):
        S.check_record(bad)
    bad = copy.deepcopy(record)
    bad['wording']['clustering'] = 'photo' if record['wording']['clustering'] == 'artwork' else 'artwork'
    with pytest.raises(AssertionError):
        S.check_record(bad)


# ------------------------------------------------------------------------------------------------ the held-data guard
def _git(cwd, *args):
    subprocess.run(['git', '-C', str(cwd), '-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false', *args],
                   check=True, capture_output=True)


def test_guard_needs_the_record_committed_and_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(S.BP, 'SMOKE', 0)
    f = tmp_path / 'stage2_emoset.json'
    assert 'does not exist' in S.guard_reason(f)
    _git(tmp_path, 'init', '-q')
    f.write_text('{"a": 1}\n')
    assert 'not in git HEAD' in S.guard_reason(f)
    _git(tmp_path, 'add', f.name)
    _git(tmp_path, 'commit', '-q', '-m', 'x')
    assert S.guard_reason(f) == ''
    f.write_text('{"a": 2}\n')
    assert 'differs from its content in git HEAD' in S.guard_reason(f)


def test_stage3_stops_before_reading_any_feature_when_the_record_is_not_committed(src, record, tmp_path, monkeypatch):
    monkeypatch.setattr(S.BP, 'SMOKE', 0)
    monkeypatch.setattr(S, 'record_file', lambda: tmp_path / 'stage2_emoset.json')
    fresh = FakeSource()
    with pytest.raises(SystemExit, match='REFUSED'):
        S.run_stage3(fresh, tmp_path / 'out', workers=WORKERS, n_clu=4, n_ret=8)
    assert fresh.calls == []                                           # no key, label or feature was read


# ------------------------------------------------------------------------------------------------ test scoring
def test_test_retrieval_fits_the_map_on_all_validation_rows_and_applies_it_to_test(src):
    kv, kt = src.keys('val'), src.keys('test')
    xv, xt = src.frozen('plain_qwen', kv), src.frozen('plain_qwen', kt)
    xt = _unit(xt + 0.8)                                               # test mean differs from validation mean
    lab = src.labels(kt)
    res = S.test_retrieval({None: xv}, {None: xt}, 'center+k0', lab, seeded=False, workers=WORKERS)
    def centre(x, mu):                                                 # float64 throughout, one cast to float32 (r11's form)
        c = x.astype(np.float64) - mu
        return (c / np.linalg.norm(c, axis=1, keepdims=True)).astype(np.float32)

    centred = centre(xt, xv.astype(np.float64).mean(axis=0))
    assert abs(res['point'] - brute_map(centred, lab)) < 1e-9
    own = centre(xt, xt.astype(np.float64).mean(axis=0))
    assert abs(brute_map(own, lab) - res['point']) > 1e-5              # fitting on test would give another number
    assert res['dropped'] == 0 and res['rows'].tolist() == list(range(len(kt)))


def test_partitions_k20_are_r10s_and_k8_has_eight_clusters(src):
    x = src.frozen('plain_qwen', src.keys('val'))
    assert np.array_equal(S.partitions_k(x, 20), ac_cluster.partitions(x))
    p8 = S.partitions_k(x, 8)
    assert p8.shape == (3, len(x)) and p8.dtype == np.int32 and all(len(np.unique(p)) <= 8 for p in p8)
    assert len(np.unique(p8[0])) > 4


def _stage3(src, record, tmp_path, **kw):
    return S.stage3(src, record, tmp_path / 'cache', workers=WORKERS, n_clu=6, n_ret=30, **kw)


@pytest.fixture(scope='module')
def part_c(src, record, tmp_path_factory):
    return S.stage3(src, record, tmp_path_factory.mktemp('cache'), workers=WORKERS, n_clu=6, n_ret=30)


def test_cells_sc7_to_sc10_follow_the_index_format_and_the_record(part_c, record):
    cells = {c['sc']: c for c in part_c['cells']}
    assert list(cells) == ['SC7', 'SC8', 'SC9', 'SC10']
    for sc, task in (('SC7', 'clu'), ('SC8', 'ret'), ('SC9', 'clu'), ('SC10', 'ret')):
        c = cells[sc]
        assert set(S.CELL_KEYS) <= set(c)
        assert (c['encoder'], c['aspect'], c['data'], c['task'], c['candidate']) == ('qwen', 'emotion', 'emoset_test', task, 'photo_qwen')
        assert c['seeds'] == [0, 1, 2, 3, 4]
        assert c['n_draws'] == (6 if task == 'clu' else 30)
        assert c['lo'] <= c['hi']
    assert cells['SC7']['comparator'] == 'plain_qwen' and cells['SC8']['comparator'] == 'plain_qwen'
    assert cells['SC7']['r'] is None and cells['SC8']['r'] is None
    w = record['teacher']['views']
    assert cells['SC9']['comparator'] == f"teacher_{w[record['wording']['clustering']]}"
    assert cells['SC10']['comparator'] == f"teacher_{w[record['wording']['retrieval']]}"
    assert cells['SC9']['r'] == record['ranges']['clu']['r'] and cells['SC10']['r'] == record['ranges']['map']['r']
    assert 'r_E,clu' in cells['SC9']['r_source'] and 'r_E,mAP' in cells['SC10']['r_source']
    assert cells['SC8']['readout_candidate'] == record['readout']['photo_qwen']['pick']
    assert cells['SC8']['readout_comparator'] == record['readout']['plain_qwen']['pick']
    json.dumps(part_c['cells'], allow_nan=False)


def test_cell_values_match_independent_computations(src, part_c, record):
    kt = src.keys('test')
    lab = src.labels(kt)
    sheet = {r['arm']: r for r in part_c['sheet']}
    plain = _unit(src.frozen('plain_qwen', kt))
    assert abs(sheet['plain_qwen']['map_r0'] - brute_map(plain, lab)) < 1e-7           # R0 is plain cosine, test rows
    ami = float(np.mean([adjusted_mutual_info_score(lab, p) for p in rs_vh.r20_partition(plain)]))
    cells = {c['sc']: c for c in part_c['cells']}
    assert abs(cells['SC7']['comparator_value'] - ami) < 1e-12 and abs(sheet['plain_qwen']['ami_k20'] - ami) < 1e-12
    photo = [src.stored('photo_qwen', 'test', s, kt) for s in range(5)]
    mean_ami = float(np.mean([np.mean([adjusted_mutual_info_score(lab, p) for p in rs_vh.r20_partition(x)]) for x in photo]))
    assert abs(cells['SC7']['candidate_value'] - mean_ami) < 1e-12
    assert abs(cells['SC7']['d'] - (cells['SC7']['candidate_value'] - cells['SC7']['comparator_value'])) < 1e-12
    assert abs(cells['SC8']['candidate_value'] - sheet['photo_qwen']['map_picked']) < 1e-12
    assert abs(cells['SC8']['comparator_value'] - sheet['plain_qwen']['map_picked']) < 1e-12
    assert abs(cells['SC8']['d'] - (cells['SC8']['candidate_value'] - cells['SC8']['comparator_value'])) < 1e-12


def test_k8_row_is_the_ami_of_eight_clusters(src, part_c):
    from sklearn.cluster import MiniBatchKMeans
    kt = src.keys('test')
    lab = src.labels(kt)
    plain = src.frozen('plain_qwen', kt)
    want = float(np.mean([adjusted_mutual_info_score(
        lab, MiniBatchKMeans(n_clusters=8, random_state=s, n_init=3, batch_size=4096).fit(plain).labels_) for s in (0, 1, 2)]))
    row = {r['arm']: r for r in part_c['sheet']}['plain_qwen']
    assert abs(row['ami_k8'] - want) < 1e-12 and abs(row['ami_k8'] - row['ami_k20']) > 1e-4


def test_each_task_uses_its_own_wording_from_the_record(src, record, tmp_path):
    import copy
    rec = copy.deepcopy(record)
    # make the two tasks pick different wordings: the photo wording wins clustering by a wide margin, retrieval stays as is
    rec['teacher']['ami'] = dict(artwork=0.1, photo=0.1 + 100 * rec['ranges']['clu']['r'])
    rec['teacher']['map_at_own_pick']['photo'] = rec['teacher']['map_at_own_pick']['artwork']
    rec['wording'] = dict(clustering='photo', retrieval='artwork')
    for w in ('artwork', 'photo'):                       # the stored tables must give those map values, so check_record passes
        rec['readout'][S.TEACHER_ARM[w]] = copy.deepcopy(record['readout'][S.TEACHER_ARM['artwork']])
    rec['teacher']['readout'] = dict(artwork=rec['readout']['teacher_emotion_focus']['pick'],
                                     photo=rec['readout']['teacher_emotion_focus']['pick'])
    S.check_record(rec)
    out = S.stage3(src, rec, tmp_path / 'cache', workers=WORKERS, n_clu=4, n_ret=8)
    cells = {c['sc']: c for c in out['cells']}
    assert cells['SC9']['comparator'] == 'teacher_emotion_focus_photo' and cells['SC10']['comparator'] == 'teacher_emotion_focus'


def test_sheet_has_k8_and_r0_rows_for_every_arm_and_makes_no_cells(part_c, record):
    arms = {r['arm'] for r in part_c['sheet']}
    assert arms == set(S.BAR_ARMS) | set(S.DESCRIPTIVE_ARMS)
    for r in part_c['sheet']:
        assert {'ami_k20', 'ami_k8', 'map_r0', 'map_picked', 'readout', 'role'} <= set(r)
        assert all(np.isfinite(r[k]) for k in ('ami_k20', 'ami_k8', 'map_r0', 'map_picked'))
        pick = (record['readout'].get(r['arm']) or record['descriptive'][r['arm']])['pick']
        assert r['readout'] == pick                                       # the sheet uses the committed pick
    assert {c['candidate'] for c in part_c['cells']} == {'photo_qwen'}      # descriptive arms never enter a cell
    assert {r['role'] for r in part_c['sheet']} == {'bar', 'descriptive'}


def test_stage3_refuses_a_record_for_other_cut_sets_or_other_exports(src, record):
    import copy
    bad = copy.deepcopy(record)
    bad['sets']['test']['keys_sha256'] = '0' * 64
    with pytest.raises(AssertionError):
        S.stage3(src, bad, Path('/nonexistent'), workers=WORKERS, n_clu=4, n_ret=8)
    bad = copy.deepcopy(record)
    bad['exports']['photo_qwen']['test__s3'] = '1' * 64
    with pytest.raises(AssertionError):
        S.stage3(src, bad, Path('/nonexistent'), workers=WORKERS, n_clu=4, n_ret=8)


def test_write_cells_is_atomic_and_complete(part_c, tmp_path):
    f = tmp_path / 'stage3' / 'cells_emoset.json'
    S.write_cells(f, part_c['cells'])
    assert json.loads(f.read_text()) == json.loads(json.dumps(part_c['cells']))
    with pytest.raises(AssertionError):
        S.write_cells(f, part_c['cells'][:3])                      # the four cells of part c, all or nothing


# ------------------------------------------------------------------------------------------------ gate 0(xi)
def _r12_docs():
    """A tiny r12-shaped validation output: tables for both wordings, picks, ranges, and r12's wording picks."""
    ranges = {'clu': {'per_seed': [0.1, 0.11, 0.12, 0.1, 0.1], 'r': 0.02}, 'map': {'per_seed': [0.3] * 5, 'r': 0.5}}
    tables = {'teacher_artwork': ART['scores'], 'teacher_photo': PHO['scores']}
    picks = {'readout': {'teacher_artwork': ART['pick'], 'teacher_photo': PHO['pick']},
             'teacher_wording': {'clustering': 'photo', 'retrieval': 'artwork'}}
    scores = {'teacher_wording': {'clustering': {'artwork': 0.1, 'photo': 0.1300},
                                  'retrieval': {'artwork': 0.35, 'photo': 0.3504}}}
    return ranges, tables, picks, scores


def test_gate_0xi_reproduces_r12s_wording_and_fails_on_a_planted_difference():
    ranges, tables, picks, scores = _r12_docs()
    res = S.gate_xi_from_documents(ranges, tables, picks, scores)
    assert res['ok'] and all(i['ok'] for i in res['items'].values())
    bad_picks = json.loads(json.dumps(picks))
    bad_picks['teacher_wording']['retrieval'] = 'photo'
    assert not S.gate_xi_from_documents(ranges, tables, bad_picks, scores)['ok']
    bad_scores = json.loads(json.dumps(scores))
    bad_scores['teacher_wording']['retrieval']['photo'] += 1e-6
    assert not S.gate_xi_from_documents(ranges, tables, picks, bad_scores)['ok']


# ------------------------------------------------------------------------------------------------ the real source
@pytest.fixture
def real_world(tmp_path, monkeypatch):
    """RealSource on a synthetic split folder, store root and exports: rows are joined by key, never by position."""
    import bs_stores
    rng = np.random.default_rng(1)
    classes = BP.EMOSET_CLASSES
    val = [[classes[i % 8], f'image/{classes[i % 8]}/{classes[i % 8]}_{i:05d}.jpg', f'annotation/x/{i}.json'] for i in range(48)]
    test = [[classes[i % 8], f'image/{classes[i % 8]}/{classes[i % 8]}_{i + 100:05d}.jpg', f'annotation/x/{i}.json']
            for i in range(56)]
    val[0][0] = 'fear'                       # first field disagrees with its path: the label is the first field
    root = tmp_path / 'emoset'
    root.mkdir()
    (root / 'val.json').write_text(json.dumps(val))
    (root / 'test.json').write_text(json.dumps(test))
    monkeypatch.setattr(E, 'ROOT', root)
    monkeypatch.setattr(E, 'COUNTS', {'val': len(val), 'test': len(test)})
    E._ENTRIES.clear()
    rec = dict(cut=BP.NEAR_COPY_CUT, reference=dict(n=E.AFFECTION_TRAIN_N, view='qwen_default'),
               splits={'val': dict(n=48, removed_keys=[val[5][1]], n_removed=1),
                       'test': dict(n=56, removed_keys=[], n_removed=0)})
    (tmp_path / 'nearcopy.json').write_text(json.dumps(rec))
    monkeypatch.setattr(E, 'record_file', lambda: tmp_path / 'nearcopy.json')
    monkeypatch.setattr(bs_stores, 'WIDTH', {k: D for k in bs_stores.WIDTH})
    monkeypatch.setenv('BS_STORE_ROOT', str(tmp_path / 'stores'))
    allkeys = [e[1] for e in val + test]
    for view in ('qwen_default', 'emotion_focus', 'emotion_focus_photo', 'siglip2'):
        folder = bs_stores.new_folder('emoset', view)
        folder.mkdir(parents=True)
        perm = rng.permutation(len(allkeys))                      # the store's row order is not the split order
        img = _unit(rng.standard_normal((len(allkeys), D))).astype(np.float16)
        np.save(folder / 'img.npy', img)
        (folder / 'keys.json').write_text(json.dumps([allkeys[i] for i in perm]))
        if view == 'qwen_default':
            stored_rows = {allkeys[i]: img[j] for j, i in enumerate(perm)}
    monkeypatch.setattr(S.BP, 'OUT', tmp_path / 'out')
    return dict(val=val, test=test, tmp=tmp_path, keys=allkeys, rows=stored_rows)


def _write_export(arm, split, seed, keys, emb, root, bad_sha=False):
    import hashlib
    p = Path(root) / 'exports/emoset' / arm / f'{split}__s{seed}.npz'
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, keys=np.asarray(keys), emb=emb.astype(np.float16))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    p.with_suffix('.json').write_text(json.dumps(dict(sha256='f' * 64 if bad_sha else sha)))
    return p


def test_real_source_reads_labels_stores_and_exports_by_key(real_world):
    rng = np.random.default_rng(2)
    rs = S.RealSource()
    keys = rs.keys('val')
    assert len(keys) == 47 and keys == E.order(keys) and real_world['val'][5][1] not in keys      # the cut set, hashed order
    lab = rs.labels(keys)
    first = {e[1]: e[0] for e in real_world['val']}
    assert lab.tolist() == [BP.EMOSET_CLASSES.index(first[k]) for k in keys]
    x = rs.frozen('plain_qwen', keys)
    assert x.dtype == np.float32 and np.abs(np.linalg.norm(x, axis=1) - 1).max() < 1e-5
    want_rows = np.stack([real_world['rows'][k].astype(np.float32) for k in keys])
    want_rows /= np.linalg.norm(want_rows, axis=1, keepdims=True)
    assert np.abs(x - want_rows).max() < 1e-6                         # joined by key, not by position
    full = [e[1] for e in real_world['val']]
    emb = _unit(rng.standard_normal((len(full), D)))
    shuffled = rng.permutation(len(full))
    _write_export('photo_qwen', 'val', 0, [full[i] for i in shuffled], emb[shuffled], real_world['tmp'] / 'out')
    got = rs.stored('photo_qwen', 'val', 0, keys)
    want = emb[[full.index(k) for k in keys]].astype(np.float16)
    assert got.dtype == np.float16 and np.array_equal(got, want)
    _write_export('photo_qwen', 'val', 1, full, emb, real_world['tmp'] / 'out', bad_sha=True)
    with pytest.raises(AssertionError, match='sidecar'):
        rs.stored('photo_qwen', 'val', 1, keys)
    sha = rs.export_sha('photo_qwen')
    assert set(sha) <= {f'{sp}__s{s}' for sp in ('val', 'test') for s in range(5)} and 'val__s0' in sha
