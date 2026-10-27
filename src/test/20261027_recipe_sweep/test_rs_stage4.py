"""Ticket 09: the stage 4 scorer. CPU only, synthetic data; the check on the real arrays is the gate
(`rs_stage4 check v4a`)."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import rs_paths as paths
import rs_sheet as sheet
import rs_stage0 as S0
import rs_stage4 as S4
import rs_full as F
import rb_part2_report
import rb_score
import rg_held

N = 80                                     # a stand-in for H's 12,281 rows
RANGES = sheet.RANGES
CANDIDATE_FIXED = S0.candidate_fixed


@pytest.fixture
def tmp_path():
    """Held smoke fixtures belong under the smoke root, also with pytest's default TMPDIR."""
    root = Path(os.environ.get('TMPDIR', paths.SMOKE_ROOT))
    if not paths.smoke_ok(root):
        root = paths.SMOKE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='rs-stage4-', dir=root))


# ------------------------------------------------------------------ outcomes at their edges
@pytest.mark.parametrize('aspect', S4.ASPECTS)
def test_t_outcome_edges_per_aspect(aspect):
    r = RANGES[aspect]
    assert S4.t_outcome(aspect, 1e-12) == 'T-WIN'
    assert S4.t_outcome(aspect, 0.0) == 'T-LEVEL'            # not T-WIN: the lower bound must be above 0
    assert S4.t_outcome(aspect, -r + 1e-12) == 'T-LEVEL'
    assert S4.t_outcome(aspect, -r) == 'T-BELOW'             # not above -r_a
    assert S4.t_outcome(aspect, -r - 1e-12) == 'T-BELOW'
    assert S4.t_outcome(aspect, 5.0) == 'T-WIN'


def test_t_outcome_uses_each_aspects_own_range():
    # -0.006 is below -r_emotion (0.0046) and inside -r_style (0.0082) and -r_genre (0.0128)
    assert [S4.t_outcome(a, -0.006) for a in S4.ASPECTS] == ['T-BELOW', 'T-LEVEL', 'T-LEVEL']


def test_t_outcome_refuses_non_finite_and_unknown_aspect():
    with pytest.raises(AssertionError):
        S4.t_outcome('genre', float('nan'))
    with pytest.raises(KeyError):
        S4.t_outcome('topic', 0.0)


@pytest.mark.parametrize('d,lower,expected', [
    (0.01, 1e-12, 'PE-GO'), (0.01, 0.0, 'PE-MM'), (0.0099999, 0.004, 'PE-IN'), (1e-12, -0.01, 'PE-IN'),
    (0.0, -0.01, 'PE-NO'), (-0.02, -0.05, 'PE-NO'), (0.0, 0.5, 'PE-NO')])
def test_pe_outcome_thresholds_are_section_4_1s(d, lower, expected):
    assert S4.pe_outcome(d, lower) == expected


# ------------------------------------------------------------------ the draws
def test_d_t_draw_order_is_i_b_then_s_b_for_c_star_and_t_draws_nothing():
    draws = rg_held.draws_d_t(N, 6)
    rng = np.random.default_rng(0)
    for rows, seeds in draws:
        assert np.array_equal(rows, rng.integers(0, N, size=N))
        assert np.array_equal(seeds['G'], rng.choice([0, 1, 2, 3, 4], size=5, replace=True))
        assert list(seeds['T']) == [0]
    assert len(draws) == 6


def test_d_g_draw_order_is_i_b_then_one_s_b_shared_by_both():
    draws = S4.draws_d_g(N, 6)
    rng = np.random.default_rng(0)
    for rows, seeds in draws:
        assert np.array_equal(rows, rng.integers(0, N, size=N))
        s = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
        assert np.array_equal(seeds['C'], s) and np.array_equal(seeds['G'], s)
    assert list(draws[0][1]) == ['C', 'G']


# ------------------------------------------------------------------ the statistic on synthetic partitions
@pytest.fixture(scope='module')
def world():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 5, size=N)
    labels[::7] = -1                                           # unlabelled rows (labels < 0)

    def part(strength):
        return np.where(rng.random((3, N)) < strength, labels % 6, rng.integers(0, 6, (3, N)))
    cand = {s: {'r20': part(.6)} for s in S4.SEEDS}
    other = {s: {'r20': part(.4)} for s in S4.SEEDS}
    t = {0: {'r20': part(.5)}}
    return labels, cand, other, t


def test_d_t_block_is_the_rb_statistic_with_the_rule_s_draws(world):
    labels, cand, _, t = world
    block = S4.d_t_block(cand, t, labels, 'genre', 30, 1)
    arms = ('G', 'T')
    ref = [rb_score.score_indices(dict(G=cand, T=t), labels, rows, seeds, arms)['d'] for rows, seeds in rg_held.draws_d_t(N, 30)]
    point = rb_score.score_indices(dict(G=cand, T=t), labels, np.arange(N), dict(G=S4.SEEDS, T=[0]), arms)
    assert block['d'] == point['d'] and block['A_cand'] == point['A_cand'] and block['A_T'] == point['A_e']['T-r20']
    assert [block['lo'], block['hi']] == np.percentile(ref, [2.5, 97.5]).tolist()
    assert block['bootstrap_d'] == ref
    assert block['d_in_ranges'] == block['d'] / RANGES['genre']
    assert block['outcome'] == S4.t_outcome('genre', block['lo'])
    assert block['labelled_rows'] == int((labels >= 0).sum()) and block['labels_masked'] == int((labels < 0).sum())


def test_labels_below_zero_are_masked_not_scored(world):
    labels, cand, _, t = world
    base = S4.d_t_block(cand, t, labels, 'style', 12, 1)
    changed = {s: {'r20': p['r20'].copy()} for s, p in cand.items()}
    for p in changed.values():
        p['r20'][:, labels < 0] = 99                          # the unlabelled rows' cluster ids change
    t2 = {0: {'r20': t[0]['r20'].copy()}}
    t2[0]['r20'][:, labels < 0] = 77
    again = S4.d_t_block(changed, t2, labels, 'style', 12, 1)
    for key in ('d', 'A_cand', 'A_T', 'lo', 'hi', 'bootstrap_d'):
        assert again[key] == base[key], key
    # a labelled row's cluster id does count
    moved = {s: {'r20': p['r20'].copy()} for s, p in cand.items()}
    for p in moved.values():
        p['r20'][:, np.flatnonzero(labels >= 0)[:20]] = 99
    assert S4.d_t_block(moved, t, labels, 'style', 12, 1)['A_cand'] != base['A_cand']


def test_d_t_block_refuses_bad_shapes(world):
    labels, cand, _, t = world
    with pytest.raises(AssertionError):
        S4.d_t_block({s: p for s, p in cand.items() if s != 3}, t, labels, 'genre', 5, 1)     # a seed missing
    with pytest.raises(AssertionError):
        S4.d_t_block(cand, {1: t[0]}, labels, 'genre', 5, 1)                                   # T has one set, seed 0
    with pytest.raises(AssertionError):
        S4.d_t_block(cand, t, labels[:-1], 'genre', 5, 1)


@pytest.mark.parametrize('aspect', S4.ASPECTS)
@pytest.mark.parametrize('lo,expected', [(1e-9, 'T-WIN'), (0.0, 'T-LEVEL'), (-1e-9, 'T-LEVEL'), (-1.0, 'T-BELOW')])
def test_d_t_block_outcome_comes_from_the_lower_bound(monkeypatch, world, aspect, lo, expected):
    labels, cand, _, t = world
    monkeypatch.setattr(rg_held, 'bootstrap_d_t', lambda *a, **k: dict(
        A_G=.3, A_T=.2, d=.1, interval=[lo, .2], bootstrap_d=[], per_seed={}))
    assert S4.d_t_block(cand, t, labels, aspect, 5, 1)['outcome'] == expected


def test_d_g_is_candidate_minus_g_with_one_seed_draw_for_both(world):
    labels, cand, other, _ = world
    got = S4.d_g_block(cand, other, labels, 20, 1)
    draws = S4.draws_d_g(N, 20)
    ref = [rb_score.score_indices(dict(C=cand, G=other), labels, rows, seeds, ('C', 'G'))['d'] for rows, seeds in draws]
    assert got['bootstrap_d'] == ref
    assert [got['lo'], got['hi']] == np.percentile(ref, [2.5, 97.5]).tolist()
    assert abs(got['d'] - (got['A_cand'] - got['A_G'])) < 1e-12
    same = S4.d_g_block(cand, cand, labels, 8, 1)
    assert same['d'] == 0.0 and same['lo'] == 0.0 and same['hi'] == 0.0


def test_per_baseline_margins_are_the_rb_margin_against_each_baseline(world):
    labels, cand, other, _ = world
    comp = {a: {s: {'r20': other[s]['r20'], 'own': other[s]['r20'][:1]} for s in S4.SEEDS} for a in ('B1', 'B4')}
    comp.update(B2={}, B3={}, B5={}, B6={})
    eligible = {'c': list(S4.SEEDS), 'B1': list(S4.SEEDS), 'B2': [], 'B3': [], 'B4': [0, 2, 4], 'B5': [], 'B6': []}
    fixed = dict(comp, c=cand)
    got = S4.per_baseline_margins(fixed, labels, eligible, 'c', 10, 1)
    assert list(got) == ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
    assert got['B2'] is None and got['B5'] is None            # no eligible seed: no margin
    ref = rb_part2_report.margin(fixed, labels, eligible, ('c', 'B4'), 10, 1)
    assert got['B4']['d'] == ref['d'] and [got['B4']['lo'], got['B4']['hi']] == ref['interval']
    assert got['B4']['eligible_seeds'] == [0, 2, 4] and got['B4']['max_entry'] == ref['max_entry']


# ------------------------------------------------------------------ code hash and the gate
def test_code_sha256_covers_the_scorer_and_what_it_imports():
    files = S4.code_sha256()
    for name in ('rs_stage4.py', 'rs_stage0.py', 'rg_held.py', 'rb_score.py', 'rb_part2_report.py', 'rs_paths.py',
                 'ct_held.py', 'ct_diag.py'):
        assert name in files and len(files[name]) == 64, name
    assert not any(n.startswith('test_') or n == 'conftest.py' for n in files)
    assert files == S4.code_sha256()
    assert files['rs_stage4.py'] == rb_score.sha_file(S4.__file__)


def test_code_sha256_changes_when_an_imported_file_changes(monkeypatch, tmp_path):
    before = S4.code_sha256()
    real = rb_score.sha_file

    def fake(path):
        return '0' * 64 if Path(path).name == 'rb_part2_report.py' else real(path)
    monkeypatch.setattr(S4, 'sha_file', fake)
    after = S4.code_sha256()
    assert after != before and after['rb_part2_report.py'] == '0' * 64


def write_gate(outputs, **changes):
    record = dict(name='v4a_scorer', code_sha256=S4.code_sha256(), smoke=False, items={})
    record['pass'] = True
    record.update(changes)
    path = Path(outputs) / 'gates/v4a_scorer.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def test_require_v4a_scorer_refuses_missing_failed_smoke_and_stale(tmp_path):
    with pytest.raises(S0.Refusal, match='missing'):
        S4.require_v4a_scorer(tmp_path)
    write_gate(tmp_path, **{'pass': False})
    with pytest.raises(S0.Refusal, match='does not pass'):
        S4.require_v4a_scorer(tmp_path)
    write_gate(tmp_path, smoke=True)
    with pytest.raises(S0.Refusal, match='smoke'):
        S4.require_v4a_scorer(tmp_path)
    stale = S4.code_sha256()
    stale['rs_stage4.py'] = '0' * 64
    write_gate(tmp_path, code_sha256=stale)
    with pytest.raises(S0.Refusal, match='other code'):
        S4.require_v4a_scorer(tmp_path)
    path = write_gate(tmp_path)
    assert S4.require_v4a_scorer(tmp_path) == rb_score.sha_file(path)


# ------------------------------------------------------------------ who the candidate is
def put(outputs, relative, record):
    path = Path(outputs) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def test_candidate_is_the_stage_3_pick(tmp_path):
    put(tmp_path, 'stage3/verdict.json', dict(outcome='PICK', pick='k6=0.0005', pick_config=dict(sheet.parse('k6=0.0005'), remove=[]), smoke=False))
    got = S4.stage_candidate(tmp_path)
    assert got['name'] == 'k6=0.0005' and got['kind'] == 'PICK'
    assert got['upstream']['file'] == 'stage3/verdict.json' and len(got['upstream']['sha256']) == 64


def test_keep_g_gives_g(tmp_path):
    put(tmp_path, 'stage3/verdict.json', dict(outcome='KEEP-G', pick=None, pick_config=None, smoke=False))
    got = S4.stage_candidate(tmp_path)
    assert got['name'] == 'G' and got['kind'] == 'KEEP-G'


@pytest.mark.parametrize('bad', [dict(outcome='PICK', pick=None), dict(outcome='KEEP-G', pick='G'),
                                 dict(outcome='PICK', pick=''), dict(outcome='MAYBE', pick=None),
                                 dict(outcome='PICK', pick='k6=0.0005', smoke=True)])
def test_stage_3_file_that_contradicts_itself_or_is_a_smoke_is_refused(tmp_path, bad):
    put(tmp_path, 'stage3/verdict.json', {'smoke': False, **bad})
    with pytest.raises(S0.Refusal):
        S4.stage_candidate(tmp_path)


def test_no_stage_3_file_is_g_only_under_sweep_none_or_top_none(tmp_path):
    put(tmp_path, 'sweep/local/stage1.json', dict(outcome='SWEEP-NONE'))
    got = S4.stage_candidate(tmp_path)
    assert got['name'] == 'G' and got['kind'] == 'SWEEP-NONE' and got['upstream']['file'] == 'sweep/local/stage1.json'
    other = tmp_path / 'b'
    put(other, 'sweep/node402/stage1.json', dict(outcome='GRID-2'))
    put(other, 'sweep/node402/stage2.json', dict(outcome='TOP-NONE'))
    got = S4.stage_candidate(other)
    assert got['name'] == 'G' and got['kind'] == 'TOP-NONE' and got['upstream']['file'] == 'sweep/node402/stage2.json'


@pytest.mark.parametrize('files', [
    {},                                                                                   # nothing
    {'sweep/local/stage1.json': dict(outcome='GRID-2')},                                  # stage 2 missing
    {'sweep/local/stage1.json': dict(outcome='GRID-2'), 'sweep/local/stage2.json': dict(outcome='TOP')},
])
def test_missing_stage_3_file_without_that_outcome_is_an_error(tmp_path, files):
    for relative, record in files.items():
        put(tmp_path, relative, record)
    with pytest.raises(S0.Refusal):
        S4.stage_candidate(tmp_path)


def test_every_machine_must_have_ended_without_candidates(tmp_path):
    put(tmp_path, 'sweep/local/stage1.json', dict(outcome='SWEEP-NONE'))
    put(tmp_path, 'sweep/node402/stage1.json', dict(outcome='GRID-1'))
    with pytest.raises(S0.Refusal):
        S4.stage_candidate(tmp_path)
    put(tmp_path, 'sweep/node402/stage2.json', dict(outcome='TOP-NONE'))
    got = S4.stage_candidate(tmp_path)
    assert got['name'] == 'G'
    assert set(got['upstream']['machines']) == {'local', 'node402'}
    for machine, record in got['upstream']['machines'].items():
        assert record['sha256'] == S4.sha_file(tmp_path / record['file'])
        assert record['file'].startswith(f'sweep/{machine}/')


def test_stage_candidate_refuses_a_pick_config_that_does_not_name_the_pick(tmp_path):
    put(tmp_path, 'stage3/verdict.json', dict(outcome='PICK', pick='k6=0.0005', pick_config={}, smoke=False))
    with pytest.raises(S0.Refusal, match='pick_config'):
        S4.stage_candidate(tmp_path)


def test_candidate_held_folder_and_sidecar_arm(tmp_path):
    cand = S4.candidate(tmp_path, 'k6=0.0005')
    assert cand.label == 'k6=0.0005' and cand.held_dir == paths.full_root(tmp_path) / 'held' / 'k6=0.0005'
    assert cand.sidecar_arm == 'k6=0.0005'
    g = S4.candidate(tmp_path, 'G')
    assert g.held_dir == paths.RG_FULL / 'held' / 'GS-0.02' and g.sidecar_arm == 'GS-0.02' and g.label == 'G'


# ------------------------------------------------------------------ the verdict
class Fake:
    """A stage 4 world without the real arrays: labels, the candidate's partitions, T's, the emotion block."""

    def __init__(self, monkeypatch, outputs, labels, cand, t, d_emotion=0.012, lo_emotion=0.004):
        self.calls = []
        self.recorded = []
        monkeypatch.setattr(S4, 'held_labels', lambda: {a: labels for a in S4.ASPECTS})

        def candidate_fixed(c, scale, folder, aspect='emotion', recorded=None):
            self.recorded.append(('fixed', aspect, recorded))
            self.calls.append(('fixed', c.label, scale, Path(folder), aspect))
            return cand, {str(s): dict(path=f'{c.label}/{aspect}__s{s}', output_sha256=f'{s}' * 64) for s in S4.SEEDS}

        def score_scale(c, scale, cache, n, w, recorded=None, prefix=None):
            self.recorded.append(('score_scale', 'emotion', recorded))
            self.calls.append(('score_scale', c.label, scale, Path(cache), prefix))
            return dict(candidate=c.label, d=d_emotion, interval=[lo_emotion, 0.02], A_cand=.1, A_e={'B1-r20': .09},
                        outcome_id=S0.outcome_id(d_emotion, lo_emotion, scale, prefix), bootstrap_d=[],
                        eligibility=dict(eligible_seeds={a: list(S4.SEEDS) for a in (c.label,) + S4.BASELINES}), arms=[c.label, 'B1'])
        monkeypatch.setattr(S0, 'candidate_fixed', candidate_fixed)
        monkeypatch.setattr(S0, 'score_scale', score_scale)
        monkeypatch.setattr(S4, 't_fixed', lambda aspect: (t, dict(path='T', sha256='0' * 64)))


@pytest.fixture
def staged(tmp_path, monkeypatch, world):
    labels, cand, _, t = world
    put(tmp_path, 'stage3/verdict.json', dict(outcome='PICK', pick='k6=0.0005', pick_config=dict(sheet.parse('k6=0.0005'), remove=[]), smoke=False))
    write_gate(tmp_path)
    put(tmp_path, 'gates/v4a_train.json', {'pass': True, 'smoke': False, 'code_sha256': F.code_sha256()})
    put(tmp_path, 'full/held/k6=0.0005/v4c.json', dict(name='k6=0.0005', rows=paths.N_HELD,
        list_sha256=paths.HELD_SHA256,
        outputs={f'{a}__s{s}': f'a{i + s}' * 32 for i, a in enumerate(S4.ASPECTS) for s in S4.SEEDS},
        output_files={f'{a}__s{s}': f'{i + s}' * 64 for i, a in enumerate(S4.ASPECTS)
                                             for s in S4.SEEDS}, **{'pass': True}))
    fake = Fake(monkeypatch, tmp_path, labels, cand, t)
    return tmp_path, fake


def test_verdict_refuses_without_the_gate_before_anything_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(S4, 'stage_candidate', lambda *a, **k: pytest.fail('read the stage 3 file before the gate'))
    monkeypatch.setattr(S4, 'held_labels', lambda: pytest.fail('read the labels before the gate'))
    with pytest.raises(S0.Refusal, match='missing'):
        S4.verdict(tmp_path, 'k6=0.0005', n_resamples=5, workers=1)


def test_verdict_refuses_any_candidate_but_c_star(staged):
    outputs, fake = staged
    for bad in ('G', 'k2=8', 'G-no_buddy'):
        with pytest.raises(S0.Refusal, match='c\\*'):
            S4.verdict(outputs, bad, n_resamples=5, workers=1)
    assert not (outputs / 'stage4/verdict.json').exists() and not fake.calls


def test_verdict_under_keep_g_takes_g_only(tmp_path, monkeypatch, world):
    labels, cand, _, t = world
    put(tmp_path, 'stage3/verdict.json', dict(outcome='KEEP-G', pick=None, pick_config=None, smoke=False))
    write_gate(tmp_path)
    fake = Fake(monkeypatch, tmp_path, labels, cand, t)
    with pytest.raises(S0.Refusal):
        S4.verdict(tmp_path, 'k6=0.0005', n_resamples=5, workers=1)
    record = S4.verdict(tmp_path, 'G', n_resamples=5, workers=1)
    assert record['candidate'] == 'G' and record['kind'] == 'KEEP-G'
    assert {c[1] for c in fake.calls} == {'G'}


def test_verdict_has_the_plain_keys_the_re_derivation_reads(staged):
    outputs, fake = staged
    record = S4.verdict(outputs, 'k6=0.0005', n_resamples=7, workers=1)
    assert record['stage'] == 4 and record['candidate'] == 'k6=0.0005' and record['kind'] == 'PICK'
    assert record['n_resamples'] == 7 and record['held_sha256'] == paths.HELD_SHA256
    assert list(record['aspects']) == list(S4.ASPECTS)
    for aspect, block in record['aspects'].items():
        assert {'d', 'lo', 'hi', 'outcome', 'A_cand', 'A_T', 'd_in_ranges'} <= set(block), aspect
        assert block['outcome'] in ('T-WIN', 'T-LEVEL', 'T-BELOW')
    margin = record['emotion_margin']
    assert margin['outcome'] == 'PE-GO' and margin['lo'] == 0.004 and margin['hi'] == 0.02 and margin['d'] == 0.012
    assert record['sources']['upstream']['file'] == 'stage3/verdict.json'
    assert record['gates']['v4a_scorer'] == rb_score.sha_file(outputs / 'gates/v4a_scorer.json')
    assert 'reported' not in record
    assert json.loads((outputs / 'stage4/verdict.json').read_text()) == record


def test_verdict_scores_the_candidates_held_folder_with_the_pe_family(staged):
    outputs, fake = staged
    S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1)
    kinds = [c for c in fake.calls if c[0] == 'score_scale']
    assert kinds == [('score_scale', 'k6=0.0005', 'full', outputs / 'stage4/partitions', S4.PE_PREFIX)]
    fixed = [c for c in fake.calls if c[0] == 'fixed']
    assert [c[4] for c in fixed] == list(S4.ASPECTS) and all(c[2] == 'full' for c in fixed)
    assert all(c[3] == outputs / 'stage4/partitions/full/k6=0.0005' for c in fixed)


def test_verdict_is_written_once_and_a_second_call_must_agree(staged, monkeypatch):
    outputs, fake = staged
    first = S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1)
    assert S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1) == first
    monkeypatch.setattr(S4, 't_fixed', lambda aspect: ({0: {'r20': np.zeros((3, N), dtype=int)}}, dict(path='T', sha256='0' * 64)))
    with pytest.raises(AssertionError, match='refusing to replace'):
        S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1)


def test_verdict_writes_no_reported_block_and_does_not_score_margins(staged, monkeypatch):
    outputs, fake = staged
    monkeypatch.setattr(S4, 'per_baseline_margins', lambda *a, **k: pytest.fail('reported block before the verdict'))
    monkeypatch.setattr(S4, 'd_g_block', lambda *a, **k: pytest.fail('d_G before the verdict'))
    S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1)
    assert not (outputs / 'stage4/reported.json').exists()


def test_report_runs_only_after_the_verdict_file_exists(staged, monkeypatch):
    outputs, fake = staged
    with pytest.raises(S0.Refusal, match='verdict'):
        S4.report(outputs, n_resamples=5, workers=1)
    assert not (outputs / 'stage4/reported.json').exists()


def test_report_needs_the_gate_for_the_current_code(staged):
    outputs, fake = staged
    S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1)
    write_gate(outputs, **{'pass': False})
    with pytest.raises(S0.Refusal):
        S4.report(outputs, n_resamples=5, workers=1)


# ------------------------------------------------------------------ the scorer check against 12-27
def fake_frozen(d=.1, lo=.05, hi=.15):
    return dict(d=d, interval=[lo, hi], A_cand=.5, A_e={'T-r20': .4})


def test_check_items_compare_d_interval_and_scores_within_1e_6():
    ours = dict(d=.1, lo=.05, hi=.15, A_cand=.5, A_T=.4, bootstrap_d=[.1, .2])
    items = S4.compare_d_t('genre', ours, dict(d=.1 + 5e-7, interval=[.05, .15 - 5e-7], A_G=.5, A_T=.4,
                                               bootstrap_d=[.1, .2 + 5e-7]))
    assert set(items) == {'d_T/genre/d', 'd_T/genre/lo', 'd_T/genre/hi', 'd_T/genre/A_cand', 'd_T/genre/A_T',
                          'd_T/genre/bootstrap_d'}
    assert all(i['pass'] for i in items.values())
    for key, change in (('d', dict(d=.1 + 2e-6)), ('lo', dict(interval=[.05 + 2e-6, .15])),
                        ('hi', dict(interval=[.05, .15 + 2e-6])), ('A_cand', dict(A_G=.5 + 2e-6)),
                        ('A_T', dict(A_T=.4 - 2e-6)), ('bootstrap_d', dict(bootstrap_d=[.1, .2 + 2e-6]))):
        theirs = {**dict(d=.1, interval=[.05, .15], A_G=.5, A_T=.4, bootstrap_d=[.1, .2]), **change}
        bad = [k for k, v in S4.compare_d_t('genre', ours, theirs).items() if not v['pass']]
        assert bad == [f'd_T/genre/{key}'], key


def test_check_reads_genre_from_12_27s_verdict_and_emotion_style_from_its_reported_margin_T():
    theirs = S4.frozen_d_t()
    assert sorted(theirs) == sorted(S4.ASPECTS)
    for aspect in S4.ASPECTS:
        assert {'d', 'interval', 'A_G', 'A_T'} <= set(theirs[aspect]), aspect
    verdict = json.loads(S0.G_FULL_VERDICT.read_text())['d_T']
    reported = json.loads(S0.G_FULL_RECORD.read_text())['G']['aspects']
    assert theirs['genre']['d'] == verdict['d'] and theirs['genre']['bootstrap_d'] == verdict['bootstrap_d']
    for aspect in ('emotion', 'style'):
        m = reported[aspect]['margins']['T']
        assert theirs[aspect]['d'] == m['d'] and theirs[aspect]['A_T'] == m['A_e']['T-r20']
        assert theirs[aspect]['interval'] == m['interval']


def test_check_gate_records_smoke_and_the_largest_difference(tmp_path, monkeypatch, world):
    labels, cand, _, t = world
    monkeypatch.setattr(S4, 'held_labels', lambda: {a: labels for a in S4.ASPECTS})
    monkeypatch.setattr(S0, 'candidate_fixed', lambda c, scale, folder, aspect='emotion', recorded=None: (cand, {}))
    monkeypatch.setattr(S4, 't_fixed', lambda aspect: (t, {}))
    monkeypatch.setattr(S4, 'frozen_partitions_equal', lambda aspect, fixed: {'pass': True, 'equal': {}})
    ours = {a: S4.d_t_block(cand, t, labels, a, 5, 1) for a in S4.ASPECTS}
    frozen = {a: dict(d=ours[a]['d'] + (3e-7 if a == 'style' else 0), interval=[ours[a]['lo'], ours[a]['hi']],
                      A_G=ours[a]['A_cand'], A_T=ours[a]['A_T'], bootstrap_d=ours[a]['bootstrap_d']) for a in S4.ASPECTS}
    monkeypatch.setattr(S4, 'frozen_d_t', lambda: frozen)
    gate = S4.check_v4a(tmp_path, workers=1, n_resamples=5, smoke=True)
    assert gate['smoke'] is True and gate['pass'] is True and gate['n_resamples'] == 5
    assert sorted(k for k in gate['items'] if k.startswith('d_T/genre')) == [
        'd_T/genre/A_T', 'd_T/genre/A_cand', 'd_T/genre/d']                     # a smoke: point values only
    assert abs(gate['max_abs_diff'] - 3e-7) < 1e-12 and gate['code_sha256'] == S4.code_sha256()
    assert json.loads((tmp_path / 'gates/v4a_scorer.json').read_text()) == gate
    with pytest.raises(S0.Refusal, match='smoke'):
        S4.require_v4a_scorer(tmp_path)
    frozen['genre']['d'] += 1e-5
    gate = S4.check_v4a(tmp_path, workers=1, n_resamples=5, smoke=True)
    assert gate['pass'] is False and gate['items']['d_T/genre/d']['pass'] is False


def test_a_real_check_is_never_shrunk(tmp_path):
    with pytest.raises(AssertionError, match='never shrunk'):
        S4.check_v4a(tmp_path, workers=1, n_resamples=5, smoke=False)


# ------------------------------------------------------------------ the reported block
def test_report_blocks_give_vs_all_per_baseline_and_d_g(world):
    labels, cand, other, _ = world
    comp = {a: {s: {'r20': other[s]['r20'], 'own': other[s]['r20'][:1]} for s in S4.SEEDS} for a in ('B1', 'B4')}
    comp.update(B2={}, B3={}, B5={}, B6={})
    eligible = {'k2=8': list(S4.SEEDS), 'B1': list(S4.SEEDS), 'B2': [], 'B3': [], 'B4': [0, 2, 4], 'B5': [], 'B6': []}
    by_aspect = {a: cand for a in S4.ASPECTS}
    g_by_aspect = {a: other for a in S4.ASPECTS}
    emotion_margin = dict(d=.01, lo=.0, hi=.02)
    got = S4.report_blocks('k2=8', by_aspect, comp, eligible, {a: labels for a in S4.ASPECTS}, emotion_margin,
                           g_by_aspect, 6, 1)
    assert list(got) == list(S4.ASPECTS)
    assert got['emotion']['vs_B1_to_B6'] is emotion_margin                  # emotion's is the verdict's
    ref = rb_part2_report.margin({**comp, 'k2=8': cand}, labels, eligible, ('k2=8',) + S4.BASELINES, 6, 1)
    for aspect in ('style', 'genre'):
        m = got[aspect]['vs_B1_to_B6']
        assert m['d'] == ref['d'] and [m['lo'], m['hi']] == ref['interval'] and m['max_entry'] == ref['max_entry']
        assert set(got[aspect]['per_baseline']) == set(S4.BASELINES) and got[aspect]['per_baseline']['B2'] is None
    for aspect in S4.ASPECTS:
        assert got[aspect]['b_matched'] == got[aspect]['per_baseline']['B1']
    d_g = got['genre']['d_G']
    assert set(d_g) == {'d', 'lo', 'hi', 'A_cand', 'A_G'} and abs(d_g['d'] - (d_g['A_cand'] - d_g['A_G'])) < 1e-12
    none = S4.report_blocks('G', by_aspect, comp, dict(eligible, G=eligible['k2=8']), {a: labels for a in S4.ASPECTS},
                            emotion_margin, None, 6, 1)
    assert all(b['d_G'] is None for b in none.values())                      # the candidate is G: nothing to subtract


@pytest.mark.parametrize('folder', [paths.ROUND, paths.TEST])
def test_code_sha256_is_independent_of_the_working_directory(monkeypatch, folder):
    monkeypatch.chdir(paths.ROUND.parents[2])
    expected = S4.code_sha256()
    monkeypatch.chdir(folder)
    assert S4.code_sha256() == expected


def test_verdict_and_v4a_use_each_aspects_labels_and_instructed_image(staged, monkeypatch, world):
    outputs, _ = staged
    labels, cand, _, t = world
    # Different row orders change the AMI; relabelling cluster IDs alone would not.
    ys = {a: np.roll(labels, i * 9) for i, a in enumerate(S4.ASPECTS)}
    ts = {a: {0: {'r20': np.roll(t[0]['r20'], i * 7, axis=1)}} for i, a in enumerate(S4.ASPECTS)}
    monkeypatch.setattr(S4, 'held_labels', lambda: ys)
    monkeypatch.setattr(S4, 't_fixed', lambda a: (ts[a], dict(path=a, sha256=a)))
    expected = {a: S4.d_t_block(cand, ts[a], ys[a], a, 5, 1) for a in S4.ASPECTS}
    record = S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1)
    for a in S4.ASPECTS:
        for key in ('d', 'lo', 'hi', 'A_cand', 'A_T', 'labelled_rows', 'outcome'):
            assert record['aspects'][a][key] == expected[a][key], (a, key)
    frozen = {a: dict(d=e['d'], interval=[e['lo'], e['hi']], A_G=e['A_cand'], A_T=e['A_T'],
                      bootstrap_d=e['bootstrap_d']) for a, e in expected.items()}
    monkeypatch.setattr(S4, 'frozen_d_t', lambda: frozen)
    monkeypatch.setattr(S4, 'frozen_partitions_equal', lambda *a: {'pass': True})
    gate = S4.check_v4a(outputs, workers=1, n_resamples=5, smoke=True)
    assert gate['pass'] is True
    assert gate['ours'] == record['aspects']


@pytest.mark.parametrize('change', [dict(smoke=True), dict(code_sha256={'rs_stage4.py': 'stale'})])
def test_full_gate_refuses_smoke_or_stale_scorer(tmp_path, change):
    import rs_full as F
    put(tmp_path, 'gates/v4a_train.json', {'pass': True, 'smoke': False, 'code_sha256': F.code_sha256()})
    write_gate(tmp_path, **change)
    with pytest.raises(S0.Refusal):
        F.require_v4a(tmp_path)


@pytest.mark.parametrize('change', [dict(smoke=True), {'pass': False}, dict(code_sha256={'rs_full.py': 'stale'})])
def test_pick_needs_the_current_training_gate_before_any_held_read(staged, change):
    outputs, fake = staged
    path = outputs / 'gates/v4a_train.json'
    put(outputs, 'gates/v4a_train.json', dict(json.loads(path.read_text()), **change))
    with pytest.raises((S0.Refusal, AssertionError)):
        S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1)
    assert fake.calls == []


def test_pick_binds_all_reads_to_v4c_and_records_both_gates(staged):
    outputs, fake = staged
    record = S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1)
    v4c = json.loads((outputs / 'full/held/k6=0.0005/v4c.json').read_text())
    for kind, aspect, recorded in fake.recorded:
        assert recorded == {str(s): v4c['output_files'][f'{aspect}__s{s}'] for s in S4.SEEDS}, (kind, aspect)
    for gate, rel in (('v4a_train', 'gates/v4a_train.json'), ('v4c', 'full/held/k6=0.0005/v4c.json')):
        assert record['gates'][gate] == S4.sha_file(outputs / rel)


@pytest.mark.parametrize('failure', ['training', 'held', 'v4c'])
def test_failed_pick_has_no_outcome_and_preserves_the_cause_without_scoring(staged, failure):
    outputs, fake = staged
    if failure == 'training':
        put(outputs, 'full/train/k6=0.0005/failed__s2.json', dict(cause='non-finite loss in style'))
    elif failure == 'held':
        put(outputs, 'full/held/k6=0.0005/failed__style__s2.json', dict(problems=['not finite', 'not finite']))
    else:
        put(outputs, 'full/held/k6=0.0005/v4c.json', {'pass': False, 'cause': 'style s2 failed twice'})
    if failure != 'v4c':
        path = outputs / 'full/held/k6=0.0005/v4c.json'
        path.rename(path.with_name('v4c.saved.json'))
    record = S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1)
    assert all(b['outcome'] is None for b in record['aspects'].values())
    assert record['emotion_margin']['outcome'] is None
    assert 'style' in record['v_failure']
    assert fake.calls == []
    assert json.loads((outputs / 'stage4/verdict.json').read_text()) == record
    with pytest.raises(S0.Refusal, match='no outcome'):
        S4.report(outputs, n_resamples=5, workers=1)


def test_pick_needs_a_complete_v4c_record(staged):
    outputs, fake = staged
    path = outputs / 'full/held/k6=0.0005/v4c.json'
    record = json.loads(path.read_text())
    record['output_files'].pop('style__s2')
    put(outputs, 'full/held/k6=0.0005/v4c.json', record)
    with pytest.raises(S0.Refusal, match='V4c'):
        S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1)
    assert fake.calls == []


def test_verdict_never_shrinks_a_real_held_claim(staged, monkeypatch):
    outputs, fake = staged
    monkeypatch.setattr(paths, 'smoke_ok', lambda out: False)
    with pytest.raises(AssertionError, match='2,000'):
        S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1)
    assert fake.calls == [] and not (outputs / 'stage4/verdict.json').exists()


def test_shrunk_scratch_verdict_is_marked_smoke(staged):
    outputs, _ = staged
    assert paths.smoke_ok(outputs)
    record = S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1)
    assert record['smoke'] is True


def test_t_fixed_reads_each_aspects_own_partition_file(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'RB_PART2', tmp_path)
    for i, aspect in enumerate(S4.ASPECTS):
        path = tmp_path / f'partitions/full/T/{aspect}__s0__r20.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, labels=np.full((3, paths.N_HELD), i, dtype=np.int32))
        put(tmp_path, str(path.with_suffix('.json').relative_to(tmp_path)), dict(
            partitions_sha256=S4.sha_file(path), identity=dict(rows=paths.N_HELD, held_sha256=paths.HELD_SHA256)))
    for i, aspect in enumerate(S4.ASPECTS):
        fixed, source = S4.t_fixed(aspect)
        assert np.all(fixed[0]['r20'] == i), aspect
        assert source['path'].endswith(f'{aspect}__s0__r20.npz')
        assert source['sha256'] == S4.sha_file(source['path'])


def test_check_v4a_rejects_a_changed_real_partition_file(tmp_path, monkeypatch, world):
    labels, cand, _, t = world
    Fake(monkeypatch, tmp_path, labels, cand, t)
    monkeypatch.setattr(paths, 'RG_STAGE_C', tmp_path / 'frozen')
    for aspect in S4.ASPECTS:
        for seed in S4.SEEDS:
            path = paths.RG_STAGE_C / f'partitions/{paths.G_ARM}/{aspect}__s{seed}__r20.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, labels=cand[seed]['r20'])
    ours = {a: S4.d_t_block(cand, t, labels, a, 5, 1) for a in S4.ASPECTS}
    monkeypatch.setattr(S4, 'frozen_d_t', lambda: {a: dict(d=e['d'], A_G=e['A_cand'], A_T=e['A_T'],
        interval=[e['lo'], e['hi']], bootstrap_d=e['bootstrap_d']) for a, e in ours.items()})
    assert S4.check_v4a(tmp_path, workers=1, n_resamples=5, smoke=True)['pass'] is True
    path = paths.RG_STAGE_C / f'partitions/{paths.G_ARM}/style__s2__r20.npz'
    changed = cand[2]['r20'].copy()
    changed[0, 0] += 1
    np.savez(path, labels=changed)
    gate = S4.check_v4a(tmp_path, workers=1, n_resamples=5, smoke=True)
    assert gate['pass'] is False and gate['failing'] == ['partitions/style']
    assert gate['items']['partitions/style']['equal']['2'] is False


@pytest.mark.parametrize('changed,allowed', [('rs_train.py', True), ('rb_score.py', False)])
def test_report_compares_only_scorer_files_and_records_both_code_stamps(staged, monkeypatch, changed, allowed):
    outputs, _ = staged
    v = S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1)
    now = dict(v['code_sha256'], **{changed: '0' * 64})
    monkeypatch.setattr(S4, 'code_sha256', lambda: now)
    write_gate(outputs)
    rcol = put(outputs, 'rcol.json', {})
    monkeypatch.setattr(rb_score, 'read_rcol_eligible', lambda scale: ({}, rcol))
    monkeypatch.setattr(rb_score, 'eligible_seeds', lambda *a: v['emotion_margin']['eligible_seeds'])
    monkeypatch.setattr(rg_held, '_frozen_comparator_partitions', lambda *a: {})
    monkeypatch.setattr(S4, 'report_blocks', lambda *a, **k: {'genre': {}})
    monkeypatch.setattr(S4, 'genre_leiden', lambda *a, **k: {})
    if not allowed:
        with pytest.raises((S0.Refusal, AssertionError), match='scor|code'):
            S4.report(outputs, n_resamples=5, workers=1)
        assert not (outputs / 'stage4/reported.json').exists()
    else:
        record = S4.report(outputs, n_resamples=5, workers=1)
        assert record['verdict_code_sha256'] == v['code_sha256']
        assert record['code_sha256'] == now



def test_verdict_refuses_a_held_file_that_changed_since_v4c(staged, monkeypatch):
    outputs, _ = staged
    path = outputs / 'full/held/k6=0.0005/style__s0.npz'
    path.write_bytes(b'changed held output')
    monkeypatch.setattr(S0, 'candidate_fixed', CANDIDATE_FIXED)
    monkeypatch.setattr(S0, 'load_candidate', lambda cand, scale, aspect, seed:
                        (np.zeros((N, 2)), S4.sha_file(path)))
    monkeypatch.setattr(rb_score, 'fixed_partitions', lambda *a, **k: pytest.fail('partitioning before hash check'))
    with pytest.raises(AssertionError, match='SHA-256 differs'):
        S4.verdict(outputs, 'k6=0.0005', n_resamples=5, workers=1)
    assert not (outputs / 'stage4/verdict.json').exists()
