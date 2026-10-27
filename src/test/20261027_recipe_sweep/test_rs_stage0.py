"""Ticket 05: stage 0. CPU only, synthetic data and tiny fake outputs; the checks on the real arrays are the gates
(`rs_stage0 check v0a`, `held`, `check v0c`, `check v0d`)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

import rs_paths as paths
import rs_sheet as sheet
import rs_stage0 as S0
import rs_train
import rb_score
import rb_student

ROWS = 40                                   # a stand-in for H's 12,281 rows (N_HELD is patched)
NAMES = [f'p{i}' for i in range(ROWS)]


# ------------------------------------------------------------------ outcomes and E0
@pytest.mark.parametrize('d,lower,scale,expected', [
    (0.01, 1e-12, 'sample', 'G-ES-GO'),         # d at its edge, lower just above 0
    (0.01, 0.0, 'sample', 'G-ES-MM'),           # lower at its edge: not above 0
    (0.01, -0.003, 'full', 'G-EF-MM'),
    (0.5, 0.2, 'full', 'G-EF-GO'),
    (0.0099999999, 0.004, 'sample', 'G-ES-IN'),  # d just below 0.01 is IN whatever the bound
    (1e-12, -0.01, 'full', 'G-EF-IN'),
    (0.0, -0.01, 'sample', 'G-ES-NO'),          # d = 0 is NO
    (-0.02, -0.05, 'full', 'G-EF-NO'),
    (0.0, 0.5, 'full', 'G-EF-NO'),
])
def test_outcome_thresholds_at_their_edges(d, lower, scale, expected):
    assert S0.outcome_id(d, lower, scale) == expected


def test_outcome_refuses_non_finite_and_unknown_scale():
    with pytest.raises(AssertionError):
        S0.outcome_id(float('nan'), 0.0, 'sample')
    with pytest.raises(AssertionError):
        S0.outcome_id(0.02, 0.01, 'held')


@pytest.mark.parametrize('sample,full,expected', [
    ('G-ES-GO', 'G-EF-GO', True), ('G-ES-MM', 'G-EF-GO', True), ('G-ES-IN', 'G-EF-GO', True),
    ('G-ES-NO', 'G-EF-GO', False),              # the sample scale is G-ES-NO
    (None, 'G-EF-GO', False),                   # no sample outcome
    ('G-ES-GO', 'G-EF-MM', False), ('G-ES-GO', 'G-EF-IN', False), ('G-ES-GO', 'G-EF-NO', False),
    ('G-ES-GO', None, False),
])
def test_e0_rule(sample, full, expected):
    assert S0.e0(dict(sample=sample, full=full)) is expected


# ------------------------------------------------------------------ candidate wiring
def partitions(rng, n, readings, n_classes=6):
    return {r: rng.integers(0, n_classes, size=(3 if r == 'r20' else 1, n)) for r in readings}


@pytest.fixture(scope='module')
def synthetic():
    rng = np.random.default_rng(7)
    n = 150
    labels = rng.integers(0, 5, size=n)
    labels[::11] = -1
    # the candidate's partitions follow the labels loosely so that d is not noise
    cand = {s: {'r20': np.where(rng.random((3, n)) < .5, labels % 6, rng.integers(0, 6, (3, n)))} for s in S0.SEEDS}
    comp = {arm: {s: partitions(rng, n, ('r20', 'own')) for s in S0.SEEDS} for arm in ('B1', 'B4', 'B5', 'B6')}
    comp['B2'], comp['B3'] = {}, {}
    rcol = {'B2': [], 'B3': [], 'B4': [0, 1, 2, 3, 4], 'B5': [0, 2, 4]}
    for s in (1, 3):
        del comp['B5'][s]
    return labels, cand, comp, rcol


def test_candidate_is_first_then_b1_to_b6():
    assert S0.arms_for('G') == ('G', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6')
    assert S0.arms_for('K4') == rb_score.arms_for('K4')      # G takes K4's place: no matched arm


def test_g_in_k4s_place_gives_the_rb_block(synthetic):
    labels, cand, comp, rcol = synthetic
    block = S0.assemble('G', cand, comp, rcol, {}, labels, 'sample', n_resamples=40, workers=1)
    arms = rb_score.arms_for('K4')
    eligible = rb_score.eligible_seeds(arms, rcol)
    ref = rb_score.candidate_block(dict(comp, K4=cand), labels, eligible, arms, 'sample', 40, 1)
    assert block['candidate'] == 'G' and block['arms'] == ['G'] + list(arms[1:])
    for key in ('d', 'A_cand', 'A_e', 'interval', 'bootstrap_d'):
        assert block[key] == ref[key], key
    assert block['outcome_id'] == S0.outcome_id(ref['d'], ref['interval'][0], 'sample')
    assert block['eligibility']['eligible_seeds'] == {
        'G': [0, 1, 2, 3, 4], 'B1': [0, 1, 2, 3, 4], 'B2': [], 'B3': [], 'B4': [0, 1, 2, 3, 4], 'B5': [0, 2, 4],
        'B6': [0, 1, 2, 3, 4]}


def test_candidate_seeds_are_0_to_4_and_bad_inputs_are_refused(synthetic):
    labels, cand, comp, rcol = synthetic
    with pytest.raises(AssertionError):                       # a missing candidate seed
        S0.assemble('G', {s: p for s, p in cand.items() if s != 3}, comp, rcol, {}, labels, 'sample', 10, 1)
    with pytest.raises(AssertionError):                       # the candidate's label cannot be a comparator
        S0.assemble('B1', cand, comp, rcol, {}, labels, 'sample', 10, 1)


def test_draw_order_is_i_b_then_s_b_for_g_then_b1_to_b6(synthetic):
    labels, cand, comp, rcol = synthetic
    arms = S0.arms_for('G')
    eligible = rb_score.eligible_seeds(arms, rcol)
    draws = rb_score.draw_bootstrap(len(labels), eligible, arms, 5)
    rng = np.random.default_rng(0)
    for rows, seeds in draws:
        assert np.array_equal(rows, rng.integers(0, len(labels), size=len(labels)))
        for arm in arms:
            if eligible[arm]:
                assert np.array_equal(seeds[arm], rng.choice(eligible[arm], size=len(eligible[arm]), replace=True))
            else:
                assert arm not in seeds
    assert list(draws[0][1]) == ['G', 'B1', 'B4', 'B5', 'B6']


def test_nonfinite_paper_seed_leaves_the_entry(synthetic):
    labels, cand, comp, rcol = synthetic
    comp = dict(comp, B4={s: p for s, p in comp['B4'].items() if s != 2})
    block = S0.assemble('G', cand, comp, rcol, {'B4': [2]}, labels, 'full', n_resamples=10, workers=1)
    assert block['eligibility']['eligible_seeds']['B4'] == [0, 1, 3, 4]
    assert block['outcome_id'].startswith('G-EF-')


# ------------------------------------------------------------------ a candidate's held outputs
@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(S0, 'N_HELD', ROWS)
    monkeypatch.setattr(rb_student, 'held_names', lambda: (NAMES, S0.HELD_SHA256))


def unit(seed, rows=ROWS):
    x = np.random.default_rng(seed).normal(size=(rows, S0.WIDTH)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def write_output(folder, arm_, scale_, aspect_, seed_, x=None, **meta):
    """A sidecar-carrying output; `meta` overrides any field of the sidecar (a wrong arm, scale, aspect, seed, ...)."""
    x = unit(seed_) if x is None else x
    info = dict(kind='held', arm=arm_, scale=scale_, aspect=aspect_, seed=seed_, rows=ROWS, width=S0.WIDTH,
                dtype='float16', list_sha256=S0.HELD_SHA256, sha256=rb_student.sha_array(x))
    info.update(meta)
    rb_student.write_outputs(Path(folder) / f'{aspect_}__s{seed_}.npz', dict(img=x), info)


def write_set(folder, arm, scale):
    for aspect in S0.ASPECTS:
        for seed in S0.SEEDS:
            write_output(folder, arm, scale, aspect, seed)


def test_load_candidate_reads_a_valid_output(small, tmp_path):
    write_set(tmp_path, 'G', 'sample')
    cand = S0.Candidate('G', tmp_path, 'G')
    x, sha = S0.load_candidate(cand, 'sample', 'emotion', 2)
    assert x.dtype == np.float32 and x.shape == (ROWS, S0.WIDTH)
    assert sha == rb_score.sha_file(tmp_path / 'emotion__s2.npz')


@pytest.mark.parametrize('bad', [dict(arm='K4'), dict(scale='full'), dict(aspect='style'), dict(seed=3),
                                 dict(list_sha256='0' * 64), dict(rows=ROWS + 1), dict(dtype='float32')])
def test_load_candidate_checks_the_sidecar(small, tmp_path, bad):
    write_output(tmp_path, 'G', 'sample', 'emotion', 2, **bad)
    with pytest.raises(AssertionError, match='sidecar'):
        S0.load_candidate(S0.Candidate('G', tmp_path, 'G'), 'sample', 'emotion', 2)


@pytest.mark.parametrize('make,message', [
    (lambda x: x * np.float16(1.5), 'V0c'),                                      # not unit norm
    (lambda x: np.where(np.arange(ROWS)[:, None] == 3, np.float16('nan'), x), 'V0c'),    # a non-finite row
    (lambda x: x[:-1], 'V0c|sidecar'),                                           # a row missing
    (lambda x: x.astype(np.float32), 'V0c|sidecar'),                             # wrong dtype
])
def test_load_candidate_applies_v0c(small, tmp_path, make, message):
    x = make(unit(1))
    info = dict(rows=len(x), dtype=str(x.dtype))
    write_output(tmp_path, 'G', 'sample', 'emotion', 1, x=x, **info)
    with pytest.raises(AssertionError, match=message):
        S0.load_candidate(S0.Candidate('G', tmp_path, 'G'), 'sample', 'emotion', 1)


def test_load_candidate_refuses_a_missing_file_a_failure_record_and_a_changed_array(small, tmp_path):
    cand = S0.Candidate('G', tmp_path, 'G')
    with pytest.raises(AssertionError, match='missing'):
        S0.load_candidate(cand, 'sample', 'emotion', 0)
    write_output(tmp_path, 'G', 'sample', 'emotion', 0)
    np.savez(tmp_path / 'emotion__s0.npz', img=unit(99))          # the array no longer matches its sidecar
    with pytest.raises(AssertionError, match='sha256'):
        S0.load_candidate(cand, 'sample', 'emotion', 0)
    write_output(tmp_path, 'G', 'sample', 'emotion', 1)
    S0.failed_path(tmp_path, 'emotion', 1).write_text('{}')
    with pytest.raises(AssertionError, match='V0d'):
        S0.load_candidate(cand, 'sample', 'emotion', 1)


def test_held_folder_is_a_parameter(small, tmp_path):
    # ticket 09 passes c*'s full-scale folder; the same loader reads it under its own sidecar arm
    write_set(tmp_path / 'c_star', 'c*', 'full')
    cand = S0.Candidate('c*', tmp_path / 'c_star', 'c*')
    assert S0.load_candidate(cand, 'full', 'genre', 4)[0].shape == (ROWS, S0.WIDTH)
    assert S0.arms_for(cand.label)[0] == 'c*'


def test_v0c_items_list_every_file(small, tmp_path):
    write_set(tmp_path, 'G', 'sample')
    write_output(tmp_path, 'G', 'sample', 'style', 3, x=unit(3) * np.float16(2))
    items = S0.v0c_items(S0.Candidate('G', tmp_path, 'G'), 'sample')
    assert len(items) == 15 and [k for k, v in items.items() if not v['pass']] == ['style__s3']
    assert 'V0c' in items['style__s3']['problem']
    (tmp_path / 'genre__s0.npz').unlink()
    assert not S0.v0c_items(S0.Candidate('G', tmp_path, 'G'), 'sample')['genre__s0']['pass']


# ------------------------------------------------------------------ gates
def test_read_gate_refuses_missing_failed_and_stale(tmp_path, monkeypatch):
    with pytest.raises(S0.Refusal, match='missing'):
        S0.read_gate(tmp_path, 'v0a')
    S0._atomic_json(S0.gate_path(tmp_path, 'v0a'), dict(**{'pass': False}, code_sha256=S0.code_sha256()))
    with pytest.raises(S0.Refusal, match='failed'):
        S0.read_gate(tmp_path, 'v0a')
    S0._atomic_json(S0.gate_path(tmp_path, 'v0a'), dict(**{'pass': True}, code_sha256='0' * 64))
    with pytest.raises(S0.Refusal, match='other stage 0 code'):
        S0.read_gate(tmp_path, 'v0a')
    S0._atomic_json(S0.gate_path(tmp_path, 'v0a'), dict(**{'pass': True}, code_sha256=S0.code_sha256()))
    assert S0.read_gate(tmp_path, 'v0a')['pass'] is True


def test_code_files_cover_the_scorer_and_the_inference():
    assert set(S0.code_files()) == set(S0.CODE_MODULES)
    assert {'rs_stage0', 'rb_score', 'rb_student', 'rb_pick', 'ct_score'} <= set(S0.code_files())
    # the trainer's modules, and every file `rb_student.code_hashes()` names (the model behind `rb_student.infer`)
    assert {'rg_train', 'rg_variants', 'rs_sheet', 'rs_train'} <= set(S0.code_files())
    behind_infer = {Path(name).stem for name in rb_student.code_hashes()}
    assert behind_infer <= set(S0.code_files()), behind_infer - set(S0.code_files())


def test_outcome_family_name_is_a_parameter():
    assert S0.outcome_id(0.02, 0.01, 'full') == 'G-EF-GO'
    prefix = {'sample': 'C-ES', 'full': 'C-EF'}
    assert S0.outcome_id(0.02, 0.01, 'full', prefix) == 'C-EF-GO'
    assert S0.outcome_id(-0.02, -0.01, 'sample', prefix) == 'C-ES-NO'


def fake_k4_block(scale):
    """A stand-in for 12-24's K4 block of its stage verdict (2,000 bootstrap d, as the scorer check asserts)."""
    return dict(candidate='K4', scale=scale, d=0.012, A_cand=0.2, A_e={'B1-r20': 0.11, 'B1-own': 0.1, 'B4-r20': 0.12},
                interval=[0.002, 0.021], bootstrap_d=[float(v) for v in np.linspace(0.0, 0.024, 2000)],
                outcome_id='K4-ES-GO' if scale == 'sample' else 'K4-EF-GO')


@pytest.fixture
def rb_verdicts(tmp_path, monkeypatch):
    """12-24's two stage verdicts, small: the entries V0a must hold come from them."""
    files = {}
    for scale in S0.SCALES:
        files[scale] = tmp_path / 'rb' / f'verdict_{scale}.json'
        files[scale].parent.mkdir(exist_ok=True)
        files[scale].write_text(json.dumps(dict(held_sha256=S0.HELD_SHA256, scale=scale,
                                                blocks=dict(K4=fake_k4_block(scale)))))
    monkeypatch.setattr(S0, 'RB_VERDICT', files)
    return files


def items_for(part, ok=True):
    return {name: {'pass': ok} for name in S0.v0a_groups()[part]}


def test_v0a_passes_only_with_both_parts_complete_and_current(tmp_path, monkeypatch, rb_verdicts):
    assert S0.write_v0a(tmp_path, 'cpu', items_for('cpu'))['pass'] is False         # the gpu part is missing
    record = S0.write_v0a(tmp_path, 'gpu', items_for('gpu'))
    assert record['pass'] is True and set(record['parts']) == {'cpu', 'gpu'}
    assert S0.write_v0a(tmp_path, 'gpu', items_for('gpu', ok=False))['pass'] is False   # a failing item
    assert S0.write_v0a(tmp_path, 'gpu', {k: v for k, v in items_for('gpu').items() if 'genre' not in k})['pass'] is False
    S0.write_v0a(tmp_path, 'gpu', items_for('gpu'))
    monkeypatch.setattr(S0, 'code_sha256', lambda: 'changed')                         # code changed after the cpu part
    assert S0.write_v0a(tmp_path, 'gpu', items_for('gpu'))['pass'] is False
    assert S0.write_v0a(tmp_path, 'cpu', items_for('cpu'))['pass'] is True


def test_v0a_expects_the_checks_the_rule_names(rb_verdicts):
    groups = S0.v0a_groups()
    cpu, gpu = groups['cpu'], groups['gpu']
    assert {'g_full/outputs_sha256', 'g_full/weights_sha256', 'g_sample/weights'} <= set(cpu)
    for scale in S0.SCALES:                     # every scorer-check item the rule names, at each scale
        assert {f'scorer/{scale}/{k}' for k in ('d', 'A_cand', 'interval_lo', 'interval_hi', 'entries', 'bootstrap_d',
                                                 'outcome', 'partitions_equal', 'A_e/B1-r20', 'A_e/B1-own',
                                                 'A_e/B4-r20')} <= set(cpu)
    assert {f'g_full/seed0_inference/{a}' for a in S0.ASPECTS} | {
        f'sample/r6_5_s1_inference/{a}' for a in S0.ASPECTS} == set(gpu)


def test_v0a_part_fails_when_any_named_item_is_missing(tmp_path, rb_verdicts):
    S0.write_v0a(tmp_path, 'gpu', items_for('gpu'))
    for name in S0.v0a_groups()['cpu']:
        items = {k: v for k, v in items_for('cpu').items() if k != name}
        record = S0.write_v0a(tmp_path, 'cpu', items)
        assert record['parts']['cpu']['pass'] is False and record['pass'] is False, name
    assert S0.write_v0a(tmp_path, 'cpu', items_for('cpu'))['pass'] is True


def test_compare_blocks_tolerance_is_1e_6(synthetic):
    labels, cand, comp, rcol = synthetic
    block = json.loads(json.dumps(S0.assemble('G', cand, comp, rcol, {}, labels, 'sample', 12, 1)))
    assert all(v['pass'] for v in S0.compare_blocks(block, block).values())
    other = json.loads(json.dumps(block))
    other['d'] += 5e-7
    other['A_e']['B1-r20'] += 2e-6
    other['interval'][1] += 9e-7
    items = S0.compare_blocks(block, other)
    assert items['d']['pass'] and items['interval_hi']['pass'] and not items['A_e/B1-r20']['pass']
    del other['A_e']['B6-own']
    assert not S0.compare_blocks(block, other)['entries']['pass']
    other = json.loads(json.dumps(block))
    other['bootstrap_d'][3] += 1e-3
    assert not S0.compare_blocks(block, other)['bootstrap_d']['pass']
    other = json.loads(json.dumps(block))
    other['outcome_id'] = 'G-ES-NO' if block['outcome_id'] != 'G-ES-NO' else 'G-ES-GO'
    assert not S0.compare_blocks(block, other)['outcome']['pass']


def test_compare_blocks_fails_on_each_field_changed_once():
    """Rule §3.1: d, A_cand, each A_e, both bounds, and (here also) the entries, every bootstrap d and the outcome."""
    theirs = fake_k4_block('sample')
    ours = lambda: json.loads(json.dumps(theirs))
    assert all(v['pass'] for v in S0.compare_blocks(ours(), theirs).values())
    changes = {
        'd': lambda b: b.update(d=b['d'] + 1e-5),
        'A_cand': lambda b: b.update(A_cand=b['A_cand'] - 1e-5),
        'interval_lo': lambda b: b['interval'].__setitem__(0, b['interval'][0] + 1e-5),
        'interval_hi': lambda b: b['interval'].__setitem__(1, b['interval'][1] + 1e-5),
        'entries': lambda b: b['A_e'].pop('B4-r20'),
        'bootstrap_d': lambda b: b['bootstrap_d'].__setitem__(7, b['bootstrap_d'][7] + 1e-5),
        'outcome': lambda b: b.update(outcome_id='G-ES-NO'),
        **{f'A_e/{e}': (lambda e: lambda b: b['A_e'].__setitem__(e, b['A_e'][e] + 1e-5))(e) for e in theirs['A_e']},
    }
    assert {k for k in changes} == {f for f in S0.compare_blocks(ours(), theirs)}
    for item, change in changes.items():
        block = ours()
        change(block)
        result = S0.compare_blocks(block, theirs)
        assert result[item]['pass'] is False, item
        failing = {k for k, v in result.items() if not v['pass']}
        assert failing == {item} | ({'A_e/B4-r20'} if item == 'entries' else set()), (item, failing)


def test_scorer_check_binds_the_items_and_compares_the_partitions_to_12_24s(tmp_path, monkeypatch, rb_verdicts):
    part2 = tmp_path / 'part2'
    cache = tmp_path / 'cache'
    rng = np.random.default_rng(3)
    for scale in S0.SCALES:
        for seed in S0.SEEDS:
            labels = rng.integers(0, 6, size=(3, 30))
            for root in (part2 / 'partitions' / scale / 'K4', cache / scale / 'K4'):
                root.mkdir(parents=True, exist_ok=True)
                np.savez(root / f'emotion__s{seed}__r20.npz', labels=labels)
    monkeypatch.setattr(paths, 'RB_PART2', part2)
    monkeypatch.setattr(S0, 'score_scale', lambda *a, **k: fake_k4_block('sample'))
    items = {}
    for scale in S0.SCALES:
        items.update(S0.scorer_check(scale, cache, 2000, 1))
    assert set(S0.v0a_groups()['cpu']) & {k for k in items} == {k for k in S0.v0a_groups()['cpu'] if k.startswith('scorer/')}
    assert all(v['pass'] for v in items.values())
    with np.load(cache / 'full' / 'K4' / 'emotion__s3__r20.npz') as z:       # one partition now differs from 12-24's
        labels = z['labels'].copy()
    labels[0, 0] = (labels[0, 0] + 1) % 6
    np.savez(cache / 'full' / 'K4' / 'emotion__s3__r20.npz', labels=labels)
    items = S0.scorer_check('full', cache, 2000, 1)
    assert items['scorer/full/partitions_equal']['pass'] is False and items['scorer/full/partitions_equal']['equal']['3'] is False
    assert S0.scorer_check('sample', cache, 2000, 1)['scorer/sample/partitions_equal']['pass'] is True


def test_bits_equal_compares_bits_not_values():
    a = unit(1)
    assert S0._bits_equal(a, a.copy()) and not S0._bits_equal(a, a.astype(np.float32))
    b = a.copy()
    b[0, 0] = np.nextafter(b[0, 0], np.float16(1))
    assert not S0._bits_equal(a, b)
    n = np.array([[np.nan]], dtype=np.float16)
    assert S0._bits_equal(n, n.copy())


@pytest.fixture
def stage_c(tmp_path, monkeypatch):
    """A tiny 12-27 stage C: held outputs, their weights, `reported.json` and `verdict.json` (genre claim)."""
    held, weights, record, claim = tmp_path / 'held', tmp_path / 'weights', {}, {}
    held.mkdir()
    weights.mkdir()
    for aspect in S0.ASPECTS:
        record[aspect] = {}
        for seed in S0.SEEDS:
            path = held / f'{aspect}__s{seed}.npz'
            path.write_bytes(f'{aspect}{seed}'.encode())
            record[aspect][str(seed)] = dict(path=f'/x/{path.name}', output_sha256=rb_score.sha_file(path))
            if aspect == 'genre':
                claim[str(seed)] = dict(path=f'/x/{path.name}', output_sha256=rb_score.sha_file(path))
            pt = weights / f'{aspect}__s{seed}.pt'
            pt.write_bytes(f'w{aspect}{seed}'.encode())
            pt.with_suffix('.json').write_text(json.dumps(dict(weights_sha256=rb_score.sha_file(pt))))
            held.joinpath(f'{aspect}__s{seed}.json').write_text(json.dumps(dict(weights_sha256=rb_score.sha_file(pt))))
    (tmp_path / 'reported.json').write_text(json.dumps(dict(G=dict(sources=dict(G_outputs=record)))))
    (tmp_path / 'verdict.json').write_text(json.dumps(dict(sources=dict(G_outputs=claim))))
    monkeypatch.setattr(S0, 'G_FULL_HELD', held)
    monkeypatch.setattr(S0, 'G_FULL_WEIGHTS', weights)
    monkeypatch.setattr(S0, 'G_FULL_RECORD', tmp_path / 'reported.json')
    monkeypatch.setattr(S0, 'G_FULL_VERDICT', tmp_path / 'verdict.json')
    return tmp_path


def test_g_full_outputs_check_against_the_stage_c_record(stage_c):
    held = stage_c / 'held'
    assert S0.g_full_outputs_check()['pass'] is True
    (held / 'style__s2.npz').write_bytes(b'changed')
    result = S0.g_full_outputs_check()
    assert result['pass'] is False and not result['files']['style__s2']['equal']


def test_g_full_outputs_check_also_against_the_stage_c_verdicts_claim(stage_c):
    claim = json.loads((stage_c / 'verdict.json').read_text())
    claim['sources']['G_outputs']['3']['output_sha256'] = '0' * 64          # the record of reported.json is right
    (stage_c / 'verdict.json').write_text(json.dumps(claim))
    result = S0.g_full_outputs_check()
    assert result['pass'] is False and not result['verdict_files']['genre__s3']['equal']
    assert all(v['equal'] for v in result['files'].values())


def test_g_full_weights_check_against_12_27s_own_records(stage_c):
    assert S0.g_full_weights_check()['pass'] is True
    (stage_c / 'weights' / 'genre__s1.pt').write_bytes(b'changed')           # the records beside it no longer match
    result = S0.g_full_weights_check()
    assert result['pass'] is False and not result['files']['genre__s1']['equal']
    (stage_c / 'weights' / 'genre__s1.pt').write_bytes(b'wgenre1')
    sidecar = stage_c / 'held' / 'style__s0.json'
    sidecar.write_text(json.dumps(dict(weights_sha256='0' * 64)))              # the held output's sidecar differs
    assert S0.g_full_weights_check()['pass'] is False


# ------------------------------------------------------------------ sample weights and held inference
ASPECT_BYTES = {a: a.encode() for a in S0.ASPECTS}


def make_head(folder, aspect, content):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{aspect}.pt').write_bytes(content)
    (folder / f'{aspect}.json').write_text(json.dumps(dict(config=dict(device='cpu'))))


T0 = 1_700_000_000_000_000_000              # ns: the V1a gate's mtime; seed 4's files are written after it


def set_mtimes(outputs, gate=T0, seed4=T0 + 10**11):
    os.utime(S0.gate_path(outputs, 'v1a_train'), ns=(gate, gate))
    folder = paths.run_dir(outputs, 'G', 4)
    for path in [folder] + list(folder.rglob('*')):
        os.utime(path, ns=(seed4, seed4))


@pytest.fixture
def world(small, tmp_path, monkeypatch):
    """12-27's GS-0.02 seeds 0 to 3 under a fake RG_TRAIN; this round's outputs with a V1a gate and a trained seed 4."""
    rg_train_dir = tmp_path / 'rg_train'
    monkeypatch.setattr(paths, 'RG_TRAIN', rg_train_dir)
    for seed in S0.SAMPLE_FROZEN_SEEDS:
        folder = rg_train_dir / paths.G_ARM / f's{seed}'
        for aspect in S0.ASPECTS:
            make_head(folder, aspect, f'frozen {aspect} {seed}'.encode())
        (folder / 'run.json').write_text(json.dumps(dict(status='ok', variant=paths.G_ARM, seed=seed)))
    outputs = tmp_path / 'out'
    gate = dict(**{'pass': True}, code_sha256=rs_train.code_sha256())
    S0._atomic_json(S0.gate_path(outputs, 'v1a_train'), gate)
    folder = paths.run_dir(outputs, 'G', 4)
    for aspect in S0.ASPECTS:
        make_head(folder, aspect, f'new {aspect}'.encode())
    run = dict(name='G', seed=4, config=rs_train.jsonable(sheet.config()), smoke=False, status='ok',
               p_sha256=paths.P_SHA256, code_dirty=False, code_sha256=rs_train.code_sha256())
    (folder / 'run.json').write_text(json.dumps(run))
    recorded = S0.g_sample_weights_record()['sha256']
    S0._atomic_json(S0.gate_path(outputs, 'v0a'), {'pass': True, 'code_sha256': S0.code_sha256(),
                                                   'parts': {'cpu': {'items': {'g_sample/weights': {'sha256': recorded}}}}})
    set_mtimes(outputs)
    return outputs, recorded


def test_g_sample_weights_record_lists_12_files(world):
    outputs, recorded = world
    assert sorted(recorded) == sorted(f's{s}/{a}' for s in range(4) for a in S0.ASPECTS)
    assert all(len(v) == 64 for v in recorded.values())


def test_sample_weights_seeds_0_to_3_are_12_27s_files_and_match_the_record(world):
    outputs, recorded = world
    path, sha = S0.sample_weights(outputs, 2, 'genre', recorded)
    assert path == paths.RG_TRAIN / paths.G_ARM / 's2/genre.pt' and sha == recorded['s2/genre']
    (path).write_bytes(b'tampered')
    with pytest.raises(AssertionError, match='V0a record'):
        S0.sample_weights(outputs, 2, 'genre', recorded)


def test_sample_weights_seed_4_comes_from_this_rounds_run(world):
    outputs, _ = world
    path, _ = S0.sample_weights(outputs, 4, 'emotion')
    assert path == paths.run_dir(outputs, 'G', 4) / 'emotion.pt'


def run_json(outputs):
    return paths.run_dir(outputs, 'G', 4) / 'run.json'


def test_seed_4_is_refused_without_a_passing_v1a_gate(world):
    outputs, _ = world
    gate = S0.gate_path(outputs, 'v1a_train')
    good = gate.read_text()
    gate.write_text(json.dumps(dict(**{'pass': False}, code_sha256=rs_train.code_sha256())))
    with pytest.raises(S0.Refusal, match='failed'):
        S0.sample_weights(outputs, 4, 'emotion')
    stale = dict(rs_train.code_sha256())
    stale['rs_knobs.py'] = '0' * 64
    gate.write_text(json.dumps(dict(**{'pass': True}, code_sha256=stale)))
    with pytest.raises(S0.Refusal, match='other trainer code'):
        S0.sample_weights(outputs, 4, 'emotion')
    gate.unlink()
    with pytest.raises(S0.Refusal, match='missing'):
        S0.sample_weights(outputs, 4, 'emotion')
    gate.write_text(good)
    assert S0.sample_weights(outputs, 4, 'emotion')[1]
    # seeds 0 to 3 do not need the V1a gate (12-27's files)
    gate.unlink()
    S0.sample_weights(outputs, 0, 'emotion')


def test_seed_4_is_refused_while_missing_failed_or_not_g(world):
    outputs, _ = world
    path = run_json(outputs)
    run = json.loads(path.read_text())
    path.write_text(json.dumps(dict(run, status='failed', cause='nonfinite')))
    with pytest.raises(S0.Refusal, match='failed'):
        S0.sample_weights(outputs, 4, 'emotion')
    path.write_text(json.dumps(dict(run, config=dict(run['config'], k6=5e-4))))
    with pytest.raises(AssertionError, match='not G'):
        S0.sample_weights(outputs, 4, 'emotion')
    stale = dict(run['code_sha256'], **{'rs_knobs.py': '0' * 64})
    path.write_text(json.dumps(dict(run, code_sha256=stale)))
    with pytest.raises(AssertionError, match='other code'):
        S0.sample_weights(outputs, 4, 'emotion')
    path.unlink()
    with pytest.raises(S0.Refusal, match='not trained'):
        S0.sample_weights(outputs, 4, 'emotion')


@pytest.fixture
def fake_infer(monkeypatch):
    calls = []

    def infer(weights, aspect, x, device):
        calls.append(Path(weights))
        out = unit(int.from_bytes(Path(weights).read_bytes()[:4], 'little') + 1)
        return out
    monkeypatch.setattr(S0, 'infer', infer)
    return calls


X = np.zeros((ROWS, S0.WIDTH), np.float32)


def weights_for(outputs, recorded):
    return lambda seed, aspect: S0.sample_weights(outputs, seed, aspect, recorded)


def test_infer_set_writes_every_output_with_its_record(world, fake_infer):
    outputs, recorded = world
    held = S0.sample_g(outputs).held_dir
    S0.infer_set(held, 'G', 'sample', [0, 4], weights_for(outputs, recorded), X, 'cpu')
    assert len(fake_infer) == 6
    cand = S0.sample_g(outputs)
    for aspect in S0.ASPECTS:
        for seed in (0, 4):
            S0.load_candidate(cand, 'sample', aspect, seed)
    info = json.loads((held / 'emotion__s4.json').read_text())
    assert info['weights_sha256'] == S0.sample_weights(outputs, 4, 'emotion')[1]
    assert info['arm'] == 'G' and info['scale'] == 'sample' and info['vc3_attempts'] == 1
    assert info['switches'] == {'m': 1, 'lam': 8, 'tau_t': 0.02, 'anchors': 0}


def test_infer_set_reuses_a_matching_output_and_refuses_changed_weights(world, fake_infer):
    outputs, recorded = world
    held = S0.sample_g(outputs).held_dir
    S0.infer_set(held, 'G', 'sample', [1], weights_for(outputs, recorded), X, 'cpu')
    fake_infer.clear()
    S0.infer_set(held, 'G', 'sample', [1], weights_for(outputs, recorded), X, 'cpu')
    assert fake_infer == []                                  # nothing recomputed
    # seed 4 retrained after its held output exists (V0d): the output no longer matches its weights
    S0.infer_set(held, 'G', 'sample', [4], weights_for(outputs, recorded), X, 'cpu')
    (paths.run_dir(outputs, 'G', 4) / 'style.pt').write_bytes(b'retrained')
    with pytest.raises(AssertionError, match='differs from this run'):
        S0.infer_set(held, 'G', 'sample', [4], weights_for(outputs, recorded), X, 'cpu')


def test_infer_set_writes_nothing_while_a_head_is_missing(world, fake_infer):
    outputs, recorded = world
    (paths.run_dir(outputs, 'G', 4) / 'genre.pt').unlink()
    with pytest.raises(AssertionError):
        S0.infer_set(S0.sample_g(outputs).held_dir, 'G', 'sample', [0, 4], weights_for(outputs, recorded), X, 'cpu')
    assert not S0.sample_g(outputs).held_dir.exists() and fake_infer == []


def test_infer_set_reruns_once_on_a_v0c_failure_and_lists_it(world, monkeypatch):
    outputs, recorded = world
    attempts = []

    def infer(weights, aspect, x, device):
        attempts.append(aspect)
        return unit(1) * np.float16(3) if len(attempts) == 1 else unit(1)      # the first attempt is not unit norm
    monkeypatch.setattr(S0, 'infer', infer)
    held = S0.sample_g(outputs).held_dir
    S0.infer_set(held, 'G', 'sample', [0], weights_for(outputs, recorded), X, 'cpu')
    info = json.loads((held / 'emotion__s0.json').read_text())
    assert info['vc3_attempts'] == 2 and 'norm error' in info['vc3_problems'][0]
    assert len(attempts) == 4                                   # one rerun for emotion, one attempt each for the rest


def test_infer_set_stops_after_a_second_failure_and_refuses_a_third_attempt(world, monkeypatch):
    outputs, recorded = world
    monkeypatch.setattr(S0, 'infer', lambda weights, aspect, x, device: unit(1) * np.float16(3))
    held = S0.sample_g(outputs).held_dir
    with pytest.raises(AssertionError, match='failed twice'):
        S0.infer_set(held, 'G', 'sample', [0], weights_for(outputs, recorded), X, 'cpu')
    record = json.loads(S0.failed_path(held, 'emotion', 0).read_text())
    assert len(record['problems']) == 2 and not (held / 'emotion__s0.npz').exists()
    with pytest.raises(AssertionError, match='V0d'):
        S0.infer_set(held, 'G', 'sample', [0], weights_for(outputs, recorded), X, 'cpu')


def test_infer_set_device_must_be_the_training_device(world, fake_infer):
    outputs, recorded = world
    with pytest.raises(AssertionError, match='device'):
        S0.infer_set(S0.sample_g(outputs).held_dir, 'G', 'sample', [0], weights_for(outputs, recorded), X, 'cuda')


def test_held_is_refused_without_the_v0a_gate_before_any_data_is_read(world, monkeypatch):
    outputs, _ = world
    S0.gate_path(outputs, 'v0a').unlink()
    monkeypatch.setattr(S0, 'held_images', lambda: pytest.fail('images were read'))
    with pytest.raises(S0.Refusal, match='missing'):
        S0.held(outputs, [0], 'cpu')


def test_held_seed_4_needs_the_v1a_gate_and_nothing_is_written(world, monkeypatch, fake_infer):
    outputs, _ = world
    monkeypatch.setattr(S0, 'held_images', lambda: X)
    monkeypatch.setattr(S0.rg_train, 'check_device', lambda d: d)
    S0.gate_path(outputs, 'v1a_train').unlink()
    with pytest.raises(S0.Refusal, match='v1a_train'):
        S0.held(outputs, [0, 1, 2, 3, 4], 'cpu')
    assert not S0.sample_g(outputs).held_dir.exists()
    S0.held(outputs, [0, 1, 2, 3], 'cpu')                       # seeds 0 to 3 do not need it
    assert len(list(S0.sample_g(outputs).held_dir.glob('*.npz'))) == 12


# ------------------------------------------------------------------ V0c and V0d gates
def full_world(world, fake_infer, monkeypatch, tmp_path):
    outputs, recorded = world
    monkeypatch.setattr(S0, 'held_images', lambda: X)
    monkeypatch.setattr(S0.rg_train, 'check_device', lambda d: d)
    S0.held(outputs, [0, 1, 2, 3, 4], 'cpu')
    full = tmp_path / 'g_full'
    write_set(full, paths.G_ARM, 'full')
    monkeypatch.setattr(S0, 'FULL_G', S0.Candidate('G', full, paths.G_ARM))
    return outputs, recorded


def test_v0c_gate_passes_on_both_sets_and_fails_on_a_bad_file(world, fake_infer, monkeypatch, tmp_path):
    outputs, _ = full_world(world, fake_infer, monkeypatch, tmp_path)
    gate = S0.check_v0c(outputs)
    assert gate['pass'] is True and len(gate['items']['sample']) == len(gate['items']['full']) == 15
    assert S0.read_gate(outputs, 'v0c')['held_sha256'] == paths.HELD_SHA256
    write_output(S0.FULL_G.held_dir, paths.G_ARM, 'full', 'genre', 2, x=unit(2) * np.float16(2))
    gate = S0.check_v0c(outputs)
    assert gate['pass'] is False and not gate['items']['full']['genre__s2']['pass']
    with pytest.raises(S0.Refusal):
        S0.read_gate(outputs, 'v0c')


def test_v0d_gate_passes_when_nothing_was_retrained_and_lists_reruns(world, fake_infer, monkeypatch, tmp_path):
    outputs, _ = full_world(world, fake_infer, monkeypatch, tmp_path)
    gate = S0.check_v0d(outputs)
    assert gate['pass'] is True and gate['items']['seed4_trained_once']['trainings'] == 1
    assert all(i['v0c_attempts'] == 1 for k, i in gate['items'].items() if '__s' in k)


def test_v0d_gate_fails_on_retrained_seed_4_or_replaced_frozen_weights(world, fake_infer, monkeypatch, tmp_path):
    outputs, _ = full_world(world, fake_infer, monkeypatch, tmp_path)
    (paths.run_dir(outputs, 'G', 4) / 'genre.pt').write_bytes(b'retrained after the held output')
    gate = S0.check_v0d(outputs)
    assert gate['pass'] is False and 'weights changed' in gate['items']['genre__s4']['problem']
    (paths.run_dir(outputs, 'G', 4) / 'genre.pt').write_bytes(b'new genre')            # restored
    assert S0.check_v0d(outputs)['pass'] is True
    (paths.RG_TRAIN / paths.G_ARM / 's1/style.pt').write_bytes(b'replaced')
    gate = S0.check_v0d(outputs)
    assert gate['pass'] is False and not gate['items']['style__s1']['pass']


def test_v0d_gate_fails_on_a_failure_record_and_on_a_second_training_of_seed_4(world, fake_infer, monkeypatch, tmp_path):
    outputs, _ = full_world(world, fake_infer, monkeypatch, tmp_path)
    held = S0.sample_g(outputs).held_dir
    S0.failed_path(held, 'style', 2).write_text('{}')
    assert not S0.check_v0d(outputs)['items']['style__s2']['pass']
    S0.failed_path(held, 'style', 2).unlink()
    for n in (1, 2):                                            # two moved-aside V1b failures: more than one rerun
        aside = outputs / 'replaced' / 'G' / f's4-{n}'
        aside.mkdir(parents=True)
        (aside / 'run.json').write_text(json.dumps(dict(variant='G', seed=4, status='failed', failure_kind='va3')))
    item = S0.check_v0d(outputs)['items']['seed4_trained_once']
    assert item['pass'] is False and item['trainings'] == 3


# ------------------------------------------------------------------ the verdict
def fake_block(d, lower, scale):
    return dict(candidate='G', scale=scale, d=d, A_cand=0.1, A_e={'B1-r20': 0.09}, interval=[lower, d + 0.002],
                outcome_id=S0.outcome_id(d, lower, scale))


def v0c_record(scale_pass=(True, True), shas=None, held_sha=None):
    """A V0c gate's content: per file pass and SHA-256 (emotion shas by seed from `shas`), a pass per scale."""
    items = {scale: {f'{a}__s{s}': {'pass': ok, 'output_sha256': f'{scale}{a}{s}'.ljust(64, '0')} if ok else
                     {'pass': False, 'problem': f'{a}__s{s}: V0c: bad'}
                     for a in S0.ASPECTS for s in S0.SEEDS} for scale, ok in zip(S0.SCALES, scale_pass)}
    if not all(scale_pass):                  # one failing file in a failing scale
        scale = S0.SCALES[scale_pass.index(False)]
        items[scale]['genre__s2'] = {'pass': False, 'problem': 'genre__s2: V0c: bad'}
    return dict(**{'pass': all(scale_pass)}, code_sha256=S0.code_sha256(), items=items,
                held_sha256=held_sha or S0.HELD_SHA256, scale_pass=dict(zip(S0.SCALES, scale_pass)))


@pytest.fixture
def gated(tmp_path, monkeypatch):
    for name in ('v0a', 'v0d'):
        S0._atomic_json(S0.gate_path(tmp_path, name), dict(**{'pass': True}, code_sha256=S0.code_sha256()))
    S0._atomic_json(S0.gate_path(tmp_path, 'v0c'), v0c_record())
    monkeypatch.setattr(S0.paths, 'check_outputs', lambda o, smoke=False: Path(o))
    monkeypatch.setattr(S0.paths, 'check_imports', lambda: None)
    return tmp_path


def score_fake(blocks, seen=None):
    def score(cand, scale, cache, n, w, recorded=None):
        if seen is not None:
            seen.append((scale, cand, cache, recorded))
        return dict(blocks[scale])
    return score


def test_verdict_needs_v0a_v0c_and_v0d(gated, monkeypatch):
    monkeypatch.setattr(S0, 'score_scale', lambda *a, **k: pytest.fail('scored before the gates held'))
    for name in ('v0a', 'v0c', 'v0d'):
        path = S0.gate_path(gated, name)
        good = path.read_text()
        for bad in (dict(json.loads(good), code_sha256='0' * 64),                    # stale code
                    *([dict(json.loads(good), **{'pass': False})] if name != 'v0c' else [])):
            path.write_text(json.dumps(bad))
            with pytest.raises(S0.Refusal):
                S0.verdict(gated)
        path.unlink()
        with pytest.raises(S0.Refusal, match='missing'):
            S0.verdict(gated)
        path.write_text(good)
    path = S0.gate_path(gated, 'v0c')                  # H is not the CT's list: a refusal, not a scale without outcome
    path.write_text(json.dumps(v0c_record(held_sha='0' * 64)))
    with pytest.raises(S0.Refusal, match='SHA-256'):
        S0.verdict(gated)


def test_verdict_records_both_scales_e0_and_is_written_once(gated, monkeypatch):
    blocks = {'sample': fake_block(0.015, 0.004, 'sample'), 'full': fake_block(0.0164, 0.011, 'full')}
    monkeypatch.setattr(S0, 'score_scale', score_fake(blocks))
    record = S0.verdict(gated)
    assert record['outcomes'] == {'sample': 'G-ES-GO', 'full': 'G-EF-GO'} and record['E0'] is True
    assert record['blocks']['full']['d'] == 0.0164 and 'A_e' in record['blocks']['sample']
    assert set(record['gates']) == {'v0a', 'v0c', 'v0d'}
    assert S0.verdict(gated) == record                          # a second call gives the same file
    blocks['sample'] = fake_block(0.02, 0.004, 'sample')
    with pytest.raises(AssertionError, match='refusing to replace'):
        S0.verdict(gated)


def test_verdict_e0_false_when_full_is_not_go_or_sample_is_no(gated, monkeypatch):
    blocks = {'sample': fake_block(-0.01, -0.02, 'sample'), 'full': fake_block(0.0164, 0.011, 'full')}
    monkeypatch.setattr(S0, 'score_scale', score_fake(blocks))
    assert S0.verdict(gated)['E0'] is False


def test_verdict_scores_g_at_both_scales_from_the_right_folders(gated, monkeypatch):
    seen = []
    blocks = {scale: fake_block(0.0, -0.1, scale) for scale in S0.SCALES}
    monkeypatch.setattr(S0, 'score_scale', score_fake(blocks, seen))
    S0.verdict(gated)
    assert [s for s, *_ in seen] == ['sample', 'full']
    assert seen[0][1] == S0.sample_g(gated) and seen[0][1].held_dir == gated / 'stage0/sample/held/G'
    assert seen[1][1] == S0.FULL_G and S0.FULL_G.held_dir == paths.RG_FULL / 'held' / 'GS-0.02'
    assert seen[0][2] == gated / 'stage0/partitions'


def test_verdict_reads_only_outputs_with_the_sha_the_v0c_gate_recorded(gated, monkeypatch):
    seen = []
    blocks = {scale: fake_block(0.0, -0.1, scale) for scale in S0.SCALES}
    monkeypatch.setattr(S0, 'score_scale', score_fake(blocks, seen))
    S0.verdict(gated)
    gate = json.loads(S0.gate_path(gated, 'v0c').read_text())
    for scale, _, _, recorded in seen:
        assert recorded == {str(s): gate['items'][scale][f'emotion__s{s}']['output_sha256'] for s in S0.SEEDS}


def test_one_scale_failing_v0c_has_no_outcome_and_the_other_is_scored(gated, monkeypatch):
    """Rule §4.1 V0d: a seed that still fails leaves that scale with no outcome (not both scales)."""
    seen = []
    blocks = {scale: fake_block(0.0164, 0.011, scale) for scale in S0.SCALES}
    monkeypatch.setattr(S0, 'score_scale', score_fake(blocks, seen))
    S0._atomic_json(S0.gate_path(gated, 'v0c'), v0c_record((False, True)))
    record = S0.verdict(gated)
    assert [s for s, *_ in seen] == ['full']
    assert record['outcomes'] == {'sample': None, 'full': 'G-EF-GO'} and record['E0'] is False
    assert record['blocks']['sample']['outcome_id'] is None
    assert 'genre__s2' in record['blocks']['sample']['v_failure'] and 'V0c' in record['blocks']['sample']['v_failure']
    assert record['blocks']['full']['d'] == 0.0164


def test_the_full_scale_failing_v0c_leaves_only_the_sample_scale_scored(tmp_path, gated, monkeypatch):
    blocks = {scale: fake_block(0.015, 0.004, scale) for scale in S0.SCALES}
    monkeypatch.setattr(S0, 'score_scale', score_fake(blocks))
    S0._atomic_json(S0.gate_path(gated, 'v0c'), v0c_record((True, False)))
    record = S0.verdict(gated)
    assert record['outcomes'] == {'sample': 'G-ES-GO', 'full': None} and record['E0'] is False


def test_both_scales_failing_v0c_have_no_outcome(gated, monkeypatch):
    monkeypatch.setattr(S0, 'score_scale', lambda *a, **k: pytest.fail('scored a failed set'))
    S0._atomic_json(S0.gate_path(gated, 'v0c'), v0c_record((False, False)))
    assert S0.verdict(gated)['outcomes'] == {'sample': None, 'full': None}


# ------------------------------------------------------------------ the score on a tiny world (M3f)
@pytest.fixture
def tiny_score(small, synthetic, tmp_path, monkeypatch):
    """`score_scale` on 150 rows: G's sample files (emotion differs from style and genre), 12-24's comparators from the
    `synthetic` partitions; `held_setup`, `read_rcol_eligible`, `comparators` and the partitioning are patched."""
    labels, _, comp, rcol = synthetic
    n = len(labels)
    monkeypatch.setattr(S0, 'N_HELD', n)
    for seed in S0.SEEDS:
        for aspect in S0.ASPECTS:
            x = unit(seed + (0 if aspect == 'emotion' else 100), n)
            write_output(tmp_path / 'g', 'G', 'sample', aspect, seed, x=x, rows=n)
    calls = []

    def fixed_partitions(x, arm, folder, aspect, seed, held_sha256, source_sha256):
        calls.append(dict(arm=arm, folder=Path(folder), aspect=aspect, seed=seed, x=x.copy()))
        lab = x[:, :6].argmax(1)
        return {'r20': np.stack([lab, (lab + 1) % 6, lab])}
    rcol_path = tmp_path / 'rcol.json'
    rcol_path.write_text('{}')
    monkeypatch.setattr(rb_score, 'fixed_partitions', fixed_partitions)
    monkeypatch.setattr(rb_score, 'held_setup', lambda: (NAMES, S0.HELD_SHA256, labels))
    monkeypatch.setattr(rb_score, 'read_rcol_eligible', lambda scale: (rcol, rcol_path))
    monkeypatch.setattr(S0, 'comparators', lambda scale, r: (comp, None, dict(outputs={'B1': {}}, nonfinite_held={})))
    return S0.Candidate('G', tmp_path / 'g', 'G'), calls, (labels, comp, rcol, n)


def test_score_scale_reads_the_emotion_files_and_gives_assembles_block(tiny_score, tmp_path):
    cand, calls, (labels, comp, rcol, n) = tiny_score
    block = S0.score_scale(cand, 'sample', tmp_path / 'cache', 20, 1)
    assert [(c['aspect'], c['seed'], c['arm']) for c in calls] == [('emotion', s, 'K4') for s in S0.SEEDS]
    assert all(c['folder'] == tmp_path / 'cache' / 'sample' / 'G' for c in calls)
    for call in calls:
        assert np.array_equal(call['x'], unit(call['seed'], n).astype(np.float32))      # the emotion file, not style's
    fixed = {c['seed']: {'r20': np.stack([c['x'][:, :6].argmax(1), (c['x'][:, :6].argmax(1) + 1) % 6,
                                          c['x'][:, :6].argmax(1)])} for c in calls}
    ref = S0.assemble('G', fixed, comp, rcol, {}, labels, 'sample', 20, 1)
    for key in ('d', 'A_cand', 'A_e', 'interval', 'bootstrap_d', 'outcome_id', 'arms', 'eligibility'):
        assert block[key] == ref[key], key
    assert block['sources']['held_outputs']['G']['0']['path'].endswith('/emotion__s0.npz')


def test_the_outcome_family_name_reaches_the_block_of_assemble_and_score_scale(tiny_score, tmp_path, synthetic):
    cand, _, (labels, comp, rcol, n) = tiny_score
    prefix = {'sample': 'C-ES', 'full': 'C-EF'}
    block = S0.score_scale(cand, 'sample', tmp_path / 'cache', 10, 1, prefix=prefix)
    assert block['outcome_id'].startswith('C-ES-') and S0.score_scale(
        cand, 'sample', tmp_path / 'cache', 10, 1)['outcome_id'].startswith('G-ES-')
    cand_fixed = synthetic[1]
    assert S0.assemble('G', cand_fixed, comp, rcol, {}, labels, 'full', 10, 1, prefix=prefix)['outcome_id'].startswith('C-EF-')


def test_score_scale_refuses_an_output_that_changed_since_the_v0c_gate(tiny_score, tmp_path):
    cand, calls, _ = tiny_score
    _, sha = S0.load_candidate(cand, 'sample', 'emotion', 0)
    recorded = {str(s): S0.load_candidate(cand, 'sample', 'emotion', s)[1] for s in S0.SEEDS}
    assert S0.score_scale(cand, 'sample', tmp_path / 'cache', 10, 1, recorded=recorded)['outcome_id'].startswith('G-ES-')
    with pytest.raises(AssertionError, match='V0c gate'):
        S0.score_scale(cand, 'sample', tmp_path / 'cache2', 10, 1, recorded=dict(recorded, **{'0': '0' * 64}))




# ------------------------------------------------------------------ review fixes: weights provenance (M4e, M4f, M4j, M4k)
def test_seed_4_run_must_be_a_real_run_on_p(world):
    outputs, _ = world
    path = run_json(outputs)
    run = json.loads(path.read_text())
    path.write_text(json.dumps(dict(run, smoke=True)))                           # a smoke run
    with pytest.raises(AssertionError, match='not G'):
        S0.sample_weights(outputs, 4, 'emotion')
    path.write_text(json.dumps(dict(run, p_sha256='0' * 64)))                    # trained on another P
    with pytest.raises(AssertionError):
        S0.sample_weights(outputs, 4, 'emotion')
    path.write_text(json.dumps(run))
    assert S0.sample_weights(outputs, 4, 'emotion')[1]


@pytest.mark.parametrize('change', [dict(status='failed'), dict(variant='GS-0.01'), dict(seed=1)])
def test_frozen_seeds_run_json_must_be_an_ok_run_of_g_at_that_seed(world, change):
    outputs, recorded = world
    path = paths.RG_TRAIN / paths.G_ARM / 's2/run.json'
    run = json.loads(path.read_text())
    path.write_text(json.dumps(dict(run, **change)))
    with pytest.raises(AssertionError):
        S0.sample_weights(outputs, 2, 'genre', recorded)
    path.write_text(json.dumps(run))
    assert S0.sample_weights(outputs, 2, 'genre', recorded)[1] == recorded['s2/genre']


def test_held_refuses_a_tampered_frozen_head_with_the_v0a_record_and_writes_nothing(world, monkeypatch, fake_infer):
    outputs, _ = world
    monkeypatch.setattr(S0, 'held_images', lambda: X)
    monkeypatch.setattr(S0.rg_train, 'check_device', lambda d: d)
    (paths.RG_TRAIN / paths.G_ARM / 's2/genre.pt').write_bytes(b'tampered')
    with pytest.raises(AssertionError, match='V0a record'):
        S0.held(outputs, [2], 'cpu')
    assert not S0.sample_g(outputs).held_dir.exists() and fake_infer == []


def test_g_sample_weights_record_is_compared_with_the_v1a_gates_reference(world):
    outputs, recorded = world
    gate = json.loads(S0.gate_path(outputs, 'v1a_train').read_text())
    S0._atomic_json(S0.gate_path(outputs, 'v1a_train'), dict(gate, input_hashes={}))
    assert S0.g_sample_weights_record(outputs)['v1a_compared'] == {}               # no record of these heads
    reference = {f'{k}/state': {'replay': v, 'reference': v} for k, v in recorded.items() if k.startswith(('s0/', 's1/'))}
    S0._atomic_json(S0.gate_path(outputs, 'v1a_train'), dict(gate, input_hashes=reference))
    result = S0.g_sample_weights_record(outputs)
    assert result['pass'] is True and sorted(result['v1a_compared']) == sorted(reference_key[:-6] for reference_key in reference)
    (paths.RG_TRAIN / paths.G_ARM / 's1/style.pt').write_bytes(b'replaced since')
    result = S0.g_sample_weights_record(outputs)
    assert result['pass'] is False and result['v1a_compared']['s1/style']['equal'] is False


# ------------------------------------------------------------------ review fixes: held inference of an existing output (M5e)
def test_infer_set_applies_v0c_to_an_existing_output(world, fake_infer):
    outputs, recorded = world
    held = S0.sample_g(outputs).held_dir
    S0.infer_set(held, 'G', 'sample', [0], weights_for(outputs, recorded), X, 'cpu')
    path = held / 'emotion__s0.npz'
    arrays, info = rb_student.read_outputs(path)
    bad = arrays['img'] * np.float16(2)                       # a consistent file whose rows are not unit norm
    meta = {k: v for k, v in info.items() if k not in ('arrays', 'code_sha256')}
    rb_student.write_outputs(path, dict(img=bad), dict(meta, sha256=rb_student.sha_array(bad)))
    with pytest.raises(AssertionError, match='V0c'):
        S0.infer_set(held, 'G', 'sample', [0], weights_for(outputs, recorded), X, 'cpu')


# ------------------------------------------------------------------ review fixes: V0c per scale and H's list
def test_v0c_has_a_pass_per_scale(world, fake_infer, monkeypatch, tmp_path):
    outputs, _ = full_world(world, fake_infer, monkeypatch, tmp_path)
    gate = S0.check_v0c(outputs)
    assert gate['scale_pass'] == {'sample': True, 'full': True} and gate['pass'] is True
    write_output(S0.FULL_G.held_dir, paths.G_ARM, 'full', 'genre', 2, x=unit(2) * np.float16(2))
    gate = S0.check_v0c(outputs)
    assert gate['scale_pass'] == {'sample': True, 'full': False} and gate['pass'] is False
    assert S0.read_gate(outputs, 'v0c', must_pass=False)['scale_pass']['sample'] is True
    (S0.sample_g(outputs).held_dir / 'style__s1.npz').unlink()
    assert S0.check_v0c(outputs)['scale_pass'] == {'sample': False, 'full': False}


def test_v0c_fails_both_scales_when_h_is_not_the_cts_list(world, fake_infer, monkeypatch, tmp_path):
    outputs, _ = full_world(world, fake_infer, monkeypatch, tmp_path)
    monkeypatch.setattr(rb_student, 'held_names', lambda: (NAMES, '0' * 64))
    gate = S0.check_v0c(outputs)
    assert gate['pass'] is False and gate['scale_pass'] == {'sample': False, 'full': False}


# ------------------------------------------------------------------ review fixes: V0d
@pytest.fixture
def flaky_infer(monkeypatch):
    """The first inference of the run fails V0c (not unit norm); every other gives a good output."""
    calls = []

    def infer(weights, aspect, x, device):
        calls.append(aspect)
        return unit(1) * np.float16(3) if len(calls) == 1 else unit(1)
    monkeypatch.setattr(S0, 'infer', infer)
    return calls


def test_v0d_passes_when_one_output_needed_one_v0c_rerun(world, flaky_infer, monkeypatch, tmp_path):
    outputs, _ = full_world(world, flaky_infer, monkeypatch, tmp_path)
    gate = S0.check_v0d(outputs)
    item = gate['items']['emotion__s0']
    assert item['v0c_attempts'] == 2 and item['pass'] is True and gate['pass'] is True


def aside(outputs, n, **record):
    folder = outputs / 'replaced' / 'G' / f's4-{n}'
    folder.mkdir(parents=True)
    (folder / 'x.pt').write_bytes(b'x')
    if record:
        (folder / 'run.json').write_text(json.dumps(dict(variant='G', seed=4, **record)))
    return folder


@pytest.mark.parametrize('record', [dict(status='ok'), dict(status='failed', failure_kind='va3'),
                                    dict(status='failed', failure_kind='nonfinite')])
def test_v0d_seed_4_trained_once_fails_on_any_other_training_with_a_run_json(world, fake_infer, monkeypatch, tmp_path,
                                                                             record):
    outputs, _ = full_world(world, fake_infer, monkeypatch, tmp_path)
    assert S0.check_v0d(outputs)['items']['seed4_trained_once']['pass'] is True
    aside(outputs, 1, **record)                   # no rerun_of in seed 4's final run.json: still a second training
    item = S0.check_v0d(outputs)['items']['seed4_trained_once']
    assert item['pass'] is False and item['trainings'] == 2 and len(item['aside_with_run_json']) == 1


def test_v0d_seed_4_rerun_of_without_an_aside_folder_fails(world, fake_infer, monkeypatch, tmp_path):
    outputs, _ = full_world(world, fake_infer, monkeypatch, tmp_path)
    path = run_json(outputs)
    path.write_text(json.dumps(dict(json.loads(path.read_text()), rerun_of=str(outputs / 'replaced/G/s4-9'))))
    item = S0.check_v0d(outputs)['items']['seed4_trained_once']
    assert item['pass'] is False and item['aside_with_run_json'] == []


def test_v0d_lists_an_interrupted_aside_folder_without_failing(world, fake_infer, monkeypatch, tmp_path):
    outputs, _ = full_world(world, fake_infer, monkeypatch, tmp_path)
    interrupted = aside(outputs, 1)               # no run.json
    item = S0.check_v0d(outputs)['items']['seed4_trained_once']
    assert item['pass'] is True and item['aside_interrupted'] == [str(interrupted)] and item['trainings'] == 1


def test_v0d_v1a_gate_must_be_older_than_seed_4s_first_file(world, fake_infer, monkeypatch, tmp_path):
    outputs, _ = full_world(world, fake_infer, monkeypatch, tmp_path)
    assert S0.check_v0d(outputs)['items']['v1a_before_seed4']['pass'] is True
    set_mtimes(outputs, gate=T0 + 10**11)         # the gate is as old as the first file: not earlier
    assert S0.check_v0d(outputs)['items']['v1a_before_seed4']['pass'] is False
    set_mtimes(outputs, gate=T0 + 2 * 10**11)     # the gate was written after seed 4 trained
    gate = S0.check_v0d(outputs)
    assert gate['items']['v1a_before_seed4']['pass'] is False and gate['pass'] is False
    set_mtimes(outputs)
    folder = aside(outputs, 1, status='failed', failure_kind='va3')                 # an earlier training moved aside
    os.utime(folder / 'x.pt', ns=(T0 - 10**9, T0 - 10**9))
    assert S0.check_v0d(outputs)['items']['v1a_before_seed4']['pass'] is False


def test_v0d_policy_followed_when_a_seed_failed_v0c_twice(world, fake_infer, monkeypatch, tmp_path):
    outputs, _ = full_world(world, fake_infer, monkeypatch, tmp_path)
    held = S0.sample_g(outputs).held_dir
    (held / 'genre__s2.npz').unlink()
    (held / 'genre__s2.json').unlink()
    S0._atomic_json(S0.failed_path(held, 'genre', 2), dict(arm='G', aspect='genre', seed=2, problems=['a', 'b']))
    gate = S0.check_v0d(outputs)
    assert gate['items']['genre__s2']['pass'] is True and gate['items']['genre__s2']['failed_twice'] and gate['pass']
    v0c = S0.check_v0c(outputs)
    assert v0c['scale_pass'] == {'sample': False, 'full': True}                # this scale alone has no outcome
    # the record must be what infer_set writes: two problems, no output beside it
    for bad in (dict(arm='G', aspect='genre', seed=2, problems=['a']), dict(arm='G', aspect='style', seed=2, problems=['a', 'b']),
                {}):
        S0.failed_path(held, 'genre', 2).write_text(json.dumps(bad))
        assert S0.check_v0d(outputs)['items']['genre__s2']['pass'] is False
    S0._atomic_json(S0.failed_path(held, 'genre', 2), dict(arm='G', aspect='genre', seed=2, problems=['a', 'b']))
    write_output(held, 'G', 'sample', 'genre', 2)                           # an output beside the failure record
    assert S0.check_v0d(outputs)['items']['genre__s2']['pass'] is False
