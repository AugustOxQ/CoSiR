"""Tests of eg_score (ticket 05) on small synthetic outputs; no real data is read."""
import json

import numpy as np
import pytest

import eg_paths
import eg_score as score
import rb_paths
import rb_score

SEEDS5 = [0, 1, 2, 3, 4]


# --- arm lists and paths (rule §2.1: entries B1 to B6 of that scale and the matched arm) ---------------------------

@pytest.mark.parametrize('candidate, matched', [('K6', 'M6'), ('K7', 'M7'), ('K8', 'M8'), ('K5', 'M5')])
def test_a_new_candidate_has_b1_to_b6_and_its_own_matched_arm(candidate, matched):
    assert score.arms_for(candidate) == (candidate, 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', matched)


def test_the_old_candidates_keep_12_24s_lists():
    for old in ('K1', 'K2', 'K3', 'K4'):
        assert score.arms_for(old) == rb_score.arms_for(old)
    assert score.arms_for('K4')[-1] == 'B6' and score.arms_for('K2')[-1] == 'M2'
    with pytest.raises(AssertionError):
        score.arms_for('K9')


def test_held_output_paths_students_and_matched_arms_under_the_read_root(tmp_path):
    for k, m in (('K5', 'M5'), ('K6', 'M6'), ('K7', 'M7'), ('K8', 'M8')):
        assert score.output_path('sample', k, 3, 'style', tmp_path) == tmp_path / f'part2/sample/held/{k}/style__s3.npz'
        assert score.output_path('full', m, 0, outputs=tmp_path) == tmp_path / f'part2/full/held/{m}/P3__s0.npz'
    assert score.output_path('full', 'K2', 1, 'genre', tmp_path) == tmp_path / 'part2/full/held/K2/genre__s1.npz'
    assert score.output_path('sample', 'M3', 1, outputs=tmp_path) == tmp_path / 'part2/sample/held/M3/P3__s1.npz'


def test_b1_to_b6_are_read_from_the_cts_saved_held_outputs(tmp_path):
    assert score.output_path('sample', 'B1', 2, outputs=tmp_path) == rb_paths.CT_OUT / 'held/percept/core__k40/P3__s2.npz'
    assert score.output_path('full', 'B6', 0, outputs=tmp_path) == rb_paths.CT_FULL / 'held/percept/core__k40/P2__s0.npz'
    assert score.output_path('full', 'B5', 4, outputs=tmp_path).name == 'P4__s4.npz'
    with pytest.raises(AssertionError):
        score.output_path('sample', 'M9', 0, outputs=tmp_path)
    with pytest.raises(AssertionError):
        score.output_path('sample', 'K5', 5, outputs=tmp_path)


# --- stage 3 and 4: the verdict file ------------------------------------------------------------------------------

def unit(n, w, seed):
    x = np.random.default_rng(seed).normal(size=(n, w)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def put_sidecar(path, arm, seed, aspect, scale, list_sha256='H'):
    """The index's output sidecar, written from the saved array (independent of the scorer)."""
    import hashlib
    with np.load(path) as z:
        key, = z.files
        x = z[key]
    path.with_suffix('.json').write_text(json.dumps(dict(
        list_sha256=list_sha256, rows=int(x.shape[0]), width=int(x.shape[1]), dtype=str(x.dtype),
        sha256=hashlib.sha256(x.tobytes()).hexdigest(), scale=scale, arm=arm, seed=seed, aspect=aspect)))


def put_student(root, scale, arm, seed, n, list_sha256='H'):
    for aspect in ('emotion', 'style', 'genre'):
        path = score.output_path(scale, arm, seed, aspect, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, img=unit(n, 2048, 100 * seed + len(aspect)))
        put_sidecar(path, arm, seed, aspect, scale, list_sha256)


def put_prob(path, arm, seed, scale, prob, list_sha256='H'):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, prob=prob.astype(np.float32))
    if arm in rb_score.CT_STEMS:
        path.with_suffix('.json').write_text(json.dumps(dict(held_sha256=list_sha256, rows=len(prob))))
    else:
        put_sidecar(path, arm, seed, 'percept', scale, list_sha256)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A 60-painting stand-in for the held set: CT-layout B files under tmp, K6 and M6 under `out`."""
    n = 60
    ct_root, out = tmp_path / 'ct', tmp_path / 'out'
    for scale in ('sample', 'full'):
        monkeypatch.setitem(rb_score.CT_ROOT, scale, ct_root / scale)
    y = np.arange(n) % 5
    for module in (score, rb_score):         # the checks also run 12-24's own scorer on the same stand-in
        monkeypatch.setattr(module, 'held_setup', lambda: (['x'] * n, 'H', y))
        monkeypatch.setattr(module, 'read_rcol_eligible', lambda scale: (
            {'B2': [], 'B3': [], 'B4': SEEDS5, 'B5': SEEDS5}, tmp_path / 'rcol.json'))
    (tmp_path / 'rcol.json').write_text('{}')
    rng = np.random.default_rng(0)
    for scale in ('sample', 'full'):
        for s in SEEDS5:
            put_student(out, scale, 'K6', s, n)
            put_prob(score.output_path(scale, 'M6', s, outputs=out), 'M6', s, scale, rng.random((n, 40)))
            for arm in ('B1', 'B4', 'B5', 'B6'):
                put_prob(score.output_path(scale, arm, s, outputs=out), arm, s, scale, rng.random((n, 40)))
    return out


def test_stage_3_writes_a_block_per_candidate_with_s1_outcome_and_caches_under_outputs(world):
    record = score.run_stage(3, ['K6'], world, workers=1, n_resamples=3)
    block = record['blocks']['K6']
    assert (record['stage'], record['scale'], record['held_sha256']) == (3, 'sample', 'H')
    assert block['arms'] == list(score.arms_for('K6')) and block['outcome_id'].startswith('S1-')
    assert block['eligibility']['eligible_seeds']['B2'] == [] and len(block['bootstrap_d']) == 3
    assert block['sources']['code']['code_commit'] and 'K6' in block['sources']['held_outputs']
    assert json.loads((world / 'part2/verdict_stage3.json').read_text())['blocks']['K6'] == block
    assert (world / 'part2/partitions/sample/K6/emotion__s0__r20.npz').exists()
    assert (world / 'part2/partitions/sample/M6/percept__s4__own.npz').exists()


def test_stage_4_is_the_same_statistic_at_full_scale_with_s2_outcomes(world):
    verdict_file(world, 3, {'K6': dict(outcome_id='S1-GO')})
    block = score.run_stage(4, ['K6'], world, workers=1, n_resamples=3)['blocks']['K6']
    assert block['scale'] == 'full' and block['outcome_id'].startswith('S2-')
    assert json.loads((world / 'part2/verdict_stage4.json').read_text())['blocks'].keys() == {'K6'}


def test_the_statistic_is_rb_scores_on_the_same_partitions(world):
    block = score.run_stage(3, ['K6'], world, workers=1, n_resamples=4)['blocks']['K6']
    arms = score.arms_for('K6')
    # an independent route: the CT-layout files are read straight, partitions made by rb_score's own cache reader
    _, sha, y = score.held_setup()
    fixed, eligible, _ = score.prepare_arms('sample', arms, score.read_rcol_eligible('sample')[0], sha,
                                            score.Roots(world, world), n_rows=len(y))
    ref = rb_score.paired_bootstrap(fixed, y, eligible, arms, 4, 1)
    assert block['d'] == ref['d'] and block['interval'] == ref['interval'] and block['bootstrap_d'] == ref['bootstrap_d']
    assert block['outcome_id'] == rb_score.outcome(ref['d'], ref['interval'][0], 'sample')


def test_a_missing_or_invalid_seed_removes_only_that_candidates_outcome(world):
    (world / 'part2/sample/held/K6/genre__s2.npz').unlink()
    record = score.run_stage(3, ['K6', 'K7'], world, workers=1, n_resamples=3)   # K7 has no files at all
    bad = record['blocks']['K6']
    assert bad['outcome_id'] is None and 'genre__s2' in bad['v_failure'] and 'bootstrap_d' not in bad
    assert bad['arms'] == list(score.arms_for('K6')) and bad['scale'] == 'sample'
    assert record['blocks']['K7']['outcome_id'] is None and 'K7' in record['blocks']['K7']['v_failure']


def test_a_sidecar_of_another_list_is_a_v3_failure(world):
    path = score.output_path('sample', 'M6', 1, outputs=world)
    put_prob(path, 'M6', 1, 'sample', np.random.default_rng(9).random((60, 40)), list_sha256='OTHER')
    block = score.run_stage(3, ['K6'], world, workers=1, n_resamples=3)['blocks']['K6']
    assert block['outcome_id'] is None and 'list_sha256' in block['v_failure']


def test_a_finished_block_is_kept_on_rerun_unless_a_reason_is_given(world):
    first = score.run_stage(3, ['K6'], world, workers=1, n_resamples=3)['blocks']['K6']
    assert score.run_stage(3, ['K6'], world, workers=1, n_resamples=3)['blocks']['K6'] == first
    (world / 'part2/sample/held/K6/genre__s2.npz').unlink()
    assert score.run_stage(3, ['K6'], world, workers=1, n_resamples=3)['blocks']['K6'] == first
    redone = score.run_stage(3, ['K6'], world, workers=1, n_resamples=3, replace='V3 redo')['blocks']['K6']
    assert redone['outcome_id'] is None


def test_reads_from_one_root_and_caches_in_another(world, tmp_path):
    cache = tmp_path / 'cache'
    block = score.run_stage(3, ['K6'], cache, workers=1, n_resamples=3,
                            roots=score.Roots(world, cache))['blocks']['K6']
    assert block['outcome_id'].startswith('S1-')
    assert (cache / 'part2/partitions/sample/K6/emotion__s0__r20.npz').exists()
    assert not (world / 'part2/partitions').exists() and not (world / 'part2/verdict_stage3.json').exists()


# --- stage 5: ArtELingo-28, scale choice per model -------------------------------------------------------------------

N28 = 40
SHA28 = 'LIST746'


def setup28():
    y = np.arange(N28) % 4
    return dict(names=[f'p{i:02d}' for i in range(N28)], sha256=SHA28, labels=dict(emotion=y, n_ties=0))


def put28(root, scale, arm, seed, x, aspect='emotion'):
    """One ArtELingo-28 output where `rb_a28` expects it, with its sidecar (written independently of the scorer)."""
    if arm[0] == 'K':
        path = root / f'part2/{scale}/a28/{arm}/{aspect}__s{seed}.npz'
        key, part = 'img', aspect
    else:
        path = root / f'part2/{scale}/a28/{arm}/{"P2" if arm == "B6" else "P3"}__s{seed}.npz'
        key, part = 'prob', 'percept'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: x.astype(np.float16 if key == 'img' else np.float32)})
    put_sidecar(path, arm, seed, part, scale, SHA28)


def aligned_k(y, seed):
    centers = unit(int(y.max()) + 1, 2048, 100 + seed).astype(np.float32)
    return centers[y]


def write28(root, scale, model, matched, seeds=(0, 1), good=True, b=True):
    y = setup28()['labels']['emotion']
    for s in seeds:
        for aspect in ('emotion', 'style', 'genre'):
            put28(root, scale, model, s, aligned_k(y, s) if good else unit(N28, 2048, s + 7), aspect)
        arms = (('B1', 'B6') if b else ()) + (matched,)
        for arm in arms:
            put28(root, scale, arm, s, np.random.default_rng(s + 3).random((N28, 40)))


def verdict_file(outputs, stage, blocks):
    header = dict(stage=3, scale='sample', held_sha256='H') if stage == 3 else dict(stage=4, scale='full',
                                                                                   held_sha256='H')
    rb_score.write_verdict(outputs / f'part2/verdict_stage{stage}.json', header, blocks)


def run5(roots, models, seeds=(0, 1), setup=None, **kw):
    # Stage 5 fixtures represent candidates that already earned a stage 3 outcome.
    path = roots.read / 'part2/verdict_stage3.json'
    blocks = json.loads(path.read_text())['blocks'] if path.exists() else {}
    for model in models:
        blocks.setdefault(model, dict(outcome_id='S1-GO'))
    verdict_file(roots.read, 3, blocks)
    return score.run_verdict(models, roots.cache, setup or setup28(), workers=1, n_resamples=10, seeds=seeds,
                             roots=roots, **kw)


@pytest.mark.parametrize('model, matched', [('K6', 'M6'), ('K7', 'M7'), ('K8', 'M8'), ('K5', 'M5')])
def test_stage_5_entries_are_b1_b6_and_the_matched_arm_and_a_holds_when_the_model_wins(tmp_path, model, matched):
    roots = score.Roots(tmp_path / 'o', tmp_path / 'o', tmp_path / 'frozen')
    pass_x2(roots.read)
    write28(roots.read, 'sample', model, matched)
    block = run5(roots, [model])['blocks'][model]
    assert block['arms'] == [model, 'B1', 'B6', matched] and block['scale'] == 'sample'
    assert block['outcome_id'] == 'A-HOLDS' and block['d'] > 0.5 and block['interval'][0] > 0
    assert sorted(block['A_e']) == sorted(f'{a}-{r}' for a in ('B1', 'B6', matched) for r in ('r20', 'own'))
    assert block['scale_reason'] == 'no stage 4 outcome' and block['code']['code_commit']
    assert (roots.cache / f'part2/partitions/a28_sample/{matched}/percept__s1__own.npz').exists()


def test_stage_5_reading_is_a_fails_when_an_entry_beats_the_model_and_a_qual_has_a_low_bound_not_above_zero(tmp_path):
    roots = score.Roots(tmp_path / 'o', tmp_path / 'o', tmp_path / 'frozen')
    write28(roots.read, 'sample', 'K6', 'M6', good=False)
    y = setup28()['labels']['emotion']
    for s in (0, 1):                                              # an aligned matched arm
        p = np.full((N28, 40), 0.1, dtype=np.float32)
        p[np.arange(N28), y * 10] = 0.9
        put28(roots.read, 'sample', 'M6', s, p)
    block = run5(roots, ['K6'])['blocks']['K6']
    assert block['outcome_id'] == 'A-FAILS' and block['d'] < -0.5
    assert score.rb_a28.outcome(0.01, -0.001) == 'A-QUAL' and score.rb_a28.outcome(0.01, 0.001) == 'A-HOLDS'


def test_the_scale_is_full_when_stage_4_gave_the_model_an_outcome_else_sample(tmp_path):
    out = tmp_path
    pass_x2(out)
    assert score.scale_and_reason('K6', out) == ('sample', 'no stage 4 outcome')
    verdict_file(out, 4, {'K6': dict(outcome_id='S2-GO'), 'K7': dict(outcome_id=None), 'K8': dict(outcome_id='S2-NO')})
    assert score.scale_and_reason('K6', out) == ('full', None)
    assert score.scale_and_reason('K7', out)[0] == 'sample' and score.scale_and_reason('K5', out)[0] == 'sample'
    assert score.scale_and_reason('K8', out)[0] == 'full'         # an S2-NO is an outcome: stage 4 ran


def test_when_x2_failed_stage_5_runs_at_the_sample_scale_whatever_stage_4_says(tmp_path):
    out = tmp_path
    verdict_file(out, 4, {'K6': dict(outcome_id='S2-GO')})
    pass_x2(out)
    rb_score.write_regression_items(out / 'part2/regression/x2_scorer.json', {'scorer/a': {'pass': False}})
    assert score.scale_and_reason('K6', out)[0] == 'sample' and 'x2_scorer.json fails' in score.scale_and_reason('K6', out)[1]
    assert score.scale_and_reason('K6', out, check_x2=False) == ('full', None)    # the checks ignore the gate


def test_full_scale_reads_b1_and_b6_from_12_24s_full_outputs_and_the_rest_from_the_read_root(tmp_path):
    roots = score.Roots(tmp_path / 'o', tmp_path / 'o', tmp_path / 'frozen')
    verdict_file(roots.read, 4, {'K6': dict(outcome_id='S2-GO')})
    pass_x2(roots.read)
    write28(roots.read, 'full', 'K6', 'M6', b=False)
    for s in (0, 1):
        for arm in ('B1', 'B6'):
            put28(roots.frozen, 'full', arm, s, np.random.default_rng(s + 3).random((N28, 40)))
    block = run5(roots, ['K6'])['blocks']['K6']
    assert block['scale'] == 'full' and 'scale_reason' not in block and block['outcome_id'] == 'A-HOLDS'
    assert str(roots.frozen) in block['sources']['B1']['0']['path']
    assert str(roots.read) in block['sources']['M6']['0']['path'] and str(roots.read) in block['sources']['K6']['0']['path']


def test_sample_scale_reads_b1_and_b6_from_this_rounds_outputs(tmp_path):
    roots = score.Roots(tmp_path / 'o', tmp_path / 'o', tmp_path / 'frozen')
    write28(roots.read, 'sample', 'K6', 'M6')
    block = run5(roots, ['K6'])['blocks']['K6']
    assert str(roots.read) in block['sources']['B6']['1']['path'] and 'frozen' not in block['sources']['B6']['1']['path']


def test_a_model_with_a_missing_or_bad_output_gets_no_outcome_and_the_others_keep_theirs(tmp_path):
    roots = score.Roots(tmp_path / 'o', tmp_path / 'o', tmp_path / 'frozen')
    write28(roots.read, 'sample', 'K6', 'M6')
    write28(roots.read, 'sample', 'K7', 'M7')
    (roots.read / 'part2/sample/a28/K7/genre__s1.npz').unlink()
    put28(roots.read, 'sample', 'K8', 0, unit(N28, 2048, 1))      # K8: a file whose sidecar says another list
    path = roots.read / 'part2/sample/a28/K8/emotion__s0.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    path.with_suffix('.json').write_text(json.dumps(dict(meta, list_sha256='OTHER')))
    record = run5(roots, ['K6', 'K7', 'K8'])
    assert record['blocks']['K6']['outcome_id'] == 'A-HOLDS'
    assert record['blocks']['K7']['outcome_id'] is None and 'genre__s1' in record['blocks']['K7']['v_failure']
    assert record['blocks']['K8']['outcome_id'] is None and 'list_sha256' in record['blocks']['K8']['v_failure']
    assert record['blocks']['K7']['scale'] == 'sample' and record['blocks']['K7']['arms'] == ['K7', 'B1', 'B6', 'M7']


# --- X1 and X2, scorer half: 12-24's scorer values from 12-24's saved outputs ----------------------------------------

@pytest.fixture
def frozen(world, tmp_path, monkeypatch):
    """A stand-in for 12-24's `part2`: outputs of K1, K2, M2 and B1/B6, and verdicts written by 12-24's own scorer
    (`rb_score.run_stage`, `rb_a28.run_verdict`) with 6 draws."""
    froz = tmp_path / 'frozen'
    n = 60
    rng = np.random.default_rng(1)
    for scale in ('sample', 'full'):
        for s in SEEDS5:
            put_student(froz, scale, 'K2', s, n)
            put_prob(score.output_path(scale, 'M2', s, outputs=froz), 'M2', s, scale, rng.random((n, 40)))
    ct_full = tmp_path / 'ct_full.json'
    ct_full.write_text('{}')
    monkeypatch.setitem(rb_score.CT_VERDICTS, 'full', ct_full)
    monkeypatch.setattr(score, 'CHECKS', {'x1': ((3, ('K2',)), (5, ('K1', 'K2'))), 'x2': ((4, ('K2',)),)})
    rb_score.run_stage(3, ['K2'], froz, workers=1, n_resamples=6, gate=False)
    rb_score.run_stage(4, ['K2'], froz, workers=1, n_resamples=6, gate=False)
    write28(froz, 'full', 'K2', 'M2', seeds=SEEDS5)
    write28(froz, 'full', 'K1', 'B1', seeds=SEEDS5, b=True)
    score.rb_a28.run_verdict(['K1', 'K2'], froz, setup28(), workers=1, n_resamples=6, seeds=SEEDS5)
    monkeypatch.setattr(score.rb_a28, 'real_setup', setup28)
    return froz


def tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob('*'))


def test_x1_passes_on_the_saved_outputs_and_writes_only_under_outputs(frozen, tmp_path):
    before = tree(frozen)
    out = tmp_path / 'check'
    record = score.run_check('x1', out, workers=1, n_resamples=6, frozen=frozen)
    assert record['pass'] is True
    names = set(record['items'])
    for stage, model in ((3, 'K2'), (5, 'K1'), (5, 'K2')):
        for item in ('A_K1', 'd', 'interval_lo', 'interval_hi', 'outcome', 'entries', 'bootstrap_d', 'arms'):
            assert f'scorer/stage{stage}/{model}/{item}' in names
    assert 'scorer/stage5/K2/A_e/M2-own' in names and 'scorer/stage3/K2/A_e/B4-r20' in names
    assert json.loads((out / 'part2/regression/x1_scorer.json').read_text()) == record
    assert tree(frozen) == before
    assert (out / 'part2/partitions/sample/K2/emotion__s0__r20.npz').exists()
    assert (out / 'part2/partitions/a28_full/K1/emotion__s0__r20.npz').exists()


def test_x2_is_the_same_check_for_stage_4(frozen, tmp_path):
    record = score.run_check('x2', tmp_path / 'check', workers=1, n_resamples=6, frozen=frozen)
    assert record['pass'] is True and 'scorer/stage4/K2/d' in record['items']
    assert not any(k.startswith(('scorer/stage3', 'scorer/stage5')) for k in record['items'])
    assert (tmp_path / 'check/part2/regression/x2_scorer.json').exists()


@pytest.mark.parametrize('stage, edit, failing', [
    (3, lambda b: b.update(d=b['d'] + 1e-4), 'scorer/stage3/K2/d'),
    (3, lambda b: b['A_e'].update({'B1-own': b['A_e']['B1-own'] + 2e-6}), 'scorer/stage3/K2/A_e/B1-own'),
    (3, lambda b: b['interval'].__setitem__(1, b['interval'][1] - 1e-5), 'scorer/stage3/K2/interval_hi'),
    (3, lambda b: b.update(outcome_id='S1-NO' if b['outcome_id'] != 'S1-NO' else 'S1-GO'), 'scorer/stage3/K2/outcome'),
    (5, lambda b: b['A_e'].update({'M2-r20': b['A_e']['M2-r20'] + 1e-3}), 'scorer/stage5/K2/A_e/M2-r20'),
])
def test_a_forced_mismatch_fails_the_check_and_names_the_item(frozen, tmp_path, stage, edit, failing):
    path = frozen / f'part2/verdict_stage{stage}.json'
    saved = json.loads(path.read_text())
    edit(saved['blocks']['K2'])
    path.write_text(json.dumps(saved))
    record = score.run_check('x1', tmp_path / 'check', workers=1, n_resamples=6, frozen=frozen)
    assert record['pass'] is False and record['items'][failing]['pass'] is False
    others = [k for k, v in record['items'].items() if not v['pass']]
    assert failing in others and not any(k.startswith('scorer/stage5/K1') for k in others)


def test_a_difference_below_1e_6_passes(frozen, tmp_path):
    path = frozen / 'part2/verdict_stage3.json'
    saved = json.loads(path.read_text())
    saved['blocks']['K2']['d'] += 5e-7
    path.write_text(json.dumps(saved))
    assert score.run_check('x1', tmp_path / 'check', workers=1, n_resamples=6, frozen=frozen)['pass'] is True


def test_a_candidate_the_scorer_cannot_read_fails_its_items_with_the_cause(frozen, tmp_path):
    (frozen / 'part2/sample/held/K2/style__s1.npz').unlink()
    record = score.run_check('x1', tmp_path / 'check', workers=1, n_resamples=6, frozen=frozen)
    assert record['pass'] is False
    assert 'style__s1' in record['items']['scorer/stage3/K2/outcome']['cause']


def test_the_check_needs_the_saved_draw_count_and_an_outputs_folder_outside_12_24s(frozen, tmp_path):
    with pytest.raises(AssertionError, match='draws'):
        score.run_check('x1', tmp_path / 'check', workers=1, n_resamples=7, frozen=frozen)
    with pytest.raises(AssertionError, match='frozen'):
        score.run_check('x1', frozen / 'inside', workers=1, n_resamples=6, frozen=frozen)


def test_cli_refuses_smoke_outside_tmp_and_never_shrinks_a_check(tmp_path, monkeypatch):
    # PYTEST_DEBUG_TEMPROOT itself may be inside the production smoke root.
    monkeypatch.setattr(eg_paths, 'SMOKE_ROOT', tmp_path / 'allowed-smoke')
    with pytest.raises(AssertionError, match='/tmp/claude-0'):
        score.main(['--stage', '3', '--candidates', 'K6', '--smoke', '--outputs', str(tmp_path / 'not-tmp')])
    with pytest.raises(AssertionError):
        score.main(['--check', 'x1', '--smoke', '--outputs', '/tmp/claude-0/x'])
    with pytest.raises(SystemExit):
        score.main(['--check', 'stage2'])


def test_k1_is_always_full_and_its_b1_is_its_only_matched_arm(tmp_path, monkeypatch):
    ct_full = tmp_path / 'ct_full.json'
    ct_full.write_text('{}')
    monkeypatch.setitem(rb_score.CT_VERDICTS, 'full', ct_full)
    assert score.scale_and_reason('K1', tmp_path) == ('full', None)
    assert score.arms_a28('K1') == ('K1', 'B1', 'B6') and score.arms_a28('K4') == ('K4', 'B1', 'B6')
    assert score.arms_a28('K2') == ('K2', 'B1', 'B6', 'M2')


# --- fix round: a kept block is checked against its recorded inputs; --replace per candidate; X2 not passed ------------

def pass_x2(out):
    for part in score.X2_PARTS:
        rb_score.write_regression_items(out / f'part2/regression/x2_{part}.json', {f'{part}/a': {'pass': True}})


def test_a_kept_stage_5_block_at_a_scale_no_longer_due_is_refused_until_a_reason_is_given(tmp_path):
    roots = score.Roots(tmp_path / 'o', tmp_path / 'o', tmp_path / 'frozen')
    pass_x2(roots.read)
    write28(roots.read, 'sample', 'K6', 'M6')
    assert run5(roots, ['K6'])['blocks']['K6']['scale'] == 'sample'
    verdict_file(roots.read, 4, {'K6': dict(outcome_id='S2-GO')})        # stage 4 now has an outcome: full is due
    assert score.scale_and_reason('K6', roots.read) == ('full', None)
    with pytest.raises(AssertionError, match='give --replace REASON'):
        run5(roots, ['K6'])
    write28(roots.read, 'full', 'K6', 'M6', b=False)                     # the reason lets it recompute at full
    for s in (0, 1):
        for arm in ('B1', 'B6'):
            put28(roots.frozen, 'full', arm, s, np.random.default_rng(s + 3).random((N28, 40)))
    block = run5(roots, ['K6'], replace='V3 wrong scale')['blocks']['K6']
    assert block['scale'] == 'full'
    assert run5(roots, ['K6'])['blocks']['K6'] == block                  # and now the rerun keeps it


def test_a_kept_stage_5_block_is_kept_while_its_scale_and_outputs_stand(tmp_path):
    roots = score.Roots(tmp_path / 'o', tmp_path / 'o', tmp_path / 'frozen')
    write28(roots.read, 'sample', 'K6', 'M6')
    first = run5(roots, ['K6'])['blocks']['K6']
    assert run5(roots, ['K6'])['blocks']['K6'] == first


def test_a_kept_stage_5_block_whose_output_was_rewritten_is_refused(tmp_path):
    roots = score.Roots(tmp_path / 'o', tmp_path / 'o', tmp_path / 'frozen')
    write28(roots.read, 'sample', 'K6', 'M6')
    run5(roots, ['K6'])
    put28(roots.read, 'sample', 'M6', 1, np.random.default_rng(99).random((N28, 40)))
    with pytest.raises(AssertionError, match='give --replace REASON'):
        run5(roots, ['K6'])


def test_a_kept_stage_block_whose_held_output_was_rewritten_is_refused(world):
    first = score.run_stage(3, ['K6'], world, workers=1, n_resamples=3)['blocks']['K6']
    p = score.output_path('sample', 'K6', 0, 'emotion', world)
    np.savez(p, img=unit(60, 2048, 999))
    put_sidecar(p, 'K6', 0, 'emotion', 'sample')
    with pytest.raises(AssertionError, match='give --replace REASON'):
        score.run_stage(3, ['K6'], world, workers=1, n_resamples=3)
    for stale in (world / 'part2/partitions/sample/K6').glob('emotion__s0__*'):
        stale.unlink()                      # the partition cache is keyed on the output hash and refuses a changed one
    redone = score.run_stage(3, ['K6'], world, workers=1, n_resamples=3, replace='V3 output rewritten')['blocks']['K6']
    assert redone['sources']['held_outputs']['K6']['0']['output_sha256'] == rb_score.sha_file(p)
    assert redone != first


def test_a_kept_stage_block_whose_rcol_file_changed_is_refused(world, tmp_path):
    score.run_stage(3, ['K6'], world, workers=1, n_resamples=3)
    (tmp_path / 'rcol.json').write_text('{"changed": 1}')
    with pytest.raises(AssertionError, match='give --replace REASON'):
        score.run_stage(3, ['K6'], world, workers=1, n_resamples=3)


def test_a_kept_block_without_an_outcome_has_no_sources_to_check_and_is_kept(world):
    (world / 'part2/sample/held/K6/genre__s2.npz').unlink()
    first = score.run_stage(3, ['K6'], world, workers=1, n_resamples=3)['blocks']['K6']
    assert first['outcome_id'] is None
    assert score.run_stage(3, ['K6'], world, workers=1, n_resamples=3)['blocks']['K6'] == first


def put_k7(world):
    rng = np.random.default_rng(5)
    for scale in ('sample', 'full'):
        for s in SEEDS5:
            put_student(world, scale, 'K7', s, 60)
            put_prob(score.output_path(scale, 'M7', s, outputs=world), 'M7', s, scale, rng.random((60, 40)))


def test_each_replaced_candidate_keeps_its_reason_and_the_previous_hash(world):
    put_k7(world)
    score.run_stage(3, ['K6', 'K7'], world, workers=1, n_resamples=3)
    path = world / 'part2/verdict_stage3.json'
    before_k6 = rb_score.sha_file(path)
    (world / 'part2/sample/held/K6/genre__s2.npz').unlink()
    score.run_stage(3, ['K6'], world, workers=1, n_resamples=3, replace='V3 first reason')
    before_k7 = rb_score.sha_file(path)
    (world / 'part2/sample/held/K7/genre__s2.npz').unlink()
    record = score.run_stage(3, ['K7'], world, workers=1, n_resamples=3, replace='V3 second reason')
    rep = record['replacement']
    assert rep['reason'] == 'V3 second reason' and rep['blocks'] == ['K7'] and rep['previous_sha256'] == before_k7
    assert [(h['blocks'], h['reason'], h['previous_sha256']) for h in rep['history']] == [
        (['K6'], 'V3 first reason', before_k6), (['K7'], 'V3 second reason', before_k7)]
    assert json.loads(path.read_text()) == record
    assert record['blocks']['K6']['outcome_id'] is None and record['blocks']['K7']['outcome_id'] is None


def test_replace_needs_explicit_candidates_on_the_command_line(tmp_path):
    with pytest.raises(SystemExit):
        score.main(['--stage', '3', '--replace', 'V3 redo', '--outputs', str(tmp_path)])


def test_x2_not_passed_means_a_missing_or_failing_file_and_the_reason_says_which(tmp_path):
    out = tmp_path
    verdict_file(out, 4, {'K6': dict(outcome_id='S2-GO')})
    scale, reason = score.scale_and_reason('K6', out)
    assert scale == 'sample' and all(f'x2_{part}.json missing' in reason for part in score.X2_PARTS)
    pass_x2(out)
    assert score.scale_and_reason('K6', out) == ('full', None)
    rb_score.write_regression_items(out / 'part2/regression/x2_matched.json', {'matched/b': {'pass': False}})
    scale, reason = score.scale_and_reason('K6', out)
    assert scale == 'sample' and 'x2_matched.json fails' in reason and 'missing' not in reason
    (out / 'part2/regression/x2_student.json').unlink()
    assert 'x2_student.json missing' in score.scale_and_reason('K6', out)[1]


def test_stage_4_needs_a_stage_3_block_with_an_outcome_other_than_s1_no(world):
    with pytest.raises(AssertionError, match='stage 3'):
        score.run_stage(4, ['K6'], world, workers=1, n_resamples=3)          # no stage 3 file
    for block in (dict(outcome_id=None), dict(outcome_id='S1-NO')):
        verdict_file(world, 3, {'K6': block})
        with pytest.raises(AssertionError, match='K6.*stage 3'):
            score.run_stage(4, ['K6'], world, workers=1, n_resamples=3)
        (world / 'part2/verdict_stage3.json').unlink()
    verdict_file(world, 3, {'K6': dict(outcome_id='S1-MM')})
    assert score.run_stage(4, ['K6'], world, workers=1, n_resamples=3)['blocks']['K6']['outcome_id'].startswith('S2-')


def test_s1_go_includes_exact_margin_threshold_through_stage_block(world, monkeypatch):
    # Stub only the statistic: stage_block must still reach candidate_block and rb_score.outcome.
    monkeypatch.setattr(rb_score, 'paired_bootstrap', lambda *a, **kw: dict(
        d=.01, interval=[.001, .02], bootstrap_d=[.01], A_cand=.11, A_e={'B1-own': .1}, max_entry='B1-own'))
    block = score.run_stage(3, ['K6'], world, workers=1, n_resamples=1)['blocks']['K6']
    assert block['d'] == .01 and block['outcome_id'] == 'S1-GO'
