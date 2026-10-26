"""Rule examples for VA2/VB2, using only this ticket's independent implementation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rg_v2 as v2


def record(a, emotion=0., style=0., status='ok'):
    effects = {k: dict(A=a, e=e, lo=0., hi=0.) for k, e in
               [('genre', .04), ('emotion', emotion), ('style', style)]}
    return v2.stage_a_record(dict(status=status, cause='nonfinite' if status == 'failed' else None,
                                 best_epoch={}), effects if status == 'ok' else None)


def test_climb_strictly_rising_runs_next_rung():
    records = {'GR-8': record(.50), 'GR-16': record(.51), 'GR-32': record(.52)}
    decision = v2.climb_decision(records)
    assert decision['next'] == 'GR-64'
    assert decision['still_rising'] is True


@pytest.mark.parametrize('newest, reason', [
    (record(.50), 'not_strictly_above'),
    (record(.49), 'not_strictly_above'),
    (record(.60, emotion=-.004600001), 'unguarded'),
    (record(.60, style=-.008200001), 'unguarded'),
    (record(.60, status='failed'), 'failed'),
])
def test_climb_stops_for_tie_lower_guard_or_failure(newest, reason):
    decision = v2.climb_decision({'GR-8': record(.50), 'GR-16': record(.48), 'GR-32': newest})
    assert decision['next'] is None
    assert decision['reason'] == reason


def test_climb_ignores_unguarded_rival_and_includes_guard_boundaries():
    decision = v2.climb_decision({'GR-8': record(.50), 'GR-16': record(.90, style=-.1),
                                 'GR-32': record(.51, emotion=-.0046, style=-.0082)})
    assert decision['next'] == 'GR-64'


def test_climb_reports_still_rising_at_cap():
    records = {v: record(.5 + i*.01) for i, v in enumerate(v2.variants.LADDER)}
    assert v2.climb_decision(records)['next'] is None
    assert v2.climb_decision(records)['still_rising'] is True


@pytest.mark.parametrize('eq, winner, outcome', [(True, 'GQ-8', 'A-NEW'), (False, 'GR-8', 'A-SAME')])
def test_seed_zero_eq_exclusion_and_tie_order(eq, winner, outcome):
    records = {'GR-8': record(.51), 'GR-16': record(.51), 'GR-32': record(.7, emotion=-.01),
               'GN-8': record(.8, status='failed'), 'GQ-8': record(.52)}
    decision = v2.choose_a(records, eq)
    assert (decision['winner'], decision['outcome']) == (winner, outcome)


def fresh_effect(e=.0128, lo=.001, emotion=-.0046, style=-.0082, a=.51):
    return {'genre': dict(Abar={'GR-8': a, 'R6-5': a-e}, e=e, lo=lo, hi=.04),
            'emotion': dict(e=emotion), 'style': dict(e=style)}


@pytest.mark.parametrize('kw, expected', [({}, 'C-HOLDS'), ({'lo': 0}, 'C-QUAL'),
    ({'e': .01279999}, 'C-QUAL'), ({'e': 0}, 'C-FAILS'), ({'e': -.01}, 'C-FAILS'),
    ({'emotion': -.00460001}, 'C-FAILS'), ({'style': -.00820001}, 'C-FAILS')])
def test_fresh_confirmation_boundaries(kw, expected):
    assert v2.confirmation(fresh_effect(**kw)) == expected


def test_fresh_pick_uses_only_confirmed_arms_and_rule_ties():
    records = {v: dict(confirmed=c, Abar_genre=a) for v, c, a in
               [('GR-2', True, .51), ('GR-4', True, .50), ('GR-8', True, .51), ('GN-8', False, .60)]}
    assert v2.choose_b(records, 'GN-8')['v_held'] == 'GR-8'
    assert v2.choose_b(records, 'GN-8')['within_one_range'] == ['GR-2', 'GR-4']
    for r in records.values():
        r['confirmed'] = False
    assert v2.choose_b(records, 'GN-8')['outcome'] == 'B-NONE'
    assert v2.choose_b(records, 'GN-8')['v_held'] is None


@pytest.fixture
def raw_tree():
    # Keep synthetic writes inside the authorized worktree, and retain everything.
    import uuid
    root = Path(__file__).resolve().parents[3] / '.scratch' / 't09-tests' / uuid.uuid4().hex
    root.mkdir(parents=True)
    import numpy as np
    names = [f'p{i:03d}' for i in range(400)]
    x = np.tile(np.eye(20, dtype=np.float16), (20, 1))
    annotations = [dict(painting=n, emotion=f'e{i%20:02d}', art_style=f's{i%20:02d}')
                   for i, n in enumerate(names)]
    p_sha = v2.sha('\n'.join(names).encode())
    rb, outputs, frozen = root/'rb', root/'out', root/'frozen'
    v2.write_json(rb/'P.json', dict(names=names, sha256=p_sha))
    for arm in [v2.BASE]+list(v2.variants.A_TIE_ORDER[:3])+list(v2.variants.A_OTHER):
        folder = rb/'train'/arm if arm == v2.BASE else frozen/arm/'s0' if arm == 'GR-8' else outputs/'train'/arm/'s0'
        folder.mkdir(parents=True, exist_ok=True)
        if arm != v2.BASE:
            v2.write_json(folder/'run.json', dict(variant=arm, seed=0, status='ok', cause=None,
                best_epoch={a: 1 for a in v2.ASPECTS}, p_sha256=p_sha,
                switches=v2.variants.eg_form(v2.variants.switches(arm)) if arm == 'GR-8' else v2.variants.switches(arm),
                code_commit='fixture'))
        for a in v2.ASPECTS:
            np.savez(folder/(f'{a}__s0__P.npz' if arm == v2.BASE else f'{a}__P.npz'), img=x)
            if arm == 'GR-16':
                continue  # RG exports need only their run.json, not an aspect sidecar.
            v2.write_json(folder/(f'{a}__s0.json' if arm == v2.BASE else f'{a}.json'),
                dict(recipe=arm, aspect=a, seed=0, best_epoch=1, p_sha256=p_sha,
                     arrays={f'{a}__P.npz': dict(img=v2.array_sha(x))}))
    v2.write_json(outputs/'regression/eq.json', {'pass': True})
    return dict(outputs=outputs, rb_part1=rb, frozen_root=frozen, n_rows=400, width=20,
                expected_p_sha=p_sha, annotations=annotations, genre_ids=np.arange(400, dtype=np.int64)%20,
                smoke=True, workers=1)


def test_synthetic_exports_rederive_exact_scores_before_opening_comparison(raw_tree):
    report = v2.run(phase='a', **raw_tree)
    assert report['recomputed']['winner'] == 'GR-8'
    assert report['recomputed']['outcome'] == 'A-SAME'
    for result in report['recomputed']['results'].values():
        assert result['effects']['genre'] == dict(A=1., e=0., lo=0., hi=0.)
    # Absence of a comparison record cannot produce agreement.
    assert not report['agree']
    assert any('climb.json' in d['theirs'] for d in report['differences'])


def standin_a(tree):
    """An independently specified scorer record: identical exports have zero effects."""
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits
    out, rb, frozen = tree['outputs'], tree['rb_part1'], tree['frozen_root']
    order = list(v2.variants.LADDER[:3])+list(v2.variants.A_OTHER)
    p_sha = tree['expected_p_sha']
    y = np.arange(400, dtype=np.int64)%20
    identity = dict(p_sha256=p_sha, label_sha256={a: v2.array_sha(y) for a in v2.ASPECTS},
                    n_draws=20, arms=order, img_sha256={}, run_sha256={}, failed={})
    results = {}
    x = np.tile(np.eye(20, dtype=np.float16), (20, 1))
    with threadpool_limits(limits=8):
        parts = np.stack([MiniBatchKMeans(n_clusters=20, random_state=s, n_init=3,
                                         batch_size=4096).fit_predict(x.astype(np.float32))
                          for s in (0, 1, 2)]).astype(np.int64)
    for v in [v2.BASE]+order:
        identity['img_sha256'][v] = {a: v2.array_sha(x) for a in v2.ASPECTS}
        for a in v2.ASPECTS:
            dest = out/'stage_a/partitions'/v/'s0'/f'{a}__partitions.npz'
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dest, labels=parts)
        if v == v2.BASE:
            continue
        folder = frozen/v/'s0' if v == 'GR-8' else out/'train'/v/'s0'
        identity['run_sha256'][v] = v2.sha((folder/'run.json').read_bytes())
        results[v] = dict(status='ok', cause=None, best_epoch={a: 1 for a in v2.ASPECTS},
            effects={a: dict(A=1., e=0., lo=0., hi=0.) for a in v2.ASPECTS},
            A_genre=1., e_genre=0., guarded=True)
    verdict = dict(stage='a', smoke=True, n_draws=20, reference='R6-5', seed=0,
        t_g=v2.variants.T_G, promising=v2.variants.T_G-.0128, ranges=v2.RANGES,
        label_counts={a: 400 for a in v2.ASPECTS}, base_A={a: 1. for a in v2.ASPECTS},
        arms=order, climb_rungs=list(v2.variants.LADDER[:3]), still_rising_at_cap=False,
        results=results, eq_pass=True, winner='GR-8', A_genre_winner=1., outcome='A-SAME',
        climb_reason='tie', levels={v: 'target' for v in order}, kept_weights=order, pending_deletions=[],
        stage_b_new=['GR-2', 'GR-4'], identity=identity)
    v2.write_json(out/'stage_a/verdict.json', verdict)
    climb_identity = dict(identity, arms=order[:3],
        img_sha256={v: identity['img_sha256'][v] for v in [v2.BASE]+order[:3]},
        run_sha256={v: identity['run_sha256'][v] for v in order[:3]})
    v2.write_json(out/'stage_a/climb.json', dict(stage='climb', smoke=True, t_g=v2.variants.T_G,
        promising=v2.variants.T_G-.0128, ranges=v2.RANGES, p_sha256=p_sha, decisions=[dict(
            rungs_run=list(v2.variants.LADDER[:3]), newest='GR-32', next=None,
            still_rising=False, reason='tie', identity=climb_identity, rivals={'GR-8': 1., 'GR-16': 1.},
            rungs={v: dict(status='ok', cause=None, A_genre=1., e_genre=0., e_emotion=0.,
                           e_style=0., guarded=True) for v in v2.variants.LADDER[:3]})]))
    return verdict


def test_synthetic_comparison_checks_values_partitions_and_decisions(raw_tree):
    verdict = standin_a(raw_tree)
    report = v2.run(phase='a', **raw_tree)
    assert report['agree'], report['differences']
    verdict['results']['GR-16']['effects']['style']['lo'] = .001
    verdict['winner'] = 'GR-16'
    v2.write_json(raw_tree['outputs']/'stage_a/verdict.json', verdict)
    report = v2.run(phase='a', **raw_tree)
    assert not report['agree']
    assert any(d['path'].endswith('/style/lo') for d in report['differences'])
    assert any(d['path'].endswith('/winner') for d in report['differences'])


def test_rung_compares_last_decision_and_exact_partitions(raw_tree):
    import numpy as np
    standin_a(raw_tree)
    report = v2.run(phase='rung', arm='GR-32', **raw_tree)
    assert report['agree'], report['differences']
    cache = raw_tree['outputs']/'stage_a/partitions/GR-32/s0/genre__partitions.npz'
    with np.load(cache) as z:
        parts = z['labels'].copy()
    parts[0, 0] = (parts[0, 0]+1)%20
    np.savez(cache, labels=parts)
    report = v2.run(phase='rung', **raw_tree)
    assert not report['agree']
    assert any(d['path'] == 'partitions/a/GR-32/s0/genre' for d in report['differences'])


def standin_b(tree):
    import shutil
    import numpy as np
    standin_a(tree)
    out, frozen = tree['outputs'], tree['frozen_root']
    arms = ['GR-2', 'GR-4', 'GR-8']
    hashes, run_hashes = {}, {}
    for v in [v2.BASE]+arms:
        hashes[v], run_hashes[v] = {}, {}
        for s in (1, 2, 3):
            folder = (frozen if v in (v2.BASE, 'GR-8') else out/'train')/v/f's{s}'
            folder.mkdir(parents=True, exist_ok=True)
            run = dict(variant=v, seed=s, status='ok', cause=None, best_epoch={a: 1 for a in v2.ASPECTS},
                p_sha256=tree['expected_p_sha'], code_commit='fixture',
                switches=v2.variants.eg_form(v2.variants.switches(v)) if v in (v2.BASE, 'GR-8') else v2.variants.switches(v))
            v2.write_json(folder/'run.json', run)
            run_hashes[v][str(s)] = v2.sha((folder/'run.json').read_bytes())
            hashes[v][str(s)] = {}
            for a in v2.ASPECTS:
                src = frozen/'GR-8/s0'
                shutil.copyfile(src/f'{a}__P.npz', folder/f'{a}__P.npz')
                shutil.copyfile(src/f'{a}.json', folder/f'{a}.json')
                with np.load(folder/f'{a}__P.npz') as z:
                    hashes[v][str(s)][a] = v2.array_sha(z['img'])
                cache = out/'stage_b/partitions'/v/f's{s}'/f'{a}__partitions.npz'
                cache.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(out/'stage_a/partitions/GR-8/s0'/f'{a}__partitions.npz', cache)
    effects = {v: {a: dict(Abar={v: 1., v2.BASE: 1.}, e=0., lo=0., hi=0.)
                   for a in v2.ASPECTS} for v in arms}
    ref = frozen/'score_fresh.json'
    v2.write_json(ref, dict(effects=effects['GR-8'], identity={'img_sha256': {v: hashes[v] for v in ('GR-8', v2.BASE)}}))
    verdict = dict(stage='b', smoke=True, n_draws=20, reference=v2.BASE, seeds=[1, 2, 3],
        t_g=v2.variants.T_G, promising=v2.variants.T_G-.0128, ranges=v2.RANGES,
        label_counts={a: 400 for a in v2.ASPECTS}, a_outcome='A-SAME', v_star_a='GR-8',
        stage_b_arms=arms, vb1_pass=True,
        results={v: dict(status='ok', cause=None, effects=effects[v], Abar_genre=1., e_genre=0.,
                         c_outcome='C-FAILS', confirmed=False) for v in arms},
        outcome='B-NONE', v_held=None, v_held_switches=None, confirmed=[], within_one_range=[],
        levels={v: 'C-FAILS' for v in arms}, identity=dict(p_sha256=tree['expected_p_sha'],
            label_sha256={a: v2.array_sha(tree['genre_ids']) for a in v2.ASPECTS},
            n_draws=20, arms=arms, img_sha256=hashes, run_sha256=run_hashes, failed={}))
    v2.write_json(out/'stage_b/verdict.json', verdict)
    # Keep the actual external JSON spelling of pass.
    v2.write_json(out/'stage_b/vb1.json', {'pass': True, 'hashes_checked': True, 'bounds_checked': False,
        'items': {'genre/e': {'value': 0., 'expected': 0., 'diff': 0., 'pass': True}},
        'p_sha256': tree['expected_p_sha'], 'reference_sha256': v2.sha(ref.read_bytes()),
        'identity': {'img_sha256': {v: hashes[v] for v in ('GR-8', v2.BASE)}}})
    return ref, verdict


def test_fresh_exports_compare_hashes_means_intervals_and_outcome(raw_tree):
    ref, verdict = standin_b(raw_tree)
    report = v2.run(phase='b', fresh_reference=ref, **raw_tree)
    assert report['agree'], report['differences']
    verdict['results']['GR-2']['effects']['genre']['Abar']['GR-2'] = .8
    verdict['outcome'] = 'B-PICK'
    v2.write_json(raw_tree['outputs']/'stage_b/verdict.json', verdict)
    report = v2.run(phase='b', fresh_reference=ref, **raw_tree)
    assert not report['agree']
    assert any(d['path'].endswith('/Abar/GR-2') for d in report['differences'])
    assert any(d['path'].endswith('/outcome') for d in report['differences'])


def test_fresh_bootstrap_draws_paintings_then_shared_seed_list():
    import numpy as np
    y = np.arange(200, dtype=np.int64)%2
    good = np.tile(y, (3, 1))
    blank = np.zeros((3, 200), dtype=np.int64)
    parts = {'R6-5': {1: blank, 2: good, 3: blank},
             'GR-8': {1: good, 2: blank, 3: good}}
    result = v2.score_effects(y, parts, n_draws=20, workers=1, fresh=True)['GR-8']
    assert result['Abar']['GR-8'] == pytest.approx(2/3)
    assert result['Abar']['R6-5'] == pytest.approx(1/3)
    assert result['e'] == pytest.approx(1/3)
    # For these exact 0/1 AMIs, each draw is 1 - 2*(count of seed 2)/3.
    # The specified stream has one all-seed-2 draw and six no-seed-2 draws.
    assert result['lo'] == pytest.approx(-.6833333333333333)
    assert result['hi'] == pytest.approx(1.)


def test_final_invalid_export_after_rerun_is_a_failed_rung(raw_tree):
    import numpy as np
    folder = raw_tree['outputs']/'train/GR-32/s0'
    run = __import__('json').loads((folder/'run.json').read_text())
    run['rerun_of'] = 'first-va3-failure'
    v2.write_json(folder/'run.json', run)
    np.savez(folder/'genre__P.npz', img=np.zeros((400, 20), dtype=np.float16))
    report = v2.run(phase='rung', **raw_tree)
    assert report['recomputed']['rungs']['GR-32']['status'] == 'failed'
    assert report['recomputed']['rungs']['GR-32']['A_genre'] is None
    assert report['recomputed']['next'] is None


@pytest.mark.parametrize('problem', ['shape', 'dtype', 'nonfinite', 'norm', 'sidecar_hash'])
def test_invalid_exports_never_produce_agreement(raw_tree, problem):
    import numpy as np
    folder = raw_tree['outputs']/'train/GR-32/s0'
    x = np.tile(np.eye(20, dtype=np.float16), (20, 1))
    if problem == 'shape':
        x = x[:-1]
    elif problem == 'dtype':
        x = x.astype(np.float32)
    elif problem == 'nonfinite':
        x[0, 0] = np.nan
    elif problem == 'norm':
        x[0] = 0
    else:
        x = np.roll(x, 1, axis=0)
    np.savez(folder/'genre__P.npz', img=x)
    report = v2.run(phase='rung', **raw_tree)
    assert not report['agree']
    assert report['recomputed'] is None


def test_failed_fresh_seed_has_no_effect_or_confirmation(raw_tree):
    import json
    ref, _ = standin_b(raw_tree)
    folder = raw_tree['outputs']/'train/GR-2/s2'
    run = json.loads((folder/'run.json').read_text())
    run.update(status='failed', cause='nonfinite loss', failure_kind='nonfinite_loss', best_epoch={})
    v2.write_json(folder/'run.json', run)
    report = v2.run(phase='b', fresh_reference=ref, **raw_tree)
    result = report['recomputed']['results']['GR-2']
    assert result['status'] == 'failed'
    assert result['effects'] is None
    assert result['confirmed'] is False
    assert result['c_outcome'] is None


def test_vb1_detects_changed_frozen_array_hash(raw_tree):
    import json
    ref, _ = standin_b(raw_tree)
    reference = json.loads(ref.read_text())
    reference['identity']['img_sha256']['GR-8']['1']['genre'] = 'wrong'
    v2.write_json(ref, reference)
    report = v2.run(phase='b', fresh_reference=ref, **raw_tree)
    assert not report['agree']
    assert report['recomputed']['vb1_pass'] is False
    assert any(d['path'].endswith('/frozen_hashes/GR-8/1/genre') for d in report['differences'])


def test_cli_returns_one_on_disagreement_and_rejects_unsafe_smoke_path(raw_tree):
    assert v2.main(['--phase', 'rung', '--outputs', str(raw_tree['outputs'])]) == 1
    with pytest.raises(SystemExit) as exc:
        v2.main(['--phase', 'a', '--smoke', '--outputs', str(raw_tree['outputs'])])
    assert exc.value.code == 2


@pytest.mark.parametrize('a, level', [(v2.variants.T_G, 'target'),
    (v2.variants.T_G-.0128, 'promising'), (v2.variants.T_G-.01280000001, 'neither'), (None, 'neither')])
def test_seed_zero_target_levels_use_unrounded_boundaries(a, level):
    assert v2.target_level(a) == level


def test_new_seed_zero_winner_uses_ladder_tie_order():
    result = v2.choose_a({'GR-8': record(.50), 'GR-16': record(.51), 'GR-32': record(.51)}, True)
    assert result['winner'] == 'GR-16'
    assert result['stage_b_new'] == ['GR-2', 'GR-4', 'GR-16']


def test_new_held_candidate_and_new_winner_tie_priority():
    records = {v: dict(confirmed=True, Abar_genre=a) for v, a in
               [('GR-8', .5), ('GR-2', .49), ('GR-4', .52), ('GN-8', .53)]}
    assert v2.choose_b(records, 'GN-8')['v_held'] == 'GN-8'
    records['GN-8']['Abar_genre'] = .52
    assert v2.choose_b(records, 'GN-8')['v_held'] == 'GR-4'


def test_labels_use_caption_plurality_smallest_tie_first_style_and_mask(raw_tree):
    import numpy as np
    # p000: one e00 and one e01, so smallest code wins; later style is ignored.
    raw_tree['annotations'].append(dict(painting='p000', emotion='e01', art_style='s19'))
    raw_tree['genre_ids'] = raw_tree['genre_ids'].copy()
    raw_tree['genre_ids'][:10] = -1
    report = v2.run(phase='rung', **raw_tree)
    y = np.arange(400, dtype=np.int64)%20
    assert report['label_sha256']['emotion'] == v2.array_sha(y)
    assert report['label_sha256']['style'] == v2.array_sha(y)
    assert report['label_counts']['genre'] == 390
    assert report['recomputed']['rungs']['GR-32']['e_genre'] == 0.


def test_independence_has_no_prohibited_scorer_imports():
    assert not {'rg_score', 'rg_report', 'eg_score3', 'rb_pick', 'rb_ami'} & sys.modules.keys()
