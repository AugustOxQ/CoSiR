"""VG2 behavior, using small inputs and the observed stage-iii file formats."""
import hashlib
import json

import numpy as np
import pytest
import scipy
import sklearn
import threadpoolctl
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

import eg_vg2 as vg


def result(genre, emotion=0., style=0.):
    effects = {a: dict(A=genre if a == 'genre' else .1, e=e, lo=e, hi=e)
               for a, e in zip(vg.ASPECTS, (emotion, style, .02))}
    return vg.variant_result(dict(status='ok', cause=None,
                                  best_epoch=dict.fromkeys(vg.ASPECTS, 1)), effects)


def test_hand_computed_ami_effect_and_interval(tmp_path):
    # For [0,0,1,1], the crossed partition has MI=0 and E[MI]=log(2)/3,
    # hence AMI=-1/2. A perfect partition has AMI=1: e=3/2.
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    crossed = np.tile([0, 1, 0, 1], (3, 1)).astype(np.int64)
    perfect = np.tile(y, (3, 1))
    # First draws: [3,2,2,1], [1,0,0,0], [0,3,2,3], [2,2,3,2].
    # Each crossed contingency has AMI=0; each perfect one has AMI=1.
    result = vg.score_effects(y, {'R6-5': {0: crossed}, 'GT-2': {0: perfect}},
                              n_draws=4, workers=1)
    assert result['R6-5']['A'] == pytest.approx(-0.5)
    assert result['GT-2'] == pytest.approx(dict(A=1, e=1.5, lo=1, hi=1))


def test_extras_follow_ladder_and_failed_runs_count_as_run(tmp_path):
    records = {v: result(.4) for v in vg.variants.FIRST}
    records['GT-4'] = result(.51)
    assert vg.choose_first(records)['extras'] == ['GT-6', 'GT-8']
    records['GT-4'] = vg.variant_result(dict(status='failed', cause='non-finite loss',
                                             best_epoch=dict.fromkeys(vg.ASPECTS, 1)), None)
    records['GT-2'] = result(.51)
    assert vg.choose_first(records)['extras'] == ['GT-1.5', 'GT-3']
    assert records['GT-4']['A_genre'] is None
    assert not records['GT-4']['guarded']


def test_ties_keep_first_and_guard_is_inclusive(tmp_path):
    records = {v: result(.51, -.0046, -.0082) for v in vg.variants.FIRST}
    choice = vg.choose_first(records)
    assert choice['closest'] == 'GT-2'
    outcome = vg.choose_all(records, [], .4989161752444698)
    assert (outcome['winner'], outcome['outcome'], outcome['fresh_seeds']) == ('GT-2', 'III-GO', True)
    records = {v: result(.51, -.00461) for v in vg.variants.FIRST}
    assert vg.choose_all(records, [], .4989161752444698)['winner'] is None
    assert vg.choose_first(records)['guarded_pool'] is False


def test_fresh_seed_pairing_and_c_outcomes(tmp_path):
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    perfect = np.tile(y, (3, 1))
    crossed = np.tile([0, 1, 0, 1], (3, 1)).astype(np.int64)
    parts = {'GT-2': {1: perfect, 2: crossed, 3: perfect},
             'R6-5': {1: crossed, 2: perfect, 3: crossed}}
    # Seedwise effects 1.5,-1.5,1.5; the point effect is their mean, .5.
    effects = vg.score_effects(y, parts, n_draws=4, workers=1, fresh=True)['GT-2']
    assert effects['e'] == pytest.approx(.5)
    assert effects['Abar']['GT-2'] == pytest.approx(.5)
    assert effects['Abar']['R6-5'] == pytest.approx(0)
    # Paired draw effects are 1, -1/3, 1/3, 1/2; linear percentiles below.
    assert effects['lo'] == pytest.approx(-17/60)
    assert effects['hi'] == pytest.approx(.9625)
    record = {a: dict(e=.02, lo=.01, hi=.03) for a in vg.ASPECTS}
    assert vg.c_outcome(record) == 'C-HOLDS'
    record['genre']['lo'] = 0
    assert vg.c_outcome(record) == 'C-QUAL'
    record['emotion']['e'] = -.005
    assert vg.c_outcome(record) == 'C-FAILS'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def versions():
    return dict(sklearn=sklearn.__version__, numpy=np.__version__,
                scipy=scipy.__version__, threadpoolctl=threadpoolctl.__version__)


def trainer_run(variant, seed, p_sha):
    """Keys from eg_train3.write_run's successful records (no cause key)."""
    return dict(variant=variant, seed=seed, switches=vg.variants.switches(variant), status='ok',
                best_epoch=dict.fromkeys(vg.ASPECTS, 1), epochs_run=dict.fromkeys(vg.ASPECTS, 16),
                best_dev_loss=dict.fromkeys(vg.ASPECTS, 1.), p_sha256=p_sha, smoke=True,
                max_epochs=200, seconds=1., code_commit='fixture', code_dirty=False, old5000_problems=[])


@pytest.fixture(autouse=True)
def smoke_root(tmp_path, monkeypatch):
    monkeypatch.setattr(vg.paths, 'SMOKE_ROOT', tmp_path)


@pytest.fixture
def files(tmp_path):
    """Constant base AMI=0; every variant's partition equals the labels: AMI=1.

    The sample has two labels, twenty times each. Bootstrap samples contain multiple
    labels, so all sampled effects are exactly one. Only cache label *numbers*
    come from sklearn: their arbitrary permutation is version dependent.
    """
    out, rb = tmp_path / 'out', tmp_path / 'rb'
    names = [f'p{i:02d}' for i in range(40)]
    p_sha = digest('\n'.join(names).encode())
    y = np.repeat(np.arange(2), 20).astype(np.int64)
    annotations = [dict(painting=n, emotion=f'e{v:02d}', art_style=f's{v:02d}')
                   for n, v in zip(names, y)]
    label_sha = dict.fromkeys(vg.ASPECTS, digest(y.tobytes()))
    dump(rb / 'P.json', dict(names=names, sha256=p_sha, counts=dict(total=40)))
    dump(rb / 'pick.json', dict(p_sha256=p_sha, n_rows=40, label_sha256=label_sha))
    target = .4989161752444698
    dump(rb / 'reported.json', dict(p_sha256=p_sha, references_p=dict(genre_word=dict(genre=target))))
    base = np.zeros((40, 20), dtype=np.float16)
    base[:, 0] = 1
    perfect = np.eye(20, dtype=np.float16)[y]
    runs, hashes, img_hashes = {}, {}, {}
    order = list(vg.variants.FIRST) + ['GT-1.5', 'GT-3']
    for v in [vg.BASE] + order:
        folder = rb / 'train' / v if v == vg.BASE else out / 'stage_iii/train' / v / 's0'
        folder.mkdir(parents=True)
        x = base if v == vg.BASE else perfect
        img_hashes[v] = dict.fromkeys(vg.ASPECTS, digest(x.tobytes()))
        if v != vg.BASE:
            run = trainer_run(v, 0, p_sha)
            dump(folder / 'run.json', run)
            runs[v] = run
            hashes[v] = digest((folder / 'run.json').read_bytes())
        with threadpool_limits(limits=8):
            parts = np.stack([MiniBatchKMeans(n_clusters=20, random_state=s, n_init=3,
                                              batch_size=4096).fit_predict(x.astype(np.float32))
                              for s in (0, 1, 2)]).astype(np.int64)
        for a in vg.ASPECTS:
            stem = f'{a}__s0__P' if v == vg.BASE else f'{a}__P'
            np.savez(folder / (stem + '.npz'), **(dict(img=x, cap=x) if v == vg.BASE else dict(img=x)))
            if v == vg.BASE:
                dump(folder / f'{a}__s0.json', dict(recipe=v, aspect=a, seed=0, p_sha256=p_sha))
            cache = out / 'stage_iii/partitions' / v / 's0' / f'{a}__partitions.npz'
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, labels=parts)
            dump(cache.with_suffix('.json'), dict(identity=dict(variant=v, aspect=a, seed=0,
                 p_sha256=p_sha, img_sha256=digest(x.tobytes()), sklearn=sklearn.__version__,
                 numpy=np.__version__, code_sha256={'rb_ami.py': 'fixture', 'rb_pick.py': 'fixture'}),
                 versions=versions(), k=20,
                 states=[0, 1, 2], array=dict(shape=[3, 40], dtype='int64', sha256=digest(parts.tobytes()))))
    header = dict(smoke=True, n_draws=20, reference=vg.BASE, seed=0, t_g=target,
                  promising=target-.0128, ranges=vg.RANGES, versions=versions(),
                  label_counts=dict.fromkeys(vg.ASPECTS, 40), code_commit='fixture', code_dirty=False,
                  code_sha256='fixture', base_A=dict.fromkeys(vg.ASPECTS, 0.))
    records = {v: dict(status='ok', cause=None, effects={a: dict(A=1., e=1., lo=1., hi=1.)
                        for a in vg.ASPECTS}, A_genre=1., e_genre=1., guarded=True,
                        best_epoch=dict.fromkeys(vg.ASPECTS, 1)) for v in order}
    def score(phase):
        vs = list(vg.variants.FIRST) if phase == 'first' else order
        d = dict(header, stage=phase, variants=vs, results={v: records[v] for v in vs},
                 extras=['GT-1.5', 'GT-3'], identity=dict(p_sha256=p_sha, label_sha256=label_sha,
                 n_draws=20, variants=vs, img_sha256={v: img_hashes[v] for v in [vg.BASE]+vs},
                 run_sha256={v: hashes[v] for v in vs}, failed={}))
        if phase == 'first':
            d.update(closest='GT-2', method='GT', rung=2, guarded_pool=True, closest_guarded=True)
        else:
            d.update(winner='GT-2', A_genre_winner=1., e_genre_winner=1., outcome='III-GO',
                     fresh_seeds=True, kept_weights=order, pending_deletions=[])
        dump(out / f'stage_iii/score_{phase}.json', d)
        return d
    first, all_score = score('first'), score('all')
    return dict(out=out, rb=rb, p_sha=p_sha, y=y, annotations=annotations, first=first,
                all_score=all_score, base=base, perfect=perfect, label_sha=label_sha)


def verify(f, phase='first', **kwargs):
    return vg.run(f['out'], phase, rb_part1=f['rb'], annotations=f['annotations'],
                  genre_loader=lambda names: f['y'], expected_p_sha=f['p_sha'], n_rows=40,
                  width=20, smoke=True, workers=1, **kwargs)


def test_ok_run_without_cause_agrees(files):
    run = json.loads((files['out']/'stage_iii/train/GT-2/s0/run.json').read_text())
    assert 'cause' not in run
    report = verify(files)
    assert report['agree'], report['differences']
    assert report['n_draws'] == 20
    saved = files['out'] / 'stage_iii/rederive_vg2_first.json'
    assert json.loads(saved.read_text()) == report


@pytest.mark.parametrize('change,path', [('effect', 'score_all/results/GT-2/effects/genre/e'),
    ('winner', 'score_all/winner'), ('outcome', 'score_all/outcome'),
    ('draws', 'score_all/n_draws'), ('partition', 'partitions/GT-2/s0/genre')])
def test_forced_difference_is_reported(files, change, path):
    record = files['all_score']
    if change == 'effect':
        record['results']['GT-2']['effects']['genre']['e'] = .123
    elif change == 'winner':
        record['winner'] = 'GT-4'
    elif change == 'outcome':
        record['outcome'] = 'III-NONE'
    elif change == 'draws':
        record['n_draws'] = 2000
    else:
        cache = files['out']/'stage_iii/partitions/GT-2/s0/genre__partitions.npz'
        with np.load(cache) as z:
            parts = z['labels']
        parts[0, 0] = (parts[0, 0]+1) % 20
        np.savez(cache, labels=parts)
    dump(files['out']/'stage_iii/score_all.json', record)
    report = verify(files, 'all')
    assert not report['agree']
    assert path in {d['path'] for d in report['differences']}


def test_failed_run_is_rederived_from_run_json(files):
    folder = files['out']/'stage_iii/train/GA-1/s0'
    run = json.loads((folder/'run.json').read_text())
    run.update(status='failed', cause='non-finite loss', failure_kind='nonfinite_loss')
    dump(folder/'run.json', run)
    record = files['first']
    record['results']['GA-1'] = dict(status='failed', cause='non-finite loss', effects=None,
        A_genre=None, e_genre=None, guarded=False, best_epoch=dict.fromkeys(vg.ASPECTS, 1))
    record['identity']['img_sha256'].pop('GA-1')
    record['identity']['failed']['GA-1'] = 'non-finite loss'
    record['identity']['run_sha256']['GA-1'] = digest((folder/'run.json').read_bytes())
    dump(files['out']/'stage_iii/score_first.json', record)
    report = verify(files)
    assert report['agree'], report['differences']


def test_all_phase_rederives_extras_and_winner(files):
    report = verify(files, 'all')
    assert report['agree'], report['differences']


def add_fresh_files(f, fresh_caches=True):
    import torch
    out, rb2 = f['out'], f['rb'].parent/'part2'
    runs, run_hashes, images = {}, {}, {}
    for v in ('GT-2', vg.BASE):
        runs[v], run_hashes[v], images[v] = {}, {}, {}
        for s in (1, 2, 3):
            folder = out/f'stage_iii/train/{v}/s{s}'
            run = trainer_run(v, s, f['p_sha'])
            dump(folder/'run.json', run)
            run_hashes[v][str(s)] = digest((folder/'run.json').read_bytes())
            runs[v][str(s)] = dict(status='ok', cause=None, best_epoch=run['best_epoch'],
                                  run_sha256=run_hashes[v][str(s)])
            x = f['base'] if v == vg.BASE else f['perfect']
            images[v][str(s)] = dict.fromkeys(vg.ASPECTS, digest(x.tobytes()))
            for a in vg.ASPECTS:
                np.savez(folder/f'{a}__P.npz', img=x)
                np.savez(folder/f'{a}__old5000.npz', img=x)
                dump(folder/f'{a}.json', dict(aspect=a, best_epoch=1, epochs_run=16, best_dev_loss=1.,
                    history=[dict(epoch=1, step=1, train_loss=1., dev_loss=1., seconds=1.)], seconds=1.,
                    arrays={f'{a}__P.npz': {'img': digest(x.tobytes())},
                            f'{a}__old5000.npz': {'img': digest(x.tobytes())}}))
                cache0 = out/f'stage_iii/partitions/{v}/s0/{a}__partitions.npz'
                with np.load(cache0) as z:
                    parts = z['labels']
                cache = out/f'stage_iii/partitions/{v}/s{s}/{a}__partitions.npz'
                side = json.loads(cache0.with_suffix('.json').read_text())
                side['identity']['seed'] = s
                if fresh_caches:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(cache, labels=parts)
                    dump(cache.with_suffix('.json'), side)
                if v == vg.BASE:
                    # Frozen K4 states and fresh R6-5 states: same tensor bytes.
                    state = {f'{arm}.{layer}.{field}': torch.ones(1) for arm in ('img', 'cap')
                             for layer in ('trunk.0', 'output') for field in ('weight', 'bias')}
                    torch.save(state, folder/f'{a}.pt')
                    frozen = rb2/'sample/weights/K4'/f'{a}__s{s}.pt'
                    frozen.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(state, frozen)
    parity = {'pass': True, 'seeds': {str(s): {'pass': True, 'run_status': 'ok', 'run_cause': None,
              'code_commit': 'fixture', 'code_dirty': False,
              'weights_equal': dict.fromkeys(vg.ASPECTS, True)} for s in (1, 2, 3)},
              'code_commit': 'fixture', 'code_dirty': False,
              'reference': str(rb2/'sample/weights/K4')}
    dump(out/'stage_iii/fresh_parity.json', parity)
    parity_hash = digest((out/'stage_iii/fresh_parity.json').read_bytes())
    fresh = {k: f['first'][k] for k in ('smoke', 'n_draws', 'reference', 'seed', 't_g', 'promising',
             'ranges', 'versions', 'label_counts', 'code_commit', 'code_dirty', 'code_sha256')}
    fresh.update(stage='fresh', winner='GT-2', seeds=[1, 2, 3], seed0_outcome='III-GO', A_genre_seed0=1.,
        parity=dict(parity, sha256=parity_hash), c_outcome='C-HOLDS', no_outcome_cause=None,
        effects={a: dict(A={'GT-2': dict.fromkeys(['1','2','3'], 1.), vg.BASE: dict.fromkeys(['1','2','3'], 0.)},
                    Abar={'GT-2': 1., vg.BASE: 0.}, e=1., lo=1., hi=1.) for a in vg.ASPECTS},
        genre_vs_targets=dict(A_bar=1., t_g=fresh['t_g'], promising=fresh['promising'],
            diff_t_g=1-fresh['t_g'], diff_promising=1-fresh['promising'], reaches_t_g=True, reaches_promising=True),
        runs=runs, reading='Seed-0 outcome III-GO stands; fresh seeds: C-HOLDS.',
        identity=dict(p_sha256=f['p_sha'], label_sha256=f['label_sha'], n_draws=20, winner='GT-2', seeds=[1,2,3],
          score_all_sha256=digest((out/'stage_iii/score_all.json').read_bytes()), fresh_parity_sha256=parity_hash,
          failed={}, run_sha256=run_hashes))
    dump(out/'stage_iii/score_fresh.json', fresh)
    return rb2, fresh


def test_real_parity_path_and_bookkeeping_agree(files):
    rb2, fresh = add_fresh_files(files)
    fresh['parity']['code_dirty'] = True
    fresh['parity']['seeds']['1']['code_commit'] = 'other bookkeeping'
    dump(files['out']/'stage_iii/score_fresh.json', fresh)
    report = verify(files, 'fresh', rb_part2=rb2)
    assert report['agree'], report['differences']


def test_failed_parity_seed_may_omit_weights_equal(tmp_path):
    ours = {'pass': False, 'seeds': {}}
    theirs = {'pass': False, 'reference': str(tmp_path/'part2/sample/weights/K4'), 'seeds': {}}
    for s in vg.SEEDS:
        ours['seeds'][str(s)] = {'pass': False, 'run_status': 'failed',
                                'weights_equal': dict.fromkeys(vg.ASPECTS, False)}
        theirs['seeds'][str(s)] = {'pass': False, 'run_status': 'failed', 'run_cause': 'loss',
                                  'code_commit': 'trainer', 'code_dirty': True}
    assert vg.parity_differences(ours, theirs) == []
    theirs['reference'] = 'K4'
    assert vg.parity_differences(ours, theirs)[0]['path'] == 'fresh_parity/reference'


def test_no_c_outcome_record_agrees(files):
    import torch
    rb2, fresh = add_fresh_files(files, fresh_caches=False)
    weight = files['out']/'stage_iii/train/R6-5/s1/emotion.pt'
    state = torch.load(weight, weights_only=True)
    state['img.output.bias'][0] = 2
    torch.save(state, weight)
    parity = fresh['parity']
    parity.pop('sha256')
    parity['pass'] = parity['seeds']['1']['pass'] = False
    parity['seeds']['1']['weights_equal']['emotion'] = False
    parity_path = files['out']/'stage_iii/fresh_parity.json'
    dump(parity_path, parity)
    parity_hash = digest(parity_path.read_bytes())
    fresh.update(parity=dict(parity, sha256=parity_hash), c_outcome=None, effects=None,
                 genre_vs_targets=None, runs=None, no_outcome_cause='R6-5 parity failed',
                 reading='Seed-0 outcome III-GO stands; no C outcome is stated.')
    fresh['identity'].pop('run_sha256')
    fresh['identity'].update(fresh_parity_sha256=parity_hash, n_draws=37, failed={}, code_dirty=True)
    dump(files['out']/'stage_iii/score_fresh.json', fresh)
    report = verify(files, 'fresh', rb_part2=rb2)
    assert report['agree'], report['differences']
    assert report['recomputed']['effects'] is None
    assert all('/s0/' in k for k in report['input_array_hashes']['img_sha256'])


def test_false_parity_pass_cannot_license_c_outcome(files):
    import torch
    rb2, _ = add_fresh_files(files)
    path = files['out']/'stage_iii/train/R6-5/s1/emotion.pt'
    state = torch.load(path, weights_only=True)
    state['img.output.bias'][0] = 2
    torch.save(state, path)
    report = verify(files, 'fresh', rb_part2=rb2)
    assert not report['agree']
    assert report['recomputed']['c_outcome'] is None
    assert 'fresh_parity/pass' in {d['path'] for d in report['differences']}


def test_fresh_failed_run_leaves_no_c_outcome(files):
    rb2, fresh = add_fresh_files(files)
    path = files['out']/'stage_iii/train/GT-2/s2/run.json'
    run = json.loads(path.read_text())
    run.update(status='failed', cause='non-finite loss', failure_kind='nonfinite_loss')
    dump(path, run)
    fresh.pop('no_outcome_cause')
    dump(files['out']/'stage_iii/score_fresh.json', fresh)
    report = verify(files, 'fresh', rb_part2=rb2)
    assert not report['agree']
    assert report['recomputed']['c_outcome'] is None
    assert 'score_fresh/c_outcome' in {d['path'] for d in report['differences']}
    assert 'score_fresh/no_outcome_cause' in {d['path'] for d in report['differences']}


def test_workers_preserve_paired_draw_stream(tmp_path):
    y = np.array([0, 0, 1, 1, -1], dtype=np.int64)
    perfect = np.tile([0, 0, 1, 1, 2], (3, 1)).astype(np.int64)
    crossed = np.tile([0, 1, 0, 1, 2], (3, 1)).astype(np.int64)
    parts = {'GT-2': {1: perfect, 2: crossed, 3: perfect},
             'R6-5': {1: crossed, 2: perfect, 3: crossed}}
    one = vg.score_effects(y, parts, n_draws=7, workers=1, fresh=True)
    two = vg.score_effects(y, parts, n_draws=7, workers=2, fresh=True)
    assert one == two


def test_cli_rejects_smoke_writes_outside_smoke_root(tmp_path):
    with pytest.raises(SystemExit) as error:
        vg.main(['--phase', 'first', '--smoke', '--outputs', str(tmp_path),
                 '--report-outputs', '/dev/null'])
    assert error.value.code == 2
    assert not (tmp_path/'stage_iii').exists()


def test_nonfinite_export_cannot_agree(files):
    path = files['out']/'stage_iii/train/GT-2/s0/emotion__P.npz'
    x = files['perfect'].copy()
    x[0, 0] = np.nan
    np.savez(path, img=x)
    report = verify(files)
    assert not report['agree'] and not report['complete']
    assert 'nonfinite image' in report['differences'][-1]['theirs']


def test_label_hash_is_checked_against_frozen_pick(files):
    path = files['rb']/'pick.json'
    pick = json.loads(path.read_text())
    pick['label_sha256']['genre'] = 'wrong'
    dump(path, pick)
    report = verify(files)
    assert 'RB_PART1/pick/label_sha256/genre' in {d['path'] for d in report['differences']}
    assert not report['agree']


def test_unsorted_p_names_are_refused(files):
    path = files['rb']/'P.json'
    p = json.loads(path.read_text())
    p['names'].reverse()
    dump(path, p)
    report = verify(files)
    assert not report['agree'] and not report['complete']
    assert 'P row count/order/unique names' in report['differences'][-1]['theirs']


def test_missing_run_provenance_is_reported(files):
    path = files['out']/'stage_iii/train/GT-2/s0/run.json'
    run = json.loads(path.read_text())
    run.pop('p_sha256')
    run.pop('code_commit')
    dump(path, run)
    report = verify(files)
    assert not report['agree']
    assert {'inputs/GT-2/s0/p_sha256', 'inputs/GT-2/s0/code_commit'} <= {
        d['path'] for d in report['differences']}


def test_reading_preserves_seed0_outcome(tmp_path):
    assert 'not repeated on fresh seeds' in vg.fresh_reading('III-PROM', 'C-FAILS')
    assert 'does not make it GO or promising' in vg.fresh_reading('III-NONE', 'C-HOLDS')
    assert 'no C outcome' in vg.fresh_reading('III-GO', None)
