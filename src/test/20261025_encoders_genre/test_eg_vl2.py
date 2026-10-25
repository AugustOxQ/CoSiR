"""Small rule-worked cases for the independent VL2 checker."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

import eg_vl2


def test_file_fixtures_use_pytests_temporary_directory(tmp_path, tmp_path_factory):
    assert tmp_path.is_relative_to(tmp_path_factory.getbasetemp())


def test_lift_matches_hand_worked_seed_and_painting_bootstrap():
    # Perfect partitions have AMI 1; constant partitions have AMI 0 unless
    # the sampled truth is constant too, when both readings have AMI 1.
    y = np.array([0, 0, 1, 1, -1], dtype=np.int64)
    perfect = np.array([0, 0, 1, 1, 9], dtype=np.int64)
    constant = np.zeros(5, dtype=np.int64)
    student = np.stack([np.tile(p, (3, 1)) for p in
                        [perfect, perfect, perfect, constant, constant]])
    r20 = np.zeros((5, 3, 5), dtype=np.int64)
    own = np.stack([p[None, :] for p in
                    [perfect, constant, constant, constant, constant]])
    frozen = np.zeros((3, 5), dtype=np.int64)
    result = eg_vl2.score_aspect(y, student, r20, own, frozen,
                                 n_draws=3, workers=1)
    assert result['A'] == pytest.approx(dict(student=.6, own_r20=0., own_own=.2, frozen=0.))
    assert result['lift'] == pytest.approx(.4)
    assert result['student_minus_frozen'] == pytest.approx(.6)
    assert result['own_best'] == 'own_own'
    assert result['n_labelled'] == 4
    # RNG(0): paintings [3,2,2,1], [3,2,2,2], [3,2,3,0].
    # Student seeds [1,0,0,0,0], [2,4,1,4,3], [0,4,0,2,0];
    # own seeds [4,3,4,2,3], [0,1,4,2,0], [1,2,2,2,0].
    # Lifts are 1, 0, .6, giving the linear percentiles .03 and .98.
    assert result['interval'] == pytest.approx([.03, .98])
    assert eg_vl2.score_aspect(y, student, r20, own, frozen, n_draws=3, workers=2) == result


def test_bootstrap_draws_only_from_noncontiguous_labelled_positions():
    # L_a = [1, 3, 4, 5]. In L_a order this is the hand-worked four-row
    # case above; the masked rows must affect neither RNG sizes nor AMIs.
    labels = np.array([-1, 0, -1, 0, 1, 1], dtype=np.int64)
    perfect = np.array([9, 0, 8, 0, 1, 1], dtype=np.int64)
    zero = np.zeros(6, dtype=np.int64)
    student = np.stack([np.tile(p, (3, 1)) for p in
                        [perfect, perfect, perfect, zero, zero]])
    r20 = np.zeros((5, 3, 6), dtype=np.int64)
    own = np.stack([p[None, :] for p in [perfect, zero, zero, zero, zero]])
    result = eg_vl2.score_aspect(labels, student, r20, own, np.tile(zero, (3, 1)),
                                 n_draws=3, workers=1)
    assert result['n_labelled'] == 4
    assert result['lift'] == pytest.approx(.4)
    assert result['interval'] == pytest.approx([.03, .98])


def test_every_changed_lift_partition_and_outcome_is_reported():
    ours = dict(lift=.4, outcome='L-HOLDS', interval=[.03, .98],
                partitions=np.array([[0, 0, 1, 1]], dtype=np.int64))
    theirs = dict(lift=.5, outcome='L-QUAL', interval=[.03 + 1e-7, .98],
                  partitions=np.array([[0, 1, 1, 0]], dtype=np.int64))
    assert eg_vl2.differences(ours, ours) == []
    found = eg_vl2.differences(ours, theirs)
    assert {d['path'] for d in found} == {
        'lift', 'outcome', 'partitions[0][1]', 'partitions[0][3]'}
    assert next(d for d in found if d['path'] == 'lift') == dict(path='lift', ours=.4, theirs=.5)


def test_nonfinite_scorer_value_is_a_serializable_disagreement():
    found = eg_vl2.differences(dict(lift=.4), dict(lift=float('nan')))
    assert json.loads(json.dumps(found, allow_nan=False)) == [
        dict(path='lift', ours=.4, theirs='nan')]


@pytest.mark.parametrize('missing_name', ['K4', 'B1', 'F4', 'qwen3vl_emb_2b', None])
def test_missing_encoder_has_no_outcome_or_partial_aspect_scores(tmp_path, missing_name):
    encoder = eg_vl2.ENCODERS[0]
    missing = {} if missing_name is None else {missing_name: 'deferred before P inference'}
    record, partitions = eg_vl2.rederive_encoder(
        encoder, tmp_path, ['a', 'b'], {}, missing, eg_vl2.Inputs(),
        store_root=tmp_path / 'stores', workers=1, n_draws=3)
    assert record['status'] == 'missing'
    assert record['outcome'] is None
    assert 'aspects' not in record
    assert record['reason']
    assert partitions == {}


def test_raw_exports_are_repartitioned_before_caches_are_compared(tmp_path):
    e = eg_vl2.ENCODERS[0]
    names = [f'p{i:02}' for i in range(24)]
    p_sha = eg_vl2.names_sha256(names)
    x = np.eye(24, 2048, dtype=np.float16)
    prob = .1 + .8 * np.eye(24, 40, dtype=np.float32)  # nonunit sigmoid rows
    for s in range(5):
        exports = [(eg_vl2.eg_paths.p_student(tmp_path, e.student, a, s),
                    'img', x, dict(aspect=a)) for a in eg_vl2.eg_paths.ASPECTS]
        exports += [(eg_vl2.eg_paths.p_own(tmp_path, e.own, s), 'prob', prob, {})]
        for path, key, array, extra in exports:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **{key: array})
            path.with_suffix('.json').write_text(json.dumps(dict(
                p_sha256=p_sha, arm=e.student if key == 'img' else e.own,
                seed=s, rows=24, width=array.shape[1], **extra)))
    store = tmp_path / 'stores' / e.store
    store.mkdir(parents=True)
    np.save(store / 'img.npy', x[::-1])  # stores need a name join, not P positions
    (store / 'index.json').write_text(json.dumps(dict(
        backbone=e.store, dataset='stores', smoke=False, dtype='float16',
        dim=e.width, n_images=24, images=[dict(painting=n) for n in reversed(names)])))
    labels = {a: np.arange(24, dtype=np.int64) for a in eg_vl2.eg_paths.ASPECTS}
    inputs = eg_vl2.Inputs()
    record, partitions = eg_vl2.rederive_encoder(
        e, tmp_path, names, labels, {}, inputs,
        store_root=store.parent, store_count=24, workers=1, n_draws=2)
    assert record['status'] == 'scored'
    assert record['outcome'] == 'L-FAILS'
    assert len(partitions) == 26
    for aspect in eg_vl2.eg_paths.ASPECTS:
        assert record['aspects'][aspect]['A']['own_own'] == 1.
        assert record['aspects'][aspect]['student_minus_frozen'] == 0.
    # The scorer cache is deliberately created only after recomputation.
    for name, array in partitions.items():
        path = tmp_path / 'stage_ii/partitions' / (name + '.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, labels=array)
        arm, partition_name = name.split('/')
        source = (store / 'img.npy' if arm == e.frozen else
                  eg_vl2.eg_paths.p_own(tmp_path, arm, int(partition_name.split('__s')[1][0]))
                  if arm == e.own else eg_vl2.eg_paths.p_student(
                      tmp_path, arm, partition_name.split('__')[0], int(partition_name[-1])))
        meta = dict(identity=dict(
            arm=arm, name=partition_name, kind='own' if partition_name.endswith('__own') else 'r20',
            p_sha256=p_sha, source_sha256=eg_vl2.file_sha256(source),
            sklearn=eg_vl2.sklearn.__version__, numpy=np.__version__,
            rb_ami_sha256=eg_vl2.file_sha256(
                eg_vl2.eg_paths.TEST / '20261025_recipe_backbones/rb_ami.py')),
            sha256=hashlib.sha256(array.tobytes()).hexdigest(), shape=list(array.shape),
            k=20, states=[0, 1, 2], threads=8)
        path.with_suffix('.json').write_text(json.dumps(meta))
    assert eg_vl2.compare_partitions(tmp_path, partitions, p_sha, 24, inputs) == []
    first = next(iter(partitions))
    path = tmp_path / 'stage_ii/partitions' / (first + '.npz')
    changed = partitions[first].copy()
    changed[0, 0] += 1
    np.savez(path, labels=changed)
    sidecar = path.with_suffix('.json')
    meta = json.loads(sidecar.read_text())
    meta['sha256'] = hashlib.sha256(changed.tobytes()).hexdigest()
    sidecar.write_text(json.dumps(meta))
    assert eg_vl2.compare_partitions(tmp_path, partitions, p_sha, 24, eg_vl2.Inputs()) == [
        dict(path=f'partitions.{first}[0][0]', ours=int(partitions[first][0, 0]),
             theirs=int(changed[0, 0]))]
    # Even equal arrays must not make an invalid sidecar hash pass.
    meta['sha256'] = '0' * 64
    sidecar.write_text(json.dumps(meta))
    assert eg_vl2.compare_partitions(tmp_path, {first: changed}, p_sha, 24, eg_vl2.Inputs()) == [
        dict(path=f'partitions.{first}.sha256',
             ours=hashlib.sha256(changed.tobytes()).hexdigest(), theirs='0' * 64)]


def test_report_checks_independent_labels_and_preserves_an_existing_result(tmp_path, monkeypatch):
    monkeypatch.setattr(eg_vl2.eg_paths, 'SMOKE_ROOT', tmp_path.resolve())   # smoke mode outside /tmp/claude-0
    names = ['a', 'b', 'c', 'd']
    p_sha = hashlib.sha256(b'a\nb\nc\nd').hexdigest()
    annotations = [dict(painting=p, emotion=e, art_style=s) for p, e, s in
                   [('a', 'z', 's'), ('a', 'a', 't'), ('b', 'a', 't'),
                    ('c', 'z', 's'), ('d', 'a', 's')]]
    annotation_path = tmp_path / 'annotations.json'
    annotation_path.write_text(json.dumps(annotations))
    genres = tmp_path / 'genre.csv'
    genres.write_text('a.jpg,0\nb.jpg,1\nc.jpg,0\n')
    p_path = tmp_path / 'P.json'
    p_path.write_text(json.dumps(dict(names=names, sha256=p_sha)))
    labels = dict(emotion=[0, 0, 1, 0], style=[0, 1, 0, 0], genre=[0, 1, 0, -1])
    label_hashes = {a: hashlib.sha256(np.array(y, dtype=np.int64).tobytes()).hexdigest()
                    for a, y in labels.items()}
    pick = dict(p_sha256=p_sha, n_rows=4, label_sha256=label_hashes,
                label_counts=dict(emotion=4, style=4, genre=3))
    pick_path = tmp_path / 'pick.json'
    pick_path.write_text(json.dumps(pick))
    stage = tmp_path / 'stage_ii'
    (stage / 'p').mkdir(parents=True)
    missing = {e.student: 'deferred' for e in eg_vl2.ENCODERS}
    (stage / 'p/index.json').write_text(json.dumps(dict(missing=missing)))
    record = dict(pick, n_draws=20, smoke=True, versions=dict(
        sklearn=eg_vl2.sklearn.__version__, numpy=np.__version__), threads=8,
                  code_commit='independent-scorer', sources={},
                  rb_ami_sha256=eg_vl2.file_sha256(
                      eg_vl2.eg_paths.TEST / '20261025_recipe_backbones/rb_ami.py'),
                  encoders={e.student: dict(
        encoder=e.store, student=e.student, own=e.own, frozen=e.frozen,
        status='missing', outcome=None, reason='deferred') for e in eg_vl2.ENCODERS})
    (stage / 'lift.json').write_text(json.dumps(record))
    options = dict(workers=1, smoke=True, n_rows=4, p_sha256=p_sha,
                   p_path=p_path, pick_path=pick_path, annotations_path=annotation_path,
                   genre_paths=[genres])
    report = eg_vl2.run(tmp_path, **options)
    assert report['agree'] and report['differences'] == []
    assert report['config']['n_draws'] == report['recomputed']['n_draws'] == 20
    assert report['recomputed']['label_sha256'] == label_hashes
    assert report['code_commit']
    assert str(annotation_path) in report['input_hashes']
    scored_without_aspects = tmp_path / 'scored-without-aspects'
    (scored_without_aspects / 'stage_ii/p').mkdir(parents=True)
    (scored_without_aspects / 'stage_ii/p/index.json').write_text(json.dumps(dict(missing=missing)))
    invalid_record = json.loads(json.dumps(record))
    invalid_record['encoders']['K4']['status'] = 'scored'
    (scored_without_aspects / 'stage_ii/lift.json').write_text(json.dumps(invalid_record))
    with pytest.raises(AssertionError, match='lift encoder schema'):
        eg_vl2.run(scored_without_aspects, **options)
    output = stage / 'rederive_vl2.json'
    original = output.read_bytes()
    assert eg_vl2.run(tmp_path, **options) == report
    assert output.read_bytes() == original
    changed = tmp_path / 'changed'
    (changed / 'stage_ii/p').mkdir(parents=True)
    (changed / 'stage_ii/p/index.json').write_text(json.dumps(dict(missing=missing)))
    record['encoders']['K4']['outcome'] = 'L-HOLDS'
    (changed / 'stage_ii/lift.json').write_text(json.dumps(record))
    disagreement = eg_vl2.run(changed, **options)
    assert not disagreement['agree']
    assert disagreement['differences'] == [dict(
        path='encoders.K4.outcome', ours=None, theirs='L-HOLDS')]
    different_threads = tmp_path / 'different-threads'
    (different_threads / 'stage_ii/p').mkdir(parents=True)
    (different_threads / 'stage_ii/p/index.json').write_text(json.dumps(dict(missing=missing)))
    record['encoders']['K4']['outcome'] = None
    (different_threads / 'stage_ii/lift.json').write_text(json.dumps(dict(record, threads=1)))
    assert eg_vl2.run(different_threads, **options)['differences'] == [
        dict(path='threads', ours=8, theirs=1)]
    different_draws = tmp_path / 'different-draws'
    (different_draws / 'stage_ii/p').mkdir(parents=True)
    (different_draws / 'stage_ii/p/index.json').write_text(json.dumps(dict(missing=missing)))
    (different_draws / 'stage_ii/lift.json').write_text(json.dumps(dict(record, n_draws=4)))
    draw_report = eg_vl2.run(different_draws, **options)
    assert draw_report['config']['n_draws'] == 20  # never borrowed from lift.json
    assert draw_report['differences'] == [dict(path='n_draws', ours=20, theirs=4)]
    normal = tmp_path / 'normal'
    (normal / 'stage_ii/p').mkdir(parents=True)
    (normal / 'stage_ii/p/index.json').write_text(json.dumps(dict(missing=missing)))
    (normal / 'stage_ii/lift.json').write_text(json.dumps(dict(record, n_draws=2000, smoke=False)))
    normal_report = eg_vl2.run(normal, **dict(options, smoke=False))
    assert normal_report['agree'] and normal_report['config']['n_draws'] == 2000
    # A changed input makes the previous verdict stale; never overwrite it.
    (stage / 'lift.json').write_text(json.dumps(dict(record, n_draws=4)))
    with pytest.raises(AssertionError, match='stale'):
        eg_vl2.run(tmp_path, **options)
    assert output.read_bytes() == original


@pytest.mark.parametrize('lift,lower,expected', [
    (0., -.1, 'L-FAILS'), (-.1, -.2, 'L-FAILS'),
    (.1, 0., 'L-QUAL'), (.1, -.1, 'L-QUAL'), (.1, .01, 'L-HOLDS')])
def test_emotion_outcome_uses_unrounded_zero_boundaries(lift, lower, expected):
    assert eg_vl2.outcome(dict(lift=lift, interval=[lower, .2])) == expected


@pytest.mark.parametrize('lift,lower,upper', [
    (np.nan, 0., .2), (np.inf, 0., .2), (-np.inf, 0., .2),
    (.1, np.nan, .2), (.1, -np.inf, .2), (.1, 0., np.nan), (.1, 0., np.inf)])
def test_nonfinite_lift_or_bound_has_no_outcome(lift, lower, upper):
    with pytest.raises(AssertionError, match='finite'):
        eg_vl2.outcome(dict(lift=lift, interval=[lower, upper]))


def test_smoke_rejects_an_output_location_outside_the_smoke_root():
    with pytest.raises(AssertionError, match='--smoke writes only'):
        eg_vl2.run('/not-a-smoke-output', smoke=True)


def test_bootstrap_selects_the_best_own_reading_again_in_each_draw():
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    zero = np.zeros(4, dtype=np.int64)
    student = np.stack([np.tile(p, (3, 1)) for p in [labels, labels, labels, zero, zero]])
    r20 = np.stack([np.tile(p, (3, 1)) for p in [zero, labels, labels, zero, zero]])
    own = np.stack([p[None, :] for p in [labels, zero, zero, zero, labels]])
    result = eg_vl2.score_aspect(labels, student, r20, own, np.tile(zero, (3, 1)),
                                 n_draws=3, workers=1)
    assert result['own_best'] == 'own_r20'  # .4 vs .4: stable first reading
    assert result['lift'] == pytest.approx(.2)
    # First draw selects own_own (.4 > .2); third selects r20 (.8 > .2).
    # Resampled lifts .6, 0, 0 have linear percentiles 0 and .57.
    assert result['interval'] == pytest.approx([0., .57])
