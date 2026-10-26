"""Independent VC2 behavior, using small CPU examples only."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import rg_held
import rg_vc2 as vc
from sklearn.metrics import adjusted_mutual_info_score


@pytest.mark.parametrize('d,lo,dr,lr,want,pc,exceeds', [
    (-.02, -.0127, 0., -1., 'G-GO', False, False),
    (.02, .001, .04, .01, 'G-GO', True, True),
    (-.0128, -.0128, .03, .001, 'G-MM', True, False),
    (-.0128, -.02, .03, 0., 'G-MM', False, False),
    (-.02, -.03, .0264, .001, 'G-PART', True, False),
    (-.02, -.03, .0263, .001, 'G-NO', False, False),
    (-.02, -.03, .04, 0., 'G-NO', False, False),
])
def test_rule_outcomes_at_strict_and_inclusive_boundaries(d, lo, dr, lr, want, pc, exceeds):
    result = vc.decide(d, lo, dr, lr, .0264)
    assert result['outcome_id'] == want
    assert result['pc'] is pc
    assert result['matches_or_exceeds'] is exceeds
    # rg_held's form: the PC value for G-MM and G-PART, None otherwise
    assert result['partial_sentence'] is (pc if want in ('G-MM', 'G-PART') else None)


def test_separate_row_then_seed_draw_streams():
    it, st = vc.draw_plan(4, {'G': range(5)}, 2)
    ir, sr = vc.draw_plan(4, {'G': range(5), 'R': range(5)}, 2)
    assert it.tolist() == [[3, 2, 2, 1], [3, 2, 3, 2]]
    assert st[0]['G'].tolist() == [1, 0, 0, 0, 0]
    assert ir[0].tolist() == [3, 2, 2, 1]
    assert sr[0]['R'].tolist() == [4, 3, 4, 2, 3]
    assert not np.array_equal(it[1], ir[1])


def synthetic_parts():
    labels = np.array([0, 0, 1, 1, 2, 2, -1, -1] * 4)
    good = np.tile(np.array([0, 0, 1, 1, 2, 2, 7, 8]), 4)
    bad = np.arange(len(labels)) % 3
    g = {s: np.stack([good if (s + k) % 2 else bad for k in range(3)]) for s in range(5)}
    r = {s: np.stack([good if (s + k) % 3 else bad for k in range(3)]) for s in range(5)}
    return labels, g, np.stack([good, bad, good]), r


def test_statistics_mask_inside_each_all_row_draw_and_resample_seeds_independently():
    labels, g, t, r = synthetic_parts()
    result = vc.calculate(labels, g, t, r, draws=6, workers=1)
    # Literal spec evaluator: individual AMIs, no production draw/statistic helper.
    expected = {}
    for stat in ('d_T', 'd_R'):
        rng = np.random.default_rng(0)
        values = []
        for _ in range(6):
            rows = rng.integers(0, len(labels), size=len(labels))
            gs = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
            rs = rng.choice([0, 1, 2, 3, 4], size=5, replace=True) if stat == 'd_R' else None
            rows = rows[labels[rows] >= 0]
            left = [adjusted_mutual_info_score(labels[rows], p[rows]) for s in gs for p in g[s]]
            right = ([adjusted_mutual_info_score(labels[rows], p[rows]) for s in rs for p in r[s]]
                     if rs is not None else [adjusted_mutual_info_score(labels[rows], p[rows]) for p in t])
            values.append(np.mean(left) - np.mean(right))
        expected[stat] = values
        np.testing.assert_allclose(result[stat]['bootstrap_d'], values, atol=1e-12)
        np.testing.assert_allclose(result[stat]['interval'], np.percentile(values, [2.5, 97.5]), atol=1e-12)
    assert not np.allclose(expected['d_T'], expected['d_R'])
    assert result['labelled_rows'] == 24
    assert result['G_half'] == pytest.approx((result['d_T']['A_T'] - result['d_R']['A_R']) / 2)
    parallel = vc.calculate(labels, g, t, r, draws=6, workers=2)
    assert parallel == result


def test_point_estimates_for_perfect_clusters_and_constant_recipe():
    labels = np.array([0, 0, 1, 1] * 8)
    perfect = np.stack([labels] * 3)
    constant = np.zeros_like(perfect)
    result = vc.calculate(labels, {s: perfect for s in range(5)}, perfect,
                          {s: constant for s in range(5)}, draws=2, workers=1)
    assert result['d_T']['A_G'] == result['d_T']['A_T'] == 1.
    assert result['d_R']['A_R'] == result['d_T']['d'] == 0.
    assert result['d_R']['d'] == 1.
    assert result['G_half'] == .5
    assert result['outcome_id'] == 'G-GO'


def toy_files(root):
    n, width = 48, 24
    names = [f'p{i:03}' for i in range(n)]
    sha = vc.sha256_names(names)
    held = root / 'held.json'
    held.write_text(json.dumps(dict(names=names, sha256=sha)))
    annotations = root / 'captions.json'
    annotations.write_text(json.dumps([dict(painting=name) for name in reversed(names)] * 2))
    csv = root / 'genre.csv'
    csv.write_text(''.join(f'art/{name}.jpg,{i % 3}\n' for i, name in enumerate(names[:-4])))
    rng = np.random.default_rng(4)
    def unit():
        x = rng.normal(size=(n, width)).astype(np.float32)
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)
    t = unit()
    teacher = root / 'teacher'
    teacher.mkdir()
    np.save(teacher / 'img.npy', t)
    np.savez(teacher / 'ids.npz', painting_names=np.array(names))
    (teacher / 'meta.json').write_text(json.dumps(dict(
        instruction='Describe the genre of this artwork in one word.',
        model_id='Qwen/Qwen3-VL-Embedding-2B', model_revision='a' * 40,
        held_sha256=sha, shape=[n, width], dtype='float16', img_sha256=vc.array_sha(t))))
    rb = root / 'rb'
    outputs = root / 'outputs'
    for arm, base, cached in [('G', outputs / 'stage_c/full/held/G', outputs / 'stage_c/partitions/G'),
                              ('K4', rb / 'full/held/K4', rb / 'partitions/full/K4')]:
        base.mkdir(parents=True)
        cached.mkdir(parents=True)
        for seed in range(5):
            x = unit()
            path = base / f'genre__s{seed}.npz'
            np.savez(path, img=x)
            meta = dict(arrays=dict(img=dict(shape=[n, width], dtype='float16', sha256=vc.array_sha(x))),
                        list_sha256=sha, rows=n, width=width, dtype='float16', arm=arm, aspect='genre',
                        scale='full', kind='held', seed=seed)
            path.with_suffix('.json').write_text(json.dumps(meta))
            np.savez(cached / f'genre__s{seed}__r20.npz', labels=vc.r20_partition(x))
    tp = rb / 'partitions/full/T'
    tp.mkdir(parents=True)
    np.savez(tp / 'genre__s0__r20.npz', labels=vc.r20_partition(t))
    inputs = vc.Inputs(held=held, annotations=annotations, genre_csvs=(csv,), teacher=teacher,
                       rb=rb, held_sha=sha, n=n, width=width, labelled=n - 4)
    return root, outputs, inputs


def test_rederive_from_saved_outputs_checks_every_partition_and_input_identity(tmp_path):
    root, outputs, inputs = toy_files(tmp_path)
    result = vc.derive(outputs, 'G', inputs, draws=3, workers=1)
    assert result['labelled_rows'] == 44
    assert result['partitions']['checked'] == 11
    assert result['partitions']['differences'] == []
    assert result['held_sha256'] == inputs.held_sha
    cached = outputs / 'stage_c/partitions/G/genre__s0__r20.npz'
    np.savez(cached, labels=np.zeros((3, inputs.n), dtype=np.int64))
    changed = vc.derive(outputs, 'G', inputs, draws=3, workers=1)
    assert len(changed['partitions']['differences']) == 1
    assert str(cached) in changed['partitions']['differences'][0]
    side = outputs / 'stage_c/full/held/G/genre__s0.json'
    meta = json.loads(side.read_text())
    meta['list_sha256'] = 'wrong order'
    side.write_text(json.dumps(meta))
    with pytest.raises(AssertionError, match='list_sha256'):
        vc.derive(outputs, 'G', inputs, draws=3, workers=1)


def held_shaped_verdict(mine):
    """The record as rg_held.run_score writes it (outcome by rg_held.outcome, per_seed keyed as its
    _bootstrap keys it), after the JSON round trip of the saved file."""
    d_t, d_r = mine['d_T'], mine['d_R']
    verdict = copy.deepcopy(mine)
    verdict.update(rg_held.outcome(d_t['d'], d_t['interval'][0], d_r['d'], d_r['interval'][0], mine['G_half']))
    g = mine['d_T']['per_seed']['G']
    verdict['d_T']['per_seed'] = {'G': {int(s): v for s, v in g.items()}, 'T': {0: d_t['A_T']}}
    verdict['d_R']['per_seed'] = {'G': {int(s): v for s, v in g.items()},
                                  'R': {int(s): v for s, v in d_r['per_seed']['R'].items()}}
    verdict.pop('partitions')      # VC2's own bookkeeping, not in rg_held's record
    return json.loads(json.dumps(verdict))


@pytest.mark.parametrize('d_t,lo_t,d_r,lo_r,want', [
    (.02, .001, .04, .01, 'G-GO'),
    (-.02, -.03, .04, 0., 'G-NO'),
    (-.0128, -.02, .06, .001, 'G-MM'),
    (-.02, -.03, .06, .001, 'G-PART'),
])
def test_comparison_agrees_with_the_record_rg_held_writes(d_t, lo_t, d_r, lo_r, want):
    labels, g, t, r = synthetic_parts()
    mine = vc.calculate(labels, g, t, r, draws=4, workers=1)
    mine.update(arm='G', held_sha256='hash', rule_commit='rule', smoke=True,
                G_half_frozen=mine['G_half'], partitions=dict(checked=11, differences=[]))
    # Force the four outcomes: the statistics only need to be self-consistent between the two records.
    mine['d_T'].update(d=d_t, interval=[lo_t, mine['d_T']['interval'][1]])
    mine['d_R'].update(d=d_r, interval=[lo_r, mine['d_R']['interval'][1]])
    mine['G_half'] = mine['G_half_frozen'] = .05
    mine.update(vc.decide(d_t, lo_t, d_r, lo_r, .05))
    assert mine['outcome_id'] == want
    verdict = held_shaped_verdict(mine)
    assert verdict['partial_sentence'] is (mine['pc'] if want in ('G-MM', 'G-PART') else None)
    comparison = vc.compare_verdict(mine, verdict)
    assert comparison['differences'] == [] and comparison['agree']


def test_comparison_reports_a_changed_per_seed_value():
    labels, g, t, r = synthetic_parts()
    mine = vc.calculate(labels, g, t, r, draws=4, workers=1)
    mine.update(arm='G', held_sha256='hash', rule_commit='rule', smoke=True,
                G_half_frozen=mine['G_half'], partitions=dict(checked=11, differences=[]))
    verdict = held_shaped_verdict(mine)
    verdict['d_R']['per_seed']['R']['3'] += 1e-3
    assert any('d_R.per_seed.R.3' in diff for diff in vc.compare_verdict(mine, verdict)['differences'])


def test_comparison_reports_numbers_draws_outcome_and_partition_disagreements():
    labels, g, t, r = synthetic_parts()
    mine = vc.calculate(labels, g, t, r, draws=4, workers=1)
    mine.update(arm='G', held_sha256='hash', rule_commit='rule', smoke=True,
                G_half_frozen=mine['G_half'], partitions=dict(checked=11, differences=[]))
    assert vc.compare_verdict(mine, copy.deepcopy(mine))['agree']
    theirs = copy.deepcopy(mine)
    theirs['d_T']['A_T'] += 2e-6
    theirs['d_R']['interval'][1] += 3e-6
    theirs['d_R']['bootstrap_d'][2] += .1
    theirs['outcome_id'] = 'wrong'
    theirs['pc_parts']['lower_gt_0'] = not mine['pc_parts']['lower_gt_0']
    theirs['held_sha256'] = 'wrong'
    mine['partitions']['differences'] = ['some/path: unequal partitions']
    comparison = vc.compare_verdict(mine, theirs)
    assert not comparison['agree']
    for field in ('A_T', 'interval', 'bootstrap_d', 'outcome_id', 'lower_gt_0', 'held_sha256', 'some/path'):
        assert any(field in diff for diff in comparison['differences'])


def test_labels_join_by_caption_painting_and_mask_missing_or_conflicting_genres(tmp_path):
    root = tmp_path
    captions = root / 'captions.json'
    captions.write_text(json.dumps([dict(painting=n) for n in ['b', 'a', 'c', 'a']]))
    first, second = root / 'train.csv', root / 'val.csv'
    first.write_text('art/a.jpg,2\nart/b.jpg,1\n')
    second.write_text('art/b.jpg,3\n')
    assert vc.genre_labels(['a', 'b', 'c'], captions, (first, second)).tolist() == [2, -1, -1]
    with pytest.raises(AssertionError, match='without captions'):
        vc.genre_labels(['d'], captions, (first, second))


def test_runner_persists_disagreement_and_invalidates_skip_when_input_changes(tmp_path, monkeypatch):
    # smoke_ok only admits /tmp/claude-0/; pytest's tmp_path lives elsewhere and is cleaned by pytest.
    monkeypatch.setattr(vc.rg_paths, 'smoke_ok', lambda path: True)
    _, outputs, inputs = toy_files(tmp_path)
    mine = vc.derive(outputs, 'G', inputs, draws=20, workers=1)
    mine['smoke'] = True
    mine['outcome_id'] = 'deliberately wrong'
    verdict = outputs / 'stage_c/verdict.json'
    verdict.write_text(json.dumps(mine))
    summary = vc.run(outputs, 'G', smoke=True, workers=1, inputs=inputs)
    assert not summary['agree']
    assert any('outcome_id' in diff for diff in summary['differences'])
    saved = json.loads(Path(summary['file']).read_text())
    assert not saved['agree'] and saved['code_commit']
    assert saved['result']['inputs_sha256'][str(verdict)] == vc.file_sha(verdict)
    assert vc.run(outputs, 'G', smoke=True, workers=1, inputs=inputs)['skipped']
    cache = outputs / 'stage_c/partitions/G/genre__s0__r20.npz'
    np.savez(cache, labels=np.zeros((3, inputs.n), dtype=np.int64))
    updated = vc.run(outputs, 'G', smoke=True, workers=1, inputs=inputs)
    assert 'skipped' not in updated
    assert any(str(cache) in diff for diff in updated['differences'])


def test_vc2_dirty_flag_covers_only_round_python_including_untracked(tmp_path, monkeypatch):
    import subprocess
    here = tmp_path / 'src/test/round'
    here.mkdir(parents=True)
    (here / 'rg_paths.py').write_text('x = 1\n')
    git = ['git', '-C', str(tmp_path), '-c', 'user.name=t', '-c', 'user.email=t@t']
    subprocess.run(git + ['init', '-q'], check=True)
    subprocess.run(git + ['add', '.'], check=True)
    subprocess.run(git + ['commit', '-qm', 'baseline'], check=True)
    monkeypatch.setattr(vc.rg_paths, 'ROUND', here)
    assert vc._code_record()['dirty'] is False
    (here / 'notes.md').write_text('run notes\n')
    (tmp_path / 'other_round.py').write_text('unrelated = True\n')
    assert vc._code_record()['dirty'] is False
    (here / 'untracked.py').write_text('new_code = True\n')
    assert vc._code_record()['dirty'] is True
    subprocess.run(git + ['add', '.'], check=True)
    subprocess.run(git + ['commit', '-qm', 'new code'], check=True)
    assert vc._code_record()['dirty'] is False
    (here / 'rg_paths.py').write_text('x = 2\n')
    assert vc._code_record()['dirty'] is True
