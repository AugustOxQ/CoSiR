"""PercepT arm cells and unchanged ablation rows from stored-result fixtures."""
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pc_ablation as ablation

ASPECTS = ('emotion', 'style', 'genre')


@pytest.fixture
def results():
    def baseline(value):
        return dict(ami={side: {a: dict(mean=value + offset + shift)
                               for a, offset in zip(ASPECTS, (0., .1, .2))}
                         for side, shift in (('img', 0.), ('cap', .02))},
                    target_ami=dict(mean=value + .03, sd=.0044), complete=True)

    aggregates = {}
    for a, offset in zip(ASPECTS, (0., .1, .2)):
        aggregates[a] = {
            'default': {g: baseline(v) for g, v in
                        (('R0', .3004), ('R1', .4004), ('R0o', .6004),
                         ('Fres1', .5004), ('Fres05', .5004), ('Fside025', .5004))},
            'prompted': {g: baseline(v) for g, v in
                         (('R0', .7004), ('Fres1', .8004), ('Fres05', .8004),
                          ('Fside025', .8004))},
        }
        for by in aggregates[a].values():
            for r in by.values():
                r['target_ami']['mean'] += offset
    references = {}
    for key, value in (('default', .1004), ('emotion_focus', .2004),
                       ('style_word', .2004), ('genre_word', .2004)):
        for side, by in baseline(value)['ami'].items():
            references[f'{key}/{side}'] = dict(ami=by)
    row1 = {}
    for key, value in (('clip_text', .0104), ('roberta', .0404)):
        by = baseline(value)['ami']
        row1[key] = dict(**by, both_mean=dict(zip(ASPECTS, (value + .01, value + .11, value + .21))))
    rows = {}
    for row, value in (('row6', .2504), ('row10', .5504), ('row13', .8504)):
        rows[row] = {a: baseline(value) for a in ASPECTS}
        for a, offset in zip(ASPECTS, (0., .1, .2)):
            rows[row][a]['target_ami']['mean'] += offset
    arms = {}
    for arm, means in (('P1', (.10049, .20049, .30049)),
                       ('P2', (.10151, .19851, .30051)),
                       ('P3', (.30049, .40049, .50049)),
                       ('P4', None), ('P5', (.30151, .39851, .50051))):
        arms[arm] = dict(established=arm not in ('P4', 'P5'),
                         reasons=['fixture gate failed'] if arm in ('P4', 'P5') else [],
                         img=None if means is None else
                         {a: dict(mean=m, sd=s) for a, m, s in
                          zip(ASPECTS, means, (.0014, .0026, .0034))})
    return dict(aggregates=aggregates, references=references), dict(row1=row1, arms=arms), rows


def test_arm_tables_and_image_changes_preserve_existing_output(results, capsys):
    ablation.main(*results)
    output = capsys.readouterr().out
    both, sides, image, changes, image_changes = output.strip().split('\n\n')
    arm_rows = {'4a', '4b', '5a', '5b', '5c'}

    def arm_lines(table):
        return [line for line in table.splitlines() if line.split(' | ')[0][2:] in arm_rows]

    assert arm_lines(both) == [
        '| 4a | — | — | — |',
        '| 4b | — | — | — |',
        '| 5a | — | — | — |',
        '| 5b | — | — | — |',
        '| 5c | — | — | — |',
    ]
    assert [line.split(' | ')[0][2:] for line in both.splitlines()[2:]] == [
        '1a', '1b', '2', '3', '4a', '4b', '5a', '5b', '5c',
        '6', '7', '8', '9', '10', '11', '12', '13',
    ]
    assert arm_lines(sides) == [
        '| 4a | 0.100 | — | 0.200 | — | 0.300 | — |',
        '| 4b | 0.102 | — | 0.199 | — | 0.301 | — |',
        '| 5a | 0.300 | — | 0.400 | — | 0.500 | — |',
        '| 5b | — | — | — | — | — | — |',
        '| 5c | 0.302* | — | 0.399* | — | 0.501* | — |',
    ]
    assert sides.endswith('\n* not established (reasons in pc_tables.md)')
    assert output.count('* not established (reasons in pc_tables.md)') == 1
    assert image == '''Image side, PercepT arms
| # | Emotion | Style | Genre |
|---|---:|---:|---:|
| 4a | 0.100 ± 0.001 | 0.200 ± 0.003 | 0.300 ± 0.003 |
| 4b | 0.102 ± 0.001 | 0.199 ± 0.003 | 0.301 ± 0.003 |
| 5a | 0.300 ± 0.001 | 0.400 ± 0.003 | 0.500 ± 0.003 |
| 5b | — | — | — |
| 5c | 0.302 ± 0.001* | 0.399 ± 0.003* | 0.501 ± 0.003* |'''
    assert image_changes == '''Image-side changes
| Change | Rows | Emotion | Style | Genre |
|---|---|---:|---:|---:|
| Stage 2 reader, CLIP | 4b vs 4a | +0.001 | -0.002 | +0.000 |
| Stage 2 reader, Qwen topics | 5b vs 5a | — | — | — |
| E: RoBERTa vs Qwen emotion | 5c vs 5a | +0.001 | -0.002 | +0.000 |
| I and T: Qwen vs CLIP | 5c vs 4b | +0.200 | +0.200 | +0.200 |'''

    def existing_lines(table):
        return '\n'.join(line for line in table.splitlines()
                         if line not in arm_lines(table) and not line.startswith('*'))

    assert existing_lines(both) == '''| # | Emotion | Style | Genre |
|---|---:|---:|---:|
| 1a | 0.020 | 0.120 | 0.220 |
| 1b | 0.050 | 0.150 | 0.250 |
| 2 | 0.110 | 0.210 | 0.310 |
| 3 | 0.210 | 0.310 | 0.410 |
| 6 | 0.280 ± 0.004 | 0.380 ± 0.004 | 0.480 ± 0.004 |
| 7 | 0.330 ± 0.004 | 0.430 ± 0.004 | 0.530 ± 0.004 |
| 8 | 0.430 ± 0.004 | 0.530 ± 0.004 | 0.630 ± 0.004 |
| 9 | 0.530 ± 0.004 | 0.630 ± 0.004 | 0.730 ± 0.004 |
| 10 | 0.580 ± 0.004 | 0.680 ± 0.004 | 0.830 ± 0.004 |
| 11 | 0.730 ± 0.004 | 0.830 ± 0.004 | 0.930 ± 0.004 |
| 12 | 0.830 ± 0.004 | 0.930 ± 0.004 | 1.030 ± 0.004 |
| 13 | 0.880 ± 0.004 | 0.980 ± 0.004 | 1.080 ± 0.004 |'''
    assert existing_lines(sides) == '''| # | Emotion img | Emotion cap | Style img | Style cap | Genre img | Genre cap |
|---|---:|---:|---:|---:|---:|---:|
| 1a | 0.010 | 0.030 | 0.110 | 0.130 | 0.210 | 0.230 |
| 1b | 0.040 | 0.060 | 0.140 | 0.160 | 0.240 | 0.260 |
| 2 | 0.100 | 0.120 | 0.200 | 0.220 | 0.300 | 0.320 |
| 3 | 0.200 | 0.220 | 0.300 | 0.320 | 0.400 | 0.420 |
| 6 | 0.250 | 0.270 | 0.350 | 0.370 | 0.450 | 0.470 |
| 7 | 0.300 | 0.320 | 0.400 | 0.420 | 0.500 | 0.520 |
| 8 | 0.400 | 0.420 | 0.500 | 0.520 | 0.600 | 0.620 |
| 9 | 0.500 | 0.520 | 0.600 | 0.620 | 0.700 | 0.720 |
| 10 | 0.550 | 0.570 | 0.650 | 0.670 | 0.800 | 0.820 |
| 11 | 0.700 | 0.720 | 0.800 | 0.820 | 0.900 | 0.920 |
| 12 | 0.800 | 0.820 | 0.900 | 0.920 | 1.000 | 1.020 |
| 13 | 0.850 | 0.870 | 0.950 | 0.970 | 1.050 | 1.070 |'''
    assert changes == '''| Change | Rows | Emotion | Style | Genre |
|---|---|---:|---:|---:|
| Instruction in the graph (default graph to plain aspect graph) | 7 vs 6 | +0.050 | +0.050 | +0.050 |
| Network on the default graph vs default embedding | 6 vs 2 | +0.170 | +0.170 | +0.170 |
| Other side alone instead of the plain aspect graph | 10 vs 7 | +0.250 | +0.250 | +0.300 |
| Prompted input instead of default, other side alone | 13 vs 10 | +0.300 | +0.300 | +0.250 |
| Encoder: Qwen default vs CLIP image + CLIP text | 2 vs 1a | +0.090 | +0.090 | +0.090 |
| Encoder: Qwen default vs CLIP image + RoBERTa | 2 vs 1b | +0.060 | +0.060 | +0.060 |'''


def test_import_does_not_read_result_files(monkeypatch):
    def no_read(*args, **kwargs):
        pytest.fail('import must not read result files')

    monkeypatch.setattr(Path, 'read_text', no_read)
    spec = importlib.util.spec_from_file_location('ablation_without_outputs', ablation.__file__)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


@pytest.mark.parametrize('key', ('P1', 'P2', 'P3', 'P4', 'P5'))
def test_reading_requires_all_five_arms(results, key):
    fu, result, rows = results
    result['arms'] = {k: r for k, r in result['arms'].items() if k != key}
    with pytest.raises(AssertionError):
        ablation.main(fu, result, rows)


@pytest.mark.parametrize('key', ('P1', 'P2', 'P3', 'P5'))
@pytest.mark.parametrize('aspect', ASPECTS)
def test_reading_requires_each_aspect_in_every_image_dict(results, key, aspect):
    fu, result, rows = results
    result['arms'][key]['img'] = {a: r for a, r in result['arms'][key]['img'].items() if a != aspect}
    with pytest.raises(AssertionError):
        ablation.main(fu, result, rows)


def test_no_footnote_for_an_incomplete_arm_without_numbers(results, capsys):
    results[1]['arms']['P5']['established'] = True
    ablation.main(*results)
    output = capsys.readouterr().out
    assert '*' not in output


def test_qwen_stage_two_reader_change_uses_image_means(results, capsys):
    results[1]['arms']['P4'] = dict(established=True, reasons=[], img={
        'emotion': dict(mean=.20049, sd=.0014),
        'style': dict(mean=.50049, sd=.0026),
        'genre': dict(mean=.40049, sd=.0034),
    })
    ablation.main(*results)
    assert '| Stage 2 reader, Qwen topics | 5b vs 5a | -0.100 | +0.100 | -0.100 |' in capsys.readouterr().out


@pytest.mark.parametrize('key,expected', [
    ('P1', '| Stage 2 reader, CLIP | 4b vs 4a | — | — | — |'),
    ('P2', '| I and T: Qwen vs CLIP | 5c vs 4b | — | — | — |'),
    ('P3', '| E: RoBERTa vs Qwen emotion | 5c vs 5a | — | — | — |'),
    ('P5', '| E: RoBERTa vs Qwen emotion | 5c vs 5a | — | — | — |'),
])
def test_image_change_is_empty_when_either_arm_is_incomplete(results, capsys, key, expected):
    results[1]['arms'][key]['img'] = None
    results[1]['arms'][key]['established'] = False
    ablation.main(*results)
    assert expected in capsys.readouterr().out
