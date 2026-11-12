"""Validation decisions: public verdict functions and record/CLI boundaries."""
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r29_decide as D


@pytest.mark.parametrize('group,edge,other,levels', [
    ('image', 0.375, 0.234375, (0, 1, 1)),
    ('text', 0.21875, 0.5, (0, 1, 1)),
    ('text', 0.25, 0.375, (1, 2, 2)),
    ('image', 0.4375, 0.5, (2, 2, 3)),
    ('text', 0.28125, 0.5, (2, 2, 3)),
])
def test_sc2_edges_are_exact(group, edge, other, levels):
    # Binary-exact worked bounds: image m=.40625, r=.03125, best=.4375;
    # text m=.25, r=.03125, best=.28125. No epsilon at any decision edge.
    bounds = dict(image=dict(m=.40625, r=.03125, best=.4375),
                  text=dict(m=.25, r=.03125, best=.28125))
    for value, level in zip((math.nextafter(edge, -math.inf), edge,
                             math.nextafter(edge, math.inf)), levels):
        image, text = (value, other) if group == 'image' else (other, value)
        assert D.sc2_level(image, text, bounds) == level


def test_sc2_level_two_has_no_image_mean_requirement():
    bounds = dict(image=dict(m=.40625, r=.03125, best=.4375),
                  text=dict(m=.25, r=.03125, best=.28125))
    assert D.sc2_level(.375, .25, bounds) == 2


def put(root, name, doc):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def text_record(arm, values, seeds=None):
    seeds = seeds if seeds is not None else (["x"] if arm.startswith('p1_') else list(range(len(values))))
    cells = {a: {d: dict(per_seed=dict(r0=values)) for d in ('t2i', 'i2t', 't2t')} for a in D.ASPECTS}
    return dict(arm=arm, seeds=seeds, val=dict(cells=cells, group_r0=dict(per_seed=values)))


@pytest.mark.parametrize('second,pick', [(.25, 'siglip1'),
    (math.nextafter(.25, -math.inf), 'siglip1'),
    (math.nextafter(.25, math.inf), 'siglip1_prefix')])
def test_sc1_seed_zero_exact_tie_and_d20_plain_tie(tmp_path, second, pick):
    for arm, value in [('siglip1', .25), ('siglip1_prefix', second),
                       ('p1_siglip1', .125), ('p1_siglip1_prefix', .125)]:
        # Later seeds deliberately reverse the winner: SC1 must only use seed 0.
        put(tmp_path, f'text_{arm}.json', text_record(arm, [value] if arm.startswith('p1_') else [value, 1-value]))
    got = D.sc1(tmp_path)
    assert got['arm'] == pick
    assert got['p1_form'] == 'siglip1'
    assert got['no_comparison'] is False
    assert json.loads((tmp_path / 'prefix_pick.json').read_text()) == got


@pytest.mark.parametrize('arm', D.CANDIDATES)
def test_sc1_one_missing_arm_is_flagged(tmp_path, arm):
    put(tmp_path, f'text_{arm}.json', text_record(arm, [.25]))
    for form in D.CANDIDATES:
        put(tmp_path, f'text_p1_{form}.json', text_record('p1_' + form, [.125]))
    got = D.sc1(tmp_path)
    assert got['arm'] == arm and got['no_comparison']


def test_sc1_both_missing_is_inconclusive(tmp_path):
    assert D.sc1(tmp_path)['status'] == 'inconclusive'


def test_sc1_refuses_misidentified_seed_zero(tmp_path):
    put(tmp_path, 'text_siglip1.json', text_record('siglip1', [.25], [1]))
    with pytest.raises(AssertionError):
        D.sc1(tmp_path)


def test_k0_bounds_replay_the_frozen_records():
    got = D.k0_baseline()
    assert got['image']['m'] == .38629170813482466
    assert got['text']['m'] == .259351288425063
    assert round(got['image']['m'], 10) == .3862917081
    assert round(got['image']['r'], 10) == .0029788602
    assert round(got['text']['m'], 10) == .2593512884
    assert round(got['text']['r'], 10) == .0020107136
    # Spec's printed floor is not the decision input.
    assert got['text']['lower'] != .2573405749
    assert got['text']['per_seed'][1] == .2587331093021101


@pytest.mark.parametrize('image,text,level,status', [(.5,.5,3,'pass'), (.38,.26,0,'fail'),
    (.385,.259,1,'pass'), (.385,.26,2,'pass')])
def test_sc2_writes_stage_two_guard_and_full_bounds(tmp_path, image, text, level, status):
    put(tmp_path, 'prefix_pick.json', dict(status='picked', arm='siglip1'))
    put(tmp_path, 'groups_siglip1.json', dict(arm='siglip1', seeds=D.SEEDS,
        image=dict(per_seed=[image]*5), text=dict(per_seed=[text]*5)))
    got = D.sc2(tmp_path)
    assert (got['level'], got['status']) == (level, status)
    assert set(got['m']) == set(got['r']) == set(got['bounds']) == {'image', 'text'}
    assert got['per_seed']['image'] == [image]*5
    assert json.loads((tmp_path/'pass_rule.json').read_text()) == got


def test_sc2_refuses_without_pick_or_five_seeds(tmp_path):
    with pytest.raises((FileNotFoundError, AssertionError)):
        D.sc2(tmp_path)
    put(tmp_path, 'prefix_pick.json', dict(status='picked', arm='siglip1'))
    put(tmp_path, 'groups_siglip1.json', dict(arm='siglip1', seeds=[0],
        image=dict(per_seed=[.5]), text=dict(per_seed=[.5])))
    with pytest.raises(AssertionError):
        D.sc2(tmp_path)


def test_groups_reads_text_values_by_recorded_seed(tmp_path, monkeypatch):
    put(tmp_path, 'text_siglip1.json', text_record('siglip1', [.125, .25, .5]))
    # The expensive export scorer is an I/O boundary; verdict reads use real JSON.
    monkeypatch.setattr(D, 'image_groups', lambda root, arm, seeds:
                        dict(cells={c: [.75, .5] for c in D.IMAGE_CELLS}, per_seed=[.75, .5], mean=.625))
    got = D.groups('siglip1', [0, 2], records=tmp_path)
    assert got['text']['per_seed'] == [.125, .5]
    assert got['image']['per_seed'] == [.75, .5]
    assert got['seeds'] == [0, 2]


@pytest.mark.parametrize('damage', ['arm', 'seed', 'nan', 'cell'])
def test_groups_rejects_bad_text_record(tmp_path, monkeypatch, damage):
    doc = text_record('siglip1', [.25])
    if damage == 'arm':
        doc['arm'] = 'siglip1_prefix'
    elif damage == 'seed':
        doc['seeds'] = [1]
    elif damage == 'nan':
        doc['val']['group_r0']['per_seed'] = [float('nan')]
    else:
        del doc['val']['cells']['style']['t2t']
    put(tmp_path, 'text_siglip1.json', doc)
    with pytest.raises(AssertionError):
        D.groups('siglip1', [0], records=tmp_path)


def weights_record(equal=False, repeat_equal=False, epoch=12):
    def comparison(equal, runs):
        return dict(equal=equal, runs=runs, heads={a: dict(tensors_equal=equal,
            recorded_sha_ok=[True, True], best_epoch=[epoch, 10]) for a in D.ASPECTS})
    return dict(gate='i_a_weights', status='pass' if equal else 'fail', tries=1, written='test',
        details=dict(seed=0, encoder='siglip1', comparisons=dict(
            new_vs_r14=comparison(equal, ['new', 'r14']),
            r14_vs_r14_b=comparison(repeat_equal, ['r14', 'r14_b']))))


@pytest.mark.parametrize('equal,repeat,epoch,value,status', [
    (True, True, 99, 99, 'pass'),
    (False, True, 10, .25, 'fail'),
    (False, False, 12, .25, 'fallback'),
    (False, False, 8, .5, 'fallback'),
    (False, False, 13, .25, 'fail'),
    (False, False, 7, .25, 'fail'),
    (False, False, 10, math.nextafter(.25, -math.inf), 'fail'),
    (False, False, 10, math.nextafter(.5, math.inf), 'fail'),
])
def test_gate_i_a_exact_equality_or_all_fallback_conditions(equal, repeat, epoch, value, status):
    cells = {c: [.25, .3125, .375, .4375, .5] for c in D.IMAGE_CELLS}
    image = {c: [value] for c in D.IMAGE_CELLS}
    got = D.gate_i_a_verdict(weights_record(equal, repeat, epoch), cells, image)
    assert got['status'] == status
    assert got['details']['cuda_fallback'] == (status == 'fallback')


def test_gate_i_a_no_repeat_cannot_fallback():
    record = weights_record()
    del record['details']['comparisons']['r14_vs_r14_b']
    assert D.gate_i_a_verdict(record, {}, None)['status'] == 'fail'


@pytest.mark.parametrize('part,status,expected', [
    ('iv_store', 'pass', 'fail'), ('iv_hash', 'fail', 'fail'),
    ('iv_hash', 'pass', 'pass'), ('i_a_weights', 'pass', 'fail'),
    ('i_a', 'pass', 'pass'), ('i_a', 'fallback', 'fallback'),
])
def test_gates_requires_all_parts(tmp_path, part, status, expected):
    if part == 'iv_hash':
        put(tmp_path, 'gates/iv_store.json', dict(gate='iv_store', status='pass', tries=1, details={}, written='test'))
    if part == 'i_a':
        put(tmp_path, 'gates/i_a_weights.json', dict(gate='i_a_weights', status='pass', tries=1, details={}, written='test'))
    put(tmp_path, f'gates/{part}.json', dict(gate=part, status=status, tries=1, details={}, written='test'))
    gate = 'iv' if part.startswith('iv') else 'i_a'
    assert D.gates(tmp_path)['gates'][gate]['status'] == expected


def test_gate_merge_does_not_hide_failed_weights_part(tmp_path):
    for part, status in [('i_a_weights','fail'), ('i_a','fallback')]:
        put(tmp_path, f'gates/{part}.json', dict(gate=part, status=status, tries=1, details={}, written='test'))
    assert D.gates(tmp_path)['gates']['i_a']['status'] == 'fail'


@pytest.mark.parametrize('difference,expected', [(1e-6, 'pass'),
    (math.nextafter(1e-6, math.inf), 'fail'), (math.nextafter(1e-6, 0), 'pass')])
def test_gate_i_c_valpick_tolerance(difference, expected):
    # Zero avoids subtraction rounding hiding a one-ULP excess over the tolerance.
    got = {c: [difference] for c in D.IMAGE_CELLS}
    frozen = {c: [0]*5 for c in D.IMAGE_CELLS}
    assert D.valpick_verdict(got, frozen)['status'] == expected


def test_diagnostics_refuses_before_pass_rule(tmp_path):
    with pytest.raises((FileNotFoundError, AssertionError)):
        D.diagnostics(tmp_path)


def test_diagnostics_uses_local_seed_zero_and_plain_text_input(tmp_path, monkeypatch):
    monkeypatch.setattr(D.P, 'OUT', tmp_path/'out')
    monkeypatch.setattr(D.P, 'K0_ROOT', tmp_path/'k0')
    put(tmp_path, 'pass_rule.json', dict(arm='siglip1_prefix', seeds=D.SEEDS, level=0, status='fail'))
    def group_record(arm, seeds, image, text):
        root, _ = D.arm_location(arm)
        return dict(arm=arm, seeds=seeds, root=str(root),
            image=dict(cells={c: [image]*len(seeds) for c in D.IMAGE_CELLS}, per_seed=[image]*len(seeds)),
            text=dict(cells={c: [text]*len(seeds) for c in D.TEXT_CELLS}, per_seed=[text]*len(seeds)))
    put(tmp_path, 'groups_siglip1_prefix.json', group_record('siglip1_prefix', D.SEEDS, .75, .625))
    put(tmp_path, 'groups_gate_i_a_new.json', group_record('gate_i_a_new', [0], .5, .5))
    put(tmp_path, 'text_p1_siglip1.json', text_record('p1_siglip1', [.125]))
    for arm in ('qwen', 'siglip1_prefix', 'gate_i_a_new'):
        root, folder = D.arm_location(arm)
        for seed in ([0] if arm == 'gate_i_a_new' else D.SEEDS):
            put(root, f'train/{folder}/run__s{seed}.json', dict(name=folder, encoder='siglip1',
                seed=seed, status='ok', best_epoch={a: 12 for a in D.ASPECTS}))
    got = D.diagnostics(tmp_path)
    assert got['e']['image']['ami/emotion']['difference'] == pytest.approx(.397363698659534)
    assert got['e']['text']['emotion/t2i']['difference'] == pytest.approx(.23873095820224344)
    assert got['a']['image']['ami/emotion']['candidate'] == .75
    assert got['c']['emotion'] == dict(candidate=.625, plain=.125, difference=.5)
    assert got['b']['candidate']['0']['emotion'] == 12
    assert got['pass_rule']['status'] == 'fail'  # diagnostics also run after a decided fail


@pytest.mark.parametrize('command', ['groups', 'sc1', 'sc2', 'gate-i-a', 'gate-i-c-valpick', 'gates', 'diagnostics'])
def test_cli_exposes_ticket_commands(command):
    parser = D.cli_parser()
    args = ['groups', '--arm', 'siglip1', '--seeds', '0'] if command == 'groups' else [command]
    assert parser.parse_args(args).command == command


@pytest.mark.parametrize('prefix,expected', [
    (math.nextafter(.125, -math.inf), 'siglip1'),
    (.125, 'siglip1'),
    (math.nextafter(.125, math.inf), 'siglip1_prefix')])
def test_d20_selects_strongest_p1_at_r0_with_exact_plain_tie(tmp_path, prefix, expected):
    for arm, value in [('siglip1', .25), ('siglip1_prefix', .5),
                       ('p1_siglip1', .125), ('p1_siglip1_prefix', prefix)]:
        put(tmp_path, f'text_{arm}.json', text_record(arm, [value]))
    assert D.sc1(tmp_path)['p1_form'] == expected


def test_k0_range_is_range_of_group_means_not_mean_of_cell_ranges(tmp_path, monkeypatch):
    # Opposing cell swings cancel in every seed. Cell ranges are .5, group range is zero.
    cells = {c: ([.25, .75, .25, .75, .5] if i % 2 else [.75, .25, .75, .25, .5])
             for i, c in enumerate(D.IMAGE_CELLS)}
    put(tmp_path, 'valpick.json', dict(kind='r28_valpick', seeds=D.SEEDS, arm='k11=40 (c*)',
        encoders=dict(siglip1=dict(encoder='siglip1', n_val=6152, score=.5,
            cells={c: dict(per_seed=v) for c, v in cells.items()}))))
    put(tmp_path, 'text.json', dict(ours_siglip1=dict(picks={a: {d: dict(per_seed=dict(r0=[.25]*5), means=dict(r0=.25))
        for d in ('t2i','i2t','t2t')} for a in D.ASPECTS})))
    for name, value in [('K0_IMAGE_MEAN', .5), ('K0_IMAGE_SEEDS', (.5,.5)),
                        ('K0_TEXT_MEAN', .25), ('K0_TEXT_SEEDS', (.25,.25))]:
        monkeypatch.setattr(D.P, name, value)
    assert D.k0_baseline(tmp_path)['image']['r'] == 0


@pytest.mark.parametrize('cell', D.IMAGE_CELLS)
def test_fallback_requires_every_image_cell_within_its_own_range(cell):
    old = {c: [.25]*5 for c in D.IMAGE_CELLS}
    new = {c: [.25] for c in D.IMAGE_CELLS}
    new[cell] = [math.nextafter(.25, math.inf)]
    assert D.gate_i_a_verdict(weights_record(), old, new)['status'] == 'fail'


def test_export_part_alone_is_not_gate_i_c_pass(tmp_path):
    put(tmp_path, 'gates/i_c.json', dict(gate='i_c', status='pass', tries=1, details={}, written='test'))
    got = D.gates(tmp_path)['gates']['i_c']
    assert got['status'] == 'fail' and got['missing_parts'] == ['i_c_valpick']


def test_gates_preserves_standalone_gates_and_rejects_wrong_identity(tmp_path):
    put(tmp_path, 'gates/ii.json', dict(gate='ii', status='pass', tries=1, details={}, written='test'))
    assert D.gates(tmp_path)['gates']['ii']['status'] == 'pass'
    put(tmp_path, 'gates/ii.json', dict(gate='iii', status='pass', tries=1, details={}, written='test'))
    with pytest.raises(AssertionError):
        D.gates(tmp_path)
