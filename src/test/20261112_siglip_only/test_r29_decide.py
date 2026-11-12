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


def gate_file(root, part, status='pass', tries=1):
    put(root, f'gates/{part}.json', dict(gate=part, status=status, tries=tries, details={}, written='test'))


def put_gates(root, tries=None, **status):
    """A merged gates.json as `gates` writes it: every stage 1 gate passes unless `status` says otherwise.
    A part's `tries` default to 3 when it is not `pass` (a gate that failed at its third try is parked), else 1."""
    tries = tries or {}
    for part in ('i_a_weights', 'i_a', 'i_c', 'i_c_valpick', 'ii', 'iii', 'iv_store', 'iv_hash'):
        st = status.get(part, 'pass')
        gate_file(root, part, st, tries.get(part, 1 if st == 'pass' else 3))
    D.gates(root)


def put_crash(root, name, **change):
    put(root, f'crash_{name}.json', dict(dict(arm=name, seed=3, status='out', tries=2, details='non-finite loss',
                                             written='2026-11-14 01:02'), **change))


def put_arms(root, k1=.25, k2=.25, p1=(.125, .125)):
    for arm, value in [('siglip1', k1), ('siglip1_prefix', k2)]:
        if value is not None:
            # Later seeds deliberately reverse the winner: SC1 must only use seed 0.
            put(root, f'text_{arm}.json', text_record(arm, [value, 1 - value]))
    for form, value in zip(D.CANDIDATES, p1):
        put(root, f'text_p1_{form}.json', text_record('p1_' + form, [value]))


@pytest.mark.parametrize('second,pick', [(.25, 'siglip1'),
    (math.nextafter(.25, -math.inf), 'siglip1'),
    (math.nextafter(.25, math.inf), 'siglip1_prefix')])
def test_sc1_seed_zero_exact_tie_and_d20_plain_tie(tmp_path, second, pick):
    put_gates(tmp_path)
    put_arms(tmp_path, k2=second)
    got = D.sc1(tmp_path)
    assert got == dict(kstar=pick, kstar_form='plain' if pick == 'siglip1' else 'prefixed', p1_form='plain', flag=None)
    assert json.loads((tmp_path / 'prefix_pick.json').read_text()) == got   # these four keys, nothing else
    assert D.read_pick(tmp_path) == got


def test_sc1_refuses_while_an_arm_is_missing_without_evidence(tmp_path):
    put_gates(tmp_path)
    put_arms(tmp_path, k2=None)           # K2 not scored, no crash record, gate (iv) passes: not out
    with pytest.raises(AssertionError, match='not out'):
        D.sc1(tmp_path)
    assert not (tmp_path / 'prefix_pick.json').exists()


def test_sc1_refuses_when_gate_iv_is_incomplete_and_no_part_failed(tmp_path):
    put_gates(tmp_path)
    (tmp_path / 'gates/iv_hash.json').unlink()
    D.gates(tmp_path)                     # merged iv is `fail` only because a part is missing: no evidence
    put_arms(tmp_path, k2=None)
    with pytest.raises(AssertionError, match='incomplete'):
        D.sc1(tmp_path)


@pytest.mark.parametrize('part,status', [('iv_store', 'fail'), ('iv_hash', 'fail'), ('iv_store', 'parked')])
def test_sc1_k2_is_out_when_gate_iv_parked_or_failed_at_try_three(tmp_path, part, status):
    put_gates(tmp_path, **{part: status})
    put_arms(tmp_path, k1=.25, k2=.75, p1=(.125, .5))   # K2 would win and be scored: gate (iv) still takes it out
    got = D.sc1(tmp_path)
    assert (got['kstar'], got['kstar_form'], got['p1_form']) == ('siglip1', 'plain', 'plain')
    assert got['flag'].startswith('siglip1_prefix out') and 'row 1a' in got['flag']


@pytest.mark.parametrize('part,n', [('iv_store', 1), ('iv_hash', 2)])
def test_sc1_gate_iv_fail_before_try_three_is_still_open_and_k2_is_not_out(tmp_path, part, n):
    put_gates(tmp_path, tries={part: n}, **{part: 'fail'})
    put_arms(tmp_path)
    with pytest.raises(AssertionError, match=rf'gate iv still open \(fail at try {n} of 3\)'):
        D.sc1(tmp_path)
    assert not (tmp_path / 'prefix_pick.json').exists()
    put_gates(tmp_path, tries={part: 3}, **{part: 'fail'})     # the third failed try parks it: K2 is out
    assert D.sc1(tmp_path)['kstar'] == 'siglip1'


@pytest.mark.parametrize('arm,kstar', [('siglip1', 'siglip1_prefix'), ('siglip1_prefix', 'siglip1')])
def test_sc1_an_arm_with_a_crash_record_is_out(tmp_path, arm, kstar):
    put_gates(tmp_path)
    put_crash(tmp_path, arm)
    put_arms(tmp_path, k1=.5 if arm == 'siglip1' else .25, k2=.25 if arm == 'siglip1' else .5, p1=(.125, .5))
    (tmp_path / f'text_{arm}.json').unlink()     # a crashed arm has no text record
    got = D.sc1(tmp_path)
    assert got['kstar'] == kstar and arm in got['flag'] and 'second crash' in got['flag']
    # K2 as the only arm: D20 still picks P1's form on validation
    assert got['p1_form'] == ('prefixed' if kstar == 'siglip1_prefix' else 'plain')


def test_sc1_both_arms_out_writes_no_pick(tmp_path):
    put_gates(tmp_path)
    put_crash(tmp_path, 'siglip1')
    put_crash(tmp_path, 'siglip1_prefix')
    got = D.sc1(tmp_path)
    assert got['status'] == 'inconclusive'
    assert not (tmp_path / 'prefix_pick.json').exists()


@pytest.mark.parametrize('change', [dict(status='ok'), dict(tries=1), dict(arm='siglip1'), dict(seed='3')])
def test_crash_reader_asserts_its_keys_and_values(tmp_path, change):
    put_crash(tmp_path, 'siglip1_prefix', **change)
    with pytest.raises(AssertionError):
        D.read_crash(tmp_path, 'siglip1_prefix')


def test_crash_reader_rejects_missing_and_extra_keys(tmp_path):
    put_crash(tmp_path, 'siglip1_prefix')
    doc = json.loads((tmp_path / 'crash_siglip1_prefix.json').read_text())
    for bad in ({k: v for k, v in doc.items() if k != 'seed'}, dict(doc, extra=1)):
        put(tmp_path, 'crash_siglip1_prefix.json', bad)
        with pytest.raises(AssertionError):
            D.read_crash(tmp_path, 'siglip1_prefix')
    assert D.read_crash(tmp_path, 'siglip1') is None


@pytest.mark.parametrize('gate,status', [('i_a', 'fail'), ('i_c', 'parked'), ('ii', 'fail'), ('iii', 'fail')])
def test_sc1_and_sc2_refuse_when_gate_i_ii_or_iii_parked(tmp_path, gate, status):
    put_gates(tmp_path, **{gate: status})
    put_arms(tmp_path)
    with pytest.raises(AssertionError, match='parked'):
        D.sc1(tmp_path)
    put(tmp_path, 'prefix_pick.json', dict(kstar='siglip1', kstar_form='plain', p1_form='plain', flag=None))
    with pytest.raises(AssertionError, match='parked'):
        D.sc2(tmp_path)


@pytest.mark.parametrize('gate,n', [('i_a', 1), ('i_c_valpick', 2), ('ii', 1), ('iii', 2)])
def test_sc1_and_sc2_refuse_a_gate_that_failed_before_try_three_as_still_open(tmp_path, gate, n):
    put_gates(tmp_path, tries={gate: n}, **{gate: 'fail'})
    put_arms(tmp_path)
    shown = {'i_c_valpick': 'i_c'}.get(gate, gate)
    with pytest.raises(AssertionError, match=rf'gate {shown} still open \(fail at try {n} of 3\)'):
        D.sc1(tmp_path)
    put(tmp_path, 'prefix_pick.json', PICK_K1)
    with pytest.raises(AssertionError, match='still open'):
        D.sc2(tmp_path)


@pytest.mark.parametrize('status,tries,message', [('fail', 3, 'parked'), ('parked', 1, 'parked'),
                                                  ('fail', 1, 'still open')])
def test_sc1_and_sc2_refuse_when_a_failing_i_b_record_exists(tmp_path, status, tries, message):
    put_gates(tmp_path)
    gate_file(tmp_path, 'i_b', status, tries)
    D.gates(tmp_path)
    put_arms(tmp_path)
    with pytest.raises(AssertionError, match=message):
        D.sc1(tmp_path)
    put(tmp_path, 'prefix_pick.json', PICK_K1)
    with pytest.raises(AssertionError, match=message):
        D.sc2(tmp_path)
    gate_file(tmp_path, 'i_b', 'pass')                # a passing i_b record does not block
    D.gates(tmp_path)
    (tmp_path / 'prefix_pick.json').unlink()
    assert D.sc1(tmp_path)['kstar'] == 'siglip1'


def test_sc1_and_sc2_refuse_without_gates_json_and_accept_a_fallback(tmp_path):
    put_arms(tmp_path)
    with pytest.raises(FileNotFoundError):
        D.sc1(tmp_path)
    put_gates(tmp_path, i_a='fallback')
    assert D.sc1(tmp_path)['kstar'] == 'siglip1'


def test_sc1_and_sc2_write_once_and_force_keeps_the_reason(tmp_path):
    put_gates(tmp_path)
    put_arms(tmp_path)
    first = D.sc1(tmp_path)
    with pytest.raises(AssertionError, match='written once'):
        D.sc1(tmp_path)
    with pytest.raises(AssertionError):
        D.sc1(tmp_path, force='')
    again = D.sc1(tmp_path, force='re-scored K2 after a store fix')
    assert again == first == json.loads((tmp_path / 'prefix_pick.json').read_text())   # exactly the four keys
    side = json.loads((tmp_path / 'prefix_pick_forced.json').read_text())
    assert side['force_reason'] == 're-scored K2 after a store fix' and side['replaced'] == first
    assert 'earlier' not in side
    D.sc1(tmp_path, force='again')                        # a second forced rewrite keeps the first side file
    side = json.loads((tmp_path / 'prefix_pick_forced.json').read_text())
    assert side['force_reason'] == 'again' and side['earlier']['force_reason'] == 're-scored K2 after a store fix'
    assert D.read_pick(tmp_path)['kstar'] == 'siglip1'
    put(tmp_path, 'groups_siglip1.json', dict(arm='siglip1', seeds=D.SEEDS,
        image=dict(per_seed=[.5]*5), text=dict(per_seed=[.5]*5)))
    rule = D.sc2(tmp_path)
    with pytest.raises(AssertionError, match='written once'):
        D.sc2(tmp_path)
    assert D.sc2(tmp_path, force='again') == rule == json.loads((tmp_path / 'pass_rule.json').read_text())
    assert 'force_reason' not in rule
    assert json.loads((tmp_path / 'pass_rule_forced.json').read_text())['force_reason'] == 'again'


def test_sc1_refuses_misidentified_seed_zero(tmp_path):
    put_gates(tmp_path)
    put(tmp_path, 'text_siglip1.json', text_record('siglip1', [.25], [1]))
    put(tmp_path, 'text_siglip1_prefix.json', text_record('siglip1_prefix', [.25]))
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


def test_sc2_uses_the_five_seed_mean_not_seed_zero(tmp_path):
    # K0: image m-r = .38331, text m-r = .25734. Seed 0 alone is level 3 and the five-seed mean level 0 (and the
    # reverse in the second case: seed 0 alone is level 0, the mean level 2), so a seed-0-only or best-seed-only mean fails one of them.
    put_gates(tmp_path)
    put(tmp_path, 'prefix_pick.json', PICK_K1)
    cases = [([.40, .30, .30, .30, .30], [.27, .20, .20, .20, .20], 0),
             ([.30, .42, .42, .42, .42], [.20, .275, .275, .275, .275], 2)]
    for i, (image, text, level) in enumerate(cases):
        put(tmp_path, 'groups_siglip1.json', dict(arm='siglip1', seeds=D.SEEDS,
            image=dict(per_seed=image), text=dict(per_seed=text)))
        got = D.sc2(tmp_path, force=None if i == 0 else 'second case')
        assert got['level'] == level
        assert got['mean']['image'] == pytest.approx(sum(image) / 5) and got['per_seed']['text'] == text


@pytest.mark.parametrize('image,text,level,status', [(.5,.5,3,'pass'), (.38,.26,0,'fail'),
    (.385,.259,1,'pass'), (.385,.26,2,'pass')])
def test_sc2_writes_stage_two_guard_and_full_bounds(tmp_path, image, text, level, status):
    put_gates(tmp_path)
    put(tmp_path, 'prefix_pick.json', PICK_K1)
    put(tmp_path, 'groups_siglip1.json', dict(arm='siglip1', seeds=D.SEEDS,
        image=dict(per_seed=[image]*5), text=dict(per_seed=[text]*5)))
    got = D.sc2(tmp_path)
    assert (got['level'], got['status']) == (level, status)
    assert set(got['m']) == set(got['r']) == set(got['bounds']) == {'image', 'text'}
    assert got['per_seed']['image'] == [image]*5
    assert json.loads((tmp_path/'pass_rule.json').read_text()) == got


PICK_K1 = dict(kstar='siglip1', kstar_form='plain', p1_form='plain', flag=None)


def test_pick_reader_asserts_its_four_keys(tmp_path):
    for bad in (dict(PICK_K1, extra=1), {k: v for k, v in PICK_K1.items() if k != 'flag'},
                dict(PICK_K1, kstar_form='prefixed'), dict(PICK_K1, p1_form='siglip1'), dict(PICK_K1, kstar='qwen')):
        put(tmp_path, 'prefix_pick.json', bad)
        with pytest.raises(AssertionError):
            D.read_pick(tmp_path)


def test_sc2_refuses_without_pick_or_five_seeds(tmp_path):
    put_gates(tmp_path)
    with pytest.raises((FileNotFoundError, AssertionError)):
        D.sc2(tmp_path)
    put(tmp_path, 'prefix_pick.json', PICK_K1)
    put(tmp_path, 'groups_siglip1.json', dict(arm='siglip1', seeds=[0],
        image=dict(per_seed=[.5]), text=dict(per_seed=[.5])))
    with pytest.raises(AssertionError):
        D.sc2(tmp_path)


def test_text_record_group_must_equal_the_mean_of_its_nine_cells(tmp_path):
    doc = text_record('siglip1', [.25, .5])
    doc['val']['cells']['style']['t2t']['per_seed']['r0'] = [.25, .625]   # nine-cell mean of seed 1 is .5139, not .5
    put(tmp_path, 'text_siglip1.json', doc)
    with pytest.raises(AssertionError, match='group_r0'):
        D.read_text(tmp_path, 'siglip1')
    doc['val']['group_r0']['per_seed'] = [.25, .5 + .125 / 9]   # the true nine-cell means: accepted
    put(tmp_path, 'text_siglip1.json', doc)
    assert D.read_text(tmp_path, 'siglip1')['per_seed'][1] == pytest.approx(.5 + .125 / 9)


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


def test_gate_i_a_without_the_r14_b_repeat_refuses_and_spends_no_try(tmp_path):
    record = weights_record()
    del record['details']['comparisons']['r14_vs_r14_b']
    with pytest.raises(AssertionError, match='repeat missing'):
        D.gate_i_a_verdict(record, {}, None)
    put(tmp_path, 'gates/i_a_weights.json', record)
    with pytest.raises(AssertionError, match='repeat missing'):
        D.gate_i_a(tmp_path)
    assert not (tmp_path / 'gates/i_a.json').exists()
    # equal weights need no repeat
    record = weights_record(equal=True)
    del record['details']['comparisons']['r14_vs_r14_b']
    assert D.gate_i_a_verdict(record, {}, None)['status'] == 'pass'


def test_gate_records_use_the_shared_writer_and_stop_after_three_tries(tmp_path):
    cells = {c: [.25] * 5 for c in D.IMAGE_CELLS}
    frozen = {c: [0] * 5 for c in D.IMAGE_CELLS}
    for n in (1, 2, 3):
        rec = D.write_gate(tmp_path, 'i_c_valpick', D.valpick_verdict({c: [0] for c in D.IMAGE_CELLS}, frozen))
        assert rec['tries'] == n and len(rec['written']) == 16 and '+' not in rec['written']
    with pytest.raises(AssertionError, match='fourth'):
        D.write_gate(tmp_path, 'i_c_valpick', D.valpick_verdict({c: [0] for c in D.IMAGE_CELLS}, frozen))
    assert D.read_json(tmp_path / 'gates/i_c_valpick.json')['tries'] == 3


@pytest.mark.parametrize('part,status,expected', [
    ('iv_store', 'pass', 'fail'), ('iv_hash', 'fail', 'fail'),
    ('iv_hash', 'pass', 'pass'), ('i_a_weights', 'pass', 'fail'),
    ('i_a', 'pass', 'pass'), ('i_a', 'fallback', 'fallback'), ('i_a', 'fail', 'fail'),
])
def test_gates_requires_all_parts(tmp_path, part, status, expected):
    if part == 'iv_hash':
        put(tmp_path, 'gates/iv_store.json', dict(gate='iv_store', status='pass', tries=1, details={}, written='test'))
    if part == 'i_a':
        put(tmp_path, 'gates/i_a_weights.json', dict(gate='i_a_weights', status='pass', tries=1, details={}, written='test'))
    put(tmp_path, f'gates/{part}.json', dict(gate=part, status=status, tries=1, details={}, written='test'))
    gate = 'iv' if part.startswith('iv') else 'i_a'
    assert D.gates(tmp_path)['gates'][gate]['status'] == expected


@pytest.mark.parametrize('weights,verdict,expected', [('fail', 'fallback', 'fallback'), ('fail', 'pass', 'pass'),
                                                      ('pass', 'fail', 'fail'), ('fail', 'fail', 'fail')])
def test_gate_i_a_merged_status_is_the_verdicts_and_the_weights_part_only_present(tmp_path, weights, verdict, expected):
    # D23: the weights part says `fail` whenever the weights differ; the verdict (pass, fallback or fail) decides.
    gate_file(tmp_path, 'i_a_weights', weights)
    gate_file(tmp_path, 'i_a', verdict)
    got = D.gates(tmp_path)['gates']['i_a']
    assert got['status'] == expected and got['missing_parts'] == []


def test_gate_i_a_verdict_without_the_weights_part_is_incomplete(tmp_path):
    gate_file(tmp_path, 'i_a', 'pass')
    got = D.gates(tmp_path)['gates']['i_a']
    assert got['status'] == 'fail' and got['missing_parts'] == ['i_a_weights']


def test_gate_iv_stays_co_equal(tmp_path):
    gate_file(tmp_path, 'iv_store', 'pass')
    gate_file(tmp_path, 'iv_hash', 'fail')
    assert D.gates(tmp_path)['gates']['iv']['status'] == 'fail'


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


@pytest.mark.parametrize('kstar,form,plain', [('siglip1_prefix', 'prefixed', .25), ('siglip1', 'plain', .125)])
def test_diagnostics_uses_local_seed_zero_and_kstars_own_input_form(tmp_path, monkeypatch, kstar, form, plain):
    monkeypatch.setattr(D.P, 'OUT', tmp_path/'out')
    monkeypatch.setattr(D.P, 'K0_ROOT', tmp_path/'k0')
    put(tmp_path, 'pass_rule.json', dict(arm=kstar, seeds=D.SEEDS, level=0, status='fail'))
    # D20's p1_form is deliberately the other form: (c) follows kstar_form, not the comparator's form
    put(tmp_path, 'prefix_pick.json', dict(kstar=kstar, kstar_form=form,
                                           p1_form='plain' if form == 'prefixed' else 'prefixed', flag=None))
    def group_record(arm, seeds, image, text):
        root, _ = D.arm_location(arm)
        return dict(arm=arm, seeds=seeds, root=str(root),
            image=dict(cells={c: [image]*len(seeds) for c in D.IMAGE_CELLS}, per_seed=[image]*len(seeds)),
            text=dict(cells={c: [text]*len(seeds) for c in D.TEXT_CELLS}, per_seed=[text]*len(seeds)))
    put(tmp_path, f'groups_{kstar}.json', group_record(kstar, D.SEEDS, .75, .625))
    put(tmp_path, 'groups_gate_i_a_new.json', group_record('gate_i_a_new', [0], .5, .5))
    put(tmp_path, 'text_p1_siglip1.json', text_record('p1_siglip1', [.125]))
    put(tmp_path, 'text_p1_siglip1_prefix.json', text_record('p1_siglip1_prefix', [.25]))
    for arm in ('qwen', kstar, 'gate_i_a_new'):
        root, folder = D.arm_location(arm)
        for seed in ([0] if arm == 'gate_i_a_new' else D.SEEDS):
            put(root, f'train/{folder}/run__s{seed}.json', dict(name=folder, encoder='siglip1',
                seed=seed, status='ok', best_epoch={a: 12 for a in D.ASPECTS}))
    got = D.diagnostics(tmp_path)
    assert got['e']['image']['ami/emotion']['difference'] == pytest.approx(.397363698659534)
    assert got['e']['text']['emotion/t2i']['difference'] == pytest.approx(.23873095820224344)
    assert got['a']['image']['ami/emotion']['candidate'] == .75
    assert got['c']['emotion'] == dict(candidate=.625, plain=plain, difference=.625 - plain)
    assert got['c_input_form'] == form
    assert got['b']['candidate']['0']['emotion'] == 12
    assert got['pass_rule']['status'] == 'fail'  # diagnostics also run after a decided fail


@pytest.mark.parametrize('command', ['groups', 'sc1', 'sc2', 'gate-i-a', 'gate-i-c-valpick', 'gates', 'diagnostics'])
def test_cli_exposes_ticket_commands(command):
    parser = D.cli_parser()
    args = ['groups', '--arm', 'siglip1', '--seeds', '0'] if command == 'groups' else [command]
    assert parser.parse_args(args).command == command


@pytest.mark.parametrize('prefix,expected', [
    (math.nextafter(.125, -math.inf), 'plain'),
    (.125, 'plain'),
    (math.nextafter(.125, math.inf), 'prefixed')])
def test_d20_selects_strongest_p1_at_r0_with_exact_plain_tie(tmp_path, prefix, expected):
    put_gates(tmp_path)
    put_arms(tmp_path, k1=.25, k2=.5, p1=(.125, prefix))
    got = D.sc1(tmp_path)
    assert got['kstar'] == 'siglip1_prefix' and got['kstar_form'] == 'prefixed' and got['p1_form'] == expected


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
