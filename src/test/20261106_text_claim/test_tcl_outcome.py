"""Fixture checks of text-claim design §3.4 (interval, outcomes, pair cells), §3.5 and §6 (verdicts, inconclusive)."""
import copy
import inspect
import json

import numpy as np
import pytest

import tcl_outcome as oc
import tcl_paths as p

DRAWS = 50                      # the value does not matter where the per-query values are constant
HELD = {'art': 'art_held', 'mm': 'mm_test'}
CELLS = [(a, d) for a in p.ASPECTS for d in p.DIRECTIONS]      # the 9 label cells, in cell order
# One read-out per row, all different from each other, so that a wrong pick reads a wrong file.
PICK = dict(cstar='center', late='center', plain='r0', teacher='center_k10')
PAIR_LATE = dict(head='style', readout='center_k10')
PAIR_PLAIN = dict(readout='center')


@pytest.fixture(autouse=True)
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'OUT', tmp_path)
    monkeypatch.setattr(p, 'SMOKE', 3)
    monkeypatch.setitem(p.N_PAINTINGS, 'art_held', 3)
    monkeypatch.setitem(p.N_PAINTINGS, 'mm_test', 3)
    monkeypatch.setitem(p.N_CAPTIONS, 'art_held', 4)


def pid_of(group, direction):
    return np.array([0, 0, 1, 2] if group == 'art' and direction != 'i2t' else [0, 1, 2], dtype=np.int64)


def rank_of(hit, n):
    return np.full(n, 1 if hit else np.nan, dtype=np.float32)


class Files:
    """Writes the score files of a fixture; fields of one path merge, so a pair file can hold labels too."""

    def __init__(self, out):
        self.out, self.fields = out, {}

    def add(self, group, kind, row, head, seed, direction, readout, ap=None, aspect=None, ranks=None):
        path = p.score_path(HELD[group], kind, row, head, seed, direction, readout, out=self.out)
        n = len(pid_of(group, direction))
        f = self.fields.setdefault(path, dict(q_pid=pid_of(group, direction),
                                              first_rank=np.full(n, 20, dtype=np.float32)))
        if ap is not None:
            f[f'ap__{aspect}'] = np.broadcast_to(np.float32(ap), n).astype(np.float32) \
                if np.isscalar(ap) else np.asarray(ap, dtype=np.float32)
        if ranks is not None:
            f['first_rank'] = np.asarray(ranks, dtype=np.float32)

    def flush(self, skip=()):
        for path, f in self.fields.items():
            if path in skip:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **f)


def build_group(out, group, nP=9, nT=9, q=(True, True), cand=0.5, lose=0.75):
    """Score files of one group. The candidate scores `cand` everywhere; plain is below it in the first nP cells
    (0.25) and `lose` elsewhere (default 0.75, above it); the teacher likewise for nT. Late fusion's pair hit is q per direction.
    Wrong read-outs hold the opposite, so a wrong pick changes the result. Returns the Files object."""
    kind = p.BAR_KIND[group]
    fl = Files(out)
    for i, (a, d) in enumerate(CELLS):
        for readout in p.READOUTS:
            right = {r: readout == PICK[r] for r in PICK}
            for cand_row in ('cstar', 'late'):
                for seed in p.SEEDS:
                    fl.add(group, kind, cand_row, a, seed, d, readout, aspect=a,
                           ap=cand if right[cand_row] else 0.0)
            fl.add(group, kind, 'plain', 'all', 'x', d, readout, aspect=a,
                   ap=(0.25 if i < nP else lose) if right['plain'] else 0.99)
            fl.add(group, kind, 'teacher', a, 'x', d, readout, aspect=a,
                   ap=(0.25 if i < nT else lose) if right['teacher'] else 0.99)
    for d, ok in zip(p.PAIR_DIRECTIONS, q):
        n = len(pid_of(group, d))
        for readout in p.READOUTS:
            for seed in p.SEEDS:
                fl.add(group, kind, 'late', PAIR_LATE['head'], seed, d, readout,
                       ranks=rank_of(ok if readout == PAIR_LATE['readout'] else not ok, n))
            fl.add(group, kind, 'plain', 'all', 'x', d, readout,
                   ranks=rank_of(True if readout == PAIR_PLAIN['readout'] else False, n))
    return fl


def stage1_record(group, ranges=None):
    kind = p.BAR_KIND[group]
    cell = lambda f: {a: {d: f(a, d) for d in p.DIRECTIONS} for a in p.ASPECTS}      # noqa: E731
    default = lambda row, a, d: 0.1 + 0.01 * CELLS.index((a, d)) + (0.05 if row == 'late' else 0)   # noqa: E731
    ranges = ranges or {}
    return dict(group=group, values={},
                picks={kind: {row: cell(lambda a, d, row=row: PICK[row]) for row in PICK}},
                ranges={kind: {row: cell(lambda a, d, row=row: ranges.get((row, a, d), default(row, a, d)))
                               for row in ('cstar', 'late')}},
                pair_picks={kind: {d: dict(late=PAIR_LATE, plain=PAIR_PLAIN) for d in p.PAIR_DIRECTIONS}})


def good_status():
    return dict(gate_i={k: True for k in oc.gate_i_keys()}, gate_ii=dict(art=True, mm=True), gate_iii=True,
                gate_iv=dict(art=True, mm=True), stage1=dict(art=True, mm=True))


def write_record(name, obj):
    path = p.record_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def setup(tmp_path, art=None, mm=None, status=None, skip=(), records=('art', 'mm'), ranges=None):
    """Write both groups (cfg dicts for build_group), the stage 1 records and status. skip: score paths to leave out."""
    files = [build_group(tmp_path, g, **c) for g, c in (('art', art or {}), ('mm', mm or {}))]
    for fl in files:
        fl.flush(skip=skip)
    for group in records:
        write_record(f'stage1_{group}.json', stage1_record(group, ranges))
    write_record('status.json', status or good_status())
    return files


def run(draws=DRAWS):
    return oc.run(draws=draws)


def verdicts(outcomes):
    return {sc: v['verdict'] for sc, v in outcomes['verdicts'].items()}


def test_all_good_fixture_holds_all_four_and_writes_every_cell(tmp_path):
    setup(tmp_path)
    cells, outcomes = run()
    assert outcomes['verdicts'] == {
        'SC1': dict(P=9, T=9, Q=None, verdict='holds'), 'SC2': dict(P=9, T=9, Q=None, verdict='holds'),
        'SC3': dict(P=9, T=9, Q=True, verdict='holds'), 'SC4': dict(P=9, T=9, Q=True, verdict='holds')}
    assert outcomes['voided'] == []
    # per group: c* 9 cells x 2 comparators, late fusion 9 x 2 + 2 pair cells
    assert len(cells) == 2 * (18 + 18 + 2)
    assert {c['claim_voided'] for c in cells} == {False}


# ------------------------------------------------------------------------------------------- the §6 table
TABLE = [(9, 9, 'holds'), (9, 5, 'holds'), (9, 4, 'qualified'), (9, 3, 'qualified'), (9, 2, 'fails'),
         (8, 9, 'fails')]


@pytest.mark.parametrize('q_ok', [True, False])
@pytest.mark.parametrize('n_plain,n_teacher,word', TABLE)
def test_verdict_table_for_both_candidates(tmp_path, n_plain, n_teacher, word, q_ok):
    cfg = dict(nP=n_plain, nT=n_teacher, q=(True, q_ok))
    setup(tmp_path, art=cfg)
    _, outcomes = run()
    v = outcomes['verdicts']
    assert v['SC1'] == dict(P=n_plain, T=n_teacher, Q=None, verdict=word)          # c*: no pair clause
    assert v['SC3'] == dict(P=n_plain, T=n_teacher, Q=q_ok, verdict=word if q_ok else 'fails')
    assert verdicts(outcomes)['SC2'] == verdicts(outcomes)['SC4'] == 'holds'      # the other set is as it was


@pytest.mark.parametrize('q', [(False, True), (True, False)])
def test_one_failed_pair_direction_fails_late_fusion_only(tmp_path, q):
    setup(tmp_path, mm=dict(q=q))
    _, outcomes = run()
    assert outcomes['verdicts']['SC4'] == dict(P=9, T=9, Q=False, verdict='fails')
    assert outcomes['verdicts']['SC2']['verdict'] == 'holds'


def test_level_over_plain_is_not_counted(tmp_path):
    # the 9th cell of plain is 0.05 above the candidate: LEVEL (seed range >= 0.1), which is no WIN
    setup(tmp_path, art=dict(nP=8, lose=0.55))
    cells, outcomes = run()
    plain = [c for c in cells if c['group'] == 'art' and c['candidate'] == 'cstar' and c['comparator'] == 'plain']
    assert [c['outcome'] for c in plain].count('LEVEL') == 1 and [c['outcome'] for c in plain].count('WIN') == 8
    assert outcomes['verdicts']['SC1'] == dict(P=8, T=9, Q=None, verdict='fails')


def test_teacher_wins_are_the_only_ones_counted_in_t(tmp_path):
    setup(tmp_path, art=dict(nT=4, lose=0.55))      # 5 teacher cells are LEVEL, not WIN
    _, outcomes = run()
    assert outcomes['verdicts']['SC1'] == dict(P=9, T=4, Q=None, verdict='qualified')


def test_value_exactly_at_the_bar_is_no_win(tmp_path):
    # candidate equals plain in every query: lower bound exactly 0, which is not strictly above 0
    build_group(tmp_path, 'art', nP=0, lose=0.5).flush()
    got = oc.label_cell('art', 'cstar', 'plain', 'emotion', 't2i', stage1_record('art'), p.N_DRAWS)
    assert (got['lower'], got['upper'], got['point']) == (0.0, 0.0, 0.0)
    assert got['outcome'] == 'LEVEL'                                    # 0 > -seed_range, and no WIN


def test_value_exactly_at_the_bar_with_zero_seed_range_is_below(tmp_path):
    build_group(tmp_path, 'art', nP=0, lose=0.5).flush()
    record = stage1_record('art', ranges={('cstar', 'emotion', 't2i'): 0.0})
    got = oc.label_cell('art', 'cstar', 'plain', 'emotion', 't2i', record, p.N_DRAWS)
    assert got['lower'] == 0.0 and got['outcome'] == 'BELOW'


def test_outcome_uses_the_candidates_seed_range_of_that_cell(tmp_path):
    # every comparator cell is 0.05 above the candidate: LEVEL iff the candidate's range exceeds 0.05
    ranges = {('cstar', a, d): 0.1 for a, d in CELLS}
    ranges.update({('late', a, d): 0.01 for a, d in CELLS})
    ranges[('cstar', 'style', 'i2t')] = 0.04
    setup(tmp_path, art=dict(nP=0, nT=0, lose=0.55), ranges=ranges)
    cells, _ = run()
    got = {(c['candidate'], c['comparator'], c['aspect'], c['direction']): (c['outcome'], c['seed_range'])
           for c in cells if c['group'] == 'art' and c['cell'] == 'label'}
    assert got[('cstar', 'plain', 'emotion', 't2i')] == ('LEVEL', 0.1)
    assert got[('cstar', 'teacher', 'genre', 't2t')] == ('LEVEL', 0.1)
    assert got[('cstar', 'plain', 'style', 'i2t')] == ('BELOW', 0.04)
    assert got[('cstar', 'teacher', 'style', 'i2t')] == ('BELOW', 0.04)
    assert got[('late', 'plain', 'emotion', 't2i')] == ('BELOW', 0.01)
    assert got[('late', 'teacher', 'genre', 't2t')] == ('BELOW', 0.01)


def test_each_row_is_read_at_its_own_pick_and_head(tmp_path):
    setup(tmp_path)
    cells, _ = run()
    for c in cells:
        if c['cell'] == 'label':
            assert c['candidate_readout'] == PICK[c['candidate']] and c['comparator_readout'] == PICK[c['comparator']]
            assert c['head'] == c['aspect']
            assert c['sc'] == p.SC_OF[(c['candidate'], c['group'])] and c['kind'] == p.BAR_KIND[c['group']]
            assert c['candidate'] in ('cstar', 'late') and c['n_units'] == 3
        else:
            assert c['candidate'] == 'late' and c['comparator'] == 'plain' and c['aspect'] is None
            assert (c['head'], c['candidate_readout'], c['comparator_readout']) == (
                PAIR_LATE['head'], PAIR_LATE['readout'], PAIR_PLAIN['readout'])
            assert c['seed_range'] is None and c['outcome'] is None and c['sc'] in ('SC3', 'SC4')
    assert {c['direction'] for c in cells if c['cell'] == 'pair'} == {'t2i', 'i2t'}
    # distractor read-outs (0.0 for the candidate, 0.99 for comparators) would have changed every outcome
    assert {c['outcome'] for c in cells if c['cell'] == 'label'} == {'WIN'}


# ----------------------------------------------------------------------------------- hand values
def write_ap(group, row, head, seed, aspect, direction, readout, ap, pid):
    path = p.score_path(HELD[group], p.BAR_KIND[group], row, head, seed, direction, readout)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, q_pid=np.asarray(pid, dtype=np.int64), first_rank=np.full(len(pid), np.nan, dtype=np.float32),
             **{f'ap__{aspect}': np.asarray(ap, dtype=np.float32)})


def test_hand_point_value_masked_queries_and_units(tmp_path):
    pid = [0, 0, 1, 2]
    nan = np.nan
    for seed in p.SEEDS:        # seed s adds s/16 to the scored values
        write_ap('art', 'cstar', 'emotion', seed, 'emotion', 't2i', 'center',
                 [0.25 + seed / 16, nan, nan, 0.5 + seed / 16], pid)
    write_ap('art', 'plain', 'all', 'x', 'emotion', 't2i', 'r0', [0.25, nan, nan, 0.125], pid)
    # By hand: scored queries are 0 and 3 (paintings 0 and 2). Candidate over five seeds:
    # (0.25 + 0.5)/2 + (0 + 1 + 2 + 3 + 4)/5/16 = 0.375 + 0.125 = 0.5. Plain: (0.25 + 0.125)/2 = 0.1875.
    # Point value 0.5 - 0.1875 = 0.3125.
    got = oc.label_cell('art', 'cstar', 'plain', 'emotion', 't2i', stage1_record('art'), DRAWS)
    assert got['point'] == pytest.approx(0.3125, abs=1e-12)
    assert (got['n_query'], got['n_units']) == (2, 2)
    assert got['lower'] <= got['point'] <= got['upper']


def test_differing_scored_mask_between_arms_crashes(tmp_path):
    pid, nan = [0, 0, 1, 2], np.nan
    for seed in p.SEEDS:
        write_ap('art', 'cstar', 'emotion', seed, 'emotion', 't2i', 'center', [0.25, 0.5, nan, 0.5], pid)
    write_ap('art', 'plain', 'all', 'x', 'emotion', 't2i', 'r0', [0.25, nan, nan, 0.125], pid)
    with pytest.raises(AssertionError, match='scored'):
        oc.label_cell('art', 'cstar', 'plain', 'emotion', 't2i', stage1_record('art'), DRAWS)


def test_differing_scored_mask_between_seeds_crashes(tmp_path):
    pid, nan = [0, 0, 1, 2], np.nan
    for seed in p.SEEDS:
        write_ap('art', 'cstar', 'emotion', seed, 'emotion', 't2i', 'center',
                 [0.25, nan, nan, 0.5] if seed < 4 else [0.25, nan, 0.5, 0.5], pid)
    write_ap('art', 'plain', 'all', 'x', 'emotion', 't2i', 'r0', [0.25, nan, nan, 0.125], pid)
    with pytest.raises(AssertionError, match='scored'):
        oc.label_cell('art', 'cstar', 'plain', 'emotion', 't2i', stage1_record('art'), DRAWS)


def test_differing_q_pid_between_arms_crashes(tmp_path):
    for seed in p.SEEDS:
        write_ap('art', 'cstar', 'emotion', seed, 'emotion', 't2i', 'center', [0.5] * 4, [0, 0, 1, 2])
    write_ap('art', 'plain', 'all', 'x', 'emotion', 't2i', 'r0', [0.25] * 4, [0, 1, 1, 2])
    with pytest.raises(AssertionError, match='queries'):
        oc.label_cell('art', 'cstar', 'plain', 'emotion', 't2i', stage1_record('art'), DRAWS)


def write_rank(group, row, head, seed, direction, readout, ranks, pid):
    path = p.score_path(HELD[group], p.BAR_KIND[group], row, head, seed, direction, readout)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, q_pid=np.asarray(pid, dtype=np.int64), first_rank=np.asarray(ranks, dtype=np.float32))


def test_pair_nan_rank_is_a_miss_and_every_query_counts(tmp_path):
    pid = [0, 0, 1, 2]
    for seed in p.SEEDS:
        write_rank('art', 'late', 'style', seed, 't2i', 'center_k10', [1, np.nan, 1, 1], pid)
    write_rank('art', 'plain', 'all', 'x', 't2i', 'center', [1, 11, 1, 10], pid)
    got = oc.pair_cell('art', 't2i', stage1_record('art'), DRAWS)
    # rank 11 and NaN are both misses, rank 10 is a hit: both arms 3 of 4, and the NaN query stays in
    assert got['n_query'] == 4 and got['point'] == 0.0 and got['n_units'] == 3
    assert got['above_limit'] is True and got['outcome'] is None


def test_pair_lower_bound_exactly_at_the_limit_is_not_q(tmp_path, monkeypatch):
    monkeypatch.setitem(p.N_PAINTINGS, 'art_held', 2)
    monkeypatch.setitem(p.N_CAPTIONS, 'art_held', 8)
    pid = [0] * 4 + [1] * 4              # identical paintings: every draw gives 3/4 - 1 = -0.25 exactly
    for seed in p.SEEDS:
        write_rank('art', 'late', 'style', seed, 't2i', 'center_k10', [1, 1, 1, np.nan] * 2, pid)
    write_rank('art', 'plain', 'all', 'x', 't2i', 'center', [1] * 8, pid)
    monkeypatch.setattr(p, 'PAIR_LIMIT', -0.25)
    got = oc.pair_cell('art', 't2i', stage1_record('art'), p.N_DRAWS)
    assert (got['lower'], got['upper']) == (-0.25, -0.25) and got['above_limit'] is False
    monkeypatch.setattr(p, 'PAIR_LIMIT', -0.2500001)
    assert oc.pair_cell('art', 't2i', stage1_record('art'), DRAWS)['above_limit'] is True


def test_spec_constants_the_verdicts_stand_on():
    assert (p.PAIR_LIMIT, p.N_DRAWS, p.TEACHER_HOLDS, p.TEACHER_QUALIFIED) == (-0.02, 10000, 5, 3)
    assert len(oc.gate_i_keys()) == 39 and 'mm_test/narrative/default/plain' in oc.gate_i_keys()


# ------------------------------------------------------------------------------------ §6 inconclusive lines
def baseline(tmp_path):
    setup(tmp_path)
    return run()


def cells_of(cells, group):
    return [c for c in cells if c['group'] == group]


def void_keys(outcomes, what):
    return sorted(v['key'] for v in outcomes['voided'] if v['what'] == what)


def assert_inconclusive(outcomes, scs):
    for sc in scs:
        assert outcomes['verdicts'][sc] == dict(P=None, T=None, Q=None, verdict='inconclusive')
    assert void_keys(outcomes, 'verdict') == sorted(scs)
    for v in outcomes['voided']:
        assert set(v) == {'what', 'key', 'why'} and v['why']


def assert_set_untouched(tmp_path, cells, outcomes, base, group, scs):
    """One set void leaves the other's verdicts, cells and claims as they were (D7)."""
    assert cells_of(cells, group) == cells_of(base[0], group)
    for sc in scs:
        assert outcomes['verdicts'][sc] == base[1]['verdicts'][sc]


def status_with(**changes):
    st = good_status()
    for key, value in changes.items():
        st[key] = value
    return st


BAR_STORES = [('art', 'art_held/caption/emotion_focus/teacher'), ('art', 'art_held/caption/genre_word/teacher'),
              ('mm', 'mm_val/narrative/style_word/teacher'), ('mm', 'mm_test/narrative/emotion_focus/teacher'),
              ('mm', 'mm_test/narrative/default/plain')]
OTHER = {'art': ('SC2', 'SC4'), 'mm': ('SC1', 'SC3')}
OWN = {'art': ('SC1', 'SC3'), 'mm': ('SC2', 'SC4')}
SET_OF = {'art': 'art_held', 'mm': 'mm_test'}


@pytest.mark.parametrize('group,key', BAR_STORES)
def test_line_a_gate_i_false_on_a_bar_store_voids_that_set(tmp_path, group, key):
    base = baseline(tmp_path)
    gate = good_status()['gate_i']
    gate[key] = False
    write_record('status.json', status_with(gate_i=gate))
    cells, outcomes = run()
    assert_inconclusive(outcomes, OWN[group])
    assert void_keys(outcomes, 'cell_claims') == [SET_OF[group]] and void_keys(outcomes, 'diagnostic') == []
    assert {c['claim_voided'] for c in cells_of(cells, group)} == {True}
    assert len(cells_of(cells, group)) == 38
    assert_set_untouched(tmp_path, cells, outcomes, base, {'art': 'mm', 'mm': 'art'}[group], OTHER[group])


def test_line_a_an_absent_bar_store_key_counts_as_failed(tmp_path):
    baseline(tmp_path)
    gate = good_status()['gate_i']
    del gate['mm_val/narrative/genre_word/teacher']
    write_record('status.json', status_with(gate_i=gate))
    _, outcomes = run()
    assert_inconclusive(outcomes, ('SC2', 'SC4'))


@pytest.mark.parametrize('key', ['art_held/caption/none/roberta', 'mm_test/formal/default/plain',
                                 'mm_val/unified/emotion_focus/teacher', 'mm_test/historical/genre_word/teacher'])
def test_line_b_gate_i_false_on_another_store_is_a_diagnostic_only(tmp_path, key):
    base = baseline(tmp_path)
    gate = good_status()['gate_i']
    gate[key] = False
    write_record('status.json', status_with(gate_i=gate))
    cells, outcomes = run()
    assert outcomes['verdicts'] == base[1]['verdicts'] and cells == base[0]
    assert [(v['what'], v['key']) for v in outcomes['voided']] == [('diagnostic', key)]


@pytest.mark.parametrize('gate,group', [('gate_ii', 'art'), ('gate_ii', 'mm'), ('gate_iv', 'art'), ('gate_iv', 'mm')])
@pytest.mark.parametrize('how', ['false', 'absent'])
def test_line_c_gate_ii_or_iv_false_voids_that_set(tmp_path, gate, group, how):
    base = baseline(tmp_path)
    st = good_status()
    if how == 'false':
        st[gate][group] = False
    else:
        del st[gate][group]
    write_record('status.json', st)
    cells, outcomes = run()
    assert_inconclusive(outcomes, OWN[group])
    assert void_keys(outcomes, 'cell_claims') == [SET_OF[group]]
    assert {c['claim_voided'] for c in cells_of(cells, group)} == {True}
    assert_set_untouched(tmp_path, cells, outcomes, base, {'art': 'mm', 'mm': 'art'}[group], OTHER[group])


def test_line_d_gate_iii_false_voids_all_four_and_all_claims(tmp_path):
    baseline(tmp_path)
    write_record('status.json', status_with(gate_iii=False))
    cells, outcomes = run()
    assert_inconclusive(outcomes, ('SC1', 'SC2', 'SC3', 'SC4'))
    assert void_keys(outcomes, 'cell_claims') == ['art_held', 'mm_test']
    assert len(cells) == 76 and {c['claim_voided'] for c in cells} == {True}


def test_line_d_gate_iii_absent_counts_as_failed(tmp_path):
    baseline(tmp_path)
    st = good_status()
    del st['gate_iii']
    write_record('status.json', st)
    _, outcomes = run()
    assert_inconclusive(outcomes, ('SC1', 'SC2', 'SC3', 'SC4'))


def test_line_e_missing_stage1_record_voids_that_set_and_does_not_crash_the_other(tmp_path):
    base = baseline(tmp_path)
    p.record_path('stage1_art.json').unlink()
    cells, outcomes = run()
    assert_inconclusive(outcomes, ('SC1', 'SC3'))
    assert cells_of(cells, 'art') == []
    assert void_keys(outcomes, 'cell_claims') == ['art_held']
    assert 'stage 1 record missing' in next(v['why'] for v in outcomes['voided'] if v['what'] == 'cell_claims')
    assert_set_untouched(tmp_path, cells, outcomes, base, 'mm', ('SC2', 'SC4'))


def test_line_e_stage1_flag_false_voids_that_set(tmp_path):
    base = baseline(tmp_path)
    write_record('status.json', status_with(stage1=dict(art=True, mm=False)))
    cells, outcomes = run()
    assert_inconclusive(outcomes, ('SC2', 'SC4'))
    assert_set_untouched(tmp_path, cells, outcomes, base, 'art', ('SC1', 'SC3'))


def test_status_and_uncommitted_records_are_not_swallowed(tmp_path, monkeypatch):
    baseline(tmp_path)
    p.record_path('status.json').unlink()
    with pytest.raises(FileNotFoundError):
        run()
    write_record('status.json', good_status())

    def refuse(name):
        raise RuntimeError('not committed')
    monkeypatch.setattr(p, 'read_committed_record', refuse)
    with pytest.raises(RuntimeError):
        run()


def path_of(group, row, head, seed, direction, readout):
    return p.score_path(HELD[group], p.BAR_KIND[group], row, head, seed, direction, readout)


def test_line_f_missing_candidate_file_voids_only_that_candidates_verdict(tmp_path):
    base = baseline(tmp_path)
    for cand, gone, kept in (('cstar', 'SC1', 'SC3'), ('late', 'SC3', 'SC1')):
        path_of('art', cand, 'genre', 2, 'i2t', 'center').rename(tmp_path / 'held_back.npz')
        cells, outcomes = run()
        assert_inconclusive(outcomes, (gone,))
        assert outcomes['verdicts'][kept] == base[1]['verdicts'][kept]          # c* and late fusion never touch
        assert void_keys(outcomes, 'cell_claims') == []
        assert {c['claim_voided'] for c in cells} == {False}
        assert len(cells) == 76 - 2                                            # the two cells of that candidate
        (tmp_path / 'held_back.npz').rename(path_of('art', cand, 'genre', 2, 'i2t', 'center'))
        assert run()[1] == base[1]


@pytest.mark.parametrize('row,head,readout', [('plain', 'all', 'r0'), ('teacher', 'style', 'center_k10')])
def test_line_f_missing_comparator_file_voids_both_verdicts_of_the_set(tmp_path, row, head, readout):
    base = baseline(tmp_path)
    path_of('mm', row, head, 'x', 't2t', readout).unlink()
    cells, outcomes = run()
    assert_inconclusive(outcomes, ('SC2', 'SC4'))
    assert_set_untouched(tmp_path, cells, outcomes, base, 'art', ('SC1', 'SC3'))


@pytest.mark.parametrize('row,head,seed,readout', [('late', 'style', 3, 'center_k10'), ('plain', 'all', 'x', 'center')])
def test_line_f_missing_pair_cell_voids_late_fusion_only(tmp_path, row, head, seed, readout):
    base = baseline(tmp_path)
    path_of('art', row, head, seed, 'i2t', readout).unlink()
    cells, outcomes = run()
    assert_inconclusive(outcomes, ('SC3',))
    assert outcomes['verdicts']['SC1'] == base[1]['verdicts']['SC1']             # c* reads no pair cell
    assert len([c for c in cells_of(cells, 'art') if c['cell'] == 'pair']) == 1


def test_line_f_missing_file_of_a_voided_verdict_is_not_an_error(tmp_path):
    baseline(tmp_path)
    path_of('art', 'cstar', 'emotion', 0, 't2i', 'center').unlink()
    path_of('art', 'plain', 'all', 'x', 't2i', 'r0').unlink()
    write_record('status.json', status_with(gate_ii=dict(art=False, mm=True)))
    cells, outcomes = run()
    assert_inconclusive(outcomes, ('SC1', 'SC3'))
    assert {c['claim_voided'] for c in cells_of(cells, 'art')} == {True}
    assert outcomes['verdicts']['SC2']['verdict'] == 'holds'


def test_two_sets_void_in_different_ways_and_neither_changes_the_other(tmp_path):
    base = baseline(tmp_path)
    st = status_with(gate_iv=dict(art=False, mm=True))
    st['gate_i']['mm_test/narrative/default/plain'] = False
    write_record('status.json', st)
    _, outcomes = run()
    assert_inconclusive(outcomes, ('SC1', 'SC2', 'SC3', 'SC4'))
    assert void_keys(outcomes, 'cell_claims') == ['art_held', 'mm_test']
    assert base[1]['voided'] == []


# --------------------------------------------------------------------------------------- the files
def test_write_is_sorted_deterministic_and_has_the_index_fields(tmp_path):
    setup(tmp_path, art=dict(nT=4))
    oc.main(draws=DRAWS)
    folder = tmp_path / 'stage2'
    first = {n: (folder / n).read_bytes() for n in ('cells.json', 'outcomes.json')}
    cells, outcomes = json.loads(first['cells.json']), json.loads(first['outcomes.json'])
    assert first['outcomes.json'].decode() == json.dumps(outcomes, sort_keys=True, indent=2, allow_nan=False) + '\n'
    assert set(outcomes) == {'verdicts', 'voided'} and set(outcomes['verdicts']) == {'SC1', 'SC2', 'SC3', 'SC4'}
    assert outcomes['verdicts']['SC1']['verdict'] == 'qualified'
    label = [c for c in cells if c['cell'] == 'label'][0]
    assert set(label) == {'sc', 'group', 'kind', 'candidate', 'comparator', 'cell', 'aspect', 'direction',
                          'candidate_readout', 'comparator_readout', 'head', 'point', 'lower', 'upper',
                          'seed_range', 'outcome', 'n_query', 'n_units', 'claim_voided'}
    oc.main(draws=DRAWS)
    assert {n: (folder / n).read_bytes() for n in first} == first


# -------------------------------------------------------------- review fault regressions
def test_fails_leaves_cell_claims_standing(tmp_path):
    setup(tmp_path, art=dict(nP=8))
    cells, outcomes = run()
    assert outcomes['verdicts']['SC1']['verdict'] == outcomes['verdicts']['SC3']['verdict'] == 'fails'
    art = [c for c in cells if c['group'] == 'art']
    assert len(art) == 38 and {c['claim_voided'] for c in art} == {False}


def wide(monkeypatch, n=8):
    monkeypatch.setattr(p, 'SMOKE', n)
    monkeypatch.setitem(p.N_PAINTINGS, 'art_held', n)
    monkeypatch.setitem(p.N_CAPTIONS, 'art_held', n)
    return list(range(n))


def test_outcome_reads_the_lower_bound_not_the_point_or_upper(tmp_path, monkeypatch):
    pid = wide(monkeypatch)
    d = np.array([.5, .5, .5, -.25, -.25, -.25, .125, .125])
    for seed in p.SEEDS:
        write_ap('art', 'cstar', 'emotion', seed, 'emotion', 't2i', 'center', 0.25 + (d > 0) * d + 0.25, pid)
    write_ap('art', 'plain', 'all', 'x', 'emotion', 't2i', 'r0', 0.5 - (d < 0) * d, pid)
    got = oc.label_cell('art', 'cstar', 'plain', 'emotion', 't2i', stage1_record('art'), 2000)
    assert got['lower'] < 0 < got['point'] < got['upper'] and got['outcome'] != 'WIN'


def test_q_reads_the_lower_bound_not_the_point(tmp_path, monkeypatch):
    pid = wide(monkeypatch)
    for seed in p.SEEDS:
        write_rank('art', 'late', 'style', seed, 't2i', 'center_k10', [1, 1, 1, 1, np.nan, np.nan, 1, 1], pid)
    write_rank('art', 'plain', 'all', 'x', 't2i', 'center', [np.nan, np.nan, 1, 1, 1, 1, 1, 1], pid)
    got = oc.pair_cell('art', 't2i', stage1_record('art'), 2000)
    assert got['point'] == 0.0 and got['lower'] < p.PAIR_LIMIT and got['above_limit'] is False


def varied_record(group='art'):
    rec = stage1_record(group)
    kind = p.BAR_KIND[group]
    for i, (a, d) in enumerate(CELLS):
        n = p.ASPECTS.index(a) + p.DIRECTIONS.index(d)
        rec['picks'][kind]['cstar'][a][d] = rec['picks'][kind]['late'][a][d] = p.READOUTS[n % 3]
        rec['picks'][kind]['plain'][a][d] = rec['picks'][kind]['teacher'][a][d] = p.READOUTS[(n + 1) % 3]
    rec['pair_picks'][kind] = {
        't2i': dict(late=dict(head='style', readout='center_k10'), plain=dict(readout='center')),
        'i2t': dict(late=dict(head='genre', readout='r0'), plain=dict(readout='center_k10'))}
    return rec


def test_each_cell_reads_its_own_pick_when_picks_differ_by_cell(tmp_path):
    setup(tmp_path)
    rec = varied_record()
    for a, d in CELLS:
        for cand in ('cstar', 'late'):
            for comp in ('plain', 'teacher'):
                got = oc.label_cell('art', cand, comp, a, d, rec, 20)
                assert got['candidate_readout'] == rec['picks']['caption'][cand][a][d]
                assert got['comparator_readout'] == rec['picks']['caption'][comp][a][d]
    for d in p.PAIR_DIRECTIONS:
        got = oc.pair_cell('art', d, rec, 20)
        want = rec['pair_picks']['caption'][d]
        assert (got['head'], got['candidate_readout'], got['comparator_readout']) == (
            want['late']['head'], want['late']['readout'], want['plain']['readout'])


def test_pair_mean_uses_all_five_seeds(tmp_path):
    pid = [0, 0, 1, 2]
    for seed in p.SEEDS:
        write_rank('art', 'late', 'style', seed, 't2i', 'center_k10', [1 if seed < 4 else 11] * 4, pid)
    write_rank('art', 'plain', 'all', 'x', 't2i', 'center', [1] * 4, pid)
    got = oc.pair_cell('art', 't2i', stage1_record('art'), 20)
    assert got['point'] == pytest.approx(-0.2, abs=1e-12)


def test_pair_nan_in_plain_stays_a_miss(tmp_path):
    pid = [0, 0, 1, 2]
    for seed in p.SEEDS:
        write_rank('art', 'late', 'style', seed, 't2i', 'center_k10', [1] * 4, pid)
    write_rank('art', 'plain', 'all', 'x', 't2i', 'center', [1, np.nan, 1, 1], pid)
    got = oc.pair_cell('art', 't2i', stage1_record('art'), 20)
    assert got['n_query'] == 4 and got['point'] == 0.25


def test_uncommitted_stage1_is_refused_not_read_as_missing(tmp_path, monkeypatch):
    baseline(tmp_path)
    real = p.read_committed_record

    def refuse(name):
        if name == 'stage1_mm.json':
            raise RuntimeError('not committed')
        return real(name)
    monkeypatch.setattr(p, 'read_committed_record', refuse)
    with pytest.raises(RuntimeError):
        run()


def test_defaults_are_the_specs_draws():
    assert inspect.signature(oc.run).parameters['draws'].default == 10000
    assert inspect.signature(oc.main).parameters['draws'].default == 10000


@pytest.mark.parametrize('sc', ['SC1', 'SC2', 'SC3', 'SC4'])
def test_c13_voids_only_the_named_verdict_and_its_cell_claims(tmp_path, sc):
    base = baseline(tmp_path)
    write_record('status.json', status_with(c13_void=[sc]))
    cells, outcomes = run()
    assert_inconclusive(outcomes, [sc])
    assert void_keys(outcomes, 'cell_claims') == []
    assert 'C13' in outcomes['voided'][0]['why']
    assert len(cells) == len(base[0])
    for cell, original in zip(cells, base[0]):
        assert cell == dict(original, claim_voided=cell['sc'] == sc)
    for other in outcomes['verdicts'].keys() - {sc}:
        assert outcomes['verdicts'][other] == base[1]['verdicts'][other]


def test_c13_list_combines_with_a_set_gate_void(tmp_path):
    base = baseline(tmp_path)
    write_record('status.json', status_with(c13_void=['SC2', 'SC3'], gate_iv=dict(art=False, mm=True)))
    cells, outcomes = run()
    assert_inconclusive(outcomes, ['SC1', 'SC2', 'SC3'])
    assert void_keys(outcomes, 'cell_claims') == ['art_held']
    assert outcomes['verdicts']['SC4'] == base[1]['verdicts']['SC4']
    assert all(c['claim_voided'] == (c['sc'] != 'SC4') for c in cells)


def test_empty_c13_list_leaves_verdicts_and_claims_standing(tmp_path):
    base = baseline(tmp_path)
    write_record('status.json', status_with(c13_void=[]))
    assert run() == base


@pytest.mark.parametrize('group', ['art', 'mm'])
@pytest.mark.parametrize('direction', p.DIRECTIONS)
@pytest.mark.parametrize('smoke', [2, 5])
def test_smoke_load_uses_selected_paintings_and_variable_caption_count(tmp_path, monkeypatch, group, direction, smoke):
    monkeypatch.setattr(p, 'SMOKE', smoke)
    n_i = min(smoke, p.N_PAINTINGS[HELD[group]])
    # All captions of the selected paintings remain; their count is not the full set's count.
    pid = [0, 0, 1, 1, 1] if group == 'art' and direction != 'i2t' else list(range(n_i))
    write_ap(group, 'plain', 'all', 'x', 'emotion', direction, 'r0', [0.5] * len(pid), pid)
    got_pid, _ = oc._load(HELD[group], p.BAR_KIND[group], 'plain', 'all', 'x', direction, 'r0',
                         'emotion', 'comparator')
    np.testing.assert_array_equal(got_pid, pid)


@pytest.mark.parametrize('direction', ['t2i', 't2t'])
def test_full_run_still_checks_the_caption_count(tmp_path, monkeypatch, direction):
    monkeypatch.setattr(p, 'SMOKE', 0)
    write_ap('art', 'plain', 'all', 'x', 'emotion', direction, 'r0', [0.5] * 3, [0, 1, 2])
    with pytest.raises(AssertionError, match='query IDs'):
        oc._load('art_held', 'caption', 'plain', 'all', 'x', direction, 'r0', 'emotion', 'comparator')


@pytest.mark.parametrize('group', ['art', 'mm'])
@pytest.mark.parametrize('voided', [True, False])
def test_bad_score_assertion_is_skipped_only_for_a_gate_voided_set(tmp_path, group, voided):
    base = baseline(tmp_path)
    write_record('status.json', status_with(gate_iv={g: not (voided and g == group) for g in OWN}))
    path = path_of(group, 'plain', 'all', 'x', 'i2t', 'r0')
    np.savez(path, q_pid=np.arange(99, dtype=np.int64), first_rank=np.ones(99, dtype=np.float32),
             **{f'ap__{a}': np.full(99, 0.5, dtype=np.float32) for a in p.ASPECTS})
    if not voided:
        with pytest.raises(AssertionError, match='query IDs'):
            run()
        return
    cells, outcomes = run()
    assert_inconclusive(outcomes, OWN[group])
    assert void_keys(outcomes, 'cell_claims') == [SET_OF[group]]
    assert all(c['claim_voided'] for c in cells_of(cells, group))
    assert not any(c['comparator'] == 'plain' and c['direction'] == 'i2t' for c in cells_of(cells, group)
                   if c['cell'] == 'label')
    assert any('query IDs' in v['why'] for v in outcomes['voided'])
    other = 'mm' if group == 'art' else 'art'
    assert_set_untouched(tmp_path, cells, outcomes, base, other, OTHER[group])


def test_extraction_store_keys_equal_the_verdict_gate_keys():
    import tcl_extract

    assert [r[0] for r in tcl_extract.stores_to_check()] == oc.gate_i_keys()
