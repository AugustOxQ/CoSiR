"""Stage 2 cells, outcomes and verdicts: text-claim design §3.4 (interval, outcomes, pair cells), §3.5, §6.

Reads stage 1 (records) and status through tcl_paths.read_committed_record, and the held score files. Writes
OUT/stage2/cells.json and OUT/stage2/outcomes.json. A difference between arms in queries or scored mask crashes; only
the lines of §6 void a verdict.
"""
import json

import numpy as np

import tcl_paths as P
from tcl_metrics import outcome, painting_interval

HELD = {'art': 'art_held', 'mm': 'mm_test'}
GROUPS = ('art', 'mm')
CANDIDATES = ('cstar', 'late')
COMPARATORS = ('plain', 'teacher')          # the rows whose stores a bar cell reads (gate (i) bar stores)
N_CELLS = len(P.ASPECTS) * len(P.DIRECTIONS)    # 9
PAIR_HIT_RANK = 10


class MissingScore(Exception):
    """A score file a bar cell reads is not there. role: 'candidate' or 'comparator'."""

    def __init__(self, path, role):
        super().__init__(str(path))
        self.path, self.role = path, role


def gate_i_keys():
    """The 39 gate (i) store keys `<set>/<kind>/<instruction>/<row>` (ticket 03's stores_to_check)."""
    keys = [f'art_held/{P.CAPTION}/{P.INSTRUCTIONS[a]}/teacher' for a in P.ASPECTS]
    keys.append(f'art_held/{P.CAPTION}/none/roberta')
    keys += [f'mm_test/{k}/default/plain' for k in P.KINDS]
    keys += [f'{s}/{k}/{i}/teacher' for s in ('mm_val', 'mm_test') for k in P.KINDS for i in P.INSTRUCTIONS.values()]
    return keys


def is_bar_store(key, group):
    """A store a bar cell of the group reads: its kind is the group's bar kind, its row plain or teacher, in either
    of the group's sets (validation stores feed picks and means)."""
    set_name, kind, _, row = key.split('/')
    return P.GROUP[set_name] == group and kind == P.BAR_KIND[group] and row in COMPARATORS


# ---------------------------------------------------------------------------------------------------- score files
def _n_queries(set_name, direction, n_caption):
    if P.GROUP[set_name] == 'art' and direction != 'i2t':
        # Smoke keeps every caption of the selected paintings, so its count varies.
        return n_caption if P.SMOKE else P.N_CAPTIONS[set_name]
    return min(P.SMOKE or P.N_PAINTINGS[set_name], P.N_PAINTINGS[set_name])


def _load(set_name, kind, row, head, seed, direction, readout, aspect, role):
    """q_pid and either the AP of `aspect` (aspect given) or the first-hit rank (aspect None) of one ranking."""
    path = P.score_path(set_name, kind, row, head, seed, direction, readout)
    if not path.exists():
        raise MissingScore(path, role)
    with np.load(path, allow_pickle=False) as score:
        column = 'first_rank' if aspect is None else f'ap__{aspect}'
        assert {'q_pid', column} <= set(score.files), f'missing score columns: {path}'
        pid, values = score['q_pid'], score[column]
    n_q = _n_queries(set_name, direction, len(pid))
    n_i = min(P.SMOKE or P.N_PAINTINGS[set_name], P.N_PAINTINGS[set_name])
    assert pid.shape == (n_q,) and pid.dtype == np.int64, f'query IDs: {path}'
    assert ((pid >= 0) & (pid < n_i)).all(), f'painting positions: {path}'
    if direction == 'i2t' or P.GROUP[set_name] == 'mm':
        assert np.array_equal(pid, np.arange(n_i)), f'painting order: {path}'
    assert values.shape == pid.shape and values.dtype == np.float32, f'{column} shape/dtype: {path}'
    finite = np.isfinite(values)
    assert not np.isinf(values).any(), f'infinite {column}: {path}'
    if aspect is None:
        assert ((values[finite] >= 1) & (values[finite] == np.floor(values[finite]))).all(), f'ranks: {path}'
    else:
        assert finite.any() and ((values[finite] >= 0) & (values[finite] <= 1)).all(), f'AP: {path}'
    return pid, values


def _interval(cand, comp, pid, draws):
    """cand float64 [seeds, n], comp float64 [n], pid int64 [n]: scored queries only, one unit per q_pid."""
    point, lower, upper, n_units = painting_interval(cand, comp, pid, draws=draws)
    return dict(point=point, lower=lower, upper=upper, n_units=n_units, n_query=int(len(pid)))


def _row(group, kind, candidate, comparator, cell, aspect, direction, cand_readout, comp_readout, head, seed_range,
         result, outcome_word):
    return dict(sc=P.SC_OF[(candidate, group)], group=group, kind=kind, candidate=candidate, comparator=comparator,
                cell=cell, aspect=aspect, direction=direction, candidate_readout=cand_readout,
                comparator_readout=comp_readout, head=head, seed_range=seed_range, outcome=outcome_word, **result)


def label_cell(group, candidate, comparator, aspect, direction, record, draws):
    """One label cell: candidate (head = aspect, seeds 0 to 4) against a comparator, each at its own picked read-out."""
    set_name, kind = HELD[group], P.BAR_KIND[group]
    c_read = record['picks'][kind][candidate][aspect][direction]
    m_read = record['picks'][kind][comparator][aspect][direction]
    seed_range = float(record['ranges'][kind][candidate][aspect][direction])      # the candidate's, not the comparator's
    m_head = aspect if comparator in P.HEAD_ROWS else 'all'
    pid, m_ap = _load(set_name, kind, comparator, m_head, 'x', direction, m_read, aspect, 'comparator')
    mask = np.isfinite(m_ap)
    assert mask.any()
    rows = []
    for seed in P.SEEDS:
        c_pid, c_ap = _load(set_name, kind, candidate, aspect, seed, direction, c_read, aspect, 'candidate')
        assert np.array_equal(c_pid, pid), f'arms differ in queries: {candidate} seed {seed} {aspect} {direction}'
        assert np.array_equal(np.isfinite(c_ap), mask), f'arms differ in scored queries: {candidate} seed {seed}'
        rows.append(c_ap[mask].astype(np.float64))
    result = _interval(np.stack(rows), m_ap[mask].astype(np.float64), pid[mask], draws)
    return _row(group, kind, candidate, comparator, 'label', aspect, direction, c_read, m_read, aspect, seed_range,
                result, outcome(result['lower'], seed_range))


def _hits(ranks):
    return np.where(np.isnan(ranks), 0.0, ranks <= PAIR_HIT_RANK).astype(np.float64)       # NaN: no hit, a miss


def pair_cell(group, direction, record, draws):
    """Late fusion against plain Qwen at pair R@10 in one direction: every query, at the pair picks."""
    set_name, kind = HELD[group], P.BAR_KIND[group]
    picks = record['pair_picks'][kind][direction]
    head, c_read, m_read = picks['late']['head'], picks['late']['readout'], picks['plain']['readout']
    pid, m_rank = _load(set_name, kind, 'plain', 'all', 'x', direction, m_read, None, 'comparator')
    rows = []
    for seed in P.SEEDS:
        c_pid, c_rank = _load(set_name, kind, 'late', head, seed, direction, c_read, None, 'candidate')
        assert np.array_equal(c_pid, pid), f'arms differ in queries: late seed {seed} {direction}'
        rows.append(_hits(c_rank))
    result = _interval(np.stack(rows), _hits(m_rank), pid, draws)
    row = _row(group, kind, 'late', 'plain', 'pair', None, direction, c_read, m_read, head, None, result, None)
    row['above_limit'] = bool(result['lower'] > P.PAIR_LIMIT)           # strict: Q needs a lower bound above the limit
    return row


# ---------------------------------------------------------------------------------------------------- verdicts
def verdict_word(n_plain, n_teacher, q):
    """§6. q None: no pair clause (c*). D4: 8 of 9 over plain Qwen fails."""
    ok = n_plain == N_CELLS and q is not False
    if ok and n_teacher >= P.TEACHER_HOLDS:
        return 'holds'
    if ok and n_teacher >= P.TEACHER_QUALIFIED:
        return 'qualified'
    return 'fails'


def _group_voids(status):
    """({group: [reasons]}, diagnostics) from status.json; a key absent counts as failed (§6, lines a to e)."""
    why = {g: [] for g in GROUPS}
    if status.get('gate_iii') is not True:
        for g in GROUPS:
            why[g].append('gate (iii) false: all verdicts and cell claims')
    gate_i = status.get('gate_i', {})
    diagnostics = []
    for key in sorted(set(gate_i_keys()) | set(gate_i)):
        if gate_i.get(key) is True:
            continue
        hit = False
        for g in GROUPS:
            if is_bar_store(key, g):
                why[g].append(f'gate (i) false on bar store {key}')
                hit = True
        if not hit:
            diagnostics.append(dict(what='diagnostic', key=key, why='gate (i) false on a store only §3.6 reads'))
    for g in GROUPS:
        for gate, label in (('gate_ii', 'gate (ii)'), ('gate_iv', 'gate (iv)')):
            if status.get(gate, {}).get(g) is not True:
                why[g].append(f'{label} false')
        if status.get('stage1', {}).get(g) is not True:
            why[g].append('stage 1 false')
    return why, diagnostics


def _read_stage1(group):
    try:
        record = P.read_committed_record(f'stage1_{group}.json')
    except FileNotFoundError:
        return None
    assert record['group'] == group
    return record


def run(draws=P.N_DRAWS):
    """(cells, outcomes) of both groups. Nothing is written."""
    status = P.read_committed_record('status.json')
    group_why, diagnostics = _group_voids(status)
    cells, why = [], {sc: [] for sc in P.SC_OF.values()}
    c13_void = status.get('c13_void', [])
    for sc in c13_void:
        why[sc].append('C13 difference over 1e-6 remains after tracing')
    for g in GROUPS:
        scs = [P.SC_OF[(c, g)] for c in CANDIDATES]
        record = _read_stage1(g)
        if record is None:
            group_why[g].append('stage 1 record missing')
        for sc in scs:
            why[sc] += group_why[g]
        if record is None:
            continue
        group_void = bool(group_why[g])
        for candidate in CANDIDATES:
            sc = P.SC_OF[(candidate, g)]
            jobs = [(f'label {c} {a} {d}', lambda c=c, a=a, d=d: label_cell(g, candidate, c, a, d, record, draws))
                    for c in COMPARATORS for a in P.ASPECTS for d in P.DIRECTIONS]
            if candidate == 'late':
                jobs += [(f'pair {d}', lambda d=d: pair_cell(g, d, record, draws)) for d in P.PAIR_DIRECTIONS]
            for name, job in jobs:
                try:
                    row = job()
                except MissingScore as miss:
                    if not group_void:
                        # a candidate's file voids its verdict; a comparator's, both verdicts of the set
                        for target in (scs if miss.role == 'comparator' and not name.startswith('pair') else [sc]):
                            if not why[target]:
                                why[target].append(f'score file missing: {miss.path} ({name})')
                    continue
                except AssertionError as error:
                    if not group_void:
                        raise
                    reason = f'invalid score cell {name}: {error}'
                    group_why[g].append(reason)
                    why[sc].append(reason)
                    continue
                row['claim_voided'] = group_void or sc in c13_void
                cells.append(row)
    outcomes = dict(verdicts={}, voided=[])
    for g in GROUPS:
        for candidate in CANDIDATES:
            sc = P.SC_OF[(candidate, g)]
            if why[sc]:
                outcomes['verdicts'][sc] = dict(P=None, T=None, Q=None, verdict='inconclusive')
                outcomes['voided'].append(dict(what='verdict', key=sc, why='; '.join(why[sc])))
                continue
            mine = [c for c in cells if c['sc'] == sc]
            n_plain = _count(mine, 'plain')
            n_teacher = _count(mine, 'teacher')
            q = None
            if candidate == 'late':
                pairs = [c for c in mine if c['cell'] == 'pair']
                assert len(pairs) == len(P.PAIR_DIRECTIONS)
                q = all(c['above_limit'] for c in pairs)
            outcomes['verdicts'][sc] = dict(P=n_plain, T=n_teacher, Q=q, verdict=verdict_word(n_plain, n_teacher, q))
        if group_why[g]:
            outcomes['voided'].append(dict(what='cell_claims', key=HELD[g], why='; '.join(group_why[g])))
    outcomes['voided'] += diagnostics
    return cells, outcomes


def _count(cells, comparator):
    own = [c for c in cells if c['cell'] == 'label' and c['comparator'] == comparator]
    assert len(own) == N_CELLS, f'{comparator}: {len(own)} label cells'
    return sum(c['outcome'] == 'WIN' for c in own)           # only WIN counts, over plain and over the teacher


def write(cells, outcomes):
    folder = P.OUT / 'stage2'
    folder.mkdir(parents=True, exist_ok=True)
    for name, obj in (('cells.json', cells), ('outcomes.json', outcomes)):
        (folder / name).write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n')


def main(draws=P.N_DRAWS):
    cells, outcomes = run(draws=draws)
    write(cells, outcomes)
    print({sc: v['verdict'] for sc, v in outcomes['verdicts'].items()}, f'voided: {len(outcomes["voided"])}')


if __name__ == '__main__':
    main()
