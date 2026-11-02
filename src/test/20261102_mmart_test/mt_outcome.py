"""r15 outcome code (ticket 06): cell outcomes, verdicts, SC4 claims, D17 voids and the licensed sentences.

Spec: docs/superpowers/specs/2026-11-02-mmart-test-design.md, §6 (whole), §3.4 "outcomes" (r14's, no multiplicity
correction), the table under §3.5 (D17), D3, D7. The cell rule and the verdict rule are r14's `bs_outcome` functions
(`cell_outcome`, `paintings_verdict`), imported, not copied.

CLI (no GPU, reads files only):
  python mt_outcome.py decide [--root OUT] [--out FILE]
    reads  OUT/stage2/cells_clu.json, cells_ret.json   cells written by the scorers (tickets 04, 05); diagnostics live
                                                       under OUT/diag and are never read
           records/status.json                         `mt_paths.record_path('status.json')`, see STATUS below
           records/cutsets.json                        same; only `test.n`, for <n> of the licensed sentences
    writes OUT/stage2/outcomes.json                    every cell with its outcome, the verdicts, the SC4 claims
  Under MT_SMOKE, `n_draws` may differ from the protocol's (2,000 clustering; 10,000 retrieval). Nothing else changes.

CELLS: data 'mmart_test' only; per encoder SC1 (3 aspects, clu), SC2 (3, ret), SC3 (3, clu), SC4 (3 clu + 3 ret) = 15,
60 in all. SC1 to SC3 are WIN-only bars: `r` null, outcome 'WIN' or 'not WIN'. SC4 carries the unrounded range `r`:
WIN (lo > 0), LEVEL (lo > -r), BELOW (strict).

STATUS (ticket 07 writes it; every key must be present, values 'ok' | 'failed'):
  {"gates": {"i_nearcopy", "iii", "vi": str;  "i_default", "ii", "iv": {enc: str};  "i_teacher", "v": {aspect: str}},
   "stage1": {"ranges": {enc: str}, "picks": {enc: str}, "teacher_pick": str}}
  `stage1.ranges` may also be one string (then it holds for every encoder).

SC4 cells of a voided encoder: `check_cell` refuses `r` null even when the cell is voided, so ticket 07 must write them with
their range when it exists, or leave them out (a missing cell is excused by the void).

D17 (cell status 'inconclusive' = the row is kept, no claim; verdict 'inconclusive' = never holds, qualified or fails):
  i_nearcopy, iii, i_default[qwen]       every cell and verdict of every encoder
  i_default[X], ii[X]                    every cell and the verdict of X
  ranges[X], picks[X]                    X's verdict, SC2 cells and SC4 cells (they use the ranges and the picks); X's SC1
                                         and SC3 cells keep their outcomes
  i_teacher[a], v[a]                     the SC4 cells of aspect a, all encoders; verdicts stand
  iv[X]                                  X's verdict and X's SC3 cells; its SC1, SC2, SC4 rows and SC4 claims stand
  vi                                     every verdict; every SC2 cell and SC4 retrieval cell; SC4 clustering claims stand
  teacher_pick                           every SC4 retrieval cell; verdicts stand. Derived: Qwen's ranges failed (the one
                                         teacher pick uses Qwen's ranges, D12), so no teacher pick can stand.
A required cell may be missing only when one of these rules voids that cell; otherwise it raises.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import mt_paths as MP

sys.path.insert(0, str(MP.BS_ROUND))
import bs_outcome as BO                                   # noqa: E402  r14: cell rule, verdict rule, aspect_list

assert tuple(BO.ASPECTS) == tuple(MP.ASPECTS)

ENCODERS, ASPECTS = MP.ENCODERS, MP.ASPECTS
WIN, LEVEL, BELOW, NOT_WIN = BO.WIN, BO.LEVEL, BO.BELOW, BO.NOT_WIN
HOLDS, QUALIFIED, FAILS, INCONCLUSIVE = BO.HOLDS, BO.QUALIFIED, BO.FAILS, BO.INCONCLUSIVE
GO_OF = BO.GO_OF
cell_outcome = BO.cell_outcome
encoder_verdict = BO.paintings_verdict                    # holds: nine WIN; qualified: SC1, SC2 all WIN; else fails
DATA = 'mmart_test'
TASKS = ('clu', 'ret')
TASK_OF_SC = {'SC1': ('clu',), 'SC2': ('ret',), 'SC3': ('clu',), 'SC4': ('clu', 'ret')}
CELL_KEYS = ('sc', 'encoder', 'aspect', 'task', 'data', 'candidate', 'comparator', 'candidate_value',
             'comparator_value', 'd', 'lo', 'hi', 'r', 'r_source', 'n_draws', 'seeds', 'n_rows')
TASK_WORD = {'clu': 'clustering', 'ret': 'retrieval'}
CLAIM_BETTER, CLAIM_LEVEL, CLAIM_NONE, CLAIM_VOID = 'better than', 'level with or better than', None, 'inconclusive'
DECIDED, VOID = 'decided', 'inconclusive'

# ------------------------------------------------------------------------------------------------ licensed sentences
# §6, copied from the spec (a test finds each fragment in the spec text; placeholders as in the spec).
HEAD = ("On <n> WikiArt paintings outside ArtELingo (MMArt; exact and near copies of ArtELingo train and validation "
        "paintings removed), c*'s recipe with the <X> image as its only input at test (frozen; never trained or "
        "picked on these paintings) clusters and retrieves paintings by emotion (MMArt's labels), style and genre ")
HOLDS_BODY = "better than the plain <X> image, and clusters them better than PercepT on the <X> image"
QUAL_PREFIX = "better than the plain <X> image; "
QUAL_BODY = "against PercepT on the <X> image, better in <aspects>"
QUAL_NONE = "against PercepT on the <X> image it is better in no aspect"
TAIL = ("(AMI with 20 clusters; mAP with read-outs picked on validation paintings with MMArt labels; 5 seeds). <X> is "
        "one of four encoders; the three besides Qwen passed an earlier held emotion test on ArtELingo.")
SC4_SENTENCE = ("<c* on X> is better than (or: level with or better than) the Qwen image instructed for <aspect> in "
                "<task> on MMArt.")


def _fill(text, **kw):
    for k, v in kw.items():
        text = text.replace(f'<{k}>', v)
    assert '<' not in text, f'unfilled placeholder in {text!r}'
    return text


def verdict_sentence(enc, verdict, n, sc3_win_aspects=()):
    X = MP.ENCODER_NAME[enc]
    n_txt = f'{n:,}'
    if verdict == HOLDS:
        return _fill(HEAD + HOLDS_BODY + ' ' + TAIL, n=n_txt, X=X)
    assert verdict == QUALIFIED
    if not sc3_win_aspects:
        return _fill(HEAD + QUAL_PREFIX + QUAL_NONE + ' ' + TAIL, n=n_txt, X=X)
    return _fill(HEAD + QUAL_PREFIX + QUAL_BODY + ' ' + TAIL, n=n_txt, X=X, aspects=BO.aspect_list(sc3_win_aspects))


def claim_sentence(enc, aspect, task, claim):
    assert claim in (CLAIM_BETTER, CLAIM_LEVEL)
    text = SC4_SENTENCE.replace('better than (or: level with or better than)', claim)
    return _fill(text.replace('<c* on X>', 'c* on ' + MP.ENCODER_NAME[enc]), aspect=aspect, task=TASK_WORD[task])


# ------------------------------------------------------------------------------------------------ cells
def cell_key(c):
    return (c['sc'], c['encoder'], c['aspect'], c['task'], c['data'])


def required_keys(enc):
    """The 15 cells of an encoder, in report order."""
    keys = [(sc, enc, a, TASK_OF_SC[sc][0], DATA) for sc in ('SC1', 'SC2', 'SC3') for a in ASPECTS]
    keys += [('SC4', enc, a, t, DATA) for a in ASPECTS for t in TASKS]
    return keys


def check_cell(c, smoke=False):
    """Asserts at the file boundary: fields, data, SC/task pairing, finite numbers, r, the protocol's draws."""
    missing = [k for k in CELL_KEYS if k not in c]
    assert not missing, f'cell lacks {missing}: {c}'
    where = cell_key(c)
    assert c['data'] == DATA, f'cell data must be {DATA!r} (diagnostics are not verdict cells): {where}'
    assert c['sc'] in TASK_OF_SC, f'unknown SC {where}'
    assert c['task'] in TASK_OF_SC[c['sc']], f'task {c["task"]!r} does not belong to {c["sc"]}: {where}'
    assert c['encoder'] in ENCODERS, f'unknown encoder {where}'
    assert c['aspect'] in ASPECTS, f'unknown aspect {where}'
    for k in ('candidate_value', 'comparator_value', 'd', 'lo', 'hi'):
        v = c[k]
        assert isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v), f'{k} not finite {where}'
    assert c['lo'] <= c['hi'], f'lo > hi {where}'
    if c['sc'] == 'SC4':
        r = c['r']
        assert isinstance(r, (int, float)) and not isinstance(r, bool) and math.isfinite(r) and r > 0, \
            f'SC4 needs a LEVEL unit r > 0, got {r!r}: {where}'
    else:
        assert c['r'] is None, f'{c["sc"]} is a WIN-only bar: r must be null, got {c["r"]!r}: {where}'
    assert isinstance(c['n_rows'], int) and c['n_rows'] > 0, f'n_rows {where}'
    if not smoke:
        assert c['n_draws'] == MP.N_DRAWS[c['task']], (
            f'n_draws {c["n_draws"]} != {MP.N_DRAWS[c["task"]]} for {where} (protocol draws; smoke allows others)')
    return where


def load_cells(stage2):
    """cells_clu.json and cells_ret.json of `stage2` (those that exist): (cells, file names). Each file holds only the
    cells of its task. Duplicates raise in `decide`."""
    stage2 = Path(stage2)
    cells, files = [], []
    for task in TASKS:
        f = stage2 / f'cells_{task}.json'
        if not f.exists():
            continue
        part = json.loads(f.read_text())
        assert isinstance(part, list), f'{f}: not a list'
        for c in part:
            assert c.get('task') == task, f'{f.name} holds a cell of task {c.get("task")!r}: {c}'
        cells += part
        files.append(f.name)
    return cells, files


# ------------------------------------------------------------------------------------------------ status
def check_status(status):
    """Every gate key present (the runner writes all of them), values 'ok' | 'failed', no extra keys."""
    def keys(d, want, what):
        assert isinstance(d, dict) and set(d) == set(want), f'status {what}: keys {sorted(d) if isinstance(d, dict) else d}, need {sorted(want)}'

    def val(v, what):
        assert v in ('ok', 'failed'), f'status {what}: {v!r} is not "ok" or "failed"'

    keys(status, ('gates', 'stage1'), 'top level')
    g, s1 = status['gates'], status['stage1']
    keys(g, ('i_nearcopy', 'iii', 'vi', 'i_default', 'ii', 'iv', 'i_teacher', 'v'), 'gates')
    keys(s1, ('ranges', 'picks', 'teacher_pick'), 'stage1')
    for k in ('i_nearcopy', 'iii', 'vi'):
        val(g[k], f'gates.{k}')
    for k in ('i_default', 'ii', 'iv'):
        keys(g[k], ENCODERS, f'gates.{k}')
        for e, v in g[k].items():
            val(v, f'gates.{k}.{e}')
    for k in ('i_teacher', 'v'):
        keys(g[k], ASPECTS, f'gates.{k}')
        for a, v in g[k].items():
            val(v, f'gates.{k}.{a}')
    if isinstance(s1['ranges'], str):
        val(s1['ranges'], 'stage1.ranges')
    else:
        keys(s1['ranges'], ENCODERS, 'stage1.ranges')
        for e, v in s1['ranges'].items():
            val(v, f'stage1.ranges.{e}')
    keys(s1['picks'], ENCODERS, 'stage1.picks')
    for e, v in s1['picks'].items():
        val(v, f'stage1.picks.{e}')
    val(s1['teacher_pick'], 'stage1.teacher_pick')
    return status


class Voids:
    """What the failed gates void (the table under §3.5), as reason lists."""

    def __init__(self, status):
        g, s1 = status['gates'], status['stage1']
        ranges = {e: s1['ranges'] for e in ENCODERS} if isinstance(s1['ranges'], str) else s1['ranges']
        bad = lambda v: v == 'failed'                                                  # noqa: E731
        self.every = [f'gate {k} failed' for k in ('i_nearcopy', 'iii') if bad(g[k])]
        if bad(g['i_default']['qwen']):
            self.every.append('gate i_default failed for qwen (the cut set is undefined)')
        self.enc = {e: [] for e in ENCODERS}                   # voids every cell of X (gates i_default[X], ii[X])
        self.stage1 = {e: [] for e in ENCODERS}                # voids X's SC2 and SC4 cells only (they use ranges, picks)
        for e in ENCODERS:
            if e != 'qwen' and bad(g['i_default'][e]):
                self.enc[e].append(f'gate i_default failed for {e}')
            if bad(g['ii'][e]):
                self.enc[e].append(f'gate ii failed for {e}')
            for what, v in (('ranges', ranges[e]), ('picks', s1['picks'][e])):
                if bad(v):
                    self.stage1[e].append(f'stage 1 {what} failed for {e}')
        self.vi = ['gate vi failed'] if bad(g['vi']) else []
        self.iv = {e: ([f'gate iv failed for {e} (matched arm)'] if bad(g['iv'][e]) else []) for e in ENCODERS}
        self.aspect = {a: [f'gate {k} failed for {a}' for k in ('i_teacher', 'v') if bad(g[k][a])] for a in ASPECTS}
        self.teacher_pick = []
        if bad(s1['teacher_pick']):
            self.teacher_pick.append('stage 1 teacher pick failed')
        if bad(ranges['qwen']):
            self.teacher_pick.append("stage 1 ranges failed for qwen (the teacher pick uses Qwen's ranges, D12)")

    def verdict(self, enc):
        return self.every + self.enc[enc] + self.stage1[enc] + self.vi + self.iv[enc]

    def cell(self, key):
        sc, enc, aspect, task, _ = key
        out = self.every + self.enc[enc]
        if sc in ('SC2', 'SC4'):                               # read-outs: need the ranges, which the picks need
            out = out + self.stage1[enc]
        if sc == 'SC2' or (sc == 'SC4' and task == 'ret'):
            out = out + self.vi
        if sc == 'SC3':
            out = out + self.iv[enc]
        if sc == 'SC4':
            out = out + self.aspect[aspect]
            if task == 'ret':
                out = out + self.teacher_pick
        return out


# ------------------------------------------------------------------------------------------------ decide
def sc4_claim(enc, aspect, task, outcome):
    """D7 cell: WIN 'better than', LEVEL 'level with or better than'. Other cell: LEVEL or WIN 'level with or better
    than'. BELOW: none."""
    if outcome == BELOW:
        return CLAIM_NONE
    if outcome == WIN and (aspect, task) in MP.D7[enc]:
        return CLAIM_BETTER
    assert outcome in (WIN, LEVEL), outcome
    return CLAIM_LEVEL


def decide(cells, status, n=None, smoke=False):
    """The whole output (a dict, JSON-able). `cells`: the stage 2 cells; `n`: the test count for the sentences (needed
    only when a verdict is holds or qualified)."""
    check_status(status)
    voids = Voids(status)
    by_key = {}
    for c in cells:
        key = check_cell(c, smoke)
        assert key not in by_key, f'duplicate cell {key}'
        by_key[key] = c
    res = dict(smoke=bool(smoke), n=n, verdicts={}, sc4={}, cells=[])
    for enc in ENCODERS:
        keys = required_keys(enc)
        absent = [k for k in keys if k not in by_key and not voids.cell(k)]
        assert not absent, f'{enc}: cells missing that no failed gate excuses: {absent}'
        oc = {}
        for k in keys:
            if k not in by_key:
                continue
            c = by_key[k]
            oc[k] = cell_outcome(c)
            why = voids.cell(k)
            res['cells'].append(dict(c, outcome=oc[k], status=VOID if why else DECIDED, void=why))
        # verdict
        why = voids.verdict(enc)
        o = {(sc, a): oc.get((sc, enc, a, TASK_OF_SC[sc][0], DATA)) for sc in ('SC1', 'SC2', 'SC3') for a in ASPECTS}
        if why:
            # a cell voided by the same failure is no evidence: the note is None when any SC1 or SC2 cell is voided
            sc12 = [None if voids.cell((sc, enc, a, TASK_OF_SC[sc][0], DATA)) else o[(sc, a)]
                    for sc in ('SC1', 'SC2') for a in ASPECTS]
            res['verdicts'][enc] = dict(
                verdict=INCONCLUSIVE, go=GO_OF[INCONCLUSIVE], reason=why, sentence=None, sc3_win_aspects=None,
                sc1_sc2_all_win=(all(x == WIN for x in sc12) if None not in sc12 else None))
        else:
            assert None not in o.values(), f'{enc}: verdict not void but SC1 to SC3 cells absent'
            v = encoder_verdict(o)
            wins = [a for a in ASPECTS if o[('SC3', a)] == WIN]
            sentence = None
            if v in (HOLDS, QUALIFIED):
                assert isinstance(n, int) and not isinstance(n, bool) and n > 0, \
                    f'{enc}: the licensed sentence needs <n> = test.n from records/cutsets.json, got {n!r}'
                assert smoke or n == MP.N_TEST, f'{enc}: test.n {n} != N_TEST {MP.N_TEST} (smoke allows others)'
                sentence = verdict_sentence(enc, v, n, wins)
            res['verdicts'][enc] = dict(verdict=v, go=GO_OF[v], reason=[], sentence=sentence, sc3_win_aspects=wins,
                                        sc1_sc2_all_win=None)
        # SC4 claims
        claims = []
        for a in ASPECTS:
            for t in TASKS:
                k = ('SC4', enc, a, t, DATA)
                why = voids.cell(k)
                d7 = (a, t) in MP.D7[enc]
                row = dict(aspect=a, task=t, d7=d7, outcome=oc.get(k), status=VOID if why else DECIDED, void=why)
                if why:
                    row.update(claim=CLAIM_VOID, sentence=None)
                else:
                    cl = sc4_claim(enc, a, t, oc[k])
                    row.update(claim=cl, sentence=claim_sentence(enc, a, t, cl) if cl else None)
                claims.append(row)
        res['sc4'][enc] = claims
    rest = [k for k in by_key if k[1] not in ENCODERS]
    assert not rest, f'cells not part of the rule: {rest}'
    return res


# ------------------------------------------------------------------------------------------------ records and CLI
def cutsets_test_n(smoke=False):
    """`test.n` of records/cutsets.json (written by ticket 07), or None when the file is absent. Outside smoke the
    record is read through `mt_paths.read_committed_record` (must equal HEAD)."""
    if not Path(MP.record_path('cutsets.json')).exists():
        return None
    rec = json.loads(Path(MP.record_path('cutsets.json')).read_text()) if smoke else MP.read_committed_record('cutsets.json')
    assert isinstance(rec, dict) and isinstance(rec.get('test'), dict) and 'n' in rec['test'], \
        f'cutsets record: needs {{"test": {{"n": ...}}}}'
    return rec['test']['n']


def table(res):
    lines = ['encoder   verdict                              SC4 claims (better / level-or-better / none / void)']
    for enc in ENCODERS:
        v = res['verdicts'][enc]
        cl = res['sc4'][enc]
        cnt = [sum(1 for c in cl if c['claim'] == x) for x in (CLAIM_BETTER, CLAIM_LEVEL, CLAIM_NONE, CLAIM_VOID)]
        lines.append(f'{enc:<9} {v["verdict"] + " / " + v["go"]:<36} {cnt[0]} / {cnt[1]} / {cnt[2]} / {cnt[3]}')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('decide', help='stage 2 cells + records/status.json -> OUT/stage2/outcomes.json')
    p.add_argument('--root', default=str(MP.OUT))
    p.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    stage2 = Path(args.root) / 'stage2'
    cells, files = load_cells(stage2)
    smoke = bool(MP.SMOKE)
    status_path = MP.record_path('status.json')
    assert Path(status_path).exists(), f'{status_path} is absent: the runner writes status.json before the verdicts'
    status = json.loads(Path(status_path).read_text()) if smoke else MP.read_committed_record('status.json')
    res = decide(cells, status, cutsets_test_n(smoke), smoke=smoke)
    res = dict(cell_files=files, status=status, **res)
    out = Path(args.out) if args.out else stage2 / 'outcomes.json'
    out.write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')
    print(table(res))
    print(f'{len(res["cells"])} cells from {len(files)} files -> {out}')


if __name__ == '__main__':
    main()
