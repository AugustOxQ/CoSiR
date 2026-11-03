"""r19 ticket 07: gates (iv) and (vii), r14 part (spec 2026-11-03-past-picks-design.md §3.5).

Gate (iv), r14 part: r14's CLIP L/14 SC1 to SC6 values and intervals, replayed by this round's stage 2 code
(`pp_score14.run_*`, the code that scores the new three) and compared with `R14_OUT/stage3/cells_*.json` within 1e-9:
  clu        9 cells (SC1, SC3, SC6 x 3 aspects): fresh partitions by r15's route (a copy of `mt_clu.replay_iii`, for CLIP L/14,
             matched arm M7), `r` and `comparator_reading(s)` included.
  ret        6 cells (SC2, SC6 x 3): the frozen image recomputed (the new three's route: `bs_artret.ENC[...]['r11_arm']` None
             for the call), at r14's stored pick; r14's replay 0(xii) showed recomputed equals r11's stored, max diff 0.0.
  affection  4 cells (SC4, SC5, SC6 x 2) from r14's photo exports.
Gate (vii), r14 part: fed r14's stored validation exports, the stage 1 code (`pp_score14.stage1_artelingo`,
`stage1_affection`) reproduces `R14_RECORDS/stage1_<enc>.json` for siglip1 (frozen pick computed, r11's rule) and clip_l14:
ranges, ArtELingo picks, Affection ranges and picks.

Outputs (OUT/stage0/): `gate_iv_r14_<part>.json`, `gate_vii_r14_<enc>.json`, merged `gate_iv_r14.json`, `gate_vii_r14.json`.
A part passes only at the protocol's draws (2,000 clustering, 10,000 retrieval); reduced draws (smoke) check the wiring.

CLI: `pp_gate14.py iv clu|ret|affection|merge`, `pp_gate14.py vii siglip1|clip_l14|merge`  (`-h` for options).
"""
import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import pp_paths as PP
import pp_register  # noqa: F401
import pp_score14 as S

import bs_artclu
import bs_artret

TOL = bs_artclu.TOL                                  # 1e-9
ENC14 = 'clip_l14'                                   # r14's encoder of gate (iv)
IV_PARTS = ('clu', 'ret', 'affection')
VII_ENCODERS = ('siglip1', 'clip_l14')
STORED = dict(clu='cells_artelingo_clu.json', ret='cells_artelingo_ret.json', affection='cells_affection.json')
EXPECT_CELLS = dict(clu=9, ret=6, affection=4)
INTERVAL_KEYS = ('lo', 'hi', 'n_draws')              # not compared at reduced draws
SKIP_ALWAYS = ('written',)


# ============================================================================================ compare
def _item(ours, theirs, ok, diff):
    return dict(ours=ours, theirs=theirs, abs_diff=float(diff), ok=bool(ok))


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _walk(ours, theirs, path, items, skip, extra_ok):
    if isinstance(theirs, dict):
        if not isinstance(ours, dict):
            items[path or '.'] = _item(repr(ours), '<dict>', False, 1.0)
            return
        if not extra_ok:
            for k in sorted(set(ours) - set(theirs) - set(SKIP_ALWAYS)):
                items[f'{path}/{k}' if path else k] = _item('<present>', '<absent>', False, 1.0)
        for k, v in theirs.items():
            if k in SKIP_ALWAYS or k in skip:
                continue
            sub = f'{path}/{k}' if path else k
            if k not in ours:
                items[sub] = _item('<absent>', repr(v) if not isinstance(v, (dict, list)) else '<value>', False, 1.0)
            else:
                _walk(ours[k], v, sub, items, skip, extra_ok)
    elif isinstance(theirs, list):
        if not isinstance(ours, list) or len(ours) != len(theirs):
            items[path or '.'] = _item(f'len {len(ours) if isinstance(ours, list) else repr(ours)}', f'len {len(theirs)}',
                                       False, 1.0)
            return
        for i, (o, t) in enumerate(zip(ours, theirs)):
            _walk(o, t, f'{path}[{i}]', items, skip, extra_ok)
    elif _is_number(theirs):
        if not _is_number(ours):
            items[path] = _item(repr(ours), theirs, False, 1.0)
        else:
            diff = abs(float(ours) - float(theirs))
            finite = math.isfinite(float(ours)) and math.isfinite(float(theirs))
            items[path] = _item(float(ours), float(theirs), finite and diff <= TOL, diff if finite else float('inf'))
    else:                                            # str, bool, None: exact (and of the same type)
        same = type(ours) is type(theirs) and ours == theirs
        items[path or '.'] = _item(ours, theirs, same, 0.0 if same else 1.0)


def compare(ours, theirs, full_draws=True, extra_ok=False):
    """Recursive compare of `ours` with the stored `theirs`. Numbers within 1e-9; strings, booleans, None exact and of the
    same type; NaN fails; a key of `theirs` missing from `ours` fails; extra keys of `ours` fail unless `extra_ok`; the
    `written` key is ignored anywhere; `lo`, `hi`, `n_draws` are skipped unless `full_draws`.
    Returns items {path: {ours, theirs, abs_diff, ok}}, `max_abs_diff`, `wiring_ok` (all items ok), `intervals_compared`
    (= full_draws), `passed` (wiring_ok and full_draws; like `mt_clu._result`)."""
    items = {}
    _walk(ours, theirs, '', items, set() if full_draws else set(INTERVAL_KEYS), extra_ok)
    assert items, 'nothing was compared'
    ok = all(i['ok'] for i in items.values())
    return dict(items=items, n_items=len(items), max_abs_diff=max(i['abs_diff'] for i in items.values()), wiring_ok=ok,
                intervals_compared=bool(full_draws), passed=bool(ok and full_draws))


def _key(c):
    return f'{c["sc"]}/{c["aspect"]}/{c["task"]}'


def compare_cells(ours, theirs, full_draws=True):
    """Cells matched on (sc, aspect, task); the two sets must be equal; our extra keys (`n_rows`, `n_labelled`) are allowed."""
    o, t = {_key(c): c for c in ours}, {_key(c): c for c in theirs}
    assert len(o) == len(ours) and len(t) == len(theirs), 'a cell appears twice'
    assert sorted(o) == sorted(t), f'cells differ: ours {sorted(o)}, stored {sorted(t)}'
    return compare(o, t, full_draws=full_draws, extra_ok=True)


# ============================================================================================ inputs
def r14_record(enc):
    """r14's committed stage 1 record (`R14_RECORDS/stage1_<enc>.json`), the frozen input of both gates."""
    assert enc in VII_ENCODERS, f'{enc}: r14 has no stage 1 record for it that this gate replays'
    return json.loads((PP.R14_RECORDS / f'stage1_{enc}.json').read_text())


def stored_cells(part):
    """All cells of r14's `stage3/cells_<part>.json` (every encoder)."""
    return json.loads((PP.R14_OUT / 'stage3' / STORED[part]).read_text())


@contextlib.contextmanager
def computed_frozen(enc):
    """`bs_artret.ENC[enc]['r11_arm']` is None inside: the frozen image is recomputed (the route of the new three) instead
    of read from r11's stored outputs. The table is restored on exit, also when the body raises."""
    spec = bs_artret.ENC[enc]
    old = spec['r11_arm']
    spec['r11_arm'] = None
    try:
        yield
    finally:
        spec['r11_arm'] = old


def _stage0():
    folder = Path(PP.OUT) / 'stage0'
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write(name, doc):
    path = _stage0() / name
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)
    return path


def _draws(part, n_draws):
    """(clu draws, ret draws) of the run: the protocol's, unless a smoke run says otherwise."""
    tasks = {'clu': ('clu',), 'ret': ('ret',), 'affection': ('clu', 'ret')}[part]
    n = {t: PP.N_DRAWS[t] if n_draws is None else int(n_draws) for t in tasks}
    if not PP.SMOKE:
        assert all(n[t] == PP.N_DRAWS[t] for t in tasks), f'{n_draws} draws: only smoke runs may use other counts'
    return n


# ============================================================================================ gate (iv)
def gate_iv(part, n_draws=None, workers=8):
    """One part of gate (iv), r14 part, for CLIP L/14. Writes `OUT/stage0/gate_iv_r14_<part>.json` and returns it."""
    assert part in IV_PARTS, part
    n = _draws(part, n_draws)
    full = all(n[t] == PP.N_DRAWS[t] for t in n)
    rec = r14_record(ENC14)
    S.check_record(ENC14, rec)                              # with r14's table: the stored frozen pick form
    t0 = time.monotonic()
    if part == 'clu':
        cells = S.run_artelingo_clu(ENC14, rec, n_draws=n['clu'], workers=workers)
    elif part == 'ret':
        with computed_frozen(ENC14):
            cells = S.run_artelingo_ret(ENC14, rec, _stage0() / 'gate_iv_r14_artret', n_draws=n['ret'], workers=workers)
    else:
        cells = S.run_affection(ENC14, rec, root=PP.R14_OUT, cache=_stage0() / 'gate_iv_r14_affection_partitions',
                                n_clu=n['clu'], n_ret=n['ret'], workers=workers)['cells']
    stored = [c for c in stored_cells(part) if c['encoder'] == ENC14]
    assert len(stored) == EXPECT_CELLS[part], (part, len(stored))
    assert len(cells) == EXPECT_CELLS[part], (part, len(cells))
    res = compare_cells(cells, stored, full_draws=full)
    doc = dict(gate='iv_r14', part=part, encoder=ENC14, n_draws=n['clu'] if part == 'clu' else n['ret'] if part == 'ret' else n,
               n_cells=len(cells), seconds=round(time.monotonic() - t0), **res)
    _write(f'gate_iv_r14_{part}.json', doc)
    return doc


def _merge(prefix, names, name_of, expect_draws=None):
    """The merged result: every part file present, passed, intervals compared, run at the protocol's draws."""
    parts, missing, bad = {}, [], []
    for n in names:
        path = Path(PP.OUT) / 'stage0' / name_of(n)
        if not path.exists():
            missing.append(n)
            continue
        d = json.loads(path.read_text())
        parts[n] = dict(passed=d['passed'], wiring_ok=d['wiring_ok'], intervals_compared=d['intervals_compared'],
                        max_abs_diff=d['max_abs_diff'], n_draws=d.get('n_draws'))
        if not (d['passed'] and d['wiring_ok'] and d['intervals_compared']):
            bad.append(n)
        if expect_draws is not None and d.get('n_draws') != expect_draws[n]:
            bad.append(n)
    passed = not missing and not bad
    return dict(gate=prefix, passed=bool(passed), missing=missing, failed=sorted(set(bad)), parts=parts,
                max_abs_diff=max([p['max_abs_diff'] for p in parts.values()], default=None))


def merge_iv():
    """`gate_iv_r14.json`: passes only when all three part files exist, ran at full draws and passed."""
    expect = dict(clu=PP.N_DRAWS['clu'], ret=PP.N_DRAWS['ret'], affection=dict(PP.N_DRAWS))
    doc = _merge('iv_r14', IV_PARTS, lambda p: f'gate_iv_r14_{p}.json', expect)
    _write('gate_iv_r14.json', doc)
    return doc


# ============================================================================================ gate (vii)
def gate_vii(enc, workers=8):
    """Gate (vii), r14 part, for one encoder: stage 1 replayed from r14's validation exports against r14's stage 1 record."""
    assert enc in VII_ENCODERS, f'{enc}: gate (vii) replays {VII_ENCODERS}'
    stored = r14_record(enc)
    t0 = time.monotonic()
    ours = S.stage1_artelingo(enc, root=PP.R14_OUT / 'cstar' / enc, workers=workers)
    ours = dict(ours, affection=S.stage1_affection(enc, root=PP.R14_OUT, workers=workers))
    blocks = ('artclu', 'artret_ranges', 'artret_picks', 'affection')
    res = compare({b: ours[b] for b in blocks}, {b: stored[b] for b in blocks}, full_draws=True)
    doc = dict(gate='vii_r14', encoder=enc, blocks=list(blocks), seconds=round(time.monotonic() - t0), **res)
    _write(f'gate_vii_r14_{enc}.json', doc)
    return doc


def merge_vii():
    doc = _merge('vii_r14', VII_ENCODERS, lambda e: f'gate_vii_r14_{e}.json')
    _write('gate_vii_r14.json', doc)
    return doc


# ============================================================================================ CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sub = p.add_subparsers(dest='gate', required=True)
    iv = sub.add_parser('iv')
    iv.add_argument('part', choices=IV_PARTS + ('merge',))
    iv.add_argument('--n-draws', type=int, default=None, help='smoke only (PP_SMOKE=1)')
    vii = sub.add_parser('vii')
    vii.add_argument('part', choices=VII_ENCODERS + ('merge',))
    args = p.parse_args(argv)
    assert Path(PP.OUT).resolve() != PP.MAIN.resolve(), 'bad output root'
    if args.gate == 'iv':
        doc = merge_iv() if args.part == 'merge' else gate_iv(args.part, args.n_draws, args.workers)
    else:
        doc = merge_vii() if args.part == 'merge' else gate_vii(args.part, args.workers)
    summary = {k: doc[k] for k in ('gate', 'passed', 'max_abs_diff') if k in doc}
    for k in ('part', 'encoder', 'wiring_ok', 'intervals_compared', 'n_items', 'missing', 'failed', 'seconds'):
        if k in doc:
            summary[k] = doc[k]
    print(json.dumps(summary))
    return 0 if doc['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
