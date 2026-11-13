"""r29 ticket 10: the stage 2 SC4 path for SigLIP 2 (E2) and CLIP L/14 (E3), each with its own text encoder.

Spec docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md §3.2 (E2, E3), §3.4 (cells table row 7), §3.5 gate (viii),
§6 SC4, D3, D13, D14, D15, D23. This module is wiring only: the trainer is ticket 03's, the cells ticket 06's, the ladder
ticket 08's, the stores ticket 02's. Names: the image encoder is bs_paths' key (`siglip2`, `clip_l14`); the arm and the
caption source are the `r29_paths.CAPTION_SOURCES` name (`siglip2`, `siglip2_prefix`, `clipl14`, `clipl14_prefix`).

What it adds
  caption source (D13)   K*'s form from `records/prefix_pick.json` (keys kstar, kstar_form, p1_form, flag; asserted):
                         plain -> the encoder's own plain text store, prefixed -> its prefixed store (ticket 02's extractor).
  gate (viii)            `gate_viii SOURCE`: the store check of ticket 02 (`r29_extract.check_store`: one row per key, finite,
                         unit norm within 1e-3, keys join to the SigLIP 1 reference by sample ID and painting, truncation count
                         recorded, width and backbone of the source) plus gate (iii)'s join checks on that source
                         (`r29_captions.gate_iii`). Record `records/gates/viii_<source>.json` through
                         `r29_captions.write_gate_record`, three tries; no training, cells or ladder without a passed record.
  D3                     every stage 2 command refuses unless `records/pass_rule.json` says `"status": "pass"` (`smoke` trains
                         nothing and reads no candidate number; it is the one command that needs no pass).
  r per encoder          ticket 06's picks read the encoder's own five seeds (its `Wiring` is built for the encoder, never SigLIP 1).
  beside                 the current r14 model on the same encoder: r17's SC1 / SC3 / SC7 and r14's SC6 ArtELingo cells of that
                         encoder (frozen), next to the new arm's cells of the same protocol (descriptive; decides nothing).

CLI (from the round folder; heavy steps through `res-queue`, see the hand-back):
  r29_encoders.py stores  --encoder siglip2|clip_l14|all [--source S]      gate (viii) on K*'s store (or the named store)
  r29_encoders.py train   --encoder E [--seeds 0 1 2 3 4] [--device cuda] [--root DIR]      ticket 03's trainer, arm = K*'s source
  r29_encoders.py export  --encoder E [--seeds ..] [--kind val held] [--device D]           ticket 03's exports (val, held)
  r29_encoders.py picks   --encoder E        validation picks and ranges -> records/image_picks_<arm>.json (ticket 06)
  r29_encoders.py cells   --encoder E        the 12 test cells (6 against plain, 6 against Q1) -> records/image_cells_<arm>.json
  r29_encoders.py ladder  --encoder E        SC4 ladder -> records/SC4_<encoder>_<arm>.json (ticket 08, written once)
  r29_encoders.py beside  --encoder E        the current r14 model's cells beside the new arm's (prints; --out FILE)
  r29_encoders.py smoke   --encoder E [--form plain|prefixed] [--stores]   wiring check on r14's exports, no training
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r29_paths as P  # noqa: E402

ENCODERS = ('siglip2', 'clip_l14')                   # E2, E3 (bs_paths' keys); SigLIP 1 is K*'s own, tickets 03 to 08
FORMS = ('plain', 'prefixed')
KSTAR_OF_FORM = {'plain': 'siglip1', 'prefixed': 'siglip1_prefix'}
STORE_SOURCES = tuple(s for enc in ENCODERS for s in P.ENCODER_CAPTION_SOURCES[enc][1:])   # the four stores of gate (viii)
EXTRACTOR_NAME = {'siglip2': 'siglip2_so400m', 'clip_l14': 'clip_l14'}   # `r29_extract.py extract --encoder`
SOURCE_ENCODER = {'siglip2': 'siglip2', 'clipl14': 'clip_l14'}           # a source's name without `_prefix` -> its encoder
HEAD_WIDTH = P.OUTPUT_WIDTH
MAX_TRIES = 3                                          # spec §3.5 stage 0: at most three tries per gate
AMI_K = {'emotion': 9, 'style': 27, 'genre': 10}       # ArtELingo class counts (r17)


# ------------------------------------------------------------------------------------------------ lazy heavy imports
def _trainer():
    import r29_trainer
    return r29_trainer


def _image_cells():
    import r29_image_cells
    return r29_image_cells


def _ladder():
    import r29_ladder
    return r29_ladder


def _captions():
    import r29_captions
    return r29_captions


def _extract():
    import r29_extract
    return r29_extract


def records_dir():
    """Committed records, or OUT/records under a smoke run (R29_OUTPUTS set): as tickets 06 and 08."""
    return P.RECORDS if P.OUT == P.DEFAULT_OUT else P.OUT / 'records'


def default_root(encoder):
    return P.OUT / 'cstar' / encoder                   # ticket 03's default root of a real run


# ------------------------------------------------------------------------------------------------ D3 and D13
def require_pass():
    """D3 as amended by A1: stage 2 only after `records/pass_rule.json` says pass, or on its fail under a committed A1 record
    (`P.stage2_gate`). Returns the gate's answer, "pass" or "A1"."""
    return P.stage2_gate(records_dir())


def read_prefix_pick(records=None):
    """`records/prefix_pick.json`, its four keys asserted (the index's Shared interfaces); the form must agree with kstar.
    `records` overrides the usual records directory for callers such as r29_decide's CLI and temporary-folder tests."""
    path = Path(records) / 'prefix_pick.json' if records is not None else records_dir() / 'prefix_pick.json'
    if not path.is_file():
        raise RuntimeError(f'no {path}: ticket 05 writes it; E2 and E3 take K*\'s caption form from it (D13)')
    doc = json.loads(path.read_text())
    for key in ('kstar', 'kstar_form', 'p1_form', 'flag'):
        assert key in doc, f'{path}: key {key} missing (keys {sorted(doc)})'
    assert doc['kstar_form'] in FORMS, f'{path}: kstar_form {doc["kstar_form"]!r}, one of {FORMS}'
    assert doc['p1_form'] in FORMS, f'{path}: p1_form {doc["p1_form"]!r}'
    assert doc['kstar'] == KSTAR_OF_FORM[doc['kstar_form']], \
        f'{path}: kstar {doc["kstar"]!r} does not match kstar_form {doc["kstar_form"]!r}'
    return doc


def arm_source(encoder, form):
    """The caption source (= arm) of an encoder for a caption form: the encoder's own plain or prefixed store."""
    assert encoder in ENCODERS, f'encoder {encoder!r}: not one of {ENCODERS}'
    assert form in FORMS, form
    source = P.ENCODER_CAPTION_SOURCES[encoder][1 + FORMS.index(form)]
    assert source in P.CAPTION_SOURCES, source
    return source


def kstar_source(encoder, records=None):
    """D13: E2's and E3's caption source is the store of K*'s form, no own check."""
    assert encoder in ENCODERS, f'encoder {encoder!r}: not one of {ENCODERS}'
    return arm_source(encoder, read_prefix_pick(records)['kstar_form'])


def arm_encoder(source):
    """The image encoder of an arm named after its caption source."""
    base = source.removesuffix('_prefix')
    assert base in SOURCE_ENCODER, f'arm {source!r}: not a caption source of siglip2 or clipl14'
    return SOURCE_ENCODER[base]


def _gate_path(source):
    return records_dir() / 'gates' / f'viii_{source}.json'


def require_store_gate(source):
    """Gate (viii) of the store passed (the spec's order: each gate before the first candidate number it serves)."""
    path = _gate_path(source)
    if not path.is_file():
        raise RuntimeError(f'gate viii_{source} not run: no {path} (run `r29_encoders.py stores` after the store exists)')
    status = json.loads(path.read_text())['status']
    if status != 'pass':
        raise RuntimeError(f'gate viii_{source} is {status}, not pass: the pieces of this store are parked (spec §3.5)')


# ------------------------------------------------------------------------------------------------ gate (viii)
def store_check(source, folder=None, reference=None, width=None):
    """Ticket 02's store check on one store: `r29_extract.check_store` against the SigLIP 1 plain store's keys (the reference:
    every store has the same 308,723 sample IDs and paintings, C6), with the prefix expected by the source's name, plus the
    source's backbone and width. Returns its dict (ok, n_nonfinite, n_bad_norm, max_norm_dev, n_texts, n_truncated,
    index_sha256) with the store's `truncation_limit` and `prefix`. A missing store raises RuntimeError (no try used)."""
    assert source in STORE_SOURCES, f'{source!r}: not one of the four E2 / E3 stores {STORE_SOURCES}'
    X, C = _extract(), _captions()
    folder = Path(folder if folder is not None else P.CAPTION_SOURCES[source][0])
    width = P.CAPTION_SOURCES[source][1] if width is None else width
    if not (folder / 'index.json').is_file():
        enc = EXTRACTOR_NAME[arm_encoder(source)]
        if source.endswith('_prefix'):
            raise RuntimeError(f'no store at {folder}: extract it first with `res-queue gpu --short --no-wait --label '
                               f'r29:t10-extract-{source} -- python r29_extract.py extract --encoder {enc}` (ticket 02)')
        raise RuntimeError(f'no plain store at {folder} (the frozen pre_extract store of {enc})')
    reference = reference if reference is not None else X._json(P.CAPTION_SOURCES['siglip1'][0] / 'index.json')
    index = X._json(folder / 'index.json')
    assert index['backbone'] == C.BACKBONE[source], f'{folder}: backbone {index["backbone"]!r}, expected {C.BACKBONE[source]!r}'
    assert index['dim'] == width, f'{folder}: width {index["dim"]}, the source\'s is {width}'
    out = X.check_store(folder, reference, expected_prefix=P.PREFIX if source.endswith('_prefix') else None)
    return dict(out, truncation_limit=index['truncation_limit'], prefix=index.get('prefix', ''), width=width, folder=str(folder))


def _join_check(source):
    """Gate (iii)'s join checks on the source's store (ticket 01): IDs and paintings against Qwen3's store and the annotations, per-painting
    caption counts, the roles, the painting means (shape, finite, unit)."""
    return _captions().gate_iii(source, write=False)


def _gates_dir():
    return records_dir() / 'gates'


def gate_viii(source, store_check=store_check, join_check=_join_check):
    """Gate (viii) on one store -> `records/gates/viii_<source>.json` (r29_captions.write_gate_record). Three tries: a fourth run, and a
    run after a pass, raise. An assertion of a check is a `fail` try (written); a missing store is not (it raises first). The truncation
    count is recorded; a count above 0 is flagged and does not fail the gate (D14: cut at the limit, count, flag)."""
    assert source in STORE_SOURCES, f'{source!r}: not one of the four E2 / E3 stores {STORE_SOURCES}'
    path = _gate_path(source)
    prior = 0                                          # recorded tries before this run
    if path.exists():
        old = json.loads(path.read_text())
        prior = old['tries']
        if old['status'] == 'pass':
            raise RuntimeError(f'gate viii_{source} already passed (try {old["tries"]}): {path}')
        if old['tries'] >= MAX_TRIES:
            raise RuntimeError(f'gate viii_{source} used its three tries ({path}): it parks what it serves')
    enc = arm_encoder(source)
    details = dict(source=source, encoder=enc, form='prefixed' if source.endswith('_prefix') else 'plain')
    status = 'fail'
    try:
        store = store_check(source)
        details['store'] = store
        n = store['n_truncated']
        details['truncation_flag'] = (f'{n} of {store["n_texts"]} captions cut at the encoder limit of {store["truncation_limit"]} '
                                      'tokens (D14: cut, counted, flagged)') if n else None
        join = join_check(source)
        details['join'] = join['details']
        status = 'pass' if join['status'] == 'pass' else 'fail'
    except AssertionError as error:
        details['error'] = f'{type(error).__name__}: {error}'
    if status == 'fail' and prior == MAX_TRIES - 1:    # a failure at try 3 parks (as ticket 14's gate_vii); before it, the gate stays open
        status = 'parked'
    return _captions().write_gate_record(_gates_dir(), dict(status=status, details=details), gate=f'viii_{source}')


# ------------------------------------------------------------------------------------------------ the current model beside
def _frozen_cells(encoder):
    """The current r14 model's ArtELingo cells of `encoder` of the six SC4 cell types against plain and against Q1, each with
    the frozen record it comes from: AMI r17 SC1 (plain) and SC3 (teacher); image mAP r17 SC7 (plain, the untold candidate) and
    r14 SC6 (teacher, the aspect head): the protocols of ticket 06's four cell types."""
    r17 = {c['id']: c for c in json.loads((P.R17_ROUND / 'records/verdicts.json').read_text())['cells']}
    r14 = json.loads((P.R14_ROUND / 'records/verdicts.json').read_text())['cells']
    out = {}
    for a in P.ASPECTS:
        for comp, sc in (('plain', 'SC1'), ('teacher', 'SC3')):
            rid = f'{sc}/artelingo/{encoder}/{a}/ami/{AMI_K[a]}/{comp}'
            c = r17[rid]
            out[(a, 'ami', comp)] = dict(source=f'r17 {rid}', value=c['cand'], comparator_value=c['comp'], d=c['d'], lo=c['lo'],
                                         hi=c['hi'], r=c['r'], outcome=c['outcome'])
        rid = f'SC7/artelingo/{encoder}/{a}/map/-/plain'
        c = r17[rid]
        out[(a, 'map', 'plain')] = dict(source=f'r17 {rid}', value=c['cand'], comparator_value=c['comp'], d=c['d'], lo=c['lo'],
                                        hi=c['hi'], r=c['r'], outcome=c['outcome'])
        (c,) = [c for c in r14 if (c['sc'], c['encoder'], c['aspect'], c['data'], c['task']) == ('SC6', encoder, a, 'artelingo_held', 'ret')]
        out[(a, 'map', 'teacher')] = dict(source=f'r14 SC6 {encoder}/{a}/ret', value=c['candidate_value'],
                                          comparator_value=c['comparator_value'], d=c['d'], lo=c['lo'], hi=c['hi'], r=c['r'],
                                          outcome=c['outcome'])
    return out


def beside(encoder, arm=None):
    """Rows {cell: '<aspect>/<ami|map>/<plain|teacher>', current: frozen cell of the r14 model on `encoder`, new: the arm's cell of
    `records/image_cells_<arm>.json` joined by aspect, measure and comparator (None when absent)}. Descriptive."""
    assert encoder in ENCODERS, f'encoder {encoder!r}: not one of {ENCODERS}'
    frozen = _frozen_cells(encoder)
    new = {}
    if arm is not None:
        assert arm_encoder(arm) == encoder, f'arm {arm!r} is not on encoder {encoder!r}'
        path = records_dir() / f'image_cells_{arm}.json'
        if path.exists():
            rec = json.loads(path.read_text())
            assert rec['encoder'] == encoder and rec['arm'] == arm, f'{path}: encoder {rec["encoder"]!r}, arm {rec["arm"]!r}'
            for c in rec['cells']:
                _, enc, arm_, aspect, mid, comp = c['id'].split('/')
                assert (enc, arm_) == (encoder, arm), c['id']
                key = (aspect, 'ami' if mid.startswith('ami') else mid, comp)
                assert key in frozen and key not in new, f'{path}: cell {c["id"]} not one of the 12 of SC4, or twice'
                new[key] = {k: c[k] for k in ('id', 'value', 'comparator_value', 'd', 'lo', 'hi', 'r', 'outcome')}
    return [dict(cell='/'.join(key), current=cur, new=new.get(key)) for key, cur in frozen.items()]


def beside_text(rows):
    f = lambda c: '        n/a' if c is None else f'{c["value"]:.3f} {c["outcome"]:<5} d {c["d"]:+.4f}'   # noqa: E731
    lines = ['cell                  | current r14 model          | new arm', '-' * 78]
    lines += [f'{r["cell"]:<21} | {f(r["current"]):<26} | {f(r["new"])}' for r in rows]
    return '\n'.join(lines)


# ------------------------------------------------------------------------------------------------ smoke
def join_report(img_names, cap_names, head_shapes, plain_shape, cap_shape, plain_width, cap_width, head_width=HEAD_WIDTH):
    """Names and shapes of one split join (the smoke's assertions, pure): the caption side's painting names equal the image side's,
    in order (the heads' names hash is checked against these by ticket 06's `Wiring`), every head is [n, 2048], the encoder's
    plain embedding [n, its width] and the caption means [n, the source's width]."""
    n = len(img_names)
    assert list(img_names) == list(cap_names), 'painting names: the caption side and the image side differ in content or order'
    for name, shape in head_shapes.items():
        assert tuple(shape) == (n, head_width), f'head {name}: shape {tuple(shape)}, expected {(n, head_width)}'
    assert tuple(plain_shape) == (n, plain_width), f'plain embedding shape {tuple(plain_shape)}, expected {(n, plain_width)}'
    assert tuple(cap_shape) == (n, cap_width), f'caption means shape {tuple(cap_shape)}, expected {(n, cap_width)}'
    return dict(ok=True, n=n, heads=len(head_shapes), plain_shape=list(plain_shape), cap_shape=list(cap_shape))


def smoke(encoder, form='plain', stores=False):
    """The wiring on r14's exports of `encoder` (no training): r14's 30 head exports (validation, held; 3 aspects x 5 seeds) read through
    ticket 06's wiring (names hash against the sorted painting names, shapes, unit norm), the encoder's plain embedding, the caption
    means of the encoder's own text store for the validation paintings in the trainer's order, the 12 frozen cells of `beside`, and the
    name maps of tickets 03, 06 and 08. `stores`: also gate (viii)'s two checks on the source's store (not recorded)."""
    assert encoder in ENCODERS
    source = arm_source(encoder, form)
    width = P.CAPTION_SOURCES[source][1]
    T, IC, L = _trainer(), _image_cells(), _ladder()
    import ns_data
    import ns_paths
    import ct2_data
    assert IC.encoder_of_arm(source) == L.encoder_of_arm(source) == arm_encoder(source) == encoder
    assert source in P.ENCODER_CAPTION_SOURCES[encoder] and encoder in T.ENCODERS
    root = P.R14_OUT / 'cstar' / encoder
    result = dict(encoder=encoder, source=source, form=form, r14_root=str(root), splits={})
    s = ct2_data.split2('full')
    caption_val_names = [str(n) for n in s['all_painting_ids'][s['dev']]]
    cap = _captions().painting_means(source, caption_val_names)
    assert np.isfinite(cap).all() and abs(np.linalg.norm(cap, axis=1) - 1).max() < 1e-5, 'caption means: not finite or not unit'
    for split in ('val', 'held'):
        w = IC.Wiring(root, P.K0_ARM, encoder)
        heads = {f'{a}__s{sd}': w.head(split, a, sd).shape for a in P.ASPECTS for sd in P.SEEDS}
        img_names = list(ns_data.keys('artelingo', split))
        plain = ns_data.plain('artelingo', split, encoder)
        if split == 'val':
            result['splits'][split] = join_report(img_names, caption_val_names, heads, plain.shape, cap.shape, ns_paths.WIDTH[encoder], width)
        else:                                      # held: image side only (the caption tower is not read on test rows)
            result['splits'][split] = join_report(img_names, img_names, heads, plain.shape, (len(img_names), width), ns_paths.WIDTH[encoder], width)
    rows = beside(encoder)
    assert len(rows) == 12 and all(np.isfinite([r['current'][k] for k in ('value', 'comparator_value', 'd', 'lo', 'hi')]).all() for r in rows)
    result['beside_cells'] = len(rows)
    if stores:
        sc = store_check(source)
        result['store_check'] = sc
        result['join_check'] = _join_check(source)['status']
    return result


# ------------------------------------------------------------------------------------------------ CLI
def _encoders(arg):
    return list(ENCODERS) if arg == 'all' else [arg]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('stores', 'train', 'export', 'picks', 'cells', 'ladder', 'beside', 'smoke'):
        s = sub.add_parser(name)
        s.add_argument('--encoder', required=True, choices=[*ENCODERS, *(['all'] if name == 'stores' else [])])
    sub.choices['stores'].add_argument('--source', choices=STORE_SOURCES, help='gate this store instead of K*\'s')
    for name in ('train', 'export'):
        sub.choices[name].add_argument('--seeds', type=int, nargs='+', choices=P.SEEDS, default=list(P.SEEDS))
        sub.choices[name].add_argument('--root', type=Path, help='default: OUT/cstar/<encoder>')
    sub.choices['train'].add_argument('--device', default='cuda')
    sub.choices['export'].add_argument('--kind', nargs='+', choices=('val', 'held'), default=['val', 'held'])
    sub.choices['export'].add_argument('--device', default=None)
    for name in ('picks', 'cells'):
        sub.choices[name].add_argument('--workers', type=int, default=8, choices=range(1, 9))
        sub.choices[name].add_argument('--root', type=Path)
    sub.choices['beside'].add_argument('--out', type=Path)
    sub.choices['smoke'].add_argument('--form', choices=FORMS, default='plain')
    sub.choices['smoke'].add_argument('--stores', action='store_true', help='also gate (viii)\'s checks on the source\'s store (not recorded)')
    args = ap.parse_args(argv)

    if args.cmd == 'smoke':                        # trains nothing, reads no candidate number: no pass needed (D3)
        result = smoke(args.encoder, args.form, args.stores)
        out = (P.SMOKE_OUT if P.OUT == P.DEFAULT_OUT else P.OUT / 'smoke') / 't10'
        out.mkdir(parents=True, exist_ok=True)
        path = out / f'smoke_{args.encoder}_{args.form}.json'
        path.write_text(json.dumps(result, indent=1, default=str) + '\n')
        print(f'smoke {args.encoder} ({result["source"]}): names and shapes join; {result["beside_cells"]} frozen cells -> {path}')
        return 0

    require_pass()                                 # D3, before anything else is read
    if args.cmd == 'stores':
        code = 0
        for enc in _encoders(args.encoder):
            source = args.source or kstar_source(enc)
            assert arm_encoder(source) == enc, f'store {source!r} is not an {enc} store'
            rec = gate_viii(source)
            print(f'gate viii_{source}: {rec["status"]} (try {rec["tries"]})')
            code = code or int(rec['status'] != 'pass')
        return code

    enc = args.encoder
    source = kstar_source(enc)
    if args.cmd not in ('beside', 'ladder'):       # the ladder reads gate viii itself: a parked store makes SC4 of that encoder inconclusive
        require_store_gate(source)                 # no training, export, picks or cells on a store that did not pass (viii)
    root = getattr(args, 'root', None) or default_root(enc)
    if args.cmd == 'train':
        _trainer().train(enc, source, args.seeds, args.device, root)
    elif args.cmd == 'export':
        _trainer().export(enc, source, args.seeds, root, kinds=tuple(args.kind), device=args.device)
    elif args.cmd in ('picks', 'cells'):
        return _image_cells().main(['--workers', str(args.workers), args.cmd, '--arm', source, '--encoder', enc, '--root', str(root)])
    elif args.cmd == 'ladder':
        return _ladder().main(['ladder', '--claim', 'SC4', '--arm', source, '--encoder', enc])
    else:
        rows = beside(enc, source)
        print(beside_text(rows))
        if args.out:
            args.out.write_text(json.dumps(dict(encoder=enc, arm=source, rows=rows), indent=1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
