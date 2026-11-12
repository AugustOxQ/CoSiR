"""r28 job E, part 1: CLAY and CRL on SigLIP 1 (handoff section 4 item 7), with the encoder as a parameter.

The existing rows replay unchanged: `enc='clip_l14'` calls r18's and r17's own loaders (`om_data.clay`, `ns_data.crl`);
`enc='siglip1'` swaps in SigLIP 1's image rows (`ns_data.plain(..., 'siglip1')`, the stores under
`/data/SSD2/pre_extract/{artelingo_full,mmart}/siglip_so400m/`) and SigLIP 1's text rows. Everything that computes a
number (CLAY's geometry `om_clay.subspace/condition`, CRL's vector `ac_crl.vectors`, k-means `ns_metrics.partitions`,
AMI, the read-out rule `ac_select.arm_table`, the AP scorer `ns_ret.arm_ap`, r18's two text-set choices) is imported
unchanged. No existing file is edited.

    python r28_clay_crl.py texts                 GPU: encode the CRL / CLAY text sets with SigLIP 1's text tower
    python r28_clay_crl.py picks  --enc siglip1  validation only: CLAY text sets and read-outs, CRL read-out (CPU)
    python r28_clay_crl.py cells  --enc siglip1  test cells with those picks (CPU)
    python r28_clay_crl.py replay                CLIP L/14: recorded picks, 24 cells against r17/r18's records (CPU)
    python r28_clay_crl.py plain                 plain SigLIP 1 beside the new cells (CPU)

Picks are made on the validation split only (`ValDataEnc` fixes `val`); `cells` reads the test split and never picks.
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
for _sub in ('20261103_outside_methods', '20261103_new_scores'):
    if str(TEST / _sub) not in sys.path:
        sys.path.insert(0, str(TEST / _sub))

import om_clay  # noqa: E402
import om_data  # noqa: E402
import om_paths as op  # noqa: E402
import om_stage1 as s1  # noqa: E402
import ns_data  # noqa: E402
import ns_metrics  # noqa: E402
import ns_picks  # noqa: E402
import ns_ret  # noqa: E402
import ac_crl  # noqa: E402
import rt_crl  # noqa: E402

OUT = Path('/data/SSD2/cosir_r28/20261112_siglip_main/clay_crl')
TEXT = OUT / 'text'
RECORDS = ROUND / 'records'
ENCODERS = ('clip_l14', 'siglip1')
BACKBONE = {'siglip1': 'siglip_so400m'}                 # scripts/extract_features.py's name of the store folder
WIDTH = {'clip_l14': 768, 'siglip1': 1152}
ASPECTS, SETS, SEEDLESS = op.ASPECTS, op.SETS, None
TOL_REPLAY = 1e-6
TEXT_SETS = ('A', 'B')
OLD_PICKS = op.OUT / 'stage1' / 'om_picks.json'          # r18's recorded CLAY picks (committed copy: records/)
R18_LABELS = TEST / '20261103_outside_methods/records/labels.json'
R17_VERDICTS = TEST / '20261103_new_scores/records/verdicts.json'
R17_PICKS = TEST / '20261103_new_scores/records/stage1_picks.json'
R14_SIGLIP1 = TEST / '20261102_backbones_sweep/records/stage1_siglip1.json'


def check_enc(enc):
    assert enc in ENCODERS, f'encoder {enc!r}: one of {ENCODERS}'
    return enc


# ================================================================================================ texts
def source_texts(text_set, aspect):
    """(texts, source path): the unchanged text lists. Set A = CRL's 50 values per aspect in r11's template (the same
    rows `om_clay.text_set('A', ...)` reads); set B = r18's 100-value lists in its template."""
    assert text_set in TEXT_SETS and aspect in ASPECTS
    base = (Path(op.CLAY_TEXT_A[aspect]) if text_set == 'A' else Path(op.CLAY_B_DIR) / f'{aspect}_text.npy').with_suffix('.json')
    side = json.loads(base.read_text())
    texts = side['texts']
    assert len(texts) == side['shape'][0] > 1 and all(isinstance(t, str) and t for t in texts)
    return texts, base


def encode_siglip1_texts(device='cuda', out=TEXT):
    """Encode every text of sets A and B with SigLIP 1's text tower exactly as the stores' text side was made:
    `src.data.feature_extract.load_encoder('siglip_so400m').encode_texts` (google/siglip-so400m-patch14-384, bf16, lower-cased,
    padded to 64 tokens, no prompt template beyond the one already in the text, L2-normalised float32)."""
    sys.path.insert(0, str(TEST.parents[1]))
    from src.data import feature_extract as fe
    enc = fe.load_encoder(BACKBONE['siglip1'], device)
    meta = {}
    for ts in TEXT_SETS:
        for a in ASPECTS:
            texts, src = source_texts(ts, a)
            x = enc.encode_texts(texts).astype(np.float32)
            assert x.shape == (len(texts), WIDTH['siglip1']) and np.isfinite(x).all()
            assert np.abs(np.linalg.norm(x.astype(np.float64), axis=1) - 1).max() < 1e-5
            d = Path(out) / ts
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f'{a}_text.npy', x)
            side = dict(model=fe.SIGLIP1_REPO, backbone=BACKBONE['siglip1'], encoder_class=type(enc).__name__,
                        prompt_template='none (the text is the unchanged r11 / r18 text, already templated)',
                        lowercase=True, max_length=64, padding='max_length', dtype=str(enc.dtype), normalised='L2, float32',
                        texts=texts, shape=list(x.shape), sha256=om_data.array_sha256(x), source=str(src), device=device)
            (d / f'{a}_text.json').write_text(json.dumps(side, indent=1) + '\n')
            meta[f'{ts}/{a}'] = dict(n=len(texts), sha256=side['sha256'])
    store_check(enc, out)
    return meta


def store_check(enc, out=TEXT, n=128):
    """The same encoder on the first `n` ArtELingo captions against the store's `txt.npy` rows (float16): lowest cosine."""
    root = Path('/data/SSD2/pre_extract/artelingo_full/siglip_so400m')
    index = json.loads((root / 'index.json').read_text())
    ann = json.load(open('/data/PDD/artelingo/artelingo_train.json'))
    texts = [ann[m['sample_id']]['caption'] for m in index['texts'][:n]]
    ref = np.load(root / 'txt.npy', mmap_mode='r')[:n].astype(np.float32)
    got = enc.encode_texts(texts).astype(np.float32)
    cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))
    res = dict(n=n, min_cos=float(cos.min()), mean_cos=float(cos.mean()))
    (Path(out) / 'store_check.json').write_text(json.dumps(res, indent=1) + '\n')
    return res


def text_rows(text_set, aspect, enc='clip_l14', root=TEXT):
    """Unit float32 text rows of `text_set` for `aspect` under encoder `enc`. CLIP L/14: r18's own arrays; SigLIP 1: the
    arrays `encode_siglip1_texts` wrote, after their sidecar's hash and text list are checked against the source lists."""
    check_enc(enc)
    if enc == 'clip_l14':
        return om_clay.text_set(text_set, aspect)
    d = Path(root) / text_set
    x = np.load(d / f'{aspect}_text.npy', allow_pickle=False)
    side = json.loads((d / f'{aspect}_text.json').read_text())
    texts, _ = source_texts(text_set, aspect)
    assert x.dtype == np.float32 and x.shape == (len(texts), WIDTH[enc]) and np.isfinite(x).all(), (x.dtype, x.shape)
    assert side['texts'] == texts, f'{d}/{aspect}: the encoded texts are not the source lists'
    assert side['sha256'] == om_data.array_sha256(x), f'{d}/{aspect}: array differs from its sidecar'
    assert np.abs(np.linalg.norm(x.astype(np.float64), axis=1) - 1).max() < 1e-4, 'text rows must be unit rows'
    return x


# ================================================================================================ rows
_MEMO = {}


def _memo(key, make, limit=14):
    if key not in _MEMO:
        if len(_MEMO) >= limit:
            _MEMO.pop(next(iter(_MEMO)))
        _MEMO[key] = make()
    return _MEMO[key]


def _no_write(*a, **k):
    raise RuntimeError('r28: a r18 loader tried to write a cache file; refusing')


@contextlib.contextmanager
def _frozen_r18_cache():
    """CLIP L/14 CLAY vectors come from r18's cache (`om_data.clay`); a sidecar mismatch would make it write: refuse."""
    old = om_data._atomic
    om_data._atomic = _no_write
    try:
        yield
    finally:
        om_data._atomic = old


def images(set_, split, enc='clip_l14'):
    """Unit float32 image rows of the split (every row, whatever its labels), in `ns_data.keys(set, split)` order."""
    check_enc(enc)
    return _memo(('img', set_, split, enc), lambda: ns_data.plain(set_, split, enc))


def clay_vectors(set_, split, aspect, text_set, enc='clip_l14', text_root=TEXT):
    """CLAY vectors [n, width] float32 unit rows. CLIP L/14: `om_data.clay` unchanged. Else: the same math (`om_clay.subspace`
    at `op.clay_rank`, `om_clay.condition` with the alignment mean over every row of the split) on `enc`'s rows and texts."""
    check_enc(enc)
    if enc == 'clip_l14':
        with _frozen_r18_cache():
            return om_data.clay(set_, split, aspect, text_set)

    def make():
        text = text_rows(text_set, aspect, enc, text_root)
        P, mu = om_clay.subspace(text, op.clay_rank(text.shape[0]))
        out = om_clay.condition(images(set_, split, enc), P, mu)
        assert out.shape == (len(ns_data.keys(set_, split)), WIDTH[enc]) and out.dtype == np.float32
        return out
    return _memo(('clay', set_, split, aspect, text_set, enc, str(text_root)), make)


def crl_vectors(set_, split, aspect, enc='clip_l14', text_root=TEXT):
    """CRL vectors [n, 50] float32. CLIP L/14: `ns_data.crl` unchanged. Else `ac_crl.vectors(unit image rows, text rows)`:
    the cosines of the 50 values (text set A = CRL's rows), as r11's form, on `enc`'s image and text rows."""
    check_enc(enc)
    if enc == 'clip_l14':
        return ns_data.crl(set_, split, aspect)

    def make():
        text = text_rows('A', aspect, enc, text_root)
        assert text.shape[0] == rt_crl.KEEP, text.shape
        return ac_crl.vectors(images(set_, split, enc), text)
    return _memo(('crl', set_, split, aspect, enc, str(text_root)), make)


class ValDataEnc(s1.ValData):
    """r18's validation reader with the encoder as a parameter: `clay` goes through `clay_vectors`; nothing takes a split."""

    def __init__(self, enc, text_root=TEXT):
        super().__init__()
        self.enc, self.text_root = check_enc(enc), text_root

    def clay(self, set_, aspect, text_set):
        return clay_vectors(set_, s1.SPLIT, aspect, text_set, self.enc, self.text_root)

    def crl(self, set_, aspect):
        return crl_vectors(set_, s1.SPLIT, aspect, self.enc, self.text_root)

    def text_sha(self, text_set, aspect):
        return self.od.array_sha256(text_rows(text_set, aspect, self.enc, self.text_root))


# ================================================================================================ picks (validation only)
def encoder_units(set_, enc):
    """(unit {aspect: r}, source): X-c*'s recorded validation mAP range on the set, the unit of the read-out score (D7)."""
    return ns_picks.ranges(set_, enc)


def crl_pick(data, set_, unit, source, workers=8, out=OUT, tag=''):
    """CRL's read-out on the set's validation split: r11's menu at width 50, r11's rule (`s1.readout_doc` = `ac_select.arm_table`),
    one pick across the three aspects, the unit given."""
    keys = s1.audit_validation_only(data, set_)
    ns_picks.check_order(set_, keys)
    labels = {a: np.asarray(data.labels(set_, a), dtype=np.int64) for a in ASPECTS}
    xs = {(a, SEEDLESS): data.crl(set_, a) for a in ASPECTS}
    for k, x in xs.items():
        assert x.dtype == np.float32 and x.shape == (len(keys), rt_crl.KEEP), (k, x.dtype, x.shape)
    doc = s1.readout_doc(xs, labels, unit, workers=workers)
    rel = f'stage1/tables/crl__{set_}__{data.enc}{tag}.json'
    ns_picks.ns_paths.write_json(Path(out) / rel, dict(arm='crl', pick=doc['pick'], table=doc['table'], ranges=doc['ranges'],
                                                       unit_source=source, dropped=doc['dropped'], n_val=len(keys)))
    return ns_picks._record(doc, source, len(keys), rel)


def make_picks(enc='clip_l14', workers=8, out=OUT, crl_unit_variants=True):
    """All validation picks of one encoder: per set, CLAY's text set for clustering (D8) and for retrieval (D7) with the
    retrieval pick, and CRL's read-out. Units: `enc`'s c* validation mAP ranges (for CLIP L/14 these are r18's CLAY units and
    r17's CRL-on-MMArt units). Validation only."""
    data = ValDataEnc(enc)
    picks = {}
    for set_ in SETS:
        unit, src = encoder_units(set_, enc)
        picks.update(s1.clay_retrieval(data, set_, unit=unit, source=src, workers=workers, out=out))
        picks[f'clay_textset/clu/{set_}'] = s1.clay_clustering_choice(data, set_)
        picks[f'crl/{set_}'] = crl_pick(data, set_, unit, src, workers=workers, out=out)
        if crl_unit_variants and set_ == 'artelingo':
            # r11 made CRL's ArtELingo pick in units of Qwen c*'s range: the pick under that unit, as a sensitivity check
            q_unit, q_src = ns_picks.ranges(set_, 'qwen')
            picks[f'crl_qwen_unit/{set_}'] = crl_pick(data, set_, q_unit, q_src, workers=workers, out=out, tag='__qwenunit')
    return picks


# ================================================================================================ test cells
def class_count(set_, aspect):
    return op.CLASS_COUNT[(set_, aspect)]


def held(set_):
    return op.HELD_SPLIT[set_]


def clu_cell(rows, set_, aspect, k=None):
    """AMI at `k` (default: the class count) on the held split: k-means as r10 (`ns_metrics.partitions`), mean over the 3
    states, labelled rows only (`ns_metrics.score_partitions`)."""
    k = class_count(set_, aspect) if k is None else k
    y = ns_data.labels(set_, held(set_), aspect)
    parts = ns_metrics.partitions(rows, k, set_)
    return dict(value=float(ns_metrics.score_partitions('ami', y, parts)),
                per_state=[float(ns_metrics.score_partitions('ami', y, parts[i:i + 1])) for i in range(parts.shape[0])],
                k=int(k), n_labelled=int((y >= 0).sum()), n_clusters_used=[int(len(np.unique(p))) for p in parts])


def ap_cell(get_rows, set_, aspect, pick):
    """Mean AP at the read-out `pick`: `ns_ret.arm_ap` with the map fitted on all validation rows, scored on the held split."""
    y = ns_data.labels(set_, held(set_), aspect).astype(np.int64)
    res = ns_ret.arm_ap(lambda s: get_rows('val'), lambda s: get_rows(held(set_)), pick, y, None)
    return dict(value=float(res['point']), pick=pick, n_queries=int(len(res['rows'])))


def score_cells(picks, enc='clip_l14', plain=False, sets=SETS, methods=('clay', 'crl'), aspects=ASPECTS, tasks=('ami', 'map')):
    """Test cells {id: {value, ...}} of CLAY and CRL: AMI at the class counts (Table 2 on ArtELingo; Table 4 on MMArt) and
    mAP at the pick. `picks` is the record of `make_picks` or of r18 / r11 / r17 (`recorded_picks`)."""
    check_enc(enc)
    cells = {}
    for set_ in sets:
        for aspect in aspects:
            if 'clay' in methods:
                ts_clu = picks[f'clay_textset/clu/{set_}']['set']
                ts_ret = picks[f'clay_textset/ret/{set_}']['set']
                if 'ami' in tasks:
                    cells[f'{set_}/{aspect}/ami/clay'] = dict(
                        clu_cell(clay_vectors(set_, held(set_), aspect, ts_clu, enc), set_, aspect), text_set=ts_clu)
                if 'map' in tasks:
                    cells[f'{set_}/{aspect}/map/clay'] = dict(
                        ap_cell(lambda split: clay_vectors(set_, split, aspect, ts_ret, enc), set_, aspect,
                                picks[f'clay/{set_}']['pick']), text_set=ts_ret)
            if 'crl' in methods:
                if 'ami' in tasks:
                    cells[f'{set_}/{aspect}/ami/crl'] = clu_cell(crl_vectors(set_, held(set_), aspect, enc), set_, aspect)
                if 'map' in tasks:
                    cells[f'{set_}/{aspect}/map/crl'] = ap_cell(lambda split: crl_vectors(set_, split, aspect, enc), set_, aspect,
                                                                  picks[f'crl/{set_}']['pick'])
            if plain:
                if 'ami' in tasks:
                    cells[f'{set_}/{aspect}/ami/plain'] = clu_cell(images(set_, held(set_), enc), set_, aspect)
                if 'map' in tasks:
                    rows = ns_ret._Rows(set_, enc)
                    y = ns_data.labels(set_, held(set_), aspect)
                    pk = ns_ret.reused_pick('plain', set_, enc)
                    res = rows.ap('plain', aspect, pk, y, 8)
                    cells[f'{set_}/{aspect}/map/plain'] = dict(value=float(res['point']), pick=pk, n_queries=int(len(res['rows'])))
    return cells


# ================================================================================================ replay of the existing CLIP L/14 rows
def recorded_picks():
    """The picks the existing CLIP L/14 rows were made with: CLAY from r18's committed record (text sets, read-outs), CRL from r11
    (ArtELingo) and r17 (MMArt)."""
    old = json.loads(OLD_PICKS.read_text())
    r17 = json.loads(R17_PICKS.read_text())
    r11 = ns_ret._r11_picks()
    picks = {k: old[k] for k in old if k.startswith('clay')}
    picks['crl/artelingo'] = dict(pick=r11['crl'])
    picks['crl/mmart'] = dict(pick=r17['crl/mmart/clip_l14']['pick'])
    return picks


def recorded_values():
    """{cell id of `score_cells`: recorded CLIP L/14 value} from r18's labels.json (CLAY) and r17's verdicts.json (CRL)."""
    lab = {c['id']: c['comp'] for c in json.loads(R18_LABELS.read_text())['cells']}
    ver = {c['id']: c['comp'] for c in json.loads(R17_VERDICTS.read_text())['cells']}
    out = {}
    for set_ in SETS:
        for a in ASPECTS:
            k = class_count(set_, a)
            out[f'{set_}/{a}/ami/clay'] = lab[f'SC3/{set_}/clip_l14/{a}/ami/{k}/clay']
            out[f'{set_}/{a}/map/clay'] = lab[f'SC4/{set_}/clip_l14/{a}/map/-/clay']
            out[f'{set_}/{a}/ami/crl'] = ver[f'SC5/{set_}/clip_l14/{a}/ami/{k}/crl']
            out[f'{set_}/{a}/map/crl'] = ver[f'SC6/{set_}/clip_l14/{a}/map/-/crl']
    return out


def compare(cells, recorded, tol=TOL_REPLAY):
    """{id: {value, recorded, diff}} for the cells that have a record; `ok` when every |diff| is within tol."""
    rows = {i: dict(value=c['value'], recorded=recorded[i], diff=c['value'] - recorded[i]) for i, c in cells.items() if i in recorded}
    return dict(cells=rows, max_abs_diff=max((abs(r['diff']) for r in rows.values()), default=None), n=len(rows),
                ok=bool(rows) and all(abs(r['diff']) <= tol for r in rows.values()))


def replay(sets=SETS, methods=('clay', 'crl')):
    cells = score_cells(recorded_picks(), 'clip_l14', sets=sets, methods=methods)
    return compare(cells, recorded_values())


# ================================================================================================ CLI
def _dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + '\n')
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('texts', 'picks', 'cells', 'replay', 'plain', 'replay-picks'))
    ap.add_argument('--enc', default=None, choices=ENCODERS, help='required for picks and cells (no default: name the encoder)')
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args(argv)
    out, t0 = Path(a.out), time.monotonic()
    assert a.enc or a.step not in ('picks', 'cells'), f'{a.step} needs --enc'
    if a.step == 'texts':
        meta = encode_siglip1_texts(out=out / 'text')
        print(f'texts: {len(meta)} arrays -> {out / "text"} ({time.monotonic() - t0:.0f}s)')
    elif a.step in ('picks', 'replay-picks'):
        enc = 'clip_l14' if a.step == 'replay-picks' else a.enc
        picks = make_picks(enc, out=out)
        name = f'picks_{enc}.json'
        _dump(out / name, picks)
        print(f'{name}: ' + ', '.join(f'{k}={v.get("pick", v.get("set"))}' for k, v in picks.items()) + f' ({time.monotonic() - t0:.0f}s)')
    elif a.step == 'cells':
        picks = json.loads((out / f'picks_{a.enc}.json').read_text())
        cells = score_cells(picks, a.enc)
        _dump(out / f'cells_{a.enc}.json', cells)
        print(f'cells_{a.enc}.json: {len(cells)} cells ({time.monotonic() - t0:.0f}s)')
    elif a.step == 'plain':
        cells = score_cells({}, 'siglip1', plain=True, methods=())
        _dump(out / 'cells_plain_siglip1.json', cells)
        print(f'cells_plain_siglip1.json: {len(cells)} cells ({time.monotonic() - t0:.0f}s)')
    else:
        res = replay()
        _dump(out / 'replay_clip_l14.json', res)
        print(f'replay: {res["n"]} cells, max |diff| {res["max_abs_diff"]:.3e}, ok={res["ok"]} ({time.monotonic() - t0:.0f}s)')
        return 0 if res['ok'] else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
