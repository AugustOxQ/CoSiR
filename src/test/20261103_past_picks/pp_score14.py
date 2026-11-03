"""r19 ticket 07: r14's ArtELingo and Affection scores (SC1 to SC6) and stage 1 for DINOv2, CLIP B/32 and CLIP B/16.

Spec: docs/superpowers/specs/2026-11-03-past-picks-design.md, SC1 to SC6, §3.4, §3.5 stages 1 and 2, D21. Nothing of r14's
statistics is rewritten: `import pp_register` puts the new three in r14's encoder tables, and each score is the function
that produced r14's numbers, called on the new encoder's own arms:

  stage 1 (validation only; written to OUT/stage1/ in r14's forms; ticket 12 merges and commits them)
    ArtELingo   `bs_artclu.clustering_ranges`, `bs_artret.validation_ranges`, `bs_artret.stage1_picks`: X-c*'s ranges, its
                read-out pick (own ranges), the frozen image's pick by r11's stage 4 rule with r11's r_a (D21; r14 D19's
                form: `bs_artret.ENC[X]['r11_arm']` is None), the teacher's stored pick.
    Affection   `bs_affection.photo_stage1` and `bs_affection.frozen_readout` (the photo model's ranges and pick; the frozen
                image's pick by r12's rule with r12's r_ret,mAP). Body copied from `bs_affection.stage1` (bs_affection.py
                :223-242) only to return the record instead of writing into `<root>/stage1`.
  stage 2 (held; reads stage 1 ONLY through `pp_paths.read_committed_record`, before any held row)
    artelingo_clu   SC1, SC3, SC6: `bs_artclu.frozen_margin`, `matched_margin`, `teacher_margin`, `aspect_cells` (r14's cells),
                    on partitions fixed as r15's replay of r14 does (`mt_clu.replay_iii`, bs_artclu's route equal to 1e-9
                    there): `ct_score.output_partitions(x, 'k1')['r20']` for X-c*, the frozen image and the teacher; the
                    matched arm's partitions from its held probabilities through `ct_score.output_partitions(p, 'percept')`
                    (M3, M6: their stored `full/held/<M>/P3__s<k>.npz`; M5: `pp_matched.held_probs`).
    artelingo_ret   SC2, SC6: `bs_artret.score_encoder` unchanged (frozen image at the record's pick through
                    `bs_artret.score_frozen_held`; teacher: r11's stored instructed outputs).
    affection       SC4, SC5, SC6 x 2: `bs_affection.score_cells`; body copied from `bs_affection.score` (:332-349) to read the
                    committed record instead of OUT/stage1.
  Cells: `OUT/stage2/cells_<part>.json`, r14's cell dicts plus `n_rows` (rows of the held population: 12,281 ArtELingo held
  paintings, 8,501 Affection test photos) and, for clustering, `n_labelled`; `n_queries` stays on retrieval cells.

CLI (`python pp_score14.py -h`): stage1 | score. Smoke: PP_OUTPUTS and PP_SMOKE=1 (fewer draws, other `records/`).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import pp_paths as PP
import pp_register  # noqa: F401  (registers the new three; second, before r14's and r15's modules are used)
import pp_outcome

import ac_data
import ac_stats
import bs_affection
import bs_artclu
import bs_artret
import bs_paths
import mt_clu
import pf_stage3
import rt_data

NEW = tuple(PP.NEW_ENCODERS)
GATE_ENCODERS = ('siglip1', 'clip_l14')            # r14's records that gate (vii) replays; never scored here
ASPECTS = tuple(PP.ASPECTS)
SEEDS = tuple(PP.SEEDS)
N_HELD = bs_artclu.rs_paths.N_HELD                  # 12,281 ArtELingo held paintings
PARTS = ('artelingo_clu', 'artelingo_ret', 'affection')
AE, AF = 'artelingo_held', 'affection_test'
EXPECT = {                                          # part -> the cells it holds: (sc, aspect, task, data)
    'artelingo_clu': sorted((sc, a, 'clu', AE) for a in ASPECTS for sc in ('SC1', 'SC3', 'SC6')),
    'artelingo_ret': sorted((sc, a, 'ret', AE) for a in ASPECTS for sc in ('SC2', 'SC6')),
    'affection': sorted([('SC4', 'emotion', 'clu', AF), ('SC5', 'emotion', 'ret', AF), ('SC6', 'emotion', 'clu', AF),
                         ('SC6', 'emotion', 'ret', AF)]),
}
STAGE1_FILES = dict(artclu='artclu_ranges_{enc}.json', artret_ranges='artret_ranges_{enc}.json',
                    artret_picks='artret_picks_{enc}.json', affection='affection_ranges_{enc}.json')


def _assert_encoder(enc):
    assert enc in NEW or enc in GATE_ENCODERS, f'{enc}: not one of the new three {NEW} (nor a gate replay encoder)'


def draws(task, n_draws=None):
    """The bootstrap draws of a run: the protocol's (clustering 2,000, retrieval 10,000); smoke runs may use fewer."""
    if n_draws is None:
        return (PP.SMOKE_DRAWS if PP.SMOKE else PP.N_DRAWS)[task]
    return int(n_draws)


def require_protocol_draws(part, n_draws):
    """Outside smoke a stage 2 run uses the protocol's draws (ticket 11 refuses cells with other counts)."""
    if n_draws is not None and not PP.SMOKE:
        tasks = {'artelingo_clu': ('clu',), 'artelingo_ret': ('ret',), 'affection': ('clu', 'ret')}[part]
        assert all(n_draws == PP.N_DRAWS[t] for t in tasks), f'{n_draws} draws: only smoke runs may use other counts'


# ============================================================================================ stage 1
def artclu_block(enc, result):
    """The clustering ranges record, as `bs_artclu.write_ranges` writes it (bs_artclu.py:302-309)."""
    return dict(encoder=enc, kind='artelingo_clustering_validation_ranges', **result)


def artret_block(enc, result):
    """The retrieval ranges record, as `bs_artret.write_ranges` writes it (bs_artret.py:275)."""
    return dict(encoder=enc, kind='artelingo_retrieval_validation_ranges', **result)


def stage1_artelingo(enc, root=None, workers=8):
    """ArtELingo stage 1 of one encoder, validation only: {'artclu', 'artret_ranges', 'artret_picks'} in r14's forms
    (r14's CLI steps `bs_artclu ranges`, `bs_artret ranges`, `bs_artret pick`, run in one call, nothing written).
    `root`: X-c*'s run root (default `PP.CSTAR_ROOT[enc]`; the gate passes r14's)."""
    _assert_encoder(enc)
    root = Path(root or PP.CSTAR_ROOT[enc])
    clu = bs_artclu.clustering_ranges(root)
    ret = bs_artret.validation_ranges(root)
    picks = bs_artret.stage1_picks(enc, root, ret['ranges'], workers)
    blocks = dict(artclu=artclu_block(enc, clu), artret_ranges=artret_block(enc, ret), artret_picks=picks)
    check_artelingo_blocks(enc, blocks)             # D21, before anything is written (r14's stage1_picks is called as is)
    return blocks


def stage1_affection(enc, root=None, workers=8):
    """Affection stage 1 of one encoder (validation only), r14's record of `bs_affection.stage1`, returned, not written.
    `root`: where `exports/affection/photo_<enc>/val__s<k>.npz` live (default `PP.OUT`; the gate passes r14's)."""
    _assert_encoder(enc)
    root = Path(PP.OUT if root is None else root)
    arm = f'photo_{enc}'
    keys, labels, dist = ac_data.keys('val'), ac_data.labels('val'), ac_data.emotion_dist('val')
    pa = bs_affection.PhotoArm(arm, root)
    embed_val = {s: pa.embed('val', s) for s in SEEDS}
    stored_val = {s: pa.stored('val', s) for s in SEEDS}
    r12_ranges = ac_stats.load_ranges(bs_affection.R12_RANGES)
    photo = bs_affection.photo_stage1(embed_val, stored_val, labels, dist, keys, r12_ranges=r12_ranges, workers=workers)
    r12_picks = json.loads(bs_affection.R12_PICKS.read_text())['affection']
    frozen = bs_affection.frozen_readout(enc, bs_affection.frozen_embed(enc, keys), labels, keys, r12_ranges, r12_picks,
                                         workers)
    frozen['view'] = bs_affection.VIEW[enc]
    sources = {f'val__s{s}': json.loads(bs_affection.export_path(arm, 'val', s, root).with_suffix('.json').read_text())['sha256']
               for s in SEEDS}
    return bs_affection.stage1_record(enc, photo, frozen, sources)


def write_stage1(enc, blocks, out=None):
    """Write stage 1 blocks to `<out>/stage1/<r14's file names>` (default `PP.OUT`). A file that exists with other content is
    refused (picks are frozen once written, as `bs_affection.write_record`). Returns {block: path}."""
    assert set(blocks) <= set(STAGE1_FILES), sorted(blocks)
    folder = Path(PP.OUT if out is None else out) / 'stage1'
    paths = {}
    for key, block in blocks.items():
        path = folder / STAGE1_FILES[key].format(enc=enc)
        text = json.dumps(block, indent=1, allow_nan=False) + '\n'
        if path.exists():
            assert path.read_text() == text, f'{path} holds other content: picks are frozen once written'
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.pending.json')
            tmp.write_text(text)
            tmp.replace(path)
        paths[key] = path
    return paths


def stage1_record(enc, out=None):
    """r14's `records/stage1_<enc>.json` content (without `written`) from the four stage 1 files of `out`. Ticket 12 commits it."""
    folder = Path(PP.OUT if out is None else out) / 'stage1'
    blocks = {k: json.loads((folder / name.format(enc=enc)).read_text()) for k, name in STAGE1_FILES.items()}
    return dict(encoder=enc, kind='stage1_record', **blocks)


# ============================================================================================ the committed record
def check_artelingo_blocks(enc, blocks):
    """The ArtELingo part of `check_record` (`blocks`: a dict with `artclu`, `artret_ranges`, `artret_picks`): encoder names,
    finite ranges, X-c*'s picks made with its own ranges, and D21 for a computed frozen pick."""
    for key in ('artclu', 'artret_ranges', 'artret_picks'):
        assert blocks[key]['encoder'] == enc, (key, blocks[key]['encoder'], enc)
    for key in ('artclu', 'artret_ranges'):
        r = blocks[key]['ranges']
        assert set(r) == set(ASPECTS) and all(np.isfinite(v) and v > 0 for v in r.values()), (key, r)
    picks = blocks['artret_picks']
    assert picks['cstar']['ranges'] == blocks['artret_ranges']['ranges'], 'X-c* picks were made with other ranges'
    assert picks['cstar']['pick'] and picks['frozen']['pick'] and picks['teacher']['pick']
    if bs_artret.ENC[enc]['r11_arm'] is None:               # a computed frozen pick (SigLIP 1, the new three): D19, D21
        assert picks['frozen']['ranges'] == bs_artret.r11_ranges(), \
            f'{enc}: the frozen pick must be made with r11\'s r_a, not X-c*\'s own ranges (D21)'
        assert picks['r11_ranges_sha256'] == bs_artret.sha_file(bs_artret.R11_RANGES_FILE), 'r11 ranges file changed'


def check_record(enc, rec):
    """The stage 1 record's blocks agree with each other and with the rules of D21: encoder names, picks made with the
    ranges they say, a new encoder's frozen picks made with r11's / r12's ranges (never X-c*'s own)."""
    assert rec['encoder'] == enc, (enc, rec['encoder'])
    for key in ('artclu', 'artret_ranges', 'artret_picks', 'affection'):
        assert key in rec, f'stage 1 record of {enc} lacks {key!r}'
    check_artelingo_blocks(enc, rec)
    aff = rec['affection']
    assert aff['encoder'] == enc, ('affection', aff['encoder'], enc)
    assert aff['kind'] == 'affection_validation_ranges_and_readouts'
    for m in ('clu', 'map', 'ndcg'):
        per_seed = aff['ranges_doc'][m]['per_seed']
        assert len(per_seed) == 5 and aff['ranges'][m] == aff['ranges_doc'][m]['r'] == ac_stats.seed_range(per_seed), m
        assert np.isfinite(aff['ranges'][m]) and aff['ranges'][m] > 0, m
    assert aff['photo_readout']['pick'] in aff['photo_readout']['scores'] and aff['frozen']['pick']
    r12_r = ac_stats.load_ranges(bs_affection.R12_RANGES)['map']['r']
    allowed = (None, r12_r) if enc in bs_affection.R12_PICK_ARM else (r12_r,)
    assert aff['frozen']['r'] in allowed, \
        f'{enc}: the Affection frozen pick must use r12\'s r_ret,mAP {r12_r} (D21), got {aff["frozen"]["r"]!r}'
    return rec


def committed_record(enc):
    """Stage 1 of `enc` as committed (`PP.read_committed_record`: refuses unless equal to git HEAD), checked."""
    rec = PP.read_committed_record(f'stage1_{enc}.json')
    return check_record(enc, rec)


# ============================================================================================ matched arm (SC3)
def _load_prob(path):
    with np.load(path, allow_pickle=False) as z:
        return z['prob']


def _list_sha256(path):
    """The held-list hash in the sidecar of a stored probabilities file (`pp_matched.held_probs` checks the same field)."""
    return json.loads(Path(path).with_suffix('.json').read_text())['list_sha256']


def matched_probs(enc):
    """{seed: float32 [12281, 40]} held probabilities of the matched arm, in r14's held order. M3, M6 (and M7 for the gate):
    the arm's stored `full/held/<M>/P3__s<k>.npz`, each sidecar's `list_sha256` equal to `rs_paths.HELD_SHA256`; DINOv2's M5:
    `pp_matched.held_probs` (checks the same)."""
    if enc == 'dinov2_large':
        import pp_matched
        probs = pp_matched.held_probs(enc)
    else:
        spec = bs_artclu.ENC[enc]
        paths = {s: Path(spec['root']) / f'full/held/{spec["matched"]}/P3__s{s}.npz' for s in SEEDS}
        for s, path in paths.items():
            assert _list_sha256(path) == bs_artclu.rs_paths.HELD_SHA256, f'{path}: not the held list'
        probs = {s: _load_prob(path) for s, path in paths.items()}
    assert sorted(probs) == list(SEEDS), f'{enc}: matched seeds {sorted(probs)}'
    for s, p in probs.items():
        assert p.shape == (N_HELD, 40) and np.isfinite(p).all(), f'{enc} seed {s}: matched probabilities {p.shape}'
    return probs


def matched_fixed(enc):
    """The matched arm's fixed partitions {seed: {'r20': [3, n], 'own': [1, n]}} (r15's route, `mt_clu.matched_from_probs`)."""
    return mt_clu.matched_from_probs(matched_probs(enc))


# ============================================================================================ ArtELingo clustering
def _unit32(x):
    x = np.asarray(x).astype(np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    assert (norm > 0).all()
    return (x / norm).astype(np.float32)


def clu_cells(enc, aspect, labels, c_fixed, frozen, matched, teacher, ranges, n_draws, workers=8):
    """SC1, SC3, SC6 cells of one aspect on arrays: r14's three margins and `bs_artclu.aspect_cells` (the same cell dicts),
    arm labels F<k> and M<k> as the older rounds name them. + `n_rows`, `n_labelled`."""
    spec = bs_artclu.ENC[enc]
    frozen_name = bs_paths.FROZEN_ARM[enc]
    margins = dict(frozen=bs_artclu.frozen_margin(c_fixed, frozen, labels, n_draws, workers, 'C', frozen_name),
                   matched=bs_artclu.matched_margin(c_fixed, matched, labels, n_draws, workers, 'C', spec['matched']),
                   teacher=bs_artclu.teacher_margin(c_fixed, teacher, labels, n_draws, workers))
    cells = bs_artclu.aspect_cells(enc, aspect, 'C', margins, ranges, n_draws, spec['matched'], frozen_name)
    for c in cells:
        c['n_rows'] = int(len(labels))
        c['n_labelled'] = int((np.asarray(labels) >= 0).sum())
    return cells


def run_artelingo_clu(enc, rec, n_draws=None, workers=8, root=None):
    """The nine ArtELingo clustering cells (held) of `enc`. `rec`: the stage 1 record (ranges for SC6's unit)."""
    n_draws = draws('clu', n_draws)
    ranges = rec['artclu']['ranges']
    held_dir = Path(root or PP.CSTAR_ROOT[enc]) / 'held' / PP.CSTAR_ARM
    names, sha, labels = bs_artclu.held_setup()
    frozen = mt_clu.single_fixed(bs_artclu.store_loader(enc, names))
    matched = matched_fixed(enc)
    cand = bs_artclu.S0.Candidate('C', held_dir, PP.CSTAR_ARM)
    cells = []
    for aspect in ASPECTS:
        c_fixed = mt_clu.cstar_fixed({s: bs_artclu.S0.load_candidate(cand, 'full', aspect, s)[0] for s in SEEDS})
        teacher = mt_clu.single_fixed(_unit32(rt_data._instructed('held', aspect, names)))
        got = clu_cells(enc, aspect, labels[aspect], c_fixed, frozen, matched, teacher, ranges, n_draws, workers)
        cells += got
        print(f'{enc} {aspect}: ' + ', '.join(f'{c["sc"]} d={c["d"]:+.4f}' for c in got), flush=True)
    return cells


# ============================================================================================ ArtELingo retrieval
def run_artelingo_ret(enc, rec, outputs, n_draws=None, workers=8, root=None):
    """The six ArtELingo retrieval cells (held) of `enc`: `bs_artret.score_encoder` with the record's ranges and picks."""
    n_draws = draws('ret', n_draws)
    ranges, picks = rec['artret_ranges']['ranges'], rec['artret_picks']
    cells = bs_artret.score_encoder(enc, Path(root or PP.CSTAR_ROOT[enc]), ranges, picks, outputs, n_draws, workers)
    for c in cells:
        c['n_rows'] = N_HELD
    return cells


# ============================================================================================ Affection
def run_affection(enc, rec, root=None, cache=None, n_clu=None, n_ret=None, workers=8):
    """SC4, SC5 and the two SC6 cells on the Affection test photos (copy of `bs_affection.score`, bs_affection.py:332-349,
    reading the record `rec['affection']` instead of OUT/stage1). `root`: where the photo model's exports live."""
    n_clu, n_ret = draws('clu', n_clu), draws('ret', n_ret)
    a = rec['affection']
    root = Path(PP.OUT if root is None else root)
    r = {'clu': a['ranges']['clu'], 'map': a['ranges']['map']}
    r12_picks = json.loads(bs_affection.R12_PICKS.read_text())['affection']
    pa = bs_affection.PhotoArm(f'photo_{enc}', root)
    val_keys, test_keys, labels = pa.keys['val'], pa.test_keys(), ac_data.labels('test')
    cache = Path(cache or Path(PP.OUT) / 'stage2' / 'partitions_affection')
    comps = pf_stage3.load_comparators(r12_picks, len(labels))
    comparators = dict(teacher=comps['teacher'],
                       frozen=bs_affection.frozen_comparator(enc, a['frozen']['pick'], val_keys, test_keys, labels, cache,
                                                             workers))
    cand = pa.candidate(a['photo_readout']['pick'])
    res = bs_affection.score_cells(enc, cand, r, labels, comparators, cache, n_clu, n_ret, workers)
    for c in res['cells']:
        c['n_rows'] = int(len(labels))
    return res


# ============================================================================================ cells
def cells_path(part, out=None):
    assert part in PARTS, part
    return Path(PP.OUT if out is None else out) / 'stage2' / f'cells_{part}.json'


def check_cells(part, enc, cells, smoke=None):
    """The cells of one part and encoder: exactly the expected (sc, aspect, task, data) set, once each; r14's labels; `data`
    values of the index; one draw count per task; every cell through ticket 11's `pp_outcome.check_cell`."""
    smoke = bool(PP.SMOKE) if smoke is None else smoke
    assert part in PARTS and enc in NEW, (part, enc)
    got = sorted((c['sc'], c['aspect'], c['task'], c['data']) for c in cells)
    assert got == EXPECT[part], f'{part}: cells {got} differ from the expected {EXPECT[part]}'
    for c in cells:
        assert c['encoder'] == enc, (c['encoder'], enc)
        assert c['data'] in PP.DATA
        pp_outcome.check_cell(c, smoke=smoke)
    for task in {c['task'] for c in cells}:
        counts = {c['n_draws'] for c in cells if c['task'] == task}
        assert len(counts) == 1, f'{part}: {task} cells have different draw counts {counts}'
    return True


def write_cells(part, enc, cells, out=None):
    """Merge into `cells_<part>.json`: the encoder's cells are replaced, the other encoders' kept (atomic; r14's writer).
    Locked, so two encoders of a part may run at once."""
    path = cells_path(part, out)
    with PP.file_lock(path):
        return bs_artclu.write_cells(path, cells, [enc])


def score(part, enc, n_draws=None, workers=8, root=None, out=None):
    """Stage 2 of one part and encoder. The committed stage 1 record is read first: nothing else is touched before it."""
    assert part in PARTS and enc in NEW, (part, enc)
    rec = committed_record(enc)
    require_protocol_draws(part, n_draws)
    out = Path(PP.OUT if out is None else out)
    if part == 'artelingo_clu':
        cells = run_artelingo_clu(enc, rec, n_draws, workers, root)
    elif part == 'artelingo_ret':
        cells = run_artelingo_ret(enc, rec, out / 'stage2' / 'artret', n_draws, workers, root)
    else:
        cells = run_affection(enc, rec, root=root, cache=out / 'stage2' / 'partitions_affection', n_clu=n_draws,
                              n_ret=n_draws, workers=workers)['cells']
    check_cells(part, enc, cells)
    write_cells(part, enc, cells, out)
    return cells


# ============================================================================================ CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sub = p.add_subparsers(dest='cmd', required=True)
    s1 = sub.add_parser('stage1', help='validation only -> OUT/stage1/ (r14 file names)')
    s1.add_argument('--enc', required=True, choices=NEW)
    s1.add_argument('--part', required=True, choices=('artelingo', 'affection'))
    s2 = sub.add_parser('score', help='held -> OUT/stage2/cells_<part>.json (refuses unless the stage 1 record is committed)')
    s2.add_argument('--enc', required=True, choices=NEW)
    s2.add_argument('--part', required=True, choices=PARTS)
    s2.add_argument('--n-draws', type=int, default=None, help='smoke only')
    args = p.parse_args(argv)
    t0 = time.monotonic()
    assert Path(PP.OUT).resolve() != PP.MAIN.resolve(), 'bad output root'
    if args.cmd == 'stage1':
        blocks = stage1_artelingo(args.enc, workers=args.workers) if args.part == 'artelingo' \
            else dict(affection=stage1_affection(args.enc, workers=args.workers))
        paths = write_stage1(args.enc, blocks)
        print(f'{args.enc} stage 1 {args.part}: {", ".join(str(v) for v in paths.values())} ({time.monotonic() - t0:.0f}s)')
        return 0
    cells = score(args.part, args.enc, args.n_draws, args.workers)
    print(f'{args.enc} {args.part}: {len(cells)} cells -> {cells_path(args.part)} ({time.monotonic() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
