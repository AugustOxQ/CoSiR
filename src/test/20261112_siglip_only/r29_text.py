"""r29 text scorer for a caption tower that reads a caption source (ticket 04; spec §3.4, §3.5 gates (ii) and (iv), D11, D20).

r28's text protocol (`r28_text.py`, which is r23's) with the caption input taken from a caption source of
`r29_paths.CAPTION_SOURCES`: the query text of a text direction is one caption's stored row of that source (`r29_captions.
caption_rows`, joined by sample ID with the painting checked, then L2-normalised as r23's plain text rows are), the caption
tower is rebuilt at its recorded width (`r29_model.load_view`), the image tower reads the SigLIP 1 image store, validation
means are refit on those rows (`tcl_score.score_unit`, r23's own function), the three read-outs are scored per aspect and
direction, and the read-out is picked by r23's rule (highest five-seed mean; ties to the first of r0, center, center_k10).
r28's pure parts (`pick_readout`, `validation_picks`, `held_cells`, `seed_mean_ap`) are imported, not copied.

Arms (the name is the output folder under TEXT_OUT and the stem of the records):
    qwen              K0: r14's c* with the Qwen3 caption input, scored through r28's own loaders (gate (ii))
    siglip1           K1: caption tower on SigLIP 1 text rows      (weights OUT/cstar/siglip1/weights/siglip1/)
    siglip1_prefix    K2: the prefixed store                       (weights OUT/cstar/siglip1/weights/siglip1_prefix/)
    gate_i_a_new      gate (i)(a)'s run of the new trainer on the Qwen3 input, seed 0 only (weights OUT/gate_i_a/new/weights/qwen/):
                      K0's protocol on it gives diagnostic (e)'s nine validation text cells; no picks, no held scoring
    p1_siglip1, p1_siglip1_prefix   P1: plain SigLIP 1 (no training), text rows of that store, image rows of the image store

    python r29_text.py score --arm A --set art_val|art_held [--aspects ..] [--seeds ..] [--device cpu]
    python r29_text.py picks --arm A          validation picks and ranges (needs all five seeds) -> records/text_<A>.json
    python r29_text.py cells --arm A          held cells at the picks                             -> records/text_<A>_held.json
    python r29_text.py replay --arm qwen      towers against r14's exports at r28's float16 rule
    python r29_text.py gate-ii                K0 against r28's records                            -> records/gates/ii.json
    python r29_text.py hash-check [--arms ..] store hash: scorer against the trainer's records     -> records/gates/iv_hash.json

`score --set art_held` refuses unless the arm's pick record (`records/text_<A>.json` with its picks) exists, is committed
unchanged in git HEAD and names the store and weights the scorer reads now (spec §3.4 "Order and C13"). Smoke: set
R29_TEXT_OUT to a scratch folder and R29_TEXT_SMOKE=1 (records then go to <it>/records, no git check; R29_TEXT_OUT without
the flag is refused) and TCL_SMOKE=N, TCL_OUTPUTS=<scratch> as r23.

Provenance: each scored unit writes `<score root>/provenance/<set>/<head>__s<seed>.json` = {weights_sha256 (the .pt file;
None for P1), store_sha256} when it is scored. A unit already on disk is reused only if both equal the current ones; the
validation record lists them per unit and the order rule checks them again.

Record `records/text_<arm>.json` (read by tickets 05 and 07):
    arm, kind ('candidate' | 'p1'), source, store_sha256, row, head ('aspect' | 'all'), score_root, seeds (ints; ['x'] for P1),
    val.cells.<aspect>.<direction> = {per_seed: {<readout>: [one value per seed in `seeds`]}, means: {<readout>: mean}},
    val.group_r0 = {per_cell: {'<aspect>/<direction>': mean over seeds}, per_seed: [mean of the nine cells, per seed], mean},
    picks.<aspect>.<direction> = {pick, mean (at the pick), range (max - min of the per-seed values at the pick; null for P1)}
    (only after `picks`), score_files = the per-query AP file pattern of the arm's held and validation scores.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_paths as P29  # noqa: E402
import r29_captions as C  # noqa: E402
import r29_model as M  # noqa: E402
import r28_text as T  # noqa: E402  (r23's protocol on r14's SigLIP 1 c*; puts the r23 folder on sys.path)
import tcl_paths as P23  # noqa: E402

KIND = P23.CAPTION
ASPECTS, SEEDS, DIRECTIONS, READOUTS = P23.ASPECTS, P23.SEEDS, P23.DIRECTIONS, P23.READOUTS
assert tuple(ASPECTS) == tuple(P29.ASPECTS) and tuple(SEEDS) == tuple(P29.SEEDS)
DEFAULT_TEXT_OUT = P29.DEFAULT_OUT / 'text'
SMOKE = os.environ.get('R29_TEXT_SMOKE') == '1'                       # explicit smoke mode: the only way past the git check
assert SMOKE or 'R29_TEXT_OUT' not in os.environ, 'R29_TEXT_OUT redirects the outputs only with R29_TEXT_SMOKE=1'
TEXT_OUT = Path(os.environ.get('R29_TEXT_OUT', DEFAULT_TEXT_OUT))     # scores and means, one folder per arm
RECORDS_DIR = P29.RECORDS if not SMOKE else TEXT_OUT / 'records'
IMAGE_ROW = 'siglip_so400m'              # tcl_data's name of the SigLIP 1 side row (images and plain text rows)
GRAPH = T.GRAPH                          # r14's validation export names
CANDIDATE_ARMS = ('qwen', 'siglip1', 'siglip1_prefix')
GATE_I_A_NEW = 'gate_i_a_new'          # gate (i)(a)'s `new` run (r29_trainer.gate_i_a_root('new')), seed 0, source qwen
P1_FORMS = ('siglip1', 'siglip1_prefix')
F16_SHARE = 0.995                        # r28's replay rule: at least this share equal in float16, the rest one float16 step
TOL = 1e-6                               # gate (ii): per-seed mAP against r28's text.json


def is_smoke():
    return SMOKE


def now():
    return datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%d %H:%M')


# ------------------------------------------------------------------------------------------------ arms
def arm_info(arm):
    """dict(kind, source, row, seeds, root, folder): where an arm's towers, rows and scores are."""
    if arm == 'qwen':
        root, folder = P29.K0_ROOT, P29.K0_ARM
        return dict(kind='candidate', source='qwen', row='cstar', seeds=SEEDS, root=root, folder=folder)
    if arm == GATE_I_A_NEW:
        return dict(kind='candidate', source='qwen', row='cstar', seeds=(0,), root=P29.OUT / 'gate_i_a/new', folder='qwen')
    if arm in CANDIDATE_ARMS:
        return dict(kind='candidate', source=arm, row='cstar', seeds=SEEDS, root=P29.OUT / 'cstar/siglip1', folder=arm)
    assert arm in ['p1_' + f for f in P1_FORMS], f'unknown arm {arm!r}'
    return dict(kind='p1', source=arm[3:], row=IMAGE_ROW, seeds=('x',), root=None, folder=None)


def units(arm, aspects=ASPECTS, seeds=None):
    """[(head, seed)] of an arm: one per aspect and seed for a candidate, the one ('all', 'x') for P1."""
    info = arm_info(arm)
    if info['kind'] == 'p1':
        return [('all', 'x')]
    return [(a, s) for a in aspects for s in (info['seeds'] if seeds is None else seeds)]


def score_root(arm):
    return Path(TEXT_OUT) / arm


def record_path(arm, held=False):
    return Path(RECORDS_DIR) / f'text_{arm}{"_held" if held else ""}.json'


def write_json(path, doc):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')


# ------------------------------------------------------------------------------------------------ rows
def text_rows(source, fr):
    """Unit float32 rows of one SigLIP-store caption each, in the frame's order (`tcl_data.frame`): joined by sample ID,
    the painting checked, then L2-normalised as r23's `tcl_data._art_plain_style_rows` does."""
    import tx_data
    names = np.asarray(fr['names'])[np.asarray(fr['tp'])]
    return tx_data.unit(C.caption_rows(source, fr['sample_id'], names))


def _wrap(model, width, device):
    """float32 [n, width] -> unit float32 [n, 2048] through a loaded tower; inputs are unit-normalised first (as r28)."""
    import torch
    import tx_data
    model.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(x):
        assert x.ndim == 2 and x.shape[1] == width and len(x), (x.shape, width)
        x = tx_data.unit(x)
        parts = [model(torch.as_tensor(x[i:i + 4096], dtype=torch.float32, device=device)).cpu().numpy()
                 for i in range(0, len(x), 4096)]
        out = np.concatenate(parts).astype(np.float32)
        assert out.shape == (len(x), P29.OUTPUT_WIDTH) and np.isfinite(out).all()
        return out
    return forward


def _load_view(path, device, expect_store_sha256):
    """(view, rb block) of one saved view: the single call site of `r29_model.load_view`, which refuses a sidecar naming
    another store than `expect_store_sha256`."""
    return M.load_view(path, device, expect_store_sha256=expect_store_sha256), M.read_sidecar(path)[1]


def weights_path(arm, aspect, seed):
    info = arm_info(arm)
    return Path(info['root']) / 'weights' / info['folder'] / f'{aspect}__s{seed}.pt'


def load_towers(arm, aspect, seed, device='cpu'):
    """(image_side, caption_side) of a candidate arm. K0 (`qwen`) goes through r28's own loader (r14's sidecars lack the
    caption fields); the others through `r29_model.load_view` at the widths the sidecar records, which must name this arm's
    caption source and the store as it is now."""
    assert arm_info(arm)['kind'] == 'candidate'
    if arm == 'qwen':
        return T.load_sides(aspect, seed, device, root=P29.K0_ROOT)
    path = weights_path(arm, aspect, seed)
    source = arm_info(arm)['source']
    view, rb = _load_view(path, device, C.store_hash(source))   # first: a sidecar that disagrees with its weights or store is refused
    config = M.read_sidecar(path)[0]
    assert config['aspect'] == aspect and config['seed'] == seed, (path, config['aspect'], config['seed'])
    assert rb['cap_source'] == source and rb['cap_store_sha256'] == C.store_hash(source), \
        f'{path}: sidecar names {rb["cap_source"]} / {rb["cap_store_sha256"][:12]}, not the store of {arm} as it is now'
    assert rb['width'] == 1152, (path, rb['width'])
    return _wrap(view['img'], rb['width'], device), _wrap(view['cap'], rb['cap_width'], device)


def rows(arm, set_name, head, seed, device='cpu'):
    """(text, image) float32 unit rows in `tcl_data.frame` order. A candidate: the caption tower on each caption's stored row
    of the arm's source, the image tower on the SigLIP 1 image. P1: the store's text rows and the SigLIP 1 images."""
    import tcl_data
    assert set_name in ('art_val', 'art_held')
    info = arm_info(arm)
    if arm == 'qwen':
        return T.rows(set_name, head, seed, device)                 # r28's route: Qwen3 plain rows, joined inside tcl_data
    # (gate_i_a_new: the same Qwen3 plain rows, the towers rebuilt from the new trainer's sidecar)
    fr = tcl_data.frame(set_name, KIND)
    image = tcl_data.image(set_name, IMAGE_ROW)
    text = tcl_data.plain_text(set_name, KIND) if info['source'] == 'qwen' else text_rows(info['source'], fr)   # as r28 for Qwen3
    if info['kind'] == 'p1':
        out = text, image
    else:
        image_side, caption_side = load_towers(arm, head, seed, device)
        out = caption_side(text), image_side(image)
    assert out[0].shape[0] == fr['n_t'] and out[1].shape[0] == fr['n_i'], (out[0].shape, out[1].shape)
    return out


# ------------------------------------------------------------------------------------------------ provenance
def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def provenance(arm, head, seed):
    """What a unit's scores depend on as of now: the weights file's sha256 (None for P1) and the caption store's hash."""
    info = arm_info(arm)
    weights = file_sha(weights_path(arm, head, seed)) if info['kind'] == 'candidate' else None
    return dict(weights_sha256=weights, store_sha256=C.store_hash(info['source']))


def provenance_path(arm, set_name, head, seed):
    return score_root(arm) / 'provenance' / set_name / f'{head}__s{seed}.json'


def write_provenance(arm, set_name, head, seed, now_prov=None):
    write_json(provenance_path(arm, set_name, head, seed), now_prov or provenance(arm, head, seed))


def check_provenance(arm, set_name, head, seed):
    """The unit's scores were made from the weights and store as they are now (a retrained or changed one is refused);
    returns the provenance."""
    path, now_prov = provenance_path(arm, set_name, head, seed), provenance(arm, head, seed)
    assert path.exists(), f'{arm}/{set_name}/{head}/s{seed}: scores exist but no provenance file {path}: rescore the unit'
    then = json.loads(path.read_text())
    for key, what in (('weights_sha256', 'weights'), ('store_sha256', 'store')):
        assert then[key] == now_prov[key], \
            f'{arm}/{set_name}/{head}/s{seed}: scored with other {what} ({then[key]}) than now ({now_prov[key]}): rescore the unit'
    return now_prov


# ------------------------------------------------------------------------------------------------ score
def require_pick_record(arm):
    """The order rule: held rows only after the arm's validation picks are on record, committed unchanged in git HEAD (not
    under smoke) and made for the store the scorer reads now. Returns the record."""
    path = record_path(arm)
    if not path.exists():
        raise RuntimeError(f'no pick record {path}: run `score --set art_val` and `picks` for {arm} first (spec §3.4)')
    text = path.read_text()
    doc = json.loads(text)
    if 'picks' not in doc:
        raise RuntimeError(f'{path} has no picks: run `picks --arm {arm}` first (spec §3.4)')
    info = arm_info(arm)
    assert doc['seeds'] == list(info['seeds']), f'{path}: seeds {doc["seeds"]}, the pick needs {list(info["seeds"])}'
    assert doc['store_sha256'] == C.store_hash(info['source']), f'{path}: the store changed since the picks were made'
    for head, seed in units(arm, seeds=doc['seeds'] if info['kind'] == 'candidate' else None):
        assert doc['provenance'][f'{head}__s{seed}'] == provenance(arm, head, seed), \
            f'{path}: the weights or store of {head}/s{seed} changed since the picks were made'
    if not is_smoke():
        rel = path.resolve().relative_to(P29.REPO.resolve())
        head = subprocess.run(['git', '-C', str(P29.REPO), 'show', f'HEAD:{rel}'], capture_output=True, text=True)
        if head.returncode != 0 or head.stdout != text:
            raise RuntimeError(f'{rel} is not committed unchanged in HEAD: commit the picks before held scoring (spec §3.4)')
    return doc


def score(arm, set_name, aspects=ASPECTS, seeds=None, device='cpu', block=1024):
    """Rank an arm on a set, per aspect and seed (P1: once): every direction and read-out, one score file each under
    TEXT_OUT/<arm> in r23's layout. A validation set writes its means first (refit on the new rows); a held set reads them."""
    import tcl_data
    import tcl_score
    assert set_name in ('art_val', 'art_held')
    if set_name == 'art_held':
        require_pick_record(arm)
    info, root = arm_info(arm), score_root(arm)
    fr = tcl_data.frame(set_name, KIND)
    flags = {a: tcl_data.names_label(set_name, KIND, a) for a in ASPECTS}
    val_n = None
    if P23.VAL_OF[set_name] != set_name:
        val = tcl_data.frame(P23.VAL_OF[set_name], KIND)
        val_n = (val['n_t'], val['n_i'])
    import time
    for head, seed in units(arm, aspects, seeds):
        if tcl_score.unit_done(set_name, KIND, info['row'], head, seed, fr, out=root):
            check_provenance(arm, set_name, head, seed)          # reuse scores only of these weights and this store
            print('done', set_name, head, seed, flush=True)
            continue
        started = time.monotonic()
        prov = provenance(arm, head, seed)                       # before the towers are read, written after the scoring
        text, image = rows(arm, set_name, head, seed, device)
        tcl_score.score_unit(set_name, KIND, info['row'], head, seed, text, image, fr, flags, device, out=root,
                             block=block, val_n=val_n)
        del text, image
        tcl_score.release(device)
        write_provenance(arm, set_name, head, seed, prov)
        print(f'{arm}/{set_name}/{head}/s{seed}: {time.monotonic() - started:.1f}s', flush=True)
    if set_name == 'art_val':
        write_val_record(arm)


# ------------------------------------------------------------------------------------------------ records
def seeds_done(arm):
    """Seeds whose validation score files are all there (every aspect, direction and read-out), in SEEDS order."""
    info, root = arm_info(arm), score_root(arm)
    done = []
    for seed in info['seeds']:
        heads = ASPECTS if info['kind'] == 'candidate' else ('all',)
        s = seed
        if all(P23.score_path('art_val', KIND, info['row'], h, s, d, r, out=root).exists()
               for h in heads for d in DIRECTIONS for r in READOUTS):
            done.append(seed)
    return done


def val_cells(arm, seeds):
    """{aspect: {direction: {per_seed: {readout: [..]}, means: {readout: mean}}}} read from the arm's validation scores."""
    info = arm_info(arm)
    got = T.validation_picks(score_root(arm), info['row'], seeds)
    return {a: {d: dict(per_seed=got[a][d]['per_seed'], means=got[a][d]['means']) for d in DIRECTIONS} for a in ASPECTS}


def group_r0(cells, n_seeds):
    """The text group at r0: the nine cells' mean, per seed and over seeds (spec §3.4)."""
    per_cell = {f'{a}/{d}': float(np.mean(cells[a][d]['per_seed']['r0'])) for a in ASPECTS for d in DIRECTIONS}
    per_seed = [float(np.mean([cells[a][d]['per_seed']['r0'][i] for a in ASPECTS for d in DIRECTIONS]))
                for i in range(n_seeds)]
    return dict(per_cell=per_cell, per_seed=per_seed, mean=float(np.mean(per_seed)))


def write_val_record(arm, with_picks=False):
    """records/text_<arm>.json from the validation score files on disk, for the seeds that are complete. The picks (r23's
    rule: `pick_readout`) and the candidate's range at each pick are added only on `with_picks`, which needs all five seeds
    (P1: its one)."""
    info = arm_info(arm)
    done = seeds_done(arm)
    if not done:
        print(f'{arm}: no seed has complete validation scores; no record written')
        return None
    cells = val_cells(arm, done)
    prov = {f'{head}__s{seed}': check_provenance(arm, 'art_val', head, seed) for head, seed in units(arm, seeds=done)}
    doc = dict(arm=arm, kind=info['kind'], source=info['source'], store_sha256=C.store_hash(info['source']), row=info['row'],
               head='aspect' if info['kind'] == 'candidate' else 'all', score_root=str(score_root(arm)),
               seeds=list(done), provenance=prov, val=dict(cells=cells, group_r0=group_r0(cells, len(done))),
               score_files=str(score_root(arm) / 'scores/<set>/caption' / info['row'] /
                               '<head>__s<seed>__<direction>__<readout>.npz'),
               written=now())
    if with_picks:
        assert done == list(info['seeds']), f'{arm}: picks need seeds {list(info["seeds"])}, validation scores have {done}'
        picks = {}
        for a in ASPECTS:
            picks[a] = {}
            for d in DIRECTIONS:
                per_seed = cells[a][d]['per_seed']
                pick, means = T.pick_readout(per_seed)
                vals = per_seed[pick]
                picks[a][d] = dict(pick=pick, mean=means[pick],
                                   range=None if info['kind'] == 'p1' else float(max(vals) - min(vals)))
        if arm == 'p1_siglip1':     # the spec uses r23's row at r23's pick: the pick must be r23's committed stage 1 pick
            r23 = json.loads(Path(T.R23_STAGE1).read_text())['picks'][KIND][IMAGE_ROW]
            for a in ASPECTS:
                for d in DIRECTIONS:
                    assert picks[a][d]['pick'] == r23[a][d], f'p1_siglip1 {a}/{d}: pick {picks[a][d]["pick"]}, r23 {r23[a][d]}'
        doc['picks'] = picks
    write_json(record_path(arm), doc)
    print('wrote', record_path(arm), 'seeds', done, 'picks' if with_picks else '')
    return doc


def picks(arm):
    return write_val_record(arm, with_picks=True)


def held_record(arm):
    """records/text_<arm>_held.json: held cells at the recorded picks, from the held score files."""
    info = arm_info(arm)
    doc = require_pick_record(arm)
    for head, seed in units(arm):
        check_provenance(arm, 'art_held', head, seed)
    pk = {a: {d: dict(pick=doc['picks'][a][d]['pick']) for d in DIRECTIONS} for a in ASPECTS}
    cells = T.held_cells(score_root(arm), info['row'], pk, info['seeds'])
    out = dict(arm=arm, kind=info['kind'], source=info['source'], store_sha256=doc['store_sha256'], row=info['row'],
               seeds=doc['seeds'], pick_record=str(record_path(arm)), picks={a: {d: pk[a][d]['pick'] for d in DIRECTIONS} for a in ASPECTS},
               cells=cells, score_files=doc['score_files'].replace('<set>', 'art_held'), written=now())
    write_json(record_path(arm, held=True), out)
    print('wrote', record_path(arm, held=True))
    return out


# ------------------------------------------------------------------------------------------------ replay (tower check)
def f16_check(mine, stored):
    """r28's float16 rule: float16(mine) against the stored float16 export. (share equal, max |diff| in float32, whether every
    unequal element is the stored value's neighbour in float16, i.e. one float16 step)."""
    stored = np.asarray(stored)
    assert mine.shape == stored.shape and stored.dtype == np.float16
    mine16 = mine.astype(np.float16)
    unequal = mine16 != stored
    up = np.nextafter(stored, np.float16(np.inf))
    down = np.nextafter(stored, np.float16(-np.inf))
    one_step = bool(((mine16 == up) | (mine16 == down) | ~unequal).all())
    return float((~unequal).mean()), float(np.abs(mine.astype(np.float64) - stored.astype(np.float64)).max()), one_step


def export_paths(arm, aspect, seed):
    info = arm_info(arm)
    root, folder = Path(info['root']), info['folder']
    return (root / f'train/{folder}/R6-5/{aspect}__{GRAPH[aspect]}__s{seed}.npz',
            root / f'held/{folder}/{aspect}__s{seed}.npz')


def replay(arm, device='cpu', write=True):
    """The inference path against the trainer's exports: the caption tower on the validation painting means of the arm's
    source (stored `cap`), the image tower on validation and held SigLIP 1 images (stored `img`), at r28's float16 rule.
    Export row orders are checked against the painting names first, as r28's replay."""
    import rt_data
    import tcl_data
    import rb_student
    import ct2_data
    info = arm_info(arm)
    assert info['kind'] == 'candidate'
    n_val, n_held = P23.N_PAINTINGS['art_val'], P23.N_PAINTINGS['art_held']
    s = ct2_data.split2('full')
    dev_names = [str(n) for n in s['all_painting_ids'][s['dev']]]
    assert dev_names == list(rt_data.names('val')), 'validation export order differs from the frame order'
    held_names, _ = rb_student.held_names()
    assert list(held_names) == list(rt_data.names('held')), 'held export order differs from the frame order'
    means = C.painting_means(info['source'], dev_names)
    img_val, img_held = tcl_data.image('art_val', IMAGE_ROW), tcl_data.image('art_held', IMAGE_ROW)
    assert len(means) == len(img_val) == n_val and len(img_held) == n_held
    files, worst = {}, dict(min_share_equal=1.0, all_one_step=True)
    for aspect in ASPECTS:
        for seed in info['seeds']:
            image_side, caption_side = load_towers(arm, aspect, seed, device)
            val_path, held_path = export_paths(arm, aspect, seed)
            with np.load(val_path, allow_pickle=False) as z:
                assert sorted(z.files) == ['cap', 'img'], val_path
                cap_stored, img_stored = z['cap'], z['img']
            with np.load(held_path, allow_pickle=False) as z:
                held_stored = z['img']
            item = {}
            for name, mine, stored in (('val_caption', caption_side(means), cap_stored),
                                       ('val_image', image_side(img_val), img_stored),
                                       ('held_image', image_side(img_held), held_stored)):
                share, d32, one_step = f16_check(mine, stored)
                item[name] = dict(share_equal_float16=share, max_abs_float32=d32, one_float16_step=one_step)
                worst['min_share_equal'] = min(worst['min_share_equal'], share)
                worst['all_one_step'] &= one_step
            files[f'{aspect}__s{seed}'] = item
            print('replay', arm, aspect, seed, {k: round(v['share_equal_float16'], 4) for k, v in item.items()}, flush=True)
    doc = dict(arm=arm, device=device, n_files=len(files), **worst,
               rule=f'share equal in float16 >= {F16_SHARE}, every other element one float16 step from the stored value',
               passed=replay_passed(worst['min_share_equal'], worst['all_one_step']), files=files)
    if write:
        write_json(Path(TEXT_OUT) / arm / 'replay.json', doc)
    return doc


# ------------------------------------------------------------------------------------------------ gates
def write_gate(name, status, details):
    """records/gates/<name>.json = {gate, status, tries, details, written} (the index's shape). This ticket's gate files are
    `ii` and `iv_hash`; ticket 02 writes `iv_store`, ticket 05 merges the parts of (iv)."""
    path = Path(RECORDS_DIR) / 'gates' / f'{name}.json'
    tries = json.loads(path.read_text())['tries'] if path.exists() else 0
    rec = dict(gate=name, status=status, tries=tries + 1, details=details, written=now())
    write_json(path, rec)
    return rec


def replay_passed(min_share_equal, all_one_step):
    """r28's float16 rule for the tower embeddings: at least F16_SHARE of the elements equal, every other one a float16 step."""
    return bool(min_share_equal >= F16_SHARE and all_one_step)


def compare_k0(mine, r28):
    """Gate (ii)'s comparison of K0's per-seed validation mAP and picks (`validation_picks` form: {aspect: {direction:
    {per_seed: {readout: [..]}, pick}}}) with r28's. Every value of both sides must be finite and the seed lists of equal
    length (a NaN or a missing seed raises); a cell whose largest per-seed gap is not <= TOL is mismatched."""
    worst, mismatched, picks_equal, n = 0.0, [], {}, 0
    for a in ASPECTS:
        for d in DIRECTIONS:
            picks_equal[f'{a}/{d}'] = dict(mine=mine[a][d]['pick'], r28=r28[a][d]['pick'])
            for r in READOUTS:
                x, y = (np.asarray(side[a][d]['per_seed'][r], dtype=np.float64) for side in (mine, r28))
                assert x.shape == y.shape == (len(SEEDS),), f'{a}/{d}/{r}: {len(x)} and {len(y)} seeds, expected {len(SEEDS)}'
                assert np.isfinite(x).all() and np.isfinite(y).all(), f'{a}/{d}/{r}: a per-seed value is not finite'
                diff = float(np.max(np.abs(x - y)))
                worst, n = max(worst, diff), n + len(x)
                if not diff <= TOL:
                    mismatched.append(f'{a}/{d}/{r}')
    picks_ok = all(v['mine'] == v['r28'] for v in picks_equal.values())
    return dict(n_values=n, tolerance=TOL, max_abs_diff_per_seed_map=worst, mismatched=mismatched, picks=picks_equal,
                picks_equal=picks_ok, ok=not mismatched and picks_ok)


def gate_ii(device='cpu', replay_too=True):
    """Gate (ii) on K0: every per-seed validation mAP at r0, center, center_k10 within 1e-6 of r28's records/text.json, the
    same nine picks, and the tower embeddings against r14's exports at r28's float16 rule. Needs `score --arm qwen --set
    art_val` done (asserts it)."""
    arm = 'qwen'
    done = seeds_done(arm)
    assert done == list(SEEDS), f'score --arm qwen --set art_val first: complete seeds {done}'
    for head, seed in units(arm):
        check_provenance(arm, 'art_val', head, seed)
    mine = T.validation_picks(score_root(arm), 'cstar', SEEDS)
    r28 = json.loads((P29.R28_RECORDS / 'text.json').read_text())['ours_siglip1']['picks']
    details = compare_k0(mine, r28)
    details['score_device_note'] = 'r28 scored on cuda; this scoring ran where `score` ran'
    ok = details.pop('ok')
    if replay_too:
        rep = replay(arm, device)
        details['replay'] = {k: rep[k] for k in ('device', 'n_files', 'min_share_equal', 'all_one_step', 'rule', 'passed')}
        ok = ok and rep['passed']
    rec = write_gate('ii', 'pass' if ok else 'fail', details)
    print('gate (ii):', rec['status'], 'max abs diff', details['max_abs_diff_per_seed_map'], 'mismatched',
          len(details['mismatched']), 'picks equal', details['picks_equal'])
    return rec


def _find_hash(doc, where):
    """`cap_store_sha256` of a trainer's sidecar or run record: at its top, in its `rb` block, or in `config.rb`."""
    found = [d['cap_store_sha256'] for d in (doc, doc.get('rb', {}), doc.get('config', {}).get('rb', {}))
             if isinstance(d, dict) and 'cap_store_sha256' in d]
    assert len(found) == 1, f'{where}: cap_store_sha256 found {len(found)} times'
    return found[0]


def hash_check(arms=('siglip1', 'siglip1_prefix')):
    """Gate (iv), hash part -> records/gates/iv_hash.json (its own file: ticket 02 writes iv_store.json, ticket 05 merges them).
    For each arm the store's index hash as the scorer records it (records/text_<arm>.json), as the
    store has it now, and as the trainer records it in every run record and weights sidecar of the arm: all equal."""
    result, ok = {}, True
    for arm in arms:
        info = arm_info(arm)
        assert info['kind'] == 'candidate' and info['source'] != 'qwen', arm
        now_hash = C.store_hash(arm)
        rec = record_path(arm)
        scorer = json.loads(rec.read_text())['store_sha256'] if rec.exists() else None
        runs = sorted((Path(info['root']) / 'train' / info['folder']).glob('run__s*.json'))
        sidecars = sorted((Path(info['root']) / 'weights' / info['folder']).glob('*__s*.json'))
        trainer = {str(p.name): _find_hash(json.loads(p.read_text()), p) for p in runs}
        trainer.update({str(p.name): _find_hash(json.loads(p.read_text()), p) for p in sidecars})
        equal = scorer is not None and bool(runs) and bool(sidecars) and set(trainer.values()) == {scorer} and scorer == now_hash
        ok &= equal
        result[arm] = dict(scorer_record=scorer, store_now=now_hash, n_run_records=len(runs), n_sidecars=len(sidecars),
                           trainer_hashes=sorted(set(trainer.values())), equal=equal)
        print('hash-check', arm, 'equal' if equal else 'NOT EQUAL')
    return write_gate('iv_hash', 'pass' if ok else 'fail', dict(arms=result))


# ------------------------------------------------------------------------------------------------ CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('score')
    a.add_argument('--arm', required=True)
    a.add_argument('--set', required=True, choices=('art_val', 'art_held'), dest='set_name')
    a.add_argument('--aspects', nargs='+', default=list(ASPECTS), choices=ASPECTS)
    a.add_argument('--seeds', nargs='+', type=int, default=None)
    a.add_argument('--device', default='cpu')
    a.add_argument('--block', type=int, default=1024)
    for name in ('picks', 'cells'):
        sub.add_parser(name).add_argument('--arm', required=True)
    b = sub.add_parser('replay')
    b.add_argument('--arm', required=True)
    b.add_argument('--device', default='cpu')
    c = sub.add_parser('gate-ii')
    c.add_argument('--device', default='cpu')
    c.add_argument('--no-replay', action='store_true')
    d = sub.add_parser('hash-check')
    d.add_argument('--arms', nargs='+', default=['siglip1', 'siglip1_prefix'])
    args = p.parse_args()
    if args.cmd == 'score':
        score(args.arm, args.set_name, args.aspects, args.seeds, args.device, args.block)
    elif args.cmd == 'picks':
        picks(args.arm)
    elif args.cmd == 'cells':
        held_record(args.arm)
    elif args.cmd == 'replay':
        doc = replay(args.arm, args.device)
        print({k: doc[k] for k in ('min_share_equal', 'all_one_step', 'passed')})
    elif args.cmd == 'gate-ii':
        gate_ii(args.device, not args.no_replay)
    else:
        hash_check(tuple(args.arms))


if __name__ == '__main__':
    main()
