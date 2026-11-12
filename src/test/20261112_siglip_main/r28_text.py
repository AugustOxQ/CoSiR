"""r28 job D: Table 3 text directions and per-seed values on SigLIP 1 (handoff section 4, items 2 and 5).

Part 1. Our SigLIP 1 model (c*, arm `k11=40`, five seeds; weights
`/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip1/weights/k11=40/<aspect>__s<seed>.pt`) in the three
text directions on ArtELingo, with exactly r23's protocol (`src/test/20261106_text_claim/`, the code that made the Table 3
text rows of the paper): rows from `tcl_data` (frames by sample id, C6), the three read-outs, means fitted on the validation
set's rows, the read-out picked per aspect and direction on validation (mean over the five seeds of the validation label
mAP, ties to the first of r0, center, center_k10), the held cell = the mean over seeds of the mean AP over scored queries.
`tcl_score.score_unit` ranks (r23's own function); only the towers are new: the image tower reads SigLIP 1 (1152), the
caption tower reads the Qwen3 plain caption embedding (2048), as r14 trained them.

    python r28_text.py replay [--device cuda]          caption tower on validation caption means and image tower on
                                                        validation and held images, against r14's exports
    python r28_text.py score --set art_val|art_held [--aspects ..] [--seeds ..] [--device cuda]
    python r28_text.py cells                           picks (validation) and held cells -> records/text.json
    python r28_text.py per-seed                        Part 2 -> records/per_seed.json

Under TCL_SMOKE=N and TCL_OUTPUTS=<scratch> (environment, as r23) the first N paintings only; smoke writes to the scratch.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
R23 = TEST / '20261106_text_claim'
if str(R23) not in sys.path:
    sys.path.insert(0, str(R23))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import tcl_paths as P  # noqa: E402  (installs the older rounds' folders)
import tx_data  # noqa: E402,F401  (r22's data module: puts the r11 / r10 folders on the path)

OUT = Path(os.environ.get('R28_TEXT_OUT', '/data/SSD2/cosir_r28/20261112_siglip_main/text'))
RECORDS = HERE / 'records'
R14_SIGLIP1 = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip1')
R23_OUT = Path('/data/SSD2/cosir_r23/20261106_text_claim')
R23_STAGE1 = R23 / 'records/stage1_art.json'
ASPECTS, SEEDS, DIRECTIONS, READOUTS = P.ASPECTS, P.SEEDS, P.DIRECTIONS, P.READOUTS
KIND = P.CAPTION
GRAPH = dict(emotion='Gres', style='Fres05', genre='R0o')       # r14 validation export names (rt_data.GRAPHS)
PAPER_TABLE = HERE.parents[2] / 'docs/paper/cvpr2027_aspect_spaces/CVPR2027-CoSiR-Paper/Tables/main_retrieval.tex'


# ------------------------------------------------------------------------------------------------ pure parts
def seed_mean_ap(ap):
    """Mean AP of one score file's column over its scored queries (r23: `ap[isfinite].mean(float64)`)."""
    ap = np.asarray(ap)
    return float(ap[np.isfinite(ap)].mean(dtype=np.float64))


def pick_readout(per_seed):
    """Validation pick of one aspect and direction. per_seed: {readout: [per-seed mean AP]}. The readout with the highest
    mean over seeds; a tie goes to the first of READOUTS (r23 `tcl_stage1.run`: `max(READOUTS, key=means.__getitem__)`)."""
    assert set(per_seed) == set(READOUTS), sorted(per_seed)
    means = {r: float(np.mean(per_seed[r])) for r in READOUTS}
    return max(READOUTS, key=means.__getitem__), means


def join_by_id(stored_ids, wanted_ids):
    """Row of each wanted id in a store whose rows carry `stored_ids` (C6: never by position). Fails on a missing or a
    repeated stored id."""
    stored = list(stored_ids)
    assert len(set(stored)) == len(stored), 'stored ids repeat'
    where = {s: i for i, s in enumerate(stored)}
    missing = [w for w in wanted_ids if w not in where]
    assert not missing, f'{len(missing)} wanted ids not in the store, first {missing[:3]}'
    return np.asarray([where[w] for w in wanted_ids], dtype=np.int64)


def f16_diff(mine, stored):
    """(max |mine - stored| in float32, max |float16(mine) - stored|, share of float16(mine) equal to stored)."""
    stored = np.asarray(stored)
    assert mine.shape == stored.shape
    a = np.abs(mine.astype(np.float64) - stored.astype(np.float64)).max()
    mine16 = mine.astype(np.float16)
    b = np.abs(mine16.astype(np.float64) - stored.astype(np.float64)).max()
    return float(a), float(b), float((mine16 == stored).mean())


# ------------------------------------------------------------------------------------------------ towers
def _towers_module():
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Side(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(d_in, 1024), nn.GELU(), nn.Dropout(0.1))
            self.output = nn.Linear(1024, 2048)

        def forward(self, x):
            return F.normalize(self.output(self.trunk(x)), dim=1)
    return torch, Side


def load_sides(aspect, seed, device='cpu', root=R14_SIGLIP1):
    """(image_side, caption_side) of r14's SigLIP 1 checkpoint: functions float32 [n, d_in] -> unit float32 [n, 2048].
    The image tower reads width 1152, the caption tower width 2048. Inputs are unit-normalised first (as r23's tx_model)."""
    import tx_data
    torch, Side = _towers_module()
    path = Path(root) / f'weights/k11=40/{aspect}__s{seed}.pt'
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert set(state) == {f'{s}.{k}' for s in ('img', 'cap') for k in
                          ('trunk.0.weight', 'trunk.0.bias', 'output.weight', 'output.bias')}, path
    widths = dict(img=1152, cap=2048)

    def make(side):
        assert tuple(state[f'{side}.trunk.0.weight'].shape) == (1024, widths[side]), (path, side)
        model = Side(widths[side])
        model.load_state_dict({k[len(side) + 1:]: v for k, v in state.items() if k.startswith(side + '.')})
        model.to(device).eval().requires_grad_(False)

        @torch.no_grad()
        def forward(x):
            assert x.ndim == 2 and x.shape[1] == widths[side] and len(x), (side, x.shape)
            x = tx_data.unit(x)
            parts = [model(torch.as_tensor(x[i:i + 4096], dtype=torch.float32, device=device)).cpu().numpy()
                     for i in range(0, len(x), 4096)]
            out = np.concatenate(parts).astype(np.float32)
            assert out.shape == (len(x), 2048) and np.isfinite(out).all()
            return out
        return forward

    return make('img'), make('cap')


def rows(set_name, aspect, seed, device):
    """(text, image) float32 unit rows of our SigLIP 1 model in `tcl_data.frame` order: the caption tower on each caption's
    Qwen3 plain embedding (joined by sample id inside `tcl_data.plain_text`), the image tower on the SigLIP 1 image."""
    import tcl_data
    assert set_name in ('art_val', 'art_held')
    image_side, caption_side = load_sides(aspect, seed, device)
    text = caption_side(tcl_data.plain_text(set_name, KIND))
    image = image_side(tcl_data.image(set_name, 'siglip_so400m'))
    fr = tcl_data.frame(set_name, KIND)
    assert text.shape == (fr['n_t'], 2048) and image.shape == (fr['n_i'], 2048)
    return text, image


# ------------------------------------------------------------------------------------------------ replay
def _val_export(aspect, seed):
    path = R14_SIGLIP1 / f'train/k11=40/R6-5/{aspect}__{GRAPH[aspect]}__s{seed}.npz'
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == ['cap', 'img'], path
        return z['cap'], z['img']


def _held_export(aspect, seed):
    path = R14_SIGLIP1 / f'held/k11=40/{aspect}__s{seed}.npz'
    with np.load(path, allow_pickle=False) as z:
        return z['img']


def replay(device):
    """The inference path against r14's exports: the caption tower on the validation caption means (stored `cap`), the image
    tower on validation and held SigLIP 1 images (stored `img`). Row orders are checked against the painting names first."""
    import rt_data
    import tx_data
    import tcl_data
    import rb_student
    import rb_encoders  # noqa: F401
    n_val, n_held = P.N_PAINTINGS['art_val'], P.N_PAINTINGS['art_held']
    captions = tx_data.val_captions()
    assert np.array_equal(captions['pid'], tcl_data._art_captions('val')['pid'])
    means = tx_data.caption_mean(tx_data.text_rows('plain', raw=True), captions['pid'], n_val)
    img_val = tcl_data.image('art_val', 'siglip_so400m')
    img_held = tcl_data.image('art_held', 'siglip_so400m')
    # export row orders: the trainer's dev order, and the CT's held order, equal to the frames' painting order
    import ct2_data
    s = ct2_data.split2('full')
    dev_names = [str(n) for n in s['all_painting_ids'][s['dev']]]
    assert dev_names == list(rt_data.names('val')), 'validation export order differs from the frame order'
    held_names, _ = rb_student.held_names()
    assert list(held_names) == list(rt_data.names('held')), 'held export order differs from the frame order'
    assert len(img_val) == n_val and len(img_held) == n_held
    files = {}
    for aspect in ASPECTS:
        for seed in SEEDS:
            image_side, caption_side = load_sides(aspect, seed, device)
            cap_stored, img_stored = _val_export(aspect, seed)
            held_stored = _held_export(aspect, seed)
            item = {}
            for name, mine, stored in (('val_caption', caption_side(means), cap_stored),
                                       ('val_image', image_side(img_val), img_stored),
                                       ('held_image', image_side(img_held), held_stored)):
                d32, d16, equal = f16_diff(mine, stored)
                item[name] = dict(max_abs_float32=d32, max_abs_float16=d16, share_equal_float16=equal)
            files[f'{aspect}__s{seed}'] = item
            print('replay', aspect, seed, {k: round(v['max_abs_float16'], 7) for k, v in item.items()}, flush=True)
    largest = {k: max(f[k]['max_abs_float16'] for f in files.values()) for k in ('val_caption', 'val_image', 'held_image')}
    largest32 = {k: max(f[k]['max_abs_float32'] for f in files.values()) for k in largest}
    doc = dict(device=device, n_files=len(files), largest_abs_float16=largest, largest_abs_float32=largest32, files=files)
    path = OUT / 'replay.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(largest_abs_float16=largest, largest_abs_float32=largest32)), flush=True)
    return doc


# ------------------------------------------------------------------------------------------------ score
def score(set_name, aspects, seeds, device, block=1024):
    """Rank our model on a set: per aspect (head) and seed, every direction and read-out, one score file each (r23's
    layout under OUT). A validation set writes its means first; a held set reads them (so score art_val first)."""
    import torch
    import tcl_data
    import tcl_score
    fr = tcl_data.frame(set_name, KIND)
    flags = {a: tcl_data.names_label(set_name, KIND, a) for a in ASPECTS}
    val_n = None
    if P.VAL_OF[set_name] != set_name:
        val = tcl_data.frame(P.VAL_OF[set_name], KIND)
        val_n = (val['n_t'], val['n_i'])
    for aspect in aspects:
        for seed in seeds:
            if tcl_score.unit_done(set_name, KIND, 'cstar', aspect, seed, fr, out=OUT):
                print('done', set_name, aspect, seed, flush=True)
                continue
            started = time.monotonic()
            text, image = rows(set_name, aspect, seed, device)
            tcl_score.score_unit(set_name, KIND, 'cstar', aspect, seed, text, image, fr, flags, device, out=OUT,
                                 block=block, val_n=val_n)
            del text, image
            tcl_score.release(device)
            print(f'{set_name}/{aspect}/s{seed}: {time.monotonic() - started:.1f}s', flush=True)


# ------------------------------------------------------------------------------------------------ cells
def _col(root, set_name, row, head, seed, direction, readout, aspect):
    path = P.score_path(set_name, KIND, row, head, seed, direction, readout, out=root)
    with np.load(path, allow_pickle=False) as z:
        return seed_mean_ap(z[f'ap__{aspect}']), int(np.isfinite(z[f'ap__{aspect}']).sum()), len(z['q_pid'])


def validation_picks(root, row, seeds_of):
    """{aspect: {direction: {pick, means, per_seed}}} for a row from its validation score files under `root`."""
    out = {}
    for aspect in ASPECTS:
        head = aspect if row in P.HEAD_ROWS else 'all'
        out[aspect] = {}
        for d in DIRECTIONS:
            per_seed = {r: [_col(root, 'art_val', row, head, s, d, r, aspect)[0] for s in seeds_of] for r in READOUTS}
            picked, means = pick_readout(per_seed)
            out[aspect][d] = dict(pick=picked, means=means, per_seed=per_seed)
    return out


def held_cells(root, row, picks, seeds_of):
    """{aspect: {direction: {readout, per_seed, mean, n_scored, n_queries}}} at the validation pick."""
    out = {}
    for aspect in ASPECTS:
        head = aspect if row in P.HEAD_ROWS else 'all'
        out[aspect] = {}
        for d in DIRECTIONS:
            r = picks[aspect][d]['pick']
            vals = [_col(root, 'art_held', row, head, s, d, r, aspect) for s in seeds_of]
            ns = {v[1] for v in vals}
            assert len(ns) == 1 or len(seeds_of) > 1, vals
            out[aspect][d] = dict(readout=r, per_seed=[v[0] for v in vals], mean=float(np.mean([v[0] for v in vals])),
                                  n_scored=[v[1] for v in vals], n_queries=vals[0][2])
    return out


def paper_text_rows():
    """Table 3's text rows as printed: {row label: {direction: [emotion, style, genre]}}, read from the paper's .tex."""
    text = PAPER_TABLE.read_text()
    want = {'Plain embedding': 'plain', 'Instructed Qwen3': 'teacher', r'\method (ours)': 'cstar'}
    out, direction = {}, None
    for line in text.splitlines():
        m = re.search(r'\\emph\{(Image to image|Text to image|Image to text|Text to text)\}', line)
        if m:
            direction = {'Image to image': 'i2i', 'Text to image': 't2i', 'Image to text': 'i2t', 'Text to text': 't2t'}[m.group(1)]
            continue
        for label, row in want.items():
            if line.strip().startswith(label) and direction:
                nums = [float(x) for x in re.findall(r'(\d\.\d{3})', line.replace(r'\textbf', ''))]
                out.setdefault(direction, {})[row] = nums[-3:]
    return out


def cells(device_unused=None):
    """Picks and held cells of our SigLIP 1 model (OUT), and the Qwen3 rows and the plain SigLIP 1 row (r23's files)."""
    ours = validation_picks(OUT, 'cstar', SEEDS)
    rec = dict(
        note='ArtELingo held, label mAP at the read-out picked on validation (mean over seeds of validation mAP)',
        ours_siglip1=dict(picks=ours, cells=held_cells(OUT, 'cstar', ours, SEEDS)))
    stage1 = json.loads(R23_STAGE1.read_text())
    paper = paper_text_rows()
    rec['paper_table_3_as_printed'] = paper
    for name, row, seeds_of in (('ours_qwen3', 'cstar', SEEDS), ('plain_qwen3', 'plain', ('x',)),
                                ('instructed_qwen3', 'teacher', ('x',)), ('plain_siglip1', 'siglip_so400m', ('x',))):
        pk = validation_picks(R23_OUT, row, seeds_of)
        for a in ASPECTS:           # the pick must be r23's committed stage 1 pick
            for d in DIRECTIONS:
                assert pk[a][d]['pick'] == stage1['picks'][KIND][row][a][d], (name, a, d)
        rec[name] = dict(picks=pk, cells=held_cells(R23_OUT, row, pk, seeds_of))
    # the Qwen3 rows rebuilt against the printed table
    check = {}
    for name, row in (('ours_qwen3', 'cstar'), ('plain_qwen3', 'plain'), ('instructed_qwen3', 'teacher')):
        for d in DIRECTIONS:
            built = [rec[name]['cells'][a][d]['mean'] for a in ASPECTS]
            printed = paper[d][row]
            check[f'{name}/{d}'] = dict(built=[round(b, 3) for b in built], printed=printed,
                                        equal=[round(b, 3) == p for b, p in zip(built, printed)])
    rec['qwen3_rows_vs_paper'] = check
    path = RECORDS / 'text.json'
    path.write_text(json.dumps(rec, indent=1, allow_nan=False) + '\n')
    print('wrote', path)
    return rec


# ------------------------------------------------------------------------------------------------ part 2: per seed
R17 = TEST / '20261103_new_scores'
R17_TABLES = Path('/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage3/tables_artelingo.json')
R14_PERQUERY = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/stage3/artret/siglip1/cstar/stage5/perquery')
R11_VERDICT = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs/stage5/verdict.json')
ENCODERS_TEX = PAPER_TABLE.parent / 'encoders.tex'
CLASS_COUNT = dict(emotion=9, style=27, genre=10)


def printed_encoder_rows():
    """Table 6 as printed ({'qwen': {'instructed': [6 numbers], 'ours': [6]}, 'siglip1': {'ours': [6]}}): AMI emotion, style,
    genre then mAP emotion, style, genre. Read from encoders.tex."""
    out, block = {}, None
    for line in ENCODERS_TEX.read_text().splitlines():
        m = re.search(r'\\emph\{(Qwen3|SigLIP 2|SigLIP 1|CLIP L/14)\}', line)
        if m:
            block = {'Qwen3': 'qwen', 'SigLIP 2': 'siglip2', 'SigLIP 1': 'siglip1', 'CLIP L/14': 'clip_l14'}[m.group(1)]
            continue
        key = 'instructed' if line.strip().startswith('Instructed Qwen3') else 'ours' if r'\method (ours)' in line else None
        if block and key:
            out.setdefault(block, {})[key] = [float(x) for x in re.findall(r'(\d\.\d{3})', line)][-6:]
    return out


def summary(per_seed, printed):
    """Seed mean and its comparison with the printed cell: the rounded mean must equal it."""
    v = np.asarray(per_seed, dtype=np.float64)
    mean = float(v.mean())
    return dict(per_seed=[float(x) for x in v], mean=mean, min=float(v.min()), max=float(v.max()),
                printed=printed, rebuilt_rounded=round(mean, 3), equals_printed=None if printed is None else (round(mean, 3) == printed))


def _ami_per_seed(enc, aspect):
    tables = json.loads(R17_TABLES.read_text())
    hit = [x for x in tables['seed_spreads'] if x['key'] == f'cstar/artelingo/{enc}/{aspect}/ami']
    assert len(hit) == 1 and hit[0]['k'] == CLASS_COUNT[aspect], (enc, aspect, hit)
    return hit[0]['test_per_seed']


def _r17_cell(sc, enc, aspect, metric='ami'):
    """The r17 verdict cell of (SC, encoder, aspect) at the class count: its candidate and comparator means."""
    cells = json.loads((R17 / 'records/verdicts.json').read_text())['cells']
    hit = [c for c in cells if c['sc'] == sc and c['set'] == 'artelingo' and c['encoder'] == enc and c['aspect'] == aspect
           and c['metric'] == metric and c['kind'] == 'class_count']
    assert len(hit) == 1, (sc, enc, aspect, len(hit))
    return hit[0]


def _map_ours_siglip1(aspect):
    out = []
    for s in SEEDS:
        with np.load(R14_PERQUERY / f'cstar__{aspect}__h{aspect}__s{s}.npz', allow_pickle=False) as z:
            out.append(seed_mean_ap(z['ap']))
    return out


def per_seed():
    """Part 2: ours on SigLIP 1 and on Qwen3, and the instructed Qwen3, per seed, every Table 2 image cell (class-count AMI)
    and every Table 3 cell (image to image and the three text directions). The instructed Qwen3 has no training seeds: one
    value per cell. Every rebuilt mean is compared with the printed value (rounded to 3 places)."""
    printed = printed_encoder_rows()
    text = json.loads((RECORDS / 'text.json').read_text())
    verdict = json.loads(R11_VERDICT.read_text())
    rec = dict(note='per-seed values; "instructed_qwen3" is unseeded (one value). printed = Table 6 (encoders.tex) for '
                    'AMI and image mAP, Table 3 (main_retrieval.tex) for text directions',
               ours_siglip1={}, ours_qwen3={}, instructed_qwen3={})
    paper = text['paper_table_3_as_printed']
    for i, a in enumerate(ASPECTS):
        # Table 2 image cells (class-count AMI)
        rec['ours_siglip1'][f'ami/{a}'] = summary(_ami_per_seed('siglip1', a), printed['siglip1']['ours'][i])
        rec['ours_qwen3'][f'ami/{a}'] = summary(_ami_per_seed('qwen', a), printed['qwen']['ours'][i])
        inst = _r17_cell('SC3', 'qwen', a)['comp']
        rec['instructed_qwen3'][f'ami/{a}'] = summary([inst], printed['qwen']['instructed'][i])
        # check the r17 candidate mean equals the per-seed mean rebuilt
        for enc, key in (('siglip1', 'ours_siglip1'), ('qwen', 'ours_qwen3')):
            cand = _r17_cell('SC1', enc, a)['cand']
            rec[key][f'ami/{a}']['r17_cell_mean'] = cand
            rec[key][f'ami/{a}']['equals_r17_cell'] = bool(abs(cand - rec[key][f'ami/{a}']['mean']) < 1e-12)
        # Table 3 image to image
        rec['ours_siglip1'][f'i2i/{a}'] = summary(_map_ours_siglip1(a), printed['siglip1']['ours'][3 + i])
        rec['ours_qwen3'][f'i2i/{a}'] = summary(verdict['map']['cstar'][a]['per_seed'], printed['qwen']['ours'][3 + i])
        rec['instructed_qwen3'][f'i2i/{a}'] = summary([verdict['map']['instructed'][a]['mean']],
                                                      printed['qwen']['instructed'][3 + i])
        # Table 3 text directions: ours SigLIP 1 (this round), ours Qwen3 and the instructed Qwen3 (r23)
        for d in DIRECTIONS:
            rec['ours_siglip1'][f'{d}/{a}'] = dict(summary(text['ours_siglip1']['cells'][a][d]['per_seed'], None),
                                                    readout=text['ours_siglip1']['cells'][a][d]['readout'])
            rec['ours_qwen3'][f'{d}/{a}'] = summary(text['ours_qwen3']['cells'][a][d]['per_seed'], paper[d]['cstar'][i])
            rec['instructed_qwen3'][f'{d}/{a}'] = summary(text['instructed_qwen3']['cells'][a][d]['per_seed'],
                                                          paper[d]['teacher'][i])
    # ours SigLIP 1 per seed against the instructed Qwen3 value, cell by cell
    for key, v in rec['ours_siglip1'].items():
        inst = rec['instructed_qwen3'][key]['mean']
        v['seeds_above_instructed_qwen3'] = int(sum(x > inst for x in v['per_seed']))
        v['instructed_qwen3'] = inst
    unequal = [f'{m}/{k}' for m in ('ours_siglip1', 'ours_qwen3', 'instructed_qwen3') for k, v in rec[m].items()
               if v.get('printed') is not None and v['equals_printed'] is False]
    rec['mean_not_equal_to_printed'] = unequal
    path = RECORDS / 'per_seed.json'
    path.write_text(json.dumps(rec, indent=1, allow_nan=False) + '\n')
    print('wrote', path, 'cells not equal to the printed value:', unequal)
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('replay'); a.add_argument('--device', default='cuda')
    b = sub.add_parser('score')
    b.add_argument('--set', required=True, choices=('art_val', 'art_held'), dest='set_name')
    b.add_argument('--aspects', nargs='+', default=list(ASPECTS), choices=ASPECTS)
    b.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    b.add_argument('--device', default='cuda'); b.add_argument('--block', type=int, default=1024)
    sub.add_parser('cells')
    sub.add_parser('per-seed')
    args = p.parse_args()
    if args.cmd == 'replay':
        replay(args.device)
    elif args.cmd == 'score':
        score(args.set_name, args.aspects, args.seeds, args.device, args.block)
    elif args.cmd == 'cells':
        cells()
    else:
        per_seed()


if __name__ == '__main__':
    main()
