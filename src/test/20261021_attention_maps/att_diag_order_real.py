"""Token-order test on real paintings: change one region, see where the image tokens change.

Y = painting; X = the same painting with a region painted over. Per image token, d = ||h_Y - h_X|| at
hidden_states[l]. Score = AUC of d for changed cells against unchanged cells (cells cut by the region edge left out),
read in the assumed order (row-major grid_hw, as the maps), the transposed order and a random order.
A correct order gives AUC near 1 where the tokens stay local; a wrong order gives about 0.5.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path('/project/CoSiR/src/test/20261021_attention_maps')
sys.path.insert(0, str(HERE))
import att_diag_spatial as S   # noqa: E402  (sets sys.path for the repo)
import att_core as C           # noqa: E402

OUT = HERE / 'outputs/spatial_diag'
LAYERS = (0, 1, 4, 8, 16)
KS = (0, 625, 1250, 1875, 3125, 4375)


def hidden(model, im):
    torch, enc = model.torch, model.enc
    enc.instruction = model.prompts['default']
    o = enc.p(text=[enc._prompt([{'type': 'image'}])], images=[im], do_resize=False, return_tensors='pt')
    ids = o['input_ids'][0].numpy()
    tt = C.token_types(ids, 'img')
    thw = o['image_grid_thw'][0].numpy()
    grid = (int(thw[1]) // 2, int(thw[2]) // 2)
    assert im.size == (grid[1] * 32, grid[0] * 32) and int((tt == 2).sum()) == grid[0] * grid[1]
    kw = {k: v.to(model.device, model.dtype) if v.is_floating_point() else v.to(model.device) for k, v in o.items()}
    with torch.inference_mode():
        out = enc.m(**kw, use_cache=False, output_hidden_states=True)
    pos = torch.as_tensor(np.flatnonzero(tt == 2), device=model.device)
    return grid, {l: out.hidden_states[l][0, pos].float().cpu().numpy() for l in LAYERS}


def masks(size):
    """(name, drawer) pairs; each returns the changed-pixel mask for a W x H image."""
    W, H = size
    a = np.zeros((H, W), bool); a[:, : (3 * W // 8) // 32 * 32 + 16] = True          # left part, edge mid-cell
    b = np.zeros((H, W), bool); b[: (H // 4) // 32 * 32 + 16, :] = True              # top part
    c = np.zeros((H, W), bool); y0, x0 = (2 * H // 3) // 32 * 32 + 16, (W // 4) // 32 * 32 + 16
    c[y0:y0 + 96, x0:x0 + 96] = True                                                # 96 px square, off-centre
    return [('left_3/8', a), ('top_1/4', b), ('square_96', c)]


def cell_truth(mask, grid):
    m = mask.reshape(grid[0], 32, grid[1], 32).mean((1, 3))
    t = np.full(grid, -1, np.int8)
    t[m == 1] = 1
    t[m == 0] = 0
    return t


def auc(score, truth):
    pos, neg = score[truth == 1], score[truth == 0]
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images = S.Images()
    model = S.PlainModel('cuda', images)
    rng = np.random.default_rng(0)
    rows = []
    for k in KS:
        pm = C.load_painting('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_attention/', k)['painting']
        Y = images.painting(pm)
        grid, hy = hidden(model, Y)
        for name, mask in masks(Y.size):
            arr = np.asarray(Y).copy()
            arr[mask] = 0
            from PIL import Image
            X = Image.fromarray(arr)
            g2, hx = hidden(model, X)
            assert g2 == grid
            truth = cell_truth(mask, grid)
            for l in LAYERS:
                d = np.linalg.norm(hy[l] - hx[l], axis=-1)
                true = d.reshape(grid)
                if grid[0] != grid[1]:
                    tr = d.reshape(grid[1], grid[0]).T
                else:
                    tr = d.reshape(grid).T
                rows.append((k, name, l, auc(true, truth), auc(tr, truth), auc(rng.permutation(d).reshape(grid), truth)))
            X.close()
        Y.close()
        print(f'k={k} grid={grid} done', flush=True)
    rows = np.array(rows, dtype=object)
    print('\nmedian AUC over paintings (true / transposed / random), rows = layer, cols = region')
    for l in LAYERS:
        cells = []
        for name in ('left_3/8', 'top_1/4', 'square_96'):
            sel = [(r[3], r[4], r[5]) for r in rows if r[1] == name and r[2] == l]
            t, tr, rd = np.median(np.array(sel, float), 0)
            mn = min(s[0] for s in sel)
            cells.append(f'{name}: {t:.3f} (min {mn:.3f}) / {tr:.3f} / {rd:.3f}')
        print(f'hidden[{l:2d}]  ' + '   '.join(cells))
    np.save(OUT / 'order_real_rows.npy', rows, allow_pickle=True)


if __name__ == '__main__':
    main()
