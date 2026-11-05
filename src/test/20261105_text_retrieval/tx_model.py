"""The two frozen cstar TrunkHead sides; no backbone model is loaded."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import tx_paths as paths
import rt_data
import tx_data


class Side(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(2048, 1024), nn.GELU(), nn.Dropout(0.1))
        self.output = nn.Linear(1024, 2048)

    def forward(self, x):
        return F.normalize(self.output(self.trunk(x)), dim=1)


def load_sides(aspect, seed=0, device='cpu'):
    assert aspect in paths.ASPECTS
    path = rt_data.CSTAR_ROOT / f'weights/k11=40/{aspect}__s{seed}.pt'
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert set(state) == {f'{s}.{k}' for s in ('img', 'cap') for k in
                         ('trunk.0.weight', 'trunk.0.bias', 'output.weight', 'output.bias')}

    def make(side):
        model = Side()
        model.load_state_dict({k[len(side) + 1:]: v for k, v in state.items() if k.startswith(side + '.')})
        model.to(device).eval().requires_grad_(False)

        @torch.no_grad()
        def forward(x):
            assert x.ndim == 2 and x.shape[1] == 2048 and len(x)
            x = tx_data.unit(x)
            result = [model(torch.as_tensor(x[i:i + 4096], dtype=torch.float32, device=device)).cpu().numpy()
                      for i in range(0, len(x), 4096)]
            out = np.concatenate(result).astype(np.float32)
            assert out.shape == x.shape and np.isfinite(out).all()
            return out
        return forward

    return make('img'), make('cap')


def gate_forward(device='cpu', n=None):
    """Compare raw-caption means and plain images to the saved validation exports."""
    count = paths.N_PAINTINGS if n is None else n
    assert 0 < count <= paths.N_PAINTINGS
    captions = tx_data.val_captions()
    keep = captions['pid'] < count
    raw = tx_data.text_rows('plain', raw=True)[keep]
    means = tx_data.caption_mean(raw, captions['pid'][keep], count)
    images = tx_data.image_rows('plain')[:count]
    results = {}
    for aspect in paths.ASPECTS:
        image_side, caption_side = load_sides(aspect, device=device)
        path = rt_data.CSTAR_ROOT / f'train/k11=40/R6-5/{aspect}__{rt_data.GRAPHS[aspect]}__s0.npz'
        with np.load(path, allow_pickle=False) as z:
            stats = {}
            for side, got in [('img', image_side(images)), ('cap', caption_side(means))]:
                saved = z[side][:count].astype(np.float32)
                assert saved.shape == got.shape
                cosine = np.sum(tx_data.unit(saved) * tx_data.unit(got), axis=1)
                mean, error = float(cosine.mean()), float(np.max(np.abs(saved - got)))
                assert mean > 0.9999 and error < 2e-3, (aspect, side, mean, error)
                stats[side] = dict(mean_cosine=mean, max_abs_difference=error, n=count)
            results[aspect] = stats
        del image_side, caption_side
    return results
