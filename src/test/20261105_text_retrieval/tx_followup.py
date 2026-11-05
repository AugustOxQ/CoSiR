"""Two short checks after the r22 read: the own-caption pair (R0-5, R6-5) and joined c* and plain scores.

Validation only, seed 0. The rule is in the round log ("Follow-up"), written before any score. Run from this folder.
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

import tx_paths as paths
import tx_data as data
import tx_metrics as metrics
import tx_model
from tx_core import READOUTS, finite_mean, release, write_json
import rt_data
import ct_d2

DIRECTIONS = ('t2i', 'i2t', 't2t')
CELLS = [(a, d) for a in paths.ASPECTS for d in DIRECTIONS]
ALPHAS = tuple(round(i / 10, 1) for i in range(11))
RECIPE_ROOT = paths.TEST / '20261025_recipe_backbones/outputs/part1/train'
RECIPES = ('R0-5', 'R6-5')
PLAIN_GAP_SHARE = 0.5
PLAIN_MARGIN = 0.02
BAR = 5


def recipe_sides(recipe, aspect, device):
    """Same Side as c*; only the weights file differs."""
    path = RECIPE_ROOT / recipe / f'{aspect}__s0.pt'
    state = torch.load(path, map_location='cpu', weights_only=True)

    def make(side):
        model = tx_model.Side()
        model.load_state_dict({k[len(side) + 1:]: v for k, v in state.items() if k.startswith(side + '.')})
        model.to(device).eval().requires_grad_(False)

        @torch.no_grad()
        def forward(x):
            x = data.unit(x)
            out = [model(torch.as_tensor(x[i:i + 4096], dtype=torch.float32, device=device)).cpu().numpy()
                   for i in range(0, len(x), 4096)]
            return np.concatenate(out).astype(np.float32)
        return forward

    return make('img'), make('cap')


def training_overlap():
    """Paintings of the recipes' training sample that are also validation paintings."""
    with np.load(ct_d2.SAMPLE, allow_pickle=False) as z:
        sample = set(z['all_painting_ids'].tolist())
    return sum(name in sample for name in rt_data.names('val'))


def rank(q, g, qp, gp, entries, self_match, readout, device):
    result = metrics.evaluate(q, g, qp, gp, entries, self_match, readout, device)
    return dict(ap={k: finite_mean(v) for k, v in result['ap'].items()},
                pair=metrics.summarize_pair(result['first_rank']), first=result['first_rank'])


@lru_cache(maxsize=None)
def side_labels(side):
    return data.caption_labels() if side == 't' else data.painting_labels()


def entry(aspect, d):
    return dict(q=side_labels(d[0])[aspect], g=side_labels(d[2])[aspect], q_keep=None, g_keep=None)


def all_rankings(text, image, cp, ip, aspect, device):
    """{direction: {readout: rank result}} with the head's own label only."""
    out = {}
    for d in DIRECTIONS:
        q, qp = (text, cp) if d[0] == 't' else (image, ip)
        g, gp = (text, cp) if d[2] == 't' else (image, ip)
        entries = {aspect: entry(aspect, d)}
        out[d] = {ro: rank(q, g, qp, gp, entries, d == 't2t', ro, device) for ro in READOUTS}
    return out


def mrr_pick(by_readout):
    return max(READOUTS, key=lambda ro: by_readout[ro]['pair']['mrr'])


def best_head(per_head, d):
    """Each head's read-out is picked on MRR; the head with the best R@10 is read (Table 2)."""
    picks = {a: mrr_pick(per_head[a][d]) for a in per_head}
    a = max(picks, key=lambda x: per_head[x][d][picks[x]]['pair']['r10'])
    return a, picks[a], per_head[a][d][picks[a]]


def table_values():
    core = json.loads((paths.OUT / 'results/core.json').read_text())
    labels, pairs = {}, {}
    for row in ('plain', 'teacher'):
        for a, d in CELLS:
            rows = [r for r in core['records'] if r['row'] == row and r['entry'] == a and r['direction'] == d]
            labels[row, a, d] = max(r['map'] for r in rows)
    for d in DIRECTIONS:
        rows = [r for r in core['pairs'] if r['row'] == 'plain' and r['direction'] == d]
        pairs[d] = max(rows, key=lambda r: r['mrr'])['r10']
    return core, labels, pairs


def gate(core, joined, device_alpha, row):
    """a = 0 must give c*'s saved scores and a = 1 plain Qwen's, ranking by ranking."""
    worst = 0.0
    for a, d in CELLS:
        for ro in READOUTS:
            want = next(r['map'] for r in core['records'] if r['row'] == row and r['entry'] == a
                        and r['direction'] == d and r['readout'] == ro
                        and (row == 'plain' or r['aspect'] == a))
            error = abs(joined[device_alpha][a][d][ro]['ap'][a] - want)
            worst = max(worst, error)
            assert error < 2e-4, (row, a, d, ro, error)
    return worst


def check_own(texts, images, cp, ip, plain_pairs, device):
    per = {}
    for recipe in RECIPES:
        per[recipe] = {}
        for a in paths.ASPECTS:
            image_side, caption_side = recipe_sides(recipe, a, device)
            per[recipe][a] = all_rankings(caption_side(texts), image_side(images), cp, ip, a, device)
            del image_side, caption_side
            release(device)
    rows = {}
    for d in DIRECTIONS:
        head_on, ro_on, on = best_head(per['R0-5'], d)
        head_off, ro_off, off = best_head(per['R6-5'], d)
        qp = cp if d[0] == 't' else ip
        diff = (on['first'] <= 10).astype(np.float64) - (off['first'] <= 10).astype(np.float64)
        mean, lo, hi = metrics.painting_bootstrap(diff, qp, paths.N_PAINTINGS)
        gap = plain_pairs[d] - off['pair']['r10']
        rows[d] = dict(on=dict(head=head_on, readout=ro_on, **on['pair']),
                       off=dict(head=head_off, readout=ro_off, **off['pair']),
                       diff=mean, lo=lo, hi=hi, plain=plain_pairs[d], gap=gap,
                       share_of_gap=mean / gap if gap > 0 else None)
    labels = {r: {f'{a}/{d}': max(per[r][a][d][ro]['ap'][a] for ro in READOUTS) for a, d in CELLS}
              for r in RECIPES}
    above = {d: rows[d]['lo'] > 0 for d in ('t2i', 'i2t')}
    half = {d: above[d] and rows[d]['share_of_gap'] is not None
            and rows[d]['share_of_gap'] >= PLAIN_GAP_SHARE for d in ('t2i', 'i2t')}
    verdict = 'supported' if all(half.values()) else 'partly' if any(above.values()) else 'not supported'
    return dict(rows=rows, labels=labels, verdict=verdict)


def check_joined(texts, images, cp, ip, core, labels, plain_pairs, device):
    joined = {x: {} for x in ALPHAS}
    plain_t, plain_i = data.unit(texts), data.unit(images)
    for a in paths.ASPECTS:
        image_side, caption_side = tx_model.load_sides(a, device=device)
        c_t, c_i = caption_side(texts), image_side(images)
        del image_side, caption_side
        for x in ALPHAS:
            w_c, w_p = float(np.sqrt(1 - x)), float(np.sqrt(x))
            t = np.concatenate((w_c * c_t, w_p * plain_t), axis=1).astype(np.float32)
            i = np.concatenate((w_c * c_i, w_p * plain_i), axis=1).astype(np.float32)
            joined[x][a] = all_rankings(t, i, cp, ip, a, device)
            print(f'joined {a} {x}', flush=True)
            del t, i
            release(device)
    gates = dict(cstar=gate(core, joined, 0.0, 'cstar'), plain=gate(core, joined, 1.0, 'plain'))
    rows = []
    for x in ALPHAS:
        cells = {f'{a}/{d}': max(joined[x][a][d][ro]['ap'][a] for ro in READOUTS) for a, d in CELLS}
        above = sum(cells[f'{a}/{d}'] > max(labels['plain', a, d], labels['teacher', a, d]) for a, d in CELLS)
        pair = {}
        for d in DIRECTIONS:
            head, ro, best = best_head(joined[x], d)
            pair[d] = dict(head=head, readout=ro, r10=best['pair']['r10'])
        close = all(pair[d]['r10'] >= plain_pairs[d] - PLAIN_MARGIN for d in ('t2i', 'i2t'))
        rows.append(dict(alpha=x, cells_above=above, label=cells, pair=pair, close_to_plain=close,
                         passes=bool(close and above >= BAR)))
    return dict(gates=gates, rows=rows, verdict=any(r['passes'] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', type=Path, default=paths.OUT / 'results/followup.json')
    args = p.parse_args()
    captions = data.val_captions()
    cp, ip = captions['pid'], np.arange(paths.N_PAINTINGS, dtype=np.int64)
    texts = data.text_rows('plain')
    images = data.image_rows('plain')[:paths.N_PAINTINGS]
    assert len(texts) == len(cp) == paths.N_CAPTIONS
    core, labels, plain_pairs = table_values()
    result = dict(training_overlap=training_overlap(), plain_pairs=plain_pairs)
    print('overlap', result['training_overlap'], 'plain', plain_pairs, flush=True)
    result['own'] = check_own(texts, images, cp, ip, plain_pairs, args.device)
    write_json(args.out, result)
    print('own', result['own']['verdict'], flush=True)
    result['joined'] = check_joined(texts, images, cp, ip, core, labels, plain_pairs, args.device)
    write_json(args.out, result)
    print(f"followup done: own {result['own']['verdict']}, joined passes {result['joined']['verdict']}", flush=True)


if __name__ == '__main__':
    main()
