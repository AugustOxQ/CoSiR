"""Validation retrieval loop. Run from the repository root; no fitting or held reads."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

import tx_paths as paths
import tx_data as data
import tx_metrics as metrics
import tx_model
import rt_data
import rt_metrics

READOUTS = ('r0', 'center', 'center_k10')
DIRECTIONS = ('t2i', 'i2t', 't2t', 'i2i')


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def release(device):
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()


def row_arrays(out, device, smoke=0):
    """Yield one row at a time; only the small saved head is constructed here."""
    c = data.val_captions()
    n = smoke or paths.N_PAINTINGS
    assert 0 < n <= paths.N_PAINTINGS
    cap_keep = c['pid'] < n
    for row in ('plain', 'clip_l14', 'siglip2_so400m', 'siglip_so400m',
                'cstar', 'cstar_free', 'teacher', 'roberta'):
        aspects = paths.ASPECTS if row in ('cstar', 'cstar_free', 'teacher') else (None,)
        for aspect in aspects:
            if row == 'teacher' and not (data.text_path(row, aspect, out) / 'txt.npy').exists():
                if smoke:
                    print(f'skip teacher/{aspect}: text store absent', flush=True)
                    continue
                raise FileNotFoundError(data.text_path(row, aspect, out))
            if row in ('cstar', 'cstar_free'):
                image_side, caption_side = tx_model.load_sides(aspect, device=device)
                text = data.text_rows('plain')[cap_keep]
                image = image_side(data.image_rows('plain')[:n])
                text = (caption_side if row == 'cstar' else image_side)(text)
                del image_side, caption_side
            else:
                text = data.text_rows(row, aspect, out)[cap_keep]
                image = None if row == 'roberta' else data.image_rows(row, aspect or 'emotion')[:n]
            yield row, aspect, text, image
            del text, image
            release(device)


def entries_for(aspects, direction, cp, n):
    caption_labels, painting_labels = data.caption_labels(), data.painting_labels()
    entries = {}
    q_text, g_text = direction[0] == 't', direction[2] == 't'
    for a in aspects:
        q = caption_labels[a][cp] if q_text else painting_labels[a][:n]
        g = caption_labels[a][cp] if g_text else painting_labels[a][:n]
        entry = dict(q=q, g=g, q_keep=None, g_keep=None)
        entries[a] = entry
        if direction != 'i2i':
            flags = ~data.names_label(a)[cp]
            entries[a + '__no_leak'] = dict(entry, q_keep=flags if q_text else None,
                                            g_keep=flags if g_text else None)
        entries[a + '__cross_painting'] = dict(entry, exclude_same_painting=True)
    return entries


def finite_mean(x):
    return float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else float('nan')


def record_ranking(out, row, aspect, direction, readout, q, g, qp, gp, entries, device,
                   records, pairs, self_match=False, save=True, extra=None):
    started = time.monotonic()
    result = metrics.evaluate(q, g, qp, gp, entries, self_match, readout, device)
    base = dict(row=row, aspect=aspect, direction=direction, readout=readout, **(extra or {}))
    for key, ap in result['ap'].items():
        records.append(dict(base, entry=key, map=finite_mean(ap),
                            macro_map=metrics.macro_map(ap, entries[key]['q']),
                            chance=result['chance'][key], n_query=result['n_kept'][key],
                            n_gallery=len(g), dropped=result['dropped'][key]))
    if result['first_rank'] is not None:
        pairs.append(dict(base, **metrics.summarize_pair(result['first_rank'])))
    if save:
        path = Path(out) / 'perquery' / f'{row}__{aspect or "all"}__{direction}__{readout}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f'ap__{k}': v for k, v in result['ap'].items()}
        if qp is not None:
            arrays['q_pid'] = qp
        if result['first_rank'] is not None:
            arrays['first_rank'] = result['first_rank']
        np.savez_compressed(path, **arrays)
    print(f'{row}/{aspect or "all"}/{direction}/{readout}: {time.monotonic() - started:.2f}s', flush=True)
    return result


def metric_gate(device):
    labels = data.painting_labels()
    result = {}
    for row in ('plain', 'cstar'):
        x = data.image_rows('plain') if row == 'plain' else rt_data.embed('cstar', 'val', 'emotion', 0)
        pid = np.arange(len(x), dtype=np.int64)
        entries = {a: dict(q=y, g=y) for a, y in labels.items()}
        got = metrics.evaluate(x, x, pid, pid, entries, True, 'r0', device)
        result[row] = {}
        for a, y in labels.items():
            expected = rt_metrics.retrieval(x, y, workers=1)
            error = abs(finite_mean(got['ap'][a]) - float(expected['ap'].mean()))
            assert error < 1e-5, (row, a, error)
            result[row][a] = dict(abs_map_difference=error)
    return result


def leak_report():
    labels = data.caption_labels()
    result = {}
    for a, y in labels.items():
        flag = data.names_label(a)
        result[a] = dict(share=float(flag.mean()), n=len(flag), per_class={
            name: dict(n=int((y == i).sum()), share=float(flag[y == i].mean()))
            for i, name in enumerate(data.label_names(a)) if (y == i).any()})
    return result


def capmean_rankings(out, device, smoke, records, pairs):
    c = data.val_captions()
    n = smoke or paths.N_PAINTINGS
    keep = c['pid'] < n
    pid = np.arange(n, dtype=np.int64)
    p_labels = {a: y[:n] for a, y in data.painting_labels().items()}
    for row in ('plain', 'cstar', 'teacher'):
        for a in (paths.ASPECTS if row != 'plain' else (None,)):
            if row == 'teacher' and not (data.text_path(row, a, out) / 'txt.npy').exists():
                if smoke:
                    continue
                raise FileNotFoundError(data.text_path(row, a, out))
            text = data.text_rows('plain' if row == 'cstar' else row, a, out, raw=True)[keep]
            q = data.caption_mean(text, c['pid'][keep], n)
            if row == 'cstar':
                image_side, caption_side = tx_model.load_sides(a, device=device)
                q, g = caption_side(q), image_side(data.image_rows('plain')[:n])
                del image_side, caption_side
            else:
                g = data.image_rows(row, a or 'emotion')[:n]
            entries = {k: dict(q=p_labels[k], g=p_labels[k]) for k in (paths.ASPECTS if a is None else (a,))}
            for readout in ('r0', 'center'):
                record_ranking(out, row, a, 'capmean2i', readout, q, g, pid, pid, entries,
                               device, records, pairs)
            del text, q, g
            release(device)


def run(out, device='cuda', smoke=0):
    out = Path(out)
    n = smoke or paths.N_PAINTINGS
    assert 0 < n <= paths.N_PAINTINGS
    gates = dict(smoke=bool(smoke), caption_mean='unit(mean(raw float16 cast to float32)); float64 accumulation',
                 forward=tx_model.gate_forward(device, n if smoke else None))
    if not smoke:
        gates['metric_parity'] = metric_gate(device)
    write_json(out / 'gates.json', gates)
    c = data.val_captions()
    keep = c['pid'] < n
    cp, ip = c['pid'][keep], np.arange(n, dtype=np.int64)
    records, pairs = [], []
    for row, aspect, text, image in row_arrays(out, device, smoke):
        aspects = ('emotion',) if row == 'roberta' else (paths.ASPECTS if aspect is None else (aspect,))
        for direction in DIRECTIONS:
            if image is None and direction != 't2t' or row == 'cstar_free' and direction == 'i2i':
                continue
            q, qp = (text, cp) if direction[0] == 't' else (image, ip)
            g, gp = (text, cp) if direction[2] == 't' else (image, ip)
            entries = entries_for(aspects, direction, keep, n)
            for readout in READOUTS:
                record_ranking(out, row, aspect, direction, readout, q, g, qp, gp, entries,
                               device, records, pairs, self_match=direction in ('i2i', 't2t'))
        del text, image, q, g
        release(device)
    capmean_rankings(out, device, smoke, records, pairs)
    write_json(out / 'results/core.json', dict(smoke=bool(smoke), n_paintings=n, records=records, pairs=pairs))
    write_json(out / 'results/leak.json', leak_report())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', type=int, default=0)
    p.add_argument('--out', type=Path, default=paths.OUT)
    args = p.parse_args()
    run(args.out, args.device, args.smoke)


if __name__ == '__main__':
    main()
