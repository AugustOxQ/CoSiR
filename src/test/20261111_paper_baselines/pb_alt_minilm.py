"""Exploratory MiniLM k-means read-out, with the unchanged r25 scoring path.

PB_SMOKE=1 uses the scorer's smoke draws and writes only logs/alt_minilm_smoke.json.
Production writes records/alt_minilm_kmeans.json. Existing records are never overwritten.
"""
import json
import time

import numpy as np

import pb_score as S
from pb_io import file_sha, names_sha, write_json
from pb_partitions import aspect_ks


def main():
    start = time.monotonic()
    P = S.P
    target = (P.ROUND / 'logs' / 'alt_minilm_smoke.json' if P.SMOKE else
              P.RECORDS / 'alt_minilm_kmeans.json')
    if target.exists():
        raise FileExistsError(target)
    names, data = S.candidate_inputs()
    root = P.DEFAULT_OUT / 'bertopic'
    emb_path, meta_path = root / 'minilm.npy', root / 'minilm.json'
    meta = json.loads(meta_path.read_text())
    digest = names_sha(names)
    if meta.get('names_sha256') != digest:
        raise ValueError('MiniLM names hash differs from scorer row order')
    emb_hash = file_sha(emb_path)
    if meta.get('sha256') != emb_hash:
        raise ValueError('MiniLM embedding hash differs from sidecar')
    rows = np.load(emb_path, allow_pickle=False)
    if rows.shape != (len(names), 384) or not np.isfinite(rows).all():
        raise ValueError('Expected finite MiniLM rows [12281, 384]')
    rows = rows.astype(np.float32)
    base = json.loads((P.RECORDS / 'cells.json').read_text())
    reference = {(c['aspect'], c['k']): c for c in base if c['comparator'] == 'vlm2vec'}
    ranges = json.loads(P.RANGES.read_text())
    hashes = {str(p): file_sha(p) for p in
              (meta_path, P.RANGES, P.RECORDS / 'cells.json')}
    hashes.update({str(emb_path): emb_hash, 'names_newline_utf8': digest})
    cells = []
    for aspect in P.ASPECTS:
        for k in aspect_ks(aspect):
            cand, y = data[aspect, k]
            # Same fitting function as pb_score.main's VLM2Vec branch.
            comp = S.ns_metrics.partitions(rows, k, P.SET)
            r = float(ranges[f'{P.SET}/{P.CSTAR_ENC}/{aspect}/ami/{k}']['r'])
            cell = S.score_cell('minilm_kmeans', aspect, cand, comp, y, r,
                                reference[aspect, k]['gate'], hashes, k)
            cell['candidate_gate_note'] = ('Not rerun: reuse exact gate and per-seed AMI from cells.json; '
                                           'score_cell still checks candidate mean against that gate.')
            cells.append(cell)
            print(json.dumps({key: cell[key] for key in
                              ('id', 'comp', 'd', 'lo', 'hi', 'label', 'n_draws')}), flush=True)
    write_json(target, cells)
    print(f'{len(cells)} cells -> {target}; {time.monotonic() - start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
