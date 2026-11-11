"""Fresh BERTopic fits at other cluster counts K, using r25's `pb_bertopic.run_seed` unchanged.
Run with /root/venvs/bertopic/bin/python. Reads r25's MiniLM cache and docs (read only); writes only under --out.

    kr_bertopic_fit.py --out <dir> --k 10            # refit gate (scratch dir)
    kr_bertopic_fit.py --out <dir> --k 5 8 16 40     # grid fits
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

R25 = Path(__file__).resolve().parents[1] / '20261111_paper_baselines'
sys.path.insert(0, str(R25))
import pb_paths as P            # noqa: E402
import pb_bertopic as B         # noqa: E402
from pb_io import file_sha, names_sha  # noqa: E402

R25_BT = P.DEFAULT_OUT / 'bertopic'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--k', type=int, nargs='+', required=True)
    a = ap.parse_args()
    assert a.out.resolve() != R25_BT.resolve() and P.DEFAULT_OUT.resolve() not in a.out.resolve().parents
    a.out.mkdir(parents=True, exist_ok=True)
    inp = P.DEFAULT_OUT / 'inputs' / 'held_docs.json'
    names, docs, _ = B.load_docs(inp)
    input_sha = file_sha(inp)
    emb_path, side = R25_BT / 'minilm.npy', R25_BT / 'minilm.json'
    emb_meta = json.loads(side.read_text())
    assert emb_meta['sha256'] == file_sha(emb_path) and emb_meta['names_sha256'] == names_sha(names)
    assert emb_meta['modules'] == B.MINILM_MODULES and emb_meta['max_seq_length'] == B.MINILM_MAX_SEQ_LENGTH
    emb = np.load(emb_path, allow_pickle=False)
    B.check_minilm_rows(emb, len(docs))
    t0 = time.time()
    for k in a.k:
        for seed in P.UMAP_SEEDS:
            B.run_seed(docs, names, emb, emb_meta, a.out, seed, input_sha, False, k)
    print(f'done in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
