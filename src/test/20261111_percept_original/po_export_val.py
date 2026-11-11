"""r26 export: Stage 2 validation probabilities with painting names, in r17's validation key order (CPU only).

    python po_export_val.py [--cuts k40 strict] [--seeds 0 1 2 3 4] [--check-keys]

For the paper's Table 3 (retrieval read-outs picked and fitted on validation; the "paper sec4 tables" chat scores
them). Stage 2 saves `eval_prob` [6152, K] with rows in the full split's dev order (sorted validation painting names)
and that order's sha256 in its sidecar (`eval_names_sha256`). This script checks the hash, reorders the rows by painting
name to r17's `ns_data.keys('artelingo', 'val')` and writes OUT/val/<cut>/L1__s<seed>.npz {prob float32 [6152, K],
names <U [6152]} with a JSON sidecar. The test (held) outputs are not touched. `--check-keys` only checks that r17's
validation keys are the 6,152 dev paintings.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import po_paths as P
import po_percept as po          # sets the import order: r17's scorer before the claim test code
import ns_data                   # noqa: E402
import ct_held                   # noqa: E402


def dev_names(s):
    names = s['all_painting_ids'][s['dev']].tolist()
    keys = ns_data.keys('artelingo', 'val')
    assert len(names) == len(keys) == P.N_VAL and len(set(keys)) == P.N_VAL
    assert set(keys) == set(names), "r17's validation keys are not the dev paintings"
    return names, keys


def export(s, cut, seed):
    src = po.stage2_path(P.OUT, cut, seed)
    if not src.exists():
        print(f'{cut} s{seed}: no Stage 2 output (a cut with fewer than 2 topics is skipped); nothing written', flush=True)
        return None
    meta = json.loads(src.with_suffix('.json').read_text())
    names, keys = dev_names(s)
    assert meta['eval_names_sha256'] == ct_held.names_sha256(names), 'Stage 2 rows are not in the dev order'
    with np.load(src, allow_pickle=False) as z:
        prob = z['eval_prob']
    assert prob.shape == (P.N_VAL, meta['K']) and prob.dtype == np.float32
    row = {name: i for i, name in enumerate(names)}
    prob = np.ascontiguousarray(prob[np.asarray([row[k] for k in keys], dtype=np.int64)])
    out = P.OUT / 'val' / cut / f'{P.ARM}__s{seed}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, prob=prob, names=np.asarray(keys))
    side = dict(source=str(src), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                rows="r17's ns_data.keys('artelingo', 'val'), joined by painting name",
                names_sha256=ct_held.names_sha256(list(keys)), shape=list(prob.shape), K=meta['K'], cut=cut, seed=seed,
                nonfinite=bool(meta.get('nonfinite', False)), dev_auc=meta.get('dev_auc'),
                collapse='judged per seed in OUT/score/po_scores.json (collapse rule on the Stage 1 arrays)')
    out.with_suffix('.json').write_text(json.dumps(side, indent=2) + '\n')
    print(f'{cut} s{seed}: wrote {out} {tuple(prob.shape)}', flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cuts', nargs='+', choices=P.CUTS, default=list(P.CUTS))
    p.add_argument('--seeds', nargs='+', type=int, choices=P.SEEDS, default=list(P.SEEDS))
    p.add_argument('--check-keys', action='store_true')
    args = p.parse_args(argv)
    s = po.split()
    if args.check_keys:
        names, keys = dev_names(s)
        print(f'r17 validation keys = the {len(keys):,} dev paintings; first {keys[0]}', flush=True)
        return 0
    for cut in args.cuts:
        for seed in args.seeds:
            export(s, cut, seed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
