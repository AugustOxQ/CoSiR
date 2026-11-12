"""r21 diagnostic (user, 2026-11-12; decides nothing): SI, DBI and CHI on the validation split, seed 0, for c* and the
stage 1a plain arms (S1 = all 50 DEC epochs kept to the end; S2 = the best c* dev-loss state among DEC epochs). Tests
whether a longer DEC makes clusters on unseen paintings much tighter, while its AMI falls.

Same procedure as the test diagnostics (`r21_table.diag`) on the validation exports: r17's k-means partitions at the class
count (`ns_metrics.partitions`, float16 rows cast to float32), SI (cosine) and DBI / CHI (unit rows) on state 0; AMI is the
mean over the 3 states (`r21_score.val_aspect`'s rule).
Run: python r21_val_internal.py > /data/SSD2/cosir_r21/20261111_dec_heads/scores/val_internal.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r21_score as SC  # noqa: E402
import r21_table as T  # noqa: E402

TRAIN_ROOT = Path('/data/SSD2/cosir_r21/20261111_dec_heads/full/train')
OUT = Path('/data/SSD2/cosir_r21/20261111_dec_heads/scores/val_internal.json')
ARMS = ['cstar', 'dec-plain_l1_w1_S2', 'dec-plain_l1_w5_S2', 'dec-plain_l10_w1_S2', 'dec-plain_l10_w5_S2',
        'dec-plain_l1_w1_S1', 'dec-plain_l1_w5_S1', 'dec-plain_l10_w1_S1', 'dec-plain_l10_w5_S1']
SEED = 0


def main():
    SC._deps()
    res = {}
    for arm in ARMS:
        res[arm] = {}
        for a in SC.ASPECTS:
            x16, side = SC.load_val(arm, a, SEED, None if arm == SC.CSTAR else TRAIN_ROOT)
            k = SC.ns_paths.CLASS_COUNT[(SC.SET, a)]
            y = SC.ns_data.labels(SC.SET, 'val', a)
            x = x16.astype(np.float32)
            parts = SC.ns_metrics.partitions(x, k, SC.SET)
            ami = SC.ns_metrics.score_partitions('ami', y, parts)
            res[arm][a] = dict(si=T.silhouette_cosine(x, parts[0]), **T.internal_scores_unit(x, parts[0]), ami=float(ami),
                               largest=float(np.bincount(parts[0]).max() / len(parts[0])),
                               best_epoch=side.get('best_epoch'), epochs_run=side.get('epochs_run'))
        print(f'{arm}: done', file=sys.stderr, flush=True)
    OUT.write_text(json.dumps(dict(seed=SEED, split='val', rows=SC.N_VAL, arms=res), indent=1) + '\n')
    ref = res['cstar']
    print('validation, seed 0. Per aspect: SI / DBI / CHI / AMI (change from c* in brackets); kept epoch / epochs run')
    for arm in ARMS:
        cells = []
        for a in SC.ASPECTS:
            r, c = res[arm][a], ref[a]
            if arm == 'cstar':
                cells.append(f"{r['si']:.4f} / {r['dbi']:.3f} / {r['chi']:.0f} / {r['ami']:.4f}; ep {r['best_epoch']}/{r['epochs_run']}")
            else:
                cells.append(f"{r['si']:.4f} ({r['si'] - c['si']:+.4f}) / {r['dbi']:.3f} ({r['dbi'] - c['dbi']:+.3f}) / "
                             f"{r['chi']:.0f} ({r['chi'] - c['chi']:+.0f}) / {r['ami']:.4f} ({r['ami'] - c['ami']:+.4f}); "
                             f"ep {r['best_epoch']}/{r['epochs_run']}")
        print(f'{arm}\n  emotion: {cells[0]}\n  style:   {cells[1]}\n  genre:   {cells[2]}')


if __name__ == '__main__':
    main()
