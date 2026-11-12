"""r28 close calls: per seed, ours on SigLIP 1 against CLAY on SigLIP 1 (Table 2 AMI, Table 3 image mAP, ArtELingo), and the
validation per-seed difference SigLIP 1 minus SigLIP 2 in the six-cell mean. Numbers are read from the record files only:
`records/per_seed.json` (ours_siglip1), `records/clay_crl.json` (CLAY, SigLIP 1), `records/valpick.json`.
CLAY is one value per cell (no seeds), so the count is how many of ours' five seeds are above it.

    python r28_close_calls.py      writes records/close_calls.json
"""
import json
from pathlib import Path

REC = Path(__file__).resolve().parent / 'records'
ASPECTS = ('emotion', 'style', 'genre')
# per_seed.json key -> clay_crl.json cell: Table 2 AMI = `ami`, Table 3 image mAP = i2i = `map`
CELLS = {'ami': 'ami', 'i2i': 'map'}


def vs_clay(per_seed, clay_crl):
    out = {}
    for kind, ckind in CELLS.items():
        for a in ASPECTS:
            seeds = per_seed['ours_siglip1'][f'{kind}/{a}']['per_seed']
            clay = clay_crl['cells'][f'artelingo/{a}/{ckind}/clay']['siglip1']
            assert len(seeds) == 5
            out[f'{kind}/{a}'] = dict(
                ours_per_seed=seeds, ours_mean=sum(seeds) / 5, clay=clay,
                diff_per_seed=[s - clay for s in seeds], seeds_above_clay=sum(s > clay for s in seeds),
                seeds_equal_clay=sum(s == clay for s in seeds), n_seeds=5,
                clay_readout=clay_crl['cells'][f'artelingo/{a}/{ckind}/clay']['readout'])
    return out


def valpick_diff(valpick, a='siglip1', b='siglip2'):
    assert valpick['seeds'] == [0, 1, 2, 3, 4]
    cells = valpick['cells']
    per = {}
    for enc in (a, b):
        c = valpick['encoders'][enc]['cells']
        assert list(c) == cells or set(c) == set(cells)
        per[enc] = [sum(c[k]['per_seed'][s] for k in cells) / len(cells) for s in range(5)]
    diff = [x - y for x, y in zip(per[a], per[b])]
    return dict(cells=cells, six_cell_mean_per_seed=per, diff_per_seed=diff, diff_mean=sum(diff) / 5,
                seeds_a_above_b=sum(d > 0 for d in diff), n_seeds=5,
                recorded_means={e: valpick['pick']['scores'][e] for e in (a, b)},
                per_cell_diff_per_seed={k: [valpick['encoders'][a]['cells'][k]['per_seed'][s] - valpick['encoders'][b]['cells'][k]['per_seed'][s]
                                            for s in range(5)] for k in cells})


def main():
    load = lambda f: json.loads((REC / f).read_text())
    res = dict(note='ours on SigLIP 1 vs CLAY on SigLIP 1, ArtELingo test (Table 2 AMI, Table 3 image mAP); CLAY is one value per cell. '
                    'valpick: validation, SigLIP 1 minus SigLIP 2, mean of the six cell means per seed.',
               sources=['records/per_seed.json ours_siglip1', 'records/clay_crl.json cells/artelingo/*/{ami,map}/clay/siglip1',
                        'records/valpick.json encoders'],
               vs_clay=vs_clay(load('per_seed.json'), load('clay_crl.json')), valpick_siglip1_minus_siglip2=valpick_diff(load('valpick.json')))
    (REC / 'close_calls.json').write_text(json.dumps(res, indent=1) + '\n')
    for k, v in res['vs_clay'].items():
        print(f"{k:12s} ours mean {v['ours_mean']:.4f}  clay {v['clay']:.4f}  seeds above {v['seeds_above_clay']}/5")
    d = res['valpick_siglip1_minus_siglip2']
    print('valpick diff per seed', [round(x, 5) for x in d['diff_per_seed']], 'mean', round(d['diff_mean'], 5), 'seeds s1>s2', d['seeds_a_above_b'])


if __name__ == '__main__':
    main()
