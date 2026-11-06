"""Part b: saved genre text-to-text AP at each arm's frozen r23 pick (CPU)."""
import numpy as np

from mc_common import P, ROUND, interval, matched, record, saved, write
import tcl_data as D
from tcl_metrics import macro_map
import mt_paths


def load(set_name, kind):
    picks = record(P.GROUP[set_name])['picks'][kind]
    pid, m = saved(set_name, kind, 'plain', 'all', 'x', 't2t', picks['plain']['genre']['t2t'], 'genre')
    cs = []
    for seed in P.SEEDS:
        cp, c = saved(set_name, kind, 'cstar', 'genre', seed, 't2t', picks['cstar']['genre']['t2t'], 'genre')
        assert np.array_equal(cp, pid)
        cs.append(c)
    c = np.stack(cs)
    mask = matched(c, m)
    f = D.frame(set_name, kind)
    assert np.array_equal(pid, f['tp'])
    return c[:, mask], m[mask], pid[mask], f['labels']['genre']['t'][mask]


def main():
    c, m, pid, y = load('mm_test', 'narrative')
    overall = float(c.mean() - m.mean())
    macro_c = float(np.mean([macro_map(x, y) for x in c]))
    macro_m = float(macro_map(m, y))
    assert abs(c.mean() - .4564) < 1e-4 and abs(m.mean() - .4701) < 1e-4
    assert abs(overall - -.0137) < 1e-4
    assert abs(macro_c - .3921) < 1e-4 and abs(macro_m - .3901) < 1e-4
    sides = {f'mm_test/{k}': load('mm_test', k) for k in P.KINDS if k != 'narrative'}
    sides['mm_val/narrative'] = load('mm_val', 'narrative')
    sides['art_held/caption'] = load('art_held', 'caption')
    rows = []
    for code, name in enumerate(mt_paths.CLASSES['genre']):
        keep = y == code
        if not keep.any():
            continue
        share = float(keep.mean())
        diff = float(c[:, keep].mean() - m[keep].mean())
        beside = {}
        for key, (cc, mm, _, yy) in sides.items():
            side_code = D.r22_names('genre').index(D.class_map('genre')[name]) if key.startswith('art') else code
            kk = yy == side_code
            beside[key] = float(cc[:, kk].mean() - mm[kk].mean()) if kk.any() else None
        rows.append(dict(class_name=name, n_query=int(keep.sum()), share=share,
                         cstar=float(c[:, keep].mean()), plain=float(m[keep].mean()), difference=diff,
                         interval=interval(c[:, keep], m[keep], pid[keep]), contribution=share * diff,
                         difference_without=float(c[:, ~keep].mean() - m[~keep].mean()), beside=beside))
    rows.sort(key=lambda r: -r['n_query'])
    assert abs(sum(r['contribution'] for r in rows) - overall) < 1e-6
    # Most gap-holding contributions first (the gap is negative here).
    ranked = sorted(rows, key=lambda r: -r['contribution'] / overall)
    result = dict(classes=rows, overall=dict(cstar=float(c.mean()), plain=float(m.mean()), difference=overall),
                  macro=dict(cstar=macro_c, plain=macro_m),
                  gap_share={f'top_{n}': sum(r['contribution'] for r in ranked[:n]) / overall for n in (1, 2)},
                  reference=dict(cstar=.4564, plain=.4701, difference=-.0137, reproduced=True))
    write(ROUND / 'records/b_classes.json', result)
    print('| Class | Queries | Share | c* AP | Plain AP | Difference | 95% interval | Contribution | Gap without | val | formal | emotional | historical | unified | Art caption |')
    print('|' + '---|' * 15)
    for r in rows:
        b = r['beside']
        extra = [b[k] for k in ('mm_val/narrative', 'mm_test/formal', 'mm_test/emotional', 'mm_test/historical', 'mm_test/unified', 'art_held/caption')]
        print(f"| {r['class_name']} | {r['n_query']} | {r['share']:.4f} | {r['cstar']:.4f} | {r['plain']:.4f} | {r['difference']:+.4f} | [{r['interval']['lower']:+.4f}, {r['interval']['upper']:+.4f}] | {r['contribution']:+.4f} | {r['difference_without']:+.4f} | " + ' | '.join('null' if v is None else f'{v:+.4f}' for v in extra) + ' |')
    print(f"Overall: c*={c.mean():.6f}, plain={m.mean():.6f}, difference={overall:+.6f}; macro={macro_c:.6f}/{macro_m:.6f}; gap shares={result['gap_share']}")


if __name__ == '__main__':
    main()
