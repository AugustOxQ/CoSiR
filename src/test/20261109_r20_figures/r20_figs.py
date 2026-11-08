"""CLI of the r20 candidate-figure build.

    r20_figs.py pd [--smoke]          Wikidata public-domain table            -> outputs/pd/
    r20_figs.py partitions [--smoke]  k-means partitions (CPU-heavy)           -> outputs/partitions/
    r20_figs.py candidates [--smoke]  every candidate panel and teaser         -> outputs/candidates/candidates.json
    r20_figs.py viewer [--smoke]      static review page with thumbnails        -> outputs/viewer/

`--smoke` writes to outputs/smoke/ and uses 1,500 evenly spaced rows of each set (partitions: one aspect only).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['pd', 'partitions', 'candidates', 'viewer'])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--sets', nargs='*', choices=['artelingo', 'mmart'], help='partitions/candidates: only these sets')
    a = ap.parse_args(argv)
    if a.step == 'pd':
        import r20_pd
        out = r20_pd.run(smoke=a.smoke)
    elif a.step == 'partitions':
        import r20_part
        out = r20_part.run(smoke=a.smoke, sets=a.sets)
    elif a.step == 'candidates':
        import r20_cand
        doc = r20_cand.run(smoke=a.smoke, sets=a.sets)
        out = {n: dict(paintings=len(s['paintings']), teaser=len(s['teaser']), **s['teaser_info']) for n, s in doc['sets'].items()}
    else:
        import r20_view
        out = r20_view.run(smoke=a.smoke)
    print(json.dumps(out, default=str))


if __name__ == '__main__':
    main()
