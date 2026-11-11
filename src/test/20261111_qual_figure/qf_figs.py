"""CLI of the qualitative-figure candidate build (paper section 4.6).

    qf_figs.py candidates [--smoke]  panel (a), panel (b), cluster ranking  -> outputs/candidates/candidates.json
    qf_figs.py viewer [--smoke]      static review page with thumbnails       -> outputs/viewer/index.html

`--smoke` writes to outputs/smoke/ and uses 800 evenly spaced rows for the candidate rules.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['candidates', 'viewer'])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(argv)
    if a.step == 'candidates':
        import qf_cand
        doc = qf_cand.run(smoke=a.smoke)
        out = dict(seconds=doc['seconds'], counts=doc['counts'], paintings=len(doc['paintings']))
    else:
        import qf_view
        out = qf_view.run(smoke=a.smoke)
    print(json.dumps(out, default=str))


if __name__ == '__main__':
    main()
