"""r29 ticket 15: write `records/crash_<arm>.json` after D21's second failure of a training step (called by the stage scripts).

    python r29_crash.py --arm A --seed S --details TEXT [--records DIR]

Keys exactly as the index says: arm, seed, status ("out"), tries (2), details, written (Amsterdam time). Written once: an
existing record is left as it is. Plumbing only; reads no result.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--details', required=True)
    ap.add_argument('--records', type=Path, default=HERE / 'records')
    a = ap.parse_args()
    path = a.records / f'crash_{a.arm}.json'
    if path.exists():
        print(f'{path} exists: left as is')
        return
    doc = dict(arm=a.arm, seed=a.seed, status='out', tries=2, details=a.details,
               written=datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%d %H:%M'))
    a.records.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + '\n')
    import r29_decide as D   # the reader's own check of the record
    assert D.read_crash(a.records, a.arm) == doc
    print('wrote', path)


if __name__ == '__main__':
    main()
