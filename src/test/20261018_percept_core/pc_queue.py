"""Queue timing output and output relocation for the fixed-path rows CLI."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def write_times(status, outdir, queue):
    outdir = Path(outdir)
    steps = {}
    for line in Path(status).read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].startswith('seconds='):
            name, code = fields[0].split('=')
            steps[name] = dict(exit_code=int(code) if code.isdigit() else code,
                               seconds=float(fields[1].split('=')[1]))
    with (outdir / 'smoke_times.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = outdir / 'smoke_times.json'
        result = json.loads(path.read_text()) if path.exists() else {}
        result[queue] = dict(smoke=True, steps=steps)
        pending = path.with_suffix('.pending.json')
        pending.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        pending.replace(path)


def main():
    if sys.argv[1] == 'rows':
        # pc_rows.main fixes base at HERE/outputs; relocate only its write root.
        import pc_rows
        pc_rows.HERE = Path(sys.argv[2]).resolve() / 'rows'
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        pc_rows.main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['times'])
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--queue', choices=['gpu', 'cpu'], required=True)
    args = parser.parse_args()
    write_times(args.status, args.outdir, args.queue)


if __name__ == '__main__':
    main()
