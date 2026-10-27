"""Names for the run chat's queues (ticket 15); plumbing, decides nothing. Prints one answer per line.

  final --outputs O [--smoke]     F's run name ('G' or c*): rs_ablation.resolve_final, the one rule of the ablation scorer
  arms  --outputs O [--smoke] [--final-name NAME]
                                  '<removal> <arm run name>' for the eight removals of F (rs_ablation.arm_names); --final-name
                                  names F without reading the stage files (the dry runs use G)
  stage1-knobs                    '<knob> <name> <name> <name>' for the 13 knobs of the sheet (the 39 stage 1 runs)
  shards N [NAME ...]             NAME... dealt to N shards, 'i name name ...' per shard, each name to the shard with the
                                  fewest runs so far (lowest index on ties). With no NAME: G, G-repeat (both first, to shard 0),
                                  then the 39 stage 1 runs
"""
import argparse
from pathlib import Path

import rs_paths
import rs_sheet


def deal(names, n):
    shards = [[] for _ in range(n)]
    for name in names:
        shards[min(range(n), key=lambda i: (len(shards[i]), i))].append(name)
    return shards


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    for cmd in ('final', 'arms'):
        p = sub.add_parser(cmd)
        p.add_argument('--outputs', type=Path, default=rs_paths.OUTPUTS)
        p.add_argument('--smoke', action='store_true')
        if cmd == 'arms':
            p.add_argument('--final-name', help='arms of this F (G, or a run name) without reading any stage file (dry runs)')
    sub.add_parser('stage1-knobs')
    p = sub.add_parser('shards')
    p.add_argument('n', type=int)
    p.add_argument('names', nargs='*')
    args = parser.parse_args(argv)
    if args.cmd in ('final', 'arms'):
        import rs_ablation
        outputs = rs_paths.check_outputs(args.outputs, args.smoke)
        if args.cmd == 'arms' and args.final_name:
            final = dict(config=rs_sheet.parse(args.final_name))
        else:
            final = rs_ablation.resolve_final(outputs, args.smoke)
        if args.cmd == 'final':
            print(final['name'])
        else:
            for removal, name in rs_ablation.arm_names(final).items():
                print(removal, name)
    elif args.cmd == 'stage1-knobs':
        by = {}
        for cfg in rs_sheet.stage1():
            by.setdefault(rs_sheet.changed(cfg)[0], []).append(rs_sheet.name(cfg))
        assert len(by) == 13 and sum(map(len, by.values())) == 39
        for knob in rs_sheet.KNOBS:
            print(knob, *by[knob])
    else:
        names = args.names or ['G', 'G-repeat'] + [rs_sheet.name(c) for c in rs_sheet.stage1()]
        if args.names:
            shards = deal(names, args.n)
        else:
            shards = deal_with_first(names, args.n)
        for i, shard in enumerate(shards):
            print(i, *shard)


def deal_with_first(names, n):
    """G and G-repeat first on shard 0, the rest to the shard with the fewest runs (G and G-repeat counted)."""
    shards = [[] for _ in range(n)]
    shards[0] = list(names[:2])
    for name in names[2:]:
        shards[min(range(n), key=lambda i: (len(shards[i]), i))].append(name)
    return shards


if __name__ == '__main__':
    main()
