"""Probe run (ticket 27): CLIP, seed 42, one dataset. Exploration: no verdict, no bar.

    python run_r9_probe.py --dataset {cub,semart} --stage {affect,groups,graph,a3,reader,recipe,episodes,score}
                           [--limit N]

Environment: CUDA_VISIBLE_DEVICES= (CPU stages) or under `flock -n -o -E 75 /tmp/gpu0.lock` (affect, graph, a3),
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1. Outputs: results/probes/<dataset>/.
Stage `recipe` = groups, graph, a3, reader in that order (GPU for graph and a3).
Exit 3: a step cannot run on this corpus; the reason is in <stage>_failed.json.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_probe_recipe as P  # noqa: E402

STAGES = {"affect": P.stage_affect, "groups": P.stage_groups, "graph": P.stage_graph, "a3": P.stage_a3,
          "reader": P.stage_reader}
RECIPE = ("groups", "graph", "a3", "reader")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("cub", "semart"), required=True)
    ap.add_argument("--stage", choices=(*STAGES, "recipe", "episodes", "score"), required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke: this many fit units, outputs in <dataset>_limit<N>")
    args = ap.parse_args(argv)
    out = P.out_dir(args.dataset, args.limit)
    rows = P.load_rows(args.dataset, args.limit)
    P.log(f"{args.dataset}: {rows.counts}")
    P.check_split(rows, out)
    for stage in (RECIPE if args.stage == "recipe" else (args.stage,)):
        if stage not in STAGES:
            raise SystemExit(f"stage {stage} is not built yet (part 2)")
        code = STAGES[stage](rows, out)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
