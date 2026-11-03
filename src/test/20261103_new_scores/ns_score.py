"""r17 shared scorer facade (spec §3.7, D13). THE INTERFACE BELOW IS FROZEN when this round merges into main.

r18 and r19 import it read-only. A later need adds a function here and never changes a signature or a return form.

    partitions(x, k, set_name) -> int32 [3, n]                  ns_metrics: MiniBatchKMeans as r10 (n_init 3, states 0 to 2)
    metric_table(arm, aspect, parts, labels, k, metrics) -> rows   ns_clu: per seed and mean, AMI NMI ARI ACC
    paired_cell(cand_parts, comp_parts, labels, k, r, metrics, draws, rng_seed) -> {metric: {cand, comp, d, lo, hi, r,
                outcome, n_draws}}                              ns_clu: joint bootstrap over rows and seeds
    ranges(parts_by_seed, labels, k, metrics) -> {metric: {r, per_seed}}   ns_clu: seed range r
    arm_ap(get_val, get_eval, pick, labels, seeds) -> {ap, point, per_seed, rows}   ns_ret: retrieval AP at a read-out pick
    paired_ret(cand_ap, comp_ap, r, draws, comp_seeds, rng_seed) -> {cand, comp, d, lo, hi, r, outcome, n_draws}   ns_ret
    outcome(lo, r) -> 'WIN' | 'LEVEL' | 'BELOW'                 r11's rt_stats.outcome (the word rule of §3.4)

Signatures are those of the defining modules (read them for the full arguments); this module only names them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths  # noqa: E402

ns_paths.add_rounds()
from ns_metrics import partitions  # noqa: E402,F401
from ns_clu import metric_table, paired_cell, ranges  # noqa: E402,F401
from ns_ret import arm_ap, paired_ret  # noqa: E402,F401
from rt_stats import outcome  # noqa: E402,F401

__all__ = ['partitions', 'metric_table', 'paired_cell', 'ranges', 'arm_ap', 'paired_ret', 'outcome']
