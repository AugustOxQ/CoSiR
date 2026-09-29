"""Is 'subreddit' a noisy/low-power proxy for topic on RedCaps, independent of the
topic-formation method?

Motivating hypothesis (user, 2026-09-27): RedCaps' negative B1/repair result may
reflect a property of the *dataset and metric*, not the method -- subreddits are
not semantically clean topic labels the way ArtELingo's emotion/genre are (lots
of near-duplicate/overlapping subreddits => false negatives), and the random,
non-stratified split may leave many subreddits with too little validation
support for a reliable lift estimate (a "test set" quality issue).

Three independent, CPU-only, read-only diagnostics, all against the already
saved B1 split (identical train/val partition, no new randomness introduced):

  A. Same- vs different-subreddit raw-CLIP similarity separability, on a
     validation sample -- estimates how many high-similarity pairs are being
     penalized as "negative" purely because they land in different subreddits
     (a direct false-negative-rate proxy for the lift metric's own label noise).
  B. Subreddit-centroid redundancy -- do distinct subreddits sit close together
     in raw CLIP space (candidate near-duplicate communities), which would mean
     "different subreddit" structurally under-counts true topic overlap?
  C. Split/support coverage -- how much validation support does each subreddit
     actually have, and how concentrated is it (a thin, long-tailed test set
     would make the lift ratio fragile regardless of model quality)?

No training, no modification to any shared/frozen file.
"""

from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ART_DIR = REPO_ROOT / "src/test/20260923_artelingo_buddy_analysis"
REDCAPS_DIR = REPO_ROOT / "src/test/20260623_redcaps_buddy"
for directory in (REPO_ROOT, ART_DIR, REDCAPS_DIR, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from redcaps_buddy import load_data
from run_b1_redcaps_single_teacher_pilot import SPLIT_PATH as B1_SPLIT_PATH, raw_concat, restrict_data


SEED = 42
SIMILARITY_SAMPLE_SIZE = 3000
MIN_SUBREDDIT_SIZE_FOR_CENTROID = 50
TOP_REDUNDANT_PAIRS = 20
REPORT_PATH = HERE / "run_subreddit_proxy_diagnostic_report.md"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def analysis_a(validation, rng) -> dict:
    """Same- vs different-subreddit raw similarity separability."""
    n = validation.n
    sample_size = min(SIMILARITY_SAMPLE_SIZE, n)
    idx = rng.choice(n, size=sample_size, replace=False)
    feats = raw_concat(validation)[idx]
    sub = validation.sub_id[idx]

    sim = feats @ feats.T
    iu, ju = np.triu_indices(sample_size, k=1)
    pair_sim = sim[iu, ju]
    pair_same = sub[iu] == sub[ju]

    same_sim = pair_sim[pair_same]
    diff_sim = pair_sim[~pair_same]
    same_median = float(np.median(same_sim)) if len(same_sim) else float("nan")
    same_p90 = float(np.percentile(same_sim, 90)) if len(same_sim) else float("nan")

    # Fraction of *different*-subreddit pairs that are "at least as similar as a
    # typical true positive" -- a direct false-negative-rate estimate for the lift
    # metric's own label, independent of any model.
    fnr_at_median = float((diff_sim >= same_median).mean()) if len(diff_sim) else float("nan")
    fnr_at_p90 = float((diff_sim >= same_p90).mean()) if len(diff_sim) else float("nan")

    # Top-1 nearest neighbor per point (excluding self): is it cross-subreddit
    # despite being at least as similar as a typical same-subreddit pair?
    np.fill_diagonal(sim, -np.inf)
    nn_idx = sim.argmax(axis=1)
    nn_sim = sim[np.arange(sample_size), nn_idx]
    nn_same = sub == sub[nn_idx]
    nn_cross_but_high_sim = float(((~nn_same) & (nn_sim >= same_median)).mean())

    return {
        "sample_size": sample_size,
        "n_same_pairs": int(pair_same.sum()),
        "n_diff_pairs": int((~pair_same).sum()),
        "same_sim_mean": float(same_sim.mean()) if len(same_sim) else float("nan"),
        "same_sim_median": same_median,
        "diff_sim_mean": float(diff_sim.mean()) if len(diff_sim) else float("nan"),
        "diff_sim_median": float(np.median(diff_sim)) if len(diff_sim) else float("nan"),
        "fnr_at_same_median": fnr_at_median,
        "fnr_at_same_p90": fnr_at_p90,
        "nn_cross_subreddit_frac": float((~nn_same).mean()),
        "nn_cross_but_high_sim_frac": nn_cross_but_high_sim,
    }


def analysis_b(train, validation) -> dict:
    """Subreddit-centroid redundancy across distinct subreddits."""
    n_sub = len(train.sub_names)
    all_img = np.concatenate([train.img, validation.img], axis=0)
    all_txt = np.concatenate([train.txt, validation.txt], axis=0)
    all_sub = np.concatenate([train.sub_id, validation.sub_id], axis=0)
    joined = np.concatenate([all_img, all_txt], axis=1)
    norms = np.linalg.norm(joined, axis=1, keepdims=True)
    joined = joined / np.maximum(norms, 1e-12)

    counts = np.bincount(all_sub, minlength=n_sub)
    qualifying = np.flatnonzero(counts >= MIN_SUBREDDIT_SIZE_FOR_CENTROID)
    centroids = np.zeros((len(qualifying), joined.shape[1]), dtype=np.float32)
    for row, sub_idx in enumerate(qualifying):
        members = all_sub == sub_idx
        centroid = joined[members].mean(axis=0)
        centroids[row] = centroid / max(np.linalg.norm(centroid), 1e-12)

    sim = centroids @ centroids.T
    np.fill_diagonal(sim, -np.inf)
    iu, ju = np.triu_indices(len(qualifying), k=1)
    pair_sim = sim[iu, ju]
    order = np.argsort(-pair_sim)[:TOP_REDUNDANT_PAIRS]
    top_pairs = []
    for rank in order:
        a, b = qualifying[iu[rank]], qualifying[ju[rank]]
        top_pairs.append({
            "name_a": train.sub_names[a], "name_b": train.sub_names[b],
            "similarity": float(pair_sim[rank]),
            "count_a": int(counts[a]), "count_b": int(counts[b]),
        })

    return {
        "n_qualifying_subreddits": int(len(qualifying)),
        "min_size_threshold": MIN_SUBREDDIT_SIZE_FOR_CENTROID,
        "centroid_sim_mean": float(pair_sim.mean()),
        "centroid_sim_p99": float(np.percentile(pair_sim, 99)),
        "frac_pairs_above_0_8": float((pair_sim >= 0.8).mean()),
        "frac_pairs_above_0_9": float((pair_sim >= 0.9).mean()),
        "top_pairs": top_pairs,
    }


def analysis_c(train, validation) -> dict:
    """Split/support coverage for the lift metric's own subreddit labels."""
    n_sub = len(train.sub_names)
    train_counts = np.bincount(train.sub_id, minlength=n_sub)
    val_counts = np.bincount(validation.sub_id, minlength=n_sub)
    present_in_train = train_counts > 0
    present_in_val = val_counts > 0
    missing_from_val = int((present_in_train & ~present_in_val).sum())
    n_present_train = int(present_in_train.sum())

    val_nonzero = val_counts[val_counts > 0]
    order = np.argsort(-val_counts)
    top10_share = float(val_counts[order[:10]].sum() / max(val_counts.sum(), 1))
    top20_share = float(val_counts[order[:20]].sum() / max(val_counts.sum(), 1))
    thin_frac = float((val_nonzero < 5).mean()) if len(val_nonzero) else float("nan")

    return {
        "n_subreddits_total": n_sub,
        "n_present_in_train": n_present_train,
        "n_missing_from_val": missing_from_val,
        "missing_from_val_frac_of_train_present": missing_from_val / max(n_present_train, 1),
        "val_count_min": int(val_nonzero.min()) if len(val_nonzero) else 0,
        "val_count_median": float(np.median(val_nonzero)) if len(val_nonzero) else float("nan"),
        "val_count_max": int(val_nonzero.max()) if len(val_nonzero) else 0,
        "top10_subreddits_val_share": top10_share,
        "top20_subreddits_val_share": top20_share,
        "frac_val_subreddits_under_5": thin_frac,
    }


def write_report(a: dict, b: dict, c: dict) -> None:
    lines = [
        "# Is 'subreddit' a noisy proxy for topic on RedCaps?\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; seed {SEED}; reuses B1's saved split ",
        f"(`{B1_SPLIT_PATH.name}`). Companion to ",
        "[`docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md`]",
        "(../../../docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md).\n\n",
        "Motivating hypothesis: the negative B1/repair result may partly reflect ",
        "the subreddit label and the split, not the topic-formation method -- ",
        "subreddits may be a noisier, more redundant, and more thinly-supported ",
        "proxy for topic than ArtELingo's emotion/genre labels.\n\n",

        "## A. Same- vs different-subreddit raw similarity separability\n\n",
        f"Sample: {a['sample_size']:,} validation points ({a['n_same_pairs']:,} same-subreddit ",
        f"pairs, {a['n_diff_pairs']:,} different-subreddit pairs). Raw cosine similarity ",
        "(unit-normalized, concatenated image+text CLIP).\n\n",
        f"- Same-subreddit similarity: mean {a['same_sim_mean']:.4f}, median {a['same_sim_median']:.4f}\n",
        f"- Different-subreddit similarity: mean {a['diff_sim_mean']:.4f}, median {a['diff_sim_median']:.4f}\n",
        f"- **False-negative-rate estimate**: {a['fnr_at_same_median']:.1%} of different-subreddit ",
        "pairs are at least as similar as the *median* same-subreddit pair ",
        f"(at the same-subreddit 90th percentile instead: {a['fnr_at_same_p90']:.1%}).\n",
        f"- Nearest-neighbor check: {a['nn_cross_subreddit_frac']:.1%} of validation points' single ",
        "nearest neighbor (by raw similarity) is in a *different* subreddit; ",
        f"{a['nn_cross_but_high_sim_frac']:.1%} of all points have a cross-subreddit nearest ",
        "neighbor that is *at least as similar as a typical same-subreddit pair* -- these are the ",
        "clearest false-negative candidates under the subreddit-lift metric.\n\n",

        "## B. Subreddit-centroid redundancy\n\n",
        f"{b['n_qualifying_subreddits']} subreddits with >= {b['min_size_threshold']} members ",
        "(train+validation combined); centroid = mean unit-normalized raw feature, ",
        "re-normalized.\n\n",
        f"- Inter-subreddit centroid similarity: mean {b['centroid_sim_mean']:.4f}, ",
        f"99th percentile {b['centroid_sim_p99']:.4f}\n",
        f"- {b['frac_pairs_above_0_8']:.2%} of distinct-subreddit pairs have centroid similarity ",
        f">= 0.8; {b['frac_pairs_above_0_9']:.2%} >= 0.9\n\n",
        f"Top {len(b['top_pairs'])} most similar *distinct* subreddit pairs (candidate redundant/",
        "overlapping communities):\n\n",
        "| Subreddit A | Subreddit B | Centroid similarity | Size A | Size B |\n",
        "|---|---|---:|---:|---:|\n",
    ]
    for pair in b["top_pairs"]:
        lines.append(
            f"| {pair['name_a']} | {pair['name_b']} | {pair['similarity']:.4f} | "
            f"{pair['count_a']:,} | {pair['count_b']:,} |\n"
        )

    lines.extend([
        "\n## C. Split/support coverage\n\n",
        f"{c['n_subreddits_total']} total subreddits; {c['n_present_in_train']} present in train. ",
        f"**{c['n_missing_from_val']} of those ",
        f"({c['missing_from_val_frac_of_train_present']:.1%}) have zero validation examples** ",
        "under the random, non-subreddit-stratified split.\n\n",
        f"- Validation count per subreddit (subreddits with >= 1 val example): ",
        f"min {c['val_count_min']}, median {c['val_count_median']:.1f}, max {c['val_count_max']:,}\n",
        f"- The top 10 subreddits by validation count hold {c['top10_subreddits_val_share']:.1%} of ",
        f"all validation examples; the top 20 hold {c['top20_subreddits_val_share']:.1%}\n",
        f"- {c['frac_val_subreddits_under_5']:.1%} of subreddits with any validation presence at ",
        "all have fewer than 5 validation examples\n\n",

        "## Reading these together\n\n",
    ])

    verdict_signals = []
    if a["fnr_at_same_median"] > 0.05:
        verdict_signals.append(
            f"(A) supports the false-negative hypothesis: {a['fnr_at_same_median']:.1%} of "
            "different-subreddit pairs look as similar as a typical true match."
        )
    else:
        verdict_signals.append(
            f"(A) does not show strong false-negative contamination: only "
            f"{a['fnr_at_same_median']:.1%} of different-subreddit pairs reach typical "
            "same-subreddit similarity."
        )
    if b["frac_pairs_above_0_9"] > 0.01:
        verdict_signals.append(
            f"(B) supports the redundancy hypothesis: {b['frac_pairs_above_0_9']:.2%} of distinct-"
            "subreddit pairs have near-duplicate centroids -- check the top-pairs table for whether "
            "these look like genuinely overlapping communities."
        )
    else:
        verdict_signals.append(
            f"(B) does not show much subreddit redundancy: only {b['frac_pairs_above_0_9']:.2%} of "
            "distinct-subreddit pairs have near-duplicate centroids."
        )
    if c["missing_from_val_frac_of_train_present"] > 0.2 or c["top20_subreddits_val_share"] > 0.5:
        verdict_signals.append(
            "(C) supports the thin-test-set hypothesis: a large fraction of subreddits are either "
            "absent from validation or the validation set is dominated by a handful of large "
            "subreddits, making the lift ratio fragile and concentrated on a few communities."
        )
    else:
        verdict_signals.append(
            "(C) does not show a strongly thin or concentrated test set by these two measures."
        )
    lines.extend(f"- {signal}\n" for signal in verdict_signals)
    lines.append(
        "\nThis diagnostic bears on B1's metric ceiling, not on the trained students' training "
        "procedure itself -- it does not retract B1's finding that neither student beats raw CLIP "
        "features, but it does inform how much of that gap (and how much of the residual occupancy "
        "collapse from B1/repair) should be attributed to subreddit being a noisy, unevenly "
        "supported proxy label versus a genuine method shortfall.\n"
    )
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    split = np.load(B1_SPLIT_PATH)
    train_idx, val_idx = split["train_idx"], split["val_idx"]
    log(f"Loaded B1 split: train={len(train_idx):,}, val={len(val_idx):,}.")

    data = load_data()
    train = restrict_data(data, train_idx)
    validation = restrict_data(data, val_idx)
    del data

    rng = np.random.default_rng(SEED)
    log("Analysis A: same- vs different-subreddit raw similarity separability.")
    a = analysis_a(validation, rng)
    log(f"A: fnr_at_same_median={a['fnr_at_same_median']:.1%} "
        f"nn_cross_but_high_sim_frac={a['nn_cross_but_high_sim_frac']:.1%}")

    log("Analysis B: subreddit-centroid redundancy.")
    b = analysis_b(train, validation)
    log(f"B: {b['n_qualifying_subreddits']} qualifying subreddits, "
        f"{b['frac_pairs_above_0_9']:.2%} pairs >= 0.9 similarity")

    log("Analysis C: split/support coverage.")
    c = analysis_c(train, validation)
    log(f"C: {c['n_missing_from_val']} subreddits missing from val "
        f"({c['missing_from_val_frac_of_train_present']:.1%} of train-present); "
        f"top20 share={c['top20_subreddits_val_share']:.1%}")

    write_report(a, b, c)
    log(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
