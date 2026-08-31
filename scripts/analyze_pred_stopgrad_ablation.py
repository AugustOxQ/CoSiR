"""
Paired analysis for Experiment 11.3 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md,
Experiment 11.3 subsection): does removing the stop-gradient on the condition-predictor
distillation term change test_oracle or test_pre_diff, relative to Experiment 11.1's existing
trained/frozen arms?

Reads wandb runs from the 'condition freeze ablation' group (shared with
scripts/run_condition_freeze_ablation.sh), pairs the new 'pred_coupled' arm against BOTH
11.1's 'trained' and 'frozen' arms WITHIN each seed, and reports mean delta +/- std and
mean/SEM (spec S5; delta = pred_coupled - baseline, so positive means pred_coupled wins) for
test_oracle/{t2i,i2t}_R1 and test_pre_diff/{t2i,i2t}_R1. Also prints each arm's
train_buddy_diag/drift_from_init and final-step train_loss/loss_pred as free diagnostic
context (does coupling shrink drift toward frozen's ~0, and does the predictor's own
reconstruction loss converge or diverge).

Usage
-----
  python scripts/analyze_pred_stopgrad_ablation.py
  python scripts/analyze_pred_stopgrad_ablation.py --selftest   # offline check, no wandb call

Requires: wandb, pandas, numpy (all already deps).
"""
import argparse

import numpy as np
import pandas as pd

T2I_ORACLE = "test_oracle/t2i_R1"
I2T_ORACLE = "test_oracle/i2t_R1"
T2I_PREDIFF = "test_pre_diff/t2i_R1"
I2T_PREDIFF = "test_pre_diff/i2t_R1"
DRIFT = "train_buddy_diag/drift_from_init"
PRED_LOSS = "train_loss/loss_pred"
METRICS = [T2I_ORACLE, I2T_ORACLE, T2I_PREDIFF, I2T_PREDIFF]
TREATMENT = "pred_coupled"
BASELINES = ["trained", "frozen"]


def cget(cfg, path, default=None):
    d = cfg
    for p in path:
        if d is None:
            return default
        try:
            d = d.get(p) if hasattr(d, "get") else getattr(d, p, None)
        except Exception:
            return default
    return default if d is None else d


def sget(summ, key, default=np.nan):
    try:
        v = summ.get(key, default)
    except Exception:
        v = getattr(summ, key, default)
    return default if v is None else v


def row_passes_filters(arm, tags, epochs, *, treatment_tag, baseline_tag, expected_epochs):
    """Pure run-selection predicate, factored out of fetch() so it's selftestable without a
    live wandb call.

    Two independent filters, both required to pass:
    - Tag filter: TREATMENT ('pred_coupled') rows must carry `treatment_tag`; BASELINE
      ('trained'/'frozen') rows must carry `baseline_tag`. These are two DIFFERENT tags
      (11.3's own sweep tag vs. 11.1's pre-existing sweep tag) because both arms' runs share
      one wandb group but were launched by different scripts at different times.
    - Epochs filter (defense in depth, applies to ALL arms regardless of tag): `epochs` must
      equal `expected_epochs` exactly. This catches contamination (e.g. a leftover smoke run
      whose tag matching has some other subtle bug) even when the tag filter alone would have
      let it through.

    Returns True iff the run should be included in the analysis.
    """
    tags = tags or []
    if arm == TREATMENT:
        if treatment_tag and treatment_tag not in tags:
            return False
    elif arm in BASELINES:
        if baseline_tag and baseline_tag not in tags:
            return False
    else:
        return False
    if expected_epochs is not None and epochs != expected_epochs:
        return False
    return True


def warn_duplicate_cells(df):
    """Loud warning -- not a silent .max() -- whenever more than one row remains for a given
    (arm, seed) cell after filtering. Per the reviewer's principle, a loud failure beats a
    plausible-looking wrong number: compute_paired_deltas still aggregates duplicate cells via
    .max() today, but this warning must fire so a human notices and investigates, rather than
    the script silently producing a number either way (this is exactly how the original
    smoke-run contamination bug went unnoticed)."""
    if df.empty:
        return
    for (arm, seed), cell in df.groupby(["arm", "seed"]):
        if len(cell) > 1:
            ids = ", ".join(cell["run_id"])
            print(f"  !! WARNING: {len(cell)} rows survived filtering for arm={arm} seed={seed} "
                  f"(run ids: {ids}) -- compute_paired_deltas will .max() through these silently. "
                  f"Investigate before trusting this cell.")


def fetch(entity, project, group, tag=None, baseline_tag=None, expected_epochs=100):
    import wandb
    api = wandb.Api()
    rows = []
    skipped_unfinished = 0
    skipped_filtered = 0
    for run in api.runs(f"{entity}/{project}", filters={"group": group}):
        # A crashed/killed/still-running run can still have partial summary metrics
        # (e.g. logged at epoch 0 before dying) that look like real numbers to
        # compute_paired_deltas's per-cell .max(). Excluding anything but a clean
        # finish is the only safe way to dedupe a cell that got re-run after a
        # crash — .max() alone can silently prefer the crashed run's number over
        # the real, fully-trained one (this happened: a driver-outage-killed run
        # logged epoch-0 metrics that were numerically higher, by epoch-0 noise,
        # than the finished run's converged ones).
        if run.state != "finished":
            skipped_unfinished += 1
            continue
        cfg, summ = run.config, run.summary
        arm = cget(cfg, ("train", "arm"))
        if not arm:
            continue
        epochs = cget(cfg, ("train", "epochs"))
        if not row_passes_filters(arm, run.tags, epochs, treatment_tag=tag,
                                   baseline_tag=baseline_tag, expected_epochs=expected_epochs):
            skipped_filtered += 1
            continue
        row = {"run_id": run.id, "state": run.state, "arm": arm, "seed": cget(cfg, ("seed",))}
        for metric in METRICS + [DRIFT, PRED_LOSS]:
            v = sget(summ, metric)
            row[metric] = float(v) if not np.isnan(v) else np.nan
        rows.append(row)
    if skipped_unfinished:
        print(f"  ({skipped_unfinished} non-finished run(s) excluded from analysis)")
    if skipped_filtered:
        print(f"  ({skipped_filtered} run(s) excluded by tag/epochs filter -- see --tag/"
              f"--baseline-tag/--expected-epochs)")
    df = pd.DataFrame(rows)
    warn_duplicate_cells(df)
    return df


def compute_paired_deltas(df, metric, baseline):
    """Pair TREATMENT ('pred_coupled') vs the given baseline arm within each seed. Returns
    list of (seed, delta) where delta = pred_coupled - baseline. Pure function - no wandb,
    no I/O."""
    deltas = []
    for seed, cell in df.groupby("seed"):
        by_arm = cell.groupby("arm")[metric].max()
        if baseline not in by_arm.index or TREATMENT not in by_arm.index:
            continue
        b, t = by_arm[baseline], by_arm[TREATMENT]
        if np.isnan(b) or np.isnan(t):
            continue
        deltas.append((seed, t - b))
    return deltas


def summarize(deltas):
    """mean, std, sem, z=mean/sem, win-rate - matches the project's existing convention."""
    n = len(deltas)
    if n == 0:
        return {"n": 0}
    arr = np.asarray([d for _, d in deltas], dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1) if n > 1 else float("nan")
    sem = std / np.sqrt(n) if n > 1 else float("nan")
    z = mean / sem if (n > 1 and sem > 0) else float("nan")
    wins = int((arr > 0).sum())
    return {"n": n, "mean": mean, "std": std, "sem": sem, "z": z, "wins": wins}


def analyze(entity, project, group, tag=None, baseline_tag=None, expected_epochs=100):
    print(f"\n{'='*78}\nExperiment 11.3 - bidirectional table<->predictor coupling  group='{group}'"
          + (f"  tag='{tag}'" if tag else "") + (f"  baseline_tag='{baseline_tag}'" if baseline_tag else "")
          + f"  expected_epochs={expected_epochs}\n{'='*78}")
    df = fetch(entity, project, group, tag=tag, baseline_tag=baseline_tag, expected_epochs=expected_epochs)
    if df.empty:
        print("  (no runs found - check --entity/--project/--tag)")
        return
    print(f"  {len(df)} run(s); arms present: {sorted(df['arm'].unique())}.")

    for baseline in BASELINES:
        print(f"\n  {'-'*70}\n  pred_coupled vs {baseline}\n  {'-'*70}")
        for metric in METRICS:
            print(f"\n    --- {metric} (pred_coupled - {baseline}) ---")
            deltas = compute_paired_deltas(df, metric, baseline)
            s = summarize(deltas)
            if s["n"] == 0:
                print("      (no paired seeds found)")
                continue
            sig = f"  mean/SEM={s['z']:+.1f}{' *' if not np.isnan(s['z']) and abs(s['z']) >= 2 else ''}" if s["n"] > 1 else ""
            print(f"      mean delta = {s['mean']:+.2f} (n={s['n']}, wins={s['wins']}/{s['n']}){sig}")
            for seed, d in sorted(deltas):
                print(f"        seed {seed}: delta = {d:+.2f}")

    print(f"\n  {'-'*70}\n  diagnostic context (not paired deltas)\n  {'-'*70}")
    for arm in ["frozen", "trained", TREATMENT]:
        drift = df.loc[df["arm"] == arm, DRIFT].dropna()
        pred_loss = df.loc[df["arm"] == arm, PRED_LOSS].dropna()
        drift_str = f"mean={drift.mean():.4f}" if len(drift) else "(none logged)"
        pred_loss_str = f"mean={pred_loss.mean():.4f}" if len(pred_loss) else "(none logged)"
        print(f"    {arm}: drift_from_init {drift_str}; final loss/loss_pred {pred_loss_str}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="augustoxq")
    ap.add_argument("--project", default="cosir_image")
    ap.add_argument("--group", default="condition freeze ablation")
    ap.add_argument("--tag", default="pred-stopgrad-ablation-redcaps_150k",
                     help="only include pred_coupled (treatment) runs carrying this wandb tag "
                          "-- matches scripts/run_pred_stopgrad_ablation.sh's own WANDB_TAG default")
    ap.add_argument("--baseline-tag", default="condition-freeze-ablation-redcaps_150k",
                     help="only include trained/frozen (baseline) runs carrying this wandb tag "
                          "-- 11.1's real tag, from scripts/run_condition_freeze_ablation.sh; "
                          "NOT the same tag as --tag, since the two arms were launched by "
                          "different scripts sharing one wandb group")
    ap.add_argument("--expected-epochs", type=int, default=100,
                     help="defense-in-depth filter, applied to ALL arms: exclude any run whose "
                          "cfg.train.epochs doesn't match this (catches leftover smoke runs "
                          "even if tag-matching has some other subtle issue)")
    ap.add_argument("--selftest", action="store_true", help="offline arithmetic check, no wandb call")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    analyze(args.entity, args.project, args.group, tag=args.tag, baseline_tag=args.baseline_tag,
            expected_epochs=args.expected_epochs)


def _selftest():
    """Offline arithmetic check - no wandb call."""
    df = pd.DataFrame([
        {"arm": "trained", "seed": 1, T2I_ORACLE: 50.0},
        {"arm": "frozen", "seed": 1, T2I_ORACLE: 54.0},
        {"arm": "pred_coupled", "seed": 1, T2I_ORACLE: 52.0},
        {"arm": "trained", "seed": 2, T2I_ORACLE: 48.0},
        {"arm": "frozen", "seed": 2, T2I_ORACLE: 52.5},
        {"arm": "pred_coupled", "seed": 2, T2I_ORACLE: 51.0},
    ])
    deltas_vs_trained = compute_paired_deltas(df, T2I_ORACLE, "trained")
    assert len(deltas_vs_trained) == 2, f"expected 2 paired cells, got {len(deltas_vs_trained)}"
    got = sorted(round(d, 2) for _, d in deltas_vs_trained)
    assert got == [2.0, 3.0], f"expected deltas [2.0, 3.0] vs trained, got {got}"

    deltas_vs_frozen = compute_paired_deltas(df, T2I_ORACLE, "frozen")
    assert len(deltas_vs_frozen) == 2, f"expected 2 paired cells, got {len(deltas_vs_frozen)}"
    got2 = sorted(round(d, 2) for _, d in deltas_vs_frozen)
    assert got2 == [-2.0, -1.5], f"expected deltas [-2.0, -1.5] vs frozen, got {got2}"

    s = summarize(deltas_vs_trained)
    assert s["n"] == 2
    assert abs(s["mean"] - 2.5) < 1e-9
    assert s["wins"] == 2

    # --- row-selection / filtering check (this experiment's history has TWO real bugs in
    # exactly this area: the wandb-key-prefix bug, and the smoke-run tag-filter-not-applied-
    # to-baseline-arms bug this fix addresses). A decoy row mimics a leftover smoke run: same
    # arm/seed as a real row, but wrong tag and wrong epochs, with a numerically HIGHER metric
    # value than the real row -- exactly the shape that let smoke runs win compute_paired_
    # deltas's per-cell .max() undetected. ---
    real_tag = "pred-stopgrad-ablation-redcaps_150k"
    base_tag = "condition-freeze-ablation-redcaps_150k"
    smoke_base_tag = "condition-freeze-ablation-redcaps_150k-smoke"
    smoke_treat_tag = "pred-stopgrad-ablation-redcaps_150k-smoke"

    # row_passes_filters itself: real rows pass, decoys (wrong tag, or right tag/wrong epochs) don't.
    assert row_passes_filters("trained", [base_tag], 100, treatment_tag=real_tag,
                               baseline_tag=base_tag, expected_epochs=100)
    assert not row_passes_filters("trained", [smoke_base_tag], 2, treatment_tag=real_tag,
                                   baseline_tag=base_tag, expected_epochs=100), \
        "decoy baseline row (smoke tag) should be excluded"
    assert not row_passes_filters("trained", [base_tag], 2, treatment_tag=real_tag,
                                   baseline_tag=base_tag, expected_epochs=100), \
        "decoy baseline row (right tag, wrong epochs) should be excluded by the epochs filter"
    assert row_passes_filters("pred_coupled", [real_tag], 100, treatment_tag=real_tag,
                               baseline_tag=base_tag, expected_epochs=100)
    assert not row_passes_filters("pred_coupled", [smoke_treat_tag], 2, treatment_tag=real_tag,
                                   baseline_tag=base_tag, expected_epochs=100), \
        "decoy treatment row (smoke tag) should be excluded"

    # End-to-end: a decoy smoke row with an INFLATED metric must not leak into
    # compute_paired_deltas once fetch()-style filtering is applied upstream of it. Without the
    # filter, .max() would silently prefer the decoy's 99.0 over the real row's 50.0 -- this is
    # exactly the shape of the real bug (a 2-epoch smoke run scored higher than the real
    # 100-epoch seed-1 run on test_oracle/test_pre_diff).
    raw_rows = [
        {"run_id": "real_trained_1", "arm": "trained", "seed": 1, "tags": [base_tag], "epochs": 100, T2I_ORACLE: 50.0},
        {"run_id": "decoy_smoke_trained_1", "arm": "trained", "seed": 1, "tags": [smoke_base_tag], "epochs": 2, T2I_ORACLE: 99.0},
        {"run_id": "real_frozen_1", "arm": "frozen", "seed": 1, "tags": [base_tag], "epochs": 100, T2I_ORACLE: 54.0},
        {"run_id": "real_pred_1", "arm": "pred_coupled", "seed": 1, "tags": [real_tag], "epochs": 100, T2I_ORACLE: 52.0},
    ]
    filtered_rows = [r for r in raw_rows if row_passes_filters(
        r["arm"], r["tags"], r["epochs"], treatment_tag=real_tag, baseline_tag=base_tag, expected_epochs=100)]
    assert len(filtered_rows) == 3, f"expected the decoy excluded (3 rows left), got {[r['run_id'] for r in filtered_rows]}"
    assert "decoy_smoke_trained_1" not in {r["run_id"] for r in filtered_rows}

    fdf = pd.DataFrame(filtered_rows)
    decoy_free_deltas = compute_paired_deltas(fdf, T2I_ORACLE, "trained")
    assert len(decoy_free_deltas) == 1
    assert abs(decoy_free_deltas[0][1] - 2.0) < 1e-9, \
        f"decoy leaked into the paired delta: expected +2.0 (52-50), got {decoy_free_deltas[0][1]:+.2f}"

    # Sanity: prove the decoy WOULD have corrupted the result if filtering were skipped (i.e.
    # this test actually exercises the bug, not a vacuous check).
    unfiltered_df = pd.DataFrame(raw_rows)
    contaminated_deltas = compute_paired_deltas(unfiltered_df, T2I_ORACLE, "trained")
    assert abs(contaminated_deltas[0][1] - (-47.0)) < 1e-9, \
        "expected the unfiltered decoy to corrupt the delta to 52-99=-47, confirming this test " \
        "exercises the real bug shape"

    print("SELFTEST OK")


if __name__ == "__main__":
    main()
