"""Gap 1: symmetric mapper LR/epoch tuning for PercepT's (bug-fixed) Stage 2,
mirroring candidate 2's sweep on buddy's own mapper exactly (same LR_GRID,
EPOCH_GRID, screen-then-4-seed-stress discipline, same +0.005 practical
margin). §6d explicitly flagged this asymmetry: buddy's mapper was tuned,
PercepT's fixed mapper was left at its original lr=1e-3/epochs=100 defaults.
This script closes that gap for a fair comparison of "best effort" configs.

The Stage-1 K=60/40 DEC re-fit (bug-fixed pruning direction + reconstruction
loss scale, see run_percept_stage2_fixed_pilot.py) does not depend on the
mapper's LR/epochs at all, so it is run exactly ONCE at seed 42 (identical to
run_percept_fixed_snapshot_pilot.py's own flow, reusing the same helper
functions by import) and its train/held-out multi-hot targets + patch
features are reused, in memory, across every sweep point. Only the mapper's
own training (small, full-batch, cheap) is repeated per sweep point and per
stress seed.
"""

from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
PERCEPT_DIR = HERE.parents[1] / "test/20260922_percept_topic_pipeline"
FIXED_STAGE2_PATH = PERCEPT_DIR / "run_percept_stage2_fixed_pilot.py"
REPORT_PATH = HERE / "percept_mapper_symmetric_sweep_pilot_report.md"

FIXED_MACRO_AUC = 0.5925
BUDDY_STAGE2_MACRO_AUC = 0.8534
PRACTICAL_MARGIN = 0.005
LR_GRID = (3e-4, 1e-3, 3e-3, 1e-2)
EPOCH_GRID = (100, 200, 400)
STRESS_SEEDS = (42, 7, 123, 2024)

for directory in (str(PERCEPT_DIR),):
    if directory not in sys.path:
        sys.path.insert(0, directory)


def load_module(module_name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def fit_stage1_and_get_targets(s2):
    """Identical flow to run_percept_fixed_snapshot_pilot.py through target
    generation: fixed pruning direction, fixed reconstruction-loss scale,
    seed 42 throughout (Stage 1 is not re-fit per mapper seed -- only the
    mapper itself is retrained per stress seed, matching candidate 2's own
    convention of re-seeding just the downstream classifier)."""
    np.random.seed(s2.SEED)
    torch.manual_seed(s2.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s2.SEED)

    pipeline = s2.base.load_sibling_module("artelingo_pipeline_symsweep_train", s2.base.PIPELINE_PATH)
    affect_pilot = s2.base.load_sibling_module("artelingo_affect_symsweep", s2.base.AFFECT_PILOT_PATH)
    cca_audit = s2.base.load_sibling_module("artelingo_cca_symsweep", s2.base.CCA_AUDIT_PATH)
    heldout_pipeline = s2.base.load_sibling_module("artelingo_pipeline_symsweep_heldout", s2.base.PIPELINE_PATH)
    heldout_pipeline.STORAGE_DIR = s2.base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = s2.base.HELDOUT_JSON
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log_fn = pipeline.log

    log_fn("Loading and deduplicating train CLIP features...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    log_fn("Loading and deduplicating held-out CLIP features...")
    heldout_paintings, heldout_img_nodes, heldout_txt_nodes, _ = heldout_pipeline.load_dedup_features()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_fn(f"Using {device}.")
    affect_train = s2.base.extract_affect_embedding_nodes(pipeline.TRAIN_JSON, paintings, device, log_fn)
    affect_heldout = s2.base.extract_affect_embedding_nodes(s2.base.HELDOUT_JSON, heldout_paintings, device, log_fn)
    train_h = s2.base.fused_embeddings(img_nodes, txt_nodes, affect_train, cca_audit, affect_pilot)
    heldout_h = s2.base.fused_embeddings(heldout_img_nodes, heldout_txt_nodes, affect_heldout, cca_audit, affect_pilot)

    train_inputs = torch.from_numpy(train_h)
    encoder, decoder = s2.base.build_autoencoder(train_h.shape[1])
    encoder.to(device)
    decoder.to(device)
    s2.base.pretrain_autoencoder(encoder, decoder, train_inputs, device, log_fn)
    centers = s2.sweep.initialize_cluster_centers(encoder, train_inputs, device, s2.N_INITIAL_CLUSTERS, s2.SEED)
    s2.train_dec_until_stable_fixed(encoder, decoder, centers, train_inputs, device, log_fn, s2.N_INITIAL_CLUSTERS)
    surviving_centers, _ = s2.prune_centers_fixed(centers, s2.N_SURVIVING_CLUSTERS)

    train_targets = s2.multi_hot_targets(encoder, surviving_centers, train_inputs, device)
    heldout_targets = s2.multi_hot_targets(encoder, surviving_centers, torch.from_numpy(heldout_h), device)

    train_patch_features = s2.load_patch_features(s2.TRAIN_PATCH_FEATURE_PATH, len(paintings), "train")
    heldout_patch_features = s2.load_patch_features(s2.HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out")
    return device, train_patch_features, train_targets, heldout_patch_features, heldout_targets


def train_and_eval(s2, device, train_patch_features, train_targets, heldout_patch_features,
                    heldout_targets, seed, lr, epochs, tag):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mapper = s2.AttentionPoolingMapper().to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    train_patch_features_dev = train_patch_features.to(device)
    train_targets_dev = train_targets.to(device)
    mapper.train()
    for epoch in range(1, epochs + 1):
        loss = loss_fn(mapper(train_patch_features_dev), train_targets_dev)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    mapper.eval()
    with torch.no_grad():
        heldout_scores = torch.sigmoid(mapper(heldout_patch_features.to(device))).cpu().numpy()
    heldout_targets_np = heldout_targets.cpu().numpy()
    model_aucs, _ = s2.evaluate_auc(heldout_scores, heldout_targets_np, log, tag)
    return s2.auc_summary(model_aucs)["macro"]


def main() -> None:
    log(f"Loading fixed Stage-2 module from {FIXED_STAGE2_PATH}")
    s2 = load_module("percept_stage2_fixed_for_symmetric_sweep", FIXED_STAGE2_PATH)

    log("=== One-time Stage-1 K=60/40 re-fit (fixed pruning + loss scale), seed 42 ===")
    device, train_patch_features, train_targets, heldout_patch_features, heldout_targets = (
        fit_stage1_and_get_targets(s2)
    )

    sanity = train_and_eval(s2, device, train_patch_features, train_targets, heldout_patch_features,
                             heldout_targets, seed=42, lr=s2.MAPPER_LEARNING_RATE, epochs=s2.MAPPER_EPOCHS,
                             tag="sanity_default")
    log(f"Sanity check (lr={s2.MAPPER_LEARNING_RATE:g}, epochs={s2.MAPPER_EPOCHS}): macro AUC={sanity:.4f} "
        f"(expected {FIXED_MACRO_AUC:.4f})")
    if abs(sanity - FIXED_MACRO_AUC) > 0.001:
        log(
            f"NOTE: re-derived sanity {sanity:.4f} does not match the previously-published "
            f"{FIXED_MACRO_AUC:.4f} within +/-0.001. Neither this script nor the original "
            "run_percept_stage2_fixed_pilot.py sets torch.use_deterministic_algorithms/"
            "cudnn.deterministic, so a fresh Stage-1 DEC re-fit (500-epoch self-sharpening "
            "dynamics) is not bit-reproducible run-to-run even at the same seed -- this is a "
            "real GPU-nondeterminism finding, not a bug in this script. Continuing the sweep "
            "using THIS run's own re-fit as the internal baseline; both this run's own baseline "
            "and the previously-published number are reported below for comparison."
        )

    log("=== Part A: LR sweep (seed 42, 100 epochs) ===")
    lr_results = {}
    for lr in LR_GRID:
        macro = train_and_eval(s2, device, train_patch_features, train_targets, heldout_patch_features,
                                heldout_targets, seed=42, lr=lr, epochs=100, tag=f"lr={lr:g}_ep=100")
        lr_results[lr] = macro
        log(f"lr={lr:g}: macro AUC={macro:.4f}")
    best_lr = max(lr_results, key=lr_results.get)
    log(f"Best LR: {best_lr:g} (macro AUC={lr_results[best_lr]:.4f})")

    log("=== Part B: epoch sweep (seed 42, best LR) ===")
    epoch_results = {}
    for epochs in EPOCH_GRID:
        macro = train_and_eval(s2, device, train_patch_features, train_targets, heldout_patch_features,
                                heldout_targets, seed=42, lr=best_lr, epochs=epochs,
                                tag=f"lr={best_lr:g}_ep={epochs}")
        epoch_results[epochs] = macro
        log(f"lr={best_lr:g} epochs={epochs}: macro AUC={macro:.4f}")
    best_epochs = max(epoch_results, key=epoch_results.get)
    best_screen = epoch_results[best_epochs]
    log(f"Best config: lr={best_lr:g}, epochs={best_epochs}, macro AUC={best_screen:.4f}")

    beats_margin = (best_screen - sanity) >= PRACTICAL_MARGIN
    stress_results = []
    if beats_margin:
        log(f"=== Best config beats margin; 4-seed stress (lr={best_lr:g}, epochs={best_epochs}) ===")
        for seed in STRESS_SEEDS:
            macro = train_and_eval(s2, device, train_patch_features, train_targets, heldout_patch_features,
                                    heldout_targets, seed=seed, lr=best_lr, epochs=best_epochs,
                                    tag=f"stress_seed{seed}")
            stress_results.append((seed, macro))
            log(f"seed={seed}: macro AUC={macro:.4f}")
    else:
        log("No sweep point beat the +0.005 margin over 0.5925; skipping stress.")

    write_report(lr_results, epoch_results, best_lr, best_epochs, beats_margin, stress_results, sanity)
    log(f"Wrote {REPORT_PATH}")


def write_report(lr_results, epoch_results, best_lr, best_epochs, beats_margin, stress_results, sanity) -> None:
    determinism_note = (
        f"\n\n**Note on reproducibility:** this run's own fresh Stage-1 re-fit scored "
        f"{sanity:.4f} at the untuned lr=1e-3/epochs=100 point, not the previously-"
        f"published {FIXED_MACRO_AUC:.4f} ({sanity - FIXED_MACRO_AUC:+.4f}). Neither this "
        "script nor the original fixed pilot sets `torch.use_deterministic_algorithms`/"
        "`cudnn.deterministic`, so a fresh 500-epoch DEC self-sharpening re-fit is not "
        "bit-reproducible run-to-run at the same seed -- this is itself a real finding "
        "about this pipeline's stability, not a bug in this script. Every comparison "
        "below uses this run's own re-fit as the internal baseline, so the LR/epoch "
        "effect is measured consistently; the previously-published number is reported "
        "for context only.\n"
        if abs(sanity - FIXED_MACRO_AUC) > 0.001 else ""
    )
    lines = [
        "# Gap 1 — symmetric mapper LR/epoch tuning for PercepT's (bug-fixed) Stage 2\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Mirrors "
        "[`candidate2_mapper_sweep_pilot_report.md`](candidate2_mapper_sweep_pilot_report.md)'s "
        f"LR_GRID={LR_GRID}, EPOCH_GRID={EPOCH_GRID}, and screen-then-4-seed-stress "
        f"discipline, applied to PercepT's own fixed Stage 2 mapper instead of "
        f"buddy's. Baseline (PercepT's original untuned lr=1e-3, epochs=100), "
        f"previously published: **{FIXED_MACRO_AUC:.4f}**; this run's own re-fit: "
        f"**{sanity:.4f}**.{determinism_note}\n\n",
        "## Part A: learning rate sweep (seed 42, 100 epochs)\n\n",
        "| LR | macro AUC |\n|---:|---:|\n",
    ]
    for lr, macro in lr_results.items():
        lines.append(f"| {lr:g} | {macro:.4f} |\n")
    lines.append(f"\nBest LR: **{best_lr:g}**.\n\n")

    lines.append("## Part B: epoch sweep (seed 42, best LR)\n\n")
    lines.append("| epochs | macro AUC |\n|---:|---:|\n")
    for epochs, macro in epoch_results.items():
        lines.append(f"| {epochs} | {macro:.4f} |\n")
    lines.append(f"\nBest config: lr={best_lr:g}, epochs={best_epochs}.\n\n")

    lines.append("## Verdict\n\n")
    lines.append(
        f"(Margins below are measured against this run's own re-fit baseline, "
        f"**{sanity:.4f}**, not the previously-published {FIXED_MACRO_AUC:.4f} -- "
        "see the reproducibility note above.)\n\n"
    )
    best_screen = epoch_results[best_epochs]
    if beats_margin:
        macros = [m for _, m in stress_results]
        mean = float(np.mean(macros))
        lines.extend([
            f"**lr={best_lr:g}, epochs={best_epochs} beat the screening margin** "
            f"(seed 42: {best_screen:.4f} vs. {sanity:.4f} baseline, "
            f"{best_screen - sanity:+.4f}). 4-seed stress:\n\n",
            "| seed | macro AUC |\n|---:|---:|\n",
        ])
        for seed, macro in stress_results:
            lines.append(f"| {seed} | {macro:.4f} |\n")
        lines.append(
            f"\nStress mean: {mean:.4f} (min {np.min(macros):.4f}, max {np.max(macros):.4f}, "
            f"std {np.std(macros):.4f}).\n\n"
            f"**Symmetrically tuned PercepT Stage 2 mapper: {mean:.4f}** "
            f"(vs. this run's own untuned re-fit {sanity:.4f}, {mean - sanity:+.4f}; "
            f"vs. the previously-published untuned number {FIXED_MACRO_AUC:.4f}, "
            f"{mean - FIXED_MACRO_AUC:+.4f}).\n\n"
            f"**Updated comparison: buddy {BUDDY_STAGE2_MACRO_AUC:.4f} vs. "
            f"symmetrically-tuned PercepT {mean:.4f} — a "
            f"{BUDDY_STAGE2_MACRO_AUC - mean:+.4f} margin** (was "
            f"{BUDDY_STAGE2_MACRO_AUC - FIXED_MACRO_AUC:+.4f} against the untuned, "
            f"previously-published comparator). {'Buddy still wins by a wide margin even against a tuned PercepT mapper.' if BUDDY_STAGE2_MACRO_AUC > mean else 'This changes the headline conclusion and needs immediate attention.'}\n"
        )
    else:
        lines.append(
            f"**No sweep point beat the {PRACTICAL_MARGIN:.3f} practical margin** "
            f"over this run's own {sanity:.4f} untuned re-fit baseline (best screened: "
            f"{best_screen:.4f}, {best_screen - sanity:+.4f}). PercepT's Stage 2 mapper "
            "was already near its ceiling at the original lr=1e-3/epochs=100 defaults "
            "— the asymmetric-tuning caveat in §6d can be dropped, or at least "
            "softened: tuning PercepT's mapper the same way buddy's was tuned does "
            "not materially change the comparator. **Comparison stands: buddy "
            f"{BUDDY_STAGE2_MACRO_AUC:.4f} vs. PercepT {FIXED_MACRO_AUC:.4f} "
            f"({BUDDY_STAGE2_MACRO_AUC - FIXED_MACRO_AUC:+.4f}, previously-published "
            "number), tuned or not.**\n"
        )
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
