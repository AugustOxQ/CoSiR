"""V1: does the harness's shared Stage 2 path reproduce the pilots' published
Stage 2 numbers when both use the same saved Stage 1 topics?

The path under test is `scripts/buddy_percept_sweep/h2h_eval.stage2_metrics`,
the call the W&B sweeps make. It builds train targets at h2h_eval.py:82-87,
trains one mapper at :89-98 and scores it against every label set at
:99-107. It is called with a lightweight store that holds the real patch
tensors from `h2h_store.load_patches`, and with subset_idx =
arange(9365): the pilots scored on every held-out painting.

Which pilot functions produce the pilot targets (all imported, nothing re-derived)
------------------------------------------------------------------------------------
buddy §6f, 0.8534 (4-seed mean). Flow: run_candidate4_fixed_stress.main, lines
34-61, in src/test/20260927_deep_stage_analysis/.
  Input snapshot: run_candidate1_min_occupancy_pilot.SNAPSHOT (line 41) =
  20260923_artelingo_buddy_analysis/attention_h1_embedding_snapshot.npz. It
  provides train/heldout_embedding_post and train_community_post (K=19).

  Train labels: run_candidate1_min_occupancy_pilot.merge_small_communities
  (lines 71-101) with SMALL_COMMUNITIES = (16, 17, 18) (line 49). It merges
  each of the three smallest communities into the larger community with the
  nearest L2-normalised centroid and compacts the result, giving K=16.

  Train targets (multi-label): two steps, both in
  run_candidate4_rich_multilabel_pilot.
    1. cosine_vote_fractions (lines 31-56). For each train row it takes the
       exact k = TRANSFER_K = 20 cosine nearest neighbours by GPU torch topk
       on L2-normalised float32 embeddings. The query is included among its
       own neighbours. It then takes the merged-label vote fractions.
    2. threshold_targets(fractions, WINNING_CUTOFF = 0.15) (lines 59-61;
       fixed_stress line 25). A topic is positive when its fraction is
       strictly greater than 0.15 times that row's largest fraction.

  Held-out targets (single-label): run_heldout_label_transfer_pilot
  .assign_to_train_communities(train, merged, heldout, k=20). It lives in
  20260923_artelingo_buddy_analysis/, lines 23-68, and is loaded with
  candidate1.load_module.
    - It uses sklearn brute cosine k-NN and takes the majority vote.
    - **A tied vote goes to the label of the nearest train point among the
      tied winners** (lines 58-63).
    - The result is passed through candidate1.one_hot(..., 16)
      (lines 65-68; fixed_stress lines 43-46).

  Mapper: run_candidate4_fixed_pilot.train_multilabel_get_scores (lines 50-86).
    - Seeds with torch.manual_seed(seed) and cuda.manual_seed_all(seed).
    - Model: run_percept_stage2_pilot.AttentionPoolingMapper(n_topics=16)
      (20260922_percept_topic_pipeline/run_percept_stage2_pilot.py:59-75),
      with one query and a linear head.
    - Optimiser: Adam, lr MAPPER_LR = 1e-2, MAPPER_EPOCHS = 400 (lines 46-47).
      Full batch, no weight decay.
    - Loss: unweighted BCEWithLogitsLoss (mean), so **no class balancing**.
    - Output: sigmoid scores for all 9365 held-out paintings.

  Scoring: run_candidate4_fixed_pilot.score_against (lines 89-92), which is
  stage2_ref.evaluate_auc + auc_summary macro (run_percept_stage2_pilot.py
  :128-162).

  Seeds: run_candidate1_stress_pilot.STRESS_SEEDS = (42, 7, 123, 2024)
  (line 28).

  Harness equivalent: Stage2Config(mapper_lr=1e-2, mapper_epochs=400,
  num_queries=1, mlp_head="linear", weight_decay_stage2=0.0,
  class_balanced_loss=False, target_cutoff=0.15, train_target_k=20).
    - Train targets: targets.cosine_vote_fractions + build_targets, both
      called inside stage2_metrics.
    - Held-out labels: h2h_eval.eval_labels, i.e.
      targets.assign_to_train_communities with k = EVAL_TRANSFER_K = 20.
      **It breaks vote ties by the lowest topic index** (votes.argmax).

PercepT fixed snapshot, 0.5925. Flow: run_percept_fixed_snapshot_pilot.main.
  Train topics: train_topic = argmax of base.soft_assignments on the 40
  surviving centres (lines 107-110).

  Train targets: s2.multi_hot_targets(encoder, surviving_centers, train_inputs)
  (line 114; run_percept_stage2_fixed_pilot.py:211-217). This is the argmax
  one-hot plus every topic with q > MULTI_LABEL_THRESHOLD = 2/40 (fixed
  pilot line 60).
    - The snapshot does NOT store the train targets or the centres, so they
      cannot be compared here.
    - percept_stage2_fixed_pilot_report.md records train 0.00% multi-labelled
      for this refit (mean 1.000 labels). The train targets therefore equal
      one_hot(train_topic), which is what target_cutoff="single_label" builds.

  Held-out targets: the same multi_hot_targets on the held-out inputs (line
  115), stored as `heldout_stage2_targets`. This script compares them with
  one_hot(heldout_topic), the single-label set stage2_metrics can express.

  Mapper: s2.AttentionPoolingMapper() with 40 topics and Adam at lr
  s2.MAPPER_LEARNING_RATE = 1e-3 for s2.MAPPER_EPOCHS = 100 (fixed pilot
  lines 61-62). Full-batch BCEWithLogitsLoss mean (snapshot pilot lines
  119-131).
    - Its init draws from the global RNG *after* the whole Stage 1 fit.
    - That RNG is seeded once at lines 56-59 and not reseeded before the
      mapper.
    - So no harness seed reproduces the pilot's mapper init exactly.

  Scoring: s2.evaluate_auc + auc_summary on heldout_stage2_targets (lines
  133-138). The result is stored as `per_topic_auc` and `heldout_stage2_scores`.

Outputs: one `V1_RESULT {json}` stdout line per system, and v1_results.json
next to this file. Pass rule (brief): targets_equal and |delta| <= 0.003.

Usage (local GPU):
    python src/test/20260930_matched_h2h/validate_v1_stage2.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEEP_DIR = REPO / "src/test/20260927_deep_stage_analysis"
PERCEPT_SNAPSHOT = DEEP_DIR / "percept_fixed_snapshot.npz"
RESULTS_PATH = HERE / "v1_results.json"

AUC_TOLERANCE = 0.003
SEEDS = (42, 7, 123, 2024)
BUDDY_REFERENCE = 0.8534            # §6f, candidate4_fixed_stress_report.md (4-seed mean)
BUDDY_REFERENCE_SEED42 = 0.8533     # same report, seed 42 row
PERCEPT_UNTUNED_REFERENCE = 0.5925  # percept_fixed_snapshot.npz per_topic_auc macro
PERCEPT_TUNED_REFERENCE = 0.9226    # §6g, a different Stage 1 refit -- informational only
_FLOAT_SLACK = 1e-9                 # keeps |delta| == tolerance inclusive under binary rounding


# ---------------------------------------------------------------- pure helpers (unit-tested)

def compare_targets(a, b) -> dict:
    """Row-wise exact comparison of two target matrices of the same shape."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"target shapes differ: {a.shape} vs {b.shape}")
    diff_rows = np.any(a.reshape(len(a), -1) != b.reshape(len(b), -1), axis=1)
    n_diff = int(diff_rows.sum())
    return {"equal": n_diff == 0, "n_diff_rows": n_diff}


def v1_verdict(aucs, reference: float, targets_equal: bool, tol: float = AUC_TOLERANCE) -> dict:
    """Mean/population-std of the macro AUCs, delta to the reference, and the
    brief's pass rule: targets_equal and |delta| <= tol."""
    values = np.asarray(aucs, dtype=np.float64)
    mean = float(values.mean())
    delta = mean - float(reference)
    return {"auc_mean": mean, "auc_std": float(values.std()), "reference": float(reference),
            "delta": delta, "pass": bool(targets_equal) and abs(delta) <= tol + _FLOAT_SLACK}


def classify_label_diffs(neighbor_labels, pilot_labels, harness_labels, n_topics: int) -> dict:
    """Explain held-out label disagreements from each query's k neighbour
    labels (nearest first). Two counts cover rows that differ AND have a
    tied vote: the pilot picking the nearest tied winner, and the harness
    picking the lowest-index tied winner."""
    neighbor_labels = np.asarray(neighbor_labels, dtype=np.int64)
    pilot, harness = np.asarray(pilot_labels), np.asarray(harness_labels)
    rows = np.arange(len(neighbor_labels))
    votes = np.zeros((len(neighbor_labels), n_topics), dtype=np.int64)
    np.add.at(votes, (np.repeat(rows, neighbor_labels.shape[1]), neighbor_labels.ravel()), 1)
    winners = votes == votes.max(axis=1, keepdims=True)
    tied = winners.sum(axis=1) > 1
    nearest_winner = neighbor_labels[rows, winners[rows[:, None], neighbor_labels].argmax(axis=1)]
    lowest_winner = votes.argmax(axis=1)
    diff = pilot != harness
    return {
        "n_rows": int(len(neighbor_labels)), "n_tied": int(tied.sum()), "n_diff": int(diff.sum()),
        "n_diff_tied": int((diff & tied).sum()),
        "n_diff_pilot_is_nearest_tied": int((diff & tied & (pilot == nearest_winner)).sum()),
        "n_diff_harness_is_lowest_index": int((diff & tied & (harness == lowest_winner)).sum()),
    }


# ---------------------------------------------------------------- real-data parts (lazy heavy imports)

def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _import_pilot_stress():
    """The §6f stress module. Importing it brings in exactly the candidate
    1/4 helpers it uses. Bytecode writing is disabled so nothing is written
    into the frozen pilot directory."""
    sys.dont_write_bytecode = True
    if str(DEEP_DIR) not in sys.path:
        sys.path.insert(0, str(DEEP_DIR))
    import run_candidate4_fixed_stress as stress
    return stress


def _load_patch_store(n_train: int, n_heldout: int, device: str):
    """An H2HStore-like object carrying only the patch tensors, loaded with
    the loader h2h_store.load_or_build_store uses. The patches are moved to
    `device` once, so train_stage2's .to(device) is a no-op."""
    from types import SimpleNamespace
    from scripts.buddy_percept_sweep.h2h_store import load_patches
    train, heldout = load_patches(n_train, n_heldout)
    return SimpleNamespace(train_patches=train.to(device), heldout_patches=heldout.to(device))


def _train_on_given_targets(s2cfg, store, train_targets, train_labels, subset_idx, label_sets, seed, device):
    """Fallback, used only when the harness's train targets differ from the
    pilot's. It repeats stage2_metrics' mapper section (h2h_eval.py:89-107)
    call for call, but trains on the pilot's own target matrix."""
    import torch
    from scripts.buddy_percept_sweep.h2h_eval import _auc_for, seed_all
    from scripts.buddy_percept_sweep.stage2 import ParameterizedAttentionPoolingMapper, train_stage2
    n_topics = train_targets.shape[1]
    seed_all(seed)
    mapper = ParameterizedAttentionPoolingMapper(
        n_topics=n_topics, num_queries=s2cfg.num_queries, mlp_head=s2cfg.mlp_head,
        d_model=store.train_patches.shape[-1],
    ).to(device)
    train_stage2(mapper, store.train_patches, torch.as_tensor(train_targets, dtype=torch.float32),
                 mapper_lr=s2cfg.mapper_lr, mapper_epochs=s2cfg.mapper_epochs,
                 weight_decay=s2cfg.weight_decay_stage2, class_balanced=s2cfg.class_balanced_loss,
                 train_labels_for_weighting=np.asarray(train_labels, dtype=np.int64), seed=seed)
    mapper.eval()
    with torch.no_grad():
        rows = torch.as_tensor(np.asarray(subset_idx), dtype=torch.long)
        scores = torch.sigmoid(mapper(store.heldout_patches[rows].to(device))).cpu().numpy()
    out = {}
    for name, labels in label_sets.items():
        out[f"auc_{name}"], out[f"skipped_{name}"] = _auc_for(scores, labels, n_topics)
    return out


def buddy_targets(stress, device: str) -> dict:
    """The pilot's train/held-out targets (fixed_stress lines 34-52, verbatim
    calls) and the harness's, from the same snapshot and merged labels."""
    from sklearn.neighbors import NearestNeighbors
    from scripts.buddy_percept_sweep.h2h_eval import eval_labels
    from scripts.buddy_percept_sweep.targets import build_targets, cosine_vote_fractions, one_hot

    with np.load(stress.SNAPSHOT, allow_pickle=True) as source:
        train_emb = source["train_embedding_post"]
        hard_train = source["train_community_post"].astype(np.int64)
        heldout_emb = source["heldout_embedding_post"]
    merged, label_map = stress.merge_small_communities(train_emb, hard_train, stress.SMALL_COMMUNITIES)
    n_topics = len(set(label_map.values()))
    transfer = stress.load_module("transfer_for_v1", stress.BUDDY_DIR / "run_heldout_label_transfer_pilot.py")
    pilot_heldout_hard = transfer.assign_to_train_communities(
        train_emb, merged, heldout_emb, k=stress.TRANSFER_K).astype(np.int64)
    pilot_heldout = stress.one_hot(pilot_heldout_hard, n_topics).cpu().numpy()
    pilot_fractions = stress.cosine_vote_fractions(train_emb, merged, train_emb, n_topics, device)
    pilot_train = stress.threshold_targets(pilot_fractions, stress.WINNING_CUTOFF)

    subset_idx = np.arange(len(heldout_emb))
    harness_fractions = cosine_vote_fractions(train_emb, merged, train_emb, n_topics, k=20)
    harness_train = build_targets(harness_fractions, n_topics, float(stress.WINNING_CUTOFF))
    harness_heldout_hard = eval_labels(train_emb, merged, heldout_emb, subset_idx)
    harness_heldout = one_hot(harness_heldout_hard, n_topics)

    knn = NearestNeighbors(n_neighbors=stress.TRANSFER_K, metric="cosine", algorithm="brute", n_jobs=1)
    neighbors = knn.fit(train_emb).kneighbors(heldout_emb, return_distance=False)
    return dict(
        train_emb=train_emb, merged=merged, n_topics=n_topics, subset_idx=subset_idx,
        pilot_train=pilot_train, pilot_heldout=pilot_heldout, pilot_heldout_hard=pilot_heldout_hard,
        harness_train=harness_train, harness_heldout=harness_heldout, harness_heldout_hard=harness_heldout_hard,
        n_train_fraction_rows_differ=int((np.abs(pilot_fractions - harness_fractions).max(axis=1) > 1e-6).sum()),
        heldout_diff=classify_label_diffs(merged[neighbors], pilot_heldout_hard, harness_heldout_hard, n_topics),
    )


def run_buddy(stress, store, device: str, seeds) -> dict:
    from scripts.buddy_percept_sweep.h2h_eval import stage2_metrics
    from scripts.buddy_percept_sweep.h2h_trial import Stage2Config

    t = buddy_targets(stress, device)
    cmp_train = compare_targets(t["harness_train"], t["pilot_train"])
    cmp_heldout = compare_targets(t["harness_heldout"], t["pilot_heldout"])
    _log(f"buddy targets: train {cmp_train}, held-out {cmp_heldout}, held-out diffs {t['heldout_diff']}")
    s2cfg = Stage2Config(mapper_lr=1e-2, mapper_epochs=400, num_queries=1, mlp_head="linear",
                         weight_decay_stage2=0.0, class_balanced_loss=False,
                         target_cutoff=stress.WINNING_CUTOFF, train_target_k=20)
    label_sets = {"pilot": t["pilot_heldout_hard"], "harness": t["harness_heldout_hard"]}
    if cmp_train["equal"]:
        path = "h2h_eval.stage2_metrics (harness train targets == pilot's)"
    else:
        path = "stage2.ParameterizedAttentionPoolingMapper + stage2.train_stage2 + h2h_eval._auc_for on pilot train targets"
    per_seed = []
    for seed in seeds:
        start = time.monotonic()
        if cmp_train["equal"]:
            out = stage2_metrics(s2cfg, store, t["train_emb"], t["merged"], t["subset_idx"], label_sets, seed, device)
        else:
            out = _train_on_given_targets(s2cfg, store, t["pilot_train"], t["merged"], t["subset_idx"],
                                          label_sets, seed, device)
        row = {"seed": seed, "auc_pilot_heldout": out["auc_pilot"], "auc_harness_heldout": out["auc_harness"],
               "skipped_pilot": out["skipped_pilot"], "seconds": round(time.monotonic() - start, 1)}
        per_seed.append(row)
        _log(f"buddy seed {seed}: {row}")

    targets_equal = cmp_train["equal"] and cmp_heldout["equal"]
    result = {"system": "buddy", "path": path, "targets_equal": targets_equal,
              "n_diff_rows": cmp_train["n_diff_rows"] + cmp_heldout["n_diff_rows"],
              "targets_train": cmp_train, "targets_heldout": cmp_heldout,
              "n_train_fraction_rows_differ": t["n_train_fraction_rows_differ"],
              "heldout_diff_diagnosis": t["heldout_diff"], "n_topics": t["n_topics"]}
    result.update(v1_verdict([r["auc_pilot_heldout"] for r in per_seed], BUDDY_REFERENCE, targets_equal))
    harness_eval = v1_verdict([r["auc_harness_heldout"] for r in per_seed], BUDDY_REFERENCE, True)
    result["mapper_pass"] = bool(abs(result["delta"]) <= AUC_TOLERANCE + _FLOAT_SLACK)
    result["harness_heldout_auc_mean"] = harness_eval["auc_mean"]
    result["harness_heldout_delta"] = harness_eval["delta"]
    result["per_seed"] = per_seed
    result["pilot_loop_seed42"] = pilot_loop_seed42(stress, store, t, device)
    harness_42 = next((r["auc_pilot_heldout"] for r in per_seed if r["seed"] == 42), None)
    if harness_42 is not None:
        result["pilot_loop_seed42"]["harness_minus_pilot_loop"] = harness_42 - result["pilot_loop_seed42"]["auc"]
    return result


def pilot_loop_seed42(stress, store, t, device: str) -> dict:
    """The pilot's OWN training loop and scorer at seed 42, on this machine.
    This separates harness-code differences (AdamW(wd=0) vs Adam, per-row
    loss mean) from hardware/software drift since the published run."""
    stage2_ref = stress.load_module("stage2_ref_for_v1", stress.PERCEPT_DIR / "run_percept_stage2_pilot.py")
    start = time.monotonic()
    scores = stress.train_multilabel_get_scores(stage2_ref, t["pilot_train"], store.train_patches,
                                                store.heldout_patches, device, 42, "v1_pilot_loop_seed42")
    result = stress.score_against(stage2_ref, scores, t["pilot_heldout"], "v1_pilot_loop_seed42")
    out = {"auc": result["macro"], "published_seed42": BUDDY_REFERENCE_SEED42,
           "seconds": round(time.monotonic() - start, 1)}
    _log(f"buddy pilot loop seed 42: {out}")
    return out


def run_percept(store, device: str, seeds) -> dict:
    from scripts.buddy_percept_sweep.h2h_eval import _auc_for, stage2_metrics
    from scripts.buddy_percept_sweep.h2h_trial import Stage2Config
    from scripts.buddy_percept_sweep.targets import one_hot

    with np.load(PERCEPT_SNAPSHOT, allow_pickle=True) as snap:
        train_emb = snap["train_embedding"]
        train_topic = snap["train_topic"].astype(np.int64)
        heldout_topic = snap["heldout_topic"].astype(np.int64)
        stored_targets = snap["heldout_stage2_targets"]
        stored_scores = snap["heldout_stage2_scores"]
        recorded_macro = float(np.nanmean(snap["per_topic_auc"]))
    n_topics = stored_targets.shape[1]
    subset_idx = np.arange(len(heldout_topic))
    cmp_heldout = compare_targets(one_hot(heldout_topic, n_topics), stored_targets)
    scoring_macro, scoring_skipped = _auc_for(stored_scores, heldout_topic, n_topics)
    scoring_check = {"harness_auc_on_pilot_scores": scoring_macro, "recorded_macro": recorded_macro,
                     "abs_diff": abs(scoring_macro - recorded_macro), "skipped": scoring_skipped}
    _log(f"percept targets: held-out {cmp_heldout}; scoring check {scoring_check}")

    result = {"system": "percept", "targets_equal": cmp_heldout["equal"],
              "n_diff_rows": cmp_heldout["n_diff_rows"], "targets_heldout": cmp_heldout,
              "train_targets_compared": False,
              "train_targets_note": ("snapshot stores neither train multi-hot targets nor centres; "
                                     "percept_stage2_fixed_pilot_report.md records train 0.00% multi-labelled, "
                                     "so the pilot's train targets = one_hot(train_topic) = the harness's "
                                     "single_label targets"),
              "scoring_check": scoring_check, "n_topics": n_topics,
              "held_out_fraction_multi_labelled": float((stored_targets.sum(axis=1) > 1).mean())}
    if not cmp_heldout["equal"]:
        result.update(path="not expressible: held-out targets are multi-hot, stage2_metrics takes label sets",
                      **v1_verdict([np.nan], PERCEPT_UNTUNED_REFERENCE, False))
        return result
    result["path"] = "h2h_eval.stage2_metrics (target_cutoff='single_label', eval set = heldout_topic)"

    def sweep(lr, epochs, run_seeds):
        cfg = Stage2Config(mapper_lr=lr, mapper_epochs=epochs, num_queries=1, mlp_head="linear",
                           weight_decay_stage2=0.0, class_balanced_loss=False, target_cutoff="single_label")
        rows = []
        for seed in run_seeds:
            start = time.monotonic()
            out = stage2_metrics(cfg, store, train_emb, train_topic, subset_idx, {"native": heldout_topic},
                                 seed, device)
            rows.append({"seed": seed, "auc": out["auc_native"], "skipped": out["skipped_native"],
                         "seconds": round(time.monotonic() - start, 1)})
            _log(f"percept lr={lr:g} epochs={epochs} seed {seed}: {rows[-1]}")
        return rows

    untuned = sweep(1e-3, 100, seeds)
    seed42 = [r["auc"] for r in untuned if r["seed"] == 42]
    result.update(v1_verdict(seed42, PERCEPT_UNTUNED_REFERENCE, cmp_heldout["equal"]))
    result["untuned_lr1e-3_ep100"] = {"per_seed": untuned,
                                      **v1_verdict([r["auc"] for r in untuned], PERCEPT_UNTUNED_REFERENCE, True)}
    tuned = sweep(1e-2, 400, seeds)
    result["tuned_lr1e-2_ep400_informational"] = {
        "per_seed": tuned, **v1_verdict([r["auc"] for r in tuned], PERCEPT_TUNED_REFERENCE, True)}
    return result


def _meta(device: str) -> dict:
    import subprocess
    import torch
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"commit": commit, "torch": torch.__version__, "device": device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"), "tolerance": AUC_TOLERANCE}


def main() -> None:
    parser = argparse.ArgumentParser(description="V1: harness Stage 2 vs pilot Stage 2 on saved pilot topics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    stress = _import_pilot_stress()
    with np.load(stress.SNAPSHOT, allow_pickle=True) as buddy_snap, \
            np.load(PERCEPT_SNAPSHOT, allow_pickle=True) as percept_snap:
        for split in ("train_paintings", "heldout_paintings"):
            if not np.array_equal(buddy_snap[split], percept_snap[split]):
                raise RuntimeError(f"{split} order differs between the snapshots; one patch store cannot serve both")
        n_train, n_heldout = len(buddy_snap["train_paintings"]), len(buddy_snap["heldout_paintings"])
    _log(f"loading patches ({n_train} train, {n_heldout} held-out)")
    store = _load_patch_store(n_train, n_heldout, args.device)

    results = {"meta": _meta(args.device)}
    for name, runner in (("buddy", lambda: run_buddy(stress, store, args.device, args.seeds)),
                         ("percept", lambda: run_percept(store, args.device, args.seeds))):
        results[name] = runner()
        print("V1_RESULT " + json.dumps(results[name]), flush=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
