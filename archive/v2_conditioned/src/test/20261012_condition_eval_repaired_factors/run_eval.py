"""Condition interface on the repaired factors: held-out evaluation (plan Task 7).

Plan: docs/superpowers/plans/2026-10-08-cosir-v2-candidate-a-factor-repair.md, Task 7. Run from the
repository root with the CoSiR environment:

    /root/miniconda3/envs/CoSiR/bin/python src/test/20261012_condition_eval_repaired_factors/run_eval.py

Inputs are Task 6's two checkpoints only (no factor training here):
  R0 = src/test/20261011_factor_repair_grid/checkpoints/R0_seed42.pt        (collapsed recipe)
  R3 = src/test/20261011_factor_repair_grid/checkpoints/selected_seed42.pt  (selected, repaired)

Step 1 (primary, paired): human-label episodes. Validation episodes are rebuilt exactly as Task 6
built them (2,048 emotion + 2,048 art style from val, seed 42) and checked against Task 6's stored
metadata and stored val lifts. Held episodes (1,024 + 1,024 from held, seed 42) are built ONCE and
used unchanged for both models. Variants naive / uniform / CLIP-only; beta per variant chosen on the
val episodes (both label types pooled) by mean bidirectional R@1 (ties -> smaller beta).

Step 2 (secondary, unpaired): factor-mined episodes, each model mining from its own codes (4,096 on
val, 1,024 on held; split-local mining under single-threaded BLAS, then remapped to global rows, as
in Task 7/8/9 of the condition-interface plan). Variants naive / naive top-k (k chosen on val) /
uniform / oracle / CLIP-only; beta chosen on val; conditional_score with a score_pool parity check.

``--smoke`` runs every code path with val rows standing in for held rows and fewer episodes, writes
only under cache/smoke/ and never encodes held rows into any reported number (its numbers are
discarded). ``--tables`` reprints the markdown tables from results/eval_results.json.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASK6 = HERE.parent / "20261011_factor_repair_grid"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASK6))
sys.path.insert(0, str(HERE.parent / "20261005_condition_ranking_evaluation"))
sys.path.insert(0, str(HERE.parent / "20261007_naive_rule_mechanism_analysis"))

from run_grid import LABEL_SETS, build_val_episodes, encode_rows, load_art_styles  # noqa: E402
from run_ranking_eval import choose_swap_pairs, episode_arrays, score_pool, validate_roles  # noqa: E402
from run_mechanism import bootstrap_difference, role_outrank, swap_reversal  # noqa: E402
from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.splits import grouped_split, leakage_groups, split_leakage  # noqa: E402
from src.eval.label_episodes import condition_lift, label_episode_recall, tie_aware_rank  # noqa: E402
from src.model.conditioning import conditional_score, naive_condition_weights, pair_codes  # noqa: E402
from src.train.episodes import EpisodeMiningConfig, mine_episodes  # noqa: E402
from src.train.train_factors import load_factor_checkpoint  # noqa: E402

SEED = 42
EXPECTED_SPLIT = (216_107, 30_872, 61_744)
BETA_GRID = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
TOPK = (1, 3, 5)
DIRECTIONS = ("i2t", "t2i")
BOOTSTRAP_RESAMPLES = 5000
PARITY_EPISODES = 16
PARITY_TOLERANCE = 1e-5
TIME_LIMIT_SECONDS = 45 * 60
CHECKPOINTS = {"R0": TASK6 / "checkpoints" / "R0_seed42.pt",
               "R3": TASK6 / "checkpoints" / "selected_seed42.pt"}
EXPECTED_CONFIG = {"R0": {"agreement": "cosine", "lambda_decorrelation": 0.0},
                   "R3": {"agreement": "infonce", "lambda_decorrelation": 1.0}}
TASK6_RESULTS = TASK6 / "results"
LABEL_VARIANTS = ("naive", "uniform", "clip_only")
MINED_VARIANTS = ("naive", *(f"top_{k}" for k in TOPK), "uniform", "oracle", "clip_only")
SIZES = {"real": dict(val_label=2048, held_label=1024, val_mined=4096, held_mined=1024),
         "smoke": dict(val_label=2048, held_label=256, val_mined=512, held_mined=256)}


def _t(values) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float32)


def pick_beta(utilities: dict) -> float:
    """Highest mean bidirectional R@1; ties go to the smaller beta (grid order)."""
    best = max(utilities.values())
    return next(b for b in BETA_GRID if utilities[b] >= best - 1e-12)


def successes(ranks: np.ndarray) -> np.ndarray:
    return np.column_stack((ranks <= 1, ranks <= 3))


def boot(a_ranks: np.ndarray, b_ranks: np.ndarray) -> dict:
    """Paired episode bootstrap (5,000 resamples, seed 42); percentile 95% CI for R@1 and R@3."""
    out = bootstrap_difference(successes(a_ranks), successes(b_ranks), seed=SEED, resamples=BOOTSTRAP_RESAMPLES)
    return {"r1": out["point"][0], "r1_ci": out["ci95"][0], "r3": out["point"][1], "r3_ci": out["ci95"][1]}


def label_sha(episodes) -> str:
    digest = hashlib.sha256()
    for field in ("anchor", "positive", "supports", "contrasts", "distractors"):
        digest.update(np.ascontiguousarray(getattr(episodes, field), dtype=np.int64).tobytes())
    digest.update("\n".join(map(str, episodes.labels.tolist())).encode())
    return digest.hexdigest()


def mined_sha(episodes: list) -> str:
    return hashlib.sha256(json.dumps([vars(ep) for ep in episodes], sort_keys=True).encode()).hexdigest()


# ----------------------------------------------------------------------------- setup

def setup(smoke: bool):
    data = load_artelingo()
    groups = leakage_groups(data.paintings, data.img_features)
    split = grouped_split(groups, fractions=(0.7, 0.1, 0.2), seed=SEED)
    split_sizes = (len(split.train), len(split.val), len(split.held))
    if split_sizes != EXPECTED_SPLIT:
        raise AssertionError(f"Unexpected split sizes {split_sizes}")
    leak = split_leakage(split, data.paintings, data.img_features)
    if any(leak.values()):
        raise AssertionError(f"Split leakage: {leak}")
    styles = load_art_styles(data.sample_ids, data.paintings, groups)
    held_rows = split.val if smoke else split.held          # smoke: val rows stand in for held
    print(f"Split {split_sizes}, leakage zero; {len(np.unique(styles))} art styles; "
          f"held part = {'val (smoke stand-in)' if smoke else 'held'}", flush=True)
    return data, groups, split, styles, held_rows, {"split_sizes": split_sizes, "leakage": leak}


def build_label_sets(data, groups, split, styles, held_rows, sizes: dict) -> tuple[dict, dict]:
    """Val episodes exactly as Task 6 (checked against its stored metadata); held episodes once."""
    labels = {"emotion": data.emotions, "art_style": styles}
    val_eps, val_meta = build_val_episodes(labels, groups, split.val, sizes["val_label"])
    held_eps, held_meta = build_val_episodes(labels, groups, held_rows, sizes["held_label"])
    stored = json.loads((TASK6_RESULTS / "summary.json").read_text())["episodes"]
    identical = json.loads(json.dumps(val_meta)) == stored
    if not identical:
        raise AssertionError("Rebuilt val label episodes differ from Task 6's stored episode metadata")
    meta = {"val": {name: {**val_meta[name], "sha256": label_sha(val_eps[name])} for name in LABEL_SETS},
            "held": {name: {**held_meta[name], "sha256": label_sha(held_eps[name])} for name in LABEL_SETS},
            "val_metadata_identical_to_task6": identical}
    for part, eps in (("val", val_eps), ("held", held_eps)):
        for name in LABEL_SETS:
            print(f"{part} {name} label episodes: {len(eps[name].anchor)} over "
                  f"{meta[part][name]['eligible_labels']} targets, sha {meta[part][name]['sha256'][:12]}", flush=True)
    return {"val": val_eps, "held": held_eps}, meta


def load_codes(data) -> tuple[dict, dict]:
    codes, meta = {}, {}
    for name, path in CHECKPOINTS.items():
        model, config = load_factor_checkpoint(path, device="cuda:0" if torch.cuda.is_available() else "cpu")
        for key, value in EXPECTED_CONFIG[name].items():
            if getattr(config, key) != value:
                raise AssertionError(f"{name} checkpoint has {key}={getattr(config, key)!r}, expected {value!r}")
        img_codes, txt_codes = encode_rows(model, data.img_features, data.txt_features)   # all rows, 8,192 batches
        if img_codes.shape != (len(data.img_features), config.num_factors) or txt_codes.shape != img_codes.shape:
            raise AssertionError(f"{name}: unexpected code shape {img_codes.shape}")
        if not (np.isfinite(img_codes).all() and np.isfinite(txt_codes).all()):
            raise AssertionError(f"{name}: non-finite codes")
        codes[name] = (img_codes, txt_codes)
        meta[name] = {"path": str(path.relative_to(ROOT)), "config": vars(config), "all_codes_finite": True,
                      "mean_code_img": float(img_codes.mean()), "mean_code_txt": float(txt_codes.mean()),
                      "active_fraction_img": float((img_codes > 0).mean()),
                      "active_fraction_txt": float((txt_codes > 0).mean())}
        print(f"{name}: loaded {path.name} ({config.agreement}, decorrelation {config.lambda_decorrelation}); "
              f"encoded {len(img_codes):,} rows, all finite", flush=True)
        del model
    return codes, meta


def task6_lift_check(data, codes: dict, val_eps: dict) -> dict:
    """condition_lift on the rebuilt val episodes must reproduce Task 6's stored val lifts."""
    out = {}
    for name, (img_codes, txt_codes) in codes.items():
        stored = json.loads((TASK6_RESULTS / f"{name}_seed42.json").read_text())["lifts"]
        mismatches = {}
        for label_set in LABEL_SETS:
            lift = condition_lift(data.img_features, data.txt_features, img_codes, txt_codes, val_eps[label_set])
            for variant in ("naive", "uniform"):
                for d in DIRECTIONS:
                    for key in ("recall1", "recall3", "tied_episodes"):
                        new, old = lift[variant][d][key], stored[label_set][variant][d][key]
                        if new != old:
                            mismatches[f"{label_set}.{variant}.{d}.{key}"] = {"recomputed": new, "task6": old}
        out[name] = {"identical": not mismatches, "n_compared": len(LABEL_SETS) * 2 * 2 * 3,
                     "mismatches": mismatches}
        print(f"{name}: Task 6 val-lift reproduction identical={not mismatches} {mismatches or ''}", flush=True)
    return out


# ----------------------------------------------------------------------------- Step 1: label episodes

def label_weights(img_codes, txt_codes, episodes) -> dict:
    support = pair_codes(_t(img_codes[episodes.supports]), _t(txt_codes[episodes.supports]))
    contrast = pair_codes(_t(img_codes[episodes.contrasts]), _t(txt_codes[episodes.contrasts]))
    naive = naive_condition_weights(support, contrast)
    return {"naive": naive, "uniform": torch.full_like(naive, 1.0 / naive.shape[-1]),
            "clip_only": torch.zeros_like(naive)}


def label_ranks(data, img_codes, txt_codes, eps: dict, weights: dict, variant: str, beta: float) -> dict:
    """{label_set: {direction: (ranks, tied_episodes)}} via label_episode_recall (conditional_score inside)."""
    out = {}
    for name in LABEL_SETS:
        result = label_episode_recall(data.img_features, data.txt_features, img_codes, txt_codes, eps[name],
                                      weights[name][variant], beta)
        out[name] = {d: (result[d]["ranks"], result[d]["tied_episodes"]) for d in DIRECTIONS}
    return out


def summarize_label(ranked: dict) -> dict:
    out = {}
    for name in (*LABEL_SETS, "pooled"):
        out[name] = {}
        for d in DIRECTIONS:
            if name == "pooled":
                ranks = np.concatenate([ranked[s][d][0] for s in LABEL_SETS])
                tied = sum(ranked[s][d][1] for s in LABEL_SETS)
            else:
                ranks, tied = ranked[name][d]
            out[name][d] = {"r1": float(np.mean(ranks <= 1)), "r3": float(np.mean(ranks <= 3)),
                            "tied_episodes": int(tied), "n": int(len(ranks))}
    return out


def pooled_ranks(ranked: dict, d: str) -> np.ndarray:
    return np.concatenate([ranked[s][d][0] for s in LABEL_SETS])


def evaluate_labels(data, codes: dict, label_eps: dict) -> tuple[dict, dict]:
    results, held_ranked = {}, {}
    for model, (img_codes, txt_codes) in codes.items():
        weights = {part: {name: label_weights(img_codes, txt_codes, label_eps[part][name]) for name in LABEL_SETS}
                   for part in ("val", "held")}
        curves, selected = {}, {}
        for variant in LABEL_VARIANTS:
            curves[variant] = {}
            for beta in BETA_GRID:
                summary = summarize_label(label_ranks(data, img_codes, txt_codes, label_eps["val"],
                                                      weights["val"], variant, beta))
                curves[variant][str(beta)] = summary
            selected[variant] = pick_beta({b: 0.5 * (curves[variant][str(b)]["pooled"]["i2t"]["r1"]
                                                     + curves[variant][str(b)]["pooled"]["t2i"]["r1"])
                                           for b in BETA_GRID})
        held, held_ranked[model] = {}, {}
        for variant in LABEL_VARIANTS:
            held[variant], held_ranked[model][variant] = {}, {}
            for setting, beta in (("beta0", 0.0), ("selected", selected[variant])):
                ranked = label_ranks(data, img_codes, txt_codes, label_eps["held"], weights["held"], variant, beta)
                held[variant][setting] = {"beta": beta, **summarize_label(ranked)}
                held_ranked[model][variant][setting] = ranked
        zero_rows = {part: {name: int((weights[part][name]["naive"].sum(dim=-1) == 0).sum()) for name in LABEL_SETS}
                     for part in ("val", "held")}
        results[model] = {"val_curves": curves, "selected_beta": selected, "held": held,
                          "naive_zero_weight_episodes": zero_rows}
        sel = held["naive"]["selected"]["pooled"]
        print(f"[label {model}] selected beta {selected}; held naive pooled R@1 i2t {sel['i2t']['r1']:.4f} "
              f"t2i {sel['t2i']['r1']:.4f}", flush=True)
    # CLIP-only does not depend on the factor model: both models must give identical ranks.
    for setting in ("beta0", "selected"):
        for s in LABEL_SETS:
            for d in DIRECTIONS:
                if not np.array_equal(held_ranked["R0"]["clip_only"][setting][s][d][0],
                                      held_ranked["R3"]["clip_only"][setting][s][d][0]):
                    raise AssertionError("CLIP-only ranks differ between models")
    return results, held_ranked


def label_bootstraps(held_ranked: dict) -> dict:
    """Paired bootstraps on the identical held label episodes, pooled and per label type."""
    comparisons = {  # name: ((model, variant), (model, variant))
        "a_R3_naive_minus_R0_naive": (("R3", "naive"), ("R0", "naive")),
        "b_R3_naive_minus_R3_uniform": (("R3", "naive"), ("R3", "uniform")),
        "c_R3_naive_minus_clip_only": (("R3", "naive"), ("R3", "clip_only")),
        "context_R0_naive_minus_R0_uniform": (("R0", "naive"), ("R0", "uniform")),
        "context_R0_naive_minus_clip_only": (("R0", "naive"), ("R0", "clip_only")),
        "context_R3_uniform_minus_R0_uniform": (("R3", "uniform"), ("R0", "uniform")),
    }
    out = {}
    for label, ((ma, va), (mb, vb)) in comparisons.items():
        out[label] = {}
        for setting in ("selected", "beta0"):
            ra, rb = held_ranked[ma][va][setting], held_ranked[mb][vb][setting]
            out[label][setting] = {"pooled": {d: boot(pooled_ranks(ra, d), pooled_ranks(rb, d)) for d in DIRECTIONS}}
            for s in LABEL_SETS:
                out[label][setting][s] = {d: boot(ra[s][d][0], rb[s][d][0]) for d in DIRECTIONS}
    return out


# ----------------------------------------------------------------------------- Step 2: factor-mined episodes

def remap(episodes: list, items: np.ndarray) -> None:
    """Split-local -> global rows, exactly as the condition-interface Task 7/8/9 scripts."""
    for ep in episodes:
        ep.anchor_idx = int(items[ep.anchor_idx])
        ep.support_idxs = items[ep.support_idxs].astype(int).tolist()
        ep.contrast_idxs = items[ep.contrast_idxs].astype(int).tolist()
        ep.positive_idx = int(items[ep.positive_idx])
        ep.hard_negative_idxs = items[ep.hard_negative_idxs].astype(int).tolist()
        ep.condition_distractor_idxs = items[ep.condition_distractor_idxs].astype(int).tolist()
        ep.anchor_distractor_idxs = items[ep.anchor_distractor_idxs].astype(int).tolist()


def mine_part(img_codes, txt_codes, rows: np.ndarray, n: int, config: EpisodeMiningConfig) -> list:
    with threadpool_limits(limits=1, user_api="blas"):
        episodes = mine_episodes(img_codes[rows], txt_codes[rows], config, n)
    remap(episodes, rows)
    validate_roles(episodes, rows)
    episode_arrays(episodes)
    return episodes


def group_diagnostic(episodes: list, groups: np.ndarray) -> dict:
    """Protocol unchanged from the row-split tasks: roles are distinct rows, not distinct paintings."""
    repeated = shares_anchor_or_positive = 0
    for ep in episodes:
        roles = [ep.anchor_idx, *ep.support_idxs, *ep.contrast_idxs, ep.positive_idx, *ep.hard_negative_idxs,
                 *ep.condition_distractor_idxs, *ep.anchor_distractor_idxs]
        role_groups = groups[roles]
        repeated += len(set(role_groups.tolist())) < len(roles)
        distractors = [*ep.hard_negative_idxs, *ep.condition_distractor_idxs, *ep.anchor_distractor_idxs]
        shares_anchor_or_positive += bool(np.isin(groups[distractors],
                                                  [groups[ep.anchor_idx], groups[ep.positive_idx]]).any()
                                          or groups[ep.positive_idx] == groups[ep.anchor_idx])
    return {"episodes": len(episodes), "any_repeated_painting_group": int(repeated),
            "candidate_shares_group_with_anchor_or_positive": int(shares_anchor_or_positive)}


def mined_weights(img_codes, txt_codes, episodes: list) -> dict:
    supports, contrasts, _, targets = episode_arrays(episodes)
    support = pair_codes(_t(img_codes[supports]), _t(txt_codes[supports]))
    contrast = pair_codes(_t(img_codes[contrasts]), _t(txt_codes[contrasts]))
    naive = naive_condition_weights(support, contrast)
    out = {"naive": naive}
    for k in TOPK:
        out[f"top_{k}"] = naive_condition_weights(support, contrast, top_k=k)
    out["uniform"] = torch.full_like(naive, 1.0 / naive.shape[-1])
    out["oracle"] = torch.nn.functional.one_hot(torch.as_tensor(targets), naive.shape[-1]).float()
    out["clip_only"] = torch.zeros_like(naive)
    return out


class MinedPart:
    """Per-direction gathered tensors for one set of mined episodes."""

    def __init__(self, data, img_codes, txt_codes, episodes: list):
        _, _, candidates, _ = episode_arrays(episodes)
        anchors = np.asarray([ep.anchor_idx for ep in episodes], dtype=np.int64)
        self.episodes, self.anchors, self.candidates = episodes, anchors, candidates
        self.raw = {"i2t": (data.img_features, data.txt_features, img_codes, txt_codes),
                    "t2i": (data.txt_features, data.img_features, txt_codes, img_codes)}
        self.tensors = {d: (_t(qf[anchors]), _t(cf[candidates]), _t(qc[anchors]), _t(cc[candidates]))
                        for d, (qf, cf, qc, cc) in self.raw.items()}

    def scores(self, d: str, weights: torch.Tensor, beta: float) -> torch.Tensor:
        return conditional_score(*self.tensors[d], weights, beta)

    def parity(self, d: str, weights: torch.Tensor, beta: float) -> float:
        qf, cf, qc, cc = self.raw[d]
        a, c = self.anchors[:PARITY_EPISODES], self.candidates[:PARITY_EPISODES]
        reference, _, _ = score_pool(qf[a], cf[c], qc[a], cc[c], weights[:PARITY_EPISODES].numpy(), beta)
        ours = self.scores(d, weights, beta)[:PARITY_EPISODES].numpy()
        return float(np.abs(ours - reference).max())


def rank_summary(scores: torch.Tensor) -> tuple[dict, np.ndarray]:
    ranks = tie_aware_rank(scores).numpy()
    tied = (scores[:, 1:] == scores[:, :1]).any(dim=1)
    all_tied = (scores[:, 1:] == scores[:, :1]).all(dim=1)
    return ({"r1": float(np.mean(ranks <= 1)), "r3": float(np.mean(ranks <= 3)),
             "tied_episodes": int(tied.sum()), "all_tied_episodes": int(all_tied.sum()), "n": int(len(ranks))},
            ranks)


def swap_counts(part: "MinedPart", weights: torch.Tensor, pairs: list, beta: float) -> dict:
    """Task 7/9 fixed-anchor two-condition reversal, scored with conditional_score."""
    out = {}
    for d in DIRECTIONS:
        qf, cf, qc, cc = part.raw[d]
        count = 0
        for a, b in pairs:
            ep_a, ep_b = part.episodes[a], part.episodes[b]
            pool = list(dict.fromkeys([
                ep_a.positive_idx, ep_b.positive_idx,
                *ep_a.hard_negative_idxs, *ep_a.condition_distractor_idxs, *ep_a.anchor_distractor_idxs,
                *ep_b.hard_negative_idxs, *ep_b.condition_distractor_idxs, *ep_b.anchor_distractor_idxs,
            ]))
            anchor = [ep_a.anchor_idx, ep_a.anchor_idx]
            scores = conditional_score(_t(qf[anchor]), _t(cf[pool])[None].expand(2, -1, -1), _t(qc[anchor]),
                                       _t(cc[pool])[None].expand(2, -1, -1), weights[[a, b]], beta)
            count += int(float(scores[0, 0] - scores[0, 1]) > 1e-8 and float(scores[1, 1] - scores[1, 0]) > 1e-8)
        out[d] = {"count": count, "total": len(pairs), "rate": count / len(pairs)}
    return out


def evaluate_mined(data, groups, split, held_rows, codes: dict, sizes: dict) -> dict:
    config = EpisodeMiningConfig(seed=SEED)
    results = {}
    for model, (img_codes, txt_codes) in codes.items():
        started = perf_counter()
        val_eps = mine_part(img_codes, txt_codes, split.val, sizes["val_mined"], config)
        held_eps = mine_part(img_codes, txt_codes, held_rows, sizes["held_mined"], config)
        mining_seconds = perf_counter() - started
        parts = {"val": MinedPart(data, img_codes, txt_codes, val_eps),
                 "held": MinedPart(data, img_codes, txt_codes, held_eps)}
        weights = {"val": mined_weights(img_codes, txt_codes, val_eps),
                   "held": mined_weights(img_codes, txt_codes, held_eps)}
        # beta per variant on val (mean bidirectional R@1, ties -> smaller beta); k for top-k on val.
        curves, selected = {}, {}
        for variant in MINED_VARIANTS:
            curves[variant] = {str(b): {d: rank_summary(parts["val"].scores(d, weights["val"][variant], b))[0]
                                        for d in DIRECTIONS} for b in BETA_GRID}
            selected[variant] = pick_beta({b: 0.5 * (curves[variant][str(b)]["i2t"]["r1"]
                                                     + curves[variant][str(b)]["t2i"]["r1"]) for b in BETA_GRID})
        topk_utility = {k: 0.5 * sum(curves[f"top_{k}"][str(selected[f"top_{k}"])][d]["r1"] for d in DIRECTIONS)
                        for k in TOPK}
        best = max(topk_utility.values())
        chosen_k = next(k for k in TOPK if topk_utility[k] >= best - 1e-12)
        # Parity with score_pool on the first 16 episodes of each part, every variant, beta 0 and selected.
        parity = max(parts[p].parity(d, weights[p][v], b) for p in ("val", "held") for v in MINED_VARIANTS
                     for b in {0.0, selected[v]} for d in DIRECTIONS)
        if not parity < PARITY_TOLERANCE:
            raise AssertionError(f"{model}: conditional_score vs score_pool parity {parity:.2e}")
        # Held: R@1/R@3 with tie-aware ranks, per-role, bootstraps.
        held, held_ranks, per_role = {}, {}, {}
        naive_beta = selected["naive"]
        for variant in MINED_VARIANTS:
            held[variant], held_ranks[variant], per_role[variant] = {}, {}, {}
            for setting, beta in (("beta0", 0.0), ("selected", selected[variant])):
                held[variant][setting], held_ranks[variant][setting] = {"beta": beta}, {}
                for d in DIRECTIONS:
                    held[variant][setting][d], held_ranks[variant][setting][d] = rank_summary(
                        parts["held"].scores(d, weights["held"][variant], beta))
            for setting, beta in (("beta0", 0.0), ("naive_selected_beta", naive_beta)):
                per_role[variant][setting] = {"beta": beta, **{
                    d: role_outrank(parts["held"].scores(d, weights["held"][variant], beta).numpy())
                    for d in DIRECTIONS}}
        bootstraps = {}
        for label, other in (("naive_minus_uniform", "uniform"), ("naive_minus_clip_only", "clip_only"),
                             (f"top_{chosen_k}_minus_uniform", "uniform")):
            first = "naive" if label.startswith("naive") else f"top_{chosen_k}"
            bootstraps[label] = {setting: {d: boot(held_ranks[first][setting][d], held_ranks[other][setting][d])
                                           for d in DIRECTIONS} for setting in ("selected", "beta0")}
        # Swap reversal on choose_swap_pairs pairs (held), each variant at its selected beta; naive also at 0.
        try:
            pairs = choose_swap_pairs(0.5 * (img_codes + txt_codes), held_rows, held_eps, config)
        except ValueError as error:      # recorded, never silently turned into a rate
            print(f"[mined {model}] choose_swap_pairs: {error}", flush=True)
            pairs = []
        swaps, swap_check = {}, {}
        for variant in (MINED_VARIANTS if pairs else ()):
            swaps[variant] = {"beta": selected[variant],
                              **swap_counts(parts["held"], weights["held"][variant], pairs, selected[variant])}
            reference = swap_reversal(data.img_features, data.txt_features, img_codes, txt_codes, held_eps,
                                      weights["held"][variant].numpy(), pairs, selected[variant])
            swap_check[variant] = {d: {"conditional_score": swaps[variant][d]["count"],
                                       "score_pool": reference[d]["count"]} for d in DIRECTIONS}
        if pairs:
            swaps["naive_beta0"] = {"beta": 0.0, **swap_counts(parts["held"], weights["held"]["naive"], pairs, 0.0)}
        results[model] = {
            "mining_seconds": mining_seconds,
            "episodes": {"val": len(val_eps), "held": len(held_eps), "val_sha256": mined_sha(val_eps),
                         "held_sha256": mined_sha(held_eps),
                         "held_targets": dict(sorted(Counter(ep.targeted_factor for ep in held_eps).items())),
                         "val_distinct_targets": len({ep.targeted_factor for ep in val_eps}),
                         "held_distinct_targets": len({ep.targeted_factor for ep in held_eps}),
                         "group_diagnostic": {"val": group_diagnostic(val_eps, groups),
                                              "held": group_diagnostic(held_eps, groups)}},
            "val_curves": curves, "selected_beta": selected, "topk_val_utility": topk_utility,
            "chosen_k": chosen_k, "parity_max_abs_diff": parity, "held": held, "per_role": per_role,
            "bootstrap": bootstraps, "swap_pairs": len(pairs), "distinct_b_episodes": len({b for _, b in pairs}),
            "swaps": swaps, "swap_scorer_check": swap_check,
            "naive_zero_weight_episodes": {p: int((weights[p]["naive"].sum(dim=-1) == 0).sum()) for p in weights},
        }
        n = held["naive"]["selected"]
        swap_text = (f"{swaps['naive']['i2t']['count']}/{swaps['naive']['t2i']['count']} of {len(pairs)}"
                     if pairs else "no valid pairs")
        print(f"[mined {model}] mining {mining_seconds:.1f}s; beta {selected}; k={chosen_k}; parity {parity:.2e}; "
              f"held naive R@1 {n['i2t']['r1']:.4f}/{n['t2i']['r1']:.4f}; swaps naive {swap_text}", flush=True)
    return results


# ----------------------------------------------------------------------------- criteria

def criteria(label_boot: dict, mined: dict) -> dict:
    def above_zero(entry: dict) -> bool:
        return all(entry[d]["r1_ci"][0] > 0 for d in DIRECTIONS)

    def crit1(setting: str) -> dict:
        a = label_boot["a_R3_naive_minus_R0_naive"][setting]["pooled"]
        b = label_boot["b_R3_naive_minus_R3_uniform"][setting]["pooled"]
        return {"R3_naive_beats_R0_naive": above_zero(a), "R3_naive_beats_R3_uniform": above_zero(b),
                "met": above_zero(a) and above_zero(b)}

    def crit2(key: str) -> dict:
        if not all(mined[m]["swap_pairs"] for m in ("R0", "R3")):
            return {"rates": None, "R3_higher": None, "met": False,
                    "note": "not evaluable: a model has no valid swap pairs"}
        rates = {m: {d: mined[m]["swaps"][key][d]["rate"] for d in DIRECTIONS} for m in ("R0", "R3")}
        by_direction = {d: rates["R3"][d] > rates["R0"][d] for d in DIRECTIONS}
        return {"rates": rates, "R3_higher": by_direction, "met": all(by_direction.values())}

    def crit3(setting: str) -> dict:
        return {"met": above_zero(mined["R3"]["bootstrap"]["naive_minus_uniform"][setting])}

    return {"1_primary": {"selected_beta": crit1("selected"), "beta0_informational": crit1("beta0")},
            "2_secondary": {"selected_beta": crit2("naive"), "beta0_informational": crit2("naive_beta0")},
            "3_floor": {"selected_beta": crit3("selected"), "beta0_informational": crit3("beta0")}}


# ----------------------------------------------------------------------------- tables

def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def ci(entry: dict, key: str = "r1") -> str:
    lo, hi = entry[f"{key}_ci"]
    return f"{100 * entry[key]:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


def clopper_pearson(count: int, total: int) -> str:
    from scipy.stats import beta as beta_dist
    lo = 0.0 if count == 0 else beta_dist.ppf(0.025, count, total - count + 1)
    hi = 1.0 if count == total else beta_dist.ppf(0.975, count + 1, total - count)
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def print_tables(res: dict) -> None:
    lab, mined = res["label"], res["mined"]
    print("\n### Label episodes: selected beta (val, pooled, mean bidirectional R@1)")
    print("| model | naive | uniform | CLIP-only |")
    print("|---|---:|---:|---:|")
    for m in ("R3", "R0"):
        print(f"| {m} | " + " | ".join(str(lab[m]["selected_beta"][v]) for v in LABEL_VARIANTS) + " |")
    print("\n### Label episodes: val curves, pooled R@1 i2t/t2i (%)")
    print("| model | variant | " + " | ".join(str(b) for b in BETA_GRID) + " |")
    print("|---|---|" + "---:|" * len(BETA_GRID))
    for m in ("R3", "R0"):
        for v in LABEL_VARIANTS:
            cells = [f"{pct(lab[m]['val_curves'][v][str(b)]['pooled']['i2t']['r1'])}/"
                     f"{pct(lab[m]['val_curves'][v][str(b)]['pooled']['t2i']['r1'])}" for b in BETA_GRID]
            print(f"| {m} | {v} | " + " | ".join(cells) + " |")
    for setting in ("selected", "beta0"):
        print(f"\n### Held label episodes ({setting}): R@1/R@3 % (tied episodes) — chance 7.7/23.1")
        print("| model | variant | beta | pooled i2t | pooled t2i | emotion i2t | emotion t2i | art i2t | art t2i |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for m in ("R3", "R0"):
            for v in LABEL_VARIANTS:
                h = lab[m]["held"][v][setting]
                cells = [f"{pct(h[s][d]['r1'])}/{pct(h[s][d]['r3'])} ({h[s][d]['tied_episodes']})"
                         for s in ("pooled", *LABEL_SETS) for d in DIRECTIONS]
                print(f"| {m} | {v} | {h['beta']} | " + " | ".join(cells) + " |")
    for setting in ("selected", "beta0"):
        print(f"\n### Held label-episode paired bootstraps ({setting}), R@1 points [95% CI]")
        print("| comparison | pooled i2t | pooled t2i | emotion i2t | emotion t2i | art i2t | art t2i |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for name, entry in res["label_bootstrap"].items():
            e = entry[setting]
            print(f"| {name} | " + " | ".join(ci(e[s][d]) for s in ("pooled", *LABEL_SETS) for d in DIRECTIONS) + " |")
        print("\nR@3 points:")
        for name, entry in res["label_bootstrap"].items():
            e = entry[setting]
            print(f"| {name} | " + " | ".join(ci(e[s][d], 'r3') for s in ("pooled", *LABEL_SETS)
                                              for d in DIRECTIONS) + " |")
    print("\n### Mined episodes: selected beta and k (val)")
    print("| model | " + " | ".join(MINED_VARIANTS) + " | chosen k | top-k val utility | parity |")
    print("|---|" + "---:|" * (len(MINED_VARIANTS) + 3))
    for m in ("R3", "R0"):
        r = mined[m]
        print(f"| {m} | " + " | ".join(str(r["selected_beta"][v]) for v in MINED_VARIANTS)
              + f" | {r['chosen_k']} | " + ", ".join(f"k{k}: {pct(u)}" for k, u in r["topk_val_utility"].items())
              + f" | {r['parity_max_abs_diff']:.1e} |")
    for setting in ("selected", "beta0"):
        print(f"\n### Held mined episodes ({setting}): R@1/R@3 % (tied / all-tied)")
        print("| model | variant | beta | i2t | t2i |")
        print("|---|---|---:|---:|---:|")
        for m in ("R3", "R0"):
            for v in MINED_VARIANTS:
                h = mined[m]["held"][v][setting]
                print(f"| {m} | {v} | {h['beta']} | " + " | ".join(
                    f"{pct(h[d]['r1'])}/{pct(h[d]['r3'])} ({h[d]['tied_episodes']}/{h[d]['all_tied_episodes']})"
                    for d in DIRECTIONS) + " |")
    print("\n### Held mined bootstraps, points [95% CI]: R@1 i2t | R@3 i2t | R@1 t2i | R@3 t2i")
    for m in ("R3", "R0"):
        for name, entry in mined[m]["bootstrap"].items():
            for setting in ("selected", "beta0"):
                e = entry[setting]
                print(f"| {m} | {name} | {setting} | {ci(e['i2t'])} | {ci(e['i2t'], 'r3')} | {ci(e['t2i'])}"
                      f" | {ci(e['t2i'], 'r3')} |")
    print("\n### Swap reversal (held mined; each variant at its selected beta)")
    print("| model | pairs (distinct B) | " + " | ".join(MINED_VARIANTS) + " | naive beta0 |")
    print("|---|---|" + "---:|" * (len(MINED_VARIANTS) + 1))
    for m in ("R3", "R0"):
        s = mined[m]["swaps"]
        if not s:
            print(f"| {m} | 0 valid swap pairs |")
            continue
        print(f"| {m} | {mined[m]['swap_pairs']} ({mined[m]['distinct_b_episodes']}) | " + " | ".join(
            f"{s[v]['i2t']['count']}/{s[v]['t2i']['count']}" for v in (*MINED_VARIANTS, "naive_beta0")) + " |")
        print(f"    scorer check: {mined[m]['swap_scorer_check']}")
        for v in ("naive", "naive_beta0"):   # descriptive only; the criterion compares the rates themselves
            print(f"    {v}: " + "; ".join(
                f"{d} {s[v][d]['count']}/{s[v][d]['total']} = {pct(s[v][d]['rate'])}% "
                f"(Clopper-Pearson 95% {clopper_pearson(s[v][d]['count'], s[v][d]['total'])})" for d in DIRECTIONS))
    print("\n### Per-role outrank (% episodes with >=1 role member strictly above the positive) H/C/A")
    print("| model | variant | beta0 i2t | beta0 t2i | naive-beta i2t | naive-beta t2i |")
    print("|---|---|---:|---:|---:|---:|")
    for m in ("R3", "R0"):
        for v in MINED_VARIANTS:
            pr = mined[m]["per_role"][v]
            cells = ["/".join(pct(pr[s][d][k]) for k in ("hard_negative", "condition_only", "anchor_only"))
                     for s in ("beta0", "naive_selected_beta") for d in DIRECTIONS]
            print(f"| {m} | {v} | " + " | ".join(cells) + f" |  (naive beta {pr['naive_selected_beta']['beta']})")
    print("\n### Mined episode diagnostics")
    for m in ("R3", "R0"):
        print(m, json.dumps({k: v for k, v in mined[m]["episodes"].items() if k != "held_targets"}),
              "zero-weight naive:", mined[m]["naive_zero_weight_episodes"], f"mining {mined[m]['mining_seconds']:.1f}s")
    print("\n### Criteria")
    print(json.dumps(res["criteria"], indent=1))
    print("\n### Codes / checks")
    print(json.dumps(res["checkpoints"], indent=1))
    print(json.dumps(res["task6_val_lift_check"], indent=1))
    print("label zero-weight naive:", {m: lab[m]["naive_zero_weight_episodes"] for m in lab})
    print(f"runtime {res['runtime_seconds']:.1f}s")


# ----------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tables", action="store_true")
    args = parser.parse_args()
    out_dir = HERE / "cache" / "smoke" if args.smoke else HERE
    results_path = out_dir / "results" / "eval_results.json"
    if args.tables:
        print_tables(json.loads(results_path.read_text()))
        return
    results_path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    mode = "smoke" if args.smoke else "real"
    sizes = SIZES[mode]
    print(f"Mode {mode.upper()}{' (numbers discarded; val stands in for held)' if args.smoke else ''}; {sizes}",
          flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    def guard(stage: str) -> None:
        if perf_counter() - started > TIME_LIMIT_SECONDS:
            raise SystemExit(f"Stopped after {stage}: runtime over {TIME_LIMIT_SECONDS // 60} minutes")

    data, groups, split, styles, held_rows, split_meta = setup(args.smoke)
    label_eps, label_meta = build_label_sets(data, groups, split, styles, held_rows, sizes)
    codes, code_meta = load_codes(data)
    lift_check = task6_lift_check(data, codes, label_eps["val"])
    guard("setup")
    label_results, held_ranked = evaluate_labels(data, codes, label_eps)
    label_boot = label_bootstraps(held_ranked)
    guard("label episodes")
    results = {"mode": mode, "sizes": sizes, "torch": torch.__version__,
               "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               **split_meta, "beta_grid": BETA_GRID, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
               "checkpoints": code_meta, "label_episodes": label_meta, "task6_val_lift_check": lift_check,
               "label": label_results, "label_bootstrap": label_boot}
    results_path.write_text(json.dumps(results, indent=1))
    mined = evaluate_mined(data, groups, split, held_rows, codes, sizes)
    results["mined"] = mined
    results["criteria"] = criteria(label_boot, mined)
    results["runtime_seconds"] = perf_counter() - started
    results_path.write_text(json.dumps(results, indent=1))
    print_tables(results)
    print(f"Total runtime {results['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
