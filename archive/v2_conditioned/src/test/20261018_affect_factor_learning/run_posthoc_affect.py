"""Post-hoc diagnostics for the affect factor-learning reports (final-review fix wave, 2026-10-01).

Every number here is post-hoc, on selection rows (or scorer-train rows), and informed no pre-registered decision.
Nothing is trained. Held and val rows are never read: captions are joined for scorer-train and selection rows only,
episodes are drawn from selection rows only, and the one held-side check (``held_bootstrap_seeds``) re-bootstraps the
stored held ranks (results file) without touching any held row. The GoEmotions model is never run: its functions are
replaced by a raising stub at import; the scorer-train affect vectors come from ``cache/affect_prepare.npz``.

Analyses (final-fix findings F1, F4, F5g, F6):
  scorer_train_facts      F1 / F5g: per-emotion rate of the label's own word in scorer-train captions, GoEmotions
                          argmax per ArtEmis emotion, affect-cluster purity.
  word_split              F6.1: D_emo (SE - C0, naive beta 0.3, mean of directions) split by whether the positive's /
                          anchor's caption contains its target emotion's word (word lists fixed below, before any split
                          was computed), with E - C0 for context.
  support_curve           F6.2: naive R@1 at 4, 8, 16, 32 supports (and as many contrasts) on new selection episodes
                          (2,048 per label per count, ``build_label_episodes`` with the standard settings otherwise).
  awe_confusion           F6.3: on selection awe episodes, the emotion carried by the top-1 wrong candidate, SE vs C0.
  cluster_bootstrap       F6.4: D_emo / D_style CIs resampling anchor paintings instead of episodes (design effect).
  held_bootstrap_seeds    F5b: the held D_emo / D_style CIs under other bootstrap seeds (stored ranks only).

Run from the repository root:
    python src/test/20261018_affect_factor_learning/run_posthoc_affect.py --run      # writes results/posthoc_affect.json
    python src/test/20261018_affect_factor_learning/run_posthoc_affect.py --tables   # reprint from the JSON
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

_AFFECT_PATH = HERE / "run_affect.py"
_aspec = importlib.util.spec_from_file_location("run_affect", _AFFECT_PATH)
aff = importlib.util.module_from_spec(_aspec)
_aspec.loader.exec_module(aff)


def _no_affect(*_args, **_kwargs):
    raise RuntimeError("the GoEmotions model must not run in the post-hoc diagnostics")


aff.goemotions_probabilities = _no_affect          # row-scope guard: no caption reaches the affect model here
aff.load_goemotions = _no_affect

from src.data.affect import GOEMOTIONS_MODEL  # noqa: E402
from src.data.artelingo import ANNOTATIONS_PATH, join_captions, load_artelingo  # noqa: E402
from src.eval.condition_eval import paired_bootstrap  # noqa: E402
from src.eval.label_episodes import (EMOTION_CATCH_ALL, STANDARD_MIN_PAINTINGS_PER_LABEL,  # noqa: E402
                                     build_label_episodes, label_episode_weights, label_episodes_sha256,
                                     tie_aware_rank)
from src.model.conditioning import conditional_score  # noqa: E402

grid, probe, sel = aff.grid, aff.probe, aff.sel
LABELS, DIRECTIONS = aff.LABELS, aff.DIRECTIONS
SEED, BETA = 42, 0.3
N_BOOT = 5000
LABEL = "post-hoc, selection rows, informed no pre-registered decision"
OUT_JSON = aff.RESULTS / "posthoc_affect.json"
HELD_NPZ = ROOT / "src/test/20261019_affect_factor_learning_held/results/held_ranks.npz"
HELD_JSON = ROOT / "src/test/20261019_affect_factor_learning_held/results/held_results.json"

# F6.1 word lists, FIXED BEFORE ANY SPLIT WAS COMPUTED (and not revised after): the label word, its inflections and
# its noun / verb / adjective / adverb forms. The only additions beyond that family are the five fear words the fix
# brief named (afraid, scared, scary, frightening, frightened). "awesome" and "awful" are left out of awe (their
# everyday meaning drifted from awe); "content" is kept for contentment although it can also mean subject matter.
# Matching: lower-cased caption, split on non-letters (so "awe-inspiring" yields "awe"), any exact token match.
EMOTION_WORDS = {
    "amusement": ("amuse", "amused", "amuses", "amusing", "amusingly", "amusement", "amusements"),
    "anger": ("anger", "angers", "angered", "angering", "angry", "angrier", "angriest", "angrily"),
    "awe": ("awe", "awed", "awes", "awestruck"),
    "contentment": ("content", "contented", "contentedly", "contentment"),
    "disgust": ("disgust", "disgusts", "disgusted", "disgusting", "disgustingly"),
    "excitement": ("excite", "excites", "excited", "exciting", "excitingly", "excitedly", "excitement",
                   "excitements"),
    "fear": ("fear", "fears", "feared", "fearing", "fearful", "fearfully",
             "afraid", "scared", "scary", "frightening", "frightened"),
    "sadness": ("sad", "sadder", "saddest", "sadly", "sadness", "sadden", "saddens", "saddened", "saddening"),
}
SUPPORT_COUNTS = (4, 8, 16, 32)
CURVE_EPISODES = 2048
CURVE_MODELS = ("C0", "E", "SE")
CURVE_BETAS = (0.3, 0.0)
PURE_CLUSTER = 0.85                                   # "near-pure proxy" threshold for listing clusters
HELD_BOOT_SEEDS = tuple(range(1, 21))
log = grid.log


# ----------------------------------------------------------------------------- helpers

def tokens(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


def has_word(captions, words) -> np.ndarray:
    words = set(words)
    return np.fromiter((bool(tokens(c) & words) for c in captions), dtype=bool, count=len(captions))


def ci_block(values: np.ndarray) -> dict:
    return probe.pct_block(paired_bootstrap(values, N_BOOT, SEED))


def goemotions_label_names() -> list:
    """The 28 class names from the model's config in the local cache (the model itself is not loaded)."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(GOEMOTIONS_MODEL, local_files_only=True)
    names = [config.id2label[i] for i in range(len(config.id2label))]
    if len(names) != 28:
        raise AssertionError(f"expected 28 GoEmotions classes, got {len(names)}")
    return names


def mean_dir_hits(ranks: dict, label: str) -> np.ndarray:
    return 0.5 * (probe.hits(ranks[label]["i2t"]) + probe.hits(ranks[label]["t2i"]))


def stored_naive(npz, model: str, beta: float) -> dict:
    return {label: {d: np.asarray(npz[f"naive__{model}__{beta:g}__{label}__{d}"]) for d in DIRECTIONS}
            for label in LABELS}


# ----------------------------------------------------------------------------- F1 / F5g: scorer-train facts

def scorer_train_facts(data, cache, captions_st) -> dict:
    st = cache["scorer_train"]
    npz = np.load(aff.CACHE / "affect_prepare.npz")
    probs, clusters = npz["affect_probs"], npz["affect_local"]
    record = json.loads((aff.CACHE / "affect_prepare.json").read_text())
    if grid.sha256_file(aff.CACHE / "affect_prepare.npz") != record["affect_npz_sha256"]:
        raise AssertionError("affect_prepare.npz differs from the file recorded at prepare")
    if probs.shape != (len(st), 28) or len(clusters) != len(st) or len(captions_st) != len(st):
        raise AssertionError("affect vectors, clusters and captions must be one per scorer-train row")
    names = goemotions_label_names()
    emo = np.asarray(data.emotions)[st]
    targets = sorted(EMOTION_WORDS)
    named = [e for e in targets if e in names]

    word_rates = {}
    for e in targets:
        hit, own = has_word(captions_st, EMOTION_WORDS[e]), emo == e
        word_rates[e] = {"n_own": int(own.sum()), "own_rate": 100 * float(hit[own].mean()),
                         "n_other": int((~own).sum()), "other_rate": 100 * float(hit[~own].mean()),
                         "n_with_word": int(hit.sum()),
                         "label_share_among_word_captions": 100 * float(own[hit].mean()) if hit.any() else None}

    neutral = names.index("neutral")
    arg_all = probs.argmax(axis=1)
    masked = probs.copy()
    masked[:, neutral] = -1.0
    arg_nn = masked.argmax(axis=1)
    argmax = {}
    for e in np.unique(emo):
        rows = emo == e
        dist = np.bincount(arg_all[rows], minlength=28) / rows.sum()
        dist_nn = np.bincount(arg_nn[rows], minlength=28) / rows.sum()
        top = np.argsort(-dist)[:3]
        argmax[str(e)] = {"n": int(rows.sum()),
                          "top3_all": [[names[i], 100 * float(dist[i])] for i in top],
                          "top_non_neutral": [names[int(dist_nn.argmax())], 100 * float(dist_nn.max())],
                          "same_name_share": 100 * float(dist[names.index(str(e))]) if str(e) in names else None,
                          "neutral_share": 100 * float(dist[neutral])}

    labels_u, emo_ids = np.unique(emo, return_inverse=True)
    base = float(np.bincount(emo_ids).max() / len(emo_ids))
    per_cluster = []
    for c in np.unique(clusters):
        rows = clusters == c
        counts = np.bincount(emo_ids[rows], minlength=len(labels_u))
        per_cluster.append({"cluster": int(c), "rows": int(rows.sum()), "majority": str(labels_u[counts.argmax()]),
                            "purity": float(counts.max() / rows.sum())})
    sizes = np.array([p["rows"] for p in per_cluster])
    purities = np.array([p["purity"] for p in per_cluster])
    weighted = float((sizes * purities).sum() / sizes.sum())
    pure = sorted([p for p in per_cluster if p["purity"] >= PURE_CLUSTER], key=lambda p: -p["rows"])
    in_majority_cluster = {}
    for e in targets:
        own_clusters = [p["cluster"] for p in per_cluster if p["majority"] == e]
        rows = emo == e
        in_majority_cluster[e] = 100 * float(np.isin(clusters[rows], own_clusters).mean())
    return {"goemotions_classes": names, "targets": targets, "targets_named_by_goemotions": named,
            "word_lists": {e: list(v) for e, v in EMOTION_WORDS.items()}, "word_rates": word_rates,
            "argmax_by_artemis_emotion": argmax,
            "cluster_purity": {"row_weighted_purity": weighted, "majority_base": base,
                               "majority_label": str(labels_u[np.bincount(emo_ids).argmax()]),
                               "n_clusters": len(per_cluster), "threshold_listed": PURE_CLUSTER,
                               "clusters_at_or_above_threshold": pure,
                               "share_of_label_rows_in_clusters_it_leads": in_majority_cluster,
                               "per_cluster": per_cluster}}


# ----------------------------------------------------------------------------- F6.1: emotion-word split

def word_split(episodes, ranks: dict, captions_by_row: dict) -> dict:
    eps = episodes["emotion"]
    targets = eps.labels.astype(str)
    pos_word = np.array([bool(tokens(captions_by_row[int(r)]) & set(EMOTION_WORDS[t]))
                         for r, t in zip(eps.positive, targets)])
    anc_word = np.array([bool(tokens(captions_by_row[int(r)]) & set(EMOTION_WORDS[t]))
                         for r, t in zip(eps.anchor, targets)])
    hits = {m: {d: probe.hits(ranks[m]["emotion"][d]) for d in DIRECTIONS} for m in ranks}
    splits = {"all": np.ones(len(targets), bool),
              "positive caption has the word": pos_word, "positive caption lacks it": ~pos_word,
              "anchor caption has the word": anc_word, "anchor caption lacks it": ~anc_word,
              "neither has it": ~pos_word & ~anc_word, "either has it": pos_word | anc_word}
    out = {}
    for name, mask in splits.items():
        block = {"n": int(mask.sum())}
        for m in ranks:
            block[f"{m}_r1"] = {**{d: 100 * float(hits[m][d][mask].mean()) for d in DIRECTIONS},
                                "mean": 100 * float((0.5 * (hits[m]["i2t"] + hits[m]["t2i"]))[mask].mean())}
        for x in ("SE", "E"):
            diff = {d: hits[x][d][mask] - hits["C0"][d][mask] for d in DIRECTIONS}
            block[f"{x}-C0"] = {"mean": ci_block(0.5 * (diff["i2t"] + diff["t2i"])),
                                **{d: ci_block(diff[d]) for d in DIRECTIONS}}
            total = (0.5 * (hits[x]["i2t"] + hits[x]["t2i"]) - 0.5 * (hits["C0"]["i2t"] + hits["C0"]["t2i"]))
            block[f"{x}-C0_share_of_total_gain"] = (float(total[mask].sum() / total.sum()) if total.sum() else None)
        out[name] = block
    per_target = {}
    for t in sorted(np.unique(targets)):
        m = targets == t
        per_target[t] = {"n": int(m.sum()), "positive_word_share": 100 * float(pos_word[m].mean()),
                         "anchor_word_share": 100 * float(anc_word[m].mean())}
    return {"splits": out, "per_target_word_shares": per_target,
            "positive_word_share": 100 * float(pos_word.mean()), "anchor_word_share": 100 * float(anc_word.mean())}


# ----------------------------------------------------------------------------- F6.4: painting-cluster bootstrap

def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Resample clusters (anchor paintings) with replacement; the statistic is the mean over the drawn episodes."""
    _, inv = np.unique(clusters, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values), np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for start in range(0, n_boot, 500):
        idx = rng.integers(0, len(sums), (min(500, n_boot - start), len(sums)))
        boots[start:start + len(idx)] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return {"point": float(values.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def design_effect(episodes, ranks: dict, groups: np.ndarray) -> dict:
    out = {}
    for x in ("SE", "E"):
        out[x] = {}
        for label in LABELS:
            values = mean_dir_hits(ranks[x], label) - mean_dir_hits(ranks["C0"], label)
            anchors = groups[episodes[label].anchor]
            episode = probe.pct_block(paired_bootstrap(values, N_BOOT, SEED))
            cluster = probe.pct_block(cluster_bootstrap(values, anchors))
            w_e, w_c = (b["ci95"][1] - b["ci95"][0] for b in (episode, cluster))
            _, per = np.unique(anchors, return_counts=True)
            out[x][label] = {"episode_bootstrap": episode, "anchor_painting_bootstrap": cluster,
                             "width_ratio": w_c / w_e, "variance_ratio": (w_c / w_e) ** 2,
                             "n_episodes": int(len(values)), "n_anchor_paintings": int(len(per)),
                             "max_episodes_per_anchor_painting": int(per.max()),
                             "mean_episodes_per_anchor_painting": float(per.mean())}
    return out


# ----------------------------------------------------------------------------- scoring

def _t(values) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float32)


def episode_scores(img, txt, ic, tc, eps, weights, beta: float) -> dict:
    cands = np.concatenate([eps.positive[:, None], eps.distractors], axis=1)
    out = {}
    for d, qf, cf, qc, cc in (("i2t", img, txt, ic, tc), ("t2i", txt, img, tc, ic)):
        out[d] = conditional_score(_t(qf[eps.anchor]), _t(cf[cands]), _t(qc[eps.anchor]), _t(cc[cands]), weights,
                                   beta)
    return out


# ----------------------------------------------------------------------------- F6.3: awe confusion

def awe_confusion(episodes, codes_sel: dict, img, txt, stored: dict, data, groups, sl) -> dict:
    eps = episodes["emotion"]
    emotions = np.asarray(data.emotions)
    targets = eps.labels.astype(str)
    awe = targets == "awe"
    cands = np.concatenate([eps.positive[:, None], eps.distractors], axis=1)
    # painting-level majority emotion over the painting's selection rows (images are shared by ~5 captions)
    sl_groups, sl_emo = groups[sl], emotions[sl]
    painting_major = {}
    for g in np.unique(groups[cands[awe]]):
        vals, cnt = np.unique(sl_emo[sl_groups == g], return_counts=True)
        painting_major[int(g)] = str(vals[cnt.argmax()])
    emo_names = sorted(set(np.unique(emotions[sl]).astype(str)))
    pool = {e: 100 * float((emotions[eps.distractors[awe]] == e).mean()) for e in emo_names}
    out = {"n_awe_episodes": int(awe.sum()), "distractor_emotion_share": pool, "models": {}}
    wrong_emotion = {}
    for m in ("SE", "C0"):
        ic, tc = codes_sel[m]
        w = label_episode_weights(ic, tc, eps)
        scores = episode_scores(img, txt, ic, tc, eps, w, BETA)
        out["models"][m] = {}
        wrong_emotion[m] = {}
        for d in DIRECTIONS:
            ranks = tie_aware_rank(scores[d]).numpy()
            if not np.array_equal(ranks, stored[m]["emotion"][d]):
                raise AssertionError(f"{m} {d}: recomputed ranks differ from selection_ranks.npz")
            top = scores[d].argmax(dim=1).numpy()
            miss = ranks > 1
            if np.any(miss & (top == 0)):
                raise AssertionError("a tie at the top: the top-1 wrong candidate is ambiguous")
            top_rows = cands[np.arange(len(top)), top]
            row_emo = np.where(miss, emotions[top_rows], "")
            paint_emo = np.array([painting_major.get(int(groups[r]), "") if mi and a else ""
                                  for r, mi, a in zip(top_rows, miss, awe)])
            wrong_emotion[m][d] = {"row": row_emo, "painting": paint_emo}
            block = {"r1": 100 * float((~miss[awe]).mean()), "misses": int(miss[awe].sum())}
            for kind, arr in (("row", row_emo), ("painting", paint_emo)):
                a = arr[awe & miss]
                block[f"wrong_top1_{kind}_emotion_share_of_misses"] = {e: 100 * float((a == e).mean())
                                                                        for e in emo_names}
                block[f"wrong_top1_{kind}_emotion_rate_per_episode"] = {e: 100 * float((arr[awe] == e).mean())
                                                                         for e in emo_names}
            out["models"][m][d] = block
    diffs = {}
    for kind in ("row", "painting"):
        diffs[kind] = {}
        for e in emo_names:
            per_dir = {d: (wrong_emotion["SE"][d][kind][awe] == e).astype(float)
                       - (wrong_emotion["C0"][d][kind][awe] == e).astype(float) for d in DIRECTIONS}
            diffs[kind][e] = {"mean": ci_block(0.5 * (per_dir["i2t"] + per_dir["t2i"])),
                              **{d: ci_block(per_dir[d]) for d in DIRECTIONS}}
    out["SE_minus_C0_rate_of_wrong_top1_by_emotion"] = diffs
    return out


# ----------------------------------------------------------------------------- F6.2: support-count curve

def support_curve(data, cache, codes_sel: dict, img, txt, stored_all: dict, stored_results: dict) -> dict:
    sl, groups = cache["selection"], cache["groups"]
    in_sel = grid._row_mask(len(groups), sl)
    label_arrays = {"emotion": np.asarray(data.emotions), "art_style": np.asarray(data.art_styles)}
    min_paintings = {}
    for label in LABELS:
        rl = label_arrays[label][sl]
        counts = {str(v): int(len(np.unique(groups[sl][rl == v]))) for v in np.unique(rl)}
        eligible = {k: v for k, v in counts.items()
                    if v >= STANDARD_MIN_PAINTINGS_PER_LABEL and not (label == "emotion" and k == EMOTION_CATCH_ALL)}
        min_paintings[label] = min(eligible.items(), key=lambda kv: kv[1])
    out = {"counts": list(SUPPORT_COUNTS), "episodes_per_label": CURVE_EPISODES, "betas": list(CURVE_BETAS),
           "smallest_eligible_label_paintings": {k: list(v) for k, v in min_paintings.items()},
           "capped": {}, "per_count": {}}
    for k in SUPPORT_COUNTS:
        capped = {label: bool(min_paintings[label][1] < k + 2) for label in LABELS}
        if any(capped.values()):
            raise AssertionError(f"a label cannot supply {k} supports + anchor + positive: {min_paintings}")
        out["capped"][str(k)] = capped
        t0 = perf_counter()
        eps = {}
        for label in LABELS:
            exclude = (EMOTION_CATCH_ALL,) if label == "emotion" else ()
            eps[label] = build_label_episodes(label_arrays[label], groups, sl, CURVE_EPISODES, seed=SEED,
                                              num_support=k, num_contrast=k, num_distractors=12,
                                              min_paintings_per_label=STANDARD_MIN_PAINTINGS_PER_LABEL,
                                              exclude_target_labels=exclude,
                                              exclude_target_paintings_from_negatives=True)
            e = eps[label]
            if not in_sel[np.column_stack([e.anchor, e.positive, e.supports, e.contrasts, e.distractors])].all():
                raise AssertionError(f"k={k} {label}: episode rows outside selection")
            if k == 4 and label_episodes_sha256(e) != grid.fin.SELECTION_SHA256[label]:
                raise AssertionError(f"k=4 {label}: episodes differ from stage (d)'s first 2,048")
        ranks = {}
        for m in CURVE_MODELS:
            ic, tc = codes_sel[m]
            w = {label: label_episode_weights(ic, tc, eps[label]) for label in LABELS}
            for beta in CURVE_BETAS:
                ranks[(m, beta)], _ = probe.fixed_weight_ranks(img, txt, ic, tc, eps, w, beta)
                if k == 4:                                   # the first 2,048 stored episodes, rank for rank
                    for label in LABELS:
                        for d in DIRECTIONS:
                            ref = stored_all[(m, beta)][label][d][:CURVE_EPISODES]
                            if not np.array_equal(np.asarray(ranks[(m, beta)][label][d]), ref):
                                raise AssertionError(f"k=4 {m} beta {beta} {label} {d}: differs from stored ranks")
        block = {"sha256": {label: label_episodes_sha256(eps[label]) for label in LABELS},
                 "targets": {label: int(len(np.unique(eps[label].labels))) for label in LABELS}}
        for beta in CURVE_BETAS:
            b = f"{beta:g}"
            block[b] = {"r1": {m: probe.r1_with_ci(ranks[(m, beta)]) for m in CURVE_MODELS},
                        "SE-C0": probe.r1_diff(ranks[("SE", beta)], ranks[("C0", beta)]),
                        "E-C0": probe.r1_diff(ranks[("E", beta)], ranks[("C0", beta)])}
        out["per_count"][str(k)] = block
        log(f"support count {k}: " + ", ".join(
            f"{m} emo {block['0.3']['r1'][m]['emotion']['mean']['point']:.2f} style "
            f"{block['0.3']['r1'][m]['art_style']['mean']['point']:.2f}" for m in CURVE_MODELS)
            + f" ({perf_counter() - t0:.1f} s)")
    r1 = stored_results["r1"]
    out["stored_4096"] = {m: {f"{s}@{b}": {scope: r1[f"{s}|{m}|{b}"][scope]["mean"]
                                            for scope in ("pooled", "emotion", "art_style")}
                              for s in ("naive", "oracle") for b in ("0.3", "0")}
                          for m in CURVE_MODELS}
    return out


# ----------------------------------------------------------------------------- F5b: held CIs across bootstrap seeds

def held_bootstrap_seeds() -> dict:
    """Re-bootstraps the STORED held ranks (results file) with other seeds; no held row is read."""
    npz = np.load(HELD_NPZ)
    held = json.loads(HELD_JSON.read_text())
    out = {}
    for label, key in (("emotion", "d_emo"), ("art_style", "d_style")):
        diff = sum(0.5 * (probe.hits(npz[f"naive__SE_seed42__0.3__{label}__{d}"])
                          - probe.hits(npz[f"naive__C0_seed42__0.3__{label}__{d}"])) for d in DIRECTIONS)
        ref = probe.pct_block(paired_bootstrap(diff, N_BOOT, SEED))
        stored = held["seeds_vs_c0"]["42"][key]
        if abs(ref["point"] - stored["point"]) > 1e-9 or max(abs(a - b) for a, b in zip(ref["ci95"], stored["ci95"])) > 1e-9:
            raise AssertionError(f"{key}: seed-42 bootstrap of the stored ranks differs from held_results.json")
        cis = {s: probe.pct_block(paired_bootstrap(diff, N_BOOT, s))["ci95"] for s in HELD_BOOT_SEEDS}
        dev = max(max(abs(c[0] - ref["ci95"][0]), abs(c[1] - ref["ci95"][1])) for c in cis.values())
        out[key] = {"seed42": ref, "other_seeds": {str(s): c for s, c in cis.items()},
                    "max_endpoint_deviation_from_seed42": dev,
                    "lower_range": [min(c[0] for c in cis.values()), max(c[0] for c in cis.values())],
                    "upper_range": [min(c[1] for c in cis.values()), max(c[1] for c in cis.values())]}
    return out


# ----------------------------------------------------------------------------- driver

def run() -> dict:
    started = perf_counter()
    data = load_artelingo()
    cache, _, _, _ = grid.load_grid()
    st, sl, groups = cache["scorer_train"], cache["selection"], cache["groups"]
    if np.intersect1d(groups[st], groups[sl]).size:
        raise AssertionError("a painting spans scorer-train and selection")
    # Captions: scorer-train and selection rows only (val and held are never joined).
    annotations = json.loads(Path(ANNOTATIONS_PATH).read_text())
    captions_st = join_captions(data.sample_ids[st], annotations)
    captions_sl = join_captions(data.sample_ids[sl], annotations)
    del annotations
    captions_by_row = {int(r): c for r, c in zip(sl, captions_sl)}
    log(f"Captions joined for {len(st):,} scorer-train and {len(sl):,} selection rows only")

    facts = scorer_train_facts(data, cache, captions_st)
    log("Scorer-train facts done")

    stored_results = json.loads(aff.SELECTION_JSON.read_text())
    episodes, _, eps_meta = aff.selection_episodes(data, cache)
    for label in LABELS:
        if eps_meta[label]["sha256"] != stored_results["episodes"][label]["sha256"]:
            raise AssertionError(f"{label}: rebuilt selection episodes differ from selection_results.json")
    npz = np.load(aff.SELECTION_NPZ)
    stored = {m: stored_naive(npz, m, BETA) for m in ("C0", "E", "SE")}
    stored_all = {(m, b): stored_naive(npz, m, b) for m in CURVE_MODELS for b in CURVE_BETAS}
    for x in ("SE", "E"):
        for label, key in (("emotion", "d_emo"), ("art_style", "d_style")):
            got = probe.r1_diff(stored[x], stored["C0"])[label]["mean"]
            ref = stored_results[key][x]
            if abs(got["point"] - ref["point"]) > 1e-9 or abs(got["ci95"][0] - ref["ci95"][0]) > 1e-9:
                raise AssertionError(f"{x} {key}: stored ranks do not reproduce selection_results.json")

    split = word_split(episodes, stored, captions_by_row)
    log(f"Word split: positive-word share {split['positive_word_share']:.1f}%")
    design = design_effect(episodes, stored, groups)
    log("Anchor-painting bootstrap done")

    prep_record = json.loads((aff.CACHE / "affect_prepare.json").read_text())
    n_rows = len(groups)
    img, txt = (sel.masked(x, sl) for x in (data.img_features, data.txt_features))
    in_sel = grid._row_mask(n_rows, sl)
    codes_sel, models = {}, {}
    for m in CURVE_MODELS:
        ic_full, tc_full, models[m] = aff.model_codes(m, data, cache, prep_record)
        ic, tc = sel.masked(ic_full, sl), sel.masked(tc_full, sl)
        for side, arr in (("img", ic), ("txt", tc)):
            probe.assert_row_scope(f"{m} {side} selection codes", arr, in_sel)
        codes_sel[m] = (ic, tc)
    awe = awe_confusion(episodes, codes_sel, img, txt, stored, data, groups, sl)
    log("Awe confusion done")
    curve = support_curve(data, cache, codes_sel, img, txt, stored_all, stored_results)
    held_seeds = held_bootstrap_seeds()

    results = {"label": LABEL,
               "rows": "captions joined for scorer-train and selection rows only; episodes, codes and CLIP features on "
                       "selection rows only; val and held rows never read (held_bootstrap_seeds re-bootstraps the "
                       "stored held ranks only); the GoEmotions model is never run",
               "settings": {"beta": BETA, "bootstrap": {"n_boot": N_BOOT, "seed": SEED},
                            "word_matching": "lower-cased, split on non-letters, exact token match"},
               "models": models, "scorer_train_facts": facts, "word_split": split, "design_effect": design,
               "awe_confusion": awe, "support_curve": curve, "held_bootstrap_seeds": held_seeds,
               "seconds": perf_counter() - started}
    OUT_JSON.write_text(json.dumps(grid._jsonable(results), indent=2))
    log(f"Post-hoc diagnostics in {results['seconds']:.1f} s -> {OUT_JSON}")
    return results


# ----------------------------------------------------------------------------- tables

def _ci(block: dict) -> str:
    return f"{block['point']:+.2f} [{block['ci95'][0]:+.2f}, {block['ci95'][1]:+.2f}]"


def _r1ci(block: dict) -> str:
    return f"{block['point']:.2f} [{block['ci95'][0]:.2f}, {block['ci95'][1]:.2f}]"


def tables(path: Path = OUT_JSON) -> None:
    r = json.loads(path.read_text())
    f = r["scorer_train_facts"]
    print(f"# {r['label']}\n")
    print(f"GoEmotions names {len(f['targets_named_by_goemotions'])} of {len(f['targets'])} targets: "
          f"{', '.join(f['targets_named_by_goemotions'])}\n")
    print("## Word lists (fixed before the split)\n")
    for e, w in f["word_lists"].items():
        print(f"- {e}: {', '.join(w)}")
    print("\n## Scorer-train word rates (% of captions containing the emotion's word)\n")
    print("| Emotion | captions with the label | with the word | other captions with the word | "
          "label share among word captions |")
    print("|---|---:|---:|---:|---:|")
    for e, v in f["word_rates"].items():
        print(f"| {e} | {v['n_own']:,} | {v['own_rate']:.1f}% | {v['other_rate']:.2f}% | "
              f"{v['label_share_among_word_captions']:.1f}% |")
    print("\n## GoEmotions argmax per ArtEmis emotion (scorer-train)\n")
    print("| ArtEmis emotion | n | top-3 argmax (share) | top non-neutral | same-name argmax | neutral |")
    print("|---|---:|---|---|---:|---:|")
    for e, v in f["argmax_by_artemis_emotion"].items():
        top3 = ", ".join(f"{n} {s:.1f}%" for n, s in v["top3_all"])
        same = f"{v['same_name_share']:.1f}%" if v["same_name_share"] is not None else "no class"
        print(f"| {e} | {v['n']:,} | {top3} | {v['top_non_neutral'][0]} {v['top_non_neutral'][1]:.1f}% | {same} | "
              f"{v['neutral_share']:.1f}% |")
    cp = f["cluster_purity"]
    print(f"\n## Affect-cluster purity: row-weighted {cp['row_weighted_purity']:.3f} against a majority base "
          f"{cp['majority_base']:.3f} ({cp['majority_label']})\n")
    for p in cp["clusters_at_or_above_threshold"]:
        print(f"- cluster {p['cluster']}: {p['rows']:,} rows, {100 * p['purity']:.1f}% {p['majority']}")
    print("Share of each label's rows in clusters it leads: "
          + ", ".join(f"{e} {v:.1f}%" for e, v in cp["share_of_label_rows_in_clusters_it_leads"].items()))

    ws = r["word_split"]
    print(f"\n## F6.1 word split of D_emo (positive-word share {ws['positive_word_share']:.1f}%, anchor-word share "
          f"{ws['anchor_word_share']:.1f}%)\n")
    print("| Split | n | C0 R@1 | SE R@1 | SE - C0 (mean) | SE - C0 i2t | SE - C0 t2i | share of SE gain | E - C0 (mean) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, b in ws["splits"].items():
        print(f"| {name} | {b['n']:,} | {b['C0_r1']['mean']:.2f} | {b['SE_r1']['mean']:.2f} | {_ci(b['SE-C0']['mean'])} | "
              f"{b['SE-C0']['i2t']['point']:+.2f} | {b['SE-C0']['t2i']['point']:+.2f} | "
              f"{100 * b['SE-C0_share_of_total_gain']:.0f}% | {_ci(b['E-C0']['mean'])} |")
    print("\nPer target: " + ", ".join(f"{t} pos {v['positive_word_share']:.1f}% / anchor {v['anchor_word_share']:.1f}%"
                                       f" (n {v['n']})" for t, v in ws["per_target_word_shares"].items()))

    print("\n## F6.4 anchor-painting bootstrap (design effect)\n")
    print("| Cell | Label | episode bootstrap | anchor-painting bootstrap | width ratio | anchor paintings |")
    print("|---|---|---:|---:|---:|---:|")
    for x, per in r["design_effect"].items():
        for label, v in per.items():
            print(f"| {x} - C0 | {label} | {_ci(v['episode_bootstrap'])} | {_ci(v['anchor_painting_bootstrap'])} | "
                  f"{v['width_ratio']:.3f} | {v['n_anchor_paintings']:,} (max {v['max_episodes_per_anchor_painting']}) |")

    a = r["awe_confusion"]
    print(f"\n## F6.3 awe confusion ({a['n_awe_episodes']} awe episodes)\n")
    emos = list(a["distractor_emotion_share"])
    print("| | " + " | ".join(emos) + " |")
    print("|---|" + "---:|" * len(emos))
    print("| distractor pool share | " + " | ".join(f"{a['distractor_emotion_share'][e]:.1f}" for e in emos) + " |")
    for m, per in a["models"].items():
        for d, b in per.items():
            for kind in ("row", "painting") if d == "t2i" else ("row",):
                sh = b[f"wrong_top1_{kind}_emotion_share_of_misses"]
                print(f"| {m} {d} ({kind}; R@1 {b['r1']:.2f}, misses {b['misses']}) | "
                      + " | ".join(f"{sh[e]:.1f}" for e in emos) + " |")
    for kind, per in a["SE_minus_C0_rate_of_wrong_top1_by_emotion"].items():
        print(f"| SE - C0 rate ({kind}, mean of dirs) | " + " | ".join(f"{per[e]['mean']['point']:+.2f}" for e in emos)
              + " |")
    for kind, per in a["SE_minus_C0_rate_of_wrong_top1_by_emotion"].items():
        print(f"\nSE - C0 rate of a wrong top-1 of each emotion ({kind} label), 95% CI: "
              + "; ".join(f"{e} {_ci(per[e]['mean'])} (i2t {per[e]['i2t']['point']:+.2f}, t2i "
                          f"{per[e]['t2i']['point']:+.2f})" for e in emos))

    c = r["support_curve"]
    print(f"\n## F6.2 support-count curve ({c['episodes_per_label']} episodes per label per count; smallest eligible "
          f"label: {c['smallest_eligible_label_paintings']}; capped: {c['capped']})\n")
    for beta in c["betas"]:
        b = f"{beta:g}"
        print(f"\n### naive R@1 at beta {b} (mean of directions)\n")
        print("| supports | scope | C0 | E | SE | SE - C0 | E - C0 |")
        print("|---:|---|---:|---:|---:|---:|---:|")
        for k, blk in c["per_count"].items():
            for scope in ("emotion", "art_style", "pooled"):
                rr = blk[b]["r1"]
                print(f"| {k} | {scope} | {_r1ci(rr['C0'][scope]['mean'])} | {_r1ci(rr['E'][scope]['mean'])} | "
                      f"{_r1ci(rr['SE'][scope]['mean'])} | {_ci(blk[b]['SE-C0'][scope]['mean'])} | "
                      f"{_ci(blk[b]['E-C0'][scope]['mean'])} |")
    print("\nStored 4,096-episode R@1 (naive 4 supports, oracle): ")
    for m, v in c["stored_4096"].items():
        print(f"- {m}: " + "; ".join(f"{k} " + "/".join(f"{x:.2f}" for x in s.values()) for k, s in v.items())
              + " (pooled/emotion/style)")

    h = r["held_bootstrap_seeds"]
    print("\n## F5b held CIs across bootstrap seeds (stored held ranks)\n")
    for key, v in h.items():
        print(f"- {key}: seed 42 {_ci(v['seed42'])}; seeds {min(map(int, v['other_seeds']))}-"
              f"{max(map(int, v['other_seeds']))}: lower {v['lower_range'][0]:+.3f} to {v['lower_range'][1]:+.3f}, "
              f"upper {v['upper_range'][0]:+.3f} to {v['upper_range'][1]:+.3f}; max endpoint deviation "
              f"{v['max_endpoint_deviation_from_seed42']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--tables", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()
    tables()


if __name__ == "__main__":
    main()
