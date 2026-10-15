"""Stage (d) selection run: five self-generated-condition scorers (G1-G5) vs the naive rule (plan Task 6).

Plan: docs/superpowers/plans/2026-09-30-cosir-v2-candidate-a-stage-d.md, Task 6 (STOP POINT).
Spec: docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-stage-d-design.md, §3-§5 and §10.

Run from the repository root with the CoSiR environment, one phase per call:

    python src/test/20261013_stage_d_selection/run_selection.py --prepare
    python src/test/20261013_stage_d_selection/run_selection.py --run G1      # ... G5, one OS process each
    python src/test/20261013_stage_d_selection/run_selection.py --evaluate
    python src/test/20261013_stage_d_selection/run_selection.py --tables      # reprint from saved JSON

--prepare   split (asserted sizes), selection sub-split, R3 checkpoint SHA + codes (finite), factor_scale,
            and every source fit on scorer-train rows only: CLIP k-means labels per view, Stage-1
            community labels (graph -> Stage 1 -> Leiden on scorer-train features). Cached in cache/.
--run Gk    rebuilds the run's source from the cached codes/labels and trains it with the untouched
            ``ScorerTrainingConfig()`` defaults (seed 42; only ``swap`` differs per run, as the plan's
            table says). Parallel OS processes give the same result as sequential ones: every run
            owns its seed-42 RNGs.
--evaluate  selection label episodes (2,048 emotion + 2,048 art style, seed 42), naive = the step-0
            model (pinned with torch.equal against ``label_episode_weights``), every run's ranks on the
            right and the wrong condition, the condition-use gain, CLIP-only / uniform / ceiling rows,
            and the pre-registered selection rule with its stop point.

Row-scope guards: val and held rows are never used. Their codes are not cached (NaN in the cached
arrays) and every post-prepare phase NaN-masks CLIP features and codes outside the rows that phase
may read (training: scorer-train only; evaluation: the train part = scorer-train + selection), so an
accidental read would surface as a non-finite loss or score instead of a silent leak.
"""

import argparse
import dataclasses
import hashlib
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.splits import grouped_split, grouped_subsplit, leakage_groups  # noqa: E402
from src.eval.condition_eval import (ceiling_ranks, condition_use_gain, label_ranks, paired_bootstrap,  # noqa: E402
                                     wrong_condition)
from src.eval.label_episodes import (LabelEpisodes, label_episode_recall, label_episode_weights,  # noqa: E402
                                     label_episodes_sha256, standard_label_episodes)
from src.model.communities import community_stats, detect_communities  # noqa: E402
from src.model.conditioning import pair_codes  # noqa: E402
from src.model.graph import GraphConfig, build_content_graph  # noqa: E402
from src.train.condition_episodes import mine_condition_episodes, mine_swap_episodes, pair_feature_units  # noqa: E402
from src.train.condition_sources import ClipClusterSource, CommunitySource, FactorComboSource  # noqa: E402
from src.train.stage1 import Stage1Config, train_stage1  # noqa: E402
from src.train.train_factors import R3_CONFIG, encode_rows, load_factor_checkpoint  # noqa: E402
from src.train.train_scorer import (ScorerTrainingConfig, load_scorer_checkpoint, save_scorer_checkpoint,  # noqa: E402
                                    train_scorer)

SEED = 42
EXPECTED_SPLIT = (216_107, 30_872, 61_744)
SELECTION_FRACTION = 0.15
R3_PATH = ROOT / "src/test/20261011_factor_repair_grid/checkpoints/selected_seed42.pt"
R3_SHA256 = "1c299fc008ca006ffd2de37315ac3557523bf06e0b755d4dcbf70b999b4e453f"
N_CLUSTERS = 64
LABELS = ("emotion", "art_style")
SCOPES = ("pooled", "emotion", "art_style")
DIRECTIONS = ("i2t", "t2i")
N_SELECTION_EPISODES = 2048
PIN_EPISODES = 16
BETA_FIXED = 0.3
TIE_POINTS = 1.0
STOP_POINTS = 0.5
RUNS = {                                   # plan Task 6 Step 3 table, in table order
    "G1": ("factor_combo", False),
    "G2": ("factor_combo", True),
    "G3": ("clip_cluster", False),
    "G4": ("clip_cluster", True),
    "G5": ("community", False),
}
NAIVE_SOURCE = "factor_combo"              # only sets naive's tau (ranking-invariant); weights are the rule
CACHE, CKPT, RESULTS = HERE / "cache", HERE / "checkpoints", HERE / "results"
PREPARE_NPZ, PREPARE_JSON = CACHE / "prepare.npz", CACHE / "prepare.json"
RESULTS_JSON, RANKS_NPZ = RESULTS / "selection_results.json", RESULTS / "selection_ranks.npz"
PARTITION_MIN_GROUP_ROWS, PARTITION_MAX_TRIES = 200, 100     # ClipClusterSource constructor defaults


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: torch.Tensor) -> float:
    return value.detach().cpu().item()


def masked(values: np.ndarray, keep_rows: np.ndarray) -> np.ndarray:
    """Copy of ``values`` (float32) with every row outside ``keep_rows`` set to NaN."""
    out = np.full(values.shape, np.nan, dtype=np.float32)
    out[keep_rows] = values[keep_rows]
    return out


def size_range(sizes) -> dict:
    sizes = np.asarray(sizes, dtype=np.int64)
    return {"count": int(len(sizes)), "min": int(sizes.min()), "median": float(np.median(sizes)),
            "max": int(sizes.max())}


# ----------------------------------------------------------------------------- prepare

def clip_source_from_labels(labels_by_view: dict, rows: np.ndarray) -> ClipClusterSource:
    """A ClipClusterSource rebuilt from cached k-means labels (no refit); prepare asserts it equals the fit."""
    source = ClipClusterSource.__new__(ClipClusterSource)
    source._setup(labels_by_view, rows, PARTITION_MIN_GROUP_ROWS, PARTITION_MAX_TRIES)
    return source


def partition_stats(source) -> dict:
    out = {}
    for view in source.labels_by_view:
        keys = [k for k in source.valid_keys if k[0] == view]
        fit = source.labels_by_view[view][source.rows]
        all_groups = np.unique(fit[fit >= 0])
        out[view] = {"groups_total": int(len(all_groups)), "groups_valid": len(keys),
                     "valid_sizes": size_range([len(source._members[k]) for k in keys]),
                     "all_sizes": size_range(np.bincount(fit[fit >= 0]))}
    return out


def check_rows_and_distinctness(source, rows, units, keys, swap: bool) -> dict:
    """Mine one real batch (seed-42 generator, same calls as train_scorer's first batch) and check scope."""
    allowed = np.zeros(len(keys), dtype=bool)
    allowed[rows] = True
    rng = np.random.default_rng(SEED)
    ep = mine_condition_episodes(source, units, keys, 64, rng, 4, hard_pool=2048)
    arrays = [np.column_stack([ep.anchor[:, None], ep.supports, ep.contrasts, ep.candidates])]
    if swap:
        sw = mine_swap_episodes(source, units, keys, 64, rng, 4, hard_pool=2048)
        arrays.append(np.column_stack([sw.anchor[:, None], sw.supports_a, sw.contrasts_a, sw.supports_b,
                                       sw.contrasts_b, sw.candidates]))
    for arr in arrays:
        if not allowed[arr].all():
            raise AssertionError(f"{source.name}: a mined row lies outside the fit rows")
        for row in arr:
            if len(np.unique(keys[row])) != len(row):
                raise AssertionError(f"{source.name}: a painting repeats inside an episode")
    return {"episodes_checked": int(sum(len(a) for a in arrays)), "all_rows_in_fit_rows": True,
            "no_painting_repeats": True}


def prepare() -> None:
    started = perf_counter()
    CACHE.mkdir(exist_ok=True)
    data = load_artelingo()
    groups = leakage_groups(data.paintings, data.img_features)
    split = grouped_split(groups, seed=SEED)
    sizes = (len(split.train), len(split.val), len(split.held))
    if sizes != EXPECTED_SPLIT:
        raise AssertionError(f"Unexpected split sizes {sizes}")
    scorer_train, selection = grouped_subsplit(groups, split.train, SELECTION_FRACTION, seed=SEED)
    if not np.array_equal(np.sort(np.concatenate([scorer_train, selection])), np.sort(split.train)):
        raise AssertionError("scorer-train + selection must equal the train part")
    if np.intersect1d(groups[scorer_train], groups[selection]).size:
        raise AssertionError("A leakage group spans scorer-train and selection")
    log(f"Split {sizes}; scorer-train {len(scorer_train):,} rows / {len(np.unique(groups[scorer_train])):,} "
        f"groups, selection {len(selection):,} rows / {len(np.unique(groups[selection])):,} groups")

    sha = sha256_file(R3_PATH)
    if sha != R3_SHA256:
        raise AssertionError(f"R3 checkpoint SHA-256 mismatch: {sha}")
    model, config = load_factor_checkpoint(R3_PATH)
    if (config.agreement, config.lambda_decorrelation, config.lambda_usage_balance) != (
            R3_CONFIG.agreement, R3_CONFIG.lambda_decorrelation, R3_CONFIG.lambda_usage_balance):
        raise AssertionError(f"Checkpoint config is not R3: {config}")
    img_codes, txt_codes = encode_rows(model, data.img_features, data.txt_features)
    if not (np.isfinite(img_codes).all() and np.isfinite(txt_codes).all()):
        raise AssertionError("R3 codes are not finite")
    pair = 0.5 * (img_codes + txt_codes)
    factor_scale = pair[scorer_train].std(axis=0) + 1e-6
    log(f"R3 codes: {img_codes.shape[0]:,} rows x {img_codes.shape[1]} factors, finite; factor_scale "
        f"[{factor_scale.min():.4g}, {factor_scale.max():.4g}]")

    # --- sources, fitted on scorer-train rows only
    t0 = perf_counter()
    factor_source = FactorComboSource(pair, scorer_train)
    stats_rng = np.random.default_rng(SEED)
    draws, inside, outside = 1000, [], []
    for _ in range(draws):
        condition = factor_source._condition(*factor_source._random_factors(stats_rng))
        if condition is not None:
            inside.append(len(condition.inside))
            outside.append(len(condition.outside))
    swap_overlaps = [len(np.intersect1d(a.inside, b.inside))
                     for a, b in (factor_source.sample_swap(stats_rng) for _ in range(100))]
    factor_stats = {"valid_fraction_of_random_draws": len(inside) / draws, "draws": draws,
                    "inside_sizes": size_range(inside), "outside_sizes": size_range(outside),
                    "swap_overlap_sizes_100_pairs": size_range(swap_overlaps), "seconds": perf_counter() - t0}
    log(f"factor_combo: {len(inside)}/{draws} random draws valid; inside {factor_stats['inside_sizes']}")

    t0 = perf_counter()
    clip_fit = ClipClusterSource(data.img_features, data.txt_features, scorer_train, n_clusters=N_CLUSTERS,
                                 seed=SEED)
    clip_seconds = perf_counter() - t0
    clip_rebuilt = clip_source_from_labels(clip_fit.labels_by_view, scorer_train)
    if clip_rebuilt.valid_keys != clip_fit.valid_keys or any(
            not np.array_equal(clip_rebuilt._members[k], clip_fit._members[k]) for k in clip_fit.valid_keys):
        raise AssertionError("ClipClusterSource rebuilt from cached labels differs from the fitted source")
    clip_overlaps = [len(np.intersect1d(a.inside, b.inside))
                     for a, b in (clip_fit.sample_swap(stats_rng) for _ in range(100))]
    clip_stats = {**partition_stats(clip_fit), "swap_overlap_sizes_100_pairs": size_range(clip_overlaps),
                  "seconds": clip_seconds}
    log(f"clip_cluster: {len(clip_fit.valid_keys)} valid groups ({clip_seconds:.1f} s)")

    t0 = perf_counter()
    train_img, train_txt = data.img_features[scorer_train], data.txt_features[scorer_train]
    graph = build_content_graph(train_img, train_txt, GraphConfig())
    stage1_out = io.StringIO()
    with redirect_stdout(stage1_out):
        _, embeddings = train_stage1(train_img, train_txt, graph, Stage1Config())
    if not np.isfinite(embeddings).all():
        raise AssertionError("Stage 1 produced non-finite embeddings")
    local_labels = np.asarray(detect_communities(embeddings), dtype=np.int64)
    community_labels = np.full(len(groups), -1, dtype=np.int64)
    community_labels[scorer_train] = local_labels
    community_source = CommunitySource(community_labels, scorer_train)
    stage1_lines = stage1_out.getvalue().strip().splitlines()
    community_stats_out = {**partition_stats(community_source), "graph_edges": int(graph.nnz // 2),
                           "communities": int(community_stats(local_labels)["num_communities"]),
                           "stage1_first_line": stage1_lines[0], "stage1_last_line": stage1_lines[-1],
                           "seconds": perf_counter() - t0}
    log(f"community: {community_stats_out['communities']} communities, {len(community_source.valid_keys)} valid; "
        f"graph {community_stats_out['graph_edges']:,} edges ({community_stats_out['seconds']:.1f} s)")

    # --- scope checks: every group and every mined row is a scorer-train row
    allowed = np.zeros(len(groups), dtype=bool)
    allowed[scorer_train] = True
    for source in (clip_fit, community_source):
        for key in source.valid_keys:
            cond = source._condition(key)
            if not (allowed[cond.inside].all() and allowed[cond.outside].all()):
                raise AssertionError(f"{source.name} group {key} has rows outside scorer-train")
    units = pair_feature_units(masked(data.img_features, scorer_train), masked(data.txt_features, scorer_train))
    mined_checks = {name: check_rows_and_distinctness(src, scorer_train, units, groups, swap)
                    for name, src, swap in (("factor_combo", factor_source, True),
                                            ("clip_cluster", clip_fit, True),
                                            ("community", community_source, False))}
    log(f"Mined-batch scope checks passed: {mined_checks}")

    keep = split.train                               # val / held codes are never cached
    np.savez(PREPARE_NPZ, groups=groups, split_train=split.train, scorer_train=scorer_train, selection=selection,
             img_codes=masked(img_codes, keep), txt_codes=masked(txt_codes, keep),
             factor_scale=factor_scale.astype(np.float32),
             clip_image=clip_fit.labels_by_view["image"], clip_caption=clip_fit.labels_by_view["caption"],
             community=community_labels)
    meta = {"split_sizes": sizes, "scorer_train_rows": int(len(scorer_train)), "selection_rows": int(len(selection)),
            "scorer_train_groups": int(len(np.unique(groups[scorer_train]))),
            "selection_groups": int(len(np.unique(groups[selection]))),
            "selection_share_of_train_rows": len(selection) / len(split.train),
            "r3_sha256": sha, "r3_config": dataclasses.asdict(config),
            "factor_scale": factor_scale.tolist(),
            "sources": {"factor_combo": factor_stats, "clip_cluster": clip_stats, "community": community_stats_out},
            "mined_checks": mined_checks, "prepare_seconds": perf_counter() - started,
            "versions": {"torch": torch.__version__, "numpy": np.__version__}}
    PREPARE_JSON.write_text(json.dumps(meta, indent=2))
    log(f"Prepared in {meta['prepare_seconds']:.1f} s -> {PREPARE_NPZ.name}, {PREPARE_JSON.name}")


# ----------------------------------------------------------------------------- shared loading

def load_prepared():
    if not PREPARE_NPZ.exists():
        raise FileNotFoundError("Run --prepare first")
    cache = dict(np.load(PREPARE_NPZ))
    return cache, json.loads(PREPARE_JSON.read_text())


def build_source(name: str, cache: dict, pair: np.ndarray):
    rows = cache["scorer_train"]
    if name == "factor_combo":
        return FactorComboSource(pair, rows)
    if name == "clip_cluster":
        return clip_source_from_labels({"image": cache["clip_image"], "caption": cache["clip_caption"]}, rows)
    if name == "community":
        return CommunitySource(cache["community"], rows)
    raise ValueError(name)


def scoped_inputs(data, cache: dict, keep_rows: np.ndarray):
    """CLIP features and R3 codes, NaN outside ``keep_rows``."""
    return (masked(data.img_features, keep_rows), masked(data.txt_features, keep_rows),
            masked(cache["img_codes"], keep_rows), masked(cache["txt_codes"], keep_rows))


def run_config(run: str) -> ScorerTrainingConfig:
    config = dataclasses.replace(ScorerTrainingConfig(), swap=RUNS[run][1])
    defaults = dataclasses.asdict(ScorerTrainingConfig())
    changed = {k for k, v in dataclasses.asdict(config).items() if v != defaults[k]}
    if not changed <= {"swap"} or config.seed != SEED:
        raise AssertionError(f"{run}: config differs from the defaults beyond swap: {changed}")
    return config


# ----------------------------------------------------------------------------- run

def run(name: str) -> None:
    started = perf_counter()
    CKPT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    cache, _ = load_prepared()
    data = load_artelingo()
    scorer_train = cache["scorer_train"]
    img, txt, img_codes, txt_codes = scoped_inputs(data, cache, scorer_train)
    source = build_source(RUNS[name][0], cache, 0.5 * (img_codes + txt_codes))
    config = run_config(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"{name}: source {source.name}, swap={config.swap}, config {dataclasses.asdict(config)}, device {device}, "
        f"torch threads {torch.get_num_threads()}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    t0 = perf_counter()
    scorer, history = train_scorer(source, img, txt, img_codes, txt_codes, cache["groups"],
                                   torch.as_tensor(cache["factor_scale"]), config, device=device)
    train_seconds = perf_counter() - t0
    finite = all(np.isfinite(history[k]).all() for k in ("loss", "loss_rank", "loss_swap", "beta", "tau"))
    if not finite:
        raise AssertionError(f"{name}: non-finite values in the training history")
    save_scorer_checkpoint(scorer.cpu(), config, CKPT / f"{name}.pt")
    record = {"run": name, "source": source.name, "config": dataclasses.asdict(config), "device": device,
              "torch_threads": torch.get_num_threads(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
              "train_seconds": train_seconds, "total_seconds": perf_counter() - started,
              "final_beta": scalar(scorer.beta), "final_tau": scalar(scorer.tau), "history": history,
              "checkpoint_sha256": sha256_file(CKPT / f"{name}.pt")}
    (RESULTS / f"history_{name}.json").write_text(json.dumps(record, indent=2))
    log(f"{name}: trained {config.steps} steps in {train_seconds / 60:.1f} min; loss {history['loss'][0]:.4f} -> "
        f"{history['loss'][-1]:.4f}; beta {record['final_beta']:.4f}, tau {record['final_tau']:.4f}")


# ----------------------------------------------------------------------------- evaluate

def concat_episodes(parts) -> LabelEpisodes:
    return LabelEpisodes(*(np.concatenate([getattr(p, f.name) for p in parts])
                           for f in dataclasses.fields(LabelEpisodes)))


def same_label_fraction(episodes: LabelEpisodes, wrong: LabelEpisodes) -> float:
    """Share of episodes whose wrong condition came from an episode with the same target label."""
    index = {tuple(row): i for i, row in enumerate(map(tuple, episodes.supports.tolist()))}
    if len(index) != len(episodes.anchor):
        raise AssertionError("support sets are not unique; cannot trace the derangement")
    source = np.asarray([index[tuple(row)] for row in wrong.supports.tolist()])
    if np.any(source == np.arange(len(source))) or not np.array_equal(episodes.contrasts[source], wrong.contrasts):
        raise AssertionError("wrong_condition is not a consistent derangement")
    return float(np.mean(episodes.labels[source] == episodes.labels))


def recall_summary(ranks: dict) -> dict:
    return {d: {"r1": float(np.mean(np.asarray(ranks[d]) <= 1)), "r3": float(np.mean(np.asarray(ranks[d]) <= 3))}
            for d in DIRECTIONS}


def pool(per_label: dict) -> dict:
    return {d: np.concatenate([per_label[label][d] for label in LABELS]) for d in DIRECTIONS}


def weight_stats(weights: torch.Tensor) -> dict:
    active = (weights > 0).sum(dim=1).float()
    return {"mean_active_factors": float(active.mean()), "all_zero_fraction": float((active == 0).float().mean()),
            "mean_max_weight": float(weights.max(dim=1).values.mean())}


def history_summary(record: dict) -> dict:
    h = record["history"]
    steps = h["step"]

    def at(key, step):
        return h[key][steps.index(step)] if step in steps and h[key] else None

    out = {"logged_points": len(steps), "train_minutes": record["train_seconds"] / 60,
           "device": record["device"], "final_beta": record["final_beta"], "final_tau": record["final_tau"]}
    for key in ("loss", "loss_rank", "loss_swap"):
        if h[key]:
            out[key] = {"step0": at(key, 0), "step500": at(key, 500), "step1000": at(key, 1000),
                        "step2000": at(key, 2000), "final": h[key][-1], "min": min(h[key]),
                        "mean_last5": float(np.mean(h[key][-5:])), "mean_first5": float(np.mean(h[key][:5]))}
    out["beta"] = {"first_logged": h["beta"][0], "final": h["beta"][-1]}
    out["tau"] = {"first_logged": h["tau"][0], "final": h["tau"][-1]}
    return out


def apply_rule(scores: dict) -> dict:
    """Plan Task 6 Step 5: highest wins; within 1.0 point of the best tie -> no-swap run, then table order."""
    best = max(scores.values())
    tied = [r for r in RUNS if scores[r] >= best - TIE_POINTS - 1e-12]
    no_swap = [r for r in tied if not RUNS[r][1]]
    winner = (no_swap or tied)[0]
    stop = best <= STOP_POINTS + 1e-12
    return {"scores_points": scores, "best_run": max(scores, key=scores.get), "best_points": best,
            "tied_within_1pt": tied, "selected": None if stop else winner, "rule_winner_if_no_stop": winner,
            "stop_point": stop,
            "verdict": ("no trained run beats the naive rule on the selection set" if stop
                        else f"selected {winner}")}


def evaluate(checkpoint_dir: Path = CKPT, results_json: Path = RESULTS_JSON, ranks_npz: Path = RANKS_NPZ) -> dict:
    started = perf_counter()
    cache, prep = load_prepared()
    data = load_artelingo()
    groups, selection, keep = cache["groups"], cache["selection"], cache["split_train"]
    img, txt, img_codes, txt_codes = scoped_inputs(data, cache, keep)

    # --- selection label episodes (selection rows only)
    episodes, wrong, meta_eps = {}, {}, {}
    in_selection = np.zeros(len(groups), dtype=bool)
    in_selection[selection] = True
    for label in LABELS:
        eps = standard_label_episodes(data, groups, selection, label, N_SELECTION_EPISODES, seed=SEED)
        rows = np.column_stack([eps.anchor, eps.positive, eps.supports, eps.contrasts, eps.distractors])
        if not in_selection[rows].all():
            raise AssertionError(f"{label} episodes use rows outside the selection set")
        episodes[label] = eps
        wrong[label] = wrong_condition(eps, seed=SEED)
        meta_eps[label] = {"n": int(len(eps.anchor)), "targets": int(len(np.unique(eps.labels))),
                           "sha256": label_episodes_sha256(eps),
                           "wrong_same_label_fraction": same_label_fraction(eps, wrong[label])}
    log(f"Selection episodes: {meta_eps}")

    # --- naive = the step-0 model (beta 0.3), pinned to the rule with torch.equal
    naive_source = build_source(NAIVE_SOURCE, cache, 0.5 * (masked(cache["img_codes"], cache["scorer_train"])
                                                           + masked(cache["txt_codes"], cache["scorer_train"])))
    naive, _ = train_scorer(naive_source, img, txt, img_codes, txt_codes, groups, torch.as_tensor(cache["factor_scale"]),
                            dataclasses.replace(ScorerTrainingConfig(), steps=0), device="cpu")
    pin = {}
    for label in LABELS:
        sub = LabelEpisodes(*(getattr(episodes[label], f.name)[:PIN_EPISODES]
                              for f in dataclasses.fields(LabelEpisodes)))
        with torch.no_grad():
            w_model = naive.weights(pair_codes(torch.as_tensor(img_codes[sub.supports]),
                                               torch.as_tensor(txt_codes[sub.supports])),
                                    pair_codes(torch.as_tensor(img_codes[sub.contrasts]),
                                               torch.as_tensor(txt_codes[sub.contrasts])))
        w_rule = label_episode_weights(img_codes, txt_codes, sub)
        if not torch.equal(w_model.cpu(), w_rule.cpu()):
            raise AssertionError(f"naive (step-0) weights differ from label_episode_weights on {label}")
        pin[label] = f"torch.equal on the first {PIN_EPISODES} {label} selection episodes"
    log(f"Naive pin passed; naive beta {scalar(naive.beta):.8f}, tau {scalar(naive.tau):.6f}")

    # --- models
    models = {"naive": naive}
    for name in RUNS:
        scorer, config = load_scorer_checkpoint(checkpoint_dir / f"{name}.pt", device="cpu")
        if dataclasses.asdict(config) != dataclasses.asdict(run_config(name)):
            raise AssertionError(f"{name}: checkpoint config differs from the plan's run config")
        models[name] = scorer

    ranks = {}
    for name, scorer in models.items():
        right = {label: label_ranks(scorer, img, txt, img_codes, txt_codes, episodes[label]) for label in LABELS}
        wrong_r = {label: label_ranks(scorer, img, txt, img_codes, txt_codes, wrong[label]) for label in LABELS}
        ranks[name] = {"right": {**right, "pooled": pool(right)}, "wrong": {**wrong_r, "pooled": pool(wrong_r)}}

    all_eps = concat_episodes([episodes[label] for label in LABELS])
    results = {"models": {}, "baselines": {}}
    for name, scorer in models.items():
        with torch.no_grad():
            weights = scorer.weights(pair_codes(torch.as_tensor(img_codes[all_eps.supports]),
                                                torch.as_tensor(txt_codes[all_eps.supports])),
                                     pair_codes(torch.as_tensor(img_codes[all_eps.contrasts]),
                                                torch.as_tensor(txt_codes[all_eps.contrasts])))
        entry = {"beta": scalar(scorer.beta), "tau": scalar(scorer.tau), "weights": weight_stats(weights),
                 "recall": {}, "recall_wrong": {}, "gain": {}, "r1_minus_naive": {}, "own_condition_use": {}}
        for scope in SCOPES:
            r, rw = ranks[name]["right"][scope], ranks[name]["wrong"][scope]
            entry["recall"][scope] = recall_summary(r)
            entry["recall_wrong"][scope] = recall_summary(rw)
            entry["own_condition_use"][scope] = {
                d: paired_bootstrap((np.asarray(r[d]) <= 1).astype(float) - (np.asarray(rw[d]) <= 1).astype(float))
                for d in DIRECTIONS}
            if name != "naive":
                n, nw = ranks["naive"]["right"][scope], ranks["naive"]["wrong"][scope]
                entry["gain"][scope] = condition_use_gain(r, rw, n, nw)
                entry["r1_minus_naive"][scope] = {
                    d: paired_bootstrap((np.asarray(r[d]) <= 1).astype(float) - (np.asarray(n[d]) <= 1).astype(float))
                    for d in DIRECTIONS}
        results["models"][name] = entry
        log(f"{name}: pooled R@1 i2t {entry['recall']['pooled']['i2t']['r1']:.4f} t2i "
            f"{entry['recall']['pooled']['t2i']['r1']:.4f}"
            + (f"; selection score {100 * entry['gain']['pooled']['mean']['point']:+.2f} pts" if name != "naive" else ""))

    # --- descriptive paired contrasts (NOT part of the pre-registered rule; added after the first evaluation):
    #     the swap term's effect within a source, and the rule's tie-break (G3 selected over the top-scoring G5)
    results["contrasts"] = {
        f"{a}_minus_{b}": {"what": what, **{scope: condition_use_gain(
            ranks[a]["right"][scope], ranks[a]["wrong"][scope], ranks[b]["right"][scope], ranks[b]["wrong"][scope])
            for scope in SCOPES}}
        for a, b, what in (("G2", "G1", "swap term, factor_combo"), ("G4", "G3", "swap term, clip_cluster"),
                           ("G5", "G3", "community vs clip_cluster (tie-break)"))}

    # --- baselines at beta 0.3 (condition-independent) and the ceiling diagnostic
    n_factors = img_codes.shape[1]
    for base, fill in (("clip_only", 0.0), ("uniform", 1.0 / n_factors)):
        per_label = {}
        for label in LABELS:
            w = torch.full((len(episodes[label].anchor), n_factors), fill)
            out = label_episode_recall(img, txt, img_codes, txt_codes, episodes[label], w, BETA_FIXED)
            per_label[label] = {d: out[d]["ranks"] for d in DIRECTIONS}
        results["baselines"][base] = {scope: recall_summary(per_label[scope] if scope != "pooled" else pool(per_label))
                                      for scope in SCOPES}
        ranks[base] = {"right": {**per_label, "pooled": pool(per_label)}}
    t0 = perf_counter()
    ceiling = {label: ceiling_ranks(img, txt, img_codes, txt_codes, episodes[label], beta=BETA_FIXED)
               for label in LABELS}
    results["baselines"]["ceiling"] = {scope: recall_summary(ceiling[scope] if scope != "pooled" else pool(ceiling))
                                       for scope in SCOPES}
    ranks["ceiling"] = {"right": {**ceiling, "pooled": pool(ceiling)}}
    log(f"Ceiling in {perf_counter() - t0:.1f} s: pooled R@1 i2t {results['baselines']['ceiling']['pooled']['i2t']['r1']:.4f} "
        f"t2i {results['baselines']['ceiling']['pooled']['t2i']['r1']:.4f}")

    # --- parity: the step-0 model equals label_episode_recall with the rule's weights at beta 0.3
    parity = {}
    for label in LABELS:
        rule = label_episode_recall(img, txt, img_codes, txt_codes, episodes[label],
                                    label_episode_weights(img_codes, txt_codes, episodes[label]), BETA_FIXED)
        parity[label] = {d: float(np.mean(rule[d]["ranks"] == ranks["naive"]["right"][label][d])) for d in DIRECTIONS}
    log(f"Naive vs label_episode_recall(rule, beta 0.3) identical-rank share: {parity}")

    # --- histories and the rule
    histories = {}
    for name in RUNS:
        path = checkpoint_dir.parent / "results" / f"history_{name}.json"
        if path.exists():
            histories[name] = history_summary(json.loads(path.read_text()))
    scores = {name: 100 * results["models"][name]["gain"]["pooled"]["mean"]["point"] for name in RUNS}
    results["selection"] = apply_rule(scores)
    results["meta"] = {"prepare": {k: prep[k] for k in ("split_sizes", "scorer_train_rows", "selection_rows",
                                                        "scorer_train_groups", "selection_groups",
                                                        "selection_share_of_train_rows", "r3_sha256",
                                                        "sources", "mined_checks", "prepare_seconds")},
                       "episodes": meta_eps, "naive_pin": pin, "naive_source_for_tau": NAIVE_SOURCE,
                       "naive_vs_rule_identical_rank_share": parity, "evaluate_seconds": perf_counter() - started,
                       "checkpoints": {name: sha256_file(checkpoint_dir / f"{name}.pt") for name in RUNS},
                       "runs": {name: {"source": RUNS[name][0], "swap": RUNS[name][1]} for name in RUNS}}
    results["histories"] = histories
    results_json.parent.mkdir(exist_ok=True)
    results_json.write_text(json.dumps(results, indent=2))
    np.savez(ranks_npz, **{f"{m}__{kind}__{scope}__{d}": np.asarray(v[scope][d])
                           for m, block in ranks.items() for kind, v in block.items()
                           for scope in SCOPES for d in DIRECTIONS})
    log(f"Rule: {results['selection']}")
    log(f"Evaluated in {results['meta']['evaluate_seconds']:.1f} s -> {results_json}")
    return results


# ----------------------------------------------------------------------------- tables

def num(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def ci(block: dict) -> str:
    lo, hi = block["ci95"]
    return f"{100 * block['point']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"


def tables(results_json: Path = RESULTS_JSON) -> None:
    res = json.loads(results_json.read_text())
    models, base, sel = res["models"], res["baselines"], res["selection"]
    out = []
    out.append("### Selection score (condition-use gain over naive, pooled, R@1 points, 95% CI)\n")
    out.append("| Run | Source | Swap | Score (mean of directions) | i2t | t2i |")
    out.append("|---|---|---|---:|---:|---:|")
    for name in RUNS:
        g = models[name]["gain"]["pooled"]
        out.append(f"| {name} | {RUNS[name][0]} | {'yes' if RUNS[name][1] else 'no'} | {ci(g['mean'])} | "
                   f"{ci(g['i2t'])} | {ci(g['t2i'])} |")
    out.append("")
    out.append(f"Rule: best {sel['best_run']} at {sel['best_points']:+.2f} pts; tied within 1 pt: "
               f"{', '.join(sel['tied_within_1pt'])}; stop point (best <= +0.5): {sel['stop_point']}; "
               f"verdict: {sel['verdict']}.\n")
    for label in LABELS:
        out.append(f"### Condition-use gain over naive, {label} episodes only (R@1 points, 95% CI)\n")
        out.append("| Run | Mean | i2t | t2i |")
        out.append("|---|---:|---:|---:|")
        for name in RUNS:
            g = models[name]["gain"][label]
            out.append(f"| {name} | {ci(g['mean'])} | {ci(g['i2t'])} | {ci(g['t2i'])} |")
        out.append("")
    if "contrasts" in res:
        out.append("### Descriptive paired contrasts of condition-use gain (not part of the rule; R@1 points, 95% CI)\n")
        out.append("| Contrast | What | Pooled mean | Pooled i2t | Pooled t2i | Emotion mean | Art style mean |")
        out.append("|---|---|---:|---:|---:|---:|---:|")
        for key, c in res["contrasts"].items():
            out.append(f"| {key} | {c['what']} | {ci(c['pooled']['mean'])} | {ci(c['pooled']['i2t'])} | "
                       f"{ci(c['pooled']['t2i'])} | {ci(c['emotion']['mean'])} | {ci(c['art_style']['mean'])} |")
        out.append("")
    out.append("### R@1 / R@3 (%, right condition) and R@1 with the wrong condition\n")
    for scope in SCOPES:
        out.append(f"**{scope}**\n")
        out.append("| Model | R@1 i2t | R@1 t2i | R@3 i2t | R@3 t2i | R@1 wrong i2t | R@1 wrong t2i |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for name in ("naive", *RUNS):
            r, rw = models[name]["recall"][scope], models[name]["recall_wrong"][scope]
            out.append(f"| {name} | {100 * r['i2t']['r1']:.2f} | {100 * r['t2i']['r1']:.2f} | {100 * r['i2t']['r3']:.2f} | "
                       f"{100 * r['t2i']['r3']:.2f} | {100 * rw['i2t']['r1']:.2f} | {100 * rw['t2i']['r1']:.2f} |")
        for bname in ("clip_only", "uniform", "ceiling"):
            r = base[bname][scope]
            wr = (f"{100 * r['i2t']['r1']:.2f} | {100 * r['t2i']['r1']:.2f}" if bname != "ceiling" else "n/a | n/a")
            out.append(f"| {bname} (beta 0.3) | {100 * r['i2t']['r1']:.2f} | {100 * r['t2i']['r1']:.2f} | "
                       f"{100 * r['i2t']['r3']:.2f} | {100 * r['t2i']['r3']:.2f} | {wr} |")
        out.append("")
    out.append("### Own condition use (R@1 right - R@1 wrong, pooled, points, 95% CI) and R@1 - naive R@1\n")
    out.append("| Model | own use i2t | own use t2i | R@1 - naive i2t | R@1 - naive t2i |")
    out.append("|---|---:|---:|---:|---:|")
    for name in ("naive", *RUNS):
        m = models[name]
        own = m["own_condition_use"]["pooled"]
        diff = m["r1_minus_naive"].get("pooled")
        dtxt = f"{ci(diff['i2t'])} | {ci(diff['t2i'])}" if diff else "0 | 0"
        out.append(f"| {name} | {ci(own['i2t'])} | {ci(own['t2i'])} | {dtxt} |")
    out.append("")
    out.append("### Learned beta and tau; condition weights on the selection episodes\n")
    out.append("| Model | beta | tau | mean active factors | all-zero weight rows | mean max weight |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for name in ("naive", *RUNS):
        m = models[name]
        w = m["weights"]
        out.append(f"| {name} | {m['beta']:.4f} | {m['tau']:.4f} | {w['mean_active_factors']:.2f} | "
                   f"{100 * w['all_zero_fraction']:.1f}% | {w['mean_max_weight']:.3f} |")
    out.append("")
    out.append("### Loss curves (multi-positive ranking loss; swap runs: total = rank + swap)\n")
    out.append("| Run | minutes | loss step 0 | mean first 5 logged | step 1000 | step 2000 | mean last 5 logged | final | "
               "rank loss first5 -> last5 | swap loss first5 -> last5 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for name, h in res["histories"].items():
        loss = h["loss"]
        rank = h["loss_rank"]
        swap = h.get("loss_swap")
        stxt = f"{swap['mean_first5']:.4f} -> {swap['mean_last5']:.4f}" if swap else "n/a"
        out.append(f"| {name} | {h['train_minutes']:.1f} | {num(loss['step0'])} | {num(loss['mean_first5'])} | "
                   f"{num(loss['step1000'])} | {num(loss['step2000'])} | {num(loss['mean_last5'])} | {num(loss['final'])} | "
                   f"{rank['mean_first5']:.4f} -> {rank['mean_last5']:.4f} | {stxt} |")
    out.append("")
    meta = res["meta"]
    out.append("### Episodes and checks\n")
    for label, m in meta["episodes"].items():
        out.append(f"- {label}: {m['n']} episodes, {m['targets']} target labels, SHA-256 `{m['sha256']}`, "
                   f"wrong condition from a same-label episode: {100 * m['wrong_same_label_fraction']:.1f}%")
    out.append(f"- naive pin: {meta['naive_pin']}")
    out.append(f"- naive vs label_episode_recall(rule, beta 0.3), identical-rank share: "
               f"{meta['naive_vs_rule_identical_rank_share']}")
    print("\n".join(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", choices=sorted(RUNS))
    group.add_argument("--evaluate", action="store_true")
    group.add_argument("--tables", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        run(args.run)
    elif args.evaluate:
        evaluate()
        tables()
    else:
        tables()


if __name__ == "__main__":
    main()
