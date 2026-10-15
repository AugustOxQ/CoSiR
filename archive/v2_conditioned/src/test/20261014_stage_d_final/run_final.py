"""Stage (d) final held-out test: G3 replication (seeds 43/44) and the pre-registered criteria (plan Task 7).

Plan: docs/superpowers/plans/2026-09-30-cosir-v2-candidate-a-stage-d.md, Task 7.
Spec: docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-stage-d-design.md, §5 (replication), §6
(final test, pre-registered) and §10 (caveats).

Run from the repository root with the CoSiR environment, one phase per call, in this order:

    python src/test/20261014_stage_d_final/run_final.py --prepare
    python src/test/20261014_stage_d_final/run_final.py --run 43       # and --run 44, one OS process each
    python src/test/20261014_stage_d_final/run_final.py --selection
    python src/test/20261014_stage_d_final/run_final.py --final --smoke   # code path on SELECTION rows, discarded
    python src/test/20261014_stage_d_final/run_final.py --final           # the single held-out run
    python src/test/20261014_stage_d_final/run_final.py --tables          # reprint from saved JSON

Reuse of Task 6: this script IMPORTS Task 6's ``run_selection.py`` (by file path; its folder name is not a
package name) instead of copying it. It reuses Task 6's cache (``cache/prepare.{npz,json}``: split,
sub-split, train-part R3 codes, factor_scale, CLIP k-means labels), its source rebuild
(``build_source`` / ``clip_source_from_labels``), its row-scope masking (``masked``, ``scoped_inputs``)
and its small helpers (``same_label_fraction``, ``recall_summary``, ``pool``, ``weight_stats``,
``concat_episodes``, ``history_summary``). The naive model is built by the same call Task 6 used.

--prepare    rebuilds the split, the selection sub-split, the R3 codes and the CLIP-cluster source from
             scratch and asserts each equals Task 6's cache (k-means refit on scorer-train rows equals the
             cached labels). Asserts the G3 checkpoint SHA-256 and its config (= the defaults). Caches the
             held rows' R3 codes (encoded with all rows, exactly as Task 6's prepare did) for --final.
--run SEED   retrains G3's config with ``seed=SEED`` (43 or 44; swap=False, everything else default) on
             the CLIP-cluster source rebuilt from Task 6's cached labels, scorer-train rows only.
--selection  selection scores (Task 6 Step 4) of G3 seeds 42/43/44 on Task 6's selection episodes
             (SHA-256s asserted); seed 42 must reproduce Task 6's recorded score exactly.
--final      held rows, first and only use: held label episodes (SHA-256s asserted; mismatch stops the
             run), criterion 1 (condition-use gain vs naive, pooled, both directions), criterion 2 (human
             swap test), seeds 43/44 alongside, CLIP-only / uniform context rows. Refuses to run twice.
             With --smoke, the same code path runs on selection rows (no held row is read) and writes to
             results/smoke_final.json; its numbers are discarded.

Row-scope guards (as in Task 6): training NaN-masks CLIP features and codes outside scorer-train rows; the
selection phase masks everything outside the train part; the final phase masks everything outside the
held rows, so an accidental read of another split surfaces as a non-finite score instead of a silent leak.
"""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

_SEL_PATH = ROOT / "src/test/20261013_stage_d_selection/run_selection.py"
_spec = importlib.util.spec_from_file_location("run_selection", _SEL_PATH)
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)

from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.splits import grouped_split, grouped_subsplit, leakage_groups  # noqa: E402
from src.eval.condition_eval import (HumanSwapEpisodes, build_human_swap_episodes, condition_use_gain,  # noqa: E402
                                     human_swap_success, label_ranks, paired_bootstrap, swap_success_difference,
                                     wrong_condition)
from src.eval.label_episodes import (LabelEpisodes, label_episode_recall, label_episode_weights,  # noqa: E402
                                     label_episodes_sha256, standard_label_episodes)
from src.model.conditioning import conditional_score, pair_codes  # noqa: E402
from src.train.condition_sources import ClipClusterSource  # noqa: E402
from src.train.train_factors import R3_CONFIG, encode_rows, load_factor_checkpoint  # noqa: E402
from src.train.train_scorer import (ScorerTrainingConfig, load_scorer_checkpoint, save_scorer_checkpoint,  # noqa: E402
                                    train_scorer)

SEED = 42
REPLICATION_SEEDS = (43, 44)
SEEDS = (42, *REPLICATION_SEEDS)
N_HELD_EPISODES = 1024
N_SWAP_EPISODES = 1024
SELECTED_RUN = "G3"
G3_PATH = sel.CKPT / "G3.pt"
G3_SHA256 = "8d9317c29a224f4cba8c5c1eca1e0bb9c59cf8877a74a673626182ed53fcfd63"
SELECTION_SHA256 = {"emotion": "8405632159883eebd627401724895e67cebca9f8326139aa2c84058033439ea0",
                    "art_style": "e1cfe1ba301ece6a4e3902ea75f2d57a78de896103a2c60be218f8c64cca4c46"}
HELD_SHA256 = {"emotion": "e62ab41f7b54b500b25900a2e82c22d50f662c3843e767aeef3a8511c16a8c85",
               "art_style": "3a58cf9dc3cb670f39a5084f786968d2186bd511f036565a6f825116422e167f"}
R3_EVAL_JSON = ROOT / "src/test/20261012_condition_eval_repaired_factors/results/eval_results.json"
LABELS, SCOPES, DIRECTIONS = sel.LABELS, sel.SCOPES, sel.DIRECTIONS
BETA_FIXED = sel.BETA_FIXED
CACHE, CKPT, RESULTS = HERE / "cache", HERE / "checkpoints", HERE / "results"
HELD_NPZ, PREPARE_JSON = CACHE / "held_codes.npz", CACHE / "prepare_checks.json"
SELECTION_JSON = RESULTS / "selection_seeds.json"
FINAL_JSON, FINAL_NPZ = RESULTS / "final_results.json", RESULTS / "final_ranks.npz"
SMOKE_JSON, SMOKE_NPZ = RESULTS / "smoke_final.json", RESULTS / "smoke_final_ranks.npz"
log = sel.log


def ckpt_path(seed: int) -> Path:
    return G3_PATH if seed == SEED else CKPT / f"G3_seed{seed}.pt"


def seed_config(seed: int) -> ScorerTrainingConfig:
    """G3's config (Task 6: the defaults, swap=False) with only the seed changed."""
    if sel.RUNS[SELECTED_RUN] != ("clip_cluster", False):
        raise AssertionError(f"Task 6's {SELECTED_RUN} is not (clip_cluster, no swap): {sel.RUNS[SELECTED_RUN]}")
    config = dataclasses.replace(sel.run_config(SELECTED_RUN), seed=seed)
    defaults = dataclasses.asdict(ScorerTrainingConfig())
    changed = {k for k, v in dataclasses.asdict(config).items() if v != defaults[k]}
    if not changed <= {"seed"} or config.swap:
        raise AssertionError(f"seed {seed}: config differs from G3's beyond the seed: {changed}")
    return config


def load_g3(seed: int):
    path = ckpt_path(seed)
    if seed == SEED and sel.sha256_file(path) != G3_SHA256:
        raise AssertionError(f"G3 checkpoint SHA-256 mismatch: {sel.sha256_file(path)}")
    scorer, config = load_scorer_checkpoint(path, device="cpu")
    if dataclasses.asdict(config) != dataclasses.asdict(seed_config(seed)):
        raise AssertionError(f"G3 seed {seed}: checkpoint config {config} differs from the plan's")
    return scorer


def build_naive(data, cache):
    """Exactly Task 6's naive: the step-0 model (beta 0.3) from ``train_scorer(..., steps=0)`` on train-part inputs."""
    img, txt, img_codes, txt_codes = sel.scoped_inputs(data, cache, cache["split_train"])
    naive_source = sel.build_source(sel.NAIVE_SOURCE, cache, 0.5 * (sel.masked(cache["img_codes"], cache["scorer_train"])
                                                                   + sel.masked(cache["txt_codes"], cache["scorer_train"])))
    naive, _ = train_scorer(naive_source, img, txt, img_codes, txt_codes, cache["groups"],
                            torch.as_tensor(cache["factor_scale"]),
                            dataclasses.replace(ScorerTrainingConfig(), steps=0), device="cpu")
    return naive


def pin_naive(naive, img_codes, txt_codes, episodes: dict) -> dict:
    """torch.equal of the naive model's weights with label_episode_weights on the first 16 episodes per label."""
    pin = {}
    for label in LABELS:
        sub = LabelEpisodes(*(getattr(episodes[label], f.name)[:sel.PIN_EPISODES]
                              for f in dataclasses.fields(LabelEpisodes)))
        with torch.no_grad():
            w_model = naive.weights(pair_codes(torch.as_tensor(img_codes[sub.supports]),
                                               torch.as_tensor(txt_codes[sub.supports])),
                                    pair_codes(torch.as_tensor(img_codes[sub.contrasts]),
                                               torch.as_tensor(txt_codes[sub.contrasts])))
        if not torch.equal(w_model.cpu(), label_episode_weights(img_codes, txt_codes, sub).cpu()):
            raise AssertionError(f"naive (step-0) weights differ from label_episode_weights on {label}")
        pin[label] = f"torch.equal on the first {sel.PIN_EPISODES} {label} episodes"
    return pin


def flatten_gain(gain: dict) -> dict:
    return {d: {"point": gain[d]["point"], "ci95": list(gain[d]["ci95"])} for d in (*DIRECTIONS, "mean")}


# ----------------------------------------------------------------------------- prepare

def prepare() -> None:
    started = perf_counter()
    CACHE.mkdir(exist_ok=True)
    cache, prep = sel.load_prepared()
    data = load_artelingo()
    groups = leakage_groups(data.paintings, data.img_features)
    split = grouped_split(groups, seed=SEED)
    sizes = (len(split.train), len(split.val), len(split.held))
    if sizes != sel.EXPECTED_SPLIT:
        raise AssertionError(f"Unexpected split sizes {sizes}")
    scorer_train, selection = grouped_subsplit(groups, split.train, sel.SELECTION_FRACTION, seed=SEED)
    for name, mine, cached in (("groups", groups, cache["groups"]), ("split.train", split.train, cache["split_train"]),
                               ("scorer_train", scorer_train, cache["scorer_train"]),
                               ("selection", selection, cache["selection"])):
        if not np.array_equal(mine, cached):
            raise AssertionError(f"Rebuilt {name} differs from Task 6's cache")
    if np.intersect1d(groups[split.held], groups[split.train]).size or np.intersect1d(groups[split.held],
                                                                                      groups[split.val]).size:
        raise AssertionError("A leakage group spans held and another split")
    log(f"Split {sizes} and sub-split ({len(scorer_train):,} / {len(selection):,}) equal Task 6's cache")

    sha = sel.sha256_file(sel.R3_PATH)
    if sha != sel.R3_SHA256:
        raise AssertionError(f"R3 checkpoint SHA-256 mismatch: {sha}")
    model, config = load_factor_checkpoint(sel.R3_PATH)
    if (config.agreement, config.lambda_decorrelation, config.lambda_usage_balance) != (
            R3_CONFIG.agreement, R3_CONFIG.lambda_decorrelation, R3_CONFIG.lambda_usage_balance):
        raise AssertionError(f"Checkpoint config is not R3: {config}")
    img_codes, txt_codes = encode_rows(model, data.img_features, data.txt_features)     # all rows, as Task 6
    if not (np.isfinite(img_codes).all() and np.isfinite(txt_codes).all()):
        raise AssertionError("R3 codes are not finite")
    train = split.train
    codes_equal = bool(np.array_equal(img_codes[train], cache["img_codes"][train])
                       and np.array_equal(txt_codes[train], cache["txt_codes"][train]))
    if not codes_equal:
        diff = max(np.abs(img_codes[train] - cache["img_codes"][train]).max(),
                   np.abs(txt_codes[train] - cache["txt_codes"][train]).max())
        raise AssertionError(f"Re-encoded train-part R3 codes differ from Task 6's cache (max |diff| {diff:.3g})")
    pair = 0.5 * (img_codes + txt_codes)
    factor_scale = (pair[scorer_train].std(axis=0) + 1e-6).astype(np.float32)
    if not np.array_equal(factor_scale, cache["factor_scale"]):
        raise AssertionError("Recomputed factor_scale differs from Task 6's cache")
    log("R3 SHA-256 and config asserted; train-part codes and factor_scale bit-identical to Task 6's cache")

    t0 = perf_counter()
    clip_fit = ClipClusterSource(data.img_features, data.txt_features, scorer_train, n_clusters=sel.N_CLUSTERS,
                                 seed=SEED)
    for view, key in (("image", "clip_image"), ("caption", "clip_caption")):
        if not np.array_equal(clip_fit.labels_by_view[view], cache[key]):
            raise AssertionError(f"k-means refit on scorer-train rows: {view} labels differ from Task 6's cache")
    rebuilt = sel.build_source("clip_cluster", cache, 0.5 * (cache["img_codes"] + cache["txt_codes"]))
    if rebuilt.valid_keys != clip_fit.valid_keys or any(
            not np.array_equal(rebuilt._members[k], clip_fit._members[k]) for k in clip_fit.valid_keys):
        raise AssertionError("CLIP-cluster source rebuilt from the cache differs from the refit source")
    log(f"CLIP-cluster source: k-means refit on scorer-train rows equals Task 6's cached labels "
        f"({len(clip_fit.valid_keys)} valid groups, {perf_counter() - t0:.1f} s)")

    g3 = load_g3(SEED)
    log(f"G3 checkpoint SHA-256 asserted; config = defaults; beta {g3.beta.item():.4f}")

    held = np.asarray(split.held, dtype=np.int64)
    np.savez(HELD_NPZ, held_rows=held, img_codes=img_codes[held], txt_codes=txt_codes[held])
    checks = {"split_sizes": sizes, "split_and_subsplit_equal_task6_cache": True, "r3_sha256": sha,
              "train_part_codes_bit_identical_to_task6": codes_equal, "factor_scale_bit_identical": True,
              "clip_kmeans_refit_equals_task6_labels": True, "clip_valid_groups": len(clip_fit.valid_keys),
              "g3_sha256": G3_SHA256, "held_rows": int(len(held)),
              "held_groups": int(len(np.unique(groups[held]))), "prepare_seconds": perf_counter() - started,
              "versions": {"torch": torch.__version__, "numpy": np.__version__}}
    PREPARE_JSON.write_text(json.dumps(checks, indent=2))
    log(f"Prepared in {checks['prepare_seconds']:.1f} s -> {HELD_NPZ.name} (held codes only), {PREPARE_JSON.name}")


# ----------------------------------------------------------------------------- run

def run(seed: int) -> None:
    if seed not in REPLICATION_SEEDS:
        raise ValueError(f"only the replication seeds {REPLICATION_SEEDS} are trained here")
    started = perf_counter()
    CKPT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    cache, _ = sel.load_prepared()
    data = load_artelingo()
    scorer_train = cache["scorer_train"]
    img, txt, img_codes, txt_codes = sel.scoped_inputs(data, cache, scorer_train)
    source = sel.build_source("clip_cluster", cache, 0.5 * (img_codes + txt_codes))
    config = seed_config(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"G3 seed {seed}: source {source.name}, config {dataclasses.asdict(config)}, device {device}, "
        f"torch threads {torch.get_num_threads()}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    t0 = perf_counter()
    scorer, history = train_scorer(source, img, txt, img_codes, txt_codes, cache["groups"],
                                   torch.as_tensor(cache["factor_scale"]), config, device=device)
    train_seconds = perf_counter() - t0
    if not all(np.isfinite(history[k]).all() for k in ("loss", "loss_rank", "loss_swap", "beta", "tau")):
        raise AssertionError(f"seed {seed}: non-finite values in the training history")
    path = ckpt_path(seed)
    save_scorer_checkpoint(scorer.cpu(), config, path)
    record = {"run": f"G3_seed{seed}", "source": source.name, "config": dataclasses.asdict(config), "device": device,
              "torch_threads": torch.get_num_threads(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
              "train_seconds": train_seconds, "total_seconds": perf_counter() - started,
              "final_beta": sel.scalar(scorer.beta), "final_tau": sel.scalar(scorer.tau), "history": history,
              "checkpoint_sha256": sel.sha256_file(path)}
    (RESULTS / f"history_G3_seed{seed}.json").write_text(json.dumps(record, indent=2))
    log(f"G3 seed {seed}: trained {config.steps} steps in {train_seconds / 60:.1f} min; loss {history['loss'][0]:.4f} "
        f"-> {history['loss'][-1]:.4f}; beta {record['final_beta']:.4f}, tau {record['final_tau']:.4f}")


# ----------------------------------------------------------------------------- shared evaluation

def model_block(name, scorer, ranks, naive_name, img_codes, txt_codes, all_eps) -> dict:
    with torch.no_grad():
        weights = scorer.weights(pair_codes(torch.as_tensor(img_codes[all_eps.supports]),
                                            torch.as_tensor(txt_codes[all_eps.supports])),
                                 pair_codes(torch.as_tensor(img_codes[all_eps.contrasts]),
                                            torch.as_tensor(txt_codes[all_eps.contrasts])))
    entry = {"beta": sel.scalar(scorer.beta), "tau": sel.scalar(scorer.tau), "weights": sel.weight_stats(weights),
             "recall": {}, "recall_wrong": {}, "gain": {}, "r1_minus_naive": {}, "own_condition_use": {}}
    for scope in SCOPES:
        r, rw = ranks[name]["right"][scope], ranks[name]["wrong"][scope]
        entry["recall"][scope] = sel.recall_summary(r)
        entry["recall_wrong"][scope] = sel.recall_summary(rw)
        entry["own_condition_use"][scope] = {
            d: paired_bootstrap((np.asarray(r[d]) <= 1).astype(float) - (np.asarray(rw[d]) <= 1).astype(float))
            for d in DIRECTIONS}
        if name != naive_name:
            n, nw = ranks[naive_name]["right"][scope], ranks[naive_name]["wrong"][scope]
            entry["gain"][scope] = flatten_gain(condition_use_gain(r, rw, n, nw))
            entry["r1_minus_naive"][scope] = {
                d: paired_bootstrap((np.asarray(r[d]) <= 1).astype(float) - (np.asarray(n[d]) <= 1).astype(float))
                for d in DIRECTIONS}
    return entry


def rank_models(models: dict, img, txt, img_codes, txt_codes, episodes: dict, wrong: dict) -> dict:
    ranks = {}
    for name, scorer in models.items():
        right = {label: label_ranks(scorer, img, txt, img_codes, txt_codes, episodes[label]) for label in LABELS}
        wrong_r = {label: label_ranks(scorer, img, txt, img_codes, txt_codes, wrong[label]) for label in LABELS}
        ranks[name] = {"right": {**right, "pooled": sel.pool(right)}, "wrong": {**wrong_r, "pooled": sel.pool(wrong_r)}}
    return ranks


def model_names() -> dict:
    return {"naive": None, **{f"G3_seed{s}": s for s in SEEDS}}


# ----------------------------------------------------------------------------- selection (seeds 43/44)

def selection() -> dict:
    started = perf_counter()
    RESULTS.mkdir(exist_ok=True)
    cache, _ = sel.load_prepared()
    data = load_artelingo()
    groups, rows = cache["groups"], cache["selection"]
    img, txt, img_codes, txt_codes = sel.scoped_inputs(data, cache, cache["split_train"])
    episodes, wrong, meta_eps = {}, {}, {}
    for label in LABELS:
        eps = standard_label_episodes(data, groups, rows, label, sel.N_SELECTION_EPISODES, seed=SEED)
        sha = label_episodes_sha256(eps)
        if sha != SELECTION_SHA256[label]:
            raise AssertionError(f"{label} selection episodes differ from Task 6's: {sha}")
        episodes[label], wrong[label] = eps, wrong_condition(eps, seed=SEED)
        meta_eps[label] = {"n": int(len(eps.anchor)), "sha256": sha}
    log(f"Selection episodes equal Task 6's (SHA-256 asserted): {meta_eps}")
    naive = build_naive(data, cache)
    pin = pin_naive(naive, img_codes, txt_codes, episodes)
    models = {name: (naive if seed is None else load_g3(seed)) for name, seed in model_names().items()}
    ranks = rank_models(models, img, txt, img_codes, txt_codes, episodes, wrong)
    all_eps = sel.concat_episodes([episodes[label] for label in LABELS])
    results = {"models": {name: model_block(name, scorer, ranks, "naive", img_codes, txt_codes, all_eps)
                          for name, scorer in models.items()}}
    task6 = json.loads(sel.RESULTS_JSON.read_text())
    reproduced = {scope: flatten_gain(task6["models"][SELECTED_RUN]["gain"][scope]) == results["models"]["G3_seed42"]["gain"][scope]
                  for scope in SCOPES}
    if not all(reproduced.values()):
        raise AssertionError(f"G3 seed 42 does not reproduce Task 6's selection gain: {reproduced}")
    histories = {}
    for seed in REPLICATION_SEEDS:
        path = RESULTS / f"history_G3_seed{seed}.json"
        histories[f"G3_seed{seed}"] = sel.history_summary(json.loads(path.read_text()))
    histories["G3_seed42"] = sel.history_summary(json.loads((sel.RESULTS / "history_G3.json").read_text()))
    results["histories"] = histories
    results["meta"] = {"episodes": meta_eps, "naive_pin": pin, "g3_seed42_reproduces_task6_gain": reproduced,
                       "checkpoints": {f"G3_seed{s}": sel.sha256_file(ckpt_path(s)) for s in SEEDS},
                       "seconds": perf_counter() - started}
    SELECTION_JSON.write_text(json.dumps(results, indent=2))
    for name in models:
        if name != "naive":
            g = results["models"][name]["gain"]["pooled"]
            log(f"{name}: selection score {100 * g['mean']['point']:+.2f} pts (i2t {100 * g['i2t']['point']:+.2f}, "
                f"t2i {100 * g['t2i']['point']:+.2f}); beta {results['models'][name]['beta']:.4f}")
    log(f"Selection phase in {results['meta']['seconds']:.1f} s -> {SELECTION_JSON}")
    return results


# ----------------------------------------------------------------------------- final (held)

class FixedWeightScorer:
    """A condition-blind scorer (CLIP-only: all-zero weights; uniform: 1/F) with the ConditionalScorer interface."""

    def __init__(self, fill: float, n_factors: int, beta: float):
        self.fill, self.n_factors, self.beta = fill, n_factors, torch.tensor(beta)

    def weights(self, support_pair, contrast_pair):
        return torch.full((support_pair.shape[0], self.n_factors), self.fill, device=support_pair.device)

    def score(self, query_feat, cand_feat, query_codes, cand_codes, weights):
        return conditional_score(query_feat, cand_feat, query_codes, cand_codes, weights, self.beta)


def swap_components(scorer, img, txt, img_codes, txt_codes, eps: HumanSwapEpisodes) -> dict:
    """Descriptive halves of the swap test: p_emo > p_style under c_emo, and p_style > p_emo under c_style."""
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    out = {}
    with torch.no_grad():
        w = {c: scorer.weights(pair_codes(t(img_codes[s]), t(txt_codes[s])), pair_codes(t(img_codes[k]), t(txt_codes[k])))
             for c, s, k in (("emo", eps.supports_emo, eps.contrasts_emo),
                             ("style", eps.supports_style, eps.contrasts_style))}
        for d in DIRECTIONS:
            qf, cf, qc, cc = (img, txt, img_codes, txt_codes) if d == "i2t" else (txt, img, txt_codes, img_codes)
            s = {c: scorer.score(t(qf[eps.anchor]), t(cf[eps.candidates]), t(qc[eps.anchor]), t(cc[eps.candidates]), w[c])
                 for c in w}
            finite = torch.isfinite(s["emo"]).all(dim=1) & torch.isfinite(s["style"]).all(dim=1)
            emo_ok = (s["emo"][:, 0] > s["emo"][:, 1]) & finite
            style_ok = (s["style"][:, 1] > s["style"][:, 0]) & finite
            out[d] = {"emo_ok": emo_ok.numpy(), "style_ok": style_ok.numpy(), "both": (emo_ok & style_ok).numpy()}
    return out


def swap_episode_checks(data, groups, rows, eps: HumanSwapEpisodes) -> dict:
    """Row scope, painting distinctness and one-aspect cleanliness of the human swap episodes."""
    emotions, styles = np.asarray(data.emotions), np.asarray(data.art_styles)
    allowed = np.zeros(len(groups), dtype=bool)
    allowed[rows] = True
    every = np.column_stack([eps.anchor[:, None], eps.supports_emo, eps.contrasts_emo, eps.supports_style,
                             eps.contrasts_style, eps.candidates])
    if not allowed[every].all():
        raise AssertionError("a human swap episode uses a row outside the evaluated rows")
    if any(len(np.unique(groups[r])) != len(r) for r in every):
        raise AssertionError("a painting (leakage group) repeats inside a human swap episode")
    has = {e: set(np.unique(groups[emotions == e]).tolist()) for e in np.unique(eps.emotions)}
    for i, (a, e, s) in enumerate(zip(eps.anchor, eps.emotions, eps.styles)):
        clean = lambda r: not any(int(g) in has[e] for g in groups[np.atleast_1d(r)])  # noqa: E731
        p_emo, p_style, negs = eps.candidates[i, 0], eps.candidates[i, 1], eps.candidates[i, 2:]
        ok = (emotions[a] == e and styles[a] == s and e != "something else"
              and np.all(emotions[eps.supports_emo[i]] == e) and np.all(styles[eps.supports_emo[i]] != s)
              and clean(eps.contrasts_emo[i])
              and np.all(styles[eps.supports_style[i]] == s) and clean(eps.supports_style[i])
              and np.all(styles[eps.contrasts_style[i]] != s)
              and emotions[p_emo] == e and styles[p_emo] != s
              and styles[p_style] == s and clean(p_style)
              and np.all(styles[negs] != s) and clean(negs))
        if not ok:
            raise AssertionError(f"human swap episode {i} is not one-aspect clean")
    digest = hashlib.sha256()
    for f in ("anchor", "supports_emo", "contrasts_emo", "supports_style", "contrasts_style", "candidates"):
        digest.update(np.ascontiguousarray(getattr(eps, f), dtype=np.int64).tobytes())
    digest.update("\n".join(map(str, eps.emotions.tolist())).encode())
    digest.update("\n".join(map(str, eps.styles.tolist())).encode())
    return {"n": int(len(eps.anchor)), "sha256": digest.hexdigest(), "rows_in_scope": True,
            "no_painting_repeats": True, "one_aspect_clean": True,
            "distinct_anchor_emotions": int(len(np.unique(eps.emotions))),
            "distinct_anchor_styles": int(len(np.unique(eps.styles))),
            "candidates_per_episode": int(eps.candidates.shape[1])}


def r3_report_consistency(naive_recall: dict, clip_recall: dict) -> dict:
    """Compare naive (beta 0.3) and CLIP-only held R@1/R@3 with the 2026-10-12 report's recorded values."""
    rec = json.loads(R3_EVAL_JSON.read_text())["label"]["R3"]["held"]
    out = {}
    for name, ours, theirs in (("naive_beta0.3", naive_recall, rec["naive"]["selected"]),
                               ("clip_only", clip_recall, rec["clip_only"]["selected"])):
        cells = {}
        for scope in SCOPES:
            for d in DIRECTIONS:
                for k in ("r1", "r3"):
                    cells[f"{scope}/{d}/{k}"] = (ours[scope][d][k], theirs[scope][d][k])
        out[name] = {"recorded_beta": theirs["beta"], "all_equal": all(a == b for a, b in cells.values()),
                     "max_abs_diff": max(abs(a - b) for a, b in cells.values())}
    return out


def final(smoke: bool = False) -> dict:
    started = perf_counter()
    RESULTS.mkdir(exist_ok=True)
    out_json, out_npz = (SMOKE_JSON, SMOKE_NPZ) if smoke else (FINAL_JSON, FINAL_NPZ)
    if not smoke and FINAL_JSON.exists():
        raise RuntimeError(f"{FINAL_JSON} exists: the held-out test is single-use; use --tables to reprint it")
    cache, _ = sel.load_prepared()
    data = load_artelingo()
    groups = cache["groups"]
    n_rows, n_factors = cache["img_codes"].shape
    if smoke:
        rows = cache["selection"]
        full_img_codes, full_txt_codes = cache["img_codes"], cache["txt_codes"]
    else:
        held = dict(np.load(HELD_NPZ))
        rows = held["held_rows"]
        full_img_codes = np.full((n_rows, n_factors), np.nan, dtype=np.float32)
        full_txt_codes = np.full((n_rows, n_factors), np.nan, dtype=np.float32)
        full_img_codes[rows], full_txt_codes[rows] = held["img_codes"], held["txt_codes"]
        if np.intersect1d(rows, cache["split_train"]).size:
            raise AssertionError("held rows overlap the train part")
    img, txt = sel.masked(data.img_features, rows), sel.masked(data.txt_features, rows)
    img_codes, txt_codes = sel.masked(full_img_codes, rows), sel.masked(full_txt_codes, rows)
    if not (np.isfinite(img_codes[rows]).all() and np.isfinite(txt_codes[rows]).all()):
        raise AssertionError("codes of the evaluated rows are not finite")
    log(f"{'SMOKE (selection rows in place of held)' if smoke else 'HELD'}: {len(rows):,} rows; "
        f"everything else NaN-masked")

    # --- label episodes (held: SHA-256s asserted against the repair plan's Task 7)
    in_rows = np.zeros(n_rows, dtype=bool)
    in_rows[rows] = True
    episodes, wrong, meta_eps = {}, {}, {}
    for label in LABELS:
        eps = standard_label_episodes(data, groups, rows, label, N_HELD_EPISODES, seed=SEED)
        sha = label_episodes_sha256(eps)
        if not smoke and sha != HELD_SHA256[label]:
            raise AssertionError(f"BLOCKED: {label} held episodes differ from the recorded ones: {sha}")
        if not in_rows[np.column_stack([eps.anchor, eps.positive, eps.supports, eps.contrasts, eps.distractors])].all():
            raise AssertionError(f"{label} episodes use rows outside the evaluated rows")
        episodes[label], wrong[label] = eps, wrong_condition(eps, seed=SEED)
        meta_eps[label] = {"n": int(len(eps.anchor)), "targets": int(len(np.unique(eps.labels))), "sha256": sha,
                           "sha256_asserted": not smoke,
                           "wrong_same_label_fraction": sel.same_label_fraction(eps, wrong[label])}
    log(f"Label episodes: {meta_eps}")

    # --- models: naive (Task 6's step-0 build), G3 seeds 42/43/44
    naive = build_naive(data, cache)
    task6_naive_tau = json.loads(sel.RESULTS_JSON.read_text())["models"]["naive"]["tau"]
    if abs(sel.scalar(naive.tau) - task6_naive_tau) > 1e-6 * task6_naive_tau:
        raise AssertionError(f"naive tau {sel.scalar(naive.tau)} differs from Task 6's {task6_naive_tau}")
    pin = pin_naive(naive, img_codes, txt_codes, episodes)
    models = {name: (naive if seed is None else load_g3(seed)) for name, seed in model_names().items()}
    ranks = rank_models(models, img, txt, img_codes, txt_codes, episodes, wrong)
    all_eps = sel.concat_episodes([episodes[label] for label in LABELS])
    results = {"models": {name: model_block(name, scorer, ranks, "naive", img_codes, txt_codes, all_eps)
                          for name, scorer in models.items()}}

    # --- criterion 1 (pre-registered): judged on seed 42; seeds 43/44 get the same test, reported alongside
    crit1 = {}
    for s in SEEDS:
        g = results["models"][f"G3_seed{s}"]["gain"]["pooled"]
        crit1[f"G3_seed{s}"] = {"i2t": g["i2t"], "t2i": g["t2i"], "mean": g["mean"],
                                "met": bool(g["i2t"]["ci95"][0] > 0 and g["t2i"]["ci95"][0] > 0)}
    results["criterion1"] = {"judged_on": "G3_seed42", "met": crit1["G3_seed42"]["met"], "by_seed": crit1,
                             "rule": "condition_use_gain pooled over emotion + art style; met iff ci95[0] > 0 "
                                     "for BOTH i2t AND t2i"}
    log(f"Criterion 1: {results['criterion1']}")

    # --- context: CLIP-only and uniform at beta 0.3 (condition-blind)
    for base, fill in (("clip_only", 0.0), ("uniform", 1.0 / n_factors)):
        per_label = {}
        for label in LABELS:
            w = torch.full((len(episodes[label].anchor), n_factors), fill)
            o = label_episode_recall(img, txt, img_codes, txt_codes, episodes[label], w, BETA_FIXED)
            per_label[label] = {d: o[d]["ranks"] for d in DIRECTIONS}
        pooled = sel.pool(per_label)
        results.setdefault("baselines", {})[base] = {
            "recall": {scope: sel.recall_summary(per_label[scope] if scope != "pooled" else pooled) for scope in SCOPES},
            "g3_seed42_minus_this_r1": {d: paired_bootstrap(
                (np.asarray(ranks["G3_seed42"]["right"]["pooled"][d]) <= 1).astype(float)
                - (np.asarray(pooled[d]) <= 1).astype(float)) for d in DIRECTIONS}}
        ranks[base] = {"right": {**per_label, "pooled": pooled}}
    parity = {}
    for label in LABELS:
        rule = label_episode_recall(img, txt, img_codes, txt_codes, episodes[label],
                                    label_episode_weights(img_codes, txt_codes, episodes[label]), BETA_FIXED)
        parity[label] = {d: float(np.mean(rule[d]["ranks"] == ranks["naive"]["right"][label][d])) for d in DIRECTIONS}
    log(f"Naive vs label_episode_recall(rule, beta 0.3) identical-rank share: {parity}")

    # --- criterion 2 (pre-registered): human emotion-vs-style swap test on the evaluated rows
    swap_eps = build_human_swap_episodes(data, groups, rows, N_SWAP_EPISODES, seed=SEED)
    swap_meta = swap_episode_checks(data, groups, rows, swap_eps)
    log(f"Human swap episodes: {swap_meta}")
    scorers = {**models, "clip_only": FixedWeightScorer(0.0, n_factors, BETA_FIXED),
               "uniform": FixedWeightScorer(1.0 / n_factors, n_factors, BETA_FIXED)}
    success, components = {}, {}
    for name, scorer in scorers.items():
        success[name] = human_swap_success(scorer, img, txt, img_codes, txt_codes, swap_eps)
        comp = swap_components(scorer, img, txt, img_codes, txt_codes, swap_eps)
        for d in DIRECTIONS:
            if not np.array_equal(comp[d]["both"], success[name][d]):
                raise AssertionError(f"{name}: swap components disagree with human_swap_success ({d})")
        components[name] = {d: {k: float(np.mean(v)) for k, v in comp[d].items()} for d in DIRECTIONS}
    crit2 = {}
    for s in SEEDS:
        diff = swap_success_difference(success[f"G3_seed{s}"], success["naive"])
        crit2[f"G3_seed{s}"] = {**diff, "met": bool(diff["pooled"]["ci95"][0] > 0)}
    results["criterion2"] = {"judged_on": "G3_seed42", "met": crit2["G3_seed42"]["met"], "by_seed": crit2,
                             "success_rates": {n: {d: float(np.mean(v[d])) for d in DIRECTIONS}
                                               for n, v in success.items()},
                             "components": components,
                             "rule": "swap_success_difference(G3, naive); met iff pooled.ci95[0] > 0"}
    log(f"Criterion 2: met={results['criterion2']['met']}; G3 seed 42 {crit2['G3_seed42']['pooled']}; "
        f"rates {crit2['G3_seed42']['rates']}")

    results["meta"] = {"smoke": smoke, "rows_evaluated": "selection (smoke)" if smoke else "held",
                       "n_rows": int(len(rows)), "episodes": meta_eps, "swap_episodes": swap_meta,
                       "naive_pin": pin, "naive_tau_equals_task6": True,
                       "naive_vs_rule_identical_rank_share": parity,
                       "checkpoints": {f"G3_seed{s}": sel.sha256_file(ckpt_path(s)) for s in SEEDS},
                       "seconds": perf_counter() - started}
    if not smoke:
        results["meta"]["r3_report_consistency"] = r3_report_consistency(
            results["models"]["naive"]["recall"], results["baselines"]["clip_only"]["recall"])
        log(f"Consistency with the 2026-10-12 report's recorded held numbers: {results['meta']['r3_report_consistency']}")
    out_json.write_text(json.dumps(results, indent=2))
    np.savez(out_npz, **{f"{m}__{kind}__{scope}__{d}": np.asarray(v[scope][d])
                         for m, block in ranks.items() for kind, v in block.items()
                         for scope in SCOPES for d in DIRECTIONS},
             **{f"swap__{n}__{d}": v[d] for n, v in success.items() for d in DIRECTIONS})
    log(f"Final phase ({'smoke' if smoke else 'held'}) in {results['meta']['seconds']:.1f} s -> {out_json}")
    return results


# ----------------------------------------------------------------------------- tables

def ci(block: dict) -> str:
    lo, hi = block["ci95"]
    return f"{100 * block['point']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def tables(final_json: Path = FINAL_JSON) -> None:
    out = []
    if SELECTION_JSON.exists():
        res = json.loads(SELECTION_JSON.read_text())
        out.append("### Selection set: G3 seeds (condition-use gain over naive, pooled, R@1 points, 95% CI)\n")
        out.append("| Model | Score (mean) | i2t | t2i | Emotion mean | Art style mean | beta | R@1 i2t | R@1 t2i |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, m in res["models"].items():
            g = m["gain"]
            gtxt = (f"{ci(g['pooled']['mean'])} | {ci(g['pooled']['i2t'])} | {ci(g['pooled']['t2i'])} | "
                    f"{ci(g['emotion']['mean'])} | {ci(g['art_style']['mean'])}" if g else "0 | 0 | 0 | 0 | 0")
            r = m["recall"]["pooled"]
            out.append(f"| {name} | {gtxt} | {m['beta']:.4f} | {pct(r['i2t']['r1'])} | {pct(r['t2i']['r1'])} |")
        out.append("")
        out.append("R@1 - naive (pooled, selection): " + "; ".join(
            f"{n}: i2t {ci(m['r1_minus_naive']['pooled']['i2t'])}, t2i {ci(m['r1_minus_naive']['pooled']['t2i'])}"
            for n, m in res["models"].items() if m["r1_minus_naive"]))
        out.append("")
        out.append("Loss / beta per seed: " + "; ".join(
            f"{n}: {h['train_minutes']:.1f} min, loss first5 {h['loss']['mean_first5']:.4f} -> last5 "
            f"{h['loss']['mean_last5']:.4f}, beta {h['final_beta']:.4f}, tau {h['final_tau']:.4f}"
            for n, h in res["histories"].items()))
        out.append(f"\nSelection meta: {res['meta']}\n")
    if not final_json.exists():
        print("\n".join(out))
        return
    res = json.loads(final_json.read_text())
    models, base, meta = res["models"], res["baselines"], res["meta"]
    out.append(f"## Final phase ({meta['rows_evaluated']}, {meta['n_rows']:,} rows)\n")
    c1 = res["criterion1"]
    out.append(f"### Criterion 1 (judged on {c1['judged_on']}): {'MET' if c1['met'] else 'NOT MET'}\n")
    out.append("| Model | Delta i2t | Delta t2i | Delta mean | same test met? |")
    out.append("|---|---:|---:|---:|---|")
    for name, c in c1["by_seed"].items():
        out.append(f"| {name} | {ci(c['i2t'])} | {ci(c['t2i'])} | {ci(c['mean'])} | {'yes' if c['met'] else 'no'} |")
    out.append("")
    for label in LABELS:
        out.append(f"Per label type, {label}: " + "; ".join(
            f"{n}: i2t {ci(m['gain'][label]['i2t'])}, t2i {ci(m['gain'][label]['t2i'])}, mean {ci(m['gain'][label]['mean'])}"
            for n, m in models.items() if m["gain"]))
        out.append("")
    c2 = res["criterion2"]
    out.append(f"### Criterion 2 (judged on {c2['judged_on']}): {'MET' if c2['met'] else 'NOT MET'}\n")
    out.append("| Model | diff pooled | diff i2t | diff t2i | model rate i2t / t2i | naive rate i2t / t2i | met? |")
    out.append("|---|---:|---:|---:|---|---|---|")
    for name, c in c2["by_seed"].items():
        rm, rn = c["rates"]["model"], c["rates"]["naive"]
        out.append(f"| {name} | {ci(c['pooled'])} | {ci(c['i2t'])} | {ci(c['t2i'])} | {pct(rm['i2t'])} / {pct(rm['t2i'])} | "
                   f"{pct(rn['i2t'])} / {pct(rn['t2i'])} | {'yes' if c['met'] else 'no'} |")
    out.append("")
    out.append("Swap components (share of episodes; emo_ok = p_emo above p_style under c_emo; style_ok = p_style "
               "above p_emo under c_style; both = success):\n")
    out.append("| Scorer | i2t emo_ok | i2t style_ok | i2t both | t2i emo_ok | t2i style_ok | t2i both |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, comp in c2["components"].items():
        out.append(f"| {name} | " + " | ".join(pct(comp[d][k]) for d in DIRECTIONS
                                                for k in ("emo_ok", "style_ok", "both")) + " |")
    out.append("")
    out.append("### R@1 / R@3 (%, right condition) and R@1 with the wrong condition\n")
    for scope in SCOPES:
        out.append(f"**{scope}**\n")
        out.append("| Model | R@1 i2t | R@1 t2i | R@3 i2t | R@3 t2i | R@1 wrong i2t | R@1 wrong t2i |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, m in models.items():
            r, rw = m["recall"][scope], m["recall_wrong"][scope]
            out.append(f"| {name} | {pct(r['i2t']['r1'])} | {pct(r['t2i']['r1'])} | {pct(r['i2t']['r3'])} | "
                       f"{pct(r['t2i']['r3'])} | {pct(rw['i2t']['r1'])} | {pct(rw['t2i']['r1'])} |")
        for bname, b in base.items():
            r = b["recall"][scope]
            out.append(f"| {bname} (beta 0.3) | {pct(r['i2t']['r1'])} | {pct(r['t2i']['r1'])} | {pct(r['i2t']['r3'])} | "
                       f"{pct(r['t2i']['r3'])} | {pct(r['i2t']['r1'])} | {pct(r['t2i']['r1'])} |")
        out.append("")
    out.append("### Own condition use (R@1 right - wrong) and R@1 - naive (pooled, points, 95% CI)\n")
    out.append("| Model | own use i2t | own use t2i | R@1 - naive i2t | R@1 - naive t2i |")
    out.append("|---|---:|---:|---:|---:|")
    for name, m in models.items():
        own, diff = m["own_condition_use"]["pooled"], m["r1_minus_naive"].get("pooled")
        dtxt = f"{ci(diff['i2t'])} | {ci(diff['t2i'])}" if diff else "0 | 0"
        out.append(f"| {name} | {ci(own['i2t'])} | {ci(own['t2i'])} | {dtxt} |")
    out.append("")
    for label in LABELS:
        out.append(f"R@1 - naive, {label}: " + "; ".join(
            f"{n}: i2t {ci(m['r1_minus_naive'][label]['i2t'])}, t2i {ci(m['r1_minus_naive'][label]['t2i'])}"
            for n, m in models.items() if m["r1_minus_naive"]))
        out.append("")
    for bname, b in base.items():
        out.append(f"G3 seed 42 - {bname} R@1 (pooled): i2t {ci(b['g3_seed42_minus_this_r1']['i2t'])}, "
                   f"t2i {ci(b['g3_seed42_minus_this_r1']['t2i'])}")
    out.append("")
    out.append("### beta, tau and condition weights on the evaluated label episodes\n")
    out.append("| Model | beta | tau | mean active factors | all-zero weight rows | mean max weight |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for name, m in models.items():
        w = m["weights"]
        out.append(f"| {name} | {m['beta']:.4f} | {m['tau']:.4f} | {w['mean_active_factors']:.2f} | "
                   f"{100 * w['all_zero_fraction']:.1f}% | {w['mean_max_weight']:.3f} |")
    out.append("")
    out.append("### Episodes and checks\n")
    for label, m in meta["episodes"].items():
        out.append(f"- {label}: {m['n']} episodes, {m['targets']} targets, SHA-256 `{m['sha256']}` "
                   f"(asserted: {m['sha256_asserted']}), wrong condition from a same-label episode: "
                   f"{100 * m['wrong_same_label_fraction']:.1f}%")
    out.append(f"- human swap episodes: {meta['swap_episodes']}")
    out.append(f"- naive pin: {meta['naive_pin']}; naive tau equals Task 6's: {meta['naive_tau_equals_task6']}")
    out.append(f"- naive vs label_episode_recall(rule, beta 0.3), identical-rank share: "
               f"{meta['naive_vs_rule_identical_rank_share']}")
    if "r3_report_consistency" in meta:
        out.append(f"- consistency with the 2026-10-12 report's recorded held numbers: {meta['r3_report_consistency']}")
    out.append(f"- checkpoints: {meta['checkpoints']}")
    print("\n".join(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", type=int, choices=REPLICATION_SEEDS)
    group.add_argument("--selection", action="store_true")
    group.add_argument("--final", action="store_true")
    group.add_argument("--tables", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="with --final: selection rows in place of held rows")
    args = parser.parse_args()
    if args.smoke and not args.final:
        parser.error("--smoke only applies to --final")
    if args.prepare:
        prepare()
    elif args.run:
        run(args.run)
    elif args.selection:
        selection()
        tables()
    elif args.final:
        final(smoke=args.smoke)
        tables(SMOKE_JSON if args.smoke else FINAL_JSON)
    else:
        tables()


if __name__ == "__main__":
    main()
