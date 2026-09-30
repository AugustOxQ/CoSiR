"""CoSiR v2 Candidate A: factor headroom probe (plan Task 1). Label oracle on other codes; condition-source AMI.

Plan: docs/superpowers/plans/2026-09-30-cosir-v2-candidate-a-factor-headroom-probe.md (a spike: the output is
an answer, not kept model code). Every number is a **post-hoc diagnostic on selection rows**.

Stage (d) found the cross-validated label oracle on frozen R3 factors about equal to the naive rule. This
script asks whether the limit is R3's code or frozen CLIP ViT-B/32: it reruns the naive rule and the label
oracle on the stage-(d) selection label episodes with other codes in place of R3,

    R3          the cached R3 codes (unchanged)
    clip512     L2-normalized CLIP features (image -> img code, text -> txt code, signed)
    pca32/128   one shared PCA basis on per-modality-centered scorer-train CLIP features (signed)
    labelprobe  DIAGNOSTIC CEILING: [P_style, P_emotion] from four multinomial logistic probes

and measures how well self-generated condition partitions align with the labels (adjusted mutual
information on scorer-train rows).

Run from the repository root with the CoSiR environment:

    python src/test/20261015_factor_headroom_probe/run_probe.py --smoke --device cpu   # code-path check
    python src/test/20261015_factor_headroom_probe/run_probe.py --run --device cpu     # the full run
    python src/test/20261015_factor_headroom_probe/run_probe.py --tables               # reprint from the JSON

--run     2,048 episodes per label (stage (d)'s, SHA-256 asserted), 200 oracle steps, fits on all scorer-train
          rows; writes results/probe_results.json and results/probe_ranks.npz; asserts the sanity checks.
--smoke   the same code path with 256 episodes per label (the first 256 of the full set; the plan's 128 leave
          one art style with a single episode, which the 2-fold label oracle rejects), 20 oracle steps and
          PCA / probe / k-means fits on a 20,000-row scorer-train subsample; writes results/smoke_probe_*;
          numbers are discarded. Sanity 2 is asserted, sanity 1 and 3 are only reported.
--tables  reprints every table from results/probe_results.json.

Reuse: stage (d)'s modules are imported by file path (as run_posthoc.py does): Task 6's cache
(``sel.load_prepared``: split, sub-split, R3 codes, cached partitions), its masking (``sel.masked``), ``sel.pool``,
``sel.LABELS/SCOPES/DIRECTIONS`` and ``fin.SELECTION_SHA256``.

Row-scope guards: val and held CLIP features are replaced by NaN right after loading, so every code exists
only on train-part rows (asserted: finite there, NaN elsewhere). PCA, probes, k-means and the code scale
scalars are fitted on scorer-train rows only (asserted). Evaluation inputs (CLIP features and codes) are
NaN outside selection rows and every episode row is asserted to be a selection row. The AMI diagnostic
reads labels of fit (scorer-train) rows only.
"""

import argparse
import dataclasses
import importlib.util
import json
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

_FINAL_PATH = ROOT / "src/test/20261014_stage_d_final/run_final.py"
_spec = importlib.util.spec_from_file_location("run_final", _FINAL_PATH)
fin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fin)
sel = fin.sel                                           # Task 6's run_selection.py, imported once

from src.data.artelingo import load_artelingo  # noqa: E402
from src.eval.condition_eval import label_oracle_ranks, paired_bootstrap  # noqa: E402
from src.eval.label_episodes import (label_episode_recall, label_episode_weights, label_episodes_sha256,  # noqa: E402
                                     standard_label_episodes)

SEED = sel.SEED
LABELS, SCOPES, DIRECTIONS = sel.LABELS, sel.SCOPES, sel.DIRECTIONS
CODES = ("R3", "clip512", "pca32", "pca128", "labelprobe")
BETAS = (0.0, 0.03, 0.1, 0.3, 1.0)
BETA_BASELINE = sel.BETA_FIXED                          # naive on R3 at beta 0.3 is every number's baseline
ORACLE_SETTINGS = {"folds": 2, "lr": 0.1, "seed": SEED}
PCA_DIMS = (32, 128)
PROBE_SETTINGS = {"C": 1.0, "max_iter": 500}
PROBE_JOBS = 4
N_CLUSTERS = 64
KMEANS_SETTINGS = {"n_clusters": N_CLUSTERS, "random_state": SEED, "n_init": 3, "batch_size": 4096}
CHANCE_R1 = 100 / 13
NULL_TOLERANCE = 2.0                                    # sanity 3: oracle nulls within 2 points of chance
ORACLE_REPRO_TOLERANCE = 0.25                           # sanity 1: oracle pooled R@1 vs stage (d), points
STAGE_D_POOLED_R1 = {"naive": {"i2t": 18.36, "t2i": 20.36}, "oracle": {"i2t": 18.41, "t2i": 21.04}}
IDENTICAL_SHARE_MIN = 0.999                             # float-rounding tolerance for "ranks identically"
LABEL = "post-hoc diagnostic on selection rows; fits on scorer-train rows; informs no pre-registered decision"
RESULTS = HERE / "results"
POSTHOC_RANKS_NPZ = sel.RESULTS / "posthoc_ranks.npz"
log = sel.log


@dataclasses.dataclass(frozen=True)
class Mode:
    name: str
    n_episodes: int
    oracle_steps: int
    fit_subsample: int | None                           # None = every scorer-train row
    prefix: str


MODES = {"run": Mode("run", sel.N_SELECTION_EPISODES, 200, None, "probe"),
         "smoke": Mode("smoke", 256, 20, 20_000, "smoke_probe")}      # 128 leaves a 1-episode art style


class Timer:
    """Wall-clock seconds per named phase (a phase entered twice accumulates)."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        t0 = perf_counter()
        yield
        self.seconds[name] = self.seconds.get(name, 0.0) + perf_counter() - t0
        log(f"[{name}] {self.seconds[name]:.1f} s")


# ----------------------------------------------------------------------------- helpers

def row_mask(n_rows: int, rows: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_rows, dtype=bool)
    mask[rows] = True
    return mask


def assert_row_scope(name: str, values: np.ndarray, allowed: np.ndarray) -> None:
    """Finite on every allowed row, NaN on every other row."""
    if not np.isfinite(values[allowed]).all():
        raise AssertionError(f"{name}: non-finite values on allowed rows")
    if not np.isnan(values[~allowed]).all():
        raise AssertionError(f"{name}: values outside the allowed rows")


def scatter_rows(values: np.ndarray, rows: np.ndarray, n_rows: int) -> np.ndarray:
    """(len(rows), F) -> (n_rows, F) float32, NaN outside ``rows``."""
    out = np.full((n_rows, values.shape[1]), np.nan, dtype=np.float32)
    out[rows] = values
    return out


def l2_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def mean_rms(img_code: np.ndarray, txt_code: np.ndarray, rows: np.ndarray) -> float:
    """Mean over dimensions of the per-dimension RMS over ``rows`` (image and text codes pooled)."""
    x = np.concatenate([img_code[rows], txt_code[rows]]).astype(np.float64)
    return float(np.sqrt(np.mean(x ** 2, axis=0)).mean())


def pct_block(block: dict) -> dict:
    return {"point": 100 * block["point"], "ci95": [100 * x for x in block["ci95"]]}


def hits(ranks) -> np.ndarray:
    return (np.asarray(ranks) <= 1).astype(np.float64)


def by_scope(per_label: dict) -> dict:
    return {**per_label, "pooled": sel.pool(per_label)}


def r1_points(per_label: dict) -> dict:
    """R@1 (%) per scope: i2t, t2i and their mean."""
    scoped = by_scope(per_label)
    out = {}
    for scope in SCOPES:
        r = {d: 100 * float(hits(scoped[scope][d]).mean()) for d in DIRECTIONS}
        out[scope] = {**r, "mean": 0.5 * (r["i2t"] + r["t2i"])}
    return out


def r1_with_ci(per_label: dict) -> dict:
    """R@1 (%) per scope with a bootstrap 95% CI (per episode; the mean column averages the directions)."""
    scoped = by_scope(per_label)
    out = {}
    for scope in SCOPES:
        h = {d: hits(scoped[scope][d]) for d in DIRECTIONS}
        out[scope] = {**{d: pct_block(paired_bootstrap(h[d])) for d in DIRECTIONS},
                      "mean": pct_block(paired_bootstrap(0.5 * (h["i2t"] + h["t2i"])))}
    return out


def r1_diff(a: dict, b: dict) -> dict:
    """Paired bootstrap of hit(a) - hit(b) per episode, R@1 points, per scope, direction and mean."""
    sa, sb = by_scope(a), by_scope(b)
    out = {}
    for scope in SCOPES:
        diff = {d: hits(sa[scope][d]) - hits(sb[scope][d]) for d in DIRECTIONS}
        out[scope] = {**{d: pct_block(paired_bootstrap(diff[d])) for d in DIRECTIONS},
                      "mean": pct_block(paired_bootstrap(0.5 * (diff["i2t"] + diff["t2i"])))}
    return out


def identical_share(a: dict, b: dict, n: int | None = None) -> float:
    """Smallest share (over labels and directions) of episodes with equal ranks; ``b`` may be longer (prefix)."""
    return min(float(np.mean(np.asarray(a[label][d]) == np.asarray(b[label][d])[:n or len(a[label][d])]))
               for label in LABELS for d in DIRECTIONS)


def key(scorer: str, code: str, beta: float) -> str:
    return f"{scorer}|{code}|{beta:g}"


# ----------------------------------------------------------------------------- Step 1: data, rows, episodes

def load_inputs(timer: Timer):
    with timer.phase("load"):
        cache, _ = sel.load_prepared()
        data = load_artelingo()
        n_rows = len(cache["groups"])
        train, scorer_train, selection = cache["split_train"], cache["scorer_train"], cache["selection"]
        if np.intersect1d(scorer_train, selection).size:
            raise AssertionError("scorer-train and selection rows overlap")
        if not (np.isin(scorer_train, train).all() and np.isin(selection, train).all()):
            raise AssertionError("scorer-train and selection must lie inside the train part")
        in_train = row_mask(n_rows, train)
        # val / held CLIP features are never read: NaN from here on
        data = dataclasses.replace(data, img_features=sel.masked(data.img_features, train),
                                   txt_features=sel.masked(data.txt_features, train))
        for name in ("img_codes", "txt_codes"):
            assert_row_scope(f"cache {name}", cache[name], in_train)
        in_scorer = row_mask(n_rows, scorer_train)
        for name in ("clip_image", "clip_caption", "community"):
            if not ((cache[name][in_scorer] >= 0).all() and (cache[name][~in_scorer] == -1).all()):
                raise AssertionError(f"cache {name}: labels are not exactly the scorer-train rows")
    log(f"Rows: train {len(train):,}, scorer-train {len(scorer_train):,}, selection {len(selection):,}; "
        f"torch threads {torch.get_num_threads()}")
    return cache, data


def build_episodes(data, cache: dict, mode: Mode) -> tuple[dict, dict, dict]:
    in_sel = row_mask(len(cache["groups"]), cache["selection"])
    episodes, meta = {}, {}
    for label in LABELS:
        eps = standard_label_episodes(data, cache["groups"], cache["selection"], label, mode.n_episodes, seed=SEED)
        sha = label_episodes_sha256(eps)
        if mode.name == "run" and sha != fin.SELECTION_SHA256[label]:
            raise AssertionError(f"{label} selection episodes differ from stage (d)'s: {sha}")
        if not in_sel[np.column_stack([eps.anchor, eps.positive, eps.supports, eps.contrasts, eps.distractors])].all():
            raise AssertionError(f"{label} episodes use rows outside the selection set")
        episodes[label] = eps
        meta[label] = {"n": int(len(eps.anchor)), "targets": int(len(np.unique(eps.labels))), "sha256": sha,
                       "equals_stage_d_sha256": sha == fin.SELECTION_SHA256[label]}
    rng = np.random.default_rng(SEED)                   # exactly stage (d)'s null targets
    nulls = {}
    for label in LABELS:
        k = episodes[label].distractors.shape[1] + 1
        if k != 13:
            raise AssertionError(f"expected 13 candidates per episode, got {k}")
        nulls[label] = rng.integers(1, k, len(episodes[label].anchor))
    log(f"Episodes: {meta}")
    return episodes, nulls, meta


def fit_rows_for(cache: dict, mode: Mode) -> np.ndarray:
    scorer_train = cache["scorer_train"]
    if mode.fit_subsample is None:
        rows = scorer_train
    else:
        rows = np.sort(np.random.default_rng(SEED).choice(scorer_train, mode.fit_subsample, replace=False))
    if not np.isin(rows, scorer_train).all() or (mode.fit_subsample is None and not np.array_equal(rows, scorer_train)):
        raise AssertionError("fit rows must equal (smoke: be a subset of) the scorer-train rows")
    return rows


# ----------------------------------------------------------------------------- Step 2: codes

def pca_codes(img: np.ndarray, txt: np.ndarray, fit_pos: np.ndarray, dims: int) -> tuple:
    """One PCA basis on stacked per-modality-centered fit rows; each modality projected (signed)."""
    mu_i, mu_t = img[fit_pos].mean(axis=0), txt[fit_pos].mean(axis=0)
    stacked = np.concatenate([img[fit_pos] - mu_i, txt[fit_pos] - mu_t])
    pca = PCA(n_components=dims, svd_solver="randomized", random_state=SEED).fit(stacked)
    basis = pca.components_.T.astype(np.float32)
    info = {"explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "fit_rows_stacked": len(stacked)}
    return (img - mu_i) @ basis, (txt - mu_t) @ basis, info


def _fit_probe(x: np.ndarray, y: np.ndarray) -> tuple[LogisticRegression, dict]:
    """One multinomial logistic probe (joblib worker); returns the model and fit diagnostics."""
    t0 = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LogisticRegression(**PROBE_SETTINGS, random_state=SEED).fit(x, y)
    msgs = sorted({str(w.message).splitlines()[0][:120] for w in caught})
    n_iter = int(np.max(model.n_iter_))
    return model, {"n_iter": n_iter, "converged": n_iter < PROBE_SETTINGS["max_iter"], "warnings": msgs,
                   "seconds": perf_counter() - t0}


def labelprobe_codes(data, feats: dict, train: np.ndarray, fit_pos: np.ndarray, selection: np.ndarray) -> tuple:
    """img code = [P_style(img), P_emotion(img)], txt code = [P_style(txt), P_emotion(txt)] (diagnostic ceiling)."""
    labels = {"art_style": np.asarray(data.art_styles)[train], "emotion": np.asarray(data.emotions)[train]}
    standardized = {}
    for modality, x in feats.items():                  # standardize with fit-row mean / std
        mu, sd = x[fit_pos].mean(axis=0), x[fit_pos].std(axis=0)
        standardized[modality] = (x - mu) / np.maximum(sd, 1e-12)
    jobs = [(m, lab) for m in ("image", "text") for lab in ("art_style", "emotion")]
    fitted = Parallel(n_jobs=PROBE_JOBS, backend="loky")(
        delayed(_fit_probe)(standardized[m][fit_pos], labels[lab][fit_pos]) for m, lab in jobs)
    models = {job: model for job, (model, _) in zip(jobs, fitted)}
    for lab in ("art_style", "emotion"):
        if not np.array_equal(models[("image", lab)].classes_, models[("text", lab)].classes_):
            raise AssertionError(f"{lab}: image and text probes have different classes")
    sel_pos = np.searchsorted(train, selection)
    info, codes = {}, {}
    for m in ("image", "text"):
        parts = []
        for lab in ("art_style", "emotion"):
            model, diag = models[(m, lab)], fitted[jobs.index((m, lab))][1]
            prob = model.predict_proba(standardized[m]).astype(np.float32)
            parts.append(prob)
            fit_labels = labels[lab][fit_pos]
            values, counts = np.unique(fit_labels, return_counts=True)
            majority = values[counts.argmax()]
            truth = labels[lab][sel_pos]
            predicted = model.classes_[prob[sel_pos].argmax(axis=1)]
            info[f"{m}->{lab}"] = {**diag, "classes": [str(c) for c in model.classes_],
                                   "selection_top1_accuracy": float(np.mean(predicted == truth)),
                                   "selection_majority_accuracy": float(np.mean(truth == majority)),
                                   "majority_class": str(majority), "fit_rows": int(len(fit_pos))}
        codes[m] = np.concatenate(parts, axis=1)
    return codes["image"], codes["text"], info


def build_codes(data, cache: dict, fit_rows: np.ndarray, timer: Timer) -> tuple[dict, dict]:
    """Every code as (img_code, txt_code) over all rows: finite on train-part rows, NaN elsewhere, rescaled."""
    train, n_rows = cache["split_train"], len(cache["groups"])
    fit_pos = np.searchsorted(train, fit_rows)
    if not np.array_equal(train[fit_pos], fit_rows):
        raise AssertionError("fit rows are not train-part rows")
    feats = {"image": data.img_features[train], "text": data.txt_features[train]}
    codes = {"R3": (cache["img_codes"], cache["txt_codes"])}
    info = {"R3": {"construction": "cache img_codes / txt_codes (stage d, R3 checkpoint)"}}
    with timer.phase("codes:clip512"):
        codes["clip512"] = tuple(scatter_rows(l2_rows(feats[m]), train, n_rows) for m in ("image", "text"))
        info["clip512"] = {"construction": "CLIP features L2-normalized per row (signed)"}
    for dims in PCA_DIMS:
        with timer.phase(f"codes:pca{dims}"):
            ic, tc, extra = pca_codes(feats["image"], feats["text"], fit_pos, dims)
            codes[f"pca{dims}"] = (scatter_rows(ic, train, n_rows), scatter_rows(tc, train, n_rows))
            info[f"pca{dims}"] = {"construction": "shared randomized PCA on stacked per-modality-centered fit rows",
                                  **extra}
    with timer.phase("codes:labelprobe"):
        ic, tc, probes = labelprobe_codes(data, feats, train, fit_pos, cache["selection"])
        codes["labelprobe"] = (scatter_rows(ic, train, n_rows), scatter_rows(tc, train, n_rows))
        info["labelprobe"] = {"construction": "[P_style, P_emotion] of per-modality logistic probes "
                                              "(diagnostic ceiling)", "probes": probes}
    in_train = row_mask(n_rows, train)
    reference = mean_rms(*codes["R3"], fit_rows)
    for name in CODES:
        ic, tc = codes[name]
        before = mean_rms(ic, tc, fit_rows)
        scale = 1.0 if name == "R3" else reference / before
        if name != "R3":
            codes[name] = (ic * np.float32(scale), tc * np.float32(scale))
        for side, arr in zip(("img", "txt"), codes[name]):
            assert_row_scope(f"{name} {side} code", arr, in_train)
        info[name].update({"dims": int(codes[name][0].shape[1]), "scale": scale, "mean_rms_before": before,
                           "mean_rms_after": mean_rms(*codes[name], fit_rows)})
    log("Codes: " + "; ".join(f"{n} {info[n]['dims']}d scale {info[n]['scale']:.4g}" for n in CODES))
    return codes, info


# ----------------------------------------------------------------------------- Step 3: scorers

def fixed_weight_ranks(img, txt, ic, tc, episodes: dict, weights: dict, beta: float) -> tuple[dict, dict]:
    per_label, ties = {}, {}
    for label in LABELS:
        out = label_episode_recall(img, txt, ic, tc, episodes[label], weights[label], beta)
        per_label[label] = {d: out[d]["ranks"] for d in DIRECTIONS}
        ties[label] = {d: out[d]["tied_episodes"] for d in DIRECTIONS}
    return per_label, ties


def oracle_ranks(img, txt, ic, tc, episodes: dict, beta: float, steps: int, device,
                 targets: dict | None = None) -> dict:
    return {label: label_oracle_ranks(img, txt, ic, tc, episodes[label], beta=beta, steps=steps, device=device,
                                      target_column=0 if targets is None else targets[label], **ORACLE_SETTINGS)
            for label in LABELS}


def best_beta(ranks: dict, scorer: str, code: str) -> float:
    """Grid beta with the highest pooled R@1 averaged over directions (first in grid order on a tie)."""
    scores = [r1_points(ranks[key(scorer, code, b)])["pooled"]["mean"] for b in BETAS]
    return BETAS[int(np.argmax(scores))]


def evaluate_codes(codes: dict, img, txt, cache: dict, episodes: dict, nulls: dict, mode: Mode, device,
                   timer: Timer) -> tuple[dict, dict, dict]:
    selection = cache["selection"]
    ranks, ties, best = {}, {}, {}
    with timer.phase("eval:clip_only"):
        ic, tc = (sel.masked(c, selection) for c in codes["R3"])
        zero = {label: torch.zeros(len(episodes[label].anchor), ic.shape[1]) for label in LABELS}
        ranks[key("clip_only", "-", BETA_BASELINE)], _ = fixed_weight_ranks(img, txt, ic, tc, episodes, zero,
                                                                             BETA_BASELINE)
    for name in CODES:
        ic, tc = (sel.masked(c, selection) for c in codes[name])
        with timer.phase(f"eval:{name}:naive"):
            weights = {label: label_episode_weights(ic, tc, episodes[label]) for label in LABELS}
            for beta in BETAS:
                ranks[key("naive", name, beta)], ties[key("naive", name, beta)] = fixed_weight_ranks(
                    img, txt, ic, tc, episodes, weights, beta)
            if name == "clip512":                        # sanity 2 input: uniform weights at beta 0
                uniform = {label: torch.full((len(episodes[label].anchor), ic.shape[1]), 1.0 / ic.shape[1])
                           for label in LABELS}
                ranks[key("uniform", name, 0.0)], _ = fixed_weight_ranks(img, txt, ic, tc, episodes, uniform, 0.0)
        with timer.phase(f"eval:{name}:oracle"):
            for beta in BETAS:
                t0 = perf_counter()
                ranks[key("oracle", name, beta)] = oracle_ranks(img, txt, ic, tc, episodes, beta, mode.oracle_steps,
                                                                device)
                log(f"  {name} oracle beta {beta:g}: pooled R@1 "
                    f"{r1_points(ranks[key('oracle', name, beta)])['pooled']} ({perf_counter() - t0:.1f} s)")
        best[name] = {"naive": best_beta(ranks, "naive", name), "oracle": best_beta(ranks, "oracle", name)}
        with timer.phase(f"eval:{name}:null"):
            for beta in sorted({0.0, best[name]["oracle"]}):
                ranks[key("oracle_null", name, beta)] = oracle_ranks(img, txt, ic, tc, episodes, beta,
                                                                     mode.oracle_steps, device, nulls)
        at_best = {s: r1_points(ranks[key(s, name, best[name][s])])['pooled'] for s in ('naive', 'oracle')}
        log(f"{name}: best beta {best[name]}; pooled R@1 at best beta {at_best}")
    return ranks, ties, best


# ----------------------------------------------------------------------------- sanity checks

def sanity_checks(ranks: dict, best: dict, mode: Mode) -> dict:
    full = mode.name == "run"
    n = mode.n_episodes
    stored_sel = np.load(sel.RANKS_NPZ)
    stored_post = np.load(POSTHOC_RANKS_NPZ)
    stored = {kind: {label: {d: npz[f"{prefix}{label}__{d}"] for d in DIRECTIONS} for label in LABELS}
              for kind, npz, prefix in (("naive", stored_sel, "naive__right__"),
                                        ("oracle", stored_post, "label_oracle__"),
                                        ("clip_only", stored_sel, "clip_only__right__"))}
    out = {}

    # 1. R3 naive and oracle at beta 0.3 reproduce stage (d) (smoke: prefix comparison, reported only)
    ours = {"naive": ranks[key("naive", "R3", 0.3)], "oracle": ranks[key("oracle", "R3", 0.3)],
            "clip_only": ranks[key("clip_only", "-", BETA_BASELINE)]}
    s1 = {kind: {"identical_rank_share": identical_share(ours[kind], stored[kind], n),
                 "pooled_r1": {d: r1_points(ours[kind])["pooled"][d] for d in DIRECTIONS},
                 "stored_pooled_r1": {d: r1_points({lab: {dd: v[dd][:n] for dd in DIRECTIONS}
                                                     for lab, v in stored[kind].items()})["pooled"][d]
                                      for d in DIRECTIONS}}
          for kind in ours}
    if full:
        naive_ok = (s1["naive"]["identical_rank_share"] >= IDENTICAL_SHARE_MIN
                    and all(round(s1["naive"]["pooled_r1"][d], 2) == STAGE_D_POOLED_R1["naive"][d] for d in DIRECTIONS))
        oracle_ok = all(abs(s1["oracle"]["pooled_r1"][d] - STAGE_D_POOLED_R1["oracle"][d]) <= ORACLE_REPRO_TOLERANCE
                        for d in DIRECTIONS)
        s1.update({"naive_passed": naive_ok, "oracle_passed": oracle_ok, "asserted": True})
        if not (naive_ok and oracle_ok):
            raise AssertionError(f"sanity 1 (stage-d reproduction) failed: {s1}")
    else:
        s1.update({"asserted": False, "note": f"smoke: first {n} episodes of stage (d)'s sets; oracle uses "
                                              f"{mode.oracle_steps} steps and a different fold split, not comparable"})
    out["1_stage_d_reproduction"] = s1

    # 2. clip512 with uniform weights at beta 0 ranks as CLIP-only (asserted in both modes)
    share = identical_share(ranks[key("uniform", "clip512", 0.0)], ranks[key("clip_only", "-", BETA_BASELINE)])
    out["2_clip512_uniform_equals_clip_only"] = {"identical_rank_share": share,
                                                 "passed": share >= IDENTICAL_SHARE_MIN, "asserted": True}
    if share < IDENTICAL_SHARE_MIN:
        raise AssertionError(f"sanity 2 failed: clip512 uniform at beta 0 vs CLIP-only identical share {share}")

    # 3. every oracle null within NULL_TOLERANCE points of chance (pooled, each direction)
    s3 = {}
    for name in CODES:
        for beta in sorted({0.0, best[name]["oracle"]}):
            r = r1_points(ranks[key("oracle_null", name, beta)])["pooled"]
            near = all(abs(r[d] - CHANCE_R1) <= NULL_TOLERANCE for d in DIRECTIONS)
            s3[key("oracle_null", name, beta)] = {**{d: r[d] for d in DIRECTIONS}, "passed": near}
    passed = all(v["passed"] for v in s3.values())
    out["3_oracle_null_near_chance"] = {"cells": s3, "passed": passed, "asserted": full, "chance": CHANCE_R1}
    if full and not passed:
        raise AssertionError(f"sanity 3 failed: {s3}")
    log(f"Sanity: 1 {s1}; 2 share {share:.4f}; 3 passed {passed}")
    return out


# ----------------------------------------------------------------------------- Step 4: comparisons

def comparisons(ranks: dict, best: dict) -> dict:
    r3_oracle = ranks[key("oracle", "R3", best["R3"]["oracle"])]
    r3_naive = ranks[key("naive", "R3", BETA_BASELINE)]
    out = {}
    for name in CODES:
        oracle = ranks[key("oracle", name, best[name]["oracle"])]
        naive = ranks[key("naive", name, best[name]["naive"])]
        block = {"oracle_minus_R3_naive0.3": r1_diff(oracle, r3_naive),
                 "naive_minus_R3_naive0.3": r1_diff(naive, r3_naive)}
        if name != "R3":
            block["oracle_minus_R3_oracle"] = r1_diff(oracle, r3_oracle)
        out[name] = block
    out["clip_only"] = {"clip_only_minus_R3_naive0.3": r1_diff(ranks[key("clip_only", "-", BETA_BASELINE)], r3_naive)}
    return out


# ----------------------------------------------------------------------------- Step 5: condition-source alignment

def caption_residual_partition(data, cache: dict, fit_rows: np.ndarray) -> tuple[np.ndarray, dict]:
    """k-means on L2-normalized (caption feature - mean caption feature of its painting's fit rows)."""
    groups = cache["groups"][fit_rows]
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    x = data.txt_features[fit_rows].astype(np.float64)
    member = csr_matrix((np.ones(len(fit_rows)), (inverse, np.arange(len(fit_rows)))))
    means = np.asarray(member @ x) / counts[:, None]
    keep = counts[inverse] >= 2
    residual = x[keep] - means[inverse[keep]]
    norms = np.linalg.norm(residual, axis=1)
    residual = (residual / np.maximum(norms, 1e-12)[:, None]).astype(np.float32)
    fit = MiniBatchKMeans(**KMEANS_SETTINGS).fit_predict(residual)
    labels = np.full(len(cache["groups"]), -1, dtype=np.int64)
    labels[fit_rows[keep]] = fit
    return labels, {"rows": int(keep.sum()), "paintings": int((counts >= 2).sum()),
                    "zero_residual_rows": int((norms < 1e-9).sum()), "clusters_used": int(len(np.unique(fit)))}


def alignment(data, cache: dict, fit_rows: np.ndarray, timer: Timer) -> dict:
    with timer.phase("ami:caption_residual_kmeans"):
        residual, residual_info = caption_residual_partition(data, cache, fit_rows)
    random_labels = np.full(len(cache["groups"]), -1, dtype=np.int64)
    random_labels[fit_rows] = np.random.default_rng(SEED).integers(0, N_CLUSTERS, len(fit_rows))
    partitions = {"clip_image": cache["clip_image"], "clip_caption": cache["clip_caption"],
                  "community": cache["community"], "caption_residual": residual, "random64": random_labels}
    labels = {"emotion": np.asarray(data.emotions), "art_style": np.asarray(data.art_styles)}
    in_scorer = row_mask(len(cache["groups"]), cache["scorer_train"])
    out = {}
    with timer.phase("ami:scores"):
        for pname, part in partitions.items():
            rows = fit_rows[part[fit_rows] >= 0]
            if not in_scorer[rows].all():
                raise AssertionError(f"{pname}: AMI rows outside scorer-train")
            out[pname] = {lab: {"ami": float(adjusted_mutual_info_score(labels[lab][rows], part[rows])),
                                "n_rows": int(len(rows)), "n_groups": int(len(np.unique(part[rows]))),
                                "n_labels": int(len(np.unique(labels[lab][rows])))}
                          for lab in LABELS}
            log(f"AMI {pname}: " + ", ".join(f"{lab} {out[pname][lab]['ami']:.4f}" for lab in LABELS))
    return {"partitions": out, "caption_residual": residual_info,
            "note": "rows = fit rows (scorer-train; smoke: subsample) with partition label >= 0; "
                    "emotion uses all 9 emotions incl. 'something else'"}


# ----------------------------------------------------------------------------- run

def run(mode: Mode, device: torch.device) -> dict:
    started = perf_counter()
    timer = Timer()
    torch.manual_seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    cache, data = load_inputs(timer)
    with timer.phase("episodes"):
        episodes, nulls, meta_eps = build_episodes(data, cache, mode)
    fit_rows = fit_rows_for(cache, mode)
    codes, code_info = build_codes(data, cache, fit_rows, timer)
    img, txt = (sel.masked(x, cache["selection"]) for x in (data.img_features, data.txt_features))
    ranks, ties, best = evaluate_codes(codes, img, txt, cache, episodes, nulls, mode, device, timer)
    del codes
    sanity = sanity_checks(ranks, best, mode)
    with timer.phase("comparisons"):
        comp = comparisons(ranks, best)
        headline = {name: {"naive": {"beta": best[name]["naive"],
                                     "r1": r1_with_ci(ranks[key("naive", name, best[name]["naive"])])},
                           "oracle": {"beta": best[name]["oracle"],
                                      "r1": r1_with_ci(ranks[key("oracle", name, best[name]["oracle"])])},
                           "oracle_null": {f"{b:g}": r1_with_ci(ranks[key("oracle_null", name, b)])
                                           for b in sorted({0.0, best[name]["oracle"]})}}
                    for name in CODES}
        headline["clip_only"] = {"beta": BETA_BASELINE, "r1": r1_with_ci(ranks[key("clip_only", "-", BETA_BASELINE)])}
    ami = alignment(data, cache, fit_rows, timer)
    results = {"label": LABEL, "mode": mode.name,
               "settings": {"betas": list(BETAS), "baseline": f"naive on R3 at beta {BETA_BASELINE}",
                            "oracle": {**ORACLE_SETTINGS, "steps": mode.oracle_steps},
                            "probe": {**PROBE_SETTINGS, "backend": "sklearn LogisticRegression (lbfgs, multinomial)",
                                      "parallel_jobs": PROBE_JOBS},
                            "pca": {"dims": list(PCA_DIMS), "svd_solver": "randomized", "random_state": SEED},
                            "kmeans": KMEANS_SETTINGS, "n_episodes_per_label": mode.n_episodes,
                            "fit_rows": int(len(fit_rows)), "fit_rows_are_all_scorer_train": mode.fit_subsample is None,
                            "best_beta_rule": "highest pooled R@1 averaged over directions, chosen in-sample from 5",
                            "null_targets": "rng(42).integers(1, 13) per label in LABELS order (stage d)"},
               "episodes": meta_eps, "codes": code_info, "best_beta": best,
               "r1": {k: r1_points(v) for k, v in ranks.items()}, "naive_tied_episodes": ties,
               "headline": headline, "chance_r1": CHANCE_R1, "comparisons": comp, "ami": ami, "sanity": sanity}
    results["meta"] = {"device": str(device), "torch_threads": torch.get_num_threads(),
                       "versions": {"torch": torch.__version__, "numpy": np.__version__},
                       "rows": {"evaluation": "selection rows only (everything else NaN-masked)",
                                "fits": "scorer-train rows only (smoke: a 20,000-row subsample)",
                                "val_held": "CLIP features NaN-masked at load; never read"},
                       "timings_seconds": timer.seconds, "seconds": perf_counter() - started}
    json_path, npz_path = RESULTS / f"{mode.prefix}_results.json", RESULTS / f"{mode.prefix}_ranks.npz"
    json_path.write_text(json.dumps(results, indent=2))
    np.savez(npz_path, **{f"{k.replace('|', '__')}__{label}__{d}": np.asarray(v[label][d])
                          for k, v in ranks.items() for label in LABELS for d in DIRECTIONS})
    log(f"Timings (s): { {k: round(v, 1) for k, v in timer.seconds.items()} }")
    log(f"{mode.name} run in {results['meta']['seconds']:.1f} s -> {json_path}")
    return results


# ----------------------------------------------------------------------------- tables

def ci(block: dict) -> str:
    lo, hi = block["ci95"]
    return f"{block['point']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def r1ci(block: dict) -> str:
    lo, hi = block["ci95"]
    return f"{block['point']:.2f} [{lo:.2f}, {hi:.2f}]"


def tables(path: Path) -> None:
    res = json.loads(path.read_text())
    out = [f"All numbers: {res['label']}. Mode: {res['mode']}; {res['settings']['n_episodes_per_label']} episodes "
           f"per label; oracle steps {res['settings']['oracle']['steps']}; fit rows {res['settings']['fit_rows']:,}.\n"]

    out.append("### Codes (rescaled to R3's mean per-dimension RMS over fit rows)\n")
    out.append("| Code | dims | scale | mean RMS before | after | construction |")
    out.append("|---|---:|---:|---:|---:|---|")
    for name in CODES:
        c = res["codes"][name]
        evr = c.get("explained_variance_ratio_sum")
        extra = f"; PCA explained variance {evr:.3f}" if evr is not None else ""
        out.append(f"| {name} | {c['dims']} | {c['scale']:.4g} | {c['mean_rms_before']:.4g} | "
                   f"{c['mean_rms_after']:.4g} | {c['construction']}{extra} |")
    out.append("")
    out.append("### Label probes (top-1 accuracy on selection rows, %)\n")
    out.append("| Probe | classes | accuracy | majority-class accuracy (majority) | lbfgs iterations | converged | "
               "fit s |")
    out.append("|---|---:|---:|---|---:|---|---:|")
    for pname, p in res["codes"]["labelprobe"]["probes"].items():
        out.append(f"| {pname} | {len(p['classes'])} | {100 * p['selection_top1_accuracy']:.2f} | "
                   f"{100 * p['selection_majority_accuracy']:.2f} ({p['majority_class']}) | {p['n_iter']} | "
                   f"{p['converged']} | {p['seconds']:.0f} |")
    out.append("")

    out.append("### R@1 (%) over the beta grid (* = best beta for that scorer)\n")
    out.append("| Code | Scorer | beta | pooled i2t | pooled t2i | pooled mean | emotion mean | art style mean |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    cl = res["r1"][key("clip_only", "-", BETA_BASELINE)]
    out.append(f"| — | CLIP-only | {BETA_BASELINE:g} | {cl['pooled']['i2t']:.2f} | {cl['pooled']['t2i']:.2f} | "
               f"{cl['pooled']['mean']:.2f} | {cl['emotion']['mean']:.2f} | {cl['art_style']['mean']:.2f} |")
    for name in CODES:
        for scorer in ("naive", "oracle"):
            for b in BETAS:
                r = res["r1"][key(scorer, name, b)]
                star = "*" if res["best_beta"][name][scorer] == b else ""
                out.append(f"| {name} | {scorer} | {b:g}{star} | {r['pooled']['i2t']:.2f} | {r['pooled']['t2i']:.2f} | "
                           f"{r['pooled']['mean']:.2f} | {r['emotion']['mean']:.2f} | {r['art_style']['mean']:.2f} |")
    out.append("")

    out.append(f"### Headline R@1 (%, mean of directions, 95% bootstrap CI); chance {res['chance_r1']:.2f}\n")
    out.append("| Code | Scorer | beta | pooled | emotion | art style |")
    out.append("|---|---|---:|---:|---:|---:|")
    h = res["headline"]
    out.append(f"| — | CLIP-only | {BETA_BASELINE:g} | "
               + " | ".join(r1ci(h["clip_only"]["r1"][s]["mean"]) for s in SCOPES) + " |")
    base = res["r1"][key("naive", "R3", BETA_BASELINE)]
    out.append(f"| R3 | naive (baseline) | {BETA_BASELINE:g} | "
               + " | ".join(f"{base[s]['mean']:.2f}" for s in SCOPES) + " |")
    for name in CODES:
        for scorer in ("naive", "oracle"):
            x = h[name][scorer]
            out.append(f"| {name} | {scorer} | {x['beta']:g} | "
                       + " | ".join(r1ci(x["r1"][s]["mean"]) for s in SCOPES) + " |")
        for b, x in h[name]["oracle_null"].items():
            out.append(f"| {name} | oracle null | {b} | " + " | ".join(r1ci(x[s]["mean"]) for s in SCOPES) + " |")
    out.append("")

    comp_names = (("oracle_minus_R3_oracle", "oracle(code, best beta) − oracle(R3, best beta)"),
                  ("oracle_minus_R3_naive0.3", "oracle(code, best beta) − naive(R3, beta 0.3)"),
                  ("naive_minus_R3_naive0.3", "naive(code, best beta) − naive(R3, beta 0.3)"))
    for cname, what in comp_names:
        out.append(f"### {what} (paired R@1 points, 95% CI)\n")
        out.append("| Code | " + " | ".join(f"{s} {d}" for s in SCOPES for d in (*DIRECTIONS, "mean")) + " |")
        out.append("|---|" + "---:|" * (3 * len(SCOPES)))
        for name in CODES:
            block = res["comparisons"][name].get(cname)
            if block is None:
                continue
            out.append(f"| {name} | "
                       + " | ".join(ci(block[s][d]) for s in SCOPES for d in (*DIRECTIONS, "mean")) + " |")
        out.append("")
    cb = res["comparisons"]["clip_only"]["clip_only_minus_R3_naive0.3"]
    out.append("CLIP-only − naive(R3, 0.3), mean of directions: "
               + "; ".join(f"{s} {ci(cb[s]['mean'])}" for s in SCOPES) + "\n")

    out.append("### Condition-source alignment: AMI with the labels (fit rows with a partition label)\n")
    out.append("| Partition | AMI emotion | AMI art style | n rows | groups | label classes (emotion / style) |")
    out.append("|---|---:|---:|---:|---:|---|")
    for pname, p in res["ami"]["partitions"].items():
        e, a = p["emotion"], p["art_style"]
        out.append(f"| {pname} | {e['ami']:.4f} | {a['ami']:.4f} | {e['n_rows']:,} | {e['n_groups']} | "
                   f"{e['n_labels']} / {a['n_labels']} |")
    out.append(f"\nCaption residual: {res['ami']['caption_residual']}. {res['ami']['note']}.\n")

    out.append("### Sanity checks\n")
    s = res["sanity"]
    s1 = s["1_stage_d_reproduction"]
    for kind in ("naive", "oracle", "clip_only"):
        ours = {d: round(v, 2) for d, v in s1[kind]["pooled_r1"].items()}
        stored = {d: round(v, 2) for d, v in s1[kind]["stored_pooled_r1"].items()}
        out.append(f"- 1 ({kind}, R3, beta 0.3): identical-rank share {s1[kind]['identical_rank_share']:.4f}; "
                   f"pooled R@1 {ours} vs stored {stored}")
    out.append(f"- 1 asserted: {s1['asserted']}; passed: naive {s1.get('naive_passed', 'n/a')}, oracle "
               f"{s1.get('oracle_passed', 'n/a')}{'; ' + s1['note'] if 'note' in s1 else ''}")
    s2 = s["2_clip512_uniform_equals_clip_only"]
    out.append(f"- 2 clip512 uniform at beta 0 vs CLIP-only: identical-rank share {s2['identical_rank_share']:.4f}; "
               f"passed {s2['passed']}")
    s3 = s["3_oracle_null_near_chance"]
    cells = "; ".join(f"{k} {v['i2t']:.2f}/{v['t2i']:.2f}{'' if v['passed'] else ' FAIL'}"
                      for k, v in s3["cells"].items())
    out.append(f"- 3 oracle nulls within {NULL_TOLERANCE:g} points of chance ({s3['chance']:.2f}): passed "
               f"{s3['passed']} (asserted: {s3['asserted']}); {cells}")
    out.append("")
    m = res["meta"]
    out.append(f"Device {m['device']}, torch threads {m['torch_threads']}, total {m['seconds']:.0f} s. Timings (s): "
               + ", ".join(f"{k} {v:.1f}" for k, v in m["timings_seconds"].items()))
    print("\n".join(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--tables", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    device = torch.device(args.device)
    if args.tables:
        tables(RESULTS / f"{MODES['run'].prefix}_results.json")
        return
    mode = MODES["run" if args.run else "smoke"]
    run(mode, device)
    tables(RESULTS / f"{mode.prefix}_results.json")


if __name__ == "__main__":
    main()
