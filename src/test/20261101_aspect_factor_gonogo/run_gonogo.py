"""E3: the aspect-factor training grid, the seed-42 selection pick (CVPR plan Task 12), and the seed-43 GO test,
held-out-genre K8 test, ablation rows and value-sharing diagnostic (Task 13).

The rules are fixed in PREREGISTRATION.md in this folder, committed before any run. Run from the repo root:

  python src/test/20261101_aspect_factor_gonogo/run_gonogo.py --train A1 --seed 42 [--smoke]    # GPU (one run)
  flock -n -o -E 75 /tmp/gpu0.lock bash src/test/20261101_aspect_factor_gonogo/run_grid.sh      # the 8 grid runs
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python src/test/20261101_aspect_factor_gonogo/run_gonogo.py --select [--smoke]
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python src/test/20261101_aspect_factor_gonogo/run_gonogo.py --gonogo [--smoke]

Row scope: training reads scorer-train rows only (local rows 0..n-1 of artelingo_splits().scorer_train); evaluation
reads selection rows only (features and codes are NaN elsewhere, asserted); val and held rows are never read.
--select and --gonogo run on the CPU. --smoke trains 50 steps on Task 10's smoke bank into checkpoints/smoke/ and
results/smoke/, and evaluates Task 9's smoke episodes with the A1 smoke checkpoint standing in for every run.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.sparse import load_npz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402
from src.eval.aspect_episodes import (AspectEpisodes, concat_episodes, eligible_values,  # noqa: E402
                                      episodes_sha256)
from src.eval.aspect_metrics import METRICS, compare, per_anchor, summarize  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, agreement_term, cosine_scores, crossfit_lambda  # noqa: E402
from src.train.train_factors import (R3_CONFIG, encode_rows, load_factor_checkpoint,  # noqa: E402
                                     save_factor_checkpoint, train_factors)

E2 = ROOT / "src/test/20261031_pseudo_partitions/results"          # Task 10: partitions, graph, banks
E1 = ROOT / "src/test/20261030_aspect_baselines/results"           # Task 9: episodes, per-anchor arrays, codes
C0_HISTORY = ROOT / "src/test/20261016_factor_learning_grid/results/history_C0_seed42.json"
FULL_STEPS, SMOKE_STEPS = 2000, 50
N_PER_PAIR = 4096
C0_CELL = {"agreement_level": "pair", "lambda_condition": 0.0}    # CELLS["C0"] of the factor-learning grid
A1 = {"lambda_aspect": 1.0, "lambda_swap": 1.0}
RUNS = {                                                           # PREREGISTRATION.md §3: run -> (bank, changes)
    "A1": ("AIC", A1),
    "A2": ("AIC", {**A1, "num_factors": 64}),
    "A3": ("AIC", {**A1, "lambda_aspect": 3.0}),
    "A4": ("AIC", {**A1, "lambda_swap": 0.0}),
    "A5": ("AIC", {**A1, "aspect_beta": 0.0}),
    "A6": ("AIC", {**A1, "lambda_sparsity": 0.0, "lambda_decorrelation": 0.1}),
    "H1": ("AI", A1),
    "S1": ("IC", A1),
}
ELIGIBLE = ("A1", "A2", "A3", "A4", "A5", "A6")
PAIRS = (("emotion", "style"), ("emotion", "genre"), ("style", "genre"))
POOLED_ORDER = [f"{a}__{b}" for a, b in PAIRS]
FIELDS = ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")
CRITERION = ("maximize 0.5 * (R@1 point + condition gain point) of the cross-fitted scores on the pooled seed-42 "
             "selection episodes, among A1..A6; ties go to the lower run id")
GO_CANDIDATES = ("diag", "diag_relu", "bilinear", "kissme", "rca", "xing", "wang", "probe", "tip")
EMOTION_PAIRS = (0, 1)                                             # emotion__style, emotion__genre
GENRE_PAIRS = (1, 2)                                               # emotion__genre, style__genre
STRONG_GO_POINTS = 4.0
MIN_PAINTINGS = 30                                                 # value eligibility, as in the episodes
SHARE_FRACTION = 0.1                                               # value sharing: >= 10% of the factor's max
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0 = perf_counter()


def log(*a):
    print(f"[{perf_counter() - T0:7.1f}s]", *a, flush=True)


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_array(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def folders(smoke: bool) -> dict:
    sub = "smoke" if smoke else ""
    return {"ckpt": HERE / "checkpoints" / sub, "res": HERE / "results" / sub, "e2": E2 / sub, "e1": E1 / sub}


def run_config(run: str, seed: int, steps: int = FULL_STEPS):
    """The C0 recipe (== the factor-learning grid's cell_config("C0", seed, steps)) plus the run's aspect fields."""
    base = dataclasses.replace(R3_CONFIG, painting_batches=True, seed=seed, epochs=steps, **C0_CELL)
    return dataclasses.replace(base, **RUNS[run][1])


def check_base_config(seed: int, steps: int) -> None:
    """The base recipe must equal the config stored by the factor-learning grid's real C0 run (seed and steps aside)."""
    base = dataclasses.asdict(dataclasses.replace(R3_CONFIG, painting_batches=True, seed=seed, epochs=steps,
                                                  **C0_CELL))
    stored = json.loads(C0_HISTORY.read_text())["config"]
    diff = {k: (base[k], v) for k, v in stored.items() if k not in ("seed", "epochs") and base[k] != v}
    if diff:
        raise AssertionError(f"base config differs from the grid's C0 run: {diff}")


def checkpoint_path(run: str, seed: int, smoke: bool) -> Path:
    """Smoke mode: the A1 smoke checkpoint stands in for every run."""
    return folders(smoke)["ckpt"] / (f"A1_seed{seed}.pt" if smoke else f"{run}_seed{seed}.pt")


def load_bank(e2: Path, name: str, record: dict) -> tuple[AspectEpisodes, str]:
    path = e2 / f"bank_{name}.npz"
    sha = sha_file(path)
    if sha != record["sha256"][f"bank_{name}.npz"]:
        raise AssertionError(f"{path}: SHA-256 differs from E2's build_record.json")
    z = np.load(path)
    bank = AspectEpisodes(str(z["aspect_a"]), str(z["aspect_b"]), *(z[f].astype(np.int64) for f in FIELDS))
    return bank, sha


def code_stats(img_codes: np.ndarray, txt_codes: np.ndarray) -> dict:
    out = {}
    for side, c in (("img", img_codes), ("txt", txt_codes)):
        out[side] = {"active_fraction": float((c > 0).mean()), "dead_factors": int(((c > 0).sum(0) == 0).sum()),
                     "mean_active_per_row": float((c > 0).sum(1).mean())}
    return out


# ---------------------------------------------------------------- --train


def train(run: str, seed: int, smoke: bool) -> None:
    f = folders(smoke)
    steps = SMOKE_STEPS if smoke else FULL_STEPS
    name = f"{run}_seed{seed}"
    ckpt, hist = f["ckpt"] / f"{name}.pt", f["res"] / f"history_{name}.json"
    if not smoke:
        for path in (ckpt, hist, f["res"] / f"failed_{name}.json"):
            if path.exists():
                raise FileExistsError(f"{path} exists: it may be evidence behind the go/no-go. Refusing to overwrite.")
    record_path = f["e2"] / "build_record.json"
    if not record_path.exists():
        raise FileNotFoundError(f"{record_path} missing: wait for E2's build to finish before training")
    record = json.loads(record_path.read_text())
    f["ckpt"].mkdir(parents=True, exist_ok=True)
    f["res"].mkdir(parents=True, exist_ok=True)

    t_load = perf_counter()
    data = load_artelingo()
    sp = artelingo_splits(data)
    st = np.asarray(sp.scorer_train)
    n = len(st)
    parts = np.load(f["e2"] / "partitions.npz")
    local_groups = parts["local_groups"].astype(np.int64)
    if not np.array_equal(local_groups, np.unique(sp.groups[st], return_inverse=True)[1]):
        raise AssertionError("partitions.npz local_groups != np.unique(groups[scorer_train], return_inverse=True)[1]")
    if sha_array(local_groups) != record["sha256"]["local_groups"]:
        raise AssertionError("local_groups SHA-256 differs from E2's build_record.json")
    graph_path = f["e2"] / "graph.npz"
    graph_sha = sha_file(graph_path)
    if graph_sha != record["sha256"]["graph.npz"]:
        raise AssertionError("graph.npz SHA-256 differs from E2's build_record.json")
    graph = load_npz(graph_path).tocsr()
    if graph.shape != (n, n):
        raise AssertionError(f"graph shape {graph.shape} != ({n}, {n})")
    bank_name = RUNS[run][0]
    bank, bank_sha = load_bank(f["e2"], bank_name, record)
    rows = bank.rows()
    if rows.min() < 0 or rows.max() >= n:
        raise AssertionError(f"bank {bank_name} rows leave 0..{n - 1}: [{rows.min()}, {rows.max()}]")
    check_base_config(seed, steps)
    config = run_config(run, seed, steps)
    load_s = perf_counter() - t_load
    log(f"{name}: loaded {n} scorer-train rows, graph {graph.shape}, bank {bank_name} ({len(bank.anchor)} episodes) "
        f"in {load_s:.1f}s; config {dataclasses.asdict(config)}")

    history: dict = {}
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = perf_counter()
    model, img_codes, txt_codes = train_factors(
        data.img_features[st], data.txt_features[st], graph, config, device=DEVICE, group_ids=local_groups,
        aspect_bank=bank, history=history, log_every=50)
    train_s = perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else 0.0
    out = {"run": run, "seed": seed, "smoke": smoke, "steps": steps, "bank": bank_name,
           "bank_sha256": bank_sha, "graph_sha256": graph_sha, "build_record_sha256": sha_file(record_path),
           "config": dataclasses.asdict(config), "load_s": load_s, "train_s": train_s,
           "wall_s": perf_counter() - t_load, "peak_gpu_gib": peak, "device": DEVICE,
           "script_sha256": sha_file(Path(__file__)), "history": history}
    if not (np.isfinite(img_codes).all() and np.isfinite(txt_codes).all()):
        (f["res"] / f"failed_{name}.json").write_text(json.dumps({**out, "failure": "non-finite codes"}, indent=1))
        raise SystemExit(f"{name}: non-finite codes; recorded as failed (not eligible, PREREGISTRATION.md §3)")
    save_factor_checkpoint(model, config, ckpt)
    out.update(checkpoint=str(ckpt.relative_to(ROOT)), checkpoint_sha256=sha_file(ckpt),
               code_stats_scorer_train=code_stats(img_codes, txt_codes))
    hist.write_text(json.dumps(out, indent=1))
    log(f"{name}: {steps} steps in {train_s:.1f}s (load {load_s:.1f}s), peak GPU {peak:.2f} GiB, "
        f"codes {out['code_stats_scorer_train']} -> {ckpt.relative_to(ROOT)}")


# ---------------------------------------------------------------- evaluation helpers


class EvalContext:
    """Selection-masked features, the pooled episodes of one episode seed (SHA-256 checked) and their cosine."""

    def __init__(self, seed: int, smoke: bool):
        f = folders(smoke)
        self.data = load_artelingo()
        sp = artelingo_splits(self.data)
        self.groups = sp.groups
        self.selection = np.asarray(sp.selection)
        self.in_sel = np.zeros(len(self.groups), dtype=bool)
        self.in_sel[self.selection] = True
        self.img = self.masked(self.data.img_features)
        self.txt = self.masked(self.data.txt_features)
        z = np.load(f["e1"] / f"episodes_seed{seed}.npz")
        self.baselines = json.loads((f["e1"] / f"baselines_seed{seed}.json").read_text())
        if list(z["pair_order"]) != POOLED_ORDER or self.baselines["pair_order"] != POOLED_ORDER:
            raise AssertionError(f"pair order must be {POOLED_ORDER}")
        parts, self.shas = [], {}
        for a, b in PAIRS:
            ep = AspectEpisodes(a, b, *(z[f"{a}__{b}__{k}"].astype(np.int64) for k in FIELDS))
            self.shas[f"{a}__{b}"] = episodes_sha256(ep)
            if self.shas[f"{a}__{b}"] != self.baselines["episodes_sha256"][f"{a}__{b}"]:
                raise AssertionError(f"episodes_seed{seed} {a}__{b}: SHA-256 differs from baselines_seed{seed}.json")
            if not self.in_sel[ep.rows()].all():
                raise AssertionError(f"episodes_seed{seed} {a}__{b}: a row outside selection")
            parts.append(ep)
        self.n_per_pair = len(parts[0].anchor)
        if any(len(p.anchor) != self.n_per_pair for p in parts) or self.n_per_pair != self.baselines["n_per_pair"]:
            raise AssertionError("episode counts differ between pairs or from the baselines record")
        if not smoke and self.n_per_pair != N_PER_PAIR:
            raise AssertionError(f"expected {N_PER_PAIR} episodes per pair, got {self.n_per_pair}")
        self.pooled = concat_episodes(parts)
        self.n = len(self.pooled.anchor)
        self.pair_index = np.repeat(np.arange(len(PAIRS)), self.n_per_pair)
        self.anchor_group = self.groups[self.pooled.anchor]
        self.parity = np.arange(self.n) % 2
        self.cos = cosine_scores(EvalInputs(self.img, self.txt), self.pooled)
        self.seed, self.smoke = seed, smoke

    def masked(self, values: np.ndarray) -> np.ndarray:
        out = np.full(values.shape, np.nan, dtype=np.float32)
        out[self.selection] = values[self.selection]
        if not (np.isnan(out[~self.in_sel]).all() and np.isfinite(out[self.in_sel]).all()):
            raise AssertionError("selection masking failed (non-NaN outside selection or non-finite inside)")
        return out

    def encode(self, ckpt: Path) -> tuple[np.ndarray, np.ndarray]:
        """Selection-row codes of a checkpoint (CPU), NaN elsewhere (asserted)."""
        model, _ = load_factor_checkpoint(ckpt, device="cpu")
        ic, tc = encode_rows(model, self.data.img_features, self.data.txt_features, rows=self.selection, device="cpu")
        out = []
        for codes in (ic, tc):
            full = np.full((len(self.groups), codes.shape[1]), np.nan, dtype=np.float32)
            full[self.selection] = codes
            if not (np.isnan(full[~self.in_sel]).all() and np.isfinite(full[self.in_sel]).all()):
                raise AssertionError(f"{ckpt.name}: codes must be finite on selection rows and NaN elsewhere")
            out.append(full)
        return out[0], out[1]

    def score(self, img_codes, txt_codes, uniform: bool) -> tuple[dict, dict]:
        """Cross-fitted (parity) z-fusion of cosine and the agreement term (or its uniform-weight control)."""
        term = agreement_term(EvalInputs(self.img, self.txt, img_codes, txt_codes), self.pooled, uniform=uniform)
        fused, picks = crossfit_lambda(self.cos, term, self.parity)
        return per_anchor(fused), {str(k): (v if np.isfinite(v) else "inf") for k, v in picks.items()}

    def summary(self, pa: dict, mask=None) -> dict:
        mask = np.ones(self.n, dtype=bool) if mask is None else mask
        return summarize({m: pa[m][mask] for m in METRICS}, self.anchor_group[mask])

    def per_pair(self, pa: dict) -> dict:
        return {name: self.summary(pa, self.pair_index == i) for i, name in enumerate(POOLED_ORDER)}


def cell(x: dict) -> str:
    return f"{x['point']:6.2f} [{x['ci95'][0]:6.2f},{x['ci95'][1]:6.2f}]"


def assert_finite_tree(obj, where: str = "") -> None:
    """Every float in a result tree must be finite (lambda picks are stored as numbers or the string 'inf')."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_finite_tree(v, f"{where}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_finite_tree(v, f"{where}[{i}]")
    elif isinstance(obj, float) and not np.isfinite(obj):
        raise AssertionError(f"non-finite number at {where}")


# ---------------------------------------------------------------- --select


def select(smoke: bool) -> None:
    f = folders(smoke)
    out_sel, out_pick = f["res"] / "select_seed42.json", f["res"] / "picked.json"
    if not smoke and (out_sel.exists() or out_pick.exists()):
        raise FileExistsError(f"{out_pick} or {out_sel} exists: the pick is frozen. Refusing to overwrite.")
    failed = {run for run in RUNS if not smoke and not checkpoint_path(run, 42, smoke).exists()
              and (f["res"] / f"failed_{run}_seed42.json").exists()}
    missing = [run for run in RUNS if run not in failed and not checkpoint_path(run, 42, smoke).exists()]
    if missing:
        raise FileNotFoundError(f"checkpoints missing (and no failed record) for {missing}: the grid is not complete")
    f["res"].mkdir(parents=True, exist_ok=True)
    ctx = EvalContext(42, smoke)
    log(f"seed-42 selection episodes: {ctx.n} pooled ({ctx.n_per_pair} per pair), SHA-256s match Task 9")
    rows, arrays = {}, {"anchor_group": ctx.anchor_group, "pair_index": ctx.pair_index}
    for run in RUNS:
        ckpt = checkpoint_path(run, 42, smoke)
        row = {"eligible": run in ELIGIBLE, "bank": RUNS[run][0]}
        if smoke:
            row["smoke_stand_in"] = "A1 smoke checkpoint (50 steps) stands in for this run"
        if run in failed:
            rows[run] = {**row, "status": "failed (non-finite codes)", "eligible": False}
            log(f"{run}: failed run, not eligible")
            continue
        ic, tc = ctx.encode(ckpt)
        pa, picks = ctx.score(ic, tc, uniform=False)
        pa_u, picks_u = ctx.score(ic, tc, uniform=True)
        overall = ctx.summary(pa)
        row.update(status="ok", checkpoint=str(ckpt.relative_to(ROOT)), checkpoint_sha256=sha_file(ckpt),
                   overall=overall, per_pair=ctx.per_pair(pa), lambda_picks=picks,
                   uniform_overall=ctx.summary(pa_u), uniform_lambda_picks=picks_u,
                   criterion=0.5 * (overall["r1"]["point"] + overall["gain"]["point"]))
        rows[run] = row
        for m in METRICS:
            arrays[f"{run}__{m}"], arrays[f"{run}_uniform__{m}"] = pa[m], pa_u[m]
        log(f"{run}: R@1 {cell(overall['r1'])} gain {cell(overall['gain'])} criterion {row['criterion']:.3f} "
            f"picks {picks}")

    scored = [r for r in ELIGIBLE if rows[r].get("status") == "ok"]
    picked = None
    if scored:
        best = max(rows[r]["criterion"] for r in scored)
        picked = next(r for r in ELIGIBLE if r in scored and rows[r]["criterion"] == best)   # ties: lower run id
    record = {"episodes_seed": 42, "smoke": smoke, "n_per_pair": ctx.n_per_pair, "episodes_sha256": ctx.shas,
              "criterion": CRITERION, "picked": picked, "runs": rows, "script_sha256": sha_file(Path(__file__))}
    assert_finite_tree(record)
    np.savez(f["res"] / "per_anchor_select_seed42.npz", **arrays)
    out_sel.write_text(json.dumps(record, indent=1))
    table = {r: {"eligible": rows[r]["eligible"], "status": rows[r]["status"],
                 **({"r1": rows[r]["overall"]["r1"]["point"], "gain": rows[r]["overall"]["gain"]["point"],
                     "criterion": rows[r]["criterion"]} if rows[r]["status"] == "ok" else {})} for r in RUNS}
    pick = {"run": picked, "criterion": CRITERION, "criterion_value": rows[picked]["criterion"] if picked else None,
            "checkpoint": rows[picked]["checkpoint"] if picked else None,
            "checkpoint_sha256": rows[picked]["checkpoint_sha256"] if picked else None,
            "smoke": smoke, "table": table}
    out_pick.write_text(json.dumps(pick, indent=1))

    print(f"\n{'run':4s} {'elig':5s} {'R@1':24s} {'gain':24s} {'other':>6s} {'swap':>6s} {'crit':>7s}  "
          f"{'unif R@1':>8s} {'unif gain':>9s}  lambda picks")
    for run in RUNS:
        r = rows[run]
        if r["status"] != "ok":
            print(f"{run:4s} {str(r['eligible']):5s} {r['status']}")
            continue
        o, u = r["overall"], r["uniform_overall"]
        print(f"{run:4s} {str(r['eligible']):5s} {cell(o['r1'])} {cell(o['gain'])} {o['other']['point']:6.2f} "
              f"{o['swap']['point']:6.2f} {r['criterion']:7.3f}  {u['r1']['point']:8.2f} {u['gain']['point']:9.2f}  "
              f"{r['lambda_picks']}")
    note = " (SMOKE: every run is the A1 smoke checkpoint; the tie goes to A1)" if smoke else ""
    print(f"\nPICKED: {picked}{note}")
    log(f"wrote {out_sel.relative_to(ROOT)} and {out_pick.relative_to(ROOT)}")


# ---------------------------------------------------------------- --gonogo (Task 13; PREREG §6 to §8, addendum)

K8_MEANING = ("a K8 pass means the mechanism reaches genre without any genre label or genre-matched partition, not "
              "that genre was unseen in training: the AI bank's image partition carries genre (E2 AMI with genre "
              "0.397, against 0.161 for the caption partition H1 leaves out; PREREGISTRATION.md addendum A3)")
SHARE_NOTE = ("share/shared/dead: the pre-registered value-sharing share (§8), disclosed as blind (one factor per "
              "value also scores 1.00; addendum A1). eta2_mean_live and spread_S (with its floor 1/V) are the "
              "descriptive measures added by addendum A1. Selection rows labelled with a value that has >= 30 "
              "selection paintings.")
CONTEXT_SCORERS = (*GO_CANDIDATES, "SE_uniform")                    # addendum A5


def go_baseline(record42: dict) -> tuple[str, dict]:
    """The first entry of Task 9's seed-42 GO-bar ranking, re-checked against the per-scorer points of that file."""
    ranking = record42["go_bar_ranking"]
    if sorted(g["scorer"] for g in ranking) != sorted(GO_CANDIDATES):
        raise AssertionError(f"GO-bar ranking must cover exactly {GO_CANDIDATES}")
    means = {s: 0.5 * (record42["scorers"][s]["overall"]["r1"]["point"]
                       + record42["scorers"][s]["overall"]["gain"]["point"]) for s in GO_CANDIDATES}
    name = ranking[0]["scorer"]
    if means[name] != max(means.values()) or ranking[0]["mean_r1_gain"] != means[name]:
        raise AssertionError(f"go_bar_ranking[0] = {name} is not the top of {means}")
    return name, means


def subset(pa: dict, mask: np.ndarray) -> dict:
    return {m: pa[m][mask] for m in METRICS}


def paired(pa: dict, pb: dict, clusters, mask=None) -> dict:
    """compare() on R@1 and condition gain; beats = both CI lower bounds above 0."""
    if mask is not None:
        pa, pb, clusters = subset(pa, mask), subset(pb, mask), clusters[mask]
    r1, gain = compare(pa, pb, clusters, "r1"), compare(pa, pb, clusters, "gain")
    return {"r1": r1, "gain": gain, "beats": bool(r1["ci95"][0] > 0 and gain["ci95"][0] > 0)}


def factor_measures(codes: np.ndarray, lab: np.ndarray, rows: np.ndarray, values: list) -> dict:
    """One model, modality and aspect: the pre-registered share (§8, blind) and, from addendum A1, the mean eta^2 over
    live factors and the value spread S = sum_f eta2_f PR_f / sum_f eta2_f, PR_f = (sum_v e_vf)^2 / (V sum_v e_vf^2),
    e_vf = m[v, f] - min_v m[v, f]. ``rows`` are the labelled rows of the aspect's eligible values."""
    x, y = codes[rows].astype(np.float64), lab[rows]
    means = np.stack([x[y == v].mean(0) for v in values])                       # m[v, f], (V, L)
    counts = np.array([(y == v).sum() for v in values], dtype=np.float64)
    top = means.max(0)
    live = top > 0
    shared = live & (2 * (means >= SHARE_FRACTION * top).sum(0) >= len(values))
    grand = x.mean(0)
    total = ((x - grand) ** 2).sum(0)
    between = (counts[:, None] * (means - grand) ** 2).sum(0)
    varying = total > 0
    eta2 = np.zeros(len(top))
    eta2[varying] = between[varying] / total[varying]
    e = means - means.min(0)
    sq = (e ** 2).sum(0)
    pr = np.zeros(len(top))
    pr[sq > 0] = e.sum(0)[sq > 0] ** 2 / (len(values) * sq[sq > 0])
    return {"share": float(shared.mean()), "shared": int(shared.sum()), "dead": int((~live).sum()),
            "factors": int(len(top)),
            "eta2_mean_live": float(eta2[varying].mean()) if varying.any() else None,
            "eta2_live_factors": int(varying.sum()),
            "spread_S": float((eta2 * pr).sum() / eta2.sum()) if eta2.sum() > 0 else None,
            "spread_floor_1_over_V": 1.0 / len(values)}


def value_sharing(codes: dict, labels: dict, groups: np.ndarray, selection: np.ndarray) -> dict:
    """Per aspect, model and modality on selection rows: factor_measures (PREREGISTRATION.md §8 and addendum A1)."""
    out = {"note": SHARE_NOTE}
    for aspect in ("emotion", "style", "genre"):
        lab = labels[aspect]
        labelled = selection[lab[selection] >= 0]
        values = eligible_values(lab, groups, labelled, MIN_PAINTINGS)
        rows = labelled[np.isin(lab[labelled], values)]
        out[aspect] = {"values": len(values), "rows": int(len(rows)), "models": {}}
        for model, (img_codes, txt_codes) in codes.items():
            entry = {side: factor_measures(c, lab, rows, values) for side, c in (("img", img_codes),
                                                                                 ("txt", txt_codes))}
            entry["mean_share"] = 0.5 * (entry["img"]["share"] + entry["txt"]["share"])
            out[aspect]["models"][model] = entry
    return out


def run_status(run: str, smoke: bool) -> str:
    """'ok' (checkpoint present) or 'failed' (no checkpoint, a failed record; addendum A4). A checkpoint missing
    without a failed record means the grid is incomplete: raise before any work."""
    if checkpoint_path(run, 42, smoke).exists():
        return "ok"
    if not smoke and (folders(smoke)["res"] / f"failed_{run}_seed42.json").exists():
        return "failed"
    raise FileNotFoundError(f"{checkpoint_path(run, 42, smoke)} missing without a failed record: grid incomplete")


def gonogo(smoke: bool) -> None:
    f = folders(smoke)
    out_json, out_npz = f["res"] / "gonogo.json", f["res"] / "per_anchor_gonogo_seed43.npz"
    if not smoke and (out_json.exists() or out_npz.exists()):
        raise FileExistsError(f"{out_json} exists: the GO test runs once. Refusing to overwrite.")
    pick_path = f["res"] / "picked.json"
    pick = json.loads(pick_path.read_text())
    if pick["smoke"] != smoke:
        raise AssertionError(f"picked.json smoke={pick['smoke']} but this run has smoke={smoke}")
    run = pick["run"]
    if run is None:
        raise SystemExit("picked.json names no run (every A run failed): GO is missed by PREREGISTRATION.md §3")
    ok = [r for r in ELIGIBLE if pick["table"][r]["status"] == "ok"]
    best = max(pick["table"][r]["criterion"] for r in ok)
    if run != next(r for r in ok if pick["table"][r]["criterion"] == best):
        raise AssertionError(f"picked.json names {run}, but its own table picks another run by the rule")
    picked_ckpt = ROOT / pick["checkpoint"]
    if picked_ckpt != checkpoint_path(run, 42, smoke):
        raise AssertionError(f"picked.json checkpoint {picked_ckpt} is not {checkpoint_path(run, 42, smoke)}")
    if sha_file(picked_ckpt) != pick["checkpoint_sha256"]:
        raise AssertionError(f"{picked_ckpt}: SHA-256 differs from picked.json; the pick is frozen")
    status = {r: ("ok" if r == run else run_status(r, smoke)) for r in ("A1", "H1", "S1")}
    e1 = f["e1"]
    inputs = {name: {"path": str((e1 / name).relative_to(ROOT)), "sha256": sha_file(e1 / name)}
              for name in ("episodes_seed43.npz", "per_anchor_seed43.npz", "baselines_seed42.json",
                           "baselines_seed43.json")}
    inputs["picked.json"] = {"path": str(pick_path.relative_to(ROOT)), "sha256": sha_file(pick_path)}
    go_name, go_means = go_baseline(json.loads((e1 / "baselines_seed42.json").read_text()))
    record43 = json.loads((e1 / "baselines_seed43.json").read_text())

    ctx = EvalContext(43, smoke)
    log(f"seed-43 test episodes: {ctx.n} pooled ({ctx.n_per_pair} per pair), SHA-256s match Task 9; "
        f"run status {status}")
    t9 = np.load(e1 / "per_anchor_seed43.npz")
    for key in t9.files:
        if len(t9[key]) != ctx.n:
            raise AssertionError(f"per_anchor_seed43.npz[{key}] has {len(t9[key])} rows, the episodes have {ctx.n}")
    if not (np.array_equal(t9["anchor_group"], ctx.anchor_group) and np.array_equal(t9["pair_index"], ctx.pair_index)):
        raise AssertionError("per_anchor_seed43.npz anchor_group / pair_index differ from the episodes")

    def t9_pa(scorer):
        return {m: t9[f"{scorer}__{m}"].astype(np.float64) for m in METRICS}

    cos_pa = per_anchor(ctx.cos)
    for m in METRICS:
        if not np.array_equal(cos_pa[m], t9[f"cosine__{m}"]):
            raise AssertionError(f"recomputed cosine {m} differs from Task 9's seed-43 array: misaligned episodes")
    if not (cos_pa["gain"] == 0).all():
        raise AssertionError("cosine condition gain must be exactly 0")
    clusters = ctx.anchor_group
    pas, picks, ckpts, codes = {}, {}, {}, {}

    def score(role: str, rid: str, uniform_too: bool) -> None:
        path = checkpoint_path(rid, 42, smoke)
        ic, tc = ctx.encode(path)
        codes[role] = (ic, tc)
        pas[role], picks[role] = ctx.score(ic, tc, uniform=False)
        if uniform_too:
            pas[f"{role}_uniform"], picks[f"{role}_uniform"] = ctx.score(ic, tc, uniform=True)
        ckpts[role] = {"run": rid, "status": "ok", "checkpoint": str(path.relative_to(ROOT)), "sha256": sha_file(path)}
        if smoke:
            ckpts[role]["smoke_stand_in"] = "A1 smoke checkpoint (50 steps) stands in for this run"
        log(f"{role} ({rid}): scored, lambda picks {picks[role]}")

    # ---- GO (PREREGISTRATION.md §6), computed before anything that a failed H1, S1 or A1 could affect
    score("picked", run, uniform_too=True)
    comparators = {"backbone_only": ("cosine", cos_pa), "go_baseline": (go_name, t9_pa(go_name)),
                   "uniform_control": (f"{run}_uniform", pas["picked_uniform"])}
    go = {"picked": run, "picked_overall": ctx.summary(pas["picked"]), "picked_per_pair": ctx.per_pair(pas["picked"]),
          "picked_lambda_picks": picks["picked"], "uniform_lambda_picks": picks["picked_uniform"], "comparators": {},
          "multiplicity": "intersection-union test over six one-sided tests; no adjustment (addendum A8)"}
    for key, (name, pa) in comparators.items():
        go["comparators"][key] = {"name": name, "overall": ctx.summary(pa), **paired(pas["picked"], pa, clusters)}
    bo = go["comparators"]["backbone_only"]
    go["GO"] = all(c["beats"] for c in go["comparators"].values())
    go["strong_go_r1_condition"] = bool(bo["r1"]["point"] >= STRONG_GO_POINTS)
    go["strong_go_gain_condition"] = bool(bo["gain"]["point"] >= STRONG_GO_POINTS)
    go["strong_go"] = bool(go["GO"] and go["strong_go_r1_condition"] and go["strong_go_gain_condition"])
    go["backbone_only_r1_measured"] = bo["overall"]["r1"]["point"]                          # addendum A7

    # ---- the other trained runs (addendum A4: a failed run is skipped, never fatal)
    for rid in ("A1", "H1", "S1"):
        if rid == run:
            for table in (pas, picks, codes, ckpts):
                table[rid] = table["picked"]
        elif status[rid] == "ok":
            score(rid, rid, uniform_too=(rid == "H1"))
        else:
            ckpts[rid] = {"run": rid, "status": "failed to train",
                          "failed_record": str((f["res"] / f"failed_{rid}_seed42.json").relative_to(ROOT))}

    # ---- K8 (PREREGISTRATION.md §7, addendum A3/A4)
    genre = np.isin(ctx.pair_index, GENRE_PAIRS)
    k8 = {"run": "H1", "bank": RUNS["H1"][0], "pairs": [POOLED_ORDER[i] for i in GENRE_PAIRS],
          "n_episodes": int(genre.sum()), "meaning": K8_MEANING}
    if status["H1"] == "ok":
        k8_cmp = paired(pas["H1"], pas["H1_uniform"], clusters, genre)
        k8.update(status="tested", gain_vs_uniform=k8_cmp["gain"], r1_vs_uniform=k8_cmp["r1"],
                  H1_genre_pairs=ctx.summary(pas["H1"], genre),
                  H1_uniform_genre_pairs=ctx.summary(pas["H1_uniform"], genre),
                  per_pair_gain_vs_uniform={POOLED_ORDER[i]: paired(pas["H1"], pas["H1_uniform"], clusters,
                                                                    ctx.pair_index == i)["gain"] for i in GENRE_PAIRS},
                  H1_all_pairs=ctx.summary(pas["H1"]), lambda_picks=picks["H1"],
                  uniform_lambda_picks=picks["H1_uniform"], K8=bool(k8_cmp["gain"]["ci95"][0] > 0))
    else:
        k8.update(status="untestable: H1 failed to train; counts as K8 not holding (addendum A4)", K8=False)

    # ---- ablation rows (descriptive; PREREGISTRATION.md §8 and addendum A2)
    emotion = np.isin(ctx.pair_index, EMOTION_PAIRS)
    have = {r: r in pas for r in ("A1", "H1", "S1")}
    ablation = {"pairs": [POOLED_ORDER[i] for i in EMOTION_PAIRS], "n_episodes": int(emotion.sum()),
                "note": "descriptive; S1_vs_A1 is the clean ablation (banks IC vs AIC, same settings; addendum A2); "
                        "SE, C0 and R3 are Task 9's seed-43 cross-fitted per-anchor arrays", "omitted": []}
    for key, a, b, mask, why in (("S1_vs_A1_emotion_pairs", "S1", "A1", emotion, "S1 or A1 failed to train"),
                                 ("S1_vs_picked_emotion_pairs", "S1", "picked", emotion, "S1 failed to train"),
                                 ("H1_vs_A1_genre_pairs", "H1", "A1", genre, "H1 or A1 failed to train")):
        if a in pas and b in pas:
            ablation[key] = paired(pas[a], pas[b], clusters, mask)
        else:
            ablation["omitted"].append(f"{key}: {why}")
    rows = {f"picked ({run})": pas["picked"], **{r: pas[r] for r in ("A1", "S1") if have[r] and r != run},
            **{s: t9_pa(s) for s in ("SE", "C0", "R3")}}
    ablation["rows"] = {name: {"emotion_pairs": ctx.summary(pa, emotion), "all_pairs": ctx.summary(pa)}
                        for name, pa in rows.items()}
    ablation["lambda_picks"] = {r: picks[r] for r in ("A1", "S1") if have[r]}

    # ---- best-baseline context (descriptive; addendum A5)
    context = {"note": "descriptive; the GO baseline stays the seed-42 pick (§6)",
               "seed43_ranking": record43["go_bar_ranking"],
               "picked_vs": {s: {"overall": ctx.summary(t9_pa(s)), **paired(pas["picked"], t9_pa(s), clusters)}
                             for s in CONTEXT_SCORERS}}

    # ---- value sharing (diagnostic, selection-row labels; addendum A1)
    t9_codes = {}
    for s in ("SE", "C0", "R3"):
        path = E1 / f"codes_{s}.npz"                      # Task 9's cache (also used in smoke mode)
        z = np.load(path)
        codes[s] = (ctx.masked(z["img"]), ctx.masked(z["txt"]))
        t9_codes[s] = {"path": str(path.relative_to(ROOT)), "sha256": sha_file(path)}
    share_models = {f"picked ({run})": codes["picked"], **{r: codes[r] for r in ("H1", "S1") if have[r]},
                    **{s: codes[s] for s in ("SE", "C0", "R3")}}
    sharing = value_sharing(share_models, artelingo_aspect_labels(ctx.data), ctx.groups, ctx.selection)

    branch = ("branch 1 (GO: method paper)" if go["GO"] else
              "GO missed: branch 2 if the Task 14 MLLM probe works, else branch 3 (the user decides on Oct 9)")
    verdict = {"picked": run, "GO": go["GO"], "strong_go": go["strong_go"], "K8": k8["K8"], "k8_status": k8["status"],
               "go_baseline": go_name, "decision_map": branch}
    record = {"episodes_seed": 43, "smoke": smoke, "n_per_pair": ctx.n_per_pair, "episodes_sha256": ctx.shas,
              "inputs": inputs, "picked_json": pick, "go_baseline": {"name": go_name, "seed42_mean_r1_gain": go_means},
              "run_status": status, "checkpoints": ckpts, "task9_codes": t9_codes, "go": go, "k8": k8,
              "ablation": ablation, "baseline_context": context, "value_sharing": sharing, "verdict": verdict,
              "script_sha256": sha_file(Path(__file__))}
    assert_finite_tree(record)
    arrays = {"anchor_group": ctx.anchor_group, "pair_index": ctx.pair_index}
    for role, pa in pas.items():
        if role in ("A1", "H1", "S1") and role == run:
            continue                                                    # same array as "picked"
        for m in METRICS:
            arrays[f"{role}__{m}"] = pa[m]
    np.savez(out_npz, **arrays)
    out_json.write_text(json.dumps(record, indent=1))

    def line(label, c):
        return (f"  {label:34s} R@1 {c['r1']['point']:+6.2f} [{c['r1']['ci95'][0]:+6.2f},{c['r1']['ci95'][1]:+6.2f}]"
                f"   gain {c['gain']['point']:+6.2f} [{c['gain']['ci95'][0]:+6.2f},{c['gain']['ci95'][1]:+6.2f}]"
                f"   beats: {'yes' if c['beats'] else 'no'}")

    po = go["picked_overall"]
    print("\n" + "=" * 34 + " E3 GO / NO-GO (seed-43 selection episodes) " + "=" * 34)
    if smoke:
        print("SMOKE RUN: every model role is the A1 smoke checkpoint (50 steps); these numbers mean nothing.")
    print(f"picked run: {run} (seed-42 criterion {pick['criterion_value']:.3f}); {ctx.n} episodes, "
          f"{len(np.unique(clusters))} anchor paintings; run status {status}")
    print(f"picked: R@1 {cell(po['r1'])}  gain {cell(po['gain'])}  swap {po['swap']['point']:.2f}  "
          f"(backbone-only R@1 {go['backbone_only_r1_measured']:.2f})")
    print("picked minus comparator (clustered bootstrap, 5,000 resamples):")
    for key, label in (("backbone_only", "backbone-only (cosine)"), ("go_baseline", f"GO baseline ({go_name})"),
                       ("uniform_control", f"uniform-weight control ({run})")):
        print(line(label, go["comparators"][key]))
    print(f"GO: {go['GO']}    strong GO: {go['strong_go']} (R@1 >= +4: {go['strong_go_r1_condition']}, "
          f"gain >= +4: {go['strong_go_gain_condition']})")
    if k8["status"] == "tested":
        g = k8["gain_vs_uniform"]
        print(f"K8 (H1 on {k8['n_episodes']} genre-pair episodes, gain vs H1 uniform): {g['point']:+.2f} "
              f"[{g['ci95'][0]:+.2f},{g['ci95'][1]:+.2f}] -> K8 {'holds' if k8['K8'] else 'fails'}")
        print("  (a pass means no genre label or genre-matched partition was used, not that genre was unseen)")
    else:
        print(f"K8: {k8['status']}")
    print("ablation and context (descriptive):")
    for key in ("S1_vs_A1_emotion_pairs", "S1_vs_picked_emotion_pairs", "H1_vs_A1_genre_pairs"):
        if key in ablation:
            print(line(key, ablation[key]))
    for item in ablation["omitted"]:
        print(f"  omitted: {item}")
    print(f"  seed-43 ranking of the GO-bar candidates: "
          + ", ".join(f"{g['scorer']} {g['mean_r1_gain']:.2f}" for g in record43["go_bar_ranking"][:3]) + ", ...")
    for s in CONTEXT_SCORERS:
        print(line(f"picked minus {s}", context["picked_vs"][s]))
    print("value sharing: pre-registered share (blind) | value spread S img/txt (floor 1/V):")
    for aspect in ("emotion", "style", "genre"):
        entry = sharing[aspect]
        print(f"  {aspect:8s} ({entry['values']:2d} values, 1/V {1 / entry['values']:.3f}): " + "; ".join(
            f"{m} {e['mean_share']:.2f} | {e['img']['spread_S'] or float('nan'):.2f}/"
            f"{e['txt']['spread_S'] or float('nan'):.2f}" for m, e in entry["models"].items()))
    print(f"decision map: {branch}")
    print("=" * 112)
    log(f"wrote {out_json.relative_to(ROOT)} and {out_npz.relative_to(ROOT)}")


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", choices=sorted(RUNS), metavar="RUN", help="train one grid run (A1..A6, H1, S1)")
    mode.add_argument("--select", action="store_true", help="seed-42 selection table and the pick")
    mode.add_argument("--gonogo", action="store_true", help="seed-43 GO test, K8, ablation rows, value sharing")
    ap.add_argument("--seed", type=int, default=42, help="model seed for --train")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    if args.train:
        train(args.train, args.seed, args.smoke)
    elif args.select:
        select(args.smoke)
    else:
        gonogo(args.smoke)
    log("done")


if __name__ == "__main__":
    main()
