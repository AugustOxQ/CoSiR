"""E3: the aspect-factor training grid and the seed-42 selection pick (CVPR plan Task 12).

The rules are fixed in PREREGISTRATION.md in this folder, committed before any run. Run from the repo root:

  python src/test/20261101_aspect_factor_gonogo/run_gonogo.py --train A1 --seed 42 [--smoke]    # GPU (one run)
  flock -n -o -E 75 /tmp/gpu0.lock bash src/test/20261101_aspect_factor_gonogo/run_grid.sh      # the 8 grid runs
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python src/test/20261101_aspect_factor_gonogo/run_gonogo.py --select [--smoke]

Row scope: training reads scorer-train rows only (local rows 0..n-1 of artelingo_splits().scorer_train); evaluation
reads selection rows only (features and codes are NaN elsewhere, asserted); val and held rows are never read.
--select runs on the CPU. --smoke trains 50 steps on Task 10's smoke bank into checkpoints/smoke/ and
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
from src.data.artelingo_splits import artelingo_splits  # noqa: E402
from src.eval.aspect_episodes import AspectEpisodes, concat_episodes, episodes_sha256  # noqa: E402
from src.eval.aspect_metrics import METRICS, per_anchor, summarize  # noqa: E402
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
    f["res"].mkdir(parents=True, exist_ok=True)
    ctx = EvalContext(42, smoke)
    log(f"seed-42 selection episodes: {ctx.n} pooled ({ctx.n_per_pair} per pair), SHA-256s match Task 9")
    rows, arrays = {}, {"anchor_group": ctx.anchor_group, "pair_index": ctx.pair_index}
    for run in RUNS:
        ckpt = checkpoint_path(run, 42, smoke)
        row = {"eligible": run in ELIGIBLE, "bank": RUNS[run][0]}
        if smoke:
            row["smoke_stand_in"] = "A1 smoke checkpoint (50 steps) stands in for this run"
        failed = f["res"] / f"failed_{run}_seed42.json"
        if not ckpt.exists():
            if failed.exists() and not smoke:
                rows[run] = {**row, "status": "failed (non-finite codes)", "eligible": False}
                log(f"{run}: failed run, not eligible")
                continue
            raise FileNotFoundError(f"{ckpt} missing (and no failed record): the grid is not complete")
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
    print(f"\nPICKED: {picked}" + (" (SMOKE: every run is the A1 smoke checkpoint; the tie goes to A1)" if smoke else ""))
    log(f"wrote {out_sel.relative_to(ROOT)} and {out_pick.relative_to(ROOT)}")


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", choices=sorted(RUNS), metavar="RUN", help="train one grid run (A1..A6, H1, S1)")
    mode.add_argument("--select", action="store_true", help="seed-42 selection table and the pick")
    ap.add_argument("--seed", type=int, default=42, help="model seed for --train")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    if args.train:
        train(args.train, args.seed, args.smoke)
    else:
        select(args.smoke)
    log("done")


if __name__ == "__main__":
    main()
