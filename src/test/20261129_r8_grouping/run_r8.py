"""Round 8 run driver (ticket 09): readers, tau^X and the seed-42 arrays.

Sub-commands (pattern of run_r7.py prep/gate; rule M14, M15, section 9):
  readers  build the A0+X reader (R7 L11) from the cross-fitted banks of heads/xheads_<X>.npz, <= 3 jobs in parallel
  taus     tau^X (R7 L12; A:D25) for a batch: seed42/taus_<batch>.json and seed42/reader_<X>.npz
  score    seed-42 arrays of a batch: seed42/arrays_<batch>.npz plus its index (R1[X], X-AFF[X], their CF, BX; AFF, B,
           B0, B1, cl, pair_index, parity as stage 0 proved them)

Every sub-command refuses to run unless stage0_regression.json has passed (rule section 9: SC1 before any candidate).
A batch is one arrays file; the main batch's index is seed42/index.json, any later batch (NBC, H<X>) gets its own
immutable index_<batch>.json, so earlier files are never rewritten; ticket 10 reads the union (--index a b).
Reader inputs, bundle and scoring functions are the ones stage 0 (run_r8_stage0) calls, so SC1 proves this path.
--smoke: seed 9001, banks of 32 episodes per block, root results/smoke/gate (override with --root, under smoke);
outputs only under results/smoke/. Never a verdict.
Stored per grouping X: seed42/reader_<X>.npz keys P__a P__b m__a m__b pick__a pick__b (R7's layout; A0 as reader_A0.npz).
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import r8_common as R
import r8_bundle as B
import r8_reader as Q
import r8_scoring_io as IO
import run_r8_stage0 as S0

METRICS = S0.METRICS
MAIN_NO_NBC = tuple(x for x in IO.CANDIDATES if x != "NBC")
BATCH_GROUPINGS = {   # default grouping IDs per batch; H<X> batches give --groupings explicitly
    "main": MAIN_NO_NBC + IO.REFERENCES + tuple(f"Z{x}" for x in MAIN_NO_NBC),
    "nbc": ("NBC", "ZNBC"),
}
READER_KEYS = tuple(f"{k}__{c}" for k in ("P", "m", "pick") for c in ("a", "b"))
SMOKE_BANK_N = 32   # as stage 0's smoke
N_REAL, N_PAIR_REAL = 12_288, 4_096


# ------------------------------------------------------------------------------------------------ paths and guards

def results_root(args):
    if args.smoke:
        root = Path(args.root) if args.root else R.SMOKE / "gate"
        assert root.resolve().is_relative_to(R.SMOKE.resolve()), f"smoke outputs must stay under {R.SMOKE}"
    else:
        assert not args.root, "--root is for smoke only; real results go to results/"
        root = R.RESULTS
    return root.resolve()


def seed_of(smoke):
    return R.SMOKE_SEEDS[0] if smoke else R.DEV_SEED


def require_stage0(root, smoke):
    """Rule section 9 order: SC1 first. Real: root/stage0_regression.json must say pass. Smoke: wiring_pass."""
    paths = [root / "stage0_regression.json"] + ([R.SMOKE / "stage0" / "stage0_regression.json"] if smoke else [])
    path = next((p for p in paths if p.is_file()), None)
    assert path is not None, f"refusing to run before SC1: no stage0_regression.json in {[str(p) for p in paths]}"
    rec = json.loads(path.read_text())
    assert rec.get("rule_sha256") == R.sha_file(R.RULE), f"{path}: rule SHA-256 differs"
    assert rec.get("smoke") is smoke, f"{path}: smoke role differs"
    ok = rec.get("wiring_pass") if smoke else rec.get("pass")   # rule gap: smoke stage-0 records carry wiring_pass, pass=false
    assert ok is True, f"refusing to run: SC1 did not pass ({path})"
    return str(path), R.sha_file(path)


def smoke_keys():
    """The input subset stage 0's smoke hashes (same filter), so the smoke needs no later-stage inputs."""
    return tuple(k for k in R.INPUTS if k.startswith((
        "20261126_design_l_style_source/", "20261121_round3_", "20261031_pseudo_", "20261116_grouping_step1_",
        "20261117_reader_fix_")) or k.startswith("/data/SSD2/pre_extract/artelingo/style_csd_vitl/")
        or k == "/data/SSD2/pre_extract/artelingo/features/metadata.json")


def start(args):
    root = results_root(args)
    smoke = bool(args.smoke)
    R.assert_rule()
    R.assert_threads()
    keys = smoke_keys() if smoke else None
    shas = R.assert_inputs(keys=keys, fresh=True)
    stage0 = require_stage0(root, smoke)
    return root, smoke, keys, shas, stage0


def end_check(smoke, keys, shas):
    """Section-10 inputs are unchanged at the end of the step (stage 0 does the same at both ends)."""
    again = R.assert_inputs(keys=keys, fresh=True)
    assert again == shas, "a section-10 input changed during the step"


def grouping_records(root):
    path = root / "groupings" / "r8_groupings.json"
    rec = json.loads(path.read_text())
    assert isinstance(rec.get("groupings"), dict) and isinstance(rec.get("dropped"), dict), f"{path}: shape"
    return rec


def wanted(args):
    names = tuple(args.groupings) if args.groupings else BATCH_GROUPINGS.get(args.batch)
    assert names, f"batch {args.batch!r} has no default groupings; give --groupings"
    assert len(set(names)) == len(names) and set(names) <= set(IO.GROUPINGS) - {"R0"}, f"bad grouping list {names}"
    return names


def split_dropped(root, names):
    """(active, {X: reason}): M16 drops from the grouping record, ticket 08's head record (if one says dropped) and the
    reader record (unfillable bank block). A dropped X has no control (rule M13): Z<X> is dropped with X."""
    # rule gap: ticket 08's drop record has no fixed name; a heads/head_<X>.json with dropped=true is honoured.
    rec = grouping_records(root)
    dropped = {x: str(v) for x, v in rec["dropped"].items()}
    out = {}
    for x in names:
        base = x[1:] if x.startswith("Z") and x[1:] in IO.CANDIDATES else None
        if x in dropped:
            out[x] = dropped[x]
        elif base is not None and base in dropped:
            out[x] = f"control of dropped {base}: {dropped[base]}"
        else:
            hp = root / "heads" / f"head_{x}.json"
            if hp.is_file():
                h = json.loads(hp.read_text())
                if h.get("dropped") is True:
                    out[x] = str(h.get("reason", "dropped by head record"))
            if x not in out:
                assert x in rec["groupings"], f"{x}: neither a grouping record nor a drop in r8_groupings.json"
    return out


# ------------------------------------------------------------------------------------------------ readers

def reader_job(name, root, smoke):
    """One reader, in a worker process; same call as stage 0's SC1b (labels, halves, post from the shared helper)."""
    labels, halves, post = IO.reader_inputs(name, root)
    path = Path(root) / "readers" / f"reader_{name}.json"
    bank_n = SMOKE_BANK_N if smoke else Q.BANK_N
    if path.exists():
        rec = Q.validate_reader(name, labels, halves, root, smoke)   # preserved reader: check, never refit
    else:
        rec = Q.build_reader(name, labels, halves, root=root, smoke=smoke, bank_n=bank_n, post=post)
    return name, bool(rec["dropped"]), rec.get("reason")


def cmd_readers(args):
    root, smoke, keys, shas, _ = start(args)
    names = wanted(args)
    drops = split_dropped(root, names)
    todo = [x for x in names if x not in drops]
    jobs = min(int(args.jobs), 3, max(1, len(todo)))
    assert 1 <= args.jobs <= 3, "at most 3 reader jobs (rule section 9)"
    t0 = time.monotonic()
    results = []
    if jobs == 1:
        results = [reader_job(x, root, smoke) for x in todo]
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(reader_job, x, root, smoke) for x in todo]
            results = [f.result() for f in futures]   # a failed job raises here
    for name, dropped, reason in results:
        print(f"reader {name}: {'DROPPED ' + str(reason) if dropped else 'ok'}", flush=True)
    for x, why in drops.items():
        print(f"reader {x}: skipped, dropped ({why})", flush=True)
    end_check(smoke, keys, shas)
    print(f"readers: {len(results)} built/checked, {len(drops)} dropped, {time.monotonic() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------ bundle and X inputs

def load_x_post(ctx, x, labels_x, root):
    """post_<X>.npz at the global-row boundary, as r8_bundle.load_posteriors but class count from X's labels
    (no head_<X>.json is needed)."""
    # rule gap: ticket 08's head record is not required; K is checked against the grouping labels instead.
    path = Path(root) / "heads" / f"post_{x}.npz"
    with np.load(path, allow_pickle=False) as z:
        assert {"img", "txt", "rows"} <= set(z.files), f"{path}: missing posterior keys"
        rows = z["rows"]
        assert rows.dtype == np.int64 and np.array_equal(rows, ctx.selection), f"{path}: rows differ from selection"
        post = {side: z[side] for side in ("img", "txt")}
    B._posterior_ok(post, ctx, x)
    k = len(np.unique(labels_x))
    assert post["img"].shape[1] == k, f"{path}: class count differs from the grouping's {k}"
    for side in post:
        v = post[side][ctx.selection]
        assert (v >= 0).all() and np.allclose(v.sum(1), 1, rtol=0, atol=1e-5), f"{path}/{side}: probability sums"
    return post


def build_state(args, root, smoke):
    """The stage-0 bundle (A0 + R0 from R7's results) with A0 posteriors replaced by R7's stored ones, as SC1a."""
    seed = seed_of(smoke)
    fusion = R.load_r7("r7_fusion")
    bundle = B.build_bundle(seed, smoke, extra=("R0",), root=S0.R7_ROOT)
    for name in B.A0:
        bundle.post[name] = S0.standard_post(bundle.ctx, name)
    bundle.stack, bundle.F = B.stack_and_features(bundle, B.A0)
    return bundle, fusion


def reader_pkl(root, x):
    return Path(root) / "readers" / f"reader_{x}.pkl"


def save_reader_npz(path, rd):
    arrays = {f"{k}__{c}": rd[k][c] for k in ("P", "m", "pick") for c in ("a", "b")}
    assert set(arrays) == set(READER_KEYS)
    if path.exists() and not R.in_smoke(path):
        with np.load(path, allow_pickle=False) as z:
            assert set(arrays) <= set(z.files), f"{path}: missing reader keys"
            assert all(z[k].dtype == v.dtype and np.array_equal(z[k], v) for k, v in arrays.items()), f"{path}: reader changed"
    else:
        R.write_npz_once(path, **arrays)


def active_readers(root, smoke, names, drops):
    """Drop X whose reader record says dropped (unfillable bank block, R7 L11); crash if a reader is missing."""
    drops = dict(drops)
    for x in names:
        if x in drops:
            continue
        rec = Q._record(Path(root) / "readers" / f"reader_{x}.json", smoke)
        assert rec["grouping"] == x, f"reader record of {x} names {rec['grouping']}"
        if rec["dropped"]:
            drops[x] = f"reader dropped: {rec.get('reason')}"
    for x in names:   # a dropped X has no control (rule M13)
        if x.startswith("Z") and x[1:] in drops and x not in drops:
            drops[x] = f"control of dropped {x[1:]}: {drops[x[1:]]}"
    return tuple(x for x in names if x not in drops), drops


def taus_for(bundle, fusion, root, smoke, active, batch, seed_dir):
    """tau^X = L12 percentiles of X's reader margins on the seed-42 episodes; one file per batch, frozen once written."""
    taus, rds = {}, {}
    for x in active:
        labels = IO.load_groupings((x,), root)[x]
        bundle.post[x] = load_x_post(bundle.ctx, x, labels, root)
        parts = B.A0 + (x,)
        rd = fusion.reader(bundle, parts, reader_pkl(root, x))
        t, n = fusion.thresholds(rd["m"])
        assert n == 2 * bundle.n, "L12 margin count differs"
        t = np.asarray(t, dtype=np.float64)
        assert t.shape == (4,) and np.isfinite(t).all() and (np.diff(t) >= 0).all(), f"{x}: taus not four ordered values"
        taus[x], rds[x] = t, rd
        del bundle.post[x]
    path = seed_dir / f"taus_{batch}.json"
    rec = {"batch": batch, "seed": bundle.seed, "smoke": smoke, "n_margins": 2 * bundle.n,
           "taus": {x: [float(v) for v in t] for x, t in taus.items()}}
    if path.exists() and not R.in_smoke(path):
        old = json.loads(path.read_text())
        assert old["taus"] == rec["taus"] and old["n_margins"] == rec["n_margins"], f"{path}: frozen taus changed"
    else:
        R.write_json_once(path, rec)
    return taus, rds, path


def cmd_taus(args):
    root, smoke, keys, shas, _ = start(args)
    names = wanted(args)
    active, drops = active_readers(root, smoke, names, split_dropped(root, names))
    seed_dir = root / "seed42"
    bundle, fusion = build_state(args, root, smoke)
    taus, rds, path = taus_for(bundle, fusion, root, smoke, active, args.batch, seed_dir)
    save_reader_npz(seed_dir / "reader_A0.npz", fusion.reader(bundle, B.A0))
    for x in active:
        save_reader_npz(seed_dir / f"reader_{x}.npz", rds[x])
    end_check(smoke, keys, shas)
    print(f"taus: {path.name}, {len(active)} groupings, {len(drops)} dropped")


# ------------------------------------------------------------------------------------------------ score

def check_batch(arr, smoke, active):
    """Cheap asserts on what was written (shapes as R7's read_arrays, keys as ticket 10's reader needs)."""
    n = len(arr["cl"])
    assert n == (192 if smoke else N_REAL), f"episode count {n}"
    for k in ("cl", "pair_index", "parity"):
        assert arr[k].dtype == np.int64 and arr[k].shape == (n,), f"{k}: dtype/shape"
    assert np.isin(arr["pair_index"], (0, 1, 2)).all() and np.isin(arr["parity"], (0, 1)).all(), "IDs"
    if not smoke:
        assert all((arr["pair_index"] == p).sum() == N_PAIR_REAL for p in (0, 1, 2)), "4,096 per pair"
    need = [f"{p}__{m}" for p in ("AFF", "AFF_cf", "B", "B0", "B1") for m in METRICS]
    need += [f"{p}_{x}{c}__{m}" for x in active for p in ("R1", "XAFF") for c in ("", "_cf") for m in METRICS]
    need += [f"BX_{x}__{m}" for x in active for m in METRICS]
    need += ["AFF__fused_cells", "AFF__cf_cells", "AFF__sigma", "AFF__gate_a", "AFF__gate_b"]
    assert not (set(need) - set(arr)), f"missing keys {sorted(set(need) - set(arr))[:5]}"
    for k, a in arr.items():
        if k.endswith(tuple(f"__{m}" for m in METRICS)):
            assert a.shape == (n,) and a.dtype == np.float64 and np.isfinite(a).all(), f"{k}: metrics"
        elif k.endswith(("__gate_a", "__gate_b")):
            assert a.shape == (4, n) and a.dtype == np.float32, f"{k}: gates"
        elif k.endswith(("__fused_cells", "__cf_cells")):
            assert a.shape == (2,) and a.dtype == np.int64, f"{k}: cells"
        elif k.endswith("__sigma"):
            assert a.shape == (2,) and a.dtype == np.float64, f"{k}: sigma"


def cmd_score(args):
    root, smoke, keys, shas, stage0 = start(args)
    batch = args.batch
    names = wanted(args)
    active, drops = active_readers(root, smoke, names, split_dropped(root, names))
    assert active, "nothing to score"
    seed_dir = root / "seed42"
    arrays_path = seed_dir / f"arrays_{batch}.npz"
    index_path = seed_dir / (args.index_name or ("index.json" if batch == "main" else f"index_{batch}.json"))
    for p in (arrays_path, index_path):
        assert smoke or not p.exists(), f"{p} exists; batches are write-once (use a new --batch name)"
    t0 = time.monotonic()
    bundle, fusion = build_state(args, root, smoke)
    taus, rds, taus_path = taus_for(bundle, fusion, root, smoke, active, batch, seed_dir)
    # Baselines exactly as SC1 (sc1a, sc1c): AFF through X-AFF on A0's frozen reader, B/B0 from the build, B1 = A0+R0.
    arr = {k: getattr(bundle, k) for k in ("cl", "pair_index", "parity")}
    for name, pa in (("B", bundle.pB), ("B0", bundle.pBp), ("B1", B.condition_free(bundle, B.A1)[0])):
        arr.update({f"{name}__{m}": v for m, v in pa.items()})
    arr.update(fusion.score_aff(bundle)["arrays"])
    save_reader_npz(seed_dir / "reader_A0.npz", fusion.reader(bundle, B.A0))
    for x in active:
        labels = IO.load_groupings((x,), root)[x]
        bundle.post[x] = load_x_post(bundle.ctx, x, labels, root)
        parts, pkl = B.A0 + (x,), reader_pkl(root, x)
        save_reader_npz(seed_dir / f"reader_{x}.npz", rds[x])
        arr.update({f"BX_{x}__{m}": v for m, v in B.condition_free(bundle, parts)[0].items()})
        arr.update(fusion.score_r1(bundle, parts, pkl, taus[x])["arrays"])
        arr.update(fusion.score_x_aff(bundle, parts, pkl, taus[x])["arrays"])
        del bundle.post[x]
        print(f"scored {x}", flush=True)
    check_batch(arr, smoke, active)
    R.write_npz_once(arrays_path, **arr)
    with np.load(arrays_path, allow_pickle=False) as z:   # read back: what was written is what was checked
        check_batch({k: z[k] for k in z.files}, smoke, active)
    end_check(smoke, keys, shas)
    entry = {"file": arrays_path.name, "sha256": R.sha_file(arrays_path), "batch": batch, "seed": bundle.seed,
             "smoke": smoke, "n_episodes": int(len(arr["cl"])), "groupings": list(active), "dropped": drops,
             "taus_file": taus_path.name, "taus_sha256": R.sha_file(taus_path),
             "readers_sha256": {x: R.sha_file(seed_dir / f"reader_{x}.npz") for x in ("A0",) + tuple(active)},
             "stage0_record": {"path": stage0[0], "sha256": stage0[1]}}
    R.write_json_once(index_path, {"batches": [entry]})
    print(f"score: {arrays_path.name} ({len(arr)} arrays, {len(active)} groupings, {len(drops)} dropped), "
          f"index {index_path.name}, {time.monotonic() - t0:.0f}s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("readers", "taus", "score"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--root", help="smoke only: results root (default results/smoke/gate)")
    p.add_argument("--batch", default="main", help="main (default groupings: 12 candidates minus NBC, refs, controls), "
                   "nbc, or a new name with --groupings (e.g. hx)")
    p.add_argument("--groupings", nargs="+", help="grouping IDs (default: the batch's list)")
    p.add_argument("--jobs", type=int, default=3, help="readers only: parallel jobs, at most 3")
    p.add_argument("--index-name", help="score only: index file name (default index.json for main, else index_<batch>.json)")
    args = p.parse_args()
    {"readers": cmd_readers, "taus": cmd_taus, "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    main()
