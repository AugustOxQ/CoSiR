"""Round 5 re-derivation: checks of the stored GoEmotions file and the CPU spot check (rule §8; D2, D3's tolerance).

Stage B only: no selection caption is passed to the model in stage A. The file's SHA-256 is read from its record
(cache/r5_goemotions_selection.json) and from the run-log line, never from r5_common.py."""
import json
import os

import numpy as np

from rd5_paths import GE_FILE, GE_RECORD, HF_FILES, HF_REV, HF_SNAPSHOT, P, RUN_LOG, SHA, sha_array, sha_file

N_SEL, N_SPOT, SPOT_SEED, TOL = 32_413, 1_024, 6, 1e-4


def spot_positions() -> np.ndarray:
    return np.sort(np.random.default_rng(SPOT_SEED).choice(N_SEL, size=N_SPOT, replace=False))


def load_ge_file(ctx, splits) -> dict:
    """Assert the file's SHA-256 against its record and the run log, its shapes, rows and sample ids."""
    sha = sha_file(GE_FILE)
    rec_text = GE_RECORD.read_text()
    rec = json.loads(rec_text)
    if sha not in rec_text:
        raise SystemExit(f"{GE_FILE.name}: SHA-256 {sha} is not the one its record states")
    if sha not in RUN_LOG.read_text():
        raise SystemExit(f"{GE_FILE.name}: SHA-256 {sha} is not on a run-log line")
    z = np.load(GE_FILE, allow_pickle=False)
    probs, rows, sids = z["probs"], z["rows"], z["sample_ids"]
    if probs.dtype != np.float32 or probs.shape != (N_SEL, 28):
        raise AssertionError(f"probs must be float32 ({N_SEL}, 28), got {probs.dtype} {probs.shape}")
    if not (np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()):
        raise AssertionError("probs must be finite and in [0, 1]")
    if sha_array(probs) not in rec_text:
        raise SystemExit("the SHA-256 of the probs bytes is not the one the record states")
    sel = np.asarray(ctx.selection)
    checks = {
        "rows_dtype_int64": rows.dtype == np.int64 and sids.dtype == np.int64,
        "rows_eq_ctx_selection": bool(np.array_equal(rows, sel)),
        "rows_eq_splits_selection": bool(np.array_equal(rows, np.asarray(splits.selection))),
        "rows_ascending_unique": bool(np.all(np.diff(rows) > 0)),
        "rows_count": int(len(rows)) == N_SEL,
        "rows_disjoint_scorer_train": not len(np.intersect1d(rows, splits.scorer_train)),
        "rows_disjoint_held": not len(np.intersect1d(rows, splits.held)),
        "sample_ids_eq_data": bool(np.array_equal(sids, ctx.data.sample_ids[rows])),
    }
    if not all(checks.values()):
        raise SystemExit(f"GoEmotions file rows or sample ids differ: {checks}")
    return {"sha256": sha, "probs_sha256": sha_array(probs), "record": rec, "checks": checks, "probs": probs,
            "rows": rows, "sample_ids": sids}


def spot_check(ctx, ge, log=print) -> dict:
    """Join the selection captions as D2 does, rerun 1,024 of them on CPU (batch 256, max_length 64) and compare."""
    from src.data import affect, artelingo
    if os.environ.get("COSIR_ARTELINGO_ANNOTATIONS"):
        raise SystemExit("COSIR_ARTELINGO_ANNOTATIONS must be unset")
    if str(artelingo.ANNOTATIONS_PATH) != str(P["annotations"]):
        raise SystemExit(f"ANNOTATIONS_PATH is {artelingo.ANNOTATIONS_PATH}")
    if sha_file(P["annotations"]) != SHA["annotations"]:
        raise SystemExit("artelingo_train.json SHA-256 differs")
    for k in ("src_affect", "src_artelingo"):
        if sha_file(P[k]) != SHA[k]:
            raise SystemExit(f"{P[k]}: SHA-256 differs")
    ref = (HF_SNAPSHOT / "refs/main").read_text().strip()
    if ref != HF_REV:
        raise SystemExit(f"refs/main names {ref}, not {HF_REV}")
    snap = HF_SNAPSHOT / "snapshots" / HF_REV
    for name, want in HF_FILES.items():
        if sha_file(snap / name) != want:
            raise SystemExit(f"{name}: SHA-256 differs from D12")
    with open(artelingo.ANNOTATIONS_PATH) as f:
        annotations = json.load(f)
    captions = artelingo.join_captions(ctx.data.sample_ids[ge["rows"]], annotations)
    if len(captions) != N_SEL or not all(isinstance(c, str) and c.strip() for c in captions):
        raise AssertionError("the selection join must give 32,413 non-empty captions")
    pos = spot_positions()
    loaded = affect.load_goemotions(device="cpu")
    dev = next(loaded[1].parameters()).device
    if dev.type != "cpu":
        raise AssertionError(f"spot check must run on CPU, model is on {dev}")
    log(f"spot check: {len(pos)} captions on CPU")
    rerun = affect.goemotions_probabilities(list(captions[pos]), loaded=loaded, batch_size=256, max_length=64)
    diff = np.abs(rerun.astype(np.float64) - ge["probs"][pos].astype(np.float64))
    return {"positions_sha256": sha_array(pos), "n": int(len(pos)), "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()), "n_above_1e-5": int((diff > 1e-5).sum()), "tolerance": TOL,
            "passed": bool(diff.max() <= TOL), "device": str(dev), "n_captions_joined": int(len(captions))}
