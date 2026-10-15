"""Round 5 GoEmotions step (rule DECISION_RULE.md of this folder: D2, D3, §5 item 2, §9, §10 list A item 2).

Computes the 28 GoEmotions sigmoid probabilities of the 32,413 selection captions once, after the regression sample
of D3 (2,048 scorer-train captions against `affect_prepare.npz`'s stored values, tolerance 1e-4) has passed with the
same loaded model. A failed sample passes no selection caption to the model. Nothing here prints a probability.
The model call is `src.data.affect.goemotions_probabilities` (batch 256, max_length 64 from r5_common).

Local constants (not in r5_common): N_SCORER_TRAIN, the size of the scorer-train split (D3's `choice(183_694, ...)`),
and the model hub folder / snapshot name used for the `refs/main` check (D2 item 2).
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r5_common as R5  # noqa: E402

from src.data.affect import goemotions_probabilities, load_goemotions  # noqa: E402,F401
from src.data.artelingo import ANNOTATIONS_PATH, join_captions  # noqa: E402
from src.data.artelingo_splits import artelingo_splits  # noqa: E402

N_SCORER_TRAIN = 183_694                                   # rule D3
ANNOTATIONS_D12 = Path("/data/PDD/artelingo/artelingo_train.json")
HUB_DIR = Path("/data/SSD2/HF_home/hub/models--SamLowe--roberta-base-go_emotions")
SNAPSHOT = Path(R5.SNAP.rstrip("/")).name
AFFECT_PREPARE = R5.ROOT / "src/test/20261018_affect_factor_learning/cache/affect_prepare.npz"
NPZ_NAME = "r5_goemotions_selection.npz"
JSON_NAME = "r5_goemotions_selection.json"
FAIL_NAME = "r5_goemotions_item2_failure.json"
INPUT_NAMES = (["src/data/affect.py", "src/data/artelingo.py", "src/data/artelingo_splits.py",
                str(ANNOTATIONS_D12), "src/test/20261018_affect_factor_learning/cache/affect_prepare.npz",
                "src/test/20261018_affect_factor_learning/run_posthoc_affect.py"]
               + [k for k in R5.INPUTS if k.startswith(R5.SNAP)])


# ---------------------------------------------------------------- environment asserts (D2 item 2)

def assert_snapshot_ref(hub_dir=HUB_DIR):
    got = (Path(hub_dir) / "refs" / "main").read_text().strip()
    if got != SNAPSHOT:
        raise SystemExit(f"the cache's refs/main names {got}, not snapshot {SNAPSHOT}")


def assert_annotation_source():
    if os.environ.get("COSIR_ARTELINGO_ANNOTATIONS"):
        raise SystemExit("COSIR_ARTELINGO_ANNOTATIONS is set; the annotations must be the D12 file")
    if Path(ANNOTATIONS_PATH) != ANNOTATIONS_D12:
        raise SystemExit(f"ANNOTATIONS_PATH is {ANNOTATIONS_PATH}, not {ANNOTATIONS_D12}")


def _annotations(annotations):
    if annotations is not None:
        return annotations
    import json
    with Path(ANNOTATIONS_PATH).open() as f:
        return json.load(f)


# ---------------------------------------------------------------- D2 item 1, D3: rows and captions

def selection_captions(ctx, annotations=None, splits=None, n_expected=None):
    """(rows, sample_ids, captions) of D2 item 1, with its assertions."""
    n_expected = R5.N_SELECTION if n_expected is None else n_expected
    splits = artelingo_splits(ctx.data) if splits is None else splits
    rows = np.asarray(ctx.selection)
    assert rows.ndim == 1 and np.issubdtype(rows.dtype, np.integer), "selection rows must be a 1-d integer array"
    rows = rows.astype(np.int64)
    assert len(rows) == n_expected, f"selection has {len(rows)} rows, not {n_expected}"
    assert np.all(np.diff(rows) > 0), "selection rows must be ascending and unique"
    assert np.array_equal(rows, np.asarray(splits.selection)), "ctx.selection differs from artelingo_splits().selection"
    assert not np.intersect1d(rows, splits.scorer_train).size, "selection overlaps scorer_train"
    assert not np.intersect1d(rows, splits.held).size, "selection overlaps held"
    sample_ids = np.asarray(ctx.data.sample_ids)[rows].astype(np.int64)
    captions = join_captions(sample_ids, _annotations(annotations))
    assert len(captions) == n_expected and all(isinstance(c, str) and c.strip() for c in captions), \
        "selection captions must be non-empty strings, one per row"
    return rows, sample_ids, captions


def regression_sample(ctx, annotations=None, splits=None, sample=None, n_train=None):
    """(pos, captions) of D3: sorted positions into scorer_train, their captions in that order."""
    seed, size = R5.REG_SAMPLE if sample is None else sample
    n_train = N_SCORER_TRAIN if n_train is None else n_train
    splits = artelingo_splits(ctx.data) if splits is None else splits
    st = np.asarray(splits.scorer_train)
    assert len(st) == n_train and np.all(np.diff(st) > 0), "scorer_train must be ascending, 183,694 rows"
    pos = np.sort(np.random.default_rng(seed).choice(n_train, size=size, replace=False))
    sample_ids = np.asarray(ctx.data.sample_ids)[st[pos]].astype(np.int64)
    captions = join_captions(sample_ids, _annotations(annotations))
    assert len(captions) == size
    return pos, captions


# ---------------------------------------------------------------- checks

def check_probs(p, n):
    assert isinstance(p, np.ndarray) and p.dtype == np.float32 and p.shape == (n, R5.N_GOEMO), \
        f"probabilities must be float32 ({n}, {R5.N_GOEMO}), got {getattr(p, 'dtype', None)} {getattr(p, 'shape', None)}"
    assert np.isfinite(p).all() and p.min() >= 0.0 and p.max() <= 1.0, "probabilities must be finite and in [0, 1]"


def compare(rerun, stored, tol):
    """Item 2's comparison: max and mean absolute difference, entries above 1e-5, pass iff max <= tol."""
    rerun, stored = np.asarray(rerun), np.asarray(stored)
    assert rerun.shape == stored.shape, f"shape {rerun.shape} differs from {stored.shape}"
    d = np.abs(rerun.astype(np.float64) - stored.astype(np.float64))
    finite = bool(np.isfinite(d).all())
    mx = float(d.max()) if finite else float("nan")
    return {"passed": bool(finite and mx <= tol), "max_abs": mx, "mean_abs": float(d.mean()) if finite else float("nan"),
            "n_above_1e-5": int((d > 1e-5).sum()), "sample_rng": None, "sample_size": int(rerun.shape[0]),
            "tol": float(tol)}


def token_stats(tokenizer, captions, limit=64, chunk=2048):
    """(largest token count, count over `limit` tokens), special tokens included, no truncation (descriptive)."""
    mx, over = 0, 0
    caps = [str(c) for c in captions]
    for s in range(0, len(caps), chunk):
        for ids in tokenizer(caps[s:s + chunk], truncation=False, padding=False)["input_ids"]:
            mx, over = max(mx, len(ids)), over + (len(ids) > limit)
    return int(mx), int(over)


def _model_device(loaded):
    return next(loaded[1].parameters()).device


def device_tag(loaded):
    d = _model_device(loaded)
    if d.type == "cuda":
        idx = d.index if d.index is not None else torch.cuda.current_device()
        return f"cuda:{idx} {torch.cuda.get_device_name(idx)}"
    return d.type


def assert_device(loaded, device):
    used = _model_device(loaded).type
    if used != device:
        raise SystemExit(f"device used ({used}) is not the device given ({device})")


def _versions():
    import sklearn
    import tokenizers
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__, "scikit-learn": sklearn.__version__}


# ---------------------------------------------------------------- the step

def run(device, loaded, out_dir=None, ctx=None, *, annotations=None, splits=None, affect_probs=None,
        file_shas=None, n_expected=None, n_train=None, sample=None):
    """Regression sample first; only if it passed, the selection captions. Writes the files once (D2 item 5)."""
    out_dir = R5.CACHE if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path, json_path, fail_path = out_dir / NPZ_NAME, out_dir / JSON_NAME, out_dir / FAIL_NAME
    R5.refuse_existing([npz_path, json_path, fail_path], smoke=False)
    assert_device(loaded, device)
    splits = artelingo_splits(ctx.data) if splits is None else splits
    annotations = _annotations(annotations)
    n_expected = R5.N_SELECTION if n_expected is None else n_expected
    seed, size = R5.REG_SAMPLE if sample is None else sample
    if affect_probs is None:
        with np.load(AFFECT_PREPARE) as z:
            affect_probs = z["affect_probs"]
    n_tr = N_SCORER_TRAIN if n_train is None else n_train
    assert len(affect_probs) == n_tr, "affect_probs must have one row per scorer-train row"

    pos, reg_caps = regression_sample(ctx, annotations, splits, (seed, size), n_tr)
    rows, sample_ids, sel_caps = selection_captions(ctx, annotations, splits, n_expected)   # joins only, no model
    rerun = goemotions_probabilities(list(reg_caps), loaded=loaded, batch_size=R5.GOEMO_BATCH,
                                     max_length=R5.GOEMO_MAXLEN)
    check_probs(rerun, size)
    item2 = compare(rerun, np.asarray(affect_probs)[pos], R5.GOEMO_TOL)
    item2["sample_rng"] = seed
    base = {"device": device_tag(loaded), "versions": _versions(), "item2": item2, "written": R5.now_ams()}
    if not item2["passed"]:
        return R5.write_json_once(fail_path, {**base, "stopped": "item 2 failed; no selection caption was passed "
                                              "to the model"}, smoke=False)

    probs = goemotions_probabilities(list(sel_caps), loaded=loaded, batch_size=R5.GOEMO_BATCH,
                                     max_length=R5.GOEMO_MAXLEN)
    check_probs(probs, n_expected)
    mx, over = token_stats(loaded[0], sel_caps, limit=R5.GOEMO_MAXLEN)
    np.savez(npz_path, probs=probs, rows=rows, sample_ids=sample_ids)
    rec = {**base, "model_snapshot": SNAPSHOT, "model_file_sha256": dict(file_shas or {}),
           "batch_size": R5.GOEMO_BATCH, "max_length": R5.GOEMO_MAXLEN, "n_rows": int(len(rows)),
           "n_captions": int(len(sel_caps)), "max_token_count": mx, "n_captions_over_64_tokens": over,
           "npz_sha256": R5.sha256_file(npz_path), "probs_sha256": hashlib.sha256(probs.tobytes()).hexdigest(),
           "rule_sha256": R5.RULE_SHA}
    return R5.write_json_once(json_path, rec, smoke=False)
