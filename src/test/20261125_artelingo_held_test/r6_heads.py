"""Round 6 head refits (DECISION_RULE.md §6 item 2, §5 item 5, §11; contracts §3; ticket 03).

The five posterior heads of the read, refit once per process on the same 60,000 scorer-train rows as before, returned
as fitted classifiers so that one fitted object predicts selection rows (the bit-for-bit check) and then held rows:

| head        | grouping                                      | classes | copied from                         |
|-------------|-----------------------------------------------|---------|-------------------------------------|
| `affect`    | D1 affect, `partition_L` (told oracle arm L)  | 41      | `run_told_oracle.fit_one_head`      |
| `affect_km` | E2 affect-km (`run_n6.PARTS` "affect")        | 64      | `run_n6.fit_heads`                  |
| `image`     | E2 image                                      | 64      | `run_n6.fit_heads`                  |
| `caption`   | E2 caption                                    | 64      | `run_n6.fit_heads`                  |
| `csd`       | step 1's `style_csd` (CLIP B/32 features)     | 17      | `run_told_oracle.fit_one_head`      |

The copies keep the originals' row draw (`default_rng(PROBE_SEED=0)`), check rows (`default_rng(1)`), the
`LogisticRegression(C=1.0, max_iter=300)` settings on `run_checks.unit` features, the order of the fits (image then
caption head per grouping; E2's groupings in `run_n6.PARTS` order) and the provenance dicts, and drop only the
prediction, which `predict` does.

Rules (cpu_path §4 and hazards 2, 18):
- Fits use the raw `data.img_features` / `data.txt_features` rows of the draw, never a masked array.
- `predict` makes ONE `predict_proba` call per (head, modality) on exactly `unit(F[rows])`, as the originals did on
  selection rows. Any other row set is predicted in its own call: never on a concatenation, since BLAS blocking can
  change bits. `rows` must be sorted and unique (a concatenation of two row sets is refused).
- OMP_NUM_THREADS and MKL_NUM_THREADS are 8 in the stage-1a run and in the read, so the coefficients hash the same.
- The csd targets are `style_csd__img` and `style_csd__txt` of step1_heads_style.npz, never the CSD-feature head
  `style_csd__img_src`. This refit was never checked before round 6; a difference is a stage-1a stop (rule §6.2),
  reported, never "fixed".

Guards carry a `# guard:<name>` marker; test_r6_heads.py deletes each on a copy and shows that its scenario then goes
through.
"""
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first: MAIN, the earlier rounds and run_n6 / run_told_oracle by path)

import numpy as np  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

n6, rto, rc, rg = R.run_n6, R.rto, R.run_checks, R.rg

HEADS = ("affect", "affect_km", "image", "caption", "csd")      # fit order and output order
E2_HEADS = {"affect_km": "affect", "image": "image", "caption": "caption"}   # r6 name -> run_n6.PARTS name
N_CLASSES = {"affect": 41, "affect_km": 64, "image": 64, "caption": 64, "csd": 17}
MODALITIES = ("img", "txt")
THREADS = "8"

HEAD_ROWS = n6.HEAD_ROWS
if not (n6.PARTS == ("affect", "image", "caption") and HEAD_ROWS == 60_000 and n6.CHECK_ROWS == 10_000
        and rc.PROBE_SEED == 0 and tuple(E2_HEADS.values()) == n6.PARTS):
    raise ImportError("run_n6 / run_checks constants differ from the ones the stored heads were fitted with")

# inputs (keys of r6_common.INPUT_SHA256, paths relative to src/test/)
PARTITION_L_REL = "20261111_community_told_oracle/results/per_anchor_told_oracle.npz"
TOLD_REL = "20261111_community_told_oracle/results/told_oracle.json"
E2_PARTITIONS_REL = "20261031_pseudo_partitions/results/partitions.npz"
CSD_GROUP_REL = "20261116_grouping_step1_style/results/step1_group_style.npz"
N6_REL = "20261108_new_method_quick_checks/results/n6_posteriors.npz"
CSD_HEADS_REL = "20261116_grouping_step1_style/results/step1_heads_style.npz"

# rule §6.2's stored selection-row posteriors: file -> {stored key: (head, modality)}. n6's "affect" is affect-km
# (E2), not D1's affect (hazard 9); the csd targets are the CLIP heads, not `style_csd__img_src`.
TARGETS = {
    N6_REL: {"affect__img": ("affect_km", "img"), "affect__txt": ("affect_km", "txt"),
             "image__img": ("image", "img"), "image__txt": ("image", "txt"),
             "caption__img": ("caption", "img"), "caption__txt": ("caption", "txt")},
    CSD_HEADS_REL: {"style_csd__img": ("csd", "img"), "style_csd__txt": ("csd", "txt")},
}
ITEM_KEYS = tuple(f"{Path(rel).stem}/{key}" for rel, keys in TARGETS.items() for key in keys)


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def require_threads():
    """OMP_NUM_THREADS = MKL_NUM_THREADS = 8, as in every earlier fit and in the read (cpu_path §4)."""
    got = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    _require(all(v == THREADS for v in got.values()),
             f"thread settings {got}: head fits and predictions run with 8 threads")  # guard:threads


def _features(data) -> dict:
    """{"img", "txt"}: data's raw CLIP B/32 arrays (float32, all rows)."""
    feats = {"img": data.img_features, "txt": data.txt_features}
    for m, F in feats.items():
        _require(isinstance(F, np.ndarray) and F.dtype == np.float32 and F.ndim == 2,
                 f"data.{m}_features must be a float32 matrix")
    _require(feats["img"].shape[0] == feats["txt"].shape[0], "image and caption features differ in rows")
    return feats


def _fit_input(F, draw):
    """`run_checks.unit(F[draw])`, the originals' fit input, from raw features (hazard 18)."""
    X = rc.unit(F[draw])
    _require(np.isfinite(X).all(),
             "a head fit got non-finite rows: fits use the raw scorer-train features, never a masked "
             "array (hazard 18)")  # guard:raw_features
    return X


# ---------------------------------------------------------------- the copies

def fit_heads_n6_r6(data, labels, scorer_train, n_rows=HEAD_ROWS):
    """`run_n6.fit_heads` (QC/run_n6.py:53-78) without the prediction: the same draw, check rows, LR settings and
    call order. ``labels`` is {run_n6.PARTS name: global labels (-1 outside scorer-train)}. Returns
    (clfs {part: {"img", "txt"}: fitted LogisticRegression}, prov), prov exactly as the original's."""
    draw = np.random.default_rng(rc.PROBE_SEED).choice(scorer_train, n_rows, replace=False)
    rest = np.setdiff1d(scorer_train, draw)
    check = np.random.default_rng(1).choice(rest, min(n6.CHECK_ROWS, len(rest)), replace=False)
    feats = _features(data)
    out, prov = {}, {"draw_rows_sha256": rg.sha_array(np.sort(draw)), "n_draw": int(n_rows)}
    for h in n6.PARTS:
        lab = labels[h]
        clfs = {m: LogisticRegression(C=1.0, max_iter=300).fit(_fit_input(F, draw), lab[draw])
                for m, F in feats.items()}
        if not np.array_equal(clfs["img"].classes_, clfs["txt"].classes_):
            raise AssertionError(f"{h}: image and caption heads have different classes")
        out[h] = clfs
        prov[h] = {"n_classes": int(len(clfs["img"].classes_)),
                   "heldout_accuracy": {m: 100 * float(clfs[m].score(rc.unit(F[check]), lab[check]))
                                        for m, F in feats.items()}}
        rc.log(f"head {h} done")
    return out, prov


def fit_one_head_r6(data, lab, scorer_train, n_rows=HEAD_ROWS):
    """`run_told_oracle.fit_one_head` (TO/run_told_oracle.py:178-201) without the prediction: the same draw, check
    rows, LR settings and call order. Returns (clfs {"img", "txt"}, prov), prov exactly as the original's."""
    draw = np.random.default_rng(rc.PROBE_SEED).choice(scorer_train, n_rows, replace=False)
    rest = np.setdiff1d(scorer_train, draw)
    check = np.random.default_rng(1).choice(rest, min(n6.CHECK_ROWS, len(rest)), replace=False)
    feats = _features(data)
    clfs = {m: LogisticRegression(C=1.0, max_iter=300).fit(_fit_input(F, draw), lab[draw]) for m, F in feats.items()}
    if not np.array_equal(clfs["img"].classes_, clfs["txt"].classes_):
        raise AssertionError("image and caption heads have different classes")
    counts = np.bincount(lab[check])
    prov = {"n_classes": int(len(clfs["img"].classes_)), "draw_rows_sha256": rg.sha_array(np.sort(draw)),
            "heldout_accuracy": {m: 100 * float(clfs[m].score(rc.unit(F[check]), lab[check]))
                                 for m, F in feats.items()},
            "check_majority_share": 100 * float(counts.max() / counts.sum()),
            "uniform": 100.0 / len(clfs["img"].classes_)}
    return clfs, prov


# ---------------------------------------------------------------- the five heads

def head_labels(groups, scorer_train) -> dict:
    """{head: global labels, int64, -1 outside scorer-train} from the SHA-asserted partitions (only the keys used)."""
    n = len(groups)
    R.assert_input(PARTITION_L_REL)
    with np.load(R.INPUT_PATHS[PARTITION_L_REL]) as z:
        partition_L = np.asarray(z["partition_L"], dtype=np.int64)
    out = {"affect": rto.global_labels(partition_L, scorer_train, n)}
    R.assert_input(E2_PARTITIONS_REL)
    e2 = n6.partition_labels(groups, scorer_train)      # its own SHA and E2 alignment checks
    out.update({name: e2[part] for name, part in E2_HEADS.items()})
    out["csd"] = csd_labels(scorer_train, n)
    return {h: out[h] for h in HEADS}


def csd_labels(scorer_train, n_rows_all):
    """step 1's `style_csd` labels on scorer-train rows (run_step1.stage_heads :523-532)."""
    R.assert_input(CSD_GROUP_REL)
    with np.load(R.INPUT_PATHS[CSD_GROUP_REL]) as z:
        st, local = np.asarray(z["scorer_train"]), np.asarray(z["style_csd"], dtype=np.int64)
    _require(np.array_equal(np.asarray(scorer_train), st),
             "scorer_train differs from step1_group_style.npz's")  # guard:csd_scorer_train
    return rto.global_labels(local, scorer_train, n_rows_all)


def fit_all(data, scorer_train, labels, n_rows=HEAD_ROWS) -> dict:
    """The five heads on ``labels`` ({head: global labels}). Order: D1 affect, E2 (affect-km, image, caption), csd.
    Returns {head: {"img": clf, "txt": clf, "prov": dict, "convergence_warnings": int}}."""
    require_threads()
    scorer_train = np.asarray(scorer_train)
    heads = {}

    def counted(fn, *a):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            res = fn(*a)
        return res, int(sum(issubclass(w.category, ConvergenceWarning) for w in caught))

    (clfs, prov), nw = counted(fit_one_head_r6, data, labels["affect"], scorer_train, n_rows)
    heads["affect"] = {**clfs, "prov": prov, "convergence_warnings": nw}
    (clfs, prov), nw = counted(fit_heads_n6_r6, data, {part: labels[name] for name, part in E2_HEADS.items()},
                               scorer_train, n_rows)
    for name, part in E2_HEADS.items():
        heads[name] = {**clfs[part],
                       "prov": {"draw_rows_sha256": prov["draw_rows_sha256"], "n_draw": prov["n_draw"], **prov[part]},
                       "convergence_warnings": nw}       # one count for E2's three groupings (one call)
    (clfs, prov), nw = counted(fit_one_head_r6, data, labels["csd"], scorer_train, n_rows)
    heads["csd"] = {**clfs, "prov": prov, "convergence_warnings": nw}
    for name in HEADS:
        for m in MODALITIES:
            _require(np.array_equal(heads[name][m].classes_, np.arange(N_CLASSES[name])),
                     f"{name}/{m}: classes are not 0..{N_CLASSES[name] - 1} (a group missing from the "
                     f"draw?)")  # guard:n_classes
    return {h: heads[h] for h in HEADS}


def fit_heads_r6(data, groups, scorer_train, n_rows=HEAD_ROWS) -> dict:
    """Contracts §3: the five heads of rule §6.2 refit on the real partitions (60,000-row draw)."""
    require_threads()
    return fit_all(data, scorer_train, head_labels(groups, scorer_train), n_rows)


# ---------------------------------------------------------------- prediction and coefficients

def predict(heads, data, rows) -> dict:
    """{head: {"img", "txt": (n_rows_all, n_cls) float32}}, NaN outside ``rows``: one `predict_proba` call per
    (head, modality) on exactly `unit(F[rows])`. Call once per row set (selection, held), never on a union."""
    require_threads()
    feats = _features(data)
    n = feats["img"].shape[0]
    rows = np.asarray(rows)
    _require(rows.ndim == 1 and rows.dtype.kind in "iu" and len(rows) > 0, "rows must be a 1-d integer array")
    _require(bool(np.all(np.diff(rows) > 0)) and rows[0] >= 0 and rows[-1] < n,
             "rows must be sorted and unique: predict one row set per call, never a "
             "concatenation")  # guard:rows_sorted
    inside = np.zeros(n, dtype=bool)
    inside[rows] = True
    out = {}
    for name, head in heads.items():
        out[name] = {}
        for m, F in feats.items():
            clf = head[m]
            full = np.full((n, len(clf.classes_)), np.nan, dtype=np.float32)
            full[rows] = clf.predict_proba(rc.unit(F[rows]))
            _require(np.isfinite(full[rows]).all() and np.isnan(full[~inside]).all(),
                     f"{name}/{m}: posteriors must be finite on the rows and NaN elsewhere")  # guard:finite_nan
            out[name][m] = full
    return out


def coef_sha256(heads) -> dict:
    """{head: {"img", "txt": SHA-256 of coef_ then intercept_ bytes (C-contiguous float64)}} (rule §5.5)."""
    out = {}
    for name, head in heads.items():
        out[name] = {}
        for m in MODALITIES:
            coef, icpt = head[m].coef_, head[m].intercept_
            _require(coef.dtype == np.float64 and icpt.dtype == np.float64, f"{name}/{m}: coefficients not float64")
            h = hashlib.sha256()
            h.update(np.ascontiguousarray(coef).tobytes())
            h.update(np.ascontiguousarray(icpt).tobytes())
            out[name][m] = h.hexdigest()
    return out


# ---------------------------------------------------------------- the stage-1a check (rule §6.2)

def compare_bits(got, want) -> dict:
    """{"equal", "n_diff", "max_abs_diff"}: bit-for-bit comparison of two float32 arrays (n_diff counts elements
    whose bits differ)."""
    got, want = np.asarray(got), np.asarray(want)
    if got.shape != want.shape or got.dtype != np.float32 or want.dtype != np.float32:
        return {"equal": False, "n_diff": int(max(got.size, want.size)), "max_abs_diff": None,
                "shape": [list(got.shape), list(want.shape)], "dtype": [str(got.dtype), str(want.dtype)]}
    differ = got.view(np.uint32) != want.view(np.uint32)
    n_diff = int(differ.sum())
    mad = float(np.max(np.abs(got[differ].astype(np.float64) - want[differ].astype(np.float64)))) if n_diff else 0.0
    return {"equal": n_diff == 0, "n_diff": n_diff, "max_abs_diff": mad}


def affect_identity(prov) -> bool:
    """R3 rule D2: the refit affect head's prov equals told_oracle.json arm L's head."""
    R.assert_input(TOLD_REL)
    stored = json.loads(Path(R.INPUT_PATHS[TOLD_REL]).read_text())
    return bool(rto.roundtrip(prov) == stored["arms"]["L"]["head"])


def check_selection(heads, data, selection) -> dict:
    """Rule §6.2: the refit posteriors on ``selection`` (one predict call, float32) against n6_posteriors.npz's six
    arrays and step1_heads_style.npz's `style_csd__img`, `style_csd__txt`, bit for bit; and the affect heads' prov
    against told_oracle arm L. {"passed", "items": {"<file stem>/<key>": compare_bits}, "affect_identity"}."""
    _require(tuple(heads) == HEADS, f"heads {tuple(heads)} are not {HEADS}")
    selection = np.asarray(selection)
    post = predict(heads, data, selection)
    items = {}
    for rel, keys in TARGETS.items():
        R.assert_input(rel)
        with np.load(R.INPUT_PATHS[rel]) as z:
            _require(np.array_equal(z["selection"], selection),
                     f"{rel} was computed on another selection row set")  # guard:stored_selection
            for key, (h, m) in keys.items():
                items[f"{Path(rel).stem}/{key}"] = compare_bits(post[h][m][selection], z[key])
    identity = affect_identity(heads["affect"]["prov"])
    _require(tuple(items) == ITEM_KEYS, "the checked items differ from rule §6.2's eight arrays")
    return {"passed": bool(all(it["equal"] for it in items.values()) and identity), "items": items,
            "affect_identity": identity}
