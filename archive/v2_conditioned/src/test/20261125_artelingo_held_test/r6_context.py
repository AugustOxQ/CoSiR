"""Round 6 row context (DECISION_RULE.md of this folder: section 5 items 3 and 5, section 6 item 4; contracts section 4;
ticket 05): what run_gonogo.EvalContext does for selection rows, for any row set of the stage (d) split.

RowContext(mode, seed, data, split, labels, heads, value_sets, n_per_pair) with mode "selection" (rows =
split.selection: the seed-42 regression and the smoke) or "held" (rows = split.held: the read). Its fields are
EvalContext's with the evaluated rows in place of the selection rows (rule section 5 item 5):

- rows, in_rows (bool mask), and `selection` (= rows). `selection` is the tripwire against stored posteriors:
  run_n6.load_posteriors refuses unless the stored array's selection equals ctx.selection, so on held rows it always
  refuses (the stored arrays hold selection rows only).
- img, txt: CLIP B/32 features, NaN outside rows and finite on them (asserted, EvalContext.masked's pattern).
- eps: the seed's episodes, built in process by r6_episodes.build_seed on rows with the development value sets
  (`index` = PaintingValueIndex over all rows); pooled, n, pair_index, parity (episode-index parity), anchor,
  anchor_group (= groups[anchor]) and episode_sha ({pair: sha}). Every member row (anchor, 13 candidates, the 4 + 4
  example pairs in both modalities) is asserted to lie in rows, here as well as in build_seed.
- cos: cosine_scores(EvalInputs(img, txt), pooled), as EvalContext.
- post: r6_heads.predict(heads, data, rows), every head's posteriors NaN outside rows and finite on them (asserted
  again here); coef_sha256 of the heads (rule section 5 item 5's record).
- encode(ckpt): a factor checkpoint's codes of rows only (encode_rows with rows=self.rows; in selection mode exactly
  rows=selection, as EvalContext.encode), NaN elsewhere (asserted).

Seeds. Selection mode admits the development seed 42 (4,096 per pair) and the smoke seeds 9001 to 9003 (64 per pair);
held mode admits only the held seeds 52, 53, 54 (4,096 per pair).

Order. The episodes are built first, right after the features are masked; `on_episodes` (optional, an addition to
contracts section 4) is called with build_seed's namespace before anything else is computed, so the held runner can
record the episode SHA-256s as soon as they exist (rule section 8 item 1).

The fits of heads and PM never see this context's masked arrays (hazard 18): heads are fitted beforehand on raw
scorer-train rows (r6_heads), the PCA basis and pair scaler by r6_bundle.fit_pm.

Guards carry a `# guard:<name>` marker; test_r6_context.py deletes each on a copy and shows that its scenario then
goes through.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_episodes as E  # noqa: E402
import r6_heads as H  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_episodes import PaintingValueIndex  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, cosine_scores  # noqa: E402
from src.train.train_factors import encode_rows, load_factor_checkpoint  # noqa: E402

MODES = ("selection", "held")
# (mode, seed) -> episodes per pair
ADMITTED = {("selection", R.DEV_SEED): R.N_PER_PAIR,
            **{("selection", s): R.N_SMOKE for s in R.SMOKE_SEEDS},
            **{("held", s): R.N_PER_PAIR for s in R.HELD_SEEDS}}
SPLIT_FIELDS = ("groups", "train", "val", "held", "scorer_train", "selection")


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def _index_array(x, what) -> np.ndarray:
    """A 1-d int64 array of sorted, unique row positions."""
    x = np.asarray(x)
    _require(x.ndim == 1 and x.dtype.kind in "iu" and x.size > 0, f"{what} must be a non-empty 1-d integer array")
    x = x.astype(np.int64, copy=False)
    _require(bool(np.all(np.diff(x) > 0)) and x[0] >= 0 and x[-1] < R.N_ROWS,
             f"{what} must be sorted, unique row positions below {R.N_ROWS}")
    return x


def check_masked(out, inside, what):
    """NaN outside the rows, finite on them (EvalContext.masked's assertion)."""
    _require(bool(np.isnan(out[~inside]).all() and np.isfinite(out[inside]).all()),
             f"{what}: must be NaN outside the evaluated rows and finite on them")


class RowContext:
    """Contracts section 4. See the module docstring for the fields."""

    def __init__(self, mode, seed, data, split, labels, heads, value_sets, n_per_pair, on_episodes=None):
        _require(mode in MODES, f"mode {mode!r} is not one of {MODES}")
        seed, n_per_pair = int(seed), int(n_per_pair)
        _require(ADMITTED.get((mode, seed)) == n_per_pair,
                 f"{mode} mode admits {sorted((s, n) for (m, s), n in ADMITTED.items() if m == mode)} as (seed, "
                 f"episodes per pair), not ({seed}, {n_per_pair})")  # guard:seed_mode
        self.mode, self.seed, self.n_per_pair = mode, seed, n_per_pair
        self.smoke = seed in R.SMOKE_SEEDS
        self.data, self.split, self.labels, self.heads, self.value_sets = data, split, labels, heads, value_sets

        # ---- rows
        for f in SPLIT_FIELDS:
            _require(hasattr(split, f), f"split has no field {f!r} (load_split's namespace)")
        self.groups = np.asarray(split.groups)
        _require(self.groups.shape == (R.N_ROWS,), f"groups has shape {self.groups.shape}, not ({R.N_ROWS},)")
        for m in ("img", "txt"):
            F = getattr(data, f"{m}_features")
            _require(isinstance(F, np.ndarray) and F.dtype == np.float32 and F.ndim == 2 and F.shape[0] == R.N_ROWS,
                     f"data.{m}_features must be a float32 ({R.N_ROWS}, d) matrix")
        rows = _index_array(split.selection if mode == "selection" else split.held, f"split.{mode}")
        scorer_train = _index_array(split.scorer_train, "split.scorer_train")
        _require(np.intersect1d(rows, scorer_train).size == 0
                 and (mode == "selection" or np.intersect1d(rows, split.selection).size == 0),
                 f"{mode} rows overlap scorer-train rows (or, held, selection rows)")  # guard:rows_split
        self.rows = rows
        self.selection = rows                  # the tripwire: run_n6.load_posteriors compares its stored selection
        self.in_rows = np.zeros(R.N_ROWS, dtype=bool)
        self.in_rows[rows] = True

        # ---- features, masked (rule section 5 item 5)
        self.img = self.masked(data.img_features)
        self.txt = self.masked(data.txt_features)

        # ---- episodes, built in process, first (rule section 5 item 3)
        self.index = PaintingValueIndex(labels, self.groups)
        self.eps = E.build_seed(labels, self.groups, rows, self.index, value_sets, seed, n_per_pair)
        self.pooled = self.eps.pooled
        self.n = int(self.eps.n)
        _require(self.n == len(R.PAIRS) * n_per_pair == len(self.pooled.anchor),
                 f"{self.n} pooled episodes, not {len(R.PAIRS)} x {n_per_pair}")
        self.pair_index = np.asarray(self.eps.pair_index, dtype=np.int64)
        self.parity = np.asarray(self.eps.parity, dtype=np.int64)
        _require(np.array_equal(self.pair_index, np.repeat(np.arange(len(R.PAIRS)), n_per_pair))
                 and np.array_equal(self.parity, np.arange(self.n) % 2),
                 "pair_index or parity is not EvalContext's (pairs in PAIRS order; episode-index parity)")
        member = np.asarray(self.pooled.rows())
        _require(bool(self.in_rows[member].all()),
                 f"{int((~self.in_rows[member]).sum())} episode member rows lie outside the {mode} "
                 f"rows")  # guard:episode_rows
        self.anchor = np.asarray(self.pooled.anchor, dtype=np.int64)
        self.anchor_group = self.groups[self.anchor]
        self.episode_sha = dict(self.eps.sha)
        _require(tuple(self.episode_sha) == R.PAIR_NAMES, f"episode hashes for {tuple(self.episode_sha)}")
        if on_episodes is not None:
            on_episodes(self.eps)

        # ---- cosine, posteriors (rule section 5 item 5)
        self.cos = cosine_scores(EvalInputs(self.img, self.txt), self.pooled)
        self.post = H.predict(heads, data, rows)
        _require(tuple(self.post) == H.HEADS, f"posteriors of {tuple(self.post)}, not of {H.HEADS}")
        for h in H.HEADS:
            for m in H.MODALITIES:
                p = self.post[h][m]
                _require(p.dtype == np.float32 and p.shape == (R.N_ROWS, H.N_CLASSES[h]),
                         f"post {h}/{m}: {p.dtype} {p.shape}, not float32 ({R.N_ROWS}, {H.N_CLASSES[h]})")
                check_masked(p, self.in_rows, f"post {h}/{m}")  # guard:post_mask
        self.coef_sha256 = H.coef_sha256(heads)

    def masked(self, values) -> np.ndarray:
        """Copy of ``values`` as float32, NaN outside the rows (asserted NaN outside and finite inside)."""
        out = np.full(values.shape, np.nan, dtype=np.float32)
        out[self.rows] = values[self.rows]
        check_masked(out, self.in_rows, "masked features")  # guard:masked
        return out

    def encode(self, ckpt):
        """(img codes, txt codes) of a factor checkpoint on the rows only (CPU), NaN elsewhere (asserted)."""
        model, _ = load_factor_checkpoint(ckpt, device="cpu")
        ic, tc = encode_rows(model, self.data.img_features, self.data.txt_features, rows=self.rows, device="cpu")
        out = []
        for codes in (ic, tc):
            full = np.full((R.N_ROWS, codes.shape[1]), np.nan, dtype=np.float32)
            full[self.rows] = codes
            check_masked(full, self.in_rows, f"{Path(ckpt).name} codes")  # guard:codes_mask
            out.append(full)
        return out[0], out[1]
