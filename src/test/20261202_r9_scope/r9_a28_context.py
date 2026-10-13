"""ArtELingo-28 row context (ticket 18; DECISION_RULE.md of this folder: section 5 items 4 and 5, section 6 item 5;
R6 rule section 5 items 4 and 5): what r9_context.RowContext does for the held rows of ArtELingo, for the read rows of
the A28 row space (r9_a28_data.load_a28). Everything that is fitted comes from ArtELingo and is passed in; nothing is
fitted here.

A28Inputs(a28, heads): the inputs of a read, with no label, episode or score (the smoke's inputs-only mode, rule
section 6 item 5).
  rows, in_rows        the read rows: the rows that are not dropped (a28.kept), sorted positions k of the A28 row space
  img, txt             the backbone's A28 features, float32, NaN outside the read rows and finite on them (asserted)
  post                 r9_heads.predict(heads, A28 features, rows): every posterior NaN outside the read rows and
                       finite on them (asserted again here); `heads` are the backbone's five heads refit on ArtELingo
                       scorer-train (r9_heads.fit_heads_r9); coef_sha256 of them
  encode(ckpt)         the factor checkpoint's codes of the read rows only, NaN elsewhere (asserted); the A3 checkpoint
                       is the backbone's own (r9_bundle.model_inputs checks its pin). T_N1u needs no scorer-train
                       side: it is the centred term of the episodes' own rows (round 6 never used the scorer-train
                       scale encode), so no ArtELingo row enters it.

A28Context(a28, eps, seed, n_per_pair, heads): the same, plus the episodes (ticket 17's
r9_a28_episodes.build_seed_a28 output, in A28 row positions) and the fields r9_bundle.build_bundle_r6 reads from a
RowContext: mode "a28", n, pooled, pair_index, parity (episode-index parity of the concatenated n_per_pair per pair, so
tune half h scores parity 1 - h), anchor, anchor_group (= the bootstrap cluster code, a28.painting_index, of the anchor),
split.groups (= a28.painting_index), episode_sha, cos, in_rows, rows, selection (= rows). Every member row of every
episode (anchor, 13 candidates, the 4 + 4 example pairs in both modalities) is asserted to be a read row, and to carry
the episode's language (a28.language at the member = eps.language[pair][i]; the episodes are language-homogeneous), so
the episodes and the features are known to share one row space (C6).

Seeds. Mode "a28" is in the one shared table r9_context.ADMITTED: the read seeds 55, 56, 57 at 4,096 episodes per pair
(rule section 5.1 item 4) and the smoke seeds 9001 to 9003 at 64 (rule section 6 item 5). RowContext does not take
mode "a28" (its MODES); A28Context does not take another mode's seeds.

Guards carry a `# guard:<name>` marker; test_r9_a28_bundle.py deletes each on a copy and shows that its scenario then
goes through.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first, before anything that imports src)
import r9_context as X  # noqa: E402
import r9_episodes as E  # noqa: E402
import r9_heads as H  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_scorers import EvalInputs, cosine_scores  # noqa: E402
from src.train.train_factors import encode_rows, load_factor_checkpoint  # noqa: E402

MODE = "a28"


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


class A28Inputs:
    """Features, posteriors, codes and their masks on the read rows of an A28 row space (see the module docstring)."""

    def __init__(self, a28, heads):
        for f in ("backbone", "n", "img", "txt", "kept", "painting_index", "language"):
            _require(hasattr(a28, f), f"a28 has no field {f!r} (load_a28's namespace)")
        n = int(a28.n)
        for m in ("img", "txt"):
            F = getattr(a28, m)
            _require(isinstance(F, np.ndarray) and F.dtype == np.float32 and F.ndim == 2 and F.shape[0] == n,
                     f"a28.{m} must be a float32 ({n}, d) matrix")
        _require(a28.img.shape == a28.txt.shape, "A28 image and caption features differ in shape")
        kept = np.asarray(a28.kept, dtype=bool)
        _require(kept.shape == (n,) and kept.any(), "a28.kept must be a boolean (n,) array with a read row")
        _require(np.asarray(a28.painting_index).shape == (n,), "a28.painting_index must have one entry per row")
        _require(np.asarray(a28.language).shape == (n,) and np.asarray(a28.language).dtype.kind in "iu",
                 "a28.language must have one integer language code (l = 1 to L) per row")
        self.a28, self.backbone, self.n_rows, self.heads = a28, a28.backbone, n, heads
        self.rows = np.flatnonzero(kept).astype(np.int64)
        self.in_rows = kept.copy()
        self.groups = np.asarray(a28.painting_index)
        self.data = SimpleNamespace(img_features=a28.img, txt_features=a28.txt)      # raw: NaN on dropped rows

        # ---- the heads are the backbone's, fitted on ArtELingo (never on A28 rows)
        for h in H.HEADS:
            for m in H.MODALITIES:
                _require(heads[h][m].coef_.shape[1] == a28.img.shape[1],
                         f"heads {h}/{m}: fitted on {heads[h][m].coef_.shape[1]}-d features, A28 "
                         f"features are {a28.img.shape[1]}-d")  # guard:a28_heads_dim
            fit_backbone = heads[h].get("fit_inputs", {}).get("backbone")      # a head without fit_inputs has none
            _require(fit_backbone == a28.backbone,
                     f"heads {h}: fitted for {fit_backbone}, A28 features are {a28.backbone}'s "
                     f"(heads need fit_inputs naming their backbone)")  # guard:a28_heads_backbone

        # ---- features, then posteriors (rule section 5 item 4)
        self.img = self.masked(a28.img, "A28 features (img)")
        self.txt = self.masked(a28.txt, "A28 features (txt)")
        self.post = H.predict(heads, self.data, self.rows)
        _require(tuple(self.post) == H.HEADS, f"posteriors of {tuple(self.post)}, not of {H.HEADS}")
        for h in H.HEADS:
            for m in H.MODALITIES:
                p = self.post[h][m]
                _require(p.dtype == np.float32 and p.shape == (n, H.N_CLASSES[h]),
                         f"post {h}/{m}: {p.dtype} {p.shape}, not float32 ({n}, {H.N_CLASSES[h]})")
                X.check_masked(p, self.in_rows, f"post {h}/{m}")  # guard:a28_post_mask
        self.coef_sha256 = H.coef_sha256(heads)

    def masked(self, values, what="masked features") -> np.ndarray:
        """float32 copy of ``values``, asserted NaN outside the read rows and finite on them. Nothing is repaired: a
        finite value on a dropped row, or a non-finite value on a read row, stops the work."""
        out = np.array(values, dtype=np.float32, copy=True)
        X.check_masked(out, self.in_rows, what)  # guard:a28_masked
        return out

    def encode(self, ckpt):
        """(img codes, txt codes) of a factor checkpoint on the read rows only (CPU), NaN elsewhere (asserted)."""
        model, _ = load_factor_checkpoint(ckpt, device="cpu")
        ic, tc = encode_rows(model, self.data.img_features, self.data.txt_features, rows=self.rows, device="cpu")
        out = []
        for codes in (ic, tc):
            _require(codes.shape[0] == len(self.rows), f"{Path(ckpt).name} codes: {codes.shape[0]} rows of codes for "
                                                       f"{len(self.rows)} read rows")
            full = np.full((self.n_rows, codes.shape[1]), np.nan, dtype=np.float32)
            full[self.rows] = codes
            X.check_masked(full, self.in_rows, f"{Path(ckpt).name} codes")  # guard:a28_codes_mask
            out.append(full)
        return out[0], out[1]


class A28Context(A28Inputs):
    """Inputs plus ticket 17's episodes: what build_bundle_r6 reads from a RowContext (see the module docstring)."""

    def __init__(self, a28, eps, seed, n_per_pair, heads):
        seed, n_per_pair = int(seed), int(n_per_pair)
        _require(X.ADMITTED.get((MODE, seed)) == n_per_pair,
                 f"{MODE} mode admits {sorted((s, n) for (m, s), n in X.ADMITTED.items() if m == MODE)} as (seed, "
                 f"episodes per pair), not ({seed}, {n_per_pair})")  # guard:a28_seed_mode
        self.mode, self.seed, self.n_per_pair = MODE, seed, n_per_pair
        self.smoke = seed in R.SMOKE_SEEDS
        super().__init__(a28, heads)
        self.split = SimpleNamespace(groups=self.groups)       # check_context reads split.groups[anchor]
        self.selection = self.rows                             # the tripwire against stored posteriors, as RowContext
        self.read_set_sha256 = getattr(a28, "read_set_sha256", None)

        # ---- the episodes (built by ticket 17 in A28 row positions)
        self.eps = eps
        self.pooled = eps.pooled
        self.n = int(eps.n)
        self.pair_index = np.asarray(eps.pair_index, dtype=np.int64)
        self.parity = np.asarray(eps.parity, dtype=np.int64)
        self.episode_sha = dict(eps.sha)
        _require(isinstance(getattr(eps, "language", None), dict) and tuple(eps.language) == R.PAIR_NAMES
                 and all(np.asarray(eps.language[p]).shape == (n_per_pair,) for p in R.PAIR_NAMES),
                 f"episodes carry no language per pair ({R.PAIR_NAMES}, {n_per_pair} codes each)")
        self.language = eps.language
        _require(int(eps.seed) == seed and self.n == len(R.PAIRS) * n_per_pair == len(self.pooled.anchor)
                 and len(eps.per_pair) == len(R.PAIRS)
                 and all(len(ep.anchor) == n_per_pair for ep in eps.per_pair)
                 and tuple(self.episode_sha) == R.PAIR_NAMES
                 and np.array_equal(self.pair_index, np.repeat(np.arange(len(R.PAIRS)), n_per_pair))
                 and np.array_equal(self.parity, np.arange(self.n) % 2),
                 f"episodes: not seed {seed}'s three pairs of {n_per_pair} (pairs in PAIRS order, hashes of "
                 f"{tuple(self.episode_sha)}, episode-index parity)")  # guard:a28_eps_shape
        member = np.asarray(self.pooled.rows())
        _require(bool(self.in_rows[member].all()),
                 f"{int((~self.in_rows[member]).sum())} episode member rows lie outside the read "
                 f"rows")  # guard:a28_episode_rows
        want = np.concatenate([np.asarray(eps.language[p], dtype=np.int64) for p in R.PAIR_NAMES])     # (n,)
        row_lang = np.asarray(a28.language, dtype=np.int64)
        bad = [int((row_lang[np.asarray(getattr(self.pooled, f)).reshape(self.n, -1)] != want[:, None]).any(axis=1).sum())
               for f in E.FIELDS]
        _require(not any(bad),
                 f"episodes with a member of another language than their own, by field {dict(zip(E.FIELDS, bad))}: "
                 f"the episodes and the A28 features are not one row space")  # guard:a28_language_join
        self.anchor = np.asarray(self.pooled.anchor, dtype=np.int64)
        self.anchor_group = self.groups[self.anchor]
        self.cos = cosine_scores(EvalInputs(self.img, self.txt), self.pooled)
