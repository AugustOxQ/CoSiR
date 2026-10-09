"""Round 6 phase-1 re-derivation (rule §8.4, constitution C13): shared helpers.

Written without reading the round-6 implementation (no r6_*, run_r6_*, test_r6_*, prep_episodes_seed42.py,
final_review/, rule_check/, design/). Imports only what rule §8.4 allows (R3 rule §8's list without EvalContext, plus
the data loaders, the episode builder and validator, the head fitters, fit_pca_basis / fit_pair_scaler / rca_term and
fused_scores). Everything else here is own code: the selection context, the values restriction of rule §5 item 3,
cosine, the per-anchor metrics, the bootstrap draws and integer counts, Holm, the variance split.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)

ROOT = Path("/project/CoSiR")
T = ROOT / "src/test"
F = T / "20261125_artelingo_held_test"
RD = F / "rederive"
OUT = RD / "out"
E1 = T / "20261030_aspect_baselines/results"
R1DIR = T / "20261117_reader_fix_csd"
R3DIR = T / "20261121_round3_affect_gate"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAIRS = (("emotion", "style", "genre"), ("emotion", "genre", "style"), ("style", "genre", "emotion"))
POOLED = [f"{a}__{b}" for a, b, _ in PAIRS]
FIELDS = ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")
CONDS, DIRS = ("a", "b"), ("i2t", "t2i")
METRICS = ("r1", "gain", "other", "swap", "strict")
A0 = ("affect", "image", "caption")
A1 = ("affect", "image", "caption", "csd")
NESTED_U = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
NESTED_A = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
PM = ("diag", "diag_relu", "bilinear", "kissme", "xing", "wang", "probe", "tip", "value_prototype")

# SHA-256s written in the rules (R3 rule header and D15; rule F §5 item 1, §6 item 1, §6 item 4)
SHA = {
    T / "20261121_round3_affect_gate/DECISION_RULE.md": "2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925",
    T / "20261111_community_told_oracle/results/per_anchor_told_oracle.npz":
        "27e8101161607da2d6936aa5a8c84f8c296629fe08e37a68c8418c9356af1366",
    T / "20261111_community_told_oracle/results/told_oracle.json":
        "76d9ec896b941a518e7db6684805fe0bc329f6afe4a48e1993992fbb221d70d2",
    T / "20261031_pseudo_partitions/results/partitions.npz":
        "cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa",
    T / "20261108_new_method_quick_checks/results/n6_posteriors.npz":
        "2ad75026e5869c461fed156ffcdcf207c66f04a59dae514358522fbbaa9df2e0",
    T / "20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt":
        "dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2",
    T / "20261116_grouping_step1_style/results/step1_group_style.npz":
        "b04d96b4798acdfdbc9ca87436012755a681c931ac4f7612fa5683350b7a20e2",
    T / "20261116_grouping_step1_style/results/step1_heads_style.npz":
        "898a37017d82d3e130b20155e90f69e51f82f30892e04316dcb56d7aaaf8df8b",
    R1DIR / "results/rb_reader_A0.pkl": "6387b469662446734e650571c88a152651da3d66bcade28fb6cdd7828f58d34c",
    R1DIR / "results/rb_reader_A0.json": "cd90da922967cf1b62e437d3927c97cd9cfdae2c139ee392744538d3e3e5c6a8",
    R1DIR / "results/rc_tau.json": "e10cf52b2e81ea5b5f7243363d7ddf226440c3a95512cbc7e0b9ed18c5e702bf",
    E1 / "per_anchor_seed42.npz": "a4818ba0fa5f7249355afe2d2483404dcd34d22cae26984b76be787bb6e9e59d",
    E1 / "baselines_seed42.json": "ce42c81e8eec256496454e88fc07dc4fcfb02e2d5f1b043f2274ba6008564ce6",
    E1 / "episodes_seed42.npz": "12af979432ff1a20c88b614ab9e672f203eeeed9c2465bff305e72d01e6c0986",
    R3DIR / "results/seed42_arrays.npz": "5ea4b09a4161a5eac6ca78942cf0f4b99b9c634edca651fba4c2689c7c24ab8a",
    T / "20261013_stage_d_selection/cache/prepare.npz":
        "d30f0281bb521d18c7a4d4adca1a7938689e2789db7fd528ce0322b90ac44c8b",
}


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_bytes(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def need(p):
    p = Path(p)
    got = sha_file(p)
    if got != SHA[p]:
        raise AssertionError(f"{p}: SHA-256 {got} differs from the rule's {SHA[p]}")
    return p


def now_ams() -> str:
    return subprocess.run(["date", "+%F %H:%M"], env={**os.environ, "TZ": "Europe/Amsterdam"},
                          capture_output=True, text=True).stdout.strip()


T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def write_json(path, obj):
    Path(path).write_text(json.dumps(jsonable(obj), indent=1))


# ---------------------------------------------------------------- data and the selection context (own code)

def load_data():
    from src.data.artelingo import load_artelingo
    from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits
    data = load_artelingo()
    sp = artelingo_splits(data)
    labels = artelingo_aspect_labels(data)
    return data, sp, labels


def check_split(sp):
    """Rule §5 item 1's selection-side assertions (no held row is read: only selection / scorer-train equality)."""
    z = np.load(need(T / "20261013_stage_d_selection/cache/prepare.npz"))
    ok = {"groups": bool(np.array_equal(sp.groups, z["groups"])),
          "scorer_train": bool(np.array_equal(np.asarray(sp.scorer_train), z["scorer_train"])),
          "selection": bool(np.array_equal(np.asarray(sp.selection), z["selection"]))}
    if not all(ok.values()):
        raise AssertionError(f"split differs from prepare.npz: {ok}")
    return ok


class SelCtx:
    """Selection-row context: features NaN outside selection rows; the interface model_inputs / fit_heads /
    fit_one_head read (data, groups, selection, in_sel, img, txt, masked, encode)."""

    def __init__(self, data, sp):
        self.data = data
        self.groups = np.asarray(sp.groups)
        self.selection = np.asarray(sp.selection)
        self.in_sel = np.zeros(len(self.groups), dtype=bool)
        self.in_sel[self.selection] = True
        self.img = self.masked(data.img_features)
        self.txt = self.masked(data.txt_features)

    def masked(self, values):
        out = np.full(values.shape, np.nan, dtype=np.float32)
        out[self.selection] = values[self.selection]
        assert np.isnan(out[~self.in_sel]).all() and np.isfinite(out[self.in_sel]).all()
        return out

    def encode(self, ckpt):
        from src.train.train_factors import encode_rows, load_factor_checkpoint
        model, _ = load_factor_checkpoint(ckpt, device="cpu")
        ic, tc = encode_rows(model, self.data.img_features, self.data.txt_features, rows=self.selection, device="cpu")
        res = []
        for codes in (ic, tc):
            full = np.full((len(self.groups), codes.shape[1]), np.nan, dtype=np.float32)
            full[self.selection] = codes
            assert np.isnan(full[~self.in_sel]).all() and np.isfinite(full[self.in_sel]).all()
            res.append(full)
        return res[0], res[1]


def unit32(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---------------------------------------------------------------- episodes with the values restriction (own code)

def pool_rows(labels, rows):
    rows = np.asarray(rows, dtype=np.int64)
    known = (labels["emotion"][rows] >= 0) & (labels["style"][rows] >= 0) & (labels["genre"][rows] >= 0)
    return rows[known]


def dev_value_sets(labels, groups, selection):
    from src.eval.aspect_episodes import eligible_values
    pool = pool_rows(labels, selection)
    return {x: eligible_values(np.asarray(labels[x], dtype=np.int64), groups, pool, 30)
            for x in ("emotion", "style", "genre")}


def build_restricted(labels, groups, rows, a, b, n, seed, third, index, values):
    """build_aspect_episodes, unchanged, with ok_a / ok_b replaced by sorted(set(ok) & set(V)) after asserting V ⊆ ok
    (rule §5 item 3). Done by wrapping the module's eligible_values for the duration of the call; the wrapper tells
    aspect a from aspect b by the identity of the label array the builder passes (index.labels[a] / [b])."""
    import src.eval.aspect_episodes as ae
    orig = ae.eligible_values
    la, lb = index.labels[a], index.labels[b]
    va, vb = (sorted(int(v) for v in x) for x in values)
    seen = []

    def restricted(labels_x, groups_x, rows_x, min_p):
        ok = orig(labels_x, groups_x, rows_x, min_p)
        if labels_x is la:
            v, tag = va, "a"
        elif labels_x is lb:
            v, tag = vb, "b"
        else:
            raise AssertionError("eligible_values called with an unexpected label array")
        if not set(v) <= set(ok):
            raise AssertionError(f"values restriction: {tag} values {sorted(set(v) - set(ok))} not eligible")
        seen.append(tag)
        return sorted(set(ok) & set(v))

    ae.eligible_values = restricted
    try:
        ep = ae.build_aspect_episodes(labels, groups, rows, a, b, n, seed, third=third, index=index)
    finally:
        ae.eligible_values = orig
    if seen != ["a", "b"]:
        raise AssertionError(f"eligible_values calls {seen}, expected a then b")
    return ep


def concat(parts):
    from src.eval.aspect_episodes import AspectEpisodes
    return AspectEpisodes("mixed", "mixed", *(np.concatenate([getattr(p, f) for p in parts]) for f in FIELDS))


# ---------------------------------------------------------------- scores and metrics (own code)

def cosine(img_u, txt_u, ep):
    out = {}
    for d in DIRS:
        if d == "i2t":
            q, c = img_u[ep.anchor], txt_u[ep.candidates]
        else:
            q, c = txt_u[ep.anchor], img_u[ep.candidates]
        out[d] = np.einsum("nd,nkd->nk", q, c)
    return {c: {d: out[d] for d in DIRS} for c in CONDS}


def _first(s, col):
    t = s[:, col:col + 1]
    rest = np.concatenate([s[:, :col], s[:, col + 1:]], axis=1)
    return ((rest < t).all(axis=1) & np.isfinite(s).all(axis=1)).astype(np.float64)


def metrics(scores):
    """Per-anchor r1, gain, other, swap, strict: per direction, then the mean of the two directions."""
    per = {m: [] for m in METRICS}
    for d in DIRS:
        sa = np.asarray(scores["a"][d], dtype=np.float64)
        sb = np.asarray(scores["b"][d], dtype=np.float64)
        aa, bb, ab, ba = _first(sa, 0), _first(sb, 1), _first(sb, 0), _first(sa, 1)
        r1 = (aa + bb) / 2
        other = (ba + ab) / 2
        fin = np.isfinite(sa).all(axis=1) & np.isfinite(sb).all(axis=1)
        swap = ((sa[:, 0] > sa[:, 1]) & (sb[:, 1] > sb[:, 0]) & fin).astype(np.float64)
        for m, v in (("r1", r1), ("gain", r1 - other), ("other", other), ("swap", swap), ("strict", aa * bb)):
            per[m].append(v)
    return {m: (v[0] + v[1]) / 2 for m, v in per.items()}


def by_parity(full_by_half, parity):
    """full_by_half[h]: scores of the pick made on tune half h, which scores the episodes of parity 1 - h."""
    out = {c: {d: np.empty(np.asarray(full_by_half[0][c][d]).shape, np.float32) for d in DIRS} for c in CONDS}
    for h in (0, 1):
        rows = parity == 1 - h
        for c in CONDS:
            for d in DIRS:
                out[c][d][rows] = np.asarray(full_by_half[h][c][d], dtype=np.float32)[rows]
    return out


# ---------------------------------------------------------------- bootstrap, integer counts, Holm (own code)

N_BOOT, BOOT_SEED, CHUNK = 5000, 42, 250


class Draws:
    """The cluster bootstrap's resample indices (rule §3 'Draws'): clusters by numpy.unique(..., return_inverse),
    rng = default_rng(42), chunks of 250 rows of rng.integers(0, k, size=(rows, k)). Generated once and shared by
    every check, as the rule requires."""

    def __init__(self, clusters):
        _, self.idx = np.unique(np.asarray(clusters), return_inverse=True)
        self.k = int(self.idx.max()) + 1
        rng = np.random.default_rng(BOOT_SEED)
        self.chunks = []
        for start in range(0, N_BOOT, CHUNK):
            self.chunks.append(rng.integers(0, self.k, size=(min(CHUNK, N_BOOT - start), self.k)))
        self.counts = np.bincount(self.idx, minlength=self.k).astype(np.float64)

    def boots(self, values):
        values = np.asarray(values, dtype=np.float64)
        sums = np.bincount(self.idx, weights=values, minlength=self.k)
        return np.concatenate([sums[dr].sum(axis=1) / self.counts[dr].sum(axis=1) for dr in self.chunks])

    def int_sums(self, values):
        """Per resample, the integer 4·Σ (cluster sums of the drawn clusters)."""
        values = np.asarray(values, dtype=np.float64)
        v4 = np.rint(4 * values).astype(np.int64)
        if not np.array_equal(v4.astype(np.float64) / 4, values):
            raise AssertionError("a per-episode difference is not a multiple of 0.25")
        s4 = np.zeros(self.k, dtype=np.int64)
        np.add.at(s4, self.idx, v4)
        return np.concatenate([s4[dr].sum(axis=1) for dr in self.chunks])


def holm(counts, m):
    """counts in check order; order by n ascending, ties by check order; the k-th passes if it and every earlier
    one pass and 40·(n_(k)+1)·(m+1−k) ≤ 5001."""
    order = sorted(range(len(counts)), key=lambda j: (counts[j], j))
    passed, ok = [False] * len(counts), True
    level = {}
    for k, j in enumerate(order, start=1):
        own = 40 * (counts[j] + 1) * (m + 1 - k) <= 5001
        ok = ok and own
        passed[j] = ok
        level[j] = {"k": k, "own_count_passes": bool(own), "alpha_one_sided": 0.025 / (m + 1 - k)}
    return order, passed, level


def variance_split(y, clusters):
    """R3 rule §6.1 one-way decomposition by anchor painting: σ_ε² = within mean square; σ_a² = max(0, (between mean
    square − σ_ε²) / n₀), n₀ = (n − Σ m_p² / n) / (P − 1)."""
    y = np.asarray(y, dtype=np.float64)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    P = int(inv.max()) + 1
    n = len(y)
    m = np.bincount(inv, minlength=P).astype(np.float64)
    means = np.bincount(inv, weights=y, minlength=P) / m
    grand = y.mean()
    msb = float(np.sum(m * (means - grand) ** 2) / (P - 1))
    msw = float(np.sum((y - means[inv]) ** 2) / (n - P))
    n0 = float((n - np.sum(m ** 2) / n) / (P - 1))
    return {"sigma_a2": max(0.0, (msb - msw) / n0), "sigma_e2": msw, "n0": n0, "n": n, "P": P,
            "msb": msb, "msw": msw, "sum_m2": float(np.sum(m ** 2))}
