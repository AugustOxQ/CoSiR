"""Round 5 re-derivation: the numerical core, written from the rules' text (round 3's D3 to D9, round 5's D5 to D7).

Grouping scores, the 18 reader features, the reader's P, T, m and pi, the gates, the 224-cell family with the nested
control sigma*, the two integer cross-fits, the assembly and the per-anchor metrics. Pure numpy (plus the allowed
`zscore_rows`); no round-2-to-5 code.

Float paths (fixed by matching the stored seed-42 arrays exactly, see rd5_stageA_report.md):
- agreements a_h(i, t) = p_h(i).p_h(t) in float32 (einsum); S, C = float32 means cast to float64; Delta = the float32
  difference S - C cast to float64; spreads = sample std (ddof 1) in float64 of the float32 agreements; match share =
  mean of 1[argmax image = argmax caption] over the 4 support pairs;
- P = mean of the two half-readers' predict_proba(scaler.transform(F)), F float64;
- T = sum_h P(h) s_h accumulated in float64 in A0 order (P float64 times float32 s_h), cast to float32.
"""
import numpy as np
import torch

CONDITIONS = ("a", "b")
DIRECTIONS = ("i2t", "t2i")
A0 = ("affect", "image", "caption")
METRICS = ("r1", "gain", "other", "swap", "strict")
NESTED_U = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
NESTED_A = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
CONTROL_SUMS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.25, 2.5, 3.0, 4.0, 4.25, 4.5, 5.0, 6.0, 8.0, 8.25, 8.5,
                9.0, 10.0, 12.0, 16.0, 16.25, 16.5, 17.0, 18.0, 20.0, 24.0, 32.0)        # round 3 D8 item 5
if sorted({u + a for u in NESTED_U for a in NESTED_A}) != list(CONTROL_SUMS):
    raise AssertionError("control sums differ from the set of lambda_u + lambda_a")
N_TAU, N_U, N_A = 4, len(NESTED_U), len(NESTED_A)
N_CELLS = N_TAU * N_U * N_A                                                        # 224


def cell_index(t: int, u: int, a: int) -> int:
    return (t * N_U + u) * N_A + a


def cell_params(cell: int) -> tuple:
    """cell -> (tau index, lambda_u, lambda_a)."""
    t, rem = divmod(int(cell), N_U * N_A)
    u, a = divmod(rem, N_A)
    return t, NESTED_U[u], NESTED_A[a]


def cell_record(cell: int, taus) -> dict:
    t, lu, la = cell_params(cell)
    return {"cell": int(cell), "tau_index": t, "tau": float(taus[t]), "lambda_u": lu, "lambda_a": la}


# ---------------------------------------------------------------- episodes

def cond_rows(ep, c):
    """(support image rows, support caption rows, contrast image rows, contrast caption rows, target column)."""
    if c == "a":
        return ep.pairs_a_img, ep.pairs_a_txt, ep.pairs_b_img, ep.pairs_b_txt, 0
    if c == "b":
        return ep.pairs_b_img, ep.pairs_b_txt, ep.pairs_a_img, ep.pairs_a_txt, 1
    raise ValueError(c)


# ---------------------------------------------------------------- grouping scores and features (round 3 D3, D4, D5)

def grouping_stack(post, ep, groupings=A0) -> dict:
    """{d: (E, H, 13) float32}: s_h(q, k) = p_h(q).p_h(k), the query's posterior from its own modality's head."""
    out = {}
    for d in DIRECTIONS:
        q, k = ("img", "txt") if d == "i2t" else ("txt", "img")
        out[d] = np.stack([np.einsum("nc,nkc->nk", post[h][q][ep.anchor], post[h][k][ep.candidates])
                           for h in groupings], axis=1)
    return out


def reader_features(post, ep, groupings=A0) -> dict:
    """{c: (E, 6H) float64}: per grouping S, C, Delta, sd_support, sd_contrast, support arg-max match share."""
    out = {}
    for c in CONDITIONS:
        si, st, ci, ct, _ = cond_rows(ep, c)
        cols = []
        for h in groupings:
            pi, pt = post[h]["img"], post[h]["txt"]
            ag_s = np.einsum("nsc,nsc->ns", pi[si], pt[st])                         # float32 (E, 4)
            ag_c = np.einsum("nsc,nsc->ns", pi[ci], pt[ct])
            s32, c32 = ag_s.mean(axis=1), ag_c.mean(axis=1)
            cols += [s32.astype(np.float64), c32.astype(np.float64), (s32 - c32).astype(np.float64),
                     ag_s.astype(np.float64).std(axis=1, ddof=1), ag_c.astype(np.float64).std(axis=1, ddof=1),
                     (pi[si].argmax(axis=-1) == pt[st].argmax(axis=-1)).mean(axis=1).astype(np.float64)]
        out[c] = np.stack(cols, axis=1)
    return out


# ---------------------------------------------------------------- reader (round 3 D5)

def half_reader_probs(halves, F) -> np.ndarray:
    ps = [h["model"].predict_proba(h["scaler"].transform(np.asarray(F, dtype=np.float64))) for h in halves]
    return (ps[0] + ps[1]) / 2


def expected_term(P, stack_d) -> np.ndarray:
    acc = P[:, 0, None] * stack_d[:, 0]                                            # float64
    for h in range(1, P.shape[1]):
        acc = acc + P[:, h, None] * stack_d[:, h]
    return acc.astype(np.float32)


def top_two_margin(P) -> np.ndarray:
    s = np.sort(P, axis=1)
    return s[:, -1] - s[:, -2]


def reader(F, stack, halves) -> dict:
    """{c: {"P", "T": {d}, "m", "pi"}} on features F (per condition) and the grouping stack."""
    out = {}
    for c in CONDITIONS:
        P = half_reader_probs(halves, F[c])
        out[c] = {"P": P, "T": {d: expected_term(P, stack[d]) for d in DIRECTIONS}, "m": top_two_margin(P),
                  "pi": np.argmax(P, axis=1)}
    return out


def taus_from_margins(m_a, m_b) -> np.ndarray:
    """numpy.percentile (linear) at 0, 25, 50, 75 of the 2E margins, condition a first, float64."""
    allm = np.concatenate([np.asarray(m_a, np.float64), np.asarray(m_b, np.float64)])
    return np.percentile(allm, [0, 25, 50, 75])


def gates_r1(m, taus) -> np.ndarray:
    return (np.asarray(m, np.float64)[None, :] >= np.asarray(taus, np.float64)[:, None]).astype(np.float32)


def gates_aff(m, pi, taus) -> np.ndarray:
    """(4, E) float32: 1[m >= tau_t] * 1[pi = affect (index 0)]."""
    return (gates_r1(m, taus).astype(bool) & (np.asarray(pi)[None, :] == 0)).astype(np.float32)


# ---------------------------------------------------------------- metrics

def first_place(S, col) -> np.ndarray:
    """bool (E,): column `col` strictly above every other candidate; ties and non-finite rows miss."""
    S = np.asarray(S)
    tgt = S[:, col:col + 1]
    others = np.delete(S, col, axis=1)
    return (others < tgt).all(axis=1) & np.isfinite(S).all(axis=1)


def per_anchor(scores) -> dict:
    """Per-episode r1, gain, other, swap, strict averaged over the two directions (round 3 §2)."""
    acc = {m: [] for m in METRICS}
    for d in DIRECTIONS:
        sa, sb = np.asarray(scores["a"][d]), np.asarray(scores["b"][d])
        aa, bb = first_place(sa, 0).astype(np.float64), first_place(sb, 1).astype(np.float64)
        ab, ba = first_place(sb, 0).astype(np.float64), first_place(sa, 1).astype(np.float64)
        r1, other = 0.5 * (aa + bb), 0.5 * (ba + ab)
        fin = np.isfinite(sa).all(axis=1) & np.isfinite(sb).all(axis=1)
        swap = ((sa[:, 0] > sa[:, 1]) & (sb[:, 1] > sb[:, 0]) & fin).astype(np.float64)
        for k, v in (("r1", r1), ("gain", r1 - other), ("other", other), ("swap", swap), ("strict", aa * bb)):
            acc[k].append(v)
    return {k: 0.5 * (v[0] + v[1]) for k, v in acc.items()}


def int4_counts(scores) -> tuple:
    """(4*R@1, 4*other) per episode as int16 from the four rankings."""
    r4 = np.zeros(len(scores["a"]["i2t"]), np.int16)
    o4 = np.zeros_like(r4)
    for d in DIRECTIONS:
        sa, sb = scores["a"][d], scores["b"][d]
        r4 += first_place(sa, 0).astype(np.int16) + first_place(sb, 1).astype(np.int16)
        o4 += first_place(sa, 1).astype(np.int16) + first_place(sb, 0).astype(np.int16)
    return r4, o4


def as_int4(values) -> np.ndarray:
    """4*x as int64, asserting every value is a multiple of 0.25."""
    v = 4.0 * np.asarray(values, np.float64)
    r = np.round(v)
    if not np.array_equal(v, r):
        raise AssertionError("per-anchor values are not multiples of 0.25")
    return r.astype(np.int64)


# ---------------------------------------------------------------- z-scores and the family (round 3 D8, D9)

def zscore(x, zscore_rows) -> np.ndarray:
    return zscore_rows(torch.as_tensor(np.asarray(x), dtype=torch.float32)).numpy().astype(np.float32, copy=False)


def condition_free(scores) -> bool:
    return all(np.array_equal(np.asarray(scores["a"][d]), np.asarray(scores["b"][d]), equal_nan=True)
               for d in DIRECTIONS)


class Family:
    """The 224 cells for one term T^c and one gate set (round 3 D8, D9), given z(B) and the parity halves."""

    def __init__(self, B, T, gates, parity, zscore_rows):
        if not condition_free(B):
            raise AssertionError("B must be condition-free")
        self.zB = {d: zscore(B["a"][d], zscore_rows) for d in DIRECTIONS}
        self.zT = {c: {d: zscore(T[c][d], zscore_rows) for d in DIRECTIONS} for c in CONDITIONS}
        self.gates = {c: np.asarray(gates[c]) for c in CONDITIONS}
        for c in CONDITIONS:
            g = self.gates[c]
            if g.dtype != np.float32 or g.shape != (N_TAU, len(parity)) or not np.isin(g, (0.0, 1.0)).all():
                raise AssertionError("gates must be float32 0/1 arrays of shape (4, E)")
        self.parity = np.asarray(parity)
        self.n = len(self.parity)

    # -------- scores
    def _base(self, d, lu):
        zb = self.zB[d]
        return zb if lu == 0 else zb + np.float32(lu) * zb

    def gated(self, c, d, t):
        return self.gates[c][t][:, None] * self.zT[c][d]

    def gcf(self, d, t):
        """G_cf,t = (g^a z(T^a) + g^b z(T^b)) / 2 in float64, cast to float32."""
        x = (self.gated("a", d, t).astype(np.float64) + self.gated("b", d, t).astype(np.float64)) / 2
        return x.astype(np.float32)

    def fused_scores(self, cell):
        t, lu, la = cell_params(cell)
        out = {c: {} for c in CONDITIONS}
        for c in CONDITIONS:
            for d in DIRECTIONS:
                s = self._base(d, lu)
                if la != 0:
                    s = s + np.float32(la) * self.gated(c, d, t)
                out[c][d] = s.astype(np.float32, copy=False)
        return out

    def cf_scores(self, cell):
        t, lu, la = cell_params(cell)
        out = {c: {} for c in CONDITIONS}
        for d in DIRECTIONS:
            s = self._base(d, lu)
            if la != 0:
                s = s + np.float32(la) * self.gcf(d, t)
            s = s.astype(np.float32, copy=False)
            out["a"][d], out["b"][d] = s, s.copy()
        return out

    def control_scores(self, sigma):
        out = {c: {} for c in CONDITIONS}
        for d in DIRECTIONS:
            s = self._base(d, sigma)
            out["a"][d], out["b"][d] = s, s.copy()
        return out

    # -------- integer statistics
    def statistics(self):
        self.f_r4 = np.zeros((N_CELLS, self.n), np.int16)
        self.f_g4 = np.zeros((N_CELLS, self.n), np.int16)
        self.c_r4 = np.zeros((N_CELLS, self.n), np.int16)
        for cell in range(N_CELLS):
            r4, o4 = int4_counts(self.fused_scores(cell))
            self.f_r4[cell], self.f_g4[cell] = r4, r4 - o4
            self.c_r4[cell] = int4_counts(self.cf_scores(cell))[0]
        self.k_r4 = np.zeros((len(CONTROL_SUMS), self.n), np.int16)
        for j, s in enumerate(CONTROL_SUMS):
            self.k_r4[j] = int4_counts(self.control_scores(s))[0]
        return self

    def crossfit(self):
        """Per tune half h: sigma*, rho_ctrl, the fused min-margin cell and the counterpart max-R@1 cell."""
        self.picks = {}
        for h in (0, 1):
            tune = self.parity == h
            rk = self.k_r4[:, tune].astype(np.int64).sum(axis=1)
            j = int(np.argmax(rk))                                  # first maximum = smallest sigma
            rho_ctrl = int(rk[j])
            rho = self.f_r4[:, tune].astype(np.int64).sum(axis=1)
            gam = self.f_g4[:, tune].astype(np.int64).sum(axis=1)
            crit = np.minimum(rho - rho_ctrl, gam)
            fused = int(np.argmax(crit))                            # ties to the lowest cell number
            rho_cf = self.c_r4[:, tune].astype(np.int64).sum(axis=1)
            cf = int(np.argmax(rho_cf))
            self.picks[h] = {"sigma": float(CONTROL_SUMS[j]), "rho_ctrl": rho_ctrl, "fused_cell": fused,
                             "fused_crit": int(crit[fused]), "cf_cell": cf, "cf_rho": int(rho_cf[cf])}
        return self.picks

    def assemble(self):
        """Assembled fused and counterpart scores (cell chosen on half h scores parity 1 - h) and their metrics."""
        shape = self.zB["i2t"].shape
        fused = {c: {d: np.empty(shape, np.float32) for d in DIRECTIONS} for c in CONDITIONS}
        cf = {c: {d: np.empty(shape, np.float32) for d in DIRECTIONS} for c in CONDITIONS}
        for h in (0, 1):
            apply = self.parity != h
            fs, cs = self.fused_scores(self.picks[h]["fused_cell"]), self.cf_scores(self.picks[h]["cf_cell"])
            for c in CONDITIONS:
                for d in DIRECTIONS:
                    fused[c][d][apply] = fs[c][d][apply]
                    cf[c][d][apply] = cs[c][d][apply]
        if not condition_free(cf):
            raise AssertionError("counterpart scores are not condition-free")
        self.fused_scores_assembled, self.cf_scores_assembled = fused, cf
        self.pa_fused, self.pa_cf = per_anchor(fused), per_anchor(cf)
        if not (self.pa_cf["gain"] == 0).all():
            raise AssertionError("counterpart gain must be 0 on every episode")
        # consistency: the assembled per-anchor R@1 equals the chosen cells' integer statistics
        r4 = np.empty(self.n, np.int64)
        for h in (0, 1):
            apply = self.parity != h
            r4[apply] = self.f_r4[self.picks[h]["fused_cell"], apply]
        if not np.array_equal(as_int4(self.pa_fused["r1"]), r4):
            raise AssertionError("assembled fused R@1 differs from the cell statistics")
        return self.pa_fused, self.pa_cf

    def run(self):
        self.statistics()
        self.crossfit()
        self.assemble()
        return self

    def summary(self, taus) -> dict:
        return {f"half{h}": {"sigma": self.picks[h]["sigma"], "rho_ctrl": self.picks[h]["rho_ctrl"],
                             "fused": cell_record(self.picks[h]["fused_cell"], taus),
                             "cf": cell_record(self.picks[h]["cf_cell"], taus)} for h in (0, 1)}


def run_family(B, T, gates, parity, zscore_rows) -> Family:
    return Family(B, T, gates, parity, zscore_rows).run()
