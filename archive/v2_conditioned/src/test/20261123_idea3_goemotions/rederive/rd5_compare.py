"""Round 5 re-derivation, phase-1 agreement: compare my hashed results with the implementation's seed-42 files under
rule §8's agreement tolerances, list every leaf of theirs without a counterpart (with its reason), and show by
deliberate perturbations that the comparator catches a 2e-9 shift, a changed Delta_k, a changed array element and a
flipped flag. Writes rederive/agreement_phase1.json (refuses to overwrite).

Run (from /project/CoSiR):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261123_idea3_goemotions/rederive/rd5_compare.py
"""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd5_paths as paths  # noqa: E402
import rd5_core as core  # noqa: E402

MINE = {"stageB": (paths.RESULTS / "rd5_stageB.json", "07e92ebdda993d6ed87e2d8571068704d24c750954c7c18d7f6dfbfb1f743ebc"),
        "stageB_arrays": (paths.RESULTS / "rd5_stageB_arrays.npz",
                          "165f4de944486e330fcc1d2b4802ced7e287aefc5a63af30d760d92098a8a2cc"),
        "stageA": (paths.RESULTS / "rd5_stageA_fix1.json",
                   "01ba2ecb7ba5888afc87b1f017ded9373436660765e340e2fcbd91e8714dc9df")}
R5 = paths.R5
THEIRS = {"carry": (R5 / "results/carry.json", "9291e3a862e84a587178cd31e4cfcfb8f3a9c6ef4ddeb7432fcdee140bcf141d"),
          "dev": (R5 / "results/dev_seed42.json", "66225589bf5e8659bda21755e4db22301b6cbb871f887adbc0806b625dfb2b7c"),
          "arrays": (R5 / "results/seed42_arrays.npz", "8dd1dd83327cd04e0f3f135c9cd3ba0fa55bf53e6e818a64d4ac5119f6eb9a14"),
          "placement": (R5 / "results/placement.json", None), "regression": (R5 / "results/regression_check.json", None),
          "ge_post": (R5 / "cache/r5_ge_posterior.npz", "081bb19e2a9b23cc5d97612f83950f77e2c9a503919e48d42fb72d94bbd7fbf7")}
PAIRS = ("emotion__style", "emotion__genre", "style__genre")
PP_TOL, TAU_ABS, TAU_REL, BOUND_EPS = 1e-9, 1e-15, 1e-9, 1e-12
N_TOTAL = 308_723
CAND_KEY = {"G-T": ("gt", "GT"), "G-TF": ("gtf", "GTF")}
CMP_NAME = {"Bprime_G": "B'_G", "Bprime_A0": "B'(A0)", "counterpart": "counterpart", "B": "B", "B_prime": "B'(A0)"}


class Cmp:
    """Agreement rows: kind in discrete / pp / tau / array; each with mine, theirs, difference and ok."""

    def __init__(self):
        self.rows = []

    def add(self, kind, name, mine, theirs, ok, diff=None, note=None):
        self.rows.append({"kind": kind, "name": name, "mine": _js(mine), "theirs": _js(theirs), "diff": diff,
                          "ok": bool(ok), **({"note": note} if note else {})})

    def disc(self, name, mine, theirs, note=None):
        self.add("discrete", name, mine, theirs, _norm(mine) == _norm(theirs), note=note)

    def pp(self, name, mine, theirs, note=None):
        d = abs(float(mine) - float(theirs))
        self.add("pp", name, mine, theirs, d <= PP_TOL, diff=d, note=note)

    def pci(self, name, mine, theirs, note=None):
        self.pp(f"{name}.point", mine["point"], theirs["point"], note)
        self.pp(f"{name}.lo", mine["ci95"][0], theirs["ci95"][0], note)
        self.pp(f"{name}.hi", mine["ci95"][1], theirs["ci95"][1], note)

    def tau(self, name, mine, theirs):
        for i, (m, t) in enumerate(zip(mine, theirs)):
            d = abs(float(m) - float(t))
            ok = d <= TAU_ABS or d <= TAU_REL * abs(float(t))
            self.add("tau", f"{name}[{i}]", m, t, ok, diff=d)
        if len(mine) != len(theirs):
            self.add("tau", f"{name}.len", len(mine), len(theirs), False)

    def arr(self, name, mine, theirs, note=None):
        m, t = np.asarray(mine), np.asarray(theirs)
        same_shape = m.shape == t.shape
        ok = same_shape and np.array_equal(m, t, equal_nan=m.dtype.kind == "f" and t.dtype.kind == "f")
        d = None
        if same_shape and m.size and m.dtype.kind in "fiub" and t.dtype.kind in "fiub":
            with np.errstate(invalid="ignore"):
                dd = np.abs(m.astype(np.float64) - t.astype(np.float64))
            dd = dd[~np.isnan(dd)]
            d = float(dd.max()) if dd.size else 0.0
        self.add("array", name, f"{m.dtype}{list(m.shape)}", f"{t.dtype}{list(t.shape)}", ok, diff=d, note=note)


def _norm(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _js(x):
    x = _norm(x)
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


class Tracked(dict):
    """A JSON dict that records which leaf paths were read."""

    def __init__(self, data, path, seen):
        super().__init__()
        self._path, self._seen = path, seen
        for k, v in data.items():
            super().__setitem__(k, _wrap(v, f"{path}/{k}", seen))

    def __getitem__(self, k):
        v = super().__getitem__(k)
        if not isinstance(v, (Tracked, TrackedList)):
            self._seen.add(f"{self._path}/{k}")
        return v


class TrackedList(list):
    def __init__(self, data, path, seen):
        super().__init__(_wrap(v, f"{path}[{i}]", seen) for i, v in enumerate(data))
        self._path, self._seen = path, seen

    def plain(self):
        self._seen.update(f"{self._path}[{i}]" for i in range(len(self)))
        for v in self:
            if isinstance(v, (Tracked, TrackedList)):
                _mark_all(v)
        return [unwrap(v) for v in self]

    def __getitem__(self, i):
        v = super().__getitem__(i)
        if not isinstance(v, (Tracked, TrackedList)):
            self._seen.add(f"{self._path}[{i}]")
        return v


def _wrap(v, path, seen):
    if isinstance(v, dict):
        return Tracked(v, path, seen)
    if isinstance(v, list):
        return TrackedList(v, path, seen)
    return v


def _mark_all(v):
    if isinstance(v, Tracked):
        for k in v:
            v[k]
            _mark_all(dict.__getitem__(v, k))
    elif isinstance(v, TrackedList):
        v.plain()


def unwrap(v):
    if isinstance(v, Tracked):
        return {k: unwrap(dict.__getitem__(v, k)) for k in v}
    if isinstance(v, TrackedList):
        return [unwrap(x) for x in list.__iter__(v)]
    return v


def pci_of(t):
    """Read a {point, ci95} leaf of theirs as plain numbers (marking the leaves seen)."""
    return {"point": t["point"], "ci95": [t["ci95"][0], t["ci95"][1]]}


def leaves(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, d


# ---------------------------------------------------------------- load

def load_inputs():
    for k, (p, sha) in MINE.items():
        if paths.sha_file(p) != sha:
            raise SystemExit(f"my {k} file changed: {p}")
    for k, (p, sha) in THEIRS.items():
        if sha and paths.sha_file(p) != sha:
            raise SystemExit(f"implementation file {p} does not have the stated SHA-256")
    for k in ("seed42_arrays", "per_anchor_seed42", "cand_rc"):
        if paths.sha_file(paths.P[k]) != paths.SHA[k]:
            raise SystemExit(f"{paths.P[k]} SHA-256 differs")
    mine = {"B": json.loads(MINE["stageB"][0].read_text()), "A": json.loads(MINE["stageA"][0].read_text()),
            "arr": dict(np.load(MINE["stageB_arrays"][0]))}
    stored = {"s4": dict(np.load(paths.P["seed42_arrays"])), "pa42": dict(np.load(paths.P["per_anchor_seed42"])),
              "rcz": dict(np.load(paths.P["cand_rc"]))}
    theirs = {k: json.loads(THEIRS[k][0].read_text()) for k in ("carry", "dev", "placement", "regression")}
    theirs["arrays"] = dict(np.load(THEIRS["arrays"][0]))
    theirs["ge_post"] = dict(np.load(THEIRS["ge_post"][0]))
    return mine, stored, theirs


# ---------------------------------------------------------------- the comparison

def compare(mine, stored, theirs_raw):
    from src.eval.aspect_metrics import cluster_bootstrap
    c = Cmp()
    seen = {k: set() for k in ("carry", "dev", "placement", "regression")}
    T = {k: Tracked(theirs_raw[k], "", seen[k]) for k in seen}
    arrs, qge = theirs_raw["arrays"], theirs_raw["ge_post"]
    B, A, ma = mine["B"], mine["A"], mine["arr"]
    s4, pa42, rcz = stored["s4"], stored["pa42"], stored["rcz"]
    cl, pair_index = s4["cl"], s4["pair_index"]
    used_arrays = set()

    def theirs_arr(k):
        used_arrays.add(k)
        return arrs[k]

    def pci_pp(values, mask=None):
        v, g = (values, cl) if mask is None else (values[mask], cl[mask])
        r = cluster_bootstrap(np.asarray(v, np.float64), g)
        return {"point": 100 * r["point"], "ci95": [100 * r["ci95"][0], 100 * r["ci95"][1]]}

    # ---- placement and Q_GE
    pl, tp = B["placement"], T["placement"]["ge_head"]
    c.disc("placement.accuracy", pl["accuracy"], tp["heldout_accuracy"])
    c.disc("placement.classes", pl["classes"], tp["classes"].plain())
    c.disc("placement.n_iter", pl["n_iter_first"], tp["n_iter"])
    c.disc("placement.n_iter_first", pl["n_iter_first"], tp["n_iter_first"])
    c.disc("placement.fallback_used", pl["fallback_used"], tp["fallback_used"])
    c.disc("placement.n_iter_fallback", pl["n_iter_fallback"], tp["n_iter_fallback"])
    c.pp("placement.check_majority_share", pl["check_majority_share"], tp["check_majority_share"])
    c.pp("placement.uniform", pl["uniform"], tp["uniform"])
    c.disc("placement.ge_posterior_sha256(record = file)", THEIRS["ge_post"][1], T["placement"]["ge_posterior_sha256"])
    ti3 = T["placement"]["item3"]
    for m in ("txt", "img"):
        c.disc(f"item3.n_iter.{m}", A["item3"][m]["n_iter"], ti3["n_iter"][m])
        c.disc(f"item3.heldout_accuracy.{m}", A["item3"][m]["accuracy"], ti3["heldout_accuracy"][m])
        c.disc(f"item3.equals_fit_one_head.{m}", A["checks"][f"item3.{m}_posterior_eq_fit_one_head"]["ok"],
               ti3["equals_fit_one_head"][m])
        c.disc(f"item3.classes_ok.{m}", A["checks"][f"item3.{m}_classes"]["ok"], ti3["classes_ok"][m])
        c.disc(f"item3.accuracy_equals_record.{m}", A["checks"][f"item3.{m}_accuracy"]["ok"],
               ti3[f"accuracy_equals_record_{m}"])
        c.disc(f"item3.clip_n_iter_expected.{m}", A["item3"][m]["n_iter"], ti3["clip_n_iter_expected_descriptive"][m])
    c.disc("item3.draw_positions_ok", A["checks"]["item3.txt_mapping"]["ok"] and A["checks"]["item3.img_mapping"]["ok"],
           ti3["draw_positions_ok"])
    c.disc("item3.head_equals_told_oracle", A["checks"]["bundle.head_identity_told_oracle"]["ok"],
           ti3["head_roundtrip_equals_told_oracle_arm_L"])
    c.disc("item3.head_equals_constants", A["checks"]["item3.fit_one_head_record"]["ok"], ti3["head_equals_constants"])
    c.disc("item3.passed", all(v["ok"] for k, v in A["checks"].items() if k.startswith("item3.")), ti3["passed"])
    c.arr("Q_GE.post_sel (selection rows, float32)", ma["Q_GE_sel"], qge["post_sel"])
    c.arr("Q_GE.rows", ma["rows"], qge["rows"])
    c.disc("Q_GE.classes", pl["classes"], qge["classes"].tolist())
    full = np.full((N_TOTAL, 41), np.nan, np.float32)
    full[qge["rows"]] = qge["post_sel"]
    c.disc("Q_GE.full array incl. NaN pattern (SHA-256 of the scattered float32 array)", pl["Q_GE_full_sha256"],
           paths.sha_array(full))
    c.disc("Q_GE.dtype", "float32", str(qge["post_sel"].dtype))

    # ---- development numbers of G-T and G-TF
    td = T["dev"]
    c.disc("dev.order", ["G-T", "G-TF"], td["order"].plain())
    for name in ("G-T", "G-TF"):
        r, t = B["records"][name], td["candidates"][name]
        c.disc(f"{name}.name", name, t["name"])
        c.pp(f"{name}.fused_r1", r["fused_r1"], t["fused_r1"])
        c.pp(f"{name}.cf_r1", r["cf_r1"], t["cf_r1"])
        for k in ("fpick", "cpick"):
            role = "fused" if k == "fpick" else "cf"
            c.disc(f"{name}.cells.{k}", [r["picks"]["half0"][role]["cell"], r["picks"]["half1"][role]["cell"]],
                   [t["cells"][k]["0"], t["cells"][k]["1"]])
        c.disc(f"{name}.sigma", [r["picks"]["half0"]["sigma"], r["picks"]["half1"]["sigma"]], [t["sigma"]["0"], t["sigma"]["1"]])
        for role in ("fused", "cf"):
            for h in (0, 1):
                mine_cell, tc = r["picks"][f"half{h}"][role], t["cell_text"][role][str(h)]
                c.disc(f"{name}.cell_text.{role}.{h}.cell", mine_cell["cell"], tc["cell"])
                c.disc(f"{name}.cell_text.{role}.{h}.tau_index", mine_cell["tau_index"], tc["tau_index"])
                c.tau(f"{name}.cell_text.{role}.{h}.tau", [mine_cell["tau"]], [tc["tau"]])
                c.disc(f"{name}.cell_text.{role}.{h}.lambda_u", mine_cell["lambda_u"], tc["lambda_u"])
                c.disc(f"{name}.cell_text.{role}.{h}.lambda_a", mine_cell["lambda_a"], tc["lambda_a"])
                c.disc(f"{name}.cell_text.{role}.{h}.k_top", 13, tc["k_top"],
                       note="k_top 13 = all 13 candidates (round 3 D8: no top-k restriction); mine has none")
        c.disc(f"{name}.bar_comparator", r["bar_comparator"], CMP_NAME[t["bar_comparator"]])
        for k, v in r["comparator_means"].items():
            inv = {v2: k2 for k2, v2 in CMP_NAME.items() if k2 != "B_prime"}
            c.pp(f"{name}.comparator_means.{k}", v, t["comparator_means"][inv[k]])
        c.pci(f"{name}.bar_margin", r["bar_margin"], pci_of(t["bar_margin"]))
        c.pci(f"{name}.margin_vs_counterpart", r["margin_vs_cf"], pci_of(t["margin_vs_counterpart"]))
        c.pci(f"{name}.gain_statistic", r["gain_statistic"], pci_of(t["gain_statistic"]))
        n_cl = int(len(np.unique(cl)))
        c.disc(f"{name}.margin_vs_counterpart.n_clusters", n_cl, t["margin_vs_counterpart"]["n_clusters"])
        c.disc(f"{name}.gain_statistic.n_clusters", n_cl, t["gain_statistic"]["n_clusters"])
        c.pp(f"{name}.either_change", r["either_change_vs_cf"], t["either_change"])
        key = CAND_KEY[name][1]
        bar_pa = {"B'(A0)": s4["Bp0__r1"], "B'_G": ma["BpG__r1"], "B": s4["B__r1"],
                  "counterpart": ma[f"{key}_cf__r1"]}[r["bar_comparator"]]
        for i, p in enumerate(PAIRS):
            tpp = t["per_pair_bar_margin"][p]
            c.pp(f"{name}.per_pair_bar_margin.{p}.point", r["per_pair_bar_margin"][p], tpp["point"])
            sup = pci_pp(ma[f"{key}_fused__r1"] - bar_pa, pair_index == i)
            c.pp(f"{name}.per_pair_bar_margin.{p}.lo", sup["ci95"][0], tpp["ci95"][0],
                 note="my interval computed at comparison time from my stage-B arrays (my record kept points only)")
            c.pp(f"{name}.per_pair_bar_margin.{p}.hi", sup["ci95"][1], tpp["ci95"][1],
                 note="my interval computed at comparison time from my stage-B arrays (my record kept points only)")
            c.pp(f"{name}.per_pair_bar_margin.{p}.point(recomputed)", sup["point"], tpp["point"])
        c.pci(f"{name}.Bprime_G_minus_Bprime_A0", B["Bprime_G"]["Bprime_G_minus_Bprime_A0"],
              pci_of(t["Bprime_G_minus_Bprime_A0"]))
        c.pp(f"{name}.beside_Bprime_A1.mean_r1", r["beside_Bp_A1"]["Bp_A1_r1"], t["beside_Bprime_A1"]["mean_r1"])
        c.pci(f"{name}.beside_Bprime_A1.candidate_minus", r["beside_Bp_A1"]["minus_Bp_A1"],
              pci_of(t["beside_Bprime_A1"]["candidate_minus"]))
        c.disc(f"{name}.delta_int", r["delta_vs_aff"]["delta_int"], t["delta_int"])
        c.pci(f"{name}.delta", r["delta_vs_aff"], pci_of(t["delta"]))
        for mk, tk in (("c1_bar_point_ge_0.5", "c1"), ("c2_bar_lower_gt_0", "c2"), ("c3_gain_lower_gt_0", "c3"),
                       ("clears", "clears")):
            c.disc(f"{name}.D10.{tk}", r["D10"][mk], t["d10"][tk])
        c.disc(f"{name}.open_tau0", [r["open_tau0"]["a"], r["open_tau0"]["b"]],
               [td["open_tau0_counts"][name]["a"], td["open_tau0_counts"][name]["b"]])
    c.tau("tau_prime", B["records"]["G-TF"]["taus"], td["tau_prime"].plain())
    c.tau("taus_AFF", B["records"]["G-T"]["taus"], td["taus_AFF"].plain())
    tcmp = td["comparators"]
    c.pp("comparators.B_mean_r1", B["records"]["G-T"]["comparator_means"]["B"], tcmp["B_mean_r1"])
    c.pp("comparators.Bprime_A0_mean_r1", B["records"]["G-T"]["comparator_means"]["B'(A0)"], tcmp["Bprime_A0_mean_r1"])
    c.pp("comparators.Bprime_G_mean_r1", B["Bprime_G"]["Bprime_G_r1"], tcmp["Bprime_G_mean_r1"])
    c.pci("comparators.Bprime_G_minus_Bprime_A0", B["Bprime_G"]["Bprime_G_minus_Bprime_A0"],
          pci_of(tcmp["Bprime_G_minus_Bprime_A0"]))
    c.pp("beside.Bprime_A1_mean_r1", A["checks"]["aff.Bp_A1_r1"]["value"], td["beside"]["Bprime_A1_mean_r1"])
    am = A["checks"]["aff.minus_Bp_A1"]["value"]
    c.pci("beside.AFF_minus_Bprime_A1", {"point": am[0], "ci95": am[1:]}, pci_of(td["beside"]["AFF_minus_Bprime_A1"]))
    c.disc("open_tau0_counts.AFF", B["aff_reference"]["open_tau0"],
           [td["open_tau0_counts"]["AFF"]["a"], td["open_tau0_counts"]["AFF"]["b"]])

    # ---- AFF's record in dev_seed42.json (my stage A values)
    ta, ck = td["aff"], A["checks"]
    c.pp("aff.fused_r1", ck["aff.fused_r1"]["value"], ta["fused_r1"])
    c.pp("aff.cf_r1", ck["aff.cf_r1"]["value"], ta["cf_r1"])
    c.disc("aff.bar_comparator", ck["aff.bar_comparator"]["value"], CMP_NAME[ta["bar_comparator"]])
    for mk, tk in (("aff.bar_margin", "bar_margin"), ("aff.margin_vs_cf", "margin_vs_counterpart"),
                   ("aff.gain_statistic", "gain_statistic")):
        v = ck[mk]["value"]
        c.pci(f"aff.{tk}", {"point": v[0], "ci95": v[1:]}, pci_of(ta[tk]))
    n_cl = int(len(np.unique(cl)))
    c.disc("aff.margin_vs_counterpart.n_clusters", n_cl, ta["margin_vs_counterpart"]["n_clusters"])
    c.disc("aff.gain_statistic.n_clusters", n_cl, ta["gain_statistic"]["n_clusters"])
    c.pp("aff.either_change", ck["aff.either_change_vs_cf"]["value"], ta["either_change"])
    for i, p in enumerate(PAIRS):
        tpp = ta["per_pair_bar_margin"][p]
        c.pp(f"aff.per_pair_bar_margin.{p}.point", ck["aff.per_pair_bar_margin"]["value"][i], tpp["point"])
        sup = pci_pp(s4["aff_fused__r1"] - s4["Bp0__r1"], pair_index == i)
        note = "interval computed at comparison time from the stored AFF and B'(A0) arrays my stage A proved equal to mine"
        c.pp(f"aff.per_pair_bar_margin.{p}.lo", sup["ci95"][0], tpp["ci95"][0], note=note)
        c.pp(f"aff.per_pair_bar_margin.{p}.hi", sup["ci95"][1], tpp["ci95"][1], note=note)
    cs = ck["aff.cells_sigma"]
    c.disc("aff.cells.fused", cs["fused"], ta["cells"]["fused"].plain())
    c.disc("aff.cells.cf", cs["cf"], ta["cells"]["cf"].plain())
    c.disc("aff.sigma", cs["sigma"], ta["sigma"].plain())
    c.disc("aff.open_tau0_counts", ck["aff.open_tau0"]["open"], [ta["open_tau0_counts"]["a"], ta["open_tau0_counts"]["b"]])
    c.pci("aff.minus_Bprime_A1", {"point": am[0], "ci95": am[1:]}, pci_of(ta["minus_Bprime_A1"]))
    pos = td["positive_check_D5"]
    for k in ("placement_is_ge", "extension_built_from_this_placement", "affect_slice_equals_independent_einsum",
              "affect_slice_differs_from_bundle_on_an_episode", "F_columns_0_to_5_differ_from_bundle_on_an_episode",
              "all_pass"):
        c.disc(f"positive_check_D5.{k}", B["positive_check"], pos[k], note="my positive check asserts all parts at once")
    for k, sha in (("goemotions_file_sha256", B["ge_file"]["sha256"]),
                   ("ge_posterior_sha256", THEIRS["ge_post"][1])):
        c.disc(f"dev.{k}", sha, td[k])
    c.disc("dev.seed42_arrays_sha256", THEIRS["arrays"][1], td["seed42_arrays_sha256"])

    # ---- arrays
    for name, (tk, mk) in CAND_KEY.items():
        rec = B["records"][name]
        for role in ("fused", "cf"):
            for m in core.METRICS:
                c.arr(f"arrays.{tk}_{role}__{m}", ma[f"{mk}_{role}__{m}"], theirs_arr(f"{tk}_{role}__{m}"))
        for cnd in core.CONDITIONS:
            c.arr(f"arrays.{tk}_gate__{cnd}", ma[f"{mk}_gate__{cnd}"], theirs_arr(f"{tk}_gate__{cnd}"))
        c.arr(f"arrays.{tk}_fused_cells", [rec["picks"]["half0"]["fused"]["cell"], rec["picks"]["half1"]["fused"]["cell"]],
              theirs_arr(f"{tk}_fused_cells"))
        c.arr(f"arrays.{tk}_cf_cells", [rec["picks"]["half0"]["cf"]["cell"], rec["picks"]["half1"]["cf"]["cell"]],
              theirs_arr(f"{tk}_cf_cells"))
        c.arr(f"arrays.{tk}_sigma", [rec["picks"]["half0"]["sigma"], rec["picks"]["half1"]["sigma"]], theirs_arr(f"{tk}_sigma"))
    for cnd in core.CONDITIONS:
        c.arr(f"arrays.gtf_P__{cnd}", ma[f"GTF_P__{cnd}"], theirs_arr(f"gtf_P__{cnd}"))
        c.arr(f"arrays.gtf_margin__{cnd}", ma[f"GTF_m__{cnd}"], theirs_arr(f"gtf_margin__{cnd}"))
        c.arr(f"arrays.gtf_pick__{cnd}", ma[f"GTF_pi__{cnd}"], theirs_arr(f"gtf_pick__{cnd}"))
        c.arr(f"arrays.aff_P__{cnd}", ma[f"GT_P__{cnd}"], theirs_arr(f"aff_P__{cnd}"), note="G-T reads AFF's reader outputs")
        c.arr(f"arrays.aff_margin__{cnd}", ma[f"GT_m__{cnd}"], theirs_arr(f"aff_margin__{cnd}"))
        c.arr(f"arrays.aff_pick__{cnd}", ma[f"GT_pi__{cnd}"], theirs_arr(f"aff_pick__{cnd}"))
    c.arr("arrays.taus", ma["GT_taus"], theirs_arr("taus"))
    c.arr("arrays.tau_prime", ma["GTF_taus"], theirs_arr("tau_prime"))
    c.tau("arrays.tau_prime(tolerance)", ma["GTF_taus"].tolist(), arrs["tau_prime"].tolist())
    for m in core.METRICS:
        c.arr(f"arrays.BpG__{m}", ma[f"BpG__{m}"], theirs_arr(f"BpG__{m}"))
    note_v = "my array = the stored round-4 seed-42 array, which my stage A asserted equal to my own (and stage B re-asserted for AFF)"
    for k in ("cl", "pair_index", "parity", "aff_gate__a", "aff_gate__b", "aff_fused_cells", "aff_cf_cells", "aff_sigma",
              *(f"aff_{r}__{m}" for r in ("fused", "cf") for m in core.METRICS),
              *(f"{s}__{m}" for s in ("B", "Bp0", "Bp1", "cosine", "rca") for m in core.METRICS)):
        c.arr(f"arrays.{k}", s4[k], theirs_arr(k), note=note_v)
    for m in core.METRICS:
        c.arr(f"arrays.cosine__{m}(per_anchor_seed42)", pa42[f"cosine__{m}"], arrs[f"cosine__{m}"])
        c.arr(f"arrays.rca__{m}(per_anchor_seed42)", pa42[f"rca__{m}"], arrs[f"rca__{m}"])

    # ---- carry
    tc, mc = T["carry"], B["carry"]
    c.disc("carry.E", mc["E"], tc["E"].plain())
    c.disc("carry.M", mc["M"], tc["M"])
    c.disc("carry.tied", mc["tied"], tc["tied"].plain())
    c.disc("carry.carried", mc["carried"], tc["carried"])
    c.disc("carry.kill", mc["decision"] == "KILL", tc["kill"])
    c.disc("carry.order", ["G-T", "G-TF"], tc["order"].plain())
    c.disc("carry.tie_band_units", 24, tc["tie_band_units"])
    for name in ("G-T", "G-TF"):
        r = B["records"][name]
        c.disc(f"carry.{name}.delta_int", r["delta_vs_aff"]["delta_int"], tc["candidates"][name]["delta_int"])
        for mk, tk in (("c1_bar_point_ge_0.5", "c1"), ("c2_bar_lower_gt_0", "c2"), ("c3_gain_lower_gt_0", "c3"),
                       ("clears", "clears")):
            c.disc(f"carry.{name}.d10.{tk}", r["D10"][mk], tc["candidates"][name]["d10"][tk])
    my_bounds = (mc["boundary"]["delta_zero"] + mc["boundary"]["gap_24"]
                 + [f"{n}.{k}" for n in ("G-T", "G-TF") for k, v in B["records"][n]["D10"]["boundary"].items() if v])
    c.disc("carry.boundaries", my_bounds, tc["boundaries"].plain())
    c.disc("carry.boundary_reported", None, tc["boundary_reported"])
    c.disc("carry.dev_seed42_sha256", THEIRS["dev"][1], tc["dev_seed42_sha256"])
    c.disc("carry.seed42_arrays_sha256", THEIRS["arrays"][1], tc["seed42_arrays_sha256"])

    # ---- regression_check.json
    reg = T["regression"]
    my_items = {1: all(v["ok"] for k, v in A["checks"].items() if k.split(".")[0] in ("bundle", "reader", "r1", "aff")
                       or k.startswith("tau_eq") or k.startswith("round1_bundle")),
                3: all(v["ok"] for k, v in A["checks"].items() if k.startswith("item3.")),
                4: all(v["ok"] for k, v in A["checks"].items() if k.startswith("item4."))}
    for i, ok in my_items.items():
        c.disc(f"regression.items.{i}.passed", ok, reg["items"][str(i)]["passed"])
    c.disc("regression.items.2.passed (my analogue: the §8 CPU spot check)", B["spot_check"]["passed"],
           reg["items"]["2"]["passed"], note="item 2 is the implementation's scorer-train sample; mine is the selection spot check")
    c.disc("regression.all_passed", A["passed"] and B["spot_check"]["passed"], reg["all_passed"])
    c.disc("regression.stopped_at_item", None, reg["stopped_at_item"])
    c.disc("regression.own_files.GOEMO_FILE_SHA", B["ge_file"]["sha256"], reg["own_files_sha256"]["GOEMO_FILE_SHA"])
    c.disc("regression.own_files.GE_POST_SHA", THEIRS["ge_post"][1], reg["own_files_sha256"]["GE_POST_SHA"])
    reg_map = regression_map(A, B, rcz)
    for j, comp in enumerate(reg["comparisons"]):
        nm = dict.__getitem__(comp, "name")
        if nm in reg_map:
            kind, mine_v = reg_map[nm]
            got = comp["got"]
            if isinstance(got, (Tracked, TrackedList)):
                _mark_all(got)
            got = unwrap(got)
            if isinstance(got, str) and got in CMP_NAME:
                got = CMP_NAME[got]
            comp["item"], comp["name"], comp["status"], comp["pass"]
            if "max_abs_diff" in comp:
                comp["max_abs_diff"]
            if isinstance(comp.get("expected"), (Tracked, TrackedList)):
                _mark_all(dict.__getitem__(comp, "expected"))
            else:
                comp["expected"]
            if kind == "pp":
                c.pp(f"regression.{nm}", mine_v, got)
            elif kind == "pci":
                c.pci(f"regression.{nm}", {"point": mine_v[0], "ci95": mine_v[1:]}, {"point": got[0], "ci95": got[1:]})
            elif kind == "tau":
                c.tau(f"regression.{nm}", mine_v, got)
            elif kind == "flag":
                c.disc(f"regression.{nm}.pass", mine_v, comp["pass"])
            else:
                c.disc(f"regression.{nm}", mine_v, got)
    unmatched_arrays = sorted(set(arrs) - used_arrays - {k for k in arrs if k.startswith(("cosine__", "rca__"))})
    return c, seen, unmatched_arrays


def regression_map(A, B, rcz):
    """Names of the implementation's regression comparisons with a counterpart in my stage A / B results."""
    ck = A["checks"]
    m = {}
    aff_keys = {"AFF.fused_r1": ("pp", ck["aff.fused_r1"]["value"]), "AFF.cf_r1": ("pp", ck["aff.cf_r1"]["value"]),
                "AFF.comparator": ("disc_name", "B'(A0)"),
                "AFF.bar_margin": ("pci", ck["aff.bar_margin"]["value"]),
                "AFF.margin_vs_counterpart": ("pci", ck["aff.margin_vs_cf"]["value"]),
                "AFF.gain_statistic": ("pci", ck["aff.gain_statistic"]["value"]),
                "AFF.either_vs_counterpart": ("pp", ck["aff.either_change_vs_cf"]["value"]),
                "AFF.aff_minus_r1_fused_r1": ("pci", ck["aff.minus_r1_fused"]["value"]),
                "AFF.aff_minus_r1_bar_margin": ("pci", ck["aff.minus_r1_bar"]["value"]),
                "AFF.fused_cells": ("disc", ck["aff.cells_sigma"]["fused"]),
                "AFF.cf_cells": ("disc", ck["aff.cells_sigma"]["cf"]),
                "AFF.sigma_star": ("disc", ck["aff.cells_sigma"]["sigma"]),
                "AFF.tau0_open_count.a": ("disc", ck["aff.open_tau0"]["open"][0]),
                "AFF.tau0_open_count.b": ("disc", ck["aff.open_tau0"]["open"][1]),
                "AFF_minus_Bprime_A1": ("pci", ck["aff.minus_Bp_A1"]["value"]),
                "Bprime_A1_mean_r1": ("pp", ck["aff.Bp_A1_r1"]["value"]),
                "R1.fused_cells": ("disc", ck["r1.cells_sigma"]["fused"]), "R1.cf_cells": ("disc", ck["r1.cells_sigma"]["cf"]),
                "R1.sigma_star": ("disc", ck["r1.cells_sigma"]["sigma"]),
                "R1.comparator": ("disc", ck["r1.bar_v_eq_stored"]["bar"]),
                "R1.bar_margin": ("pci", ck["r1.bar_margin_gain_statistic"]["bar_margin"]),
                "R1.gain_statistic": ("pci", ck["r1.bar_margin_gain_statistic"]["gain_statistic"]),
                "R1.tau_recomputed_equals_rc_tau": ("tau", A["taus"]),
                "R1.fused_r1": ("pp", 100 * float(rcz["fused__r1"].mean())),
                "R1.cf_r1": ("pp", 100 * float(rcz["cf__r1"].mean())),
                "R1.n_margins": ("disc", 24576),
                "item3_heldout_accuracy_txt": ("disc", A["item3"]["txt"]["accuracy"]),
                "item3_heldout_accuracy_img": ("disc", A["item3"]["img"]["accuracy"]),
                "goemotions_file_sha256_equals_r5_common": ("disc", B["ge_file"]["sha256"]),
                "record_npz_sha256_equals_file": ("disc", B["ge_file"]["sha256"]),
                "probs_sha256_equals_record": ("disc", B["ge_file"]["probs_sha256"]),
                "batch_size": ("disc", 256), "max_length": ("disc", 64), "n_rows": ("disc", 32413),
                "n_captions": ("disc", B["spot_check"]["n_captions_joined"]),
                "ge_classes_are_0_to_40": ("flag", B["placement"]["classes"] == list(range(41))),
                "rows_equal_ctx_selection": ("flag", B["ge_file"]["checks"]["rows_eq_ctx_selection"]),
                "ge_rows_equal_ctx_selection": ("flag", B["ge_file"]["checks"]["rows_eq_ctx_selection"]),
                "sample_ids_equal_ctx_sample_ids_at_selection": ("flag", B["ge_file"]["checks"]["sample_ids_eq_data"])}
    for k, v in aff_keys.items():
        m[k] = ("disc", v[1]) if v[0] == "disc_name" else v
    for cell in (116, 119, 58, 123, 39, 149, 10):
        t, lu, la = core.cell_params(cell)
        m[f"R1.cell_{cell}_is_the_rule_text"] = ("disc", [t, lu, la])
        m[f"AFF.cell_{cell}_is_the_rule_text"] = ("disc", [t, lu, la])
    # array-equality comparisons against stored files: my own pass flag for the same equality
    flag = lambda k: ("flag", ck[k]["ok"])  # noqa: E731
    for cnd in core.CONDITIONS:
        m[f"R1.margin__{cnd}"] = flag(f"reader.margin_{cnd}_eq_stored")
        m[f"R1.pick__{cnd}"] = flag(f"reader.pick_{cnd}_eq_stored")
        m[f"R1.gates__{cnd}"] = flag("r1.gates_eq_stored")
        for d in core.DIRECTIONS:
            m[f"R1.T__{cnd}__{d}"] = flag(f"reader.T_{cnd}_{d}_eq_stored")
    for mt in core.METRICS:
        m[f"R1.fused__{mt}"] = flag("r1.per_anchor_eq_stored")
        m[f"R1.cf__{mt}"] = flag("r1.per_anchor_eq_stored")
        for r in ("fused", "cf"):
            m[f"round4_arrays.r1_{r}__{mt}"] = flag("r1.per_anchor_eq_stored")
            m[f"round4_arrays.aff_{r}__{mt}"] = flag("aff.per_anchor_eq_stored")
    m["R1.bar_v"] = flag("r1.bar_v_eq_stored")
    for cnd in core.CONDITIONS:
        m[f"round4_arrays.r1_gate__{cnd}"] = flag("r1.gates_eq_stored")
        m[f"round4_arrays.aff_gate__{cnd}"] = flag("aff.gates_eq_stored")
    for k in ("fused_cells", "cf_cells", "sigma"):
        m[f"round4_arrays.r1_{k}"] = flag("r1.cells_sigma")
        m[f"round4_arrays.aff_{k}"] = flag("aff.cells_sigma")
    return m


# ---------------------------------------------------------------- reasons for leaves without a counterpart

def reason(path):
    rules = [
        ("/provenance/", "provenance metadata (git head, time, smoke flag)"),
        ("written", "write time of the implementation's file"), ("/time", "write time of the implementation's file"),
        ("/what", "description text"), ("rule_sha256", "the rule's SHA-256 (asserted by my code too, not a result)"),
        ("inputs_sha256", "the implementation's input-hash table (my code asserts the inputs it reads itself)"),
        ("modules_sha256", "SHA-256s of modules the implementation imports (my code imports none of rounds 2 to 5)"),
        ("regression_check_sha256", "hash of the implementation's own regression record"),
        ("/gates_checked/", "the implementation's own gate self-check flags; the gates themselves are compared array by array"),
        ("redundancy_D7", "round 3's D7 redundancy values: not part of the re-derivation's phase-1 list (§8)"),
        ("D7.", "round 3's D7 redundancy values: not part of the re-derivation's phase-1 list (§8)"),
        ("/dry_run", "run-mode flag"), ("/runtime_s", "run time"), ("/n_comparisons", "count of the implementation's own checks"),
        ("/failed", "list of the implementation's own failed checks (empty)"),
        ("/items/", "count of the implementation's own checks per item"),
        ("r3.", "round 3's bundle-equality self-checks (mine: bundle equal to round 1's load_bundle and the stored arrays, stage A)"),
        ("clip_ext.", "item-4 self-check flags (mine: item4.* checks of stage A, all passed)"),
        ("G-T.", "item-4 self-check flags (mine: item4.G-T.* checks of stage A, all passed)"),
        ("G-TF.", "item-4 self-check flags (mine: item4.G-TF.* checks of stage A, all passed)"),
        ("clip_placement_kind", "item-4 self-check flag"),
        ("(no_", "the implementation's failure-record self-check"),
        ("item2_", "the implementation's D3 scorer-train sample (rng 5, 2,048); my analogue is the §8 selection spot check"),
        ("goemotions_", "the implementation's record-presence self-check"), ("record_", "the implementation's record self-check"), ("item3_", "item-3 self-check flag (mine: item3.* of stage A)"),
        ("placement_record_present", "self-check flag"), ("ge_posterior_sha", "self-check of the implementation's constants"),
        ("ge_head_record_present", "self-check flag"), ("probs_float32_n_by_28", "self-check (mine asserted the same in load_ge_file)"),
        ("probs_finite_in_0_1", "self-check (mine asserted the same in load_ge_file)"),
        ("AFF.r5_constants_equal_round3s", "the implementation's constants vs round 3's (no computation)"),
        ("AFF.r5_cells_equal_round3s", "the implementation's constants vs round 3's (no computation)"),
        ("AFF.rule_constants_equal", "the implementation's constants vs the brainstorm's files (no computation)"),
        ("AFF.per_pair_bar_margin", "per-pair AFF points (compared in dev_seed42.json aff.per_pair_bar_margin)"),
        ("R1.stored_extra_taus", "stored-file self-consistency check"), ("R1.bar_margin_equals_stored_json", "stored-file check"),
        ("R1.gain_statistic_equals_stored_json", "stored-file check"),
        ("AFF_minus_Bprime_A1_constant_equals", "constant vs round 4's stored file"),
        ("comparison_names_equal_the_pin", "the implementation's check-list pin"),
        ("cf_gain_exactly_0", "self-check flag (mine: asserted on every episode in rd5_core.Family.assemble and rd5_stats.dev_record)"),
        ("a1.", "round 4's A1 extension self-checks (csd posteriors, A1 features, B'(A1) scores): not re-derived; my B'(A1) per anchor is round 4's stored array, which my stage A found equal to round 1's load_bundle pBp['A1']"),
        ("auc_", "item 4's AUC part: not in the re-derivation's list (§8 names item 4's tau and B' parts; the final review re-derives the diagnostics)"),
        ("pair_lift.", "item 4's pair-lift part: not in the re-derivation's list (§8 names item 4's tau and B' parts)"),
        ("external_cosine_rca_loaded", "self-check flag (mine: bundle.cosine_rca_eq_stored of stage A)"),
        ("/ge_file", "file hash"), ("dev_seed42_sha256", "hash"), ("seed42_arrays_sha256", "hash"),
    ]
    for key, why in rules:
        if key in path:
            return why
    return "UNEXPLAINED"


def unmatched_leaves(theirs_raw, seen):
    out = []
    names = [c["name"] for c in theirs_raw["regression"]["comparisons"]]
    for k in ("carry", "dev", "placement", "regression"):
        for path, val in leaves(theirs_raw[k]):
            if path not in seen[k]:
                label = path
                if k == "regression" and path.startswith("/comparisons["):
                    label = f"{path} ({names[int(path.split('[')[1].split(']')[0])]})"
                out.append({"file": k, "path": label,
                            "value": _js(val) if not isinstance(val, str) or len(val) < 80 else val[:80],
                            "reason": reason(label)})
    return out


# ---------------------------------------------------------------- perturbation test

def perturbation_tests(mine, stored, theirs_raw):
    out = {}
    base, _, _ = compare(mine, stored, theirs_raw)
    base_fail = {r["name"] for r in base.rows if not r["ok"]}

    def run(label, mutate, expect):
        t = copy.deepcopy({k: v for k, v in theirs_raw.items() if k not in ("arrays", "ge_post")})
        t["arrays"] = {k: v.copy() for k, v in theirs_raw["arrays"].items()}
        t["ge_post"] = theirs_raw["ge_post"]
        mutate(t)
        c, _, _ = compare(mine, stored, t)
        failed = sorted({r["name"] for r in c.rows if not r["ok"]} - base_fail)
        out[label] = {"new_failures": failed, "caught": expect in failed}

    run("bar margin point shifted by 2e-9 pp",
        lambda t: t["dev"]["candidates"]["G-T"]["bar_margin"].__setitem__(
            "point", t["dev"]["candidates"]["G-T"]["bar_margin"]["point"] + 2e-9), "G-T.bar_margin.point")
    run("tau' element shifted by 2e-9 relative",
        lambda t: t["dev"]["tau_prime"].__setitem__(1, t["dev"]["tau_prime"][1] * (1 + 2e-9)), "tau_prime[1]")
    run("Delta_k changed by 1 (G-TF in dev_seed42.json)",
        lambda t: (t["dev"]["candidates"]["G-TF"].__setitem__("delta_int", t["dev"]["candidates"]["G-TF"]["delta_int"] + 1)),
        "G-TF.delta_int")
    run("Delta_k changed by 1 (G-T in carry.json)",
        lambda t: (t["carry"]["candidates"]["G-T"].__setitem__("delta_int", t["carry"]["candidates"]["G-T"]["delta_int"] - 1)),
        "carry.G-T.delta_int")

    def arr_mut(t):
        a = t["arrays"]["gtf_fused__r1"]
        a[17] = a[17] + 0.25 if a[17] < 1 else a[17] - 0.25
    run("one per-anchor array element changed (gtf_fused__r1[17])", arr_mut, "arrays.gtf_fused__r1")
    run("flag flipped (G-T d10.c2)",
        lambda t: t["dev"]["candidates"]["G-T"]["d10"].__setitem__("c2", not t["dev"]["candidates"]["G-T"]["d10"]["c2"]),
        "G-T.D10.c2")
    run("flag flipped (carry kill)", lambda t: t["carry"].__setitem__("kill", not t["carry"]["kill"]), "carry.kill")
    return out


def main():
    out_path = paths.HERE / "agreement_phase1.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; refusing to overwrite")
    mine, stored, theirs = load_inputs()
    c, seen, unmatched_arrays = compare(mine, stored, theirs)
    um = unmatched_leaves(theirs, seen)
    pert = perturbation_tests(mine, stored, theirs)
    kinds = {}
    for r in c.rows:
        kinds.setdefault(r["kind"], {"n": 0, "failed": 0})
        kinds[r["kind"]]["n"] += 1
        kinds[r["kind"]]["failed"] += int(not r["ok"])
    num = [r for r in c.rows if r["kind"] in ("pp", "tau") and r["diff"] is not None]
    largest = max(num, key=lambda r: r["diff"])
    arr_max = max((r for r in c.rows if r["kind"] == "array" and r["diff"] is not None), key=lambda r: r["diff"])
    boundaries = boundary_flags(mine["B"], theirs)
    result = {"what": "rule §8 phase-1 agreement on seed 42: my re-derivation vs the implementation",
              "written": paths.now_ams(), "rule_sha256": paths.sha_file(paths.R5 / "DECISION_RULE.md"),
              "mine_sha256": {k: v[1] for k, v in MINE.items()},
              "theirs_sha256": {k: paths.sha_file(v[0]) for k, v in THEIRS.items()},
              "tolerances": {"pp": PP_TOL, "tau_abs": TAU_ABS, "tau_rel": TAU_REL, "boundary": BOUND_EPS,
                             "discrete_and_arrays": "identical"},
              "agreement": all(r["ok"] for r in c.rows), "counts": kinds, "n_compared": len(c.rows),
              "failed": [r for r in c.rows if not r["ok"]],
              "largest_numeric_difference": largest, "largest_array_difference": arr_max,
              "boundaries": boundaries, "carry_mine": mine["B"]["carry"], "carry_theirs": theirs["carry"]["kill"],
              "perturbation_tests": pert, "perturbations_all_caught": all(v["caught"] for v in pert.values()),
              "unmatched_leaves": um, "unmatched_unexplained": [u for u in um if u["reason"] == "UNEXPLAINED"],
              "unmatched_arrays": unmatched_arrays, "rows": c.rows}
    out_path.write_text(json.dumps(result, indent=1))
    print(f"agreement={result['agreement']} n={len(c.rows)} counts={kinds}")
    print(f"largest numeric diff {largest['name']} {largest['diff']}; largest array diff {arr_max['name']} {arr_max['diff']}")
    print(f"perturbations caught: {result['perturbations_all_caught']} {[(k, v['caught']) for k, v in pert.items()]}")
    print(f"unmatched leaves {len(um)}, unexplained {len(result['unmatched_unexplained'])}, unmatched arrays {unmatched_arrays}")
    print(f"boundaries {boundaries}")
    print(f"WROTE {out_path.relative_to(paths.ROOT)} sha256 {paths.sha_file(out_path)}")


def boundary_flags(B, theirs):
    """Lower bounds and D10 clauses within 1e-12 of their thresholds, Delta_k = 0, tie gap 24 (both sides)."""
    flags = []
    for name in ("G-T", "G-TF"):
        r, t = B["records"][name], theirs["dev"]["candidates"][name]
        for who, bp, blo, glo, dk in (("mine", r["bar_margin"]["point"], r["bar_margin"]["ci95"][0],
                                       r["gain_statistic"]["ci95"][0], r["delta_vs_aff"]["delta_int"]),
                                      ("theirs", t["bar_margin"]["point"], t["bar_margin"]["ci95"][0],
                                       t["gain_statistic"]["ci95"][0], t["delta_int"])):
            if abs(bp - 0.5) <= BOUND_EPS:
                flags.append(f"{who}.{name}.c1")
            if abs(blo) <= BOUND_EPS:
                flags.append(f"{who}.{name}.c2")
            if abs(glo) <= BOUND_EPS:
                flags.append(f"{who}.{name}.c3")
            if dk == 0:
                flags.append(f"{who}.{name}.delta_zero")
    E = B["carry"]["E"]
    if len(E) == 2 and abs(B["carry"]["deltas"]["G-T"] - B["carry"]["deltas"]["G-TF"]) == 24:
        flags.append("mine.gap_24")
    return flags


if __name__ == "__main__":
    main()
