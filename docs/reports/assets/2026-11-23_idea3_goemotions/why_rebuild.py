"""Descriptive rebuild for section 6 of docs/reports/auto/v2/2026-11-23_idea3_goemotions.md (reader fix, round 5).

Computed after the kill (recorded 2026-10-07 18:34); seed 42 only; decides nothing. It rebuilds the seed-42 bundle and
the GE extension with the round's own code (round 4's r4_bundle.build_bundle, round 5's r5_bundle.extend, r5_fusion
.candidate and r5_diag's re-assembly), releasing round 5's guard from the stored results/regression_check.json and
minting the GE placement from the committed cache/r5_ge_posterior.npz (SHA-256 asserted by r5_guard). It writes
nothing under src/: its only output is why_rebuild.json next to this file.

Before any number is kept, the rebuild must reproduce the stored run exactly: AFF's, G-T's and G-TF's gates at every
tau index (seed42_arrays.npz), tau', B'_G's per-anchor arrays, and the fused and counterpart per-anchor arrays of all
three scorers re-assembled at their stored cells (r5_diag._reassemble asserts per_anchor equality).

What it computes (all descriptive; evaluation labels only to group rankings or rows, never inside a scorer):
  per (pair, condition): fused R@1, gain and either, candidate minus AFF, with net rankings (integers)
  fixed-cell decomposition: G-T and G-TF assembled at AFF's fused cells (39, 119); AFF at G-T's (46, 117)
  in-sample family: the 224 cells' whole-seed fused R@1 and gain, and the counterpart's R@1, for AFF, G-T, G-TF
  the affect grouping score alone (CLIP against GE placement): first-place rates of p_A and p_B per pair and direction
  redundancy: per-row correlation of z(s_affect) with z(B) and with z(cosine) (round 3's D7 definition), and of the
     gated term z(T^c) with z(B) on the values where each scorer's gate is open as scored
  pair lifts through the heads (run_told_oracle.pair_stats_heads): image x caption, caption x caption, image x image
  same-painting agreement: image of one row with the caption of another row of the same painting (different emotion)
     against different paintings (different emotion): content similarity carried by the agreement
  posterior sharpness on selection rows; B'_G's and B'(A0)'s cross-fitted weights

Run from the repo root (CPU only, one process):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python docs/reports/assets/2026-11-23_idea3_goemotions/why_rebuild.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
R5DIR = ROOT / "src/test/20261123_idea3_goemotions"
RES = R5DIR / "results"
sys.path.insert(0, str(R5DIR))

import r5_common as R5  # noqa: E402
import r5_guard as R5G  # noqa: E402
import r5_bundle as R5B  # noqa: E402
import r5_fusion as R5F  # noqa: E402
import r5_diag as R5D  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, first_place, per_anchor  # noqa: E402

R3, RB3, RB4, RF3, RTO, C = R5.R3, R5.RB3, R5.RB4, R5.RF3, R5.RTO, R5.C
TAUS = tuple(R3.TAUS)
PAIRS = ("emotion__style", "emotion__genre", "style__genre")
NAMES = ("AFF", "G-T", "G-TF")
KEY = {"AFF": "aff", "G-T": "gt", "G-TF": "gtf"}
OUT = HERE / "why_rebuild.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ci(values, cl):
    r = C.point_ci(np.asarray(values, dtype=np.float64), np.asarray(cl))
    return {"point": float(r["point"]), "ci95": [float(x) for x in r["ci95"]]}


def stack_gates(g, c):
    return np.stack([np.asarray(g[t][c]) for t in range(len(g))])


def as_int(x, mult):
    q = mult * np.asarray(x, dtype=np.float64)
    r = np.rint(q)
    assert np.array_equal(q, r), "not an integer multiple"
    return r.astype(np.int64)


def main():
    t0 = time.time()
    assert sha(R5DIR / "DECISION_RULE.md") == R5.RULE_SHA
    z = np.load(RES / "seed42_arrays.npz", allow_pickle=False)
    dev = json.loads((RES / "dev_seed42.json").read_text())
    assert dev["seed42_arrays_sha256"] == sha(RES / "seed42_arrays.npz")

    b = RB4.build_bundle(42, False)
    cl, pidx, par = np.asarray(b.cl), np.asarray(b.pair_index), np.asarray(b.parity)
    assert np.array_equal(cl, z["cl"]) and np.array_equal(pidx, z["pair_index"]) and np.array_equal(par, z["parity"])
    E = int(b.n)
    clip = R5G.clip_from_bundle(b)
    R5G.release(RES / "regression_check.json")                      # read only: items 1 to 4 recorded as passed
    ge = R5G.ge_from_file(R5DIR / "cache/r5_ge_posterior.npz")      # SHA-256 = r5_common.GE_POST_SHA asserted
    ext = R5B.extend(b, ge)
    ext_clip = R5B.extend(b, clip)                                  # = the bundle's own stack, F, B'(A0)
    for m in METRICS:
        assert np.array_equal(np.asarray(ext.pBp[m], np.float64), z[f"BpG__{m}"]), "B'_G differs from the run"
        assert np.array_equal(np.asarray(ext_clip.pBp[m], np.float64), z[f"Bp0__{m}"]), "B'(A0) differs"

    # ---- the three scorers, rebuilt and checked against the stored run
    rd = RF3.reader(b, readers=b.readers)
    g_aff = RF3.gates_aff(rd["m"], rd["pick"], TAUS)
    cand = {n: R5F.candidate(n, b, ext) for n in ("G-T", "G-TF")}
    assert tuple(cand["G-TF"]["taus"]) == tuple(float(x) for x in z["tau_prime"]), "tau' differs from the run"
    T = {"AFF": rd["T"], "G-T": cand["G-T"]["T"], "G-TF": cand["G-TF"]["T"]}
    G = {"AFF": g_aff, "G-T": cand["G-T"]["gates"], "G-TF": cand["G-TF"]["gates"]}
    P = {"AFF": rd["P"], "G-T": rd["P"], "G-TF": cand["G-TF"]["P"]}
    for n in NAMES:
        for c in CONDITIONS:
            assert np.array_equal(stack_gates(G[n], c), z[f"{KEY[n]}_gate__{c}"]), f"{n} gates differ from the run"
    fam = {n: {"fpick": {h: int(z[f"{KEY[n]}_fused_cells"][h]) for h in (0, 1)},
               "cpick": {h: int(z[f"{KEY[n]}_cf_cells"][h]) for h in (0, 1)},
               "fused": {m: z[f"{KEY[n]}_fused__{m}"] for m in METRICS},
               "cf": {m: z[f"{KEY[n]}_cf__{m}"] for m in METRICS}} for n in NAMES}
    S = {n: R5D._reassemble(b, T[n], G[n], fam[n]) for n in NAMES}  # asserts per_anchor == the stored arrays
    print(f"rebuild equals the stored run (gates, tau', B'_G, per-anchor arrays) [{time.time() - t0:.0f}s]", flush=True)

    out = {"what": "descriptive rebuild for the report's section 6 (after the kill; seed 42; decides nothing)",
           "rebuild_equals_stored_run": True}
    info = RF3.F.rank_info(b.B)
    terms = {n: RF3._terms(b, T[n], G[n]) for n in NAMES}          # zB, gated, G_cf

    # ---- (1) per (pair, condition): fused R@1, gain and either, candidate minus AFF, with net rankings
    et = {n: R5D.episode_terms(S[n]["fused"]) for n in NAMES}
    etc = {n: R5D.episode_terms(S[n]["cf"]) for n in NAMES}
    cells = {}
    for n in ("G-T", "G-TF"):
        cells[n] = {}
        tot = 0
        for i, p in enumerate(PAIRS):
            m = pidx == i
            for c in CONDITIONS:
                d_r1 = et[n]["cond"][c]["r1"] - et["AFF"]["cond"][c]["r1"]
                d_ot = et[n]["cond"][c]["other"] - et["AFF"]["cond"][c]["other"]
                net = int(as_int(d_r1[m], 2).sum())
                tot += net
                cells[n][f"{p}|{c}"] = {"r1": ci(d_r1[m], cl[m]), "gain": ci((d_r1 - d_ot)[m], cl[m]),
                                        "either": ci((d_r1 + d_ot)[m], cl[m]), "net_rankings": net}
        assert tot == dev["candidates"][n]["delta_int"], f"{n}: per-cell net rankings do not sum to Delta_k"
        cells[n]["total_net_rankings"] = tot
    out["per_cell_minus_AFF"] = cells
    out["per_cell_levels"] = {n: {f"{p}|{c}": {"fused_r1": 100 * float(np.mean(et[n]["cond"][c]["r1"][pidx == i])),
                                               "cf_r1": 100 * float(np.mean(etc[n]["cond"][c]["r1"][pidx == i]))}
                                  for i, p in enumerate(PAIRS) for c in CONDITIONS} for n in NAMES}

    # ---- (2) fixed-cell decomposition
    def at_cells(n, cells_):
        zB, gated, _ = terms[n]
        s = RF3.F.assemble(zB, info, gated, {0: int(cells_[0]), 1: int(cells_[1])}, par)
        return s, per_anchor(s)

    aff_cells = (int(z["aff_fused_cells"][0]), int(z["aff_fused_cells"][1]))
    gt_cells = (int(z["gt_fused_cells"][0]), int(z["gt_fused_cells"][1]))
    gtf_cells = (int(z["gtf_fused_cells"][0]), int(z["gtf_fused_cells"][1]))
    aff_r1 = z["aff_fused__r1"]
    fixed = {}
    for lab, n, cc in (("G-T_at_AFF_cells", "G-T", aff_cells), ("G-TF_at_AFF_cells", "G-TF", aff_cells),
                       ("AFF_at_G-T_cells", "AFF", gt_cells), ("AFF_at_G-TF_cells", "AFF", gtf_cells)):
        s, pa = at_cells(n, cc)
        e = R5D.episode_terms(s)
        row = {"cells": list(cc), "fused_r1": 100 * float(np.mean(pa["r1"])),
               "net_vs_AFF": int((as_int(pa["r1"], 4) - as_int(aff_r1, 4)).sum()),
               "minus_AFF": ci(pa["r1"] - aff_r1, cl), "gain_minus_AFF": ci(pa["gain"] - z["aff_fused__gain"], cl),
               "either_minus_AFF": ci((pa["r1"] + pa["other"]) - (z["aff_fused__r1"] + z["aff_fused__other"]), cl),
               "per_cell_net_vs_AFF": {}}
        for i, p in enumerate(PAIRS):
            m = pidx == i
            for c in CONDITIONS:
                row["per_cell_net_vs_AFF"][f"{p}|{c}"] = int(
                    as_int((e["cond"][c]["r1"] - et["AFF"]["cond"][c]["r1"])[m], 2).sum())
        fixed[lab] = row
    # own cells reproduce the stored arrays (sanity of at_cells)
    for n, cc in (("AFF", aff_cells), ("G-T", gt_cells), ("G-TF", gtf_cells)):
        assert np.array_equal(at_cells(n, cc)[1]["r1"], z[f"{KEY[n]}_fused__r1"])
    out["fixed_cells"] = fixed

    # ---- (3) in-sample family: whole-seed R@1 of each of the 224 cells
    fam_is = {}
    for n in NAMES:
        zB, gated, Gc = terms[n]
        fri, fgi, cri = RF3.F.cell_statistics(zB, info, gated, Gc, 1)
        fam_is[n] = {"fused_r1": (100 * fri.sum(1, dtype=np.int64) / (4 * E)).tolist(),
                     "fused_gain": (100 * fgi.sum(1, dtype=np.int64) / (4 * E)).tolist(),
                     "cf_r1": (100 * cri.sum(1, dtype=np.int64) / (4 * E)).tolist()}
        # the chosen cells' whole-seed values are consistent with the int statistics
        print(f"family statistics {n} done [{time.time() - t0:.0f}s]", flush=True)
    out["in_sample_family"] = {"cell_order": "cell = (tau_index*7 + u)*8 + a; lambda_u = NESTED_U[u], "
                                             "lambda_a = NESTED_A[a]", **fam_is}

    # ---- (4) the affect grouping score alone, CLIP against GE placement
    alone = {}
    for lab, st in (("CLIP", b.stack), ("GE", ext.stack)):
        alone[lab] = {}
        for d in DIRECTIONS:
            s = np.asarray(st[d][:, 0, :], dtype=np.float64)          # affect slice (A0 order: affect first)
            pa_first, pb_first = first_place(s, 0), first_place(s, 1)
            above = (s[:, 0] > s[:, 1]).astype(np.float64)
            for i, p in enumerate(PAIRS):
                m = pidx == i
                alone[lab][f"{p}|{d}"] = {"pA_first": 100 * float(pa_first[m].mean()),
                                          "pB_first": 100 * float(pb_first[m].mean()),
                                          "pA_above_pB": 100 * float(above[m].mean())}
    out["affect_score_alone"] = alone

    # ---- (5) redundancy with B and with cosine (round 3's D7 row correlation)
    red = {"CLIP_D7": RB3.redundancy(b), "GE_D7": RB3.redundancy(SimpleNamespace(B=b.B, stack=ext.stack))}
    red["with_cosine"] = {}
    for lab, st in (("CLIP", b.stack), ("GE", ext.stack)):
        red["with_cosine"][lab] = {}
        for j, h in enumerate(R3.A0):
            red["with_cosine"][lab][h] = {d: RB3.row_corr(RB3._zrows64(st[d][:, j]), RB3._zrows64(b.cos["a"][d]))
                                          for d in DIRECTIONS}
    red["B_with_cosine"] = {d: RB3.row_corr(RB3._zrows64(b.B["a"][d]), RB3._zrows64(b.cos["a"][d]))
                            for d in DIRECTIONS}
    # the gated term on the values where each scorer's gate is open as scored (cell chosen on half h scores parity 1-h)
    tau_idx = {n: (int(z[f"{KEY[n]}_fused_cells"][0]) // 56, int(z[f"{KEY[n]}_fused_cells"][1]) // 56) for n in NAMES}
    ar = np.arange(E)
    term_red = {}
    for n in NAMES:
        eff = np.where(par == 1, tau_idx[n][0], tau_idx[n][1])
        term_red[n] = {}
        for c in CONDITIONS:
            gs = stack_gates(G[n], c)[eff, ar].astype(bool)
            term_red[n][c] = {"open_as_scored": int(gs.sum())}
            for d in DIRECTIONS:
                zt = RB3._zrows64(np.asarray(T[n][c][d])[gs])
                zb = RB3._zrows64(np.asarray(b.B[c][d])[gs])
                term_red[n][c][d] = RB3.row_corr(zt, zb)
                if n != "AFF":
                    za = RB3._zrows64(np.asarray(T["AFF"][c][d])[gs])
                    term_red[n][c][f"{d}_corr_with_AFF_term"] = RB3.row_corr(zt, za)
    red["gated_term_with_B"] = term_red
    red["tau_index_as_scored"] = {n: list(v) for n, v in tau_idx.items()}
    out["redundancy"] = red

    # ---- (6) placements on the selection rows: lifts, same-painting agreement, sharpness
    from src.data.artelingo_splits import artelingo_aspect_labels
    sel = np.asarray(b.ctx.selection)
    lab_all = artelingo_aspect_labels(b.ctx.data)
    labS = {a: lab_all[a][sel] for a in ("emotion", "style", "genre")}
    gS = np.asarray(b.ctx.groups)[sel]
    Pi = np.asarray(b.post["affect"]["img"])[sel].astype(np.float64)
    Qc = np.asarray(clip.Q)[sel].astype(np.float64)
    Qg = np.asarray(ge.Q)[sel].astype(np.float64)
    lifts = {}
    for lab, (x, y) in (("image_x_caption_CLIP", (Pi, Qc)), ("image_x_caption_GE", (Pi, Qg)),
                        ("caption_x_caption_CLIP", (Qc, Qc)), ("caption_x_caption_GE", (Qg, Qg)),
                        ("image_x_image", (Pi, Pi)), ("caption_CLIP_x_caption_GE", (Qc, Qg))):
        full = RTO.pair_stats_heads(x, y, labS, gS)
        lifts[lab] = {"ratio_same_over_diff": full["by_aspect"]["ratio_same_over_diff"],
                      "mean_same": full["by_aspect"]["mean_same"], "mean_diff": full["by_aspect"]["mean_diff"],
                      "emotionxstyle": full["contrast"]["emotionxstyle"]["ratio"],
                      "emotionxgenre": full["contrast"]["emotionxgenre"]["ratio"]}
    assert lifts["image_x_caption_CLIP"]["ratio_same_over_diff"] == R5.PAIR_RATIOS["ratio_same_over_diff"]
    diag = json.loads((RES / "diagnostics_seed42.json").read_text())
    assert lifts["image_x_caption_GE"]["ratio_same_over_diff"] == diag["c_pair_lift"]["GE"]["ratio_same_over_diff"]
    out["pair_lifts"] = lifts

    # same painting (another row, another emotion) against different paintings (another emotion), image x caption
    m = labS["emotion"] >= 0
    e_, g_, pi_ = labS["emotion"][m], gS[m], Pi[m]

    def grouped(keys, A, Bm):
        inv = np.unique(np.stack(keys, 1), axis=0, return_inverse=True)[1].ravel()
        k = inv.max() + 1
        sa = np.zeros((k, A.shape[1]))
        sb = np.zeros((k, Bm.shape[1]))
        np.add.at(sa, inv, A)
        np.add.at(sb, inv, Bm)
        cnt = np.bincount(inv, minlength=k).astype(np.float64)
        return float((sa * sb).sum()), float((cnt ** 2).sum())

    same_p = {}
    for lab, q in (("CLIP", Qc[m]), ("GE", Qg[m]), ("image_x_image", pi_)):
        diag_s, diag_n = float((pi_ * q).sum()), float(len(pi_))
        all_s, all_n = float(pi_.sum(0) @ q.sum(0)), float(len(pi_)) ** 2
        gp_s, gp_n = grouped([g_], pi_, q)                    # same painting, diagonal included
        gpe_s, gpe_n = grouped([g_, e_], pi_, q)              # same painting and emotion, diagonal included
        e_s, e_n = grouped([e_], pi_, q)                      # same emotion, any painting, diagonal included
        sp_de = (gp_s - gpe_s) / (gp_n - gpe_n)               # same painting, different emotion (never the diagonal)
        dp_de = (all_s - gp_s - (e_s - gpe_s)) / (all_n - gp_n - (e_n - gpe_n))   # different painting and emotion
        same_p[lab] = {"same_row": diag_s / diag_n, "same_painting_diff_emotion": sp_de,
                       "diff_painting_diff_emotion": dp_de, "ratio_same_painting_over_diff": sp_de / dp_de,
                       "ratio_same_row_over_diff": (diag_s / diag_n) / dp_de,
                       "n_same_painting_diff_emotion_pairs": int(gp_n - gpe_n)}
    out["same_painting"] = same_p

    def sharp(x):
        p = np.clip(x, 1e-12, 1.0)
        return {"mean_max": float(x.max(1).mean()), "mean_entropy_nats": float(-(p * np.log(p)).sum(1).mean()),
                "max_entropy_nats": float(np.log(x.shape[1]))}

    out["sharpness"] = {"image_head": sharp(Pi), "caption_CLIP": sharp(Qc), "caption_GE": sharp(Qg),
                        "argmax_same_row_image_vs_caption_CLIP": 100 * float((Pi.argmax(1) == Qc.argmax(1)).mean()),
                        "argmax_same_row_image_vs_caption_GE": 100 * float((Pi.argmax(1) == Qg.argmax(1)).mean()),
                        "argmax_caption_CLIP_vs_GE": 100 * float((Qc.argmax(1) == Qg.argmax(1)).mean())}

    # ---- (7) B'_G and B'(A0): cross-fitted weights
    out["Bprime_picks"] = {"Bprime_A0": ext_clip.Bp_picks, "Bprime_G": ext.Bp_picks,
                           "order": "crossfit_condition_free weights per tune half (cosine, method-A term, probe term)"}

    out["provenance"] = {"inputs_sha256": {f: sha(RES / f) for f in ("seed42_arrays.npz", "dev_seed42.json",
                                                                     "regression_check.json", "diagnostics_seed42.json")}
                                          | {"cache/r5_ge_posterior.npz": sha(R5DIR / "cache/r5_ge_posterior.npz"),
                                             "DECISION_RULE.md": sha(R5DIR / "DECISION_RULE.md")},
                         "written": R5.now_ams(), "runtime_s": round(time.time() - t0)}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float) + "\n")
    print(f"{OUT.name} written [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
