"""Rule check of round 5's draft DECISION_RULE.md (idea 3): reproduces the earlier-round constants the draft quotes and
checks the feasibility of the code paths it names. CPU only. Computes NO number of G-T, G-TF or B'_G, fits no GE head,
passes no selection caption to GoEmotions, and computes no AUC or pair statistic with a GoEmotions placement.

What it does (seed 42, earlier rounds' quantities only):
  1. round 4's bundle (r4_bundle.build_bundle(42, False)), round 3's reader on it;
  2. the reader's P against round 2's stored cand_R1_A0.npz and round 4's seed42_arrays.npz (bitwise), and the
     bs_07 AUC (emotion conditions) from the pipeline's P, from the stored P and from round 4's stored P;
  3. rc_core.thresholds on the pipeline's margins against r3_common.TAUS (==);
  4. R1's and AFF's families, cells, sigma*, and every development number the draft quotes (bar margin, margin, gain
     statistic, either change, per-pair bar margins, AFF minus R1, AFF minus B'(A1), open counts), plus
     seed42_arrays.npz equality;
  5. the draft's D5 extension with Q = Q_CLIP (the bundle's own caption posterior): stack, F, B' and the G-T / G-TF
     reader paths and tau' (all must equal AFF's);
  6. a one-modality placement function on the CLIP features (fit_one_head's recipe) against fit_one_head's posteriors
     (exact), n_iter_, held-out accuracy, the draw SHA-256 and the scorer-train position mapping;
  7. run_told_oracle.pair_stats_heads with the CLIP posteriors against told_oracle.json arms.L.pairs.heads;
  8. the row sets (selection, scorer_train) against the grid cache that run_affect.py / run_posthoc_affect.py used;
     the selection caption join's count (no model call);
  9. GoEmotions on the D3 regression sample of SCORER-TRAIN captions only (CPU), against affect_prepare.npz.
Writes rule_check/rc5_check.json only.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ROOT = Path("/project/CoSiR")
T = ROOT / "src/test"
R4DIR = T / "20261122_round4_aff_vetoes"
OUT = Path(__file__).resolve().parent / "rc5_check.json"
sys.path.insert(0, str(R4DIR))

import r4_common as R4  # noqa: E402
import r4_bundle as R4B  # noqa: E402
import r4_stats as R4S  # noqa: E402

R3, RB3, RF3, C = R4.R3, R4.RB3, R4.RF3, R4.C
K = R3.K
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_quick_checks import crossfit_condition_free, uniform_probe_scores  # noqa: E402

RES = {}
t00 = time.time()


def put(k, v):
    RES[k] = C.jsonable(v)
    print(f"[rc5] {k}: {json.dumps(RES[k]) if not isinstance(RES[k], (dict, list)) or len(json.dumps(RES[k])) < 400 else '(dict)'}",
          flush=True)


def ci3(r):
    return [r["point"], r["ci95"][0], r["ci95"][1]]


def same(x, y, nan=False):
    x, y = np.asarray(x), np.asarray(y)
    return bool(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=nan))


# ---------------------------------------------------------------- 1. bundle and reader
put("seed_guard_TEST_SEEDS_R3_R4", [list(R3.TEST_SEEDS), list(R4.TEST_SEEDS)])
for s, sm in ((42, False), (52, False), (53, False), (54, False), (49, False), (55, False), (9001, True), (9001, False)):
    try:
        R4B._check_seed(s, sm)
        ok = True
    except ValueError:
        ok = False
    put(f"seed_guard_admits_{s}_smoke{sm}", ok)

b = R4B.build_bundle(42, False)
ctx, ep = b.ctx, b.ctx.pooled
cl, pi = np.asarray(b.cl), np.asarray(b.pair_index)
put("bundle_fields_after_r4_extension_post_keys", list(b.post))
put("affect_txt_is_HEADS_cached_object", bool(b.post["affect"]["txt"] is RB3._HEADS[60000]["post"]["txt"]))
put("affect_dict_is_HEADS_cached_dict", bool(b.post["affect"] is RB3._HEADS[60000]["post"]))
rd = RF3.reader(b, readers=b.readers)

# ---------------------------------------------------------------- 2. P against stored P; the AUC
z2 = np.load(T / "20261118_reader_fix_round2/results/cand_R1_A0.npz")
s42 = np.load(R4DIR / "results/seed42_arrays.npz")
for c in CONDITIONS:
    put(f"P_pipeline_equals_cand_R1_A0_probs__{c}_bitwise", same(rd["P"][c], z2[f"probs__{c}"]))
    put(f"P_pipeline_dtype_{c}_vs_stored", [str(np.asarray(rd["P"][c]).dtype), str(z2[f"probs__{c}"].dtype)])
    put(f"P_pipeline_maxabs_vs_cand_R1_A0_{c}",
        float(np.abs(np.asarray(rd["P"][c], np.float64) - z2[f"probs__{c}"].astype(np.float64)).max()))
    put(f"P_pipeline_equals_seed42_arrays_P__{c}", same(rd["P"][c], s42[f"P__{c}"]))
    put(f"pick_pipeline_equals_cand_R1_A0_pick__{c}",
        bool(np.array_equal(np.asarray(rd["pick"][c], np.int64), z2[f"pick__{c}"].astype(np.int64))))
from sklearn.metrics import roc_auc_score  # noqa: E402

PAIRS = C.PAIRS
emo = {"a": np.array([PAIRS[i][0] == "emotion" for i in pi]), "b": np.array([PAIRS[i][1] == "emotion" for i in pi])}
yall = np.concatenate([emo["a"], emo["b"]])
put("auc_sets_positives_count", int(yall.sum()))
bs07 = json.loads((T / "20261120_r1_levers_brainstorm/results/bs_07_detector.json").read_text())
put("bs07_R1aff_auc_emotion", bs07["results"]["R1aff"]["auc_emotion"])
for nm, PP in (("pipeline", rd["P"]), ("cand_R1_A0", {c: z2[f"probs__{c}"] for c in CONDITIONS}),
               ("seed42_arrays", {c: s42[f"P__{c}"] for c in CONDITIONS})):
    auc = float(roc_auc_score(yall, np.concatenate([np.asarray(PP["a"])[:, 0], np.asarray(PP["b"])[:, 0]])))
    put(f"auc_from_{nm}", auc)
    put(f"auc_from_{nm}_equals_0.7870951145887375", bool(auc == 0.7870951145887375))

# ---------------------------------------------------------------- 3. tau
taus = R3.assert_taus()
tau_rc, n_m = K.thresholds(rd["m"])
put("rc_core_thresholds_pipeline_margins", [tau_rc, n_m])
put("rc_core_thresholds_equal_TAUS_elementwise", [bool(a == b_) for a, b_ in zip(tau_rc, taus)])
put("np_percentile_a_first_equals", [float(x) for x in np.percentile(np.concatenate([rd["m"]["a"], rd["m"]["b"]]),
                                                                      [0, 25, 50, 75])] == list(taus))
put("rc_core_thresholds_return_type", type(K.thresholds(rd["m"])).__name__)

# ---------------------------------------------------------------- 4. R1 and AFF families
g_r1 = RF3.gates_r1(rd["m"], taus)
fam_r1 = RF3.run_family(b, rd["T"], g_r1)
g_aff = RF3.gates_aff(rd["m"], rd["pick"], taus)
fa = RF3.run_family(b, rd["T"], g_aff)
put("R1_cells_fused_cf_sigma", [[int(fam_r1["fpick"][h]) for h in (0, 1)], [int(fam_r1["cpick"][h]) for h in (0, 1)],
                               [float(fam_r1["sigma"][h]) for h in (0, 1)]])
put("AFF_cells_fused_cf_sigma", [[int(fa["fpick"][h]) for h in (0, 1)], [int(fa["cpick"][h]) for h in (0, 1)],
                                [float(fa["sigma"][h]) for h in (0, 1)]])
bar_v_r1, bar_r1 = C.bar_info(fam_r1["fused"], fam_r1["cf"], b.pBp, b.pB, cl, pi)
put("R1_bar_comparator", bar_r1["comparator"])
put("R1_bar_margin", ci3(bar_r1["r1"]))
put("R1_gain_statistic", ci3(C.diff3(fam_r1["fused"], fam_r1["cf"], cl)["gain"]))
dra = R4S.dev_record("AFF", fa, None, b.pB, b.pBp, b.pBp1, cl, pi)
put("AFF_fused_r1", dra["fused_r1"])
put("AFF_cf_r1", dra["cf_r1"])
put("AFF_bar_comparator", dra["bar_comparator"])
put("AFF_bar_margin", ci3(dra["bar_margin"]))
put("AFF_margin_vs_cf", ci3(dra["margin_vs_counterpart"]))
put("AFF_gain_statistic", ci3(dra["gain_statistic"]))
put("AFF_either_change", dra["either_change"])
put("AFF_per_pair_bar", {p: dra["per_pair_bar_margin"][p]["point"] for p in C.POOLED_ORDER})
bar_v_aff, _ = C.bar_info(fa["fused"], fa["cf"], b.pBp, b.pB, cl, pi)
put("AFF_minus_R1_fused", ci3(C.point_ci(np.asarray(fa["fused"]["r1"], np.float64)
                                         - np.asarray(fam_r1["fused"]["r1"], np.float64), cl)))
put("AFF_minus_R1_bar", ci3(C.point_ci(bar_v_aff - bar_v_r1, cl)))
put("AFF_tau0_open_counts", [RF3.open_count(g_aff[0], "a"), RF3.open_count(g_aff[0], "b")])
put("Bprime_A1_mean", 100 * float(np.mean(b.pBp1["r1"])))
put("Bprime_A0_mean", 100 * float(np.mean(b.pBp["r1"])))
put("B_mean", 100 * float(np.mean(b.pB["r1"])))
put("AFF_minus_Bprime_A1", ci3(C.point_ci(np.asarray(fa["fused"]["r1"], np.float64)
                                          - np.asarray(b.pBp1["r1"], np.float64), cl)))
eq = {}
for who, fam in (("r1", fam_r1), ("aff", fa)):
    for part in ("fused", "cf"):
        for m in METRICS:
            eq[f"{who}_{part}__{m}"] = same(fam[part][m], s42[f"{who}_{part}__{m}"])
for who, g in (("r1", g_r1), ("aff", g_aff)):
    for c in CONDITIONS:
        eq[f"{who}_gate__{c}"] = same(np.stack([g[t][c] for t in range(4)]), s42[f"{who}_gate__{c}"])
put("seed42_arrays_r1_aff_equal_all", all(eq.values()))
put("seed42_arrays_r1_aff_unequal_keys", [k for k, v in eq.items() if not v])
put("either_per_gain_AFF", -dra["either_change"] / dra["gain_statistic"]["point"])
put("either_per_gain_literal", 1.629638671875 / 3.110758463541667)

# ---------------------------------------------------------------- 5. D5 with Q = Q_CLIP; G-T, G-TF paths
QC = b.post["affect"]["txt"]
heads_fp_before = R4B._fp(RB3._HEADS[60000]["post"], 0, set())
post_Q = {"affect": {"img": b.post["affect"]["img"], "txt": QC}, "image": b.post["image"], "caption": b.post["caption"]}
A0 = R3.A0
stack_Q = C.grouping_stack(post_Q, ep, A0)
F_Q, fch = R3.rbe.seed42_features(SimpleNamespace(ctx=ctx, post=post_Q), A0)
Bp_Q = crossfit_condition_free(ctx.cos, b.t_n1u, uniform_probe_scores(post_Q, ep, A0), ctx.parity)[0]
put("Q_CLIP_stack_equal", all(same(stack_Q[d], b.stack[d]) for d in DIRECTIONS))
put("Q_CLIP_F_equal", all(same(F_Q[c], b.F[c]) for c in CONDITIONS))
put("Q_CLIP_Bprime_equal", all(same(Bp_Q[c][d], b.Bp[c][d]) for c in CONDITIONS for d in DIRECTIONS))
put("Q_CLIP_pBprime_equal", all(same(per_anchor(Bp_Q)[m], b.pBp[m]) for m in METRICS))
put("HEADS_cache_unchanged", R4B._fp(RB3._HEADS[60000]["post"], 0, set()) == heads_fp_before)
rdT = RF3.reader(SimpleNamespace(F=b.F, stack=stack_Q), readers=b.readers)
rdTF = RF3.reader(SimpleNamespace(F=F_Q, stack=stack_Q), readers=b.readers)
for nm, r in (("GT", rdT), ("GTF", rdTF)):
    ok = all(same(r["P"][c], rd["P"][c]) and same(r["m"][c], rd["m"][c]) and same(r["pick"][c], rd["pick"][c])
             and all(same(r["T"][c][d], rd["T"][c][d]) for d in DIRECTIONS) for c in CONDITIONS)
    put(f"Q_CLIP_{nm}_reader_equal_AFF", ok)
tp = K.thresholds(rdTF["m"])[0]
put("Q_CLIP_tau_prime_equal_TAUS", [bool(a == b_) for a, b_ in zip(tp, taus)])
gTF = RF3.gates_aff(rdTF["m"], rdTF["pick"], tp)
put("Q_CLIP_GTF_gates_equal_AFF", all(same(gTF[t][c], g_aff[t][c]) for t in range(4) for c in CONDITIONS))
lab, comp, means = R4S.bar_comparator([("Bprime_G", per_anchor(Bp_Q)), ("Bprime_A0", b.pBp), ("counterpart", fa["cf"]),
                                       ("B", b.pB)])
put("Q_CLIP_bar_comparator_label", lab)
put("gate_dtype", str(np.asarray(g_aff[0]["a"]).dtype))

# ---------------------------------------------------------------- 6. placement function on CLIP features
from sklearn.linear_model import LogisticRegression  # noqa: E402
from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402

M = RB3.modules()
sp = artelingo_splits(ctx.data)
st = np.asarray(sp.scorer_train)
put("scorer_train_ascending_unique", bool(np.all(np.diff(st) > 0)))
put("selection_ascending_unique_and_equal_ctx", [bool(np.all(np.diff(ctx.selection) > 0)),
                                                 bool(np.array_equal(sp.selection, ctx.selection))])
put("selection_len", int(len(ctx.selection)))
put("selection_disjoint_scorer_train_held", [int(np.intersect1d(ctx.selection, st).size),
                                             int(np.intersect1d(ctx.selection, sp.held).size)])
with np.load(T / "20261111_community_told_oracle/results/per_anchor_told_oracle.npz") as zz:
    partition_L = np.asarray(zz["partition_L"], np.int64)
put("partition_L_len_classes_minsize", [int(len(partition_L)), int(partition_L.max()) + 1,
                                        int(np.bincount(partition_L).min())])
lab_g = M.rto.global_labels(partition_L, st, len(ctx.groups))


def place(Ffull, transform, rows, max_iter):
    draw = np.random.default_rng(M.rc.PROBE_SEED).choice(st, M.n6.HEAD_ROWS, replace=False)
    rest = np.setdiff1d(st, draw)
    check = np.random.default_rng(1).choice(rest, M.n6.CHECK_ROWS, replace=False)
    clf = LogisticRegression(C=1.0, max_iter=max_iter).fit(transform(Ffull[draw]), lab_g[draw])
    full = np.full((len(ctx.groups), len(clf.classes_)), np.nan, dtype=np.float32)
    full[rows] = clf.predict_proba(transform(Ffull[rows]))
    acc = 100 * float(clf.score(transform(Ffull[check]), lab_g[check]))
    return full, clf.classes_, [int(x) for x in np.atleast_1d(clf.n_iter_)], acc, draw, check


for side, Ff in (("txt", ctx.data.txt_features), ("img", ctx.data.img_features)):
    t0 = time.time()
    full, classes, n_iter, acc, draw, check = place(Ff, M.rc.unit, ctx.selection, 300)
    put(f"place_{side}_equals_fit_one_head_exact", same(full, b.post["affect"][side], nan=True))
    put(f"place_{side}_classes_0_40", bool(np.array_equal(classes, np.arange(41))))
    put(f"place_{side}_n_iter", n_iter)
    put(f"place_{side}_acc", acc)
    put(f"place_{side}_acc_equals_head_record", bool(json.loads(json.dumps(acc))
                                                     == b.affect_head["heldout_accuracy"][side]))
    put(f"place_{side}_seconds", round(time.time() - t0))
put("draw_sha256", M.rg.sha_array(np.sort(draw)))
pos = np.searchsorted(st, draw)
put("scorer_train_pos_maps_draw", bool(np.array_equal(st[pos], draw)))
put("check_rows_all_scorer_train", bool(np.isin(check, st).all()))
put("affect_head_record", b.affect_head)
told = json.loads((T / "20261111_community_told_oracle/results/told_oracle.json").read_text())
put("affect_head_record_equals_told_L_head", M.rto.roundtrip(b.affect_head) == told["arms"]["L"]["head"])

# ---------------------------------------------------------------- 7. pair_stats_heads with the CLIP posteriors
sel = ctx.selection
lab_all = artelingo_aspect_labels(ctx.data)
labS = {a: lab_all[a][sel] for a in ("emotion", "style", "genre")}
ps = M.rto.pair_stats_heads(b.post["affect"]["img"][sel].astype(np.float64),
                            b.post["affect"]["txt"][sel].astype(np.float64), labS, ctx.groups[sel])
put("pair_stats_heads_equals_told_L_heads", M.rto.roundtrip(ps) == told["arms"]["L"]["pairs"]["heads"])
put("pair_stats_heads_n_values", len(ps["by_aspect"]) + sum(len(v) for v in ps["contrast"].values()))
put("told_L_group_lift", told["arms"]["L"]["pairs"]["groups"]["lift"]["lift"])

# ---------------------------------------------------------------- 8. row sets against the grid cache; the join count
with np.load(T / "20261013_stage_d_selection/cache/prepare.npz") as zp:
    put("grid_scorer_train_equals_artelingo_splits", bool(np.array_equal(zp["scorer_train"], st)))
    put("grid_selection_equals_ctx_selection", bool(np.array_equal(zp["selection"], ctx.selection)))
from src.data.artelingo import ANNOTATIONS_PATH, join_captions  # noqa: E402

put("ANNOTATIONS_PATH", str(ANNOTATIONS_PATH))
ann = json.loads(Path(ANNOTATIONS_PATH).read_text())
caps_sel = join_captions(ctx.data.sample_ids[ctx.selection], ann)          # count only; never passed to a model
put("selection_join_count_nonempty", [int(len(caps_sel)), bool(all(isinstance(x, str) and x.strip() for x in caps_sel))])
del caps_sel

# ---------------------------------------------------------------- 9. GoEmotions on the D3 regression sample (scorer-train only, CPU)
zA = np.load(T / "20261018_affect_factor_learning/cache/affect_prepare.npz")
AP = zA["affect_probs"]
put("affect_probs_shape_dtype_min_max", [list(AP.shape), str(AP.dtype), float(AP.min()), float(AP.max())])
posr = np.sort(np.random.default_rng(5).choice(183_694, size=2_048, replace=False))
caps = join_captions(ctx.data.sample_ids[st[posr]], ann)
caps_full = join_captions(ctx.data.sample_ids[st], ann)
put("regression_join_equals_full_join_at_pos", bool(np.array_equal(caps, caps_full[posr])))
del caps_full, ann
import torch  # noqa: E402
from src.data.affect import goemotions_probabilities, load_goemotions  # noqa: E402

loaded = load_goemotions(device="cpu")
put("ge_model_device", str(next(loaded[1].parameters()).device))
ntok = np.array([len(loaded[0](x, truncation=False)["input_ids"]) for x in caps])
put("regression_sample_token_len_max_and_count_over_64", [int(ntok.max()), int((ntok > 64).sum())])
t0 = time.time()
pr = goemotions_probabilities(list(caps), loaded=loaded, batch_size=256, max_length=64)
put("ge_regression_seconds_cpu", round(time.time() - t0, 1))
d = np.abs(pr.astype(np.float64) - AP[posr].astype(np.float64))
put("ge_regression_maxabs_meanabs_count_gt_1e-5", [float(d.max()), float(d.mean()), int((d > 1e-5).sum())])
put("ge_regression_pass_1e-3", bool(d.max() <= 1e-3))
pr64 = goemotions_probabilities(list(caps), loaded=loaded, batch_size=64, max_length=64)
d2 = np.abs(pr.astype(np.float64) - pr64.astype(np.float64))
put("ge_batch256_vs_batch64_cpu_maxabs", float(d2.max()))
# misalignment sensitivity: shift by one row
dsh = np.abs(pr.astype(np.float64) - AP[np.clip(posr + 1, 0, 183_693)].astype(np.float64))
put("ge_shifted_by_one_maxabs", float(dsh.max()))
put("runtime_s", round(time.time() - t00))
OUT.write_text(json.dumps(RES, indent=1))
print("[rc5] done", flush=True)
