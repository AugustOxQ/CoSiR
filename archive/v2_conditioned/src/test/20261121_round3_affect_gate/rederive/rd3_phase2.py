"""Phase 2 of the round-3 independent re-derivation (rule §8), on the test seeds 49, 50, 51.

  hash     §6.2 hash check: per-pair episode SHA-256s (9 distinct, none equal to seeds 42/43/45/47/48), the three files
           per seed against results/build_seed{s}.json, codes_provenance.json against D15 -> out/phase2_hash.json
  go       per seed (bundle from rd3_bundle.py --seed s): reader, R1 and AFF gates at tau_0..3, sigma*, AFF's fused and
           counterpart cross-fits, R1's fused cross-fit only (its counterpart is never computed, §6.4), per-anchor
           arrays; pooled over 49, 50, 51 in that order with painting clusters: the seven GO checks (§6.5) and the
           secondary check (§6.6) -> out/phase2.json, out/phase2_arrays.npz
  compare  only after `go` has written its files: the implementation's results/go_seed{s}.npz and go_pooled.json
           against ours under §8's tolerances -> out/phase2_agreement.json

Computes nothing that §6.4 reserves for after the verdict: no per-seed or per-pair summaries, no bar margin or bar
comparator, nothing of R1's counterpart, no gate shares, no pick accuracy. Integer cell statistics are used to choose
cells and are not written. Nothing is printed except shapes, cells, sigma*, pass/fail and the pooled check values.
"""
import argparse
import json
import time

import numpy as np

import rd3_core as K
import rd3_family as FAM
from rd3_core import COND, DIRS, METRICS, T

SEEDS = (49, 50, 51)
EARLIER = (42, 43, 45, 47, 48)
E1 = T / "20261030_aspect_baselines/results"
R3RES = K.R3DIR / "results"
D15_BASELINES = {42: "ce42c81e8eec256496454e88fc07dc4fcfb02e2d5f1b043f2274ba6008564ce6",
                 43: "b250e89caadb1f5ad5937fb36b92ccf1f0f3b63a82b71056dc1ce2b9cc7e2859",
                 45: "ecbcf5e900a5f844af9a3e986d1154d2764bb10ba2e2123aba2aaf77917c9890",
                 47: "56e5c0661ad20372cd1c9976d2e48f07c995e44ce33f0649da5cc351f8652386",
                 48: "feed925f3eddbf2957b4bce284e1bbf5e9d85132b760f94f6d2c76cb08654286"}
D15_CODES_PROV = "8e6a517b73610d4d21b42e2eb39f6fd4118dc4132049364c8e89eecb7716bfaf"
FIELDS = ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")
CHECKS = (   # (name, fused metric, comparator, comparator metric)
    ("R1_minus_cosine", "r1", "cosine", "r1"),
    ("R1_minus_RCA", "r1", "rca", "r1"),
    ("R1_minus_B", "r1", "B", "r1"),
    ("R1_minus_Bprime", "r1", "Bprime", "r1"),
    ("R1_minus_counterpart", "r1", "AFF_cf", "r1"),
    ("gain_statistic", "gain", "AFF_cf", "gain"),
    ("gain_minus_RCA", "gain", "rca", "gain"),
)
SECONDARY = ("secondary_R1_AFF_minus_R1", "r1", "R1_fused", "r1")


def baselines_fields(s):
    """Only the hash fields of baselines_seed{s}.json (its scorer tables are not read)."""
    b = json.loads((E1 / f"baselines_seed{s}.json").read_text())
    return {"episodes_sha256": b["episodes_sha256"], "n_per_pair": b["n_per_pair"], "pair_order": b["pair_order"],
            "episodes_seed": b.get("episodes_seed")}


def cmd_hash():
    from src.eval.aspect_episodes import AspectEpisodes, episodes_sha256
    K.assert_rule()
    rec = {"seeds": {}, "earlier": {}}
    problems = []
    for s in EARLIER:
        got = K.sha_file(E1 / f"baselines_seed{s}.json")
        if got != D15_BASELINES[s]:
            problems.append(f"baselines_seed{s}.json SHA-256 {got} differs from D15")
        rec["earlier"][s] = baselines_fields(s)["episodes_sha256"]
    new = {}
    for s in SEEDS:
        K.guard_seed(s)
        bj = json.loads((R3RES / f"build_seed{s}.json").read_text())
        files = {"episodes": E1 / f"episodes_seed{s}.npz", "per_anchor": E1 / f"per_anchor_seed{s}.npz",
                 "baselines": E1 / f"baselines_seed{s}.json"}
        sha = {k: K.sha_file(p) for k, p in files.items()}
        eq = {k: sha[k] == bj["sha256"][k] for k in files}
        z = np.load(files["episodes"])
        pair = {}
        for p in K.PAIR_NAMES:
            a, b = p.split("__")
            ep = AspectEpisodes(a, b, *(z[f"{p}__{f}"].astype(np.int64) for f in FIELDS))
            pair[p] = episodes_sha256(ep)
            new[(s, p)] = pair[p]
        bf = baselines_fields(s)
        r = {"file_sha256": sha, "file_sha256_equal_build_json": eq, "pair_sha256": pair,
             "pair_sha256_equal_baselines_json": pair == bf["episodes_sha256"],
             "pair_sha256_equal_build_json": pair == bj["episode_pair_sha256"],
             "n_per_pair": bf["n_per_pair"], "n_per_pair_4096": bf["n_per_pair"] == 4096,
             "pair_order": bf["pair_order"], "pair_order_ok": list(bf["pair_order"]) == list(K.PAIR_NAMES),
             "episode_counts_4096": {p: int(len(z[f"{p}__anchor"])) == 4096 for p in K.PAIR_NAMES},
             "build_json_rule_sha256_ok": bj["rule_sha256"] == K.RULE_SHA,
             "build_json_codes_provenance": [bj["codes_provenance_before"], bj["codes_provenance_after"]],
             "build_json_passed": bj["passed"], "build_json_hash_check_pass": bj["hash_check"]["pass"]}
        for k in ("pair_sha256_equal_baselines_json", "pair_sha256_equal_build_json", "n_per_pair_4096",
                  "pair_order_ok", "build_json_rule_sha256_ok"):
            if not r[k]:
                problems.append(f"seed {s}: {k} false")
        if not all(eq.values()):
            problems.append(f"seed {s}: file SHA-256 differs from build_seed{s}.json: {eq}")
        if not all(r["episode_counts_4096"].values()):
            problems.append(f"seed {s}: episode count not 4096")
        if any(v != D15_CODES_PROV for v in r["build_json_codes_provenance"]):
            problems.append(f"seed {s}: codes_provenance before/after differs from D15")
        rec["seeds"][s] = r
    vals = list(new.values())
    rec["nine_distinct"] = len(set(vals)) == 9
    earlier_all = {v for s in EARLIER for v in rec["earlier"][s].values()}
    rec["n_earlier_pair_sha256"] = len(earlier_all)
    rec["none_equal_earlier"] = not (set(vals) & earlier_all)
    if not rec["nine_distinct"]:
        problems.append("the 9 new per-pair SHA-256s are not distinct")
    if not rec["none_equal_earlier"]:
        problems.append("a new per-pair SHA-256 equals an earlier seed's")
    cp = K.sha_file(E1 / "codes_provenance.json")
    rec["codes_provenance_sha256_now"] = cp
    rec["codes_provenance_equal_D15"] = cp == D15_CODES_PROV
    if cp != D15_CODES_PROV:
        problems.append("codes_provenance.json changed")
    rec["problems"] = problems
    rec["PASSED"] = not problems
    rec["written_amsterdam"] = K.now_ams()
    K.save_json(K.OUT / "phase2_hash.json", rec)
    print(f"hash check {'PASSED' if not problems else 'FAILED'}: {problems}")


def strip(rec):
    """Only the chosen cells (with ties count) and sigma*; no integer statistics or tune-half rates."""
    out = {"sigma_star": {h: rec["control"][h]["sigma_star"] for h in ("0", "1")},
           "fused_cells": {h: {k: v for k, v in rec["fused_cells"][h].items()
                               if k in ("cell", "tau_index", "tau", "lambda_u", "lambda_a", "n_cells_at_max",
                                        "scores_parity")} for h in ("0", "1")},
           "literal_assembly_equal": rec["literal_assembly_equal"]}
    if isinstance(rec["counterpart_cells"], dict):
        out["counterpart_cells"] = {h: {k: v for k, v in rec["counterpart_cells"][h].items()
                                        if k in ("cell", "tau_index", "tau", "lambda_u", "lambda_a", "n_cells_at_max",
                                                 "scores_parity")} for h in ("0", "1")}
        out["counterpart_condition_free_all_cells"] = rec["counterpart_condition_free_all_cells"]
    return out


def seed42_fused_only_check(BU, pk, taus):
    """The fused-only path (R1 on the test seeds) equals phase 1's full path on seed 42."""
    bd = BU.load(42)
    P, _ = K.reader_probs(pk, bd["F"])
    pm = {c: K.picks_margins(P[c]) for c in COND}
    Tr = K.weighted_term(bd["stack"], P)
    rec, pn, pc, _ = FAM.run_family(bd["B"], Tr, FAM.gates_r1({c: pm[c][1] for c in COND}, taus), taus,
                                    bd["parity"], with_counterpart=False)
    z = np.load(K.OUT / "phase1_arrays.npz")
    ok = (pc is None and [rec["fused_cells"][h]["cell"] for h in "01"] == [116, 119]
          and all(np.array_equal(pn[m], z[f"R1__fused__{m}"]) for m in METRICS))
    if not ok:
        raise AssertionError("fused-only path differs from phase 1 on seed 42")
    return True


def cmd_go():
    t0 = time.time()
    K.assert_rule()
    K.assert_inputs(["20261117_reader_fix_csd/results/rc_tau.json", "20261117_reader_fix_csd/results/rb_reader_A0.pkl",
                     "20261117_reader_fix_csd/results/rb_reader_A0.json"])
    hj = json.loads((K.OUT / "phase2_hash.json").read_text())
    if not hj["PASSED"]:
        raise SystemExit("hash check did not pass")
    import rd3_bundle as BU    # noqa: E402
    pk, _, _ = BU.rbb.load_readers("A0", False)
    taus = [float(x) for x in json.loads((T / "20261117_reader_fix_csd/results/rc_tau.json").read_text())["taus"]]
    if tuple(taus) != K.RULE_TAUS:
        raise AssertionError("rc_tau.json differs from the rule's text")
    out = {"what": "round-3 independent re-derivation, phase 2 (test seeds 49, 50, 51)", "rule_sha256": K.RULE_SHA,
           "taus": taus, "hash_check": {"PASSED": hj["PASSED"]}, "seeds": {}}
    out["seed42_fused_only_path_equals_phase1"] = seed42_fused_only_check(BU, pk, taus)
    arrays, npz = {}, {}
    for s in SEEDS:
        K.guard_seed(s)
        bjson = json.loads((K.OUT / f"rd3_bundle_seed{s}.json").read_text())
        if K.sha_file(BU.cache_path(s)) != bjson["cache_sha256"]:
            raise SystemExit(f"bundle cache seed {s} changed since it was written")
        bd = BU.load(s)
        cl, parity, pair_index = bd["cl"], bd["parity"], bd["pair_index"]
        E = len(parity)
        # §4 item 2
        pa = np.load(E1 / f"per_anchor_seed{s}.npz")
        pcos = K.metrics(bd["cos"])
        a2 = {"anchor_group_equal": bool(np.array_equal(pa["anchor_group"], cl)),
              "pair_index_equal": bool(np.array_equal(pa["pair_index"], pair_index)),
              "cosine_equal_own_per_anchor": {m: bool(np.array_equal(pcos[m], pa[f"cosine__{m}"])) for m in METRICS}}
        if not (a2["anchor_group_equal"] and a2["pair_index_equal"] and all(a2["cosine_equal_own_per_anchor"].values())):
            raise AssertionError(f"seed {s}: §4 item 2 assertion failed: {a2}")
        prca = {m: pa[f"rca__{m}"].astype(np.float64) for m in METRICS}
        pB, pBp = K.metrics(bd["B"]), K.metrics(bd["Bp"])
        P, _ = K.reader_probs(pk, bd["F"])
        pm = {c: K.picks_margins(P[c]) for c in COND}
        picks = {c: pm[c][0] for c in COND}
        margins = {c: pm[c][1] for c in COND}
        Tr = K.weighted_term(bd["stack"], P)
        gR1 = FAM.gates_r1(margins, taus)
        gA = FAM.gates_aff(margins, picks, taus)
        recA, pnA, pcA, _ = FAM.run_family(bd["B"], Tr, gA, taus, parity, with_counterpart=True)
        recR, pnR, pcR, _ = FAM.run_family(bd["B"], Tr, gR1, taus, parity, with_counterpart=False)
        assert pcR is None
        if not np.all(pcA["gain"] == 0):
            raise AssertionError(f"seed {s}: AFF counterpart gain not 0")
        out["seeds"][s] = {"n_episodes": int(E), "assert_4_item_2": a2, "bundle_picks": bjson["picks"],
                           "affect_head_equals_told_oracle_L": bjson["affect_head_equals_told_oracle_L"],
                           "AFF": strip(recA), "R1_fused_only": strip(recR),
                           "B_condition_free": True, "Bprime_condition_free": True}
        K.log(f"seed {s}: AFF fused {[recA['fused_cells'][h]['cell'] for h in '01']}, cf "
              f"{[recA['counterpart_cells'][h]['cell'] for h in '01']}, R1 fused "
              f"{[recR['fused_cells'][h]['cell'] for h in '01']}, sigma* {out['seeds'][s]['AFF']['sigma_star']}", t0)
        arrays[s] = {"AFF_fused": pnA, "AFF_cf": pcA, "R1_fused": pnR, "B": pB, "Bprime": pBp, "cosine": pcos,
                     "rca": prca, "cl": cl}
        for m in METRICS:
            for name, p in (("AFF__fused", pnA), ("AFF__cf", pcA), ("R1__fused", pnR), ("B", pB), ("Bprime", pBp),
                            ("cosine", pcos), ("rca", prca)):
                npz[f"s{s}__{name}__{m}"] = p[m]
        for c in COND:
            npz[f"s{s}__pick__{c}"] = picks[c]
            npz[f"s{s}__margin__{c}"] = margins[c]
            npz[f"s{s}__R1__gate__{c}"] = np.stack([g[c] for g in gR1])
            npz[f"s{s}__AFF__gate__{c}"] = np.stack([g[c] for g in gA])
        npz[f"s{s}__anchor_group"], npz[f"s{s}__pair_index"], npz[f"s{s}__parity"] = cl, pair_index, parity

    # ---- pooled over 49, 50, 51 in that order; clusters = anchor paintings across seeds
    cat = lambda name, m: np.concatenate([np.asarray(arrays[s][name][m], np.float64) for s in SEEDS])  # noqa: E731
    cl = np.concatenate([arrays[s]["cl"] for s in SEEDS])
    pooled = {"n_episodes": int(len(cl)), "n_clusters": int(len(np.unique(cl)))}
    cmp_name = {"cosine": "cosine", "rca": "rca", "B": "B", "Bprime": "Bprime", "AFF_cf": "AFF_cf",
                "R1_fused": "R1_fused"}
    checks = {}
    for name, fm, comp, cm in CHECKS + (SECONDARY,):
        v = cat("AFF_fused", fm) - cat(cmp_name[comp], cm)
        r = K.point_ci(v, cl)
        lo = r["ci95"][0]
        checks[name] = {"point": r["point"], "ci95": r["ci95"], "pass": bool(lo > 0),
                        "lower_within_1e-12_of_0": bool(abs(lo) <= 1e-12)}
        npz[f"pooled__{name}__diff"] = v
    pooled["checks"] = {k: checks[k] for k, *_ in CHECKS}
    pooled["secondary"] = checks[SECONDARY[0]]
    pooled["all_seven_pass"] = all(c["pass"] for c in pooled["checks"].values())
    pooled["counterpart_gain_zero_all"] = bool(np.all(cat("AFF_cf", "gain") == 0))
    out["pooled"] = pooled
    npz["pooled__clusters"] = cl
    out["written_amsterdam"] = K.now_ams()
    out["runtime_s"] = round(time.time() - t0, 1)
    K.save_json(K.OUT / "phase2.json", out)
    np.savez(K.OUT / "phase2_arrays.npz", **npz)
    for k, c in checks.items():
        K.log(f"{k}: {c['point']!r} [{c['ci95'][0]!r}, {c['ci95'][1]!r}] pass={c['pass']}", t0)
    K.log(f"all seven pass: {pooled['all_seven_pass']}", t0)


POOLED_MAP = {"R1_minus_cosine": "r1_vs_cosine", "R1_minus_RCA": "r1_vs_rca", "R1_minus_B": "r1_vs_B",
              "R1_minus_Bprime": "r1_vs_Bprime", "R1_minus_counterpart": "r1_vs_counterpart",
              "gain_statistic": "gain_statistic", "gain_minus_RCA": "gain_vs_rca"}
ARRAY_MAP = {"AFF__fused": "aff_fused", "AFF__cf": "aff_cf", "R1__fused": "r1_fused", "B": "B", "Bprime": "Bp",
             "cosine": "cosine", "rca": "rca"}


def cmd_compare():
    """Run only after `go` wrote out/phase2.json and out/phase2_arrays.npz."""
    K.assert_rule()
    mine = json.loads((K.OUT / "phase2.json").read_text())
    za = np.load(K.OUT / "phase2_arrays.npz")
    items = []

    def add(q, agree, **detail):
        items.append({"quantity": q, "agree": bool(agree), **K.jsonable(detail)})

    impl_pool = json.loads((R3RES / "go_pooled.json").read_text())
    add("go_pooled.json rule_sha256", impl_pool["provenance"]["rule_sha256"] == K.RULE_SHA)
    add("go_pooled.json seeds order", impl_pool["seeds"] == list(SEEDS), impl=impl_pool["seeds"])
    add("taus", impl_pool["taus"] == mine["taus"], impl=impl_pool["taus"], ours=mine["taus"])
    for s in SEEDS:
        zi = np.load(R3RES / f"go_seed{s}.npz")
        zr = np.load(R3RES / f"cache_reader_seed{s}.npz")
        zc = np.load(R3RES / f"cache_seed{s}.npz")
        meta = json.loads(str(zi["meta"]))
        add(f"seed {s}: go npz meta seed / smoke / rule", meta["seed"] == s and meta["smoke"] is False
            and meta["rule_sha256"] == K.RULE_SHA)
        add(f"seed {s}: cache file SHA-256 equal go meta", K.sha_file(R3RES / f"cache_seed{s}.npz") == meta["cache_sha256"]
            and K.sha_file(R3RES / f"cache_reader_seed{s}.npz") == meta["cache_reader_sha256"])
        for ours, theirs in (("anchor_group", "cl"), ("pair_index", "pair_index"), ("parity", "parity")):
            add(f"seed {s}: {ours}", np.array_equal(za[f"s{s}__{ours}"], zi[theirs]))
        sd = mine["seeds"][str(s)]
        for name, key, ours in (("AFF fused cells", "aff_fused_cells", sd["AFF"]["fused_cells"]),
                                ("AFF counterpart cells", "aff_cf_cells", sd["AFF"]["counterpart_cells"]),
                                ("R1 fused cells", "r1_fused_cells", sd["R1_fused_only"]["fused_cells"])):
            o = [ours["0"]["cell"], ours["1"]["cell"]]
            add(f"seed {s}: {name} (tune half 0, 1)", o == [int(x) for x in zi[key]], ours=o, impl=zi[key])
        o = [sd["AFF"]["sigma_star"]["0"], sd["AFF"]["sigma_star"]["1"]]
        add(f"seed {s}: sigma* (tune half 0, 1)", o == [float(x) for x in zi["sigma"]]
            and sd["R1_fused_only"]["sigma_star"] == sd["AFF"]["sigma_star"], ours=o, impl=zi["sigma"])
        for ours, theirs in ARRAY_MAP.items():
            for m in METRICS:
                a, b = za[f"s{s}__{ours}__{m}"], zi[f"{theirs}__{m}"]
                add(f"seed {s}: per-anchor {ours}.{m}", a.dtype == b.dtype and np.array_equal(a, b),
                    n_diff=int((a != b).sum()))
        for c in COND:
            add(f"seed {s}: picks {c}", np.array_equal(za[f"s{s}__pick__{c}"], zr[f"pick__{c}"]),
                n_diff=int((za[f"s{s}__pick__{c}"] != zr[f"pick__{c}"]).sum()))
            add(f"seed {s}: margins {c}", np.array_equal(za[f"s{s}__margin__{c}"], zr[f"m__{c}"]))
            # the implementation stores the gate inputs (margins, picks), not the gates: apply D6 to them
            taus = mine["taus"]
            g1 = np.stack([zr[f"m__{c}"] >= np.float64(t) for t in taus])
            ga = g1 & (zr[f"pick__{c}"] == 0)[None, :]
            add(f"seed {s}: R1 gates {c}, tau_0..3 (D6 on the implementation's margins)",
                np.array_equal(za[f"s{s}__R1__gate__{c}"], g1))
            add(f"seed {s}: AFF gates {c}, tau_0..3 (D6 on the implementation's margins and picks)",
                np.array_equal(za[f"s{s}__AFF__gate__{c}"], ga))
        # bundle pieces (scores, features, grouping scores, reader probabilities, term)
        bd = np.load(K.OUT / f"rd3_bundle_seed{s}.npz")
        pieces = [(f"{k}__{c}__{d}", f"{k}__{c}__{d}") for k in ("cos", "B") for c in COND for d in DIRS]
        pieces += [(f"Bp__{c}__{d}", f"Bp__{c}__{d}") for c in COND for d in DIRS]
        pieces += [(f"F__{c}", f"F__{c}") for c in COND] + [(f"stack__{d}", f"stack__{d}") for d in DIRS]
        for ours, theirs in pieces:
            add(f"seed {s}: bundle {ours}", bd[ours].dtype == zc[theirs].dtype and np.array_equal(bd[ours], zc[theirs]))
        add(f"seed {s}: anchors", np.array_equal(bd["anchor"], zc["anchor"]))
    pooled = mine["pooled"]
    add("pooled n_episodes", pooled["n_episodes"] == impl_pool["n_episodes_pooled"])
    add("pooled n_clusters", pooled["n_clusters"] == impl_pool["n_clusters_pooled"], ours=pooled["n_clusters"])
    allchk = {**pooled["checks"], SECONDARY[0]: pooled["secondary"]}
    implchk = {**{k: impl_pool["checks"][v] for k, v in POOLED_MAP.items()}, SECONDARY[0]: impl_pool["secondary"]}
    near_zero = []
    for k, o in allchk.items():
        i = implchk[k]
        vo = [o["point"], o["ci95"][0], o["ci95"][1]]
        vi = [i["point"], i["ci95"][0], i["ci95"][1]]
        d = [abs(a - b) for a, b in zip(vo, vi)]
        add(f"pooled {k}: point and 95% bounds within 1e-9 pp", all(x <= 1e-9 for x in d), ours=vo, impl=vi,
            abs_diff=d, exact=[a == b for a, b in zip(vo, vi)])
        add(f"pooled {k}: pass/fail", o["pass"] == i["pass"], ours=o["pass"], impl=i["pass"])
        for who, v in (("ours", vo[1]), ("impl", vi[1])):
            if abs(v) <= 1e-12:
                near_zero.append({"check": k, "who": who, "lower": v})
    add("all seven pass (ours) = go (impl)", pooled["all_seven_pass"] == impl_pool["go"],
        ours=pooled["all_seven_pass"], impl=impl_pool["go"])
    rec = {"all_agree": all(it["agree"] for it in items), "rule_sha256": K.RULE_SHA, "written": K.now_ams(),
           "n_compared": len(items), "n_disagree": sum(not it["agree"] for it in items),
           "lower_bounds_within_1e-12_of_0": near_zero,
           "own_files_sha256_before_opening_impl": {"phase2.json": "02000158d00983cc815ca49751cb2ed03e6c97bb884b8b7e281c3c97b82679ae",
                                                    "phase2_arrays.npz": "b23cf5c017788d69607cf7223bbfae6b46b1d3ef71d25fe2380165985859fc2d"},
           "own_files_sha256_now": {"phase2.json": K.sha_file(K.OUT / "phase2.json"),
                                    "phase2_arrays.npz": K.sha_file(K.OUT / "phase2_arrays.npz")},
           "impl_files_sha256": {f.name: K.sha_file(f) for f in [R3RES / "go_pooled.json"]
                                 + [R3RES / f"go_seed{s}.npz" for s in SEEDS]},
           "compared": items}
    K.save_json(K.OUT / "phase2_agreement.json", rec)
    print(f"all_agree {rec['all_agree']}: {rec['n_compared']} compared, {rec['n_disagree']} disagree; near zero {near_zero}")
    for it in items:
        if not it["agree"]:
            print("DISAGREE", it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("hash", "go", "compare"))
    a = ap.parse_args()
    {"hash": cmd_hash, "go": cmd_go, "compare": cmd_compare}[a.cmd]()


if __name__ == "__main__":
    main()
