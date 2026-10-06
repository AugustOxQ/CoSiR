"""Round 3 on the test seeds: the GO pass (DECISION_RULE.md §6.3 to §6.6, order of computation §6.4) and the descriptive
pass after the verdict (§7).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261121_round3_affect_gate/run_r3_test.py --phase go --seeds 49 50 51
    ... --phase descriptive --seeds 49 50 51          only after results/test_verdict.json exists
    ... --seeds 9001 9002 9003 --smoke                the wiring smoke test (results/smoke/; prints no value)

--phase go: per seed (49, 50, 51 in that order): the build record's SHA-256s against the files and the episode-hash
cross check (§6.2, D15); r3_bundle.build_bundle(s); load_external; the bundle cache results/cache_seed{s}.npz
(r3_bundle.save_bundle) and the reader arrays results/cache_reader_seed{s}.npz (P, T, margins, picks); AFF's family
(fused reader and its matched counterpart) and R1's fused reader (run_family fused_only: R1's counterpart is not
cross-fitted, assembled or written). results/go_seed{s}.npz holds ONLY the per-anchor arrays of §6.4 (AFF fused and
counterpart, R1 fused, B, B'(A0), cosine, RCA), cl, pair_index, parity, the chosen cells and sigma* (go_npz_keys()).
Then results/go_pooled.json = r3_stats.go_checks over the three seeds (the seven checks and the secondary check). No
per-seed summary, bar margin, gate share, pick accuracy or anything of R1's counterpart is computed.

--phase descriptive: refuses unless results/test_verdict.json exists under this rule (and names the current
go_pooled.json). From the caches: §7 items 1 to 7 -> results/descriptive.json and descriptive.txt. The cached bundle and
reader arrays are checked to reproduce the GO pass exactly before anything is computed.
Non-smoke outputs are never overwritten.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r3_bundle as RB  # noqa: E402
import r3_common as R3  # noqa: E402
import r3_fusion as RF  # noqa: E402
import r3_stats as RS  # noqa: E402
import run_r3_build as BLD  # noqa: E402

C = R3.C
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS  # noqa: E402

GO_ARRAYS = ("aff_fused", "aff_cf", "r1_fused", "B", "Bp", "cosine", "rca")
GO_EXTRA = ("cl", "pair_index", "parity", "aff_fused_cells", "aff_cf_cells", "r1_fused_cells", "sigma", "meta")
TOLD_A0 = {"emotion": "affect", "style": "image", "genre": "image"}          # rule D14


def say(msg):
    print(f"[run_r3_test] {msg}", flush=True)


def go_npz_keys() -> tuple:
    """The only arrays results/go_seed{s}.npz may hold (rule §6.4)."""
    return tuple(f"{w}__{m}" for w in GO_ARRAYS for m in METRICS) + GO_EXTRA


def _paths(out, seeds):
    return {"cache": {s: out / f"cache_seed{s}.npz" for s in seeds},
            "reader": {s: out / f"cache_reader_seed{s}.npz" for s in seeds},
            "go": {s: out / f"go_seed{s}.npz" for s in seeds},
            "pooled": out / "go_pooled.json", "verdict": out / "test_verdict.json",
            "desc_json": out / "descriptive.json", "desc_txt": out / "descriptive.txt"}


def _f64(pa):
    return {m: np.asarray(pa[m], dtype=np.float64) for m in METRICS}


def _pair(d):
    return [int(d[0]), int(d[1])]


def _save_reader(path, rd, seed, smoke, n):
    arr = {}
    for c in CONDITIONS:
        arr[f"P__{c}"] = np.asarray(rd["P"][c], np.float64)
        arr[f"m__{c}"] = np.asarray(rd["m"][c], np.float64)
        arr[f"pick__{c}"] = np.asarray(rd["pick"][c], np.int64)
        for d in DIRECTIONS:
            arr[f"T__{c}__{d}"] = np.asarray(rd["T"][c][d], np.float32)
    arr["meta"] = np.array(json.dumps({"seed": int(seed), "smoke": bool(smoke), "n": int(n),
                                       "rule_sha256": R3.RULE_SHA, "what": "reader arrays of rule §4 item 3"}))
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arr)
    tmp.replace(path)
    return R3.sha_file(path)


# ---------------------------------------------------------------- GO pass (§6.4 to §6.6)

def go_phase(seeds, smoke):
    seeds = BLD.check_seeds(seeds, smoke, full=True)
    smoke = bool(smoke)
    R3.assert_rule()
    taus = R3.assert_taus()
    out = R3.res_dir(smoke)
    P = _paths(out, seeds)
    R3.refuse_existing([*P["cache"].values(), *P["reader"].values(), *P["go"].values(), P["pooled"]], smoke)
    t_start = time.time()
    records = {s: BLD.verify_build_record(s, smoke) for s in seeds}
    BLD.cross_check(records)
    say(f"build records of seeds {list(seeds)}: file SHA-256s and the episode-hash cross check PASS")
    per_seed, files = [], {}
    for s in seeds:
        t0 = time.time()
        b = RB.build_bundle(s, smoke)
        if dict(b.episodes_sha256) != records[s]["episode_pair_sha256"]:
            raise SystemExit(f"seed {s}: the bundle's episode SHA-256s differ from build_seed{s}.json")
        ext = RB.load_external(b)
        files[P["cache"][s].name] = RB.save_bundle(b, P["cache"][s])
        rd = RF.reader(b, readers=b.readers)
        files[P["reader"][s].name] = _save_reader(P["reader"][s], rd, s, smoke, b.n)
        fa = RF.run_family(b, rd["T"], RF.gates_aff(rd["m"], rd["pick"], taus))
        fr = RF.run_family(b, rd["T"], RF.gates_r1(rd["m"], taus), fused_only=True)
        if fr["cf"] is not None or fr["cpick"] is not None or "counterpart" in fr["details"]:
            raise AssertionError("R1's counterpart must not be cross-fitted or assembled before the verdict (§6.4)")
        if fa["sigma"] != fr["sigma"]:
            raise AssertionError("the nested control's sigma* depends on B only, so it must be the same for both readers")
        if not (np.asarray(fa["cf"]["gain"]) == 0).all():
            raise AssertionError("AFF's matched counterpart must have condition gain exactly 0")
        arrays = {}
        for who, pa in (("aff_fused", fa["fused"]), ("aff_cf", fa["cf"]), ("r1_fused", fr["fused"]), ("B", b.pB),
                        ("Bp", b.pBp), ("cosine", ext["cosine"]), ("rca", ext["rca"])):
            for m in METRICS:
                arrays[f"{who}__{m}"] = np.asarray(pa[m], np.float64)
        arrays["cl"], arrays["pair_index"], arrays["parity"] = (np.asarray(b.cl), np.asarray(b.pair_index),
                                                                np.asarray(b.parity))
        arrays["aff_fused_cells"] = np.array(_pair(fa["fpick"]), np.int64)
        arrays["aff_cf_cells"] = np.array(_pair(fa["cpick"]), np.int64)
        arrays["r1_fused_cells"] = np.array(_pair(fr["fpick"]), np.int64)
        arrays["sigma"] = np.array([fa["sigma"][0], fa["sigma"][1]], np.float64)
        arrays["meta"] = np.array(json.dumps({"seed": int(s), "smoke": smoke, "n": int(b.n), "rule_sha256": R3.RULE_SHA,
                                              "what": "rule §6.4 per-anchor arrays, chosen cells and sigma*",
                                              "cache_sha256": files[P["cache"][s].name],
                                              "cache_reader_sha256": files[P["reader"][s].name]}))
        if set(arrays) != set(go_npz_keys()):
            raise AssertionError("go_seed arrays differ from rule §6.4's list")
        np.savez_compressed(P["go"][s], **arrays)
        files[P["go"][s].name] = R3.sha_file(P["go"][s])
        per_seed.append({"cl": arrays["cl"], "pair_index": arrays["pair_index"], "aff": _f64(fa["fused"]),
                         "cf": _f64(fa["cf"]), "r1": _f64(fr["fused"]), "cosine": ext["cosine"], "rca": ext["rca"],
                         "B": _f64(b.pB), "Bp": _f64(b.pBp)})
        say(f"seed {s}: {b.n} episodes; {P['cache'][s].name}, {P['reader'][s].name}, {P['go'][s].name} written "
            f"[{time.time() - t0:.0f}s]")
        del b, rd, fa, fr, ext, arrays
        gc.collect()
    pooled = RS.go_checks(per_seed)
    cl_all = np.concatenate([p["cl"] for p in per_seed])
    rec = {"what": "rule §6.5 and §6.6: the seven GO checks and the secondary check, pooled over the seeds (pp, 95% "
                   "painting-bootstrap intervals, clusters = anchor paintings shared across seeds); the verdict is "
                   "r3_apply_rule.py's", "seeds": list(seeds), "smoke": smoke,
           "n_episodes": {str(s): int(len(p["cl"])) for s, p in zip(seeds, per_seed)},
           "n_episodes_pooled": int(len(cl_all)), "n_clusters_pooled": int(len(np.unique(cl_all))),
           "checks": pooled["checks"], "go": pooled["go"], "secondary": pooled["secondary"],
           "files_sha256": files, "build_record_sha256": {str(s): R3.sha_file(BLD.record_path(s, smoke)) for s in seeds},
           "taus": list(taus), "runtime_s": round(time.time() - t_start, 1)}
    R3.write_json_once(P["pooled"], rec, smoke)
    if smoke:
        say(f"{P['pooled'].name} written: {len(pooled['checks'])} checks and the secondary check (smoke: no value "
            f"printed)")
    else:
        say("pooled over seeds 49, 50, 51 (pp; pass = 95% lower bound > 0):")
        for k, r in list(pooled["checks"].items()) + [("secondary (never changes GO)", pooled["secondary"])]:
            print(f"  {k:30s} {r['point']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
                  f"{'pass' if r['pass'] else 'FAIL'}", flush=True)
    say("GO_PHASE PASS")


# ---------------------------------------------------------------- descriptive pass (§7)

def require_verdict(out, smoke) -> dict:
    p = out / "test_verdict.json"
    if not p.exists():
        raise SystemExit(f"{p} is missing: the descriptive pass runs only after the verdict (rule §6.4, §7); refusing")
    v = json.loads(p.read_text())
    if v.get("rule_sha256") != R3.RULE_SHA:
        raise SystemExit(f"{p.name}: written under another rule; refusing")
    if v.get("smoke") is not bool(smoke):
        raise SystemExit(f"{p.name}: its smoke flag differs from this run's; refusing")
    if v.get("verdict") not in ("GO", "NO-GO"):
        raise SystemExit(f"{p.name}: no GO or NO-GO verdict; refusing")
    return v


def _cat(dicts):
    return {m: np.concatenate([np.asarray(d[m], np.float64) for d in dicts]) for m in dicts[0]}


def _inp(S, fused, cf, second):
    """go_checks / bar_info_pooled input of one seed for a (fused, counterpart, secondary comparator) triple."""
    return {"cl": S["cl"], "pair_index": S["pair_index"], "aff": S[fused], "cf": S[cf], "r1": S[second],
            "cosine": S["cosine"], "rca": S["rca"], "B": S["B"], "Bp": S["Bp"]}


def _sub(inp, mask):
    out = {"cl": np.asarray(inp["cl"])[mask], "pair_index": np.asarray(inp["pair_index"])[mask]}
    for k in ("aff", "cf", "r1", "cosine", "rca", "B", "Bp"):
        out[k] = {m: np.asarray(v)[mask] for m, v in inp[k].items()}
    return out


def _seven(inps, secondary=True):
    r = RS.go_checks(inps)
    out = {"checks": r["checks"], "all_seven_lower_above_0": r["go"],
           "note": "descriptive (rule §7); decides nothing"}
    if secondary:
        out["secondary"] = r["secondary"]
    return out


def _bar(inp):
    v, info = C.bar_info(inp["aff"], inp["cf"], inp["Bp"], inp["B"], np.asarray(inp["cl"]), np.asarray(inp["pair_index"]))
    return v, info


def _cells(fam):
    out = {"fused": {str(h): RF.describe(int(c)) for h, c in fam["fpick"].items()},
           "sigma": {str(h): float(s) for h, s in fam["sigma"].items()}}
    if fam.get("cpick") is not None:
        out["counterpart"] = {str(h): RF.describe(int(c)) for h, c in fam["cpick"].items()}
    return out


def _same_pa(a, b):
    return all(np.array_equal(np.asarray(a[m], np.float64), np.asarray(b[m], np.float64)) for m in METRICS)


def _concat_gates(gs):
    return [{c: np.concatenate([g[t][c] for g in gs]) for c in CONDITIONS} for t in range(len(gs[0]))]


def descriptive_phase(seeds, smoke):
    seeds = BLD.check_seeds(seeds, smoke, full=True)
    smoke = bool(smoke)
    R3.assert_rule()
    out = R3.res_dir(smoke)
    verdict = require_verdict(out, smoke)
    P = _paths(out, seeds)
    R3.refuse_existing([P["desc_json"], P["desc_txt"]], smoke)
    if not P["pooled"].exists() or R3.sha_file(P["pooled"]) != verdict["inputs"]["go_pooled"]["sha256"]:
        raise SystemExit("go_pooled.json is missing or differs from the one the verdict was written from; refusing")
    go_rec = json.loads(P["pooled"].read_text())
    if go_rec["seeds"] != list(seeds) or go_rec["smoke"] is not smoke:
        raise SystemExit("go_pooled.json: other seeds or smoke flag")
    taus = R3.assert_taus()
    t_start = time.time()
    records = {s: BLD.verify_build_record(s, smoke) for s in seeds}
    BLD.cross_check(records)
    data = []
    for s in seeds:
        t0 = time.time()
        for k in ("cache", "reader", "go"):
            if R3.sha_file(P[k][s]) != go_rec["files_sha256"][P[k][s].name]:
                raise SystemExit(f"{P[k][s].name}: SHA-256 differs from go_pooled.json's record")
        with np.load(P["go"][s]) as z:
            g = {k: z[k] for k in z.files}
        b = RB.load_bundle_cache(P["cache"][s], seed=s, smoke=smoke, sha256=go_rec["files_sha256"][P["cache"][s].name])
        rd = RF.reader(b, readers=b.readers)
        with np.load(P["reader"][s]) as z:
            same = all(np.array_equal(np.asarray(rd[k][c]), z[f"{k}__{c}"]) for k in ("P", "m") for c in CONDITIONS)
            same &= all(np.array_equal(np.asarray(rd["pick"][c], np.int64), z[f"pick__{c}"]) for c in CONDITIONS)
            same &= all(np.array_equal(rd["T"][c][d], z[f"T__{c}__{d}"]) for c in CONDITIONS for d in DIRECTIONS)
        if not same:
            raise AssertionError(f"seed {s}: the reader on the cached bundle differs from the GO pass's reader arrays")
        T = rd["T"]
        g_aff, g_r1 = RF.gates_aff(rd["m"], rd["pick"], taus), RF.gates_r1(rd["m"], taus)
        fa = RF.run_family(b, T, g_aff)
        fr = RF.run_family(b, T, g_r1)                             # R1's counterpart cross-fit, run now (§7 item 3)
        unpack = (lambda who: {m: g[f"{who}__{m}"] for m in METRICS})
        consistent = (_same_pa(fa["fused"], unpack("aff_fused")) and _same_pa(fa["cf"], unpack("aff_cf"))
                      and _same_pa(fr["fused"], unpack("r1_fused")) and _same_pa(b.pB, unpack("B"))
                      and _same_pa(b.pBp, unpack("Bp")) and _pair(fa["fpick"]) == g["aff_fused_cells"].tolist()
                      and _pair(fa["cpick"]) == g["aff_cf_cells"].tolist()
                      and _pair(fr["fpick"]) == g["r1_fused_cells"].tolist()
                      and [fa["sigma"][0], fa["sigma"][1]] == g["sigma"].tolist())
        if not consistent:
            raise AssertionError(f"seed {s}: the cached bundle does not reproduce the GO pass exactly")
        fz_a = RF.score_frozen(b, T, g_aff, R3.AFF_CELLS["fused"], R3.AFF_CELLS["cf"])
        fz_r = RF.score_frozen(b, T, g_r1, R3.RC_CELLS["fused"], R3.RC_CELLS["cf"])
        shares = RF.affect_shares(g_aff)
        rnd = {}
        for r in (0, 1):
            rnd[r] = RF.run_family(b, T, RF.gates_random(g_r1, shares, 100 * s + r, b.n))
        red = RB.redundancy(b)
        S = {"seed": s, "cl": g["cl"], "pair_index": g["pair_index"],
             "aff": _f64(fa["fused"]), "aff_cf": _f64(fa["cf"]), "r1": _f64(fr["fused"]), "r1_cf": _f64(fr["cf"]),
             "B": unpack("B"), "Bp": unpack("Bp"), "cosine": unpack("cosine"), "rca": unpack("rca"),
             "fz_aff": _f64(fz_a["fused"]), "fz_aff_cf": _f64(fz_a["cf"]), "fz_r1": _f64(fz_r["fused"]),
             "fz_r1_cf": _f64(fz_r["cf"]), "g_aff": g_aff, "g_r1": g_r1, "pick": rd["pick"],
             "cells": {"AFF": _cells(fa), "R1": _cells(fr), **{f"random_r{r}": _cells(rnd[r]) for r in (0, 1)}},
             "red": red, "shares": shares}
        for r in (0, 1):
            S[f"rnd{r}"], S[f"rnd{r}_cf"] = _f64(rnd[r]["fused"]), _f64(rnd[r]["cf"])
        data.append(S)
        say(f"seed {s}: cache reproduces the GO pass; descriptive families done [{time.time() - t0:.0f}s]")
        del b, rd, fa, fr, fz_a, fz_r, rnd
        gc.collect()
    desc = describe_all(seeds, data)
    desc.update(verdict=verdict["verdict"], verdict_sha256=R3.sha_file(P["verdict"]),
                go_pooled_sha256=R3.sha_file(P["pooled"]), runtime_s=round(time.time() - t_start, 1))
    rec = R3.write_json_once(P["desc_json"], desc, smoke)
    text = descriptive_text(rec, smoke)
    P["desc_txt"].write_text(text)
    if smoke:
        say(f"{P['desc_json'].name} and {P['desc_txt'].name} written (smoke: no value printed)")
    else:
        print(text, flush=True)
    say("DESCRIPTIVE_PHASE PASS")


def describe_all(seeds, data) -> dict:
    """Rule §7 items 1 to 7 from the per-seed records."""
    cl_all = np.concatenate([S["cl"] for S in data])
    pi_all = np.concatenate([S["pair_index"] for S in data])
    aff_in = [_inp(S, "aff", "aff_cf", "r1") for S in data]
    r1_in = [_inp(S, "r1", "r1_cf", "r1") for S in data]

    # item 1: per seed, and per pair pooled over the seeds
    item1 = {"per_seed": {str(S["seed"]): _seven([i]) for S, i in zip(data, aff_in)},
             "per_pair_pooled": {p: _seven([_sub(i, np.asarray(i["pair_index"]) == k) for i in aff_in])
                                 for k, p in enumerate(C.POOLED_ORDER)},
             "note": "per-seed and per-pair results never change the verdict; per-pair results are not tested (§6.8)"}

    # item 2: bar margin (pooled scope and per seed), margin / gain / either, AFF - R1 bar margin, cells, frozen line
    bar_v_aff, bar_aff = RS.bar_info_pooled(aff_in)
    bar_v_r1, bar_r1 = RS.bar_info_pooled(r1_in)
    per_seed_bar = {}
    for S, ia, ir in zip(data, aff_in, r1_in):
        va, ba = _bar(ia)
        vr, br = _bar(ir)
        per_seed_bar[str(S["seed"])] = {"AFF": ba, "R1": br, "AFF_minus_R1_bar_margin": C.point_ci(va - vr, S["cl"]),
                                        "AFF_vs_counterpart": C.diff3(S["aff"], S["aff_cf"], S["cl"])}
    fz_aff_in = [_inp(S, "fz_aff", "fz_aff_cf", "fz_r1") for S in data]
    fz_r1_in = [_inp(S, "fz_r1", "fz_r1_cf", "fz_r1") for S in data]
    item2 = {"AFF_bar_margin_pooled": bar_aff, "per_seed": per_seed_bar,
             "AFF_vs_counterpart_pooled": C.diff3(_cat([S["aff"] for S in data]), _cat([S["aff_cf"] for S in data]),
                                                  cl_all),
             "AFF_minus_R1_bar_margin_pooled": C.point_ci(bar_v_aff - bar_v_r1, cl_all),
             "chosen_cells": {str(S["seed"]): S["cells"] for S in data},
             "frozen_cell_line": {
                 "cells": {"AFF": {"fused": list(R3.AFF_CELLS["fused"]), "counterpart": list(R3.AFF_CELLS["cf"])},
                           "R1": {"fused": list(R3.RC_CELLS["fused"]), "counterpart": list(R3.RC_CELLS["cf"])},
                           "note": "the cell chosen on seed-42 tune half h scores the test seed's parity 1 - h"},
                 "AFF_pooled": {**_seven(fz_aff_in), "bar": RS.bar_info_pooled(fz_aff_in)[1]},
                 "AFF_per_seed": {str(S["seed"]): _seven([i]) for S, i in zip(data, fz_aff_in)},
                 "R1_pooled": {**_seven(fz_r1_in, secondary=False), "bar": RS.bar_info_pooled(fz_r1_in)[1]},
                 "R1_per_seed": {str(S["seed"]): _seven([i], secondary=False) for S, i in zip(data, fz_r1_in)},
                 "secondary_note": "AFF_pooled.secondary is frozen AFF fused minus frozen R1 fused"}}

    # item 3: R1's own seven checks (its counterpart cross-fit run now), with its bar margin
    item3 = {**_seven(r1_in, secondary=False), "bar_margin_pooled": bar_r1,
             "per_seed": {str(S["seed"]): {**_seven([i], secondary=False), "bar": per_seed_bar[str(S["seed"])]["R1"]}
                          for S, i in zip(data, r1_in)},
             "note": "descriptive; R1 never gives a second verdict (rule §7 item 3)"}

    # item 4: gate-open counts and shares at each tau, per condition and pair, per seed and pooled
    item4 = {"AFF": {"pooled": RF.open_shares(_concat_gates([S["g_aff"] for S in data]), pi_all),
                     "per_seed": {str(S["seed"]): RF.open_shares(S["g_aff"], S["pair_index"]) for S in data}},
             "R1": {"pooled": RF.open_shares(_concat_gates([S["g_r1"] for S in data]), pi_all),
                    "per_seed": {str(S["seed"]): RF.open_shares(S["g_r1"], S["pair_index"]) for S in data}}}

    # item 5: pick accuracy under the told mapping (D14), pooled
    if dict(C.TOLD["A0"]) != TOLD_A0:
        raise AssertionError("round 1's told mapping for A0 differs from rule D14")
    picks = {c: np.concatenate([np.asarray(S["pick"][c]) for S in data]) for c in CONDITIONS}
    acc, share = C.pick_statistics(picks, R3.A0, TOLD_A0, pi_all, cl_all)
    item5 = {"pick_accuracy": acc, "pick_share": share, "told_mapping": TOLD_A0,
             "note": "diagnostic only; enters no rule (D14)"}

    # item 6: D7 redundancy per seed
    item6 = {str(S["seed"]): {"redundancy": S["red"], "affect_least_redundant_both_directions":
                              RB.affect_least_redundant(S["red"])} for S in data}

    # item 7: random-share control, per draw, pooled
    item7 = {"definition": "keep^c = 1[u < share_c], u = default_rng(100*s + r).random(E), condition a drawn first; "
                           "gates = R1's gates times keep^c at every tau; own counterpart (D9); comparators as AFF's",
             "shares": {str(S["seed"]): S["shares"] for S in data}, "draws": {}}
    for r in (0, 1):
        rin = [_inp(S, f"rnd{r}", f"rnd{r}_cf", "r1") for S in data]
        bar_v_rnd, bar_rnd = RS.bar_info_pooled(rin)
        item7["draws"][f"r{r}"] = {
            "generator_seeds": {str(S["seed"]): 100 * S["seed"] + r for S in data},
            "bar_margin_pooled": bar_rnd,
            "AFF_minus_control_fused_r1": C.point_ci(_cat([S["aff"] for S in data])["r1"]
                                                     - _cat([S[f"rnd{r}"] for S in data])["r1"], cl_all),
            "AFF_minus_control_bar_margin": C.point_ci(bar_v_aff - bar_v_rnd, cl_all),
            "cells": {str(S["seed"]): S["cells"][f"random_r{r}"] for S in data}}

    return {"what": "rule §7: the descriptive pass after the verdict; decides nothing",
            "seeds": list(seeds), "n_episodes_pooled": int(len(cl_all)), "n_clusters_pooled": int(len(np.unique(cl_all))),
            "item1_per_seed_and_per_pair": item1, "item2_bar_margin_cells_frozen": item2, "item3_R1_checks": item3,
            "item4_gate_open_shares": item4, "item5_pick_accuracy": item5, "item6_redundancy": item6,
            "item7_random_share_control": item7, "disclosure": "rule §6.11 applies to every AFF number"}


# ---------------------------------------------------------------- readable text

def _ci(r):
    return f"{r['point']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"


def _seven_lines(x, indent="    "):
    L = [f"{indent}{k:20s} {_ci(r)} {'pass' if r['pass'] else 'fail'}" for k, r in x["checks"].items()]
    if "secondary" in x:
        L.append(f"{indent}{'secondary (AFF-R1)':20s} {_ci(x['secondary'])} {'pass' if x['secondary']['pass'] else 'fail'}")
    return L


def descriptive_text(d, smoke) -> str:
    i1, i2, i3 = d["item1_per_seed_and_per_pair"], d["item2_bar_margin_cells_frozen"], d["item3_R1_checks"]
    L = [f"Round 3 descriptive pass (rule §7; decides nothing). Verdict: {d['verdict']}. Seeds {d['seeds']}, "
         f"{d['n_episodes_pooled']} episodes, {d['n_clusters_pooled']} paintings.{'  [SMOKE: not a result]' if smoke else ''}",
         "1. AFF per seed (seven checks and the secondary, pp):"]
    for s, x in i1["per_seed"].items():
        L.append(f"  seed {s}")
        L += _seven_lines(x)
    L.append("   AFF per aspect pair, pooled over the seeds (not tested):")
    for p, x in i1["per_pair_pooled"].items():
        L.append(f"  {p}")
        L += _seven_lines(x)
    b = i2["AFF_bar_margin_pooled"]
    L += ["2. AFF bar margin, pooled (comparator by the pooled mean R@1): "
          f"{_ci(b['r1'])} vs {b['comparator']}; per pair: "
          + "; ".join(f"{p} {_ci(v)}" for p, v in b["per_pair_r1"].items()),
          "   per seed: " + "; ".join(f"{s} {_ci(v['AFF']['r1'])} vs {v['AFF']['comparator']}"
                                      for s, v in i2["per_seed"].items()),
          f"   AFF vs its counterpart, pooled: margin {_ci(i2['AFF_vs_counterpart_pooled']['r1'])}, gain "
          f"{_ci(i2['AFF_vs_counterpart_pooled']['gain'])}, either {_ci(i2['AFF_vs_counterpart_pooled']['either'])}",
          f"   AFF minus R1, bar margin, pooled: {_ci(i2['AFF_minus_R1_bar_margin_pooled'])}",
          "   chosen cells (half 0 / half 1):"]
    for s, cells in i2["chosen_cells"].items():
        L.append(f"    seed {s}: " + "; ".join(
            f"{who} fused {v['fused']['0']['cell']}/{v['fused']['1']['cell']}"
            + (f", cf {v['counterpart']['0']['cell']}/{v['counterpart']['1']['cell']}" if "counterpart" in v else "")
            for who, v in cells.items()) + f"; sigma* {cells['AFF']['sigma']['0']}/{cells['AFF']['sigma']['1']}")
    fz = i2["frozen_cell_line"]
    L.append(f"   frozen-cell line, AFF (fused 39/119, cf 149/10), pooled; bar {_ci(fz['AFF_pooled']['bar']['r1'])} vs "
             f"{fz['AFF_pooled']['bar']['comparator']}:")
    L += _seven_lines(fz["AFF_pooled"])
    L.append(f"   frozen-cell line, R1 (fused 116/119, cf 58/123), pooled; bar {_ci(fz['R1_pooled']['bar']['r1'])} vs "
             f"{fz['R1_pooled']['bar']['comparator']}:")
    L += _seven_lines(fz["R1_pooled"])
    L.append(f"3. R1's own seven checks, pooled (descriptive; no second verdict); bar margin "
             f"{_ci(i3['bar_margin_pooled']['r1'])} vs {i3['bar_margin_pooled']['comparator']}:")
    L += _seven_lines(i3)
    L.append("4. Gate open share (%), pooled: overall / a / b per tau")
    for who in ("AFF", "R1"):
        g = d["item4_gate_open_shares"][who]["pooled"]
        L.append(f"   {who}: " + "; ".join(f"{k} {v['overall']:.2f}/{v['a']:.2f}/{v['b']:.2f}" for k, v in g.items()))
    acc = d["item5_pick_accuracy"]["pick_accuracy"]
    L.append(f"5. Pick accuracy (told mapping, pooled): {_ci(acc['correct_share'])} (chance {acc['chance']:.1f}); per "
             "pair a/b: " + "; ".join(f"{p} {v['a']:.1f}/{v['b']:.1f}" for p, v in acc["per_pair_condition"].items()))
    L.append("6. Redundancy with B (D7), i2t/t2i:")
    for s, x in d["item6_redundancy"].items():
        L.append(f"   seed {s}: " + "; ".join(f"{h} {v['i2t']:.4f}/{v['t2i']:.4f}" for h, v in x["redundancy"].items())
                 + f"; affect least redundant: {x['affect_least_redundant_both_directions']}")
    L.append("7. Random-share control (pooled):")
    for r, x in d["item7_random_share_control"]["draws"].items():
        L.append(f"   {r}: bar margin {_ci(x['bar_margin_pooled']['r1'])} vs {x['bar_margin_pooled']['comparator']}; "
                 f"AFF minus control: fused R@1 {_ci(x['AFF_minus_control_fused_r1'])}, bar margin "
                 f"{_ci(x['AFF_minus_control_bar_margin'])}")
    L.append("Disclosure (rule §6.11) applies to every AFF number above.")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("go", "descriptive"), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.phase == "go":
        go_phase(args.seeds, args.smoke)
    else:
        descriptive_phase(args.seeds, args.smoke)


if __name__ == "__main__":
    main()
