"""Phase-1 agreement (round-4 rule §8): compare the re-derivation's out/rd4_phase1.json with the implementation's
results/regression_check.json, results/dev_seed42.json, results/carry.json and results/sensitivity.json (and any
per-anchor .npz arrays under results/, top level only) under the rule's agreement tolerances:

  every discrete quantity identical (picks, gates, chosen cells, sigma*, the bar comparator, Delta_k, the carry, each
  clause's pass or fail); v75 within 1e-15 absolute or 1e-9 relative; every margin, gain statistic, point and bound
  within 1e-9 percentage points; every per-anchor array exactly.

The re-derivation was written without reading the implementation, so its JSON key layout is unknown here. Each of our
quantities is located in the implementation's files by (1) the explicit map IMPL_MAP below (dotted paths; to be filled
in once the files exist), else (2) a token heuristic over the flattened key paths (every required token group present,
no forbidden token, value type compatible); several candidate paths with different values make the quantity
"ambiguous". (3) For an unmatched or ambiguous number, paths holding the same value are listed as hints; hints never
count as agreement. all_agree is true only if every quantity is located and agrees, and every per-anchor array of the
fused readers and counterparts of AFF, V4, V2 and V24 has an identical twin under a key naming its scorer.

Writes out/agreement_phase1.json. Run only after out/rd4_phase1.json is written and its SHA-256 recorded
(out/rd4_phase1.sha256, asserted). Usage: python rd4_compare_phase1.py
"""
import json
import re
from pathlib import Path

import numpy as np

import rd4_core as K

MINE = K.OUT / "rd4_phase1.json"
MINE_SHA = K.OUT / "rd4_phase1.sha256"
MINE_NPZ = K.OUT / "rd4_phase1_arrays.npz"
RES = K.IMPL_RESULTS
IMPL_FILES = ("regression_check.json", "dev_seed42.json", "carry.json", "sensitivity.json")
OUT = K.OUT / "agreement_phase1.json"
TOL = 1e-9

# Explicit map, quantity name -> [(file name, dotted path), ...]; list indices as integers in the path ("a.ci95.0").
# Filled in after the implementation's files exist (the controller resumes the re-derivation for that).
IMPL_MAP = {}

SCORER_TOKENS = {"R1": {"r1"}, "AFF": {"aff"},
                 "IMGABST": {"imgabst", "q75", "xav", "r1xav", "r1timesav", "r1av", "imgabstq75", "r1imgabst",
                             "r1imgabstq75", "abst", "abstention"},
                 "V4": {"v4"}, "V2": {"v2"}, "V24": {"v24"}}
CONTEXT_FORBID = {"emotion", "style", "genre", "sha256", "sha", "smoke", "seed49", "seed50", "seed51", "seed52",
                  "seed53", "seed54"}


# ---------------------------------------------------------------- flattening and tokens

def flatten(obj, prefix=()):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, prefix + (str(k),)))
    elif isinstance(obj, list) and not obj:
        out[prefix] = obj                                            # empty list (E, tied in a kill)
    elif isinstance(obj, list) and all(not isinstance(v, (dict, list)) for v in obj) and len(obj) <= 8:
        out[prefix] = obj                                            # short leaf list (cells, E, tied, ci95)
        for i, v in enumerate(obj):
            out[prefix + (i,)] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, prefix + (i,)))
    else:
        out[prefix] = obj
    return out


SCORER_WHOLE = {"r1xav", "r1timesav", "r1av", "imgabstq75", "imgabst", "r1imgabst", "r1imgabstq75"}


def tokens(path):
    t = set()
    for p in path:
        if isinstance(p, int):
            t.add(f"#{p}")
        else:
            s = re.sub(r"([a-z])([A-Z])", r"\1_\2", p).lower().replace("′", "prime").replace("'", "prime")
            whole = re.sub(r"[^a-z0-9]+", "", s)                      # the whole key, separators removed
            t.add(whole)
            if whole not in SCORER_WHOLE:                             # a scorer alias stays whole ("R1_x_a_v")
                t.update(x for x in re.split(r"[^a-z0-9#]+", s) if x)
    return t


def dotted(path):
    return ".".join(str(p) for p in path)


def get_dotted(obj, path):
    for p in path.split("."):
        if isinstance(obj, list):
            obj = obj[int(p)]
        else:
            obj = obj[p]
    return obj


# ---------------------------------------------------------------- normalisation and comparison

COMP = {"bprimea1": "Bprime_A1", "bpa1": "Bprime_A1", "bp1": "Bprime_A1", "bprime1": "Bprime_A1",
        "bprimea0": "Bprime_A0", "bpa0": "Bprime_A0", "bp0": "Bprime_A0", "bprime": "Bprime_A0", "bp": "Bprime_A0",
        "bprime0": "Bprime_A0", "counterpart": "counterpart", "cf": "counterpart", "matchedcounterpart": "counterpart",
        "b": "B"}


def norm_name(x):
    if isinstance(x, str):
        s = re.sub(r"[^a-z0-9]+", "", x.lower().replace("′", "prime").replace("'", "prime"))
        return COMP.get(s, x)
    return x


def as_num(x):
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    return None


def agree(kind, mine, theirs):
    if kind == "float":
        a, b = as_num(mine), as_num(theirs)
        return a is not None and b is not None and abs(a - b) <= TOL, (None if a is None or b is None else abs(a - b))
    if kind == "v75":
        a, b = as_num(mine), as_num(theirs)
        if a is None or b is None:
            return False, None
        d = abs(a - b)
        return d <= 1e-15 or d <= 1e-9 * abs(a), d
    if kind == "int":
        a, b = as_num(mine), as_num(theirs)
        return a is not None and b is not None and a == b and float(theirs) == int(float(theirs)), \
            (None if a is None or b is None else abs(a - b))
    if kind == "bool":
        return isinstance(theirs, (bool, np.bool_)) and bool(mine) == bool(theirs), None
    if kind == "name":
        return norm_name(mine) == norm_name(theirs), None
    if kind == "names":
        if not isinstance(theirs, (list, tuple)):
            return False, None
        return [norm_name(x) for x in mine] == [norm_name(x) for x in theirs], None
    raise ValueError(kind)


def type_ok(kind, v):
    if kind in ("float", "v75"):
        return as_num(v) is not None
    if kind == "int":
        return as_num(v) is not None and float(v) == int(float(v))
    if kind == "bool":
        return isinstance(v, (bool, np.bool_))
    if kind == "name":
        return isinstance(v, str) or v is None
    if kind == "names":
        return isinstance(v, (list, tuple))
    return True


# ---------------------------------------------------------------- our quantities

def _spec(scorer, groups, forbid, allow=()):
    gs = [set(g) for g in groups]
    if scorer:
        gs = [SCORER_TOKENS[scorer]] + gs
    asked = set().union(*gs) if gs else set()
    fb = set(forbid) | (CONTEXT_FORBID - asked)
    for s, toks in SCORER_TOKENS.items():
        if s != scorer and not (scorer == "IMGABST" and s == "R1"):     # "R1_x_a_v" names R1 x a_v
            fb |= (toks - asked)
    return {"groups": gs, "forbid": fb - asked - set(allow)}


def Q(name, kind, value, scorer, groups, forbid=(), files=None, alt=(), allow=()):
    """groups: list of token alternatives (each a set; the path must hold one token of each); scorer adds its own
    group; tokens of the other scorers and the context tokens are forbidden unless a group asks for them. alt: further
    (groups, forbid) variants, tried in order when the first finds nothing."""
    variants = [_spec(scorer, groups, forbid, allow)] + [_spec(scorer, g, f, allow) for g, f in alt]
    return {"name": name, "kind": kind, "mine": value, "variants": variants, "files": files}


PT = {"point", "pt", "mean", "value"}
CI_TOK = {"ci95", "ci", "lo", "hi", "lower", "upper", "low", "high", "#0", "#1"}


def pci_quantities(prefix, scorer, field_groups, value, forbid=(), files=None):
    fb = set(forbid)
    return [Q(f"{prefix}.point", "float", value["point"], scorer, field_groups + [PT], fb | {"#0", "#1"}, files,
              alt=[(field_groups, fb | CI_TOK)]),
            Q(f"{prefix}.lo", "float", value["ci95"][0], scorer, field_groups + [{"ci95", "ci"}, {"#0"}], fb, files,
              alt=[(field_groups + [{"lo", "lower", "low"}], fb)]),
            Q(f"{prefix}.hi", "float", value["ci95"][1], scorer, field_groups + [{"ci95", "ci"}, {"#1"}], fb, files,
              alt=[(field_groups + [{"hi", "upper", "high"}], fb)])]


def cell_quantities(prefix, scorer, s, files=None):
    out = []
    for kind, g in (("fused_cells", {"fused"}), ("counterpart_cells", {"counterpart", "cf"})):
        for h in ("0", "1"):
            out.append(Q(f"{prefix}.{kind}.half{h}", "int", s[kind][h]["cell"], scorer,
                         [g, {"cell", "cells"}, {f"#{h}", h, f"half{h}", f"h{h}"}],
                         forbid={"tau", "lambda", "u", "a"}, files=files))
    for h in ("0", "1"):
        out.append(Q(f"{prefix}.sigma_star.half{h}", "float", s["sigma_star"][h], scorer,
                     [{"sigma", "sigmastar", "ctrl", "control"}, {f"#{h}", h, f"half{h}", f"h{h}"}],
                     forbid={"rho"}, files=files))
    return out


def scorer_quantities(prefix, scorer, s, files, with_r1=True, with_margin=True, with_either=True):
    out = []
    if with_r1:
        out += [Q(f"{prefix}.fused_r1", "float", s["fused_r1"], scorer, [{"fused"}, {"r1", "fusedr1"}],
                  forbid={"minus", "bar", "margin", "gain", "either", "delta"}, files=files),
                Q(f"{prefix}.counterpart_r1", "float", s["counterpart_r1"], scorer, [{"counterpart", "cf"},
                                                                                      {"r1", "cfr1"}],
                  forbid={"minus", "bar", "margin", "gain", "either", "delta"}, files=files)]
    out.append(Q(f"{prefix}.bar_comparator", "name", s["bar_comparator"], scorer, [{"comparator", "barcomparator"}],
                 files=files))
    out += pci_quantities(f"{prefix}.bar_margin", scorer, [{"bar"}], s["bar_margin"], forbid={"minus", "x", "se"},
                          files=files)
    out += pci_quantities(f"{prefix}.gain_statistic", scorer, [{"gain"}], s["gain_statistic"],
                          forbid={"minus", "rca", "x", "se", "per"}, files=files)
    if with_margin:
        out += pci_quantities(f"{prefix}.margin_vs_counterpart", scorer, [{"margin"}], s["margin_vs_counterpart"],
                              forbid={"bar", "minus", "x", "se"}, files=files)
    if with_either:
        out.append(Q(f"{prefix}.either_change.point", "float", s["either_change_vs_counterpart"]["point"], scorer,
                     [{"either"}, PT], forbid={"#0", "#1"}, files=files))
    out += cell_quantities(prefix, scorer, s, files)
    return out


def build_quantities(m):
    q = []
    i1, i2, i3, i5 = m["item1_bundle"], m["item2_R1_AFF"], m["item3_IMGABST_q75"], m["item5_gate_algebra"]
    reg = ("regression_check.json",)
    q.append(Q("item1.Bprime_A1_r1", "float", i1["Bprime_A1_r1"], None, [{"bprimea1", "bpa1", "bp1", "a1"}, {"r1"}],
               forbid={"fused", "cf", "counterpart", "minus"}, files=reg))
    q += scorer_quantities("item2.R1", "R1", i2["summary"]["R1"], reg, with_r1=False, with_margin=False,
                           with_either=False)
    q += scorer_quantities("item2.AFF", "AFF", i2["summary"]["AFF"], reg)
    for c in ("a", "b"):
        q.append(Q(f"item2.AFF.tau0_open_count.{c}", "int", i2["AFF"]["tau0_open_counts"][c], "AFF",
                   [{"open", "count", "counts"}, {c, f"cond{c}"}], forbid={"tau1", "tau2", "tau3", "share", "pct"},
                   files=reg))
    for nm, key in (("AFF_minus_R1_fused", "fused"), ("AFF_minus_R1_bar", "bar")):
        v = i2["AFF"][nm]["ours"]
        q += pci_quantities(f"item2.{nm}", None, [{"aff"}, {"r1"}, {"minus", "vs", "diff"}, {key}],
                            {"point": v[0], "ci95": [v[1], v[2]]}, files=reg)
    s3 = i3["summary"]
    q.append(Q("item3.v75", "v75", i3["v75_recomputed"], None, [{"v75", "v"}], forbid={"count"}, files=reg,
               allow=SCORER_TOKENS["IMGABST"]))
    q.append(Q("item3.a_v_count", "int", i3["a_v_count"], None, [{"count", "n", "ones"},
                                                                  {"av", "v", "abstention", "avcount"}],
               files=reg, allow=SCORER_TOKENS["IMGABST"]))
    q += scorer_quantities("item3.IMGABST", "IMGABST", s3, reg)
    for g in ("AFF", "V4", "V2", "V24"):
        for c in ("a", "b"):
            q.append(Q(f"item5.tau0_open_count.{g}.{c}", "int", i5["tau0_open_counts"][g][c], g,
                       [{"open", "count", "counts"}, {c, f"cond{c}"}],
                       forbid={"tau1", "tau2", "tau3", "share", "pct"}, files=reg))
    q.append(Q("item5.PASSED", "bool", i5["PASSED"], None, [{"algebra", "item5", "gate"}, {"pass", "passed", "ok"}],
               files=reg))
    dv = ("dev_seed42.json",)
    for k in K.CANDIDATES:
        s = m["item6_dev"][k]
        q += scorer_quantities(f"item6.{k}", k, s, dv)
        dk = s["Delta_k"]
        q.append(Q(f"item6.{k}.Delta_k.int", "int", dk["int"], k, [{"delta", "deltak", "dk"}],
                   forbid={"point", "pp", "ci95", "#0", "#1", "lo", "hi"}, files=dv))
        q += pci_quantities(f"item6.{k}.Delta_k", k, [{"delta", "deltak", "dk"}],
                            {"point": dk["point_pp"], "ci95": dk["ci95"]}, forbid={"int"}, files=dv)
        for cl_ in ("c1", "c2", "c3"):
            key = [x for x in s["D10_clauses"] if x.startswith(cl_)][0]
            q.append(Q(f"item6.{k}.D10.{cl_}", "bool", s["D10_clauses"][key], k, [{cl_, f"clause{cl_[1]}"}],
                       files=dv + ("carry.json",)))
        q.append(Q(f"item6.{k}.D10.clears_bar", "bool", s["D10_clauses"]["clears_bar"], k,
                   [{"clears", "clearsbar", "passes", "pass", "passed"}], forbid={"c1", "c2", "c3"},
                   files=dv + ("carry.json",)))
        for p in K.PAIR_NAMES:
            a, b = p.split("__")
            q.append(Q(f"item6.{k}.per_pair_bar.{p}.point", "float", s["per_pair_bar_margin"][p]["point"], k,
                       [{"bar"}, {a}, {b}, PT], forbid={"#0", "#1", "gain"} - {a, b},
                       files=dv))
    cr = m["item8_carry"]
    cf = ("carry.json",)
    q.append(Q("item8.E", "names", cr["E"], None, [{"e", "eligible", "set"}], forbid={"tied"}, files=cf))
    q.append(Q("item8.tied", "names", cr["tied"], None, [{"tied", "tie"}], files=cf))
    q.append(Q("item8.carried", "name", cr["carried"], None, [{"carried", "carry"}],
               forbid={"tied", "e", "m", "order"}, files=cf))
    q.append(Q("item8.M", "int" if cr["M"] is not None else "name", cr["M"], None, [{"m", "max", "largest"}],
               forbid={"tied"}, files=cf))
    q.append(Q("item8.kill", "bool", cr["kill"], None, [{"kill", "killed"}], files=cf))
    for k in K.CANDIDATES:
        q.append(Q(f"item8.Delta_k.{k}", "int", cr["Delta_k"][k], k, [{"delta", "deltak", "dk"}],
                   forbid={"point", "pp", "ci95", "#0", "#1"}, files=cf))
    sn = m.get("sensitivity_6_1", {})
    if sn.get("candidate"):
        sf = ("sensitivity.json",)
        k = sn["candidate"]
        for chk, s in sn["checks"].items():
            toks = [t for t in re.split(r"_+", chk.lower()) if t not in ("r1", "minus", "check")] or [chk.lower()]
            g = [{t} for t in toks]
            for fld, kind_t in (("SE", {"se"}), ("half_width", {"half", "halfwidth"}), ("x", {"x"}),
                                ("seed42_bootstrap_half_width", {"bootstrap", "seed42"})):
                fb = {"bootstrap", "seed42"} if fld in ("half_width", "SE", "x") else set()
                if chk == "gain_statistic":
                    fb |= {"rca"}
                q.append(Q(f"sensitivity.{k}.{chk}.{fld}", "float", s[fld], None, g + [kind_t], forbid=fb, files=sf,
                           allow=SCORER_TOKENS[k] | {"aff", "r1"}))
    return q


# ---------------------------------------------------------------- locating

def locate(qq, impl_flat):
    if qq["name"] in IMPL_MAP:
        hits = []
        for fname, path in IMPL_MAP[qq["name"]]:
            try:
                hits.append((fname, path, get_dotted(IMPL_JSON[fname], path)))
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        return "map", hits
    for i, var in enumerate(qq["variants"]):
        cands = []
        for fname, flat in impl_flat.items():
            if qq["files"] and fname not in qq["files"]:
                continue
            for path, v in flat.items():
                tk = tokens(path)
                if any(not (g & tk) for g in var["groups"]) or (tk & var["forbid"]) or not type_ok(qq["kind"], v):
                    continue
                cands.append((fname, dotted(path), v, len(tk)))
        if cands:
            best = min(c[3] for c in cands)
            return f"heuristic_variant{i}", [(f, p, v) for f, p, v, n in cands if n == best]
    return "heuristic", []


def value_hints(qq, impl_flat, limit=5):
    a = as_num(qq["mine"])
    if a is None:
        return []
    out = []
    for fname, flat in impl_flat.items():
        for path, v in flat.items():
            b = as_num(v)
            if b is not None and abs(a - b) <= TOL:
                out.append(f"{fname}:{dotted(path)}")
                if len(out) >= limit:
                    return out
    return out


IMPL_JSON = {}


def main():
    import argparse
    global RES, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl-dir", default=None, help="test only: another folder in place of ../results")
    ap.add_argument("--out", default=None, help="test only: another output path")
    args = ap.parse_args()
    if args.impl_dir:
        RES = Path(args.impl_dir)
    if args.out:
        OUT = Path(args.out)
    K.assert_rules()
    sha = K.sha_file(MINE)
    recorded = MINE_SHA.read_text().split()[0]
    if sha != recorded:
        raise SystemExit(f"rd4_phase1.json SHA-256 {sha} differs from the recorded {recorded}")
    mine = json.loads(MINE.read_text())
    kill = bool(mine["item8_carry"]["kill"])
    expected = [f for f in IMPL_FILES if not (kill and f == "sensitivity.json")]
    res = {"rule_sha256": K.RULE4_SHA, "rd4_phase1_sha256": sha, "written_amsterdam": K.now_ams(),
           "impl_dir": str(RES), "rederive_carry": mine["item8_carry"]["carried"], "rederive_kill": kill,
           "expected_files": expected,
           "tolerances": {"discrete": "identical", "v75": "1e-15 abs or 1e-9 rel", "numbers_pp": TOL,
                          "per_anchor_arrays": "identical"},
           "impl_files": {}, "IMPL_MAP_entries": len(IMPL_MAP)}
    for f in IMPL_FILES:
        p = RES / f
        if p.exists():
            IMPL_JSON[f] = json.loads(p.read_text())
            res["impl_files"][f] = {"present": True, "sha256": K.sha_file(p)}
        else:
            res["impl_files"][f] = {"present": False}
    impl_flat = {f: flatten(o) for f, o in IMPL_JSON.items()}
    used_paths = set()
    rows, counts = [], {"agree": 0, "disagree": 0, "unmatched": 0, "ambiguous": 0}
    for qq in build_quantities(mine):
        how, hits = locate(qq, impl_flat)
        row = {"name": qq["name"], "kind": qq["kind"], "mine": qq["mine"], "located_by": how}
        if not hits:
            row["status"] = "unmatched"
            row["value_hints"] = value_hints(qq, impl_flat)
        else:
            vals = [h[2] for h in hits]
            oks = [agree(qq["kind"], qq["mine"], v) for v in vals]
            row["impl"] = [{"file": f, "path": p, "value": v, "agree": ok[0], "abs_diff": ok[1]}
                           for (f, p, v), ok in zip(hits, oks)]
            for f, p, _ in hits:
                used_paths.add(f"{f}:{p}")
            distinct = {json.dumps(K.jsonable(v), sort_keys=True) for v in vals}
            if len(distinct) > 1 and not all(o[0] for o in oks):
                row["status"] = "ambiguous"
                row["value_hints"] = value_hints(qq, impl_flat)
            else:
                row["status"] = "agree" if all(o[0] for o in oks) else "disagree"
        counts[row["status"]] += 1
        rows.append(row)
    res["quantities"] = rows
    res["counts"] = counts

    # ---- per-anchor arrays: every one of ours that the rule compares must have an identical twin
    my_arr = mine.get("per_anchor_sha256", {})
    impl_arr = {}
    npz_files = sorted(p for p in RES.glob("*.npz") if "smoke" not in p.name) if RES.exists() else []
    for p in npz_files:
        z = np.load(p, allow_pickle=False)
        for key in z.files:
            a = np.asarray(z[key])
            if a.ndim == 1 and a.shape[0] == K.N_EP_SEED42 and a.dtype.kind in "fiub":
                impl_arr.setdefault(K.sha_arr(a.astype(np.float64)), []).append(f"{p.name}:{key}")
    res["impl_npz_files"] = {p.name: K.sha_file(p) for p in npz_files}
    arr_rows, arr_fail = {}, []
    for name, h in my_arr.items():
        twins = impl_arr.get(h, [])
        scorer = name.split("__")[0]
        named = [t for t in twins if (SCORER_TOKENS.get(scorer, {scorer.lower()}) & tokens((t.split(":", 1)[1],)))]
        arr_rows[name] = {"sha256": h, "twins": twins, "twins_naming_scorer": named}
        if scorer in ("AFF", "V4", "V2", "V24") and not named:
            arr_fail.append(name)
    res["per_anchor_arrays"] = arr_rows
    res["per_anchor_arrays_without_named_twin"] = arr_fail
    if not npz_files:
        res["per_anchor_arrays_note"] = "no implementation .npz found under results/; per-anchor arrays not compared"

    # ---- implementation leaves no quantity matched (for adapting IMPL_MAP)
    res["impl_paths_not_used"] = sorted(f"{f}:{dotted(p)}" for f, fl in impl_flat.items() for p in fl
                                        if f"{f}:{dotted(p)}" not in used_paths)[:2000]
    res["all_files_present"] = all(res["impl_files"][f]["present"] for f in expected)
    res["all_agree"] = bool(res["all_files_present"] and counts["disagree"] == 0 and counts["unmatched"] == 0
                            and counts["ambiguous"] == 0 and npz_files and not arr_fail)
    res["note"] = ("key mapping is heuristic until IMPL_MAP is filled from the implementation's actual layout; "
                   "unmatched or ambiguous quantities are not failures of agreement until mapped")
    K.save_json(OUT, res)
    print(json.dumps({"all_agree": res["all_agree"], "counts": counts, "files": res["impl_files"],
                      "arrays_without_twin": len(arr_fail), "npz_files": list(res["impl_npz_files"])}, indent=1))


if __name__ == "__main__":
    main()
