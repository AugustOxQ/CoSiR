"""Final review A1, item 1: r9_stats against my own reference (fr_a1_lib) on real-shape arrays with planted counts.

    python fr_a1_stats.py --part pure|seven|a28|sens   -> fr_a1_stats_<part>.json beside this file
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
F = HERE.parent
sys.path.insert(0, str(F))
sys.path.insert(0, str(HERE))
import r9_common as CM  # noqa: E402,F401
import r9_stats as ST  # noqa: E402
import fr_a1_lib as L  # noqa: E402

TOL = 1e-9   # R3 rule §8: points and bounds within 1e-9 R@1 points


def part_pure():
    out = {"int_vs_frac_vs_float_mismatch": [], "boundary": {}, "holm_random": 0, "holm_diffs": []}
    for m in (2, 5, 7):
        for k in range(1, m + 1):
            for n in range(0, 5001):
                a, b, c = L.own_int(n, k, m), L.own_frac(n, k, m), L.own_float(n, k, m)
                if not (a == b == c):
                    out["int_vs_frac_vs_float_mismatch"].append([m, k, n, a, b, c])
        out["boundary"][m] = {"mine": [L.nstar(k, m) for k in range(1, m + 1)],
                              "ST.boundary": [ST.boundary(k, m) for k in range(1, m + 1)]}
    rng = np.random.default_rng(7)
    fams = [(L.CHECKS, 7), (L.L1, 2), (L.L2, 5)]
    for t in range(30000):
        names, m = fams[t % 3]
        mode = t % 5
        if mode == 0:
            vals = rng.integers(0, 5001, m)
        elif mode == 1:
            vals = rng.integers(0, 130, m)
        elif mode == 2:
            vals = np.full(m, rng.integers(0, 130))
        else:
            bnd = [L.nstar(k, m) for k in range(1, m + 1)]
            vals = np.array([bnd[rng.integers(0, m)] + rng.integers(-1, 2) for _ in range(m)])
        counts = {c: int(v) for c, v in zip(names, vals)}
        order, mine = L.my_holm(counts, names)
        got = ST.holm(counts, names, m)
        out["holm_random"] += 1
        if [g["name"] for g in got] != order:
            out["holm_diffs"].append(["order", counts])
            continue
        for g in got:
            r = mine[g["name"]]
            for f_mine, f_code in (("holm_k", "k"), ("passes", "passes"), ("own_count_passes", "own_count_passes"),
                                   ("level_two_sided", "level_two_sided")):
                if r[f_mine] != g[f_code]:
                    out["holm_diffs"].append([f_mine, counts, g["name"], r[f_mine], g[f_code]])
            if ST.near_boundary(g["n"], g["k"], m) != r["near_boundary"]:
                out["holm_diffs"].append(["near_boundary", counts, g["name"]])
    z = L.z_table()
    out["z"] = {"mine": z, "ST.Z": ST.Z, "ST.Z_95": ST.Z_95,
                "round3_ok": all(round(z[k], 3) == ST.Z[k] for k in ST.Z) and round(z["x95"], 2) == ST.Z_95}
    return out


def compare_checks(rec, mine_stats, counts, design):
    """Every field of rec['checks'] and holm_order against mine. -> list of differences."""
    diffs = []
    fams = [("all", L.CHECKS)] if design == "seven" else [("level1", L.L1), ("level2", L.L2)]
    orders = {}
    for label, names in fams:
        m = len(names)
        order, h = L.my_holm(counts, names)
        orders[label] = order
        for c in names:
            e, s, r = rec["checks"][c], mine_stats[c], h[c]
            want = {"n": s["n"], "holm_k": r["holm_k"], "passes": r["passes"], "own_count_passes": r["own_count_passes"],
                    "near_boundary": r["near_boundary"]}
            for f, v in want.items():
                if e[f] != v or type(e[f]) is not type(v):
                    diffs.append([c, f, e[f], v])
            if abs(e["level_two_sided"] - r["level_two_sided"]) > 1e-15:
                diffs.append([c, "level_two_sided", e["level_two_sided"], r["level_two_sided"]])
            if abs(e["point"] - s["point"]) > TOL:
                diffs.append([c, "point", e["point"], s["point"]])
            for f, v in (("ci95", s["ci95"]), ("ci_holm", L.holm_ci(s["b"], r["holm_k"], m))):
                if len(e[f]) != 2 or max(abs(x - y) for x, y in zip(e[f], v)) > TOL:
                    diffs.append([c, f, e[f], v])
            if design != "seven" and e.get("level") != (1 if label == "level1" else 2):
                diffs.append([c, "level", e.get("level")])
            if s["n"] != s["n_float"]:
                diffs.append([c, "note: float form count differs from integer form", s["n_float"], s["n"]])
    want_order = orders["all"] if design == "seven" else orders
    if rec["holm_order"] != want_order:
        diffs.append(["holm_order", rec["holm_order"], want_order])
    return diffs


def scenarios(design):
    """{name: counts} with every boundary n*-1, n*, n*+1 per rank, ties and all-equal."""
    sc = {}
    fams = [L.CHECKS] if design == "seven" else [L.L1, L.L2]
    for fi, names in enumerate(fams):
        m = len(names)
        for k in range(1, m + 1):
            for off in (-1, 0, 1):
                t = L.nstar(k, m) + off
                cnt = {}
                for j, c in enumerate(names):
                    cnt[c] = 0 if j < k - 1 else (t if j == k - 1 else t + 2 * (j - k + 1))
                if design != "seven":                  # the other level: all zero (passes)
                    for c in fams[1 - fi]:
                        cnt[c] = 0
                sc[f"f{fi + 1}_k{k}_off{off:+d}"] = cnt
                # the same with the names reversed (ties broken by the rule's order, not the input order)
                rev = dict(zip(names, [cnt[c] for c in reversed(names)]))
                if design != "seven":
                    rev.update({c: 0 for c in fams[1 - fi]})
                sc[f"f{fi + 1}_k{k}_off{off:+d}_rev"] = rev
        for v in sorted({L.nstar(1, m) - 1, L.nstar(1, m), L.nstar(1, m) + 1, 40, 0, 5000}):
            cnt = {c: v for c in names}
            if design != "seven":
                cnt.update({c: 0 for c in fams[1 - fi]})
            sc[f"f{fi + 1}_allequal_{v}"] = cnt
        if m > 1:   # a tie between rank 1 and rank 2 exactly at rank 2's n*
            t = L.nstar(2, m)
            cnt = {c: 0 for c in names}
            cnt[names[-1]] = t
            cnt[names[-2]] = t
            if design != "seven":
                cnt.update({c: 0 for c in fams[1 - fi]})
            sc[f"f{fi + 1}_tie_last_two_at_{t}"] = cnt
    if design != "seven":   # both levels failing; level 1 failing with level 2 passing counts
        sc["both_fail"] = {**{c: 200 for c in L.L1}, **{c: 300 for c in L.L2}}
        sc["l1_fail_l2_zero"] = {**{"P4": 0, "P6": 125}, **{c: 0 for c in L.L2}}
    return sc


def part_pass(design):
    n_paint = 248 if design == "seven" else 746
    cls = L.clusters(n_paint, 12288, seed=11 if design == "seven" else 12)
    cl = np.concatenate(cls)
    dr = L.Draws(cl)
    rng = np.random.default_rng(5)
    sc = scenarios(design)
    need = sorted({v for cnt in sc.values() for v in cnt.values()})
    lib, missing = {}, []
    t0 = time.time()
    for v in need:
        q = L.plant(dr, v, rng)
        if q is None:
            missing.append(v)
        else:
            lib[v] = q
    t_plant = time.time() - t0
    results, n_diff = {}, 0
    seeds = [52, 53, 54] if design == "seven" else [55, 56, 57]
    for name, cnt in sc.items():
        if any(v not in lib for v in cnt.values()):
            results[name] = {"skipped": "a planted count was not reached", "counts": cnt}
            continue
        # distinct checks with equal counts get the same quarters; the comparison is per check, so that is fine
        qs = {c: lib[cnt[c]] for c in L.CHECKS}
        per_seed = L.per_seed_dicts(cls, qs, rng)
        extra = L.extra_for(seeds)
        if design == "seven":
            rec = ST.pass_record(per_seed, "held", seeds, extra, secondary=False)
        else:
            sa = [{"language": "Kyrgyz", "pair": "emotion__genre", "seed": 56}]
            rec = ST.pass_record_a28(per_seed, "a28", seeds, extra, set_aside=sa, flagged_languages=["Emakhuwa"],
                                     near_duplicates={"cos_099": 2, "cos_098": 5})
        mine = {c: dr.stats(qs[c]) for c in L.CHECKS}
        diffs = compare_checks(rec, mine, cnt, design)
        top = {"n_episodes": (rec["n_episodes"], len(cl)), "n_clusters": (rec["n_clusters"], dr.k),
               "mode": (rec["mode"], "held"), "seeds": (rec["seeds"], seeds),
               "design": (rec.get("design"), None if design == "seven" else "two_level"),
               "secondary absent": ("secondary" in rec, False)}
        if design != "seven":
            top.update(set_aside=(rec["set_aside"], sa), flagged=(rec["flagged_languages"], ["Emakhuwa"]),
                       nd=(rec["near_duplicates"], {"cos_099": 2, "cos_098": 5}))
        diffs += [[k, a, b] for k, (a, b) in top.items() if a != b]
        n_diff += len([d for d in diffs if not str(d[1]).startswith("note")])
        results[name] = {"counts": cnt, "diffs": diffs,
                         "passes": {c: rec["checks"][c]["passes"] for c in L.CHECKS}, "order": rec["holm_order"]}
        results[name]["pass_record"] = {c: {k: v for k, v in rec["checks"][c].items()} for c in L.CHECKS}
    return {"design": design, "n_paintings": dr.k, "n_episodes": len(cl), "planted": len(lib), "missing": missing,
            "plant_seconds": round(t_plant, 1), "scenarios": len(sc), "field_differences": n_diff, "results": results}


def part_sens():
    out = {}
    for design, n_paint in (("seven", 248), ("two_level", 746)):
        cls = L.clusters(n_paint, 12288, seed=21)
        seed42 = L.clusters(n_paint, 12288, seeds=1, seed=22)[0]
        rng = np.random.default_rng(3)
        sig = {}
        for c in L.CHECKS:
            q = np.clip(np.rint(rng.normal(0, 1.2, n_paint)[np.unique(seed42, return_inverse=True)[1]]
                                + rng.normal(0.3, 1.8, len(seed42))), -4, 4)
            code = ST.sigma_split(q / 4.0, seed42)
            mine = L.variance_split(100 * q / 4.0, seed42)
            sig[c] = code
            out.setdefault("sigma_diffs", [])
            for k in ("sigma_a2", "sigma_eps2"):
                if not np.isclose(code[k], mine[k], rtol=1e-12, atol=1e-12):
                    out["sigma_diffs"].append([design, c, k, code[k], mine[k]])
        cl = np.concatenate(cls)
        _, m_p = np.unique(cl, return_counts=True)
        N = len(cl)
        rec = ST.sensitivity_record(sig, m_p, N, design)
        z = L.z_table()
        diffs = []
        for c in L.CHECKS:
            se = float(np.sqrt((sig[c]["sigma_a2"] * np.sum(m_p.astype(float) ** 2) + sig[c]["sigma_eps2"] * N) / N ** 2))
            kind = "seven" if design == "seven" else ("level1" if c in L.L1 else "level2")
            want = {"SE": se, "x": round(z[kind], 3) * se, "x95": 2.80 * se}
            for k, v in want.items():
                if not np.isclose(rec[c][k], v, rtol=1e-12):
                    diffs.append([c, k, rec[c][k], v])
        out[design] = {"N": N, "n_paintings": int(m_p.size), "diffs": diffs,
                       "x_over_SE": {c: rec[c]["x"] / rec[c]["SE"] for c in L.CHECKS}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("pure", "seven", "a28", "sens"))
    a = ap.parse_args()
    t0 = time.time()
    res = {"pure": part_pure, "seven": lambda: part_pass("seven"), "a28": lambda: part_pass("two_level"),
           "sens": part_sens}[a.part]()
    res["seconds"] = round(time.time() - t0, 1)
    (HERE / f"fr_a1_stats_{a.part}.json").write_text(json.dumps(res, indent=1, default=str))
    keys = ("field_differences", "missing", "scenarios", "planted", "seconds", "holm_random")
    print(a.part, {k: res[k] for k in keys if k in res},
          {k: (len(v) if isinstance(v, list) else v) for k, v in res.items() if k.endswith(("mismatch", "diffs"))})


if __name__ == "__main__":
    main()
