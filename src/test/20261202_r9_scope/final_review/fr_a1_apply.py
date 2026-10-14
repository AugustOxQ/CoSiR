"""Final review A1, items 2 and 3: real r9_stats pass files (planted counts) through run_r9_apply_rule (decide, apply,
main) against my own reading of rule §4, §4.3, §7.6, §10 and R6 rule §4; the agreement is made by ticket 25's own
comparer (rd9_compare.compare_stats + provenance) from ticket 25's own re-derivation (rd9_stats.pass_record).

    python fr_a1_apply.py --part seven|two|main|c13   -> fr_a1_apply_<part>.json beside this file
Everything is written under the scratch folder; the real results/, rederive/rd9_out/ and ledgers are never touched.
"""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
F = HERE.parent
sys.path.insert(0, str(F))
sys.path.insert(0, str(HERE))
import r9_common as CM  # noqa: E402
import r9_stats as ST  # noqa: E402
import run_r9_apply_rule as AR  # noqa: E402
import fr_a1_lib as L  # noqa: E402

sys.path.insert(0, str(F / "rederive"))
import rd9_stats as RS9  # noqa: E402
import rd9_compare as RC  # noqa: E402

SCR = Path("/tmp/claude-0/-project-CoSiR/18057a5a-4e60-48cd-83b0-ff14c8a319ae/scratchpad/fr_a1")
BACKBONE = {"H6": "Qwen3-VL-Embedding-2B", "H7": "SigLIP 2 so400m/14-384", "H8": "CLIP ViT-B/32",
            "H9": "Qwen3-VL-Embedding-2B", "H10": "SigLIP 2 so400m/14-384"}
SHORT = {"H6": "Qwen", "H7": "SigLIP 2"}
NAME = {"P1": "cosine", "P2": "RCA", "P3": "B", "P4": "B′(A0)", "P5": "its matched control",
        "P6": "the condition-free scorers on condition gain", "P7": "RCA on condition gain"}


# ---------------------------------------------------------------- data
class Data:
    def __init__(self, design):
        self.design = design
        n_paint = 248 if design == "seven" else 746
        self.cls = L.clusters(n_paint, 12288, seed=31 if design == "seven" else 32)
        self.cl = np.concatenate(self.cls)
        self.dr = L.Draws(self.cl)
        self.rng = np.random.default_rng(9)
        self.lib = {}
        self.seeds = [52, 53, 54] if design == "seven" else [55, 56, 57]
        s42 = L.clusters(n_paint, 12288, seeds=1, seed=33)[0]
        self.sig = {}
        for c in L.CHECKS:
            q = np.clip(np.rint(self.rng.normal(0.4, 2.0, len(s42))), -4, 4)
            self.sig[c] = ST.sigma_split(q / 4.0, s42)
        _, self.m_p = np.unique(self.cl, return_counts=True)

    def q(self, spec):
        """spec: an int count, or 'zero' (mean exactly 0)."""
        if spec not in self.lib:
            if spec == "zero":
                q = np.clip(np.rint(self.rng.normal(0, 1.5, len(self.cl))), -4, 4).astype(np.int64)
                s = int(q.sum())
                for e in self.rng.permutation(len(q)):
                    if s == 0:
                        break
                    if s > 0 and q[e] > -4:
                        q[e] -= 1
                        s -= 1
                    elif s < 0 and q[e] < 4:
                        q[e] += 1
                        s += 1
                assert q.sum() == 0
                self.lib[spec] = q
            else:
                q = L.plant(self.dr, spec, self.rng)
                assert q is not None, spec
                self.lib[spec] = q
        return self.lib[spec]

    def build(self, specs, set_aside=(), flagged=(), nd=None):
        qs = {c: self.q(specs[c]) for c in L.CHECKS}
        per_seed = L.per_seed_dicts(self.cls, qs, self.rng)
        extra = L.extra_for(self.seeds)
        if self.design == "seven":
            rec = ST.pass_record(per_seed, "held", self.seeds, extra, secondary=False)
        else:
            rec = ST.pass_record_a28(per_seed, "a28", self.seeds, extra, set_aside=list(set_aside),
                                     flagged_languages=list(flagged),
                                     near_duplicates=nd or {"cos_099": 1, "cos_098": 4})
        arrays = {"cl": self.cl}
        for key in ("aff_fused", "aff_cf", "cosine", "rca", "B", "B0"):
            for m in ("r1", "gain"):
                arrays[f"{key}__{m}"] = np.concatenate([s[key][m] for s in per_seed])
        own = RS9.pass_record(arrays, a28=self.design != "seven")
        own.update(seeds=self.seeds, episodes_sha256={str(s): v for s, v in extra["episodes_sha256"].items()})
        sens = ST.sensitivity_record(self.sig, self.m_p, len(self.cl), self.design)
        return rec, own, sens


def agreement(own, pass_path, read, report_extra=None):
    """Ticket 25's comparer on its own re-derivation against the pass file (as rd9_compare.main, scalars only)."""
    raw = pass_path.read_bytes()
    report = json.loads(raw)
    c = RC.Comparison()
    extra = RC.provenance({"stats": own}, pass_path, raw, report)
    RC.compare_stats(c, own, report)
    for k, (mine, theirs) in (report_extra or {}).items():
        c.value(k, mine, theirs)
    res = c.record()
    res.update(independent_identity={"record_sha256": "1" * 64, "arrays_sha256": "2" * 64}, read=read,
               backbone="x", **extra)
    return res


def stage(d, read, rec, own, sens, a28_items=None):
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"held_pass_{read}.json"
    p.write_text(json.dumps(rec, allow_nan=False))
    (d / f"sensitivity_{read}.json").write_text(json.dumps(sens, allow_nan=False))
    agr = agreement(own, p, read, a28_items)
    (d / f"rd9_{read}_agreement.json").write_text(json.dumps(agr))
    if not (d / "smoke_record.json").exists():
        (d / "smoke_record.json").write_text(json.dumps({"passed": True, "module_sha256": CM.r9_module_shas()}))
    return agr


# ---------------------------------------------------------------- my reading of the rule
def expect(read, rec, sens, design):
    checks = rec["checks"]
    counts = {c: checks[c]["n"] for c in L.CHECKS}
    point = {c: checks[c]["point"] for c in L.CHECKS}
    z = {"seven": 3.532, "level1": 3.083, "level2": 3.417}
    out = {"checks": {}, "boundary": []}

    def fam(names, kind, level):
        order, h = L.my_holm(counts, names)
        stop = next((c for c in order if not h[c]["own_count_passes"]), None)
        for c in names:
            r = h[c]
            e = {"passes": r["passes"], "level": level}
            if not r["passes"]:
                e["reading_kind"] = "inconclusive" if point[c] > 0 else "did not beat"
                e["not_reached_after"] = stop if r["own_count_passes"] else None
                e["x"] = z[kind] * sens[c]["SE"] if point[c] > 0 else None
                if point[c] <= 0:
                    e["text"] = f"AFF did not beat {NAME[c]} on new paintings"
            out["checks"][c] = e
            if r["near_boundary"]:
                out["boundary"].append((c, r["holm_k"], counts[c], r["n_star"]))
        return all(h[c]["passes"] for c in names)

    if design == "seven":
        ok = fam(L.CHECKS, "seven", None)
        failed = [c for c in L.CHECKS if not out["checks"][c]["passes"]]
        out["verdict"] = "GO" if ok else "NO-GO"
        out["kind"] = None if ok else ("inconclusive" if all(point[c] > 0 for c in failed) else "did not beat")
        out["claim"] = (f"on ArtELingo's held paintings, AFF re-fitted by its fixed recipe on frozen {SHORT[read]} "
                        f"features beats cosine, RCA, B, B′(A0) and its matched control on R@1, with a positive "
                        f"condition gain over its control and RCA, pooled over three aspect pairs") if ok else None
    else:
        ok1 = fam(L.L1, "level1", 1)
        ok2 = None
        if ok1:
            ok2 = fam(L.L2, "level2", 2)
        else:
            for c in L.L2:
                out["checks"][c] = {"passes": None, "level": 2}
        failed = [c for c in L.CHECKS if out["checks"][c]["passes"] is False]
        out["verdict"] = "full claim" if ok2 else ("small claim" if ok1 else "NO-GO")
        out["kind"] = None if ok1 else ("inconclusive" if all(point[c] > 0 for c in failed) else "did not beat")
        n_lang = 27 - len({s["language"] for s in rec["set_aside"]})
        out["n_languages"] = n_lang
        if ok2:
            out["claim_has"] = [f"with captions from the {n_lang} languages that entered the read",
                                "machine-translated to English, AFF trained on English ArtELingo keeps the same "
                                "sentence on that backbone", "Not native-language text, not any single language."]
        elif ok1:
            out["claim_has"] = ["beats B′(A0) on R@1 with a positive condition gain over its matched control",
                                "Not native-language text, not any single language."]
        else:
            out["claim_has"] = None
        if out["claim_has"] is not None:
            out["claim_has"] += [BACKBONE[read]] + [s["language"] for s in rec["set_aside"]] + list(
                rec["flagged_languages"])
    out["failed"] = failed
    return out


def compare(v, want, rd):
    diffs = []
    for k in ("verdict", "kind", "failed"):
        if v[k] != want[k]:
            diffs.append([k, v[k], want[k]])
    for c in L.CHECKS:
        got, w = v["checks"][c], want["checks"][c]
        if got["passes"] != w["passes"]:
            diffs.append([c, "passes", got["passes"], w["passes"]])
        if w["passes"] is False:
            if got.get("reading_kind") != w["reading_kind"]:
                diffs.append([c, "reading_kind", got.get("reading_kind"), w["reading_kind"]])
            if got.get("not_reached_after") != w["not_reached_after"]:
                diffs.append([c, "not_reached_after", got.get("not_reached_after"), w["not_reached_after"]])
            if w["x"] is not None and abs(got.get("x", -1) - w["x"]) > 1e-9:
                diffs.append([c, "x", got.get("x"), w["x"]])
            if w["x"] is not None and f"{w['x']:.3f}" not in got.get("reading", ""):
                diffs.append([c, "x not in reading", got.get("reading")])
            if "text" in w and w["text"] not in got.get("reading", ""):
                diffs.append([c, "reading text", got.get("reading"), w["text"]])
            if w["not_reached_after"] and "not reached: the Holm procedure stopped at " + w["not_reached_after"] \
                    not in got.get("reading", ""):
                diffs.append([c, "not reached text", got.get("reading")])
        if w["passes"] is None and (got.get("reading") is not None or got.get("reading_kind") is not None):
            diffs.append([c, "untested check carries a reading", got.get("reading")])
    bnd = [(b["check"], b["holm_k"], b["n"], b["n_star"]) for b in v["boundary_report"]]
    if sorted(bnd) != sorted(want["boundary"]):
        diffs.append(["boundary", bnd, want["boundary"]])
    for b in v["boundary_report"]:
        if b["rederivation_n"] != rd[b["check"]]:
            diffs.append(["boundary rederivation_n", b])
    if "claim_has" in want:
        if want["claim_has"] is None:
            if v["claim"] is not None:
                diffs.append(["claim should be None", v["claim"]])
        else:
            miss = [p for p in want["claim_has"] if p not in (v["claim"] or "")]
            if miss:
                diffs.append(["claim lacks", miss, v["claim"]])
        if v["n_languages"] != want["n_languages"]:
            diffs.append(["n_languages", v["n_languages"], want["n_languages"]])
    elif v["claim"] != (want["claim"] + ". Not each pair, not another dataset, not a margin size." if want["claim"]
                        else None):
        diffs.append(["claim", v["claim"], want["claim"]])
    return diffs


def run_case(D, d, read, specs, **kw):
    rec, own, sens = D.build(specs, **{k: v for k, v in kw.items() if k in ("set_aside", "flagged", "nd")})
    a28_items = None
    if D.design != "seven":
        a28_items = {"set_aside": (RC.reported_set_aside(rec["set_aside"]), RC.reported_set_aside(rec["set_aside"])),
                     "flagged_languages": (sorted(rec["flagged_languages"]), sorted(rec["flagged_languages"]))}
    agr = stage(d, read, rec, own, sens, a28_items)
    want = expect(read, rec, sens, D.design)
    # ticket 25's own decision from its own record
    rd_dec = RS9.decide({c: own["checks"][c]["n"] for c in L.CHECKS}, a28=D.design != "seven")["outcome"]
    try:
        _, v = AR.apply(read, results=d, agreement_dir=d, ledger=d / "ledger.md")
    except AR.Stop as e:
        return {"specs": specs, "stop": [e.code, e.msg], "agreement": agr["agreement"]}
    diffs = compare(v, want, agr["rederived_counts"])
    if rd_dec != v["verdict"]:
        diffs.append(["rd9 decide", rd_dec, v["verdict"]])
    return {"specs": {c: specs[c] for c in L.CHECKS}, "counts": {c: rec["checks"][c]["n"] for c in L.CHECKS},
            "points": {c: round(rec["checks"][c]["point"], 3) for c in L.CHECKS}, "agreement": agr["agreement"],
            "agreement_failed": agr["failed"], "verdict": v["verdict"], "kind": v["kind"],
            "readings": {c: v["checks"][c].get("reading") for c in L.CHECKS}, "claim": v["claim"],
            "boundary": v["boundary_report"], "diffs": diffs}


def cases_seven():
    cs = {"all_pass": {c: 0 for c in L.CHECKS}}
    for c in L.CHECKS:
        for sign, spec in (("pos", 200), ("neg", 4000), ("zero", "zero")):
            s = {x: 0 for x in L.CHECKS}
            s[c] = spec
            cs[f"{c}_alone_{sign}"] = s
    # Holm stops at ranks 1, 2 and 4; later ones not reached (their own counts pass)
    cs["stop_rank1"] = dict(zip(("P5", "P1", "P2", "P3", "P4", "P6", "P7"), (17, 17, 18, 20, 25, 35, 50)))
    cs["stop_rank2"] = dict(zip(("P1", "P7", "P2", "P3", "P4", "P5", "P6"), (0, 20, 21, 25, 35, 50, 60)))
    cs["stop_rank4"] = dict(zip(("P1", "P2", "P3", "P6", "P4", "P5", "P7"), (0, 0, 0, 31, 35, 50, 100)))
    cs["stop_rank2_did_not_beat"] = dict(zip(("P1", "P7", "P2", "P3", "P4", "P5", "P6"), (0, 4000, 4000, 4000, 4000, 4000, 4000)))
    cs["boundary_all_at_nstar"] = dict(zip(L.CHECKS, (16, 19, 24, 30, 40, 61, 124)))
    cs["boundary_rank3_plus1"] = dict(zip(L.CHECKS, (16, 19, 25, 30, 40, 61, 124)))
    return cs


def cases_two():
    z = {c: 0 for c in L.CHECKS}
    cs = {"full": dict(z), "full_at_boundaries": {"P4": 61, "P6": 124, "P1": 24, "P2": 30, "P3": 40, "P5": 61, "P7": 124}}
    for c in L.L1:
        for sign, spec in (("pos", 200), ("neg", 4000), ("zero", "zero")):
            cs[f"L1_{c}_{sign}"] = {**z, c: spec}
    cs["L1_both_fail_pos"] = {**z, "P4": 200, "P6": 300}
    cs["L1_both_fail_mixed"] = {**z, "P4": 200, "P6": 4000}
    cs["L1_fail_L2_also_bad"] = {**{c: 4000 for c in L.L2}, "P4": 62, "P6": 0}
    cs["L1_tie_at_62"] = {**z, "P4": 62, "P6": 62}
    for c in L.L2:
        for sign, spec in (("pos", 200), ("neg", 4000), ("zero", "zero")):
            cs[f"L2_{c}_{sign}"] = {**z, c: spec}
    cs["L2_stop_rank2"] = {**z, "P1": 0, "P7": 31, "P2": 35, "P3": 45, "P5": 60}
    return cs


def part(design):
    D = Data(design)
    reads = ("H6", "H7") if design == "seven" else ("H8", "H9", "H10")
    cs = cases_seven() if design == "seven" else cases_two()
    out, n_diff = {}, 0
    for j, (name, specs) in enumerate(cs.items()):
        read = reads[j % len(reads)]
        kw = {}
        if design != "seven" and j % 3 == 1:
            kw = {"set_aside": [{"language": "Kyrgyz", "pair": "emotion__genre", "seed": 56},
                                {"language": "Kyrgyz", "pair": "style__genre", "seed": 57},
                                {"language": "Emakhuwa", "pair": "emotion__genre", "seed": 56}],
                  "flagged": ["Emakhuwa", "Igbo"]}
        d = SCR / f"apply_{design}" / name
        if d.exists():
            shutil.rmtree(d)
        r = run_case(D, d, read, specs, **kw)
        r["read"] = read
        n_diff += len(r.get("diffs", [1]))
        out[name] = r
    return {"cases": len(cs), "differences": n_diff, "results": out}


# ---------------------------------------------------------------- main, refusals, not-run
def part_main():
    res = {}
    D = Data("seven")
    d = SCR / "main_seven"
    if d.exists():
        shutil.rmtree(d)
    rec, own, sens = D.build({**{c: 0 for c in L.CHECKS}, "P3": 31, "P6": 35})
    stage(d, "H6", rec, own, sens)
    led = d / "ledger.md"
    shutil.copy(CM.CHECKOUT / "docs/superpowers/held_ledger.md", led)
    saved = (AR.RESULTS, AR.AGREEMENT_DIR, AR.LEDGER)
    AR.RESULTS, AR.AGREEMENT_DIR, AR.LEDGER = d, d, led
    try:
        for attempt in (1, 2):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    code = AR.main(["--read", "H6"])
                except SystemExit as e:
                    code = e.code
            vp = d / "held_verdict_H6.json"
            res[f"H6_run{attempt}"] = {"exit": code, "stdout": buf.getvalue().splitlines(),
                                       "verdict_sha": hashlib.sha256(vp.read_bytes()).hexdigest() if vp.exists() else None}
        # not-run rows
        nr = SCR / "main_notrun"
        if nr.exists():
            shutil.rmtree(nr)
        nr.mkdir(parents=True)
        lines = ["| # | Date | Dataset | Backbone | Purpose | Episode | Script | Report |", "|---|---|---|---|---|---|---|---|",
                 "| H7 | 2026-10-14 | x | SigLIP 2 | z | not run: switched off | abc | - |",
                 "| H10 | 2026-10-14 | x | SigLIP 2 | z | not run: switched off | abc | - |",
                 "| H8 | 2026-10-14 | x | CLIP | z | not run: switched off | abc | - |",
                 "| H9 | 2026-10-14 | x | Qwen | z | dropped under the rule | abc | - |",
                 "| H6 | 2026-10-14 | x | Qwen | z | (pending) | abc | - |"]
        (nr / "ledger.md").write_text("\n".join(lines) + "\n")
        AR.RESULTS, AR.LEDGER = nr, nr / "ledger.md"
        for read, kind in (("H10", "switched_off"), ("H7", "switched_off"), ("H10", "switched_off"),
                           ("H8", "switched_off"), ("H9", "dropped"), ("H6", "dropped"), ("H9", "switched_off")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    code = AR.main(["--read", read, "--not-run", kind])
                except SystemExit as e:
                    code = e.code
            vp = nr / f"held_verdict_{read}.json"
            v = json.loads(vp.read_text()) if vp.exists() else None
            res.setdefault("not_run", []).append({"read": read, "kind": kind, "exit": code,
                                                   "stdout": buf.getvalue().strip()[-200:],
                                                   "status": v and v.get("status"), "verdict": v and v.get("verdict")})
    finally:
        AR.RESULTS, AR.AGREEMENT_DIR, AR.LEDGER = saved
    return res


# ---------------------------------------------------------------- C13: the comparer sees each mutated field
def part_c13():
    out = {}
    for design, read in (("seven", "H6"), ("two_level", "H9")):
        D = Data(design)
        specs = {**{c: 0 for c in L.CHECKS}, "P3": 31, "P6": 50} if design == "seven" else \
            {**{c: 0 for c in L.CHECKS}, "P4": 20, "P2": 29}
        rec, own, sens = D.build(specs, set_aside=[{"language": "Thai", "pair": "style__genre", "seed": 55}]
                                 if design != "seven" else ())
        base = SCR / f"c13_{design}"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)

        def run(mut):
            r = copy.deepcopy(rec)
            mut(r)
            p = base / f"held_pass_{read}.json"
            p.write_text(json.dumps(r))
            items = None
            if design != "seven":
                items = {"set_aside": (sorted({("Thai", "style__genre", 55)}), RC.reported_set_aside(r["set_aside"])),
                         "flagged_languages": (sorted([]), sorted(r["flagged_languages"]))}
            a = agreement(own, p, read, items)
            return a["agreement"], [x["key"] for x in a["items"] if not x["equal"]][:4]

        first = "P1"
        o = rec["holm_order"] if design == "seven" else rec["holm_order"]["level2"]
        muts = {
            "none": lambda r: None,
            "n+1": lambda r: r["checks"]["P3"].__setitem__("n", r["checks"]["P3"]["n"] + 1),
            "passes flip": lambda r: r["checks"]["P3"].__setitem__("passes", not r["checks"]["P3"]["passes"]),
            "own flip": lambda r: r["checks"]["P2"].__setitem__("own_count_passes",
                                                                 not r["checks"]["P2"]["own_count_passes"]),
            "holm_k": lambda r: r["checks"][first].__setitem__("holm_k", 7),
            "near_boundary flip": lambda r: r["checks"]["P5"].__setitem__("near_boundary",
                                                                         not r["checks"]["P5"]["near_boundary"]),
            "point sign": lambda r: r["checks"]["P3"].__setitem__("point", -r["checks"]["P3"]["point"]),
            "point +1e-6": lambda r: r["checks"]["P3"].__setitem__("point", r["checks"]["P3"]["point"] + 1e-6),
            "ci95 hi": lambda r: r["checks"]["P3"]["ci95"].__setitem__(1, r["checks"]["P3"]["ci95"][1] + 0.01),
            "ci_holm lo": lambda r: r["checks"]["P3"]["ci_holm"].__setitem__(0, r["checks"]["P3"]["ci_holm"][0] - .01),
            "level_two_sided": lambda r: r["checks"]["P3"].__setitem__("level_two_sided", 0.95),
            "holm_order swap": (lambda r: r.__setitem__("holm_order", [o[1], o[0]] + o[2:])) if design == "seven"
            else (lambda r: r["holm_order"].__setitem__("level2", [o[1], o[0]] + o[2:])),
            "n_clusters": lambda r: r.__setitem__("n_clusters", r["n_clusters"] - 1),
            "seeds": lambda r: r.__setitem__("seeds", [r["seeds"][0], r["seeds"][2], r["seeds"][1]]),
            "mode": lambda r: r.__setitem__("mode", "regression"),
            "rule_sha256": lambda r: r.__setitem__("rule_sha256", "0" * 64),
            "key quantity": lambda r: r["checks"]["P3"].__setitem__("quantity", "x"),
        }
        if design != "seven":
            muts.update({
                "level swap": lambda r: (r["checks"]["P4"].__setitem__("level", 2)),
                "design drop": lambda r: r.pop("design"),
                "set_aside drop": lambda r: r.__setitem__("set_aside", []),
                "set_aside add": lambda r: r["set_aside"].append({"language": "Urdu", "pair": "emotion__style",
                                                                  "seed": 57}),
                "flagged add": lambda r: r["flagged_languages"].append("Hausa"),
            })
        out[design] = {k: run(m) for k, m in muts.items()}
    # the sensitivity file is not read by the comparer at all: show it by construction
    out["comparer_reads_sensitivity"] = "sensitivity" in Path(RC.__file__).read_text()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("seven", "two", "main", "c13"))
    a = ap.parse_args()
    t0 = time.time()
    res = {"seven": lambda: part("seven"), "two": lambda: part("two_level"), "main": part_main,
           "c13": part_c13}[a.part]()
    res["seconds"] = round(time.time() - t0, 1)
    (HERE / f"fr_a1_apply_{a.part}.json").write_text(json.dumps(res, indent=1, default=str, ensure_ascii=False))
    print(a.part, {k: res[k] for k in ("cases", "differences", "seconds") if k in res})


if __name__ == "__main__":
    main()
