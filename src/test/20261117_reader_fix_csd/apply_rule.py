"""Apply DECISION_RULE.md §5 items 3 to 5 (development bar, carry, kill) to the stored candidate results of seed 42.
Reads results/cand_<name>.json for the seven candidates of item 1 and writes results/rule_application.json and .txt
(refuses to overwrite). Run from /project/CoSiR:
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261117_reader_fix_csd/apply_rule.py
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RULE_SHA = "613d8c9d13fb7d4787ec0fb91dc65cfd4a192d07c88ab8f961da3adc1c35d23c"
BASE = ("Ra_A1", "Rb_argmax_A1", "Rb_expected_A1", "Ra_A0", "Rb_argmax_A0", "Rb_expected_A0")
ORDER = ("Ra", "Rb_argmax", "Rb_expected", "Rc")          # item 4 tie order
BAR, TIE = 0.5, 0.05


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def find(o, key):
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = find(v, key)
            if r is not None:
                return r
    return None


def main():
    if sha(HERE / "DECISION_RULE.md") != RULE_SHA:
        sys.exit("DECISION_RULE.md differs from the committed rule (SHA-256)")
    out_json, out_txt = RES / "rule_application.json", RES / "rule_application.txt"
    if out_json.exists() or out_txt.exists():
        sys.exit(f"{out_json} exists; refusing to overwrite")
    rc = sorted(p.stem[len("cand_"):] for p in RES.glob("cand_Rc_*.json"))
    if len(rc) > 1:
        sys.exit(f"more than one R-c candidate: {rc}")
    cands = [n for n in BASE if (RES / f"cand_{n}.json").exists()] + rc
    rows = {}
    for n in cands:
        d = json.loads((RES / f"cand_{n}.json").read_text())
        bar = find(d, "bar")
        gain = find(d, "gain_statistic")
        r = {"config": n.rsplit("_", 1)[-1], "reader": next(o for o in ORDER if n.startswith(o + "_")),
             "bar_margin": bar["r1"]["point"], "bar_ci95": bar["r1"]["ci95"], "comparator": bar["comparator"],
             "gain_statistic": gain["point"], "gain_ci95": gain["ci95"], "file_sha256": sha(RES / f"cand_{n}.json")}
        r["clause1"] = r["bar_margin"] >= BAR
        r["clause2"] = r["bar_ci95"][0] > 0
        r["clause3"] = r["gain_ci95"][0] > 0
        r["clears"] = r["clause1"] and r["clause2"] and r["clause3"]
        rows[n] = r
    missing = [n for n in BASE if n not in rows] + ([] if rc else ["Rc_<parent>"])
    E = [n for n in rows if rows[n]["clears"]]
    pool = [n for n in E if rows[n]["config"] == "A1"] or E
    if pool:
        M = max(rows[n]["bar_margin"] for n in pool)
        tied = [n for n in pool if rows[n]["bar_margin"] >= M - TIE]
        carried = min(tied, key=lambda n: ORDER.index(rows[n]["reader"]))
        action = f"carry {carried} to the fresh-seed test (DECISION_RULE.md §6)"
    else:
        tied, carried = [], None
        action = ("item 5: no candidate clears the development bar; no test is built; the result goes to the user "
                  "for the Friday choice (design L, a change of course, or another reader round under a new rule)")
    now = datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d %H:%M")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    result = {"rule_sha256": RULE_SHA, "time_amsterdam": now, "git_head": head, "script_sha256": sha(__file__),
              "candidates": rows, "missing": missing, "eligible": E, "pool": pool, "tied": tied,
              "carried": carried, "action": action}
    out_json.write_text(json.dumps(result, indent=1))
    L = [f"Rule application (DECISION_RULE.md {RULE_SHA[:12]}), {now}",
         f"{'candidate':18s} {'bar margin':>10s}  {'95% interval':>18s}  {'comparator':11s} {'gain stat':>9s}  {'gain lower':>10s}  clauses  clears"]
    for n, r in rows.items():
        L.append(f"{n:18s} {r['bar_margin']:+10.4f}  [{r['bar_ci95'][0]:+.3f}, {r['bar_ci95'][1]:+.3f}]  "
                 f"{r['comparator']:11s} {r['gain_statistic']:+9.3f}  {r['gain_ci95'][0]:+10.3f}  "
                 f"{int(r['clause1'])}{int(r['clause2'])}{int(r['clause3'])}      {r['clears']}")
    L += [f"missing: {missing or 'none'}", f"eligible: {E or 'none'}", f"carried: {carried}", f"action: {action}"]
    out_txt.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
