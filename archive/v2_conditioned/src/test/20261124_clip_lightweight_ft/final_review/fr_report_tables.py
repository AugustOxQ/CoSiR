"""Final review: rebuild the report's numeric tables from fr_episodes.json (own derivation) and diff them against the
report's markdown rows (2-decimal rounding, Unicode minus)."""
import json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
R = json.loads((HERE / "fr_episodes.json").read_text())
REP = (ROOT / "docs/reports/auto/v2/2026-11-24_clip_lightweight_ft.md").read_text().splitlines()
def s(x, sign=True):
    t = f"{x:+.2f}" if sign else f"{x:.2f}"
    return t.replace("-", "−")
def ci(d, sign=True):
    return f"{s(d['point'], sign)} [{s(d['ci95'][0], sign)}, {s(d['ci95'][1], sign)}]"
def row(cells): return "| " + " | ".join(cells) + " |"
exp = []
P = R["pooled"]; sc = P["scorers"]; cmp = P["comparisons"]
V = ["LP", "LB", "LoRA"]
# Table S1 (AFF-minus intervals for plain/B/Bp0 come from round 3; recomputed here in fr_extra)
for v in V:
    exp.append(("S1 " + v, f"| {v}, fine-tuned (cosine) | {ci(sc['ft:'+v]['r1'], False)} | {s(sc['ft:'+v]['either']['point'], False)} | {ci(cmp[v]['AFF_minus_ft']['r1'])} | {ci(cmp[v]['ft_minus_plain']['r1'])} |"))
# Table 3
for lab, c, m in (("fine-tuned minus plain CLIP, R@1", "ft_minus_plain", "r1"), ("fine-tuned minus plain CLIP, either", "ft_minus_plain", "either"),
                  ("fine-tuned minus B′(A0), R@1", "ft_minus_Bp0", "r1"), ("fine-tuned minus B′(A0), either", "ft_minus_Bp0", "either"),
                  ("AFF minus fine-tuned, R@1", "AFF_minus_ft", "r1"), ("AFF minus fine-tuned, either", "AFF_minus_ft", "either")):
    exp.append(("T3", row([lab] + [ci(cmp[v][c][m]) for v in V])))
# Tables 4, 5
names = [("plain CLIP", "plain"), ("cache-path reference", "ft:CLIPcache"), ("LP", "ft:LP"), ("LB", "ft:LB"), ("LoRA", "ft:LoRA"),
         ("B", "B"), ("B′(A0)", "Bp0"), ("B′(A1)", "Bp1"), ("AFF", "AFF")]
for tab, m in (("T4", "r1"), ("T5", "either")):
    for lab, k in names:
        if tab == "T5" and k == "ft:CLIPcache":
            continue
        cells = []
        for sd in ("42", "49", "50", "51"):
            d = R["seeds"][sd]["scorers"].get(k)
            cells.append(s(d[m]["point"], False) if d else "")
        cells.append(s(P["scorers"][k][m]["point"], False) if k in P["scorers"] else "")
        exp.append((tab, row([lab] + cells)))
# Table 6
for c, lab in (("AFF_minus_ft", "AFF minus {}"), ("ft_minus_plain", "{} minus plain"), ("ft_minus_Bp0", "{} minus B′(A0)")):
    for v in V:
        exp.append(("T6", row([lab.format(v)] + [ci(R["seeds"][sd]["comparisons"][v][c]["r1"]) for sd in ("42", "49", "50", "51")])))
# Table 8
for lab, k in (("plain CLIP", "plain"), ("LP", "ft:LP"), ("LB", "ft:LB"), ("LoRA", "ft:LoRA"), ("B", "B"), ("B′(A0)", "Bp0"), ("AFF", "AFF")):
    exp.append(("T8", row([lab] + [s(R["pairs"][str(i)]["scorers"][k]["r1"]["point"], False) for i in range(3)])))
exp.append(("T8", row(["AFF's condition gain"] + [s(R["pairs"][str(i)]["scorers"]["AFF"]["gain"]["point"], False) for i in range(3)])))
# Table 9
for c, lab in (("ft_minus_plain", "{} minus plain"), ("ft_minus_Bp0", "{} minus B′(A0)"), ("AFF_minus_ft", "AFF minus {}")):
    for v in V:
        exp.append(("T9", row([lab.format(v)] + [ci(R["pairs"][str(i)]["comparisons"][v][c]["r1"]) for i in range(3)])))
# Table 10
lab = {0: ("emotion × style", "emotion", "style"), 1: ("emotion × genre", "emotion", "genre"), 2: ("style × genre", "style", "genre")}
for i in range(3):
    sr = R["pairs"][str(i)]["side_rates"]
    for side, nm in (("A", lab[i][1]), ("B", lab[i][2])):
        vals = [sr[k][side] for k in ("plain", "ft:LP", "ft:LB", "ft:LoRA")]
        lifts = [v - vals[0] for v in vals[1:]]
        exp.append(("T10", row([f"{lab[i][0]}, {nm} candidate"] + [s(v, False) for v in vals] + [f"{s(min(lifts))} to {s(max(lifts))}"])))
lines = set(l.strip() for l in REP)
bad = [(t, e) for t, e in exp if e not in lines]
print(f"{len(exp)} expected rows; {len(exp) - len(bad)} found verbatim; {len(bad)} not found")
for t, e in bad:
    # show the closest report line (same first cell)
    first = e.split("|")[1].strip()
    cand = [l for l in REP if l.strip().startswith("| " + first + " |")]
    print(t, "EXPECTED", e)
    for c in cand: print("    REPORT ", c.strip())
