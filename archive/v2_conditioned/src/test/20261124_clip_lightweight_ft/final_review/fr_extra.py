"""Final review extras: AFF minus plain/B/B'(A0) pooled intervals (quoted from round 3 in Table S1), per-seed
per-side lifts (robustness of the genre finding), exact values behind a few rounded text numbers."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import fr_episodes as E
R = json.loads((HERE / "fr_episodes.json").read_text())
P = R["pooled"]["scorers"]
print("cache ref pooled r1 diff", R["pooled"]["comparisons"]["CLIPcache"]["ft_minus_plain"]["r1"])
print("cache ref per seed r1 diffs", {s: round(R["seeds"][s]["comparisons"]["CLIPcache"]["ft_minus_plain"]["r1"]["point"], 4) for s in R["seeds"]})
print("cache ref per pair r1 diffs", {i: round(R["pairs"][i]["comparisons"]["CLIPcache"]["ft_minus_plain"]["r1"]["point"], 4) for i in R["pairs"]})
s42 = R["seeds"]["42"]["scorers"]
print("seed42 Bp1 - ft", {v: round(s42["Bp1"]["r1"]["point"] - s42["ft:" + v]["r1"]["point"], 4) for v in ("LP", "LB", "LoRA")})
print("e x g emotion-side lifts exact", {v: R["pairs"]["1"]["side_rates"]["ft:" + v]["A"] - R["pairs"]["1"]["side_rates"]["plain"]["A"] for v in ("LP", "LB", "LoRA")})
print("pooled sides lifts", {v: (round(R["pooled"]["side_rates"]["ft:" + v]["A"] - R["pooled"]["side_rates"]["plain"]["A"], 3), round(R["pooled"]["side_rates"]["ft:" + v]["B"] - R["pooled"]["side_rates"]["plain"]["B"], 3)) for v in ("LP", "LB", "LoRA")})
print("PER SEED side lifts (A-side, B-side) per pair:")
for s in ("49", "50", "51"):
    for i in range(3):
        sr = R["pairs_per_seed"][f"{s}:{i}"]["side_rates"]
        print("  seed", s, "pair", i, {v: (round(sr["ft:" + v]["A"] - sr["plain"]["A"], 2), round(sr["ft:" + v]["B"] - sr["plain"]["B"], 2)) for v in ("LP", "LB", "LoRA")})
for i in ("42",):
    for p in range(3):
        sr = R["pairs_seed42"][str(p)]["side_rates"]
        print("  seed 42 pair", p, {v: (round(sr["ft:" + v]["A"] - sr["plain"]["A"], 2), round(sr["ft:" + v]["B"] - sr["plain"]["B"], 2)) for v in ("LP", "LB", "LoRA")})
# per-seed per-pair floor lifts
print("PER SEED pair r1 lifts")
for s in ("49", "50", "51"):
    print("  seed", s, {i: [round(R["pairs_per_seed"][f"{s}:{i}"]["comparisons"][v]["ft_minus_plain"]["r1"]["point"], 2) for v in ("LP", "LB", "LoRA")] for i in range(3)})
# per-seed intervals excluding zero
ok = all((R["seeds"][s]["comparisons"][v][c]["r1"]["ci95"][0] > 0) == (c != "ft_minus_Bp0") and (R["seeds"][s]["comparisons"][v][c]["r1"]["ci95"][1] < 0) == (c == "ft_minus_Bp0")
         for s in R["seeds"] for v in ("LP", "LB", "LoRA") for c in ("AFF_minus_ft", "ft_minus_plain", "ft_minus_Bp0"))
print("all per-seed ft comparisons exclude 0 with the stated sign:", ok)
# per-pair ft comparisons exclude 0?
okp = all((R["pairs"][i]["comparisons"][v][c]["r1"]["ci95"][0] > 0) or (R["pairs"][i]["comparisons"][v][c]["r1"]["ci95"][1] < 0)
          for i in R["pairs"] for v in ("LP", "LB", "LoRA") for c in ("AFF_minus_ft", "ft_minus_plain", "ft_minus_Bp0"))
print("all per-pair (pooled) ft comparisons exclude 0:", okp)
# 'sits between plain CLIP and B on every seed, every pair pooled and on seed 42'
btw = []
for sc_ in [R["seeds"][s]["scorers"] for s in R["seeds"]] + [R["pairs"][i]["scorers"] for i in R["pairs"]] + [R["pairs_seed42"][i]["scorers"] for i in R["pairs_seed42"]]:
    btw.append(all(sc_["plain"]["r1"]["point"] < sc_["ft:" + v]["r1"]["point"] < sc_["B"]["r1"]["point"] for v in ("LP", "LB", "LoRA")))
print("ft between plain and B on every seed, pooled pair and seed-42 pair:", all(btw), btw)
