"""Collect every exploratory row of results/bs_*.json into one markdown table (results/bs_summary.md). Exploratory,
seed 42, decides nothing."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
rows = []


def add(src, key, r):
    if not isinstance(r, dict) or "bar" not in r:
        return
    b = r["bar"]
    pp = r.get("per_pair_bar", {})
    rows.append(f"| {src} | {key} | {r['fused_r1']:.3f} | {r['cf_r1']:.3f} | {r['comparator']} | {b['point']:+.3f} "
                f"[{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] | {r['gain']['point']:+.3f} | {r['either']['point']:+.3f} | "
                + " / ".join(f"{v:+.3f}" for v in pp.values()) + " |")


for f in sorted(R.glob("bs_*.json")):
    d = json.loads(f.read_text())
    for k, v in d.items():
        if isinstance(v, dict) and "bar" in v:
            add(f.stem, k, v)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict) and "bar" in v2:
                    add(f.stem, f"{k}/{k2}", v2)
hdr = ("| script | row | fused R@1 | counterpart R@1 | bar comparator | bar margin [95%] | gain stat | either | "
       "per pair e×s / e×g / s×g |\n|---|---|---|---|---|---|---|---|---|")
(R / "bs_summary.md").write_text("Exploratory, seed 42, decides nothing.\n\n" + hdr + "\n" + "\n".join(rows) + "\n")
print(hdr)
print("\n".join(rows))
