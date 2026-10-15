"""Round 7 facts: style x genre label statistics on the scorer-train rows (the rows of the earlier AMIs; no held row is read).
Run from /project/CoSiR:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261126_design_l_style_source/r7_label_stats.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami, normalized_mutual_info_score as nmi

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.artelingo import load_artelingo
from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits, encode_labels
from src.data.wikiart_genre import GENRE_NAMES

data = load_artelingo()
sp = artelingo_splits(data)
st = np.asarray(sp.scorer_train)
lab = artelingo_aspect_labels(data)
_, style_names = encode_labels(data.art_styles)
style, genre = lab["style"][st], lab["genre"][st]
paint = np.asarray(sp.groups)[st]
print("scorer-train rows", len(st), "paintings (leakage groups)", len(np.unique(paint)))
print("rows with style", int((style >= 0).sum()), "with genre", int((genre >= 0).sum()),
      "with both", int(((style >= 0) & (genre >= 0)).sum()))
print("style classes", len(style_names), "genre classes", len(GENRE_NAMES))
m = (style >= 0) & (genre >= 0)
for nm, mk in (("rows with both", m),):
    print(f"AMI(style, genre) on {nm}: {ami(style[mk], genre[mk]):.4f}   NMI: {nmi(style[mk], genre[mk]):.4f}")
# AMI on the same row sets as the earlier AMIs: style on all rows, genre on genre-labelled rows (two different sets)
# painting level: first row of each painting among rows with both
_, first = np.unique(paint[m], return_index=True)
sp_, gp_ = style[m][first], genre[m][first]
print(f"painting-level (one row per painting, both labels, n={len(first)}): AMI {ami(sp_, gp_):.4f} NMI {nmi(sp_, gp_):.4f}")


def table(s, g, unit):
    print(f"\n[{unit}] style counts and largest-genre share, 10 most frequent styles")
    cnt = Counter(s.tolist())
    top = [k for k, _ in cnt.most_common(10)]
    print("style | n | largest genre | share of styles' labelled items")
    for k in top:
        gg = Counter(g[s == k].tolist())
        gk, n = gg.most_common(1)[0]
        print(f"{style_names[k]} | {cnt[k]} | {GENRE_NAMES[gk]} | {n / sum(gg.values()):.3f}")
    ok = 0
    for k in cnt:
        gg = Counter(g[s == k].tolist())
        ok += sum(1 for v in gg.values() if v >= 50) >= 3
    print(f"styles (of {len(cnt)}) with >=3 genres of >=50 items: {ok}")
    return cnt


table(style[m], genre[m], "rows with both labels")
table(sp_, gp_, "paintings with both labels")
print("\nstyle class sizes (rows, scorer-train):", sorted(Counter(style[style >= 0].tolist()).values(), reverse=True))
print("style names:", style_names)
print("genre counts (rows):", {GENRE_NAMES[k]: v for k, v in sorted(Counter(genre[genre >= 0].tolist()).items())})
