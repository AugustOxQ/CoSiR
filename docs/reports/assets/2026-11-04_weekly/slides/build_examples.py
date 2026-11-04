"""One test item per dataset for the weekly deck: MMArt, Affection, EmoSet.

The images belong to their datasets. Read the items from /data; labels come from the same files the tests use.
"""
from pathlib import Path
import textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

INK, SECONDARY = "#0b0b0b", "#52514e"
OUT = Path(__file__).with_name("examples_slide.png")

m = pd.read_parquet("/data/SSD/mmart/mmart_index.parquet")
row = m[m.key.str.contains("julius-leblanc-stewart_summer-s-promenade-1880", regex=False)].iloc[0]
assert row.split == "new"
a = pd.read_csv("/data/SSD/affection/language/version_0/affection_preprocessed_public_version_0.csv",
                usecols=["image_name", "utterance", "emotion", "split"])
a = a[a.image_name == "awe_0292.jpg"]
assert set(a.split) == {"test"}
votes = a.emotion.value_counts()
assert votes.idxmax() == "awe"
quote = a[a.utterance.str.contains("stars spinning")].utterance.iloc[0].strip()

items = [
    ("MMArt (painting)", row.image_path,
     [f"Emotion: {row.emotion}", f"Style: {row['style']}", f"Genre: {row.genre_name}"]),
    ("Affection (photo)", "/data/SSD/affection/fi_t4sa/dataset/benchmark/FI/images/awe/awe_0292.jpg",
     [f"Emotion: awe ({votes['awe']} of {votes.sum()} votes)"] + textwrap.wrap(f"Explanation: \"{quote}\"", 38)),
    ("EmoSet (photo)", "/data/SSD/emoset/image/amusement/amusement_11233.jpg", ["Emotion: amusement"]),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
W, H = 12.9, 6.0
fig = plt.figure(figsize=(W, H), dpi=200, facecolor="white")
cw = W / 3
for i, (name, path, lines) in enumerate(items):
    x0 = i * cw
    fig.text((x0 + cw / 2) / W, 5.72 / H, name, ha="center", va="center", fontsize=20, weight="bold")
    im = Image.open(path).convert("RGB")
    bw, bh = cw - .4, 3.55
    asp = im.width / im.height
    iw, ih = (bw, bw / asp) if bw / asp <= bh else (bh * asp, bh)
    ax = fig.add_axes([(x0 + (cw - iw) / 2) / W, (1.72 + (bh - ih) / 2) / H, iw / W, ih / H])
    ax.imshow(im)
    ax.axis("off")
    y = 1.46
    for k, line in enumerate(lines):
        bold = k == 0
        fig.text((x0 + cw / 2) / W, y / H, line, ha="center", va="center", fontsize=16 if bold else 15,
                 weight="bold" if bold else "normal", color=INK if bold else SECONDARY)
        y -= .33
fig.savefig(OUT, dpi=200)
print("wrote", OUT, "votes", votes.to_dict())
