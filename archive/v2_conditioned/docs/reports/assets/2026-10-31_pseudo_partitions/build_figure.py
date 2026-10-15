"""AMI heatmap (3 partitions x 3 labels) and shuffled-label null for E2. Run from the repo root."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402

RES = ROOT / "src/test/20261031_pseudo_partitions/results"
rec = json.loads((RES / "build_record.json").read_text())["ami"]
parts, labs = ["affect", "image", "caption"], ["emotion", "style", "genre"]
m = np.array([[rec[p][l] for l in labs] for p in parts])
fig, ax = plt.subplots(figsize=(5.6, 3.6))
im = ax.imshow(m, cmap="viridis", vmin=0, vmax=0.45)
ax.set_xticks(range(3), labs); ax.set_yticks(range(3), parts)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", color="w" if m[i, j] < 0.25 else "k")
ax.set_title("AMI of each k=64 partition with each label\n(scorer-train rows, labelled rows only)", fontsize=9)
fig.colorbar(im, ax=ax, label="AMI"); fig.tight_layout()
fig.savefig(Path(__file__).parent / "ami_heatmap.png", dpi=160)

data = load_artelingo(); st = artelingo_splits(data).scorer_train
labels = artelingo_aspect_labels(data); P = np.load(RES / "partitions.npz")
rng = np.random.default_rng(0)
out = {"seed": 0, "shuffles": 5, "null": {}, "catch_all_kept": {}}
for p, l in (("affect", "emotion"), ("image", "genre"), ("caption", "genre")):
    y = labels[l][st]; ok = y >= 0
    null = [float(ami(rng.permutation(y[ok]), P[p][ok])) for _ in range(5)]
    out["null"][f"{p}_vs_{l}"] = {"values": null, "mean": float(np.mean(null))}
    print(f"null {p} vs {l}: mean {np.mean(null):.5f} min {min(null):.5f} max {max(null):.5f}")
from src.data.artelingo_splits import encode_labels  # noqa: E402
em_all = encode_labels(data.emotions)[0][st]          # catch-all kept as its own class
for p in parts:
    out["catch_all_kept"][f"{p}_vs_emotion"] = float(ami(em_all, P[p]))
print(out["catch_all_kept"])
(Path(__file__).parent / "ami_null.json").write_text(json.dumps(out, indent=2))
