from pathlib import Path
import os
os.environ["MPLCONFIGDIR"] = str(Path(__file__).parent / ".mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREY, BLUE, GREEN = "#a6a49e", "#1d6fd1", "#4a9c5b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": SECONDARY,
    "ytick.color": INK, "figure.facecolor": "white", "savefig.facecolor": "white"})

# Source: weekly draft v2, Table 4, lines 283–287.
fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=200)
fig.subplots_adjust(left=.10, right=.96, bottom=.17, top=.70)
fig.text(.055, .94, "One instruction per aspect raises own-aspect AMI\nwhere the aspect is visible", fontsize=24, weight="bold", va="top")
fig.text(.055, .80, "Qwen3-VL-Embedding, k-means; 99,999 captions and 5,000 development paintings", fontsize=16, color=SECONDARY)
x = [0, 1.25, 2.5]
for offset, values, color, label in [(-.19, [.122,.222,.069], GREY, "Default prompt"), (.19,[.278,.395,.093], GREEN,"Aspect instruction")]:
    bars=ax.bar([v+offset for v in x], values, width=.32, color=color, label=label)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=7, fontsize=18)
ax.set(ylim=(0,.45), ylabel="AMI (0 = chance)", xticks=x, xticklabels=["Captions, emotion", "Images, style", "Images, emotion"])
ax.set_yticks([0,.1,.2,.3,.4])
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.spines[["top","right","left"]].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="both", length=0, pad=12)
ax.legend(loc="lower left", bbox_to_anchor=(0,1.02), ncol=2, frameon=False)
fig.savefig(Path(__file__).with_name("instruction_effect.png"), dpi=200)
plt.close(fig)
