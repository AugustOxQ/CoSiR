"""Figure A: dollar weight per round, stacked by role (analysis notes, role table)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
roles = [  # name, R4, R5, R6 (role-table dollars, 1h write price for all), colour
 ("Main chat (coordinator)", 139, 111, 167, "#8a8984"),
 ("Decide (facts, literature, rule)", 9, 27, 39, "#d9a21b"),
 ("Code mapping", 1, 5, 10, "#8fc1e3"),
 ("Implementers", 39, 80, 293, "#2a78d6"),
 ("Ticket/task reviews", 16, 20, 74, "#7b5ea7"),
 ("Final review, fix wave, re-review", 39, 41, 107, "#eb6834"),
 ("Independent re-derivation", 22, 22, 13, "#1baf7a"),
 ("Reports (writer, user-read, checks)", 31, 35, 3, "#c4576b"),
]
totals = ["$295", "$341", "$706"]
labels = ["Round 4\n(2 experiments)", "Round 5", "Round 6\n(claim test)"]
fig, ax = plt.subplots(figsize=(9, 5.6))
bottoms = [0, 0, 0]
for name, a, b, c, col in roles:
    vals = [a, b, c]
    ax.bar(labels, vals, bottom=bottoms, color=col, label=name, width=0.55, edgecolor="white", linewidth=0.6)
    for i, v in enumerate(vals):
        if v >= 30:
            ax.text(i, bottoms[i] + v / 2, f"${v}", ha="center", va="center", fontsize=8.5, color="white")
    bottoms = [x + y for x, y in zip(bottoms, vals)]
for i, t in enumerate(totals):
    ax.text(i, bottoms[i] + 10, f"{t} total", ha="center", fontsize=9, color="#0b0b0b")
ax.set_ylabel("Dollar weight (API-price equivalent, USD)")
ax.set_ylim(0, 790)
ax.set_title("Round 6 cost 2x round 5, mostly in implementers and reviews", fontsize=12, color="#0b0b0b", loc="left")
ax.grid(axis="y", color="#e4e3df"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="upper left", fontsize=8, frameon=False)
fig.text(0.01, 0.01, "Dollar weights are API-price equivalents, not what the Max plan charges.", fontsize=7.5, color="#52514e")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(__file__.replace("fig_dollar_by_role.py", "fig_dollar_by_role.png"), dpi=150)
