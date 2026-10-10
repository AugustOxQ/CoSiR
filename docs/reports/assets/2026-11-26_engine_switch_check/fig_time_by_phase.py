"""Figure B: wall-clock hours per phase for each round (analysis notes and rounds_facts.md section 1)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
h = lambda hh, mm=0: hh + mm / 60
rows = [  # label, decide, idle wait for user, build, run to verdict, verdict to report
 ("Round 4a\n(vetoes)", h(1, 11), 0, h(0, 54), h(0, 11), h(1, 3)),
 ("Round 4b\n(CLIP FT)", 0, 0, h(0, 52), h(0, 32), h(0, 53)),
 ("Round 5", h(2, 18), h(5, 57), h(2, 20), h(0, 53), h(1, 18)),
 ("Round 6\n(claim test)", h(2, 36), 0, h(18, 49), h(4, 13), h(4, 50)),
]
phases = [("Decide", "#d9a21b"), ("Idle, waiting for the user", "#e4e3df"), ("Build", "#2a78d6"),
          ("Run to verdict", "#1baf7a"), ("Verdict to report", "#eb6834")]
fig, ax = plt.subplots(figsize=(9.5, 5))
ys = list(range(len(rows)))[::-1]
for y, r in zip(ys, rows):
    left = 0
    for k, (pname, col) in enumerate(phases):
        v = r[k + 1]
        ax.barh(y, v, left=left, color=col, edgecolor="white", height=0.55,
                label=pname if r[0] == "Round 5" else None,
                hatch="//" if k == 1 else None)
        if v >= 1.6:
            ax.text(left + v / 2, y, f"{int(v)}h{int(round((v-int(v))*60)):02d}", ha="center", va="center", fontsize=8.5,
                    color="#0b0b0b" if k == 1 else "white")
        left += v
    ax.text(left + 0.25, y, f"{left:.1f} h", va="center", fontsize=9, color="#0b0b0b")
# round 6 marks (below the bar)
y6 = ys[-1]
bstart = h(2, 36)
ax.plot([bstart + 0.9, bstart + 6.9], [y6 - 0.33, y6 - 0.33], color="#52514e", lw=3)
ax.annotate("6 h review stall (permission prompt)", xy=(bstart + 3.9, y6 - 0.36), xytext=(bstart + 3.9, y6 - 0.75),
            ha="center", fontsize=8.5, arrowprops=dict(arrowstyle="-", color="#52514e"))
vstart = h(2, 36) + h(18, 49) + h(4, 13)
ax.plot([vstart + 0.8, vstart + 4.8], [y6 - 0.33, y6 - 0.33], color="#52514e", lw=3)
ax.annotate("of which about 4 h GPU jobs after the verdict\n(descriptive rows only)", xy=(vstart + 2.8, y6 - 0.36), xytext=(vstart + 0.5, y6 - 0.85),
            ha="center", fontsize=8.5, arrowprops=dict(arrowstyle="-", color="#52514e"))
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("Duration (hours)")
ax.set_xlim(0, 36); ax.set_ylim(-1.3, 3.5)
ax.set_title("Round 6's extra time is build time, including one 6-hour stall", fontsize=12, loc="left", color="#0b0b0b")
ax.grid(axis="x", color="#e4e3df"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="upper right", fontsize=8, frameon=False)
fig.text(0.01, 0.01, "Phase durations from the analysis notes; phases overlap slightly (round 5's run started before its build ended), so bars can exceed the whole-loop wall time (12h09).\n"
         "Round 4a and 4b ran in one chat. The stall and post-verdict marks show where inside the phase bars they fall.", fontsize=7.5, color="#52514e")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(__file__.replace("fig_time_by_phase.py", "fig_time_by_phase.png"), dpi=150)
