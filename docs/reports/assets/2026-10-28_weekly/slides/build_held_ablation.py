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

# Source: weekly draft v2, Table 17, held columns, lines 883–890.
rows=["Instructed graphs", "Buddy terms", "Content correction", "Relational term", "Sharper teacher row", "Direct teacher", "Caption endpoints", "Shared trunk"]
data=[
[(-.0327,-.0361,-.0292),(-.0277,-.0305,-.0250),(-.0090,-.0119,-.0062),(.0009,-.0013,.0032),(.0016,-.0015,.0046),(.0035,.0008,.0056),(.0009,-.0024,.0044),(-.0001,-.0025,.0027)],
[(-.1903,-.1959,-.1822),(-.1512,-.1547,-.1451),(-.0752,-.0814,-.0679),(0,-.0054,.0048),(.0008,-.0030,.0048),(.0041,.0016,.0067),(.0072,.0015,.0121),(.0023,-.0022,.0069)],
[(.0054,-.0014,.0132),(-.0015,-.0059,.0027),(.0072,.0030,.0113),(-.0440,-.0512,-.0372),(-.0038,-.0074,-.0002),(-.0031,-.0080,.0014),(-.0034,-.0119,.0045),(-.0001,-.0057,.0059)]]
fig, axes=plt.subplots(1,3, figsize=(13.33,7.5), dpi=200, sharey=True)
fig.subplots_adjust(left=.255,right=.96,bottom=.21,top=.65,wspace=.22)
fig.text(.035,.95,"Graphs and buddy terms carry emotion and style;\nthe relational term carries genre", fontsize=24, weight="bold",va="top")
fig.text(.035,.79,"Held paintings: each part removed, minus the sweep’s pick;\n12,281 paintings (genre 9,949), 5 seeds; nothing retuned",fontsize=16,color=SECONDARY,linespacing=1.4)
for ax, title, vals, limits, ticks in zip(axes,["Emotion","Style","Genre"],data,[(-.11,.075),(-.56,.37),(-.14,.105)],[[-.1,0,.05],[-.4,-.2,0,.2],[-.05,0,.05]]):
    ax.set_title(title,fontsize=20,pad=30,weight="bold")
    ax.axvline(0,color=SECONDARY,lw=1)
    ax.text(0,1.025,"the pick",transform=ax.get_xaxis_transform(),ha="center",fontsize=16,color=SECONDARY)
    for y,(v,lo,hi) in enumerate(vals):
        resolved=lo>0 or hi<0
        color=BLUE if resolved else GREY
        ax.barh(y,v,height=.48,color=color)
        ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt="none",ecolor=INK,elinewidth=1,capsize=3)
        if resolved:
            # Each label shares its bar’s row and clears the outer whisker.
            ax.annotate(f"{v:+.4f}".replace("-","−"),
                        xy=(lo if v < 0 else hi,y),
                        xytext=(-6 if v < 0 else 6,0),
                        textcoords="offset points", fontsize=16,
                        ha="right" if v < 0 else "left", va="center", color=INK)
    ax.set(xlim=limits,ylim=(7.7,-.85),xticks=ticks,yticks=range(8))
    if title == "Genre":
        ax.set_xticklabels(["−0.05", "0", "+0.05"])
    ax.grid(axis="x",color=GRID);ax.set_axisbelow(True)
    ax.spines[["top","right","left"]].set_visible(False);ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0,pad=10)
axes[0].set_yticklabels(rows,fontsize=16)
fig.text(.61,.12,"Change in AMI after removal",ha="center",fontsize=17)
fig.text(.035,.055,"Blue: interval excludes zero; grey: interval includes zero.\nRemoval effects inside this recipe, not the path that built it.",fontsize=16,color=SECONDARY,linespacing=1.4)
fig.savefig(Path(__file__).with_name("held_ablation.png"),dpi=200)
plt.close(fig)
