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

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
# Current model line only. Weekly draft: experiments, coverage, and next steps.
fig=plt.figure(figsize=(13.33,7.5),dpi=200)
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,13.33),ylim=(0,7.5));ax.axis("off")
def text(x,y,s,size=18,**kw):
    return ax.text(x,y,s,fontsize=size,ha=kw.pop("ha","center"),va="center",**kw)
yaxis=3.5
text(.45,7.0,"One image read: more datasets, retrieval and coverage",24,ha="left",weight="bold")
text(.45,6.50,"Current painting model and its photo fine-tune",18,ha="left",color=SECONDARY)
ax.add_patch(Rectangle((2.9,1.0),9.95,5.0,fc=BLUE,alpha=.12,ec="none"))
text(7.875,5.72,"THIS WEEK",18,color=BLUE,weight="bold")
text(1.55,5.72,"LAST WEEK",16,color=SECONDARY,weight="bold")
ax.plot([.55,12.8],[yaxis,yaxis],color=SECONDARY,lw=1.5)
# The weekly steps are ordered milestones, not invented daily timestamps.
events=[
(1.55,4.65,"Recipe sweep:\nheld test and ablation",BLUE,"o"),
(3.95,2.35,"Retrieval\non paintings",BLUE,"o"),
(5.95,4.55,"Photo model\nAffection + EmoSet",BLUE,"o"),
(7.90,2.35,"Three more\nencoders",BLUE,"o"),
(9.85,4.55,"MMArt\non four encoders",BLUE,"o"),
(11.65,2.35,"Coverage check, new scores\nTwo rounds in progress",GREY,"o")]
for x,ty,label,c,m in events:
    endy=ty-.65 if ty>yaxis else ty+.50
    ax.plot([x,x],[yaxis,endy],color=c,lw=1.1,zorder=2)
    ax.plot(x,yaxis,marker=m,ms=9,color=c,zorder=3)
    text(x,ty,label,size=17)
text(7.875,1.25,"Rounds in progress have no results yet",16,color=SECONDARY)
handles=[Line2D([],[],marker="o",color=c,ls="none",ms=9,label=s)
         for s,c in [("completed step",BLUE),("coverage check",GREY)]]
ax.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.025),ncol=2,frameon=False,fontsize=16,handletextpad=.5,columnspacing=2)
fig.savefig(Path(__file__).with_name("progress_timeline.png"),dpi=200);plt.close(fig)
