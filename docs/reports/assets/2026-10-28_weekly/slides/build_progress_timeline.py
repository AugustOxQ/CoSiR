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
# Sources: reports_sum.md weekly table; weekly draft v2 lines 167–197, 372–379.
fig=plt.figure(figsize=(13.33,7.5),dpi=200)
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,13.33),ylim=(0,7.5));ax.axis("off")
def text(x,y,s,size=16,**kw):
    return ax.text(x,y,s,fontsize=size,ha=kw.pop("ha","center"),va="center",**kw)
def octpos(d):
    # 15–28 October occupies 64.5% of the axis; the earlier section is compressed.
    if d < 15: return 3.15+(d-1)*(1.75/14)
    return 4.9+(d-15)*(7.9/13)
yaxis=3.35
text(.45,7.02,"From the buddy model to a condition-free model\nread once on the default image",24,ha="left",weight="bold")
x15,x28=octpos(15),octpos(28)
ax.add_patch(Rectangle((x15,.75),x28-x15,5.58,fc=BLUE,alpha=.12,ec="none"))
text((x15+x28)/2,6.12,"this report",color=BLUE)
# Date-labelled phase summaries occupy the compressed history region.
for y,h,label in [
    (6.11,.80,"Jun–early Sep\nConditional buddy model"),
    (5.15,.95,"Late Sep · PercepT topic\npipeline; buddy model\nagainst PercepT"),
    (4.00,1.20,"Late Sep–14 Oct\nConditioned line (CoSiR v2):\naspect episodes,\none-sided affect steering")]:
    ax.add_patch(Rectangle((.40,y-h/2),3.45,h,fc=GREY,ec="none"))
    text(2.125,y,label)
text(12.8,6.5,"2026",ha="right",color=SECONDARY)
ax.plot([.55,12.8],[yaxis,yaxis],color=SECONDARY,lw=1.5)
for xx in [1.90,2.02]:
    ax.plot([xx-.035,xx+.035],[yaxis-.10,yaxis+.10],color=INK,lw=1.8)
for x,label in [(.55,"Jun"),(1.40,"2 Sep"),(octpos(1),"1 Oct"),(x15,"15 Oct")]:
    ax.plot([x,x],[yaxis-.05,yaxis+.05],color=SECONDARY)
    text(x,3.03,label,color=SECONDARY)
ax.plot([x28,x28],[yaxis-.05,yaxis+.05],color=SECONDARY)
text(x28,3.63,"28 Oct",ha="right",color=SECONDARY)
# Each callout remains beside its own date. Alternating levels separate nearby dates.
events=[
(15,4.95,4.00,"15 Oct\nConditioned line\nstopped",INK,"D",False),
(17,6.10,2.35,"17 Oct\nOne Qwen\ninstruction\nper aspect",GREY,"o",False),
(17.5,6.42,5.05,"17–18 Oct\nImage-only\nstudents","#eb6834","o",False),
(19,7.60,2.00,"19 Oct\nFirst shared\nmodel",BLUE,"o",False),
(20,8.15,4.15,"20 Oct\nHeld emotion\nabove PercepT",BLUE,"o",True),
(25,9.72,2.35,"25 Oct · Recipe pick;\nencoders pass\nheld emotion",BLUE,"o",True),
(27,11.28,4.52,"27 Oct\nRelational genre term:\nheld genre level",BLUE,"o",True),
(27.7,11.95,2.10,"27–28 Oct\nRecipe sweep:\nheld test and\nablation",BLUE,"o",True)]
for d,tx,ty,label,c,m,held in events:
    x=octpos(d)
    halfheight=.62 if d in (17,27.7) else .49
    endy=ty-halfheight if ty>yaxis else ty+halfheight
    ax.plot([x,tx],[yaxis,endy],color=c,lw=1.1,zorder=2)
    ax.plot(x,yaxis,marker=m,ms=9,color=c,zorder=3)
    if held: ax.plot(x,yaxis-.18,marker="*",ms=11,color=INK,zorder=3)
    text(tx,ty,label)
handles=[Line2D([],[],marker=m,color=c,ls="none",ms=9,label=s) for s,c,m in [("worked",BLUE,"o"),("did not work","#eb6834","o"),("check",GREY,"o"),("decision",INK,"D"),("held data",INK,"*")]]
ax.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.015),ncol=5,frameon=False,fontsize=16,handletextpad=.5,columnspacing=1.4)
fig.savefig(Path(__file__).with_name("progress_timeline.png"),dpi=200);plt.close(fig)
