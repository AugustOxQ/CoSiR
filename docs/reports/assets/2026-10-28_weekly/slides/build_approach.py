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

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
# Source: weekly draft v2, lines 60–82 and 732–753. No recipe settings.
fig=plt.figure(figsize=(13.33,7.5),dpi=200)
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,13.33),ylim=(0,7.5));ax.axis("off")
def text(x,y,s,size=16,**kw):
    return ax.text(x,y,s,fontsize=size,ha=kw.pop("ha","center"),va="center",**kw)
def box(x,y,w,h,s,edge=GRID,fill="white",color=INK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=.09",ec=edge,fc=fill,lw=1.5))
    text(x+w/2,y+h/2,s,color=color)
def arrow(a,b,color=SECONDARY,dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=14,lw=1.5,color=color,linestyle="--" if dashed else "-",shrinkA=2,shrinkB=2))
def painting(x,y):
    ax.add_patch(Rectangle((x,y),.65,.65,fc="#f0efed",ec=GREY,lw=1.5))
    ax.add_patch(Circle((x+.47,y+.46),.08,fc=GREY,ec="none"))
    ax.add_patch(Polygon([(x+.05,y+.07),(x+.27,y+.38),(x+.44,y+.14),(x+.55,y+.25),(x+.60,y+.07)],fc=GREY,ec="none"))
text(.45,7.02,"Train on Qwen’s instructed views; read the default image once",24,ha="left",weight="bold")
text(.45,6.35,"Training (training paintings, no labels)",18,ha="left",weight="bold")
painting(.55,4.8);text(.88,4.48,"+ captions")
arrow((1.28,5.13),(1.83,5.13))
box(1.85,4.52,2.5,1.2,"Qwen3-VL-\nEmbedding\n(frozen)")
arrow((4.36,5.13),(4.72,5.13))
for y,s,c in [(5.55,"default",GREY),(5.0,"emotion instruction",GREEN),(4.45,"style instruction","#74b382"),(3.90,"genre instruction","#a2cdaa")]:
    box(4.78,y,2.55,.43,s,c,"#f0efed" if c==GREY else c)
# Default input enters above the graph path.
ax.plot([7.33,10.18,10.18],[5.77,5.77,5.48],color=GREY,lw=1.5)
arrow((10.18,5.48),(10.18,5.32),GREY)
text(8.65,6.01,"default image",color=SECONDARY)
text(8.35,5.48,"buddy graph\nper aspect")
for cy,c in [(5.0,GREEN),(4.55,"#74b382"),(4.10,"#a2cdaa")]:
    arrow((7.35,{5.0:5.215,4.55:4.665,4.10:4.115}[cy]),(7.88,cy),c)
    pts=[(8.05,cy-.12),(8.22,cy+.13),(8.44,cy+.04),(8.42,cy-.15)]
    for a,b in [(0,1),(1,2),(2,3),(3,0),(1,3)]:
        ax.plot([pts[a][0],pts[b][0]],[pts[a][1],pts[b][1]],color=c,lw=1.5)
    for xx,yy in pts:ax.add_patch(Circle((xx,yy),.045,fc=c,ec="white",lw=.5))
    ax.plot([8.52,8.83],[cy,cy],color=c,lw=1.2)
ax.plot([8.83,8.83],[4.10,5.0],color=GREEN,lw=1.5)
arrow((8.83,4.665),(9.24,4.665),GREEN)
box(9.25,4.05,3.55,1.25,"network: one trunk,\nthree heads",BLUE,BLUE,"white")
ax.plot([7.32,7.65,11],[3.90,3.50,3.50],color=GREEN,lw=1.5,ls="--")
arrow((11,3.5),(11,4.04),GREEN,True)
text(8.9,3.23,"teacher: instructed image",color=SECONDARY)
ax.plot([.45,12.85],[2.95,2.95],color=GRID,lw=1)
text(.45,2.59,"Test (one pass, no instruction, no caption)",18,ha="left",weight="bold")
painting(.55,1.34);arrow((1.28,1.66),(1.83,1.66))
box(1.85,1.25,2,.82,"default image",GREY,"#f0efed")
arrow((3.88,1.66),(4.38,1.66));box(4.4,1.2,1.7,.92,"network",BLUE,BLUE,"white")
arrow((6.12,1.66),(6.62,1.66))
for y,s in [(2.0,"emotion"),(1.55,"style"),(1.1,"genre")]:box(6.65,y,1.65,.34,s,BLUE,BLUE,"white")
arrow((8.35,1.66),(8.88,1.66))
box(8.9,1.12,3.9,1.1,"k-means per aspect,\nscored by AMI\nagainst labels")
text(.45,.43,"Labels only score the clusters; they never enter training.",16,ha="left",color=SECONDARY)
fig.savefig(Path(__file__).with_name("approach.png"),dpi=200);plt.close(fig)
