"""Figure 10 override: read exactly the draft coverage table and section F states.
Line wrapping changes presentation only; cell wording is retained verbatim.
"""
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[4]
DRAFT=ROOT/'.scratch/stage_pages/2026-11-04_datasets_and_retrieval.draft.md'
CHANGES=ROOT/'.scratch/stage_pages/2026-11-04_datasets_and_retrieval.changes.md'
def table(path,heading):
    section=path.read_text().split(heading,1)[1]
    lines=[line for line in section.split('\n\n') if line.startswith('|')][0].splitlines()
    return [[cell.strip() for cell in line.strip('|').split('|')] for line in lines if not line.startswith('|---')]
data=table(DRAFT,'### What we have and what is missing')
states=table(CHANGES,'## F. For the HTML builder: coverage table cell states')
assert [r[0] for r in data[1:]]==[r[0] for r in states[1:]]
assert data[0][1:]==states[0][1:]
# Weekly only: the new-scores round finished, so its cells are scored. The stage page keeps its own 08:30 table.
OVERRIDES={('PercepT on the same encoder, retrieval',1):('4 encoders','scored'),
           ('PercepT on the same encoder, retrieval',2):('4 encoders','scored'),
           ('CRL, clustering',1):('CLIP L/14','scored'),
           ('CRL, clustering',2):('CLIP L/14','scored'),
           ('CRL, retrieval',1):('CLIP L/14; Qwen, score only','scored'),
           ('CRL, retrieval',2):('CLIP L/14','scored')}
for r,s in zip(data[1:],states[1:]):
    for (name,col),(val,st) in OVERRIDES.items():
        if r[0]==name: r[col],s[col]=val,st
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'text.color':'#0b0b0b'})
fig=plt.figure(figsize=(24,13.5),dpi=160,facecolor='white')
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,2400); ax.set_ylim(1350,0); ax.axis('off')
ax.text(35,42,'What we have and what is missing',fontsize=30,weight='bold',va='top')
colors={'scored':'#b9d3f2','running':'#fbe3a6','none':'#eeede9'}
for i,state in enumerate(['scored','running','mixed','none']):
    x=35+i*340
    ax.add_patch(Rectangle((x,115),32,32,facecolor=colors.get(state,colors['scored'])))
    if state=='mixed': ax.add_patch(Polygon([(x,115),(x+32,115),(x+32,147)],color=colors['running']))
    ax.text(x+45,131,{'running':'in progress','mixed':'both'}.get(state,state),va='center',fontsize=20)
startx,cw,top,rh=490,375,265,96
for j,label in enumerate(data[0][1:]):
    ax.text(startx+j*cw+cw/2,220,label.replace(' (','\n('),ha='center',va='center',fontsize=20,weight='bold')
for i,(row,srow) in enumerate(zip(data[1:],states[1:])):
    y=top+i*rh
    ax.text(35,y+rh/2,textwrap.fill(row[0],26),va='center',fontsize=20,weight='bold')
    for j,(value,state) in enumerate(zip(row[1:],srow[1:])):
        x=startx+j*cw
        ax.add_patch(Rectangle((x+2,y+2),cw-4,rh-4,facecolor=colors.get(state,colors['scored'])))
        if state=='mixed':
            ax.add_patch(Polygon([(x+2,y+2),(x+cw-2,y+2),(x+cw-2,y+rh-2)],facecolor=colors['running']))
        if state!='none':
            artist=ax.text(x+cw/2,y+rh/2,textwrap.fill(value,23),ha='center',va='center',fontsize=20,color='#0b0b0b',linespacing=1.05)
ax.text(35,1295,'Three encoders: DINOv2, CLIP B/32, CLIP B/16. Four encoders: Qwen, SigLIP 2, SigLIP 1, CLIP L/14.',fontsize=20,color='#52514e',va='center')
fig.savefig(OUT/'coverage.png')
print('coverage.png: 10 rows × 5 columns, matching source labels and states')
