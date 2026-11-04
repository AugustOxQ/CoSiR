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
ROOT=OUT.parents[3]
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
fig=plt.figure(figsize=(16,19),dpi=100,facecolor='white')
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1600); ax.set_ylim(1900,0); ax.axis('off')
ax.text(35,42,'Our own comparisons are in place;\ncurrent rounds add the outside comparisons',fontsize=24,weight='bold',va='top')
ax.text(35,155,'Status on 2026-11-04, with the finished round scored. In-progress cells have no results yet',fontsize=16,color='#52514e',va='center')
colors={'scored':'#b9d3f2','running':'#fbe3a6','none':'#eeede9'}
for i,state in enumerate(['scored','running','mixed','none']):
    x=35+i*285
    ax.add_patch(Rectangle((x,208),32,32,facecolor=colors.get(state,colors['scored'])))
    if state=='mixed': ax.add_patch(Polygon([(x,208),(x+32,208),(x+32,240)],color=colors['running']))
    ax.text(x+45,224,{'running':'in progress','mixed':'both'}.get(state,state),va='center',fontsize=16)
startx,cw,top,rh=390,234,340,140
for j,label in enumerate(data[0][1:]):
    ax.text(startx+j*cw+cw/2,292,label.replace(' (','\n('),ha='center',va='center',fontsize=16,weight='bold')
for i,(row,srow) in enumerate(zip(data[1:],states[1:])):
    y=top+i*rh
    ax.text(35,y+rh/2,textwrap.fill(row[0],24),va='center',fontsize=16,weight='bold')
    for j,(value,state) in enumerate(zip(row[1:],srow[1:])):
        x=startx+j*cw
        ax.add_patch(Rectangle((x+2,y+2),cw-4,rh-4,facecolor=colors.get(state,colors['scored'])))
        if state=='mixed':
            ax.add_patch(Polygon([(x+2,y+2),(x+cw-2,y+2),(x+cw-2,y+rh-2)],facecolor=colors['running']))
        if state!='none':
            artist=ax.text(x+cw/2,y+rh/2,textwrap.fill(value,18),ha='center',va='center',fontsize=16,color='#0b0b0b',linespacing=1.15)
ax.text(35,1820,'Three encoders: DINOv2, CLIP B/32, CLIP B/16. Four encoders: Qwen, SigLIP 2, SigLIP 1, CLIP L/14.',fontsize=16,color='#52514e',va='center')
fig.savefig(OUT/'coverage.png')
print('coverage.png: 10 rows × 5 columns, matching source labels and states')
