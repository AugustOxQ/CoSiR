"""Figure 5. Values transcribed from its spec and Tables 7–13 of the draft.
Only teacher d/lo/hi fields are selected from the two specified JSON sources.
No estimates, resampling, or other result calculations are performed.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
SOURCE = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/stage3')
BLUE, PHOTO, INK, GREY, ORANGE, GRID = '#1d6fd1','#0b3d91','#0b0b0b','#a6a49e','#eb6834','#e4e3df'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK})
# Each tuple is the reported point, lower bound, upper bound, in metric points.
left = [
 [(0.1066-0.0708,None,None),(.0208,.0198,.0217)],
 [(.0477,.0439,.0514),(.0249,.0238,.0260)],
 [(.0399,.0363,.0434),(.0236,.0225,.0247)],
 [(.0377,.0343,.0411),(.0192,.0181,.0203)],
 [(.0201,.0178,.0227),(.0075,.0070,.0082)],
 [(.0276,.0252,.0301),(.0120,.0114,.0125)],
 [(.0249,.0226,.0274),(.0108,.0103,.0113)],
 [(.0204,.0182,.0227),(.0098,.0093,.0103)],
 [(.0797,.0741,.0857),(.0720,.0699,.0742)],
 [(.0782,.0724,.0845),(.0711,.0673,.0749)],
 [(.0728,.0668,.0787),(.0688,.0652,.0722)],
 [(.0815,.0746,.0880),(.0640,.0605,.0671)],
 [(.0653,.0589,.0719),(.0975,.0948,.1004)],
 [(.0343,.0298,.0387),(.0304,.0293,.0315)],
]
right = [
 [(.0145,.0113,.0177),(.0007,-.0002,.0016)],
 [None,(.0008,-.0001,.0017)], [None,(.0004,-.0005,.0013)], [None,(-.0013,-.0023,-.0003)],
 [(.0043,.0023,.0063),(-.0010,-.0015,-.0004)],
 [(.0043,.0021,.0065),(-.0005,-.0010,.0001)],
 [(.0054,.0033,.0076),(-.0000,-.0005,.0004)],
 [(.0024,.0004,.0045),(-.0011,-.0015,-.0006)],
 [(.0172,.0121,.0225),(.0227,.0210,.0244)],
 [None,None],[None,None],[None,None],
 [(.0181,.0121,.0242),(.0320,.0295,.0347)],
 [(-.0282,-.0333,-.0233),(-.0190,-.0205,-.0176)],
]
missing=[]
for filename, start, tasks in [('cells_artelingo_clu.json',1,['clu']),('cells_affection.json',9,['clu','ret'])]:
    records=json.loads((SOURCE/filename).read_text())
    for j, encoder in enumerate(['siglip2','siglip1','clip_l14']):
        for k, task in enumerate(tasks):
            matches=[r for r in records if r.get('encoder')==encoder and r.get('aspect')=='emotion' and r.get('task')==task and r.get('comparator','').startswith('teacher')]
            vals=tuple(matches[0].get(key) for key in ('d','lo','hi')) if len(matches)==1 else None
            if vals is None or any(v is None for v in vals):
                missing.append(f'{filename}: {encoder}, {task} teacher difference')
            else:
                right[start+j][k]=vals

fig=plt.figure(figsize=(16,15),dpi=100,facecolor='white')
fig.text(.035,.966,'Emotion gains over the plain image stay positive;\nagainst the teacher, retrieval on paintings sits near zero',fontsize=24,weight='bold',va='top')
fig.text(.035,.89,'Emotion only, the weakest aspect. Five seeds, 95% intervals.\nPainting model on paintings; photo model on photos, except the marked row.',fontsize=16,va='top',color='#52514e')
axes=[fig.add_axes([.255,.15,.325,.625]),fig.add_axes([.655,.15,.305,.625])]
ys=[1,2,3,4,6,7,8,9,11,12,13,14,16,18]
groups=[('ArtELingo held',0),('MMArt test',5),('Affection test · photo model',10),('EmoSet test · photo model',15),('Affection · frozen\npainting model',17)]
for ax,title,limits in zip(axes,["Gain over the encoder’s\nown plain image",'Difference from the teacher\n(instructed Qwen image)'],[(-.004,.107),(-.037,.037)]):
    ax.set_title(title,fontsize=18,loc='left',pad=24)
    ax.set_ylim(19, -.5); ax.set_xlim(*limits)
    ax.axvline(0,color=INK,lw=1.3)
    ax.set_yticks([]); ax.set_xlabel('Difference (AMI or mAP points)',labelpad=15,fontsize=16)
    ax.tick_params(axis='x',labelsize=16)
    ax.xaxis.grid(True,color=GRID,zorder=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    for y in [4.7,9.7,14.7,16.7]: ax.axhline(y,color=GRID,lw=1)
axes[0].set_xticks([0,.025,.05,.075,.1],['0','.025','.050','.075','.100'])
axes[1].set_xticks([-.03,-.015,0,.015,.03],['−.030','−.015','0','+.015','+.030'])
for title,y in groups:
    fig.text(.035, .15+.625*(19-y)/19.5,title,fontsize=16,weight='bold',va='center')
for i,y in enumerate(ys):
    enc=['Qwen','SigLIP 2','SigLIP 1','CLIP L/14'][i%4] if i<12 else ('Qwen' if i==12 else 'Qwen · failed test')
    fig.text(.05,.15+.625*(19-y)/19.5,enc,fontsize=16,va='center',color='#52514e')
    color=PHOTO if 8<=i<=12 else BLUE
    for panel, data in enumerate([left,right]):
        for metric,(marker,offset) in enumerate([('o',-.17),('s',.17)]):
            value=data[i][metric]
            if value is None: continue
            d,lo,hi=value; yy=y+offset
            if lo is not None:
                axes[panel].plot([lo,hi],[yy,yy],color=color,lw=2.6,zorder=2)
                axes[panel].vlines([lo,hi],yy-.075,yy+.075,color=color,lw=1.3)
            axes[panel].plot(d,yy,marker=marker,color=color,ms=8,zorder=4)
            state=('below' if i==13 or (metric==1 and i in [3,4,7]) else 'level' if metric==1 and i in [0,1,2,5,6] else None) if panel else None
            if state:
                axes[panel].plot(d,yy,marker='o',mfc='none',mec=ORANGE if state=='below' else GREY,mew=2.4,ms=16,zorder=5)
                axes[panel].text(.035,yy+(-.16 if metric==0 else .16) if i==13 else yy,state,ha='right',va='center',fontsize=16,color='#52514e')
handles=[Line2D([],[],marker='o',color=INK,lw=0,label='AMI'),Line2D([],[],marker='s',color=INK,lw=0,label='mAP'),Line2D([],[],color=BLUE,lw=3,label='Painting model'),Line2D([],[],color=PHOTO,lw=3,label='Photo model')]
fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.03,.066),ncol=4,frameon=False,fontsize=16)
fig.text(.035,.035,'ArtELingo Qwen AMI vs plain image: point only; no interval is available in the sources.',fontsize=16,color='#52514e')
fig.savefig(OUT/'results_overview.png')
print('results_overview.png; missing teacher values:',missing or 'none')
