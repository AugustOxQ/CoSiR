"""Figure 9: literal outcome cells from the Figure 9 specification."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'text.color':'#0b0b0b'})
fig=plt.figure(figsize=(16,11),dpi=100,facecolor='white')
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1600); ax.set_ylim(1100,0); ax.axis('off')
def text(x,y,s,**kwargs): return ax.text(x,y,s,va='center',fontsize=16,**kwargs)
ax.text(35,45,'Against the teacher, clustering is mostly clearly above;\nemotion retrieval is the weak column',fontsize=24,weight='bold',va='top')
text(35,150,'Each cell is one comparison with the instructed Qwen image, five seeds.',color='#52514e')
ax.text(35,188,'Key: above = clearly above (whole 95% interval above zero); level = not worse than one seed range; below = worse than that.',color='#52514e',fontsize=15,va='center')
A=[['win','win','level','level','level','win'],['win','win','level','level','win','win'],['win','win','level','level','win','win'],['win','win','level','below','win','win']]
B=[['win','win','win','below','level','win'],['win','win','win','level','win','win'],['win','win','win','level','win','win'],['win','win','level','below','win','win']]
colors={'win':'#1d6fd1','level':'#e4e3df','below':'#eb6834'}
WORD={'win':'above','level':'level','below':'below'}
for label,x,labelx,data in [('ArtELingo held',145,30,A),('MMArt test',705,590,B)]:
    text(labelx,235,label,weight='bold')
    text(x+102,290,'AMI',ha='center',weight='bold'); text(x+320,290,'mAP',ha='center',weight='bold')
    for j,aspect in enumerate(['Emotion','Style','Genre']*2):
        xx=x+j*68+(14 if j>=3 else 0)
        # Full aspect names are turned to keep the square cells and large type.
        text(xx+34,365,aspect,ha='center',rotation=60)
    for i,row in enumerate(data):
        y=425+i*96
        text(labelx,y+34,['Qwen','SigLIP 2','SigLIP 1','CLIP L/14'][i])
        for j,state in enumerate(row):
            xx=x+j*68+(14 if j>=3 else 0)
            ax.add_patch(Rectangle((xx,y),66,68,facecolor=colors[state]))
            ax.text(xx+33,y+34,WORD[state],va='center',fontsize=13,ha='center',color='#0b0b0b' if state=='level' else 'white')
text(1165,235,'Photos, emotion',weight='bold')
for j,metric in enumerate(['AMI','mAP']): text(1440+j*72+34,365,metric,ha='center',weight='bold')
labels=['Affection · photo\nQwen','Affection · photo\nSigLIP 2','Affection · photo\nSigLIP 1','Affection · photo\nCLIP L/14','EmoSet · photo\nQwen','Affection · frozen\npainting model\nQwen · failed test']
for i,label in enumerate(labels):
    y=425+i*85
    text(1165,y+34,label)
    for j in range(2):
        state='below' if i==5 else 'win'; xx=1440+j*72
        ax.add_patch(Rectangle((xx,y),68,68,facecolor=colors[state]))
        ax.text(xx+34,y+34,WORD[state],va='center',fontsize=13,ha='center',color='white')
text(35,1020,'Counts on MMArt: 18 above, 4 level, 2 below. The level margin is each cell’s own validation seed range;',color='#52514e')
text(35,1055,'the Qwen ArtELingo clustering cells used last week’s rule.',color='#52514e')
fig.savefig(OUT/'teacher_cells.png')
print('teacher_cells.png')
