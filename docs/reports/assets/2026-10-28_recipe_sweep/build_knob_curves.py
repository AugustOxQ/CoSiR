"""Rebuild the thirteen single-setting curves after checking the published tables."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from source_checks import stage1, style, quiet, NAMES, BLUE, OLD_BLUE, SECONDARY, GRID


def main():
    d=stage1(); style()
    order=sorted(d['curves'],key=lambda k:max(p['M'] for p in d['curves'][k]),reverse=True)
    fig,ax=plt.subplots(figsize=(10,7.5),dpi=200)
    fig.subplots_adjust(left=.335,right=.965,top=.765,bottom=.12)
    fig.text(.035,.975,'One setting at a time: only 40 graph neighbours\ngives a large gain over the previous recipe',fontsize=20,va='top',linespacing=1.12)
    fig.text(.035,.845,'Single-seed sweep · no aspect may lose more than half a seed range',fontsize=12,color=SECONDARY)
    fig.legend(handles=[Line2D([],[],ls='none',marker='o',color=BLUE,label='passes guard',markersize=6),
        Line2D([],[],ls='none',marker='o',markerfacecolor='white',markeredgecolor=BLUE,label='fails guard',markersize=6),
        Patch(facecolor='#edf4fc',edgecolor='none',label='to the joint grid')],
        loc='center left',bbox_to_anchor=(.025,.802),frameon=False,ncol=3,fontsize=12,columnspacing=1.5)
    ax.axvline(0,color=OLD_BLUE,lw=1.5,zorder=1)
    ax.text(0,1.015,'previous recipe G',transform=ax.get_xaxis_transform(),ha='center',fontsize=12,color=SECONDARY)
    for i,k in enumerate(order):
        if k in d['grid_knobs']: ax.axhspan(i-.47,i+.47,color='#edf4fc',zorder=0)
        ax.axhline(i,color=GRID,lw=.5,zorder=0)
        points=sorted(d['curves'][k],key=lambda p:p['M'])
        # Only close clusters need vertical separation. Keep points at their exact gains.
        close=any(b['M']-a['M']<.23 for a,b in zip(points,points[1:]))
        for j,p in enumerate(points):
            y=i+((j-1)*.25 if close else 0)
            ax.plot(p['M'],y,'o',ms=6,mec=BLUE,mfc=BLUE if p['guard'] else 'white',mew=1.2,zorder=3)
            dx,dy,ha=7,0,'left'
            if close:
                dx,dy,ha=([(-7,5,'right'),(7,0,'left'),(7,-3,'left')][j])
            if k=='k4':
                dx,dy,ha=[(-7,4,'right'),(7,-4,'left'),(7,-4,'left')][j]
            if k=='k13':
                dx,dy,ha=[(-36,0,'right'),(12,0,'left'),(48,0,'left')][j]
            ax.annotate(f"{p['value']:g}",(p['M'],y),xytext=(dx,dy),textcoords='offset points',
                        va='center',ha=ha,fontsize=12,color=BLUE,
                        arrowprops=dict(arrowstyle='-',color=BLUE,lw=.6) if k=='k13' else None)
    ax.set(xlim=(-1.32,1.12),ylim=(12.6,-.6),xticks=[-1.2,-.8,-.4,0,.4,.8])
    ax.set_xticklabels(['−1.2','−0.8','−0.4','0','+0.4','+0.8'],fontsize=12)
    ax.set_yticks(range(13)); ax.set_yticklabels([NAMES[k] for k in order],fontsize=12.5)
    for label,k in zip(ax.get_yticklabels(),order):
        if k in d['grid_knobs']: label.set_weight('bold')
    ax.set_xlabel('mean gain over the previous recipe (seed ranges)',fontsize=12.5,labelpad=12)
    quiet(ax)
    fig.text(.035,.025,'Mean gain = average of the three AMI gains divided by their seed ranges; labels show tested values.',fontsize=11.5,color=SECONDARY)
    fig.savefig(Path(__file__).with_name('knob_curves.png'),dpi=200); plt.close(fig)

if __name__=='__main__': main()
