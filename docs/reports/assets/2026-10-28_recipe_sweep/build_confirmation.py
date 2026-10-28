"""Rebuild local confirmation; all per-aspect intervals exist in the JSON and log."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from source_checks import stage3, style, quiet, BLUE, OLD_BLUE, SECONDARY, GRID

LABELS=['40 neighbours', '40 neighbours + style relational 2 + trunk 512', '40 neighbours + teacher row 0.035']
COLORS=[BLUE,'#5292dc','#78a9e3']


def main():
    d=stage3(); style()
    fig=plt.figure(figsize=(12,5),dpi=200)
    fig.text(.035,.97,'Confirmation on three seeds: 40 graph neighbours\nhas the largest mean gain, mostly from style',fontsize=21,va='top',linespacing=1.12)
    fig.text(.035,.80,'Seeds 1 to 3 · 29,232 development paintings · paired 95% intervals',fontsize=13,color=SECONDARY)
    fig.legend(handles=[Line2D([],[],ls='none',marker='o',color=c,label=l,markersize=7) for c,l in zip(COLORS,LABELS)],
               loc='upper left',bbox_to_anchor=(.025,.775),ncol=1,frameon=False,fontsize=12.5,labelspacing=.5,handletextpad=.6)
    left=fig.add_axes([.12,.23,.40,.32]); right=fig.add_axes([.65,.23,.25,.32])
    for ax in (left,right):
        quiet(ax); ax.axvline(0,color=OLD_BLUE,lw=1.4,zorder=1)
        ax.text(0,1.03,'previous recipe G',transform=ax.get_xaxis_transform(),ha='center',fontsize=12,color=SECONDARY)
    for a,aspect in enumerate(('emotion','style','genre')):
        left.axhline(a,color=GRID,lw=.6,zorder=0)
        for i,run in enumerate(d['top']):
            e=d['results'][run]['effects'][aspect]; y=a+(i-1)*.22
            err=[[e['e']-e['lo']],[e['hi']-e['e']]] if 'lo' in e else None
            left.errorbar(e['e'],y,xerr=err,fmt='o',color=COLORS[i],ms=6,capsize=3,elinewidth=1.7)
    left.set(ylim=(2.5,-.5),xlim=(-.0115,.0185),yticks=[0,1,2],yticklabels=['emotion','style','genre'],xticks=[-.01,0,.01])
    left.xaxis.set_major_formatter(FuncFormatter(lambda x,_:'0' if x==0 else f'{x:+.2f}'.replace('-','−')))
    left.set_xlabel('AMI gain over the previous recipe',fontsize=13,labelpad=12)
    short=['40 neighbours','+ style 2\n+ trunk 512','+ teacher row\n0.035']
    for i,run in enumerate(d['top']):
        r=d['results'][run]; m=r['M']; right.axhline(i,color=GRID,lw=.6,zorder=0)
        right.errorbar(m['M'],i,xerr=[[m['M']-m['lo']],[m['hi']-m['M']]],fmt='o',color=COLORS[i],ms=7,capsize=4,elinewidth=1.8)
        right.text(1.08,i, 'qualifies' if r['qualifies'] else 'does not qualify\n(lower bound\nbelow zero)',
                   fontsize=12,va='center',linespacing=1.2)
        right.text(m['M'],i-.22,f"{m['M']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}]",ha='center',fontsize=11.5,color=SECONDARY)
    right.set(ylim=(2.5,-.5),xlim=(-.25,1.4),yticks=[0,1,2],yticklabels=short,xticks=[0,.5,1])
    right.set_xlabel('mean gain (seed ranges)',fontsize=13,labelpad=12)
    fig.text(.035,.035,'Each row: the candidate minus the previous recipe G, mean over seeds 1 to 3; whiskers: paired 95% intervals (2,000 joint draws).',fontsize=11.5,color=SECONDARY)
    fig.savefig(Path(__file__).with_name('confirmation.png'),dpi=200); plt.close(fig)

if __name__=='__main__': main()
