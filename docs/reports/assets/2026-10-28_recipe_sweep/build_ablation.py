"""B2: unretuned removal effects from checked development and held JSON."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator
from figure_data import ASPECTS, ROWS, BLUE, LIGHT, GRID, SECONDARY, REFERENCE, ablation_data, style, clean


def main():
    data=ablation_data(); style(plt)
    fig,axes=plt.subplots(1,3,figsize=(13,6.5),dpi=200)
    fig.subplots_adjust(left=.225,right=.935,bottom=.21,top=.64,wspace=.30)
    fig.suptitle('Removing one part at a time: the graphs, buddy terms and content correction\ncarry emotion and style; the relational term carries genre',
                 x=.5,y=.97,fontsize=18,linespacing=1.18)
    fig.text(.5,.845,'Removal minus the pick; paired 95% intervals; nothing retuned',ha='center',fontsize=14,color=SECONDARY)
    fig.legend(handles=[Line2D([],[],marker=m,color=c,ls='none',ms=7,label=l) for m,c,l in
        [('o',BLUE,'Held'),('s',LIGHT,'Development')]],loc='center',bbox_to_anchor=(.5,.785),ncol=2,frameon=False,fontsize=14)
    for j,(ax,a) in enumerate(zip(axes,ASPECTS)):
        ax.set_xscale('symlog',linthresh=.005,linscale=1)
        ax.set_xlim(-.28,.035); ax.set_ylim(7.6,-.7)
        ax.set_title(a.title(),fontsize=17,weight='bold',pad=28)
        ax.axvline(0,color=REFERENCE,lw=1.2)
        ax.text(0,1.018,'the pick',transform=ax.get_xaxis_transform(),ha='center',fontsize=13,color=SECONDARY)
        for i,(key,label) in enumerate(ROWS):
            ax.axhline(i,color=GRID,lw=.65,zorder=0)
            for split,offset,color,marker in [('dev',-.16,LIGHT,'s'),('held',.16,BLUE,'o')]:
                v=data[split]['results'][key]['aspects'][a]
                d,lo,hi=[v[k] for k in ('effect','lo','hi')]
                ax.errorbar(d,i+offset,xerr=[[d-lo],[hi-d]],fmt=marker,color=color,
                            ms=6,elinewidth=1.7,capsize=3,zorder=3)
                if split=='held' and (lo>0 or hi<0):
                    ax.text(1.015,i+offset,'higher' if lo>0 else 'lower',
                            transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=13,color=BLUE)
        ax.set_yticks(range(8),[label for _,label in ROWS] if j==0 else [])
        ax.tick_params(axis='y',pad=12)
        ax.set_xticks([-.2,-.01,0,.01])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:'0' if x==0 else f'{x:+g}'.replace('-','−')))
        ax.tick_params(axis='x',labelsize=13,pad=8)
        clean(ax)
    fig.text(.60,.105,'AMI change, symmetric log scale',ha='center',fontsize=15)
    fig.text(.60,.065,'Linear within ±0.005 AMI; identical axes in all panels',ha='center',fontsize=13,color=SECONDARY)
    fig.text(.035,.025,'Development: 29,232 paintings, seeds 1–3  •  Held: 12,281 paintings, seeds 0–4',fontsize=14,color=SECONDARY)
    fig.savefig(Path(__file__).with_name('ablation.png'),dpi=200)
    plt.close(fig)


if __name__=='__main__': main()
