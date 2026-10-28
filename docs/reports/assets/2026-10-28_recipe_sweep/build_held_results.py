"""B1: checked held AMI and paired differences. Run with the CoSiR Python."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from figure_data import ASPECTS, BLUE, LIGHT, GREEN, ORANGE, GRID, SECONDARY, REFERENCE, held_data, style, clean


def main():
    bars, verdict = held_data()
    style(plt)
    fig = plt.figure(figsize=(13,5.5), dpi=200)
    fig.suptitle('Held paintings: the pick is above the instructed Qwen image\non emotion and style, and level on genre',
                 x=.5, y=.97, fontsize=19, linespacing=1.16)
    fig.text(.035,.735,'a   Held AMI', fontsize=16, weight='bold')
    fig.text(.035,.68,'Each AMI axis starts above zero; breaks marked',fontsize=13,color=SECONDARY)
    colors = [ORANGE,GREEN,LIGHT,BLUE]
    limits = [(.08,.13),(.20,.54),(.40,.59)]
    ticks = [[.08,.10,.12],[.20,.30,.40],[.40,.46,.52]]
    for j,a in enumerate(ASPECTS):
        ax = fig.add_axes([.05+j*.21,.255,.155,.345])
        low,high=limits[j]
        ax.bar(range(4), [v-low for v in bars[a]], bottom=low, color=colors, width=.65, zorder=3)
        for i,v in enumerate(bars[a]):
            ax.annotate(f'{v:.4f}', (i,v), xytext=(0,5), textcoords='offset points',
                        ha='center', va='bottom', rotation=60, fontsize=13, zorder=5)
            # White gap with two dark slashes across every bar marks the omitted origin.
            ax.plot([i-.36,i+.36],[low+(high-low)*.014]*2,color='white',lw=7,zorder=4)
            for x in (i-.12,i+.10):
                ax.plot([x-.06,x+.06],[low,low+(high-low)*.03],color=REFERENCE,lw=1,clip_on=False,zorder=5)
        ax.set_ylim(low, high)
        ax.set_xlim(-.65,3.65)
        ax.set_yticks(ticks[j])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:.2f}'))
        ax.set_xticks(range(4),['P','Q','G','c*'])
        ax.set_title(a.title(),fontsize=15,weight='bold',pad=12)
        ax.set_axisbelow(True); ax.grid(axis='y',color=GRID,lw=.7); clean(ax)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,[
        'P  PercepT (strongest entry)', 'Q  Instructed Qwen image', 'G  Previous recipe G', 'c*  The pick'])],
        loc='lower left',bbox_to_anchor=(.025,.075),ncol=2,frameon=False,fontsize=13,
        columnspacing=1.1,handlelength=1)
    fig.text(.69,.735,'b   Pick minus instructed image',fontsize=15,weight='bold')
    ax=fig.add_axes([.715,.255,.23,.345])
    ax.axvline(0,color=REFERENCE,lw=1.2)
    for j,a in enumerate(ASPECTS):
        v=verdict['aspects'][a]; d,lo,hi=[v[k] for k in ('d','lo','hi')]
        ax.hlines(j,-.015,.020,color=GRID,lw=.7)
        ax.errorbar(d,j,xerr=[[d-lo],[hi-d]],fmt='o',color=BLUE,ms=7,capsize=4,lw=2)
        interval = f'{d:+.4f} [{lo:+.4f}, {hi:+.4f}]'.replace('-', '−')
        ax.text(.02,j-.19,interval,transform=ax.get_yaxis_transform(),va='bottom',fontsize=13,
                bbox=dict(facecolor='white',edgecolor='none',pad=.8))
        ax.text(1.025,j,'above' if v['outcome']=='T-WIN' else 'level',transform=ax.get_yaxis_transform(),va='center',fontsize=14)
    margin=-verdict['ranges']['genre']
    ax.vlines(margin,1.70,2.35,color=SECONDARY,linestyle=(0,(4,3)),lw=2.0,zorder=5)
    ax.text(margin+.0004,2.48,
            f'level bar: minus one\nseed range, {margin:.4f}'.replace('-', '−'),
            va='top',fontsize=13,color=SECONDARY,linespacing=1.15,
            bbox=dict(facecolor='white',edgecolor='none',pad=.8))
    ax.set_yticks(range(3),[a.title() for a in ASPECTS]); ax.set_ylim(3.35,-.65)
    ax.set_xlim(-.015,.020); ax.set_xticks([-.01,0,.01,.02])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:'0' if x==0 else f'{x:+.2f}'.replace('-','−')))
    ax.set_xlabel('AMI difference',labelpad=10); clean(ax)
    fig.text(.035,.025,'12,281 held paintings (genre labels on 9,949), 5 seeds; paired 95% intervals',fontsize=14,color=SECONDARY)
    fig.savefig(Path(__file__).with_name('held_results.png'),dpi=200)
    plt.close(fig)


if __name__=='__main__': main()
