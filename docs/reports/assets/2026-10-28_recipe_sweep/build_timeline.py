"""Rebuild the Amsterdam timeline from the report and ablation run log."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from source_checks import style, REPORT, LOG, BLUE, SECONDARY, GRID

# Times: report 'What ran'; held ablation completion: log 'Held (done ...)'.
STEPS = [
    ('17:08','17:20',220,True,365,390,'Stage 0 · held emotion\ntest + analysis','passed (both scales)'),
    ('17:08','18:29',460,False,370,590,'Stage 1 · 13 settings\none at a time (39 runs)','four settings to the grid'),
    ('18:30','19:39',460,False,910,590,'Stage 2 · joint grid\n(18 new runs)','40 neighbours leads'),
    ('19:39','20:13',220,True,820,390,'Stage 3 · 3 candidates\n× 3 seeds','pick: 40 neighbours'),
    ('20:14','21:09',220,True,925,175,'Stage 4 · held test + analysis','emotion/style above; genre level'),
    ('21:11','22:50',220,False,1250,390,'Development ablation\n(8 removals)','decides nothing'),
    ('22:50','03:53',220,False,1950,390,'Held ablation (8 removals)','decides nothing'),
]


def main():
    style()
    report, log = REPORT.read_text(), LOG.read_text()
    for start,end,*_ in STEPS:
        assert start in report or start in log
        assert end in report or end in log
    assert '23:00' in report
    # Explicit timezone also handles the midnight date transition.
    def x(time):
        hour,minute=map(int,time.split(':'))
        dt=datetime(2026,10,27 if hour>=17 else 28,hour,minute,tzinfo=ZoneInfo('Europe/Amsterdam'))
        origin=datetime(2026,10,27,17,tzinfo=dt.tzinfo)
        return 360+(dt-origin).total_seconds()/3600*(1960/11)
    fig=plt.figure(figsize=(12,4.5),dpi=200)
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,2400),ylim=(0,900)); ax.set_axis_off()
    ax.text(70,865,'The recipe sweep ran overnight: picked by 20:13,\ntested on held paintings by 21:09',fontsize=21,va='top',linespacing=1.12)
    ax.text(70,700,'27–28 October 2026 · Amsterdam time · bars include analysis and checks',color=SECONDARY,fontsize=13)
    ax.legend(handles=[Line2D([],[],marker='o',ls='none',color=BLUE,label='test or pick',markersize=7),
                       Line2D([],[],marker='o',ls='none',color='#a6a49e',label='sweep or check',markersize=7)],
              loc='center left',bbox_to_anchor=(.025,.74),frameon=False,ncol=2,fontsize=13)
    for y,label in [(460,'cluster node\n(3 GPUs)'),(220,'local GPU\n(RTX 3090)')]:
        ax.text(70,y,label,va='center',fontsize=13,linespacing=1.25)
        ax.plot([360,2320],[y,y],color=GRID,lw=1,zorder=0)
    ax.plot([x('23:00')]*2,[95,605],color=SECONDARY,lw=1,ls=(0,(4,4)),zorder=0)
    ax.text(x('23:00')+12,615,'planned end · 23:00',fontsize=12,color=SECONDARY)
    for start,end,y,test,tx,ty,label,outcome in STEPS:
        a,b=x(start),x(end); mid=(a+b)/2; color=BLUE if test else '#a6a49e'
        ax.plot([a,b],[y,y],color=color,lw=12,solid_capstyle='butt',zorder=3)
        ax.plot([b,b],[y-8,y+8],color='white',lw=1,zorder=4)
        ax.scatter([a],[y],s=45,color=color,zorder=4)
        ha='center' if start=='22:50' else 'left'
        anchor=tx if ha=='center' else tx+50
        # Leaders attach to the bar midpoint, with all local labels outside the lane.
        target=ty-120 if ty>y and '\n' in label else ty-85 if ty>y else ty+14
        ax.plot([mid,mid,anchor],[y+10 if ty>y else y-10,target,target],color=GRID,lw=1,zorder=1)
        ax.text(tx,ty,label,ha=ha,va='top',fontsize=12.5,linespacing=1.12,
                bbox=dict(facecolor='white',edgecolor='none',pad=1.5),zorder=5)
        offset=76 if '\n' in label else 40
        ax.text(tx,ty-offset,outcome,ha=ha,va='top',fontsize=12,color=SECONDARY,linespacing=1.12,
                bbox=dict(facecolor='white',edgecolor='none',pad=1.5),zorder=5)
    ax.plot([360,2320],[75,75],color=SECONDARY,lw=1)
    for hour in range(17,29):
        xx=x(f'{hour%24:02}:00'); ax.plot([xx,xx],[70,80],color=SECONDARY,lw=1)
        ax.text(xx,54,f'{hour%24:02}:00',ha='center',va='top',fontsize=12,color=SECONDARY)
    fig.savefig(Path(__file__).with_name('timeline.png'),dpi=200); plt.close(fig)

if __name__=='__main__': main()
