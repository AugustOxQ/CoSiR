"""Validation only: can the pair bar pass? Late fusion a=0.7 against plain Qwen, 5 seeds, ArtELingo and MMArt validation.
Also the coder differences in the bootstrap form. Read only; writes to the scratchpad."""
import sys, json, time, pickle
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,'/project/CoSiR/src/test/20261105_text_retrieval')
import tx_paths as paths, tx_data as data, tx_metrics as metrics, tx_model
import rt_data
torch.set_num_threads(4)
OUT=Path('/tmp/claude-0/-project-CoSiR/88455088-ddbb-4074-8cbe-a09cc630ef5c/scratchpad')
a=0.7; wc,wp=float(np.sqrt(1-a)),float(np.sqrt(a))
def join(c,p): return np.concatenate((wc*c,wp*p),axis=1).astype(np.float32)
def rank(q,g,qp,gp,ro):
    return metrics.evaluate(q,g,qp,gp,{},False,ro,'cpu')['first_rank']
res={}
# ---- MMArt validation, unified text: all heads and read-outs at seed 0, then 5 seeds at the pick
idx,texts=data.mmart_texts()
import tx_extras
path=paths.OUT/'text/mmart_val/e_unified'
T=tx_extras.store_text(path,idx['key'].tolist())
rows=data.gather(rt_data.names('val'),idx['painting'].tolist())
I=data.image_rows('plain')[rows]
n=len(T); pid=np.arange(n,dtype=np.int64)
def mmart_side(h,s):
    im,ca=tx_model.load_sides(h,seed=s,device='cpu')
    return ca(T),im(I)
t0=time.time()
plain={}
for d,(q,g) in {'t2i':(T,I),'i2t':(I,T)}.items():
    plain[d]={ro:rank(q,g,pid,pid,ro) for ro in ('r0','center','center_k10')}
table={}
for h in paths.ASPECTS:
    ct,ci=mmart_side(h,0); jt,ji=join(ct,T),join(ci,I)
    for d,(q,g) in {'t2i':(jt,ji),'i2t':(ji,jt)}.items():
        for ro in ('r0','center','center_k10'):
            table[h,d,ro]=rank(q,g,pid,pid,ro)
print('mmart seed0 done',round(time.time()-t0),flush=True)
summ={k:metrics.summarize_pair(v) for k,v in table.items()}
psumm={(d,ro):metrics.summarize_pair(v) for d in plain for ro,v in plain[d].items()}
picks={}
for d in ('t2i','i2t'):
    per={h:max(('r0','center','center_k10'),key=lambda ro:summ[h,d,ro]['mrr']) for h in paths.ASPECTS}
    h=max(per,key=lambda x:summ[x,d,per[x]]['r10'])
    pro=max(('r0','center','center_k10'),key=lambda ro:psumm[d,ro]['mrr'])
    picks[d]=(h,per[h],pro)
    print('mmart',d,'pick',picks[d],'LF r10',round(summ[h,d,per[h]]['r10'],4),'plain r10',round(psumm[d,pro]['r10'],4),{x:round(summ[x,d,per[x]]['r10'],4) for x in per},flush=True)
M={}
for d in ('t2i','i2t'):
    h,ro,pro=picks[d]
    hits=[]
    for s in range(5):
        ct,ci=mmart_side(h,s); jt,ji=join(ct,T),join(ci,I)
        q,g=(jt,ji) if d=='t2i' else (ji,jt)
        hits.append((rank(q,g,pid,pid,ro)<=10).astype(np.float64))
    M[d]=dict(lf=np.stack(hits),plain=(plain[d][pro]<=10).astype(np.float64),pid=pid)
    print('mmart',d,'seed r10',np.round(M[d]['lf'].mean(1),4).tolist(),flush=True)
res['mmart']=M
pickle.dump(res,open(OUT/'pairs.pkl','wb'))
# ---- ArtELingo validation, genre head, centering (the seed 0 pick in followup.json)
c=data.val_captions(); cp=c['pid']; ip=np.arange(6152,dtype=np.int64)
tA=data.text_rows('plain'); iA=data.image_rows('plain')[:6152]
A={}
pl={d:rank(*( (tA,iA,cp,ip) if d=='t2i' else (iA,tA,ip,cp) ),'center') for d in ('t2i','i2t')}
print('plain r10',{d:round(float((v<=10).mean()),4) for d,v in pl.items()},round(time.time()-t0),flush=True)
for d in ('t2i','i2t'):
    hits=[]
    for s in range(5):
        im,ca=tx_model.load_sides('genre',seed=s,device='cpu')
        jt,ji=join(ca(tA),tA),join(im(iA),iA)
        fr=rank(jt,ji,cp,ip,'center') if d=='t2i' else rank(ji,jt,ip,cp,'center')
        hits.append((fr<=10).astype(np.float64)); print(d,s,round(hits[-1].mean(),4),round(time.time()-t0),flush=True)
    A[d]=dict(lf=np.stack(hits),plain=(pl[d]<=10).astype(np.float64),pid=cp if d=='t2i' else ip)
res['artelingo']=A
pickle.dump(res,open(OUT/'pairs.pkl','wb'))
print('done',round(time.time()-t0))
