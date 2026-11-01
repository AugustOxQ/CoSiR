import json, numpy as np, collections, time
A='/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b/'
d=json.load(open(A+'index.json'))
ims=d['images']
split=[]; src=[]; name=[]
for im in ims:
    s=[x for x in im['splits'] if x!='ignored']; split.append(s[0]); src.append(im['source']); name.append(im['image_name'])
split=np.array(split); src=np.array(src)
X=np.load(A+'img.npy').astype(np.float32)
n=np.linalg.norm(X,axis=1); print('norm min max',n.min(),n.max())
X/=n[:,None]
tr=np.where(split=='train')[0]; vt=np.where(split!='train')[0]
V=X[vt]
best=np.empty(len(tr),np.float32); arg=np.empty(len(tr),np.int64)
t=time.time()
for i in range(0,len(tr),4096):
    S=X[tr[i:i+4096]]@V.T
    a=S.argmax(1); best[i:i+4096]=S[np.arange(len(a)),a]; arg[i:i+4096]=vt[a]
print('secs',time.time()-t)
np.savez('/tmp/claude-0/-project-CoSiR/b74d81b8-4841-446a-8ba1-944bc40c5fd4/scratchpad/nearcopy.npz',tr=tr,best=best,arg=arg)
for th in [0.90,0.95,0.97,0.98,0.99,0.995,0.999]:
    m=best>=th; print('th',th,'n',m.sum())
m=best>=0.95
c=collections.Counter((src[tr[j]],src[arg[j]],split[arg[j]]) for j in np.where(best>=0.98)[0]); print('>=0.98 pairs',c.most_common(20))
c=collections.Counter((src[tr[j]],src[arg[j]]) for j in np.where((best>=0.95)&(best<0.98))[0]); print('0.95-0.98 pairs',c.most_common(20))
h,e=np.histogram(best[best>=0.9],bins=np.arange(0.9,1.0001,0.005)); print(list(zip(np.round(e[:-1],3),h)))
# examples in band
for j in np.where((best>=0.95)&(best<0.98))[0][:15]:
    print(round(float(best[j]),4), src[tr[j]], name[tr[j]], '|', src[arg[j]], name[arg[j]], split[arg[j]])
