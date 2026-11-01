import json, numpy as np, sys, collections
sys.path.insert(0,'/project/CoSiR/scripts')
import extract_features as ef
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
A='/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b/'
d=json.load(open(A+'index.json')); ims=d['images']
z=np.load('/tmp/claude-0/-project-CoSiR/b74d81b8-4841-446a-8ba1-944bc40c5fd4/scratchpad/nearcopy.npz')
tr,best,arg=z['tr'],z['best'],z['arg']
R=ef.AFFECTION_IMAGE_ROOTS
def path(i):
    s=ims[i]['source']; n=ims[i]['image_name']
    if s=='you_aaai_affective': return Path(ef.AFFECTION_FI_ROOT)/n.split('_')[0]/n
    if s=='coco_affective': return Path(R[s]['train2014' if 'train2014' in n else 'val2014'])/n
    return Path(R[s])/n
def thumb(i):
    im=Image.open(path(i)).convert('L')
    a=np.asarray(im.resize((32,32)),dtype=np.float32).ravel(); a=a-a.mean(); return a/ (np.linalg.norm(a)+1e-6)
for c in [0.80,0.85,0.90]: print('n best>=',c,(best>=c).sum())
rng=np.random.default_rng(1)
sel=np.where(best>=0.90)[0]
low=np.where((best>=0.80)&(best<0.90))[0]; sel=np.concatenate([sel,rng.choice(low,400,replace=False)])
def f(j):
    try: return float(thumb(tr[j])@thumb(arg[j]))
    except Exception as e: return np.nan
with ThreadPoolExecutor(8) as ex: pc=np.array(list(ex.map(f,sel)))
print('nan',np.isnan(pc).sum())
b=best[sel]
for lo,hi in [(0.80,0.85),(0.85,0.90),(0.90,0.93),(0.93,0.95),(0.95,0.97),(0.97,0.98),(0.98,1.01)]:
    m=(b>=lo)&(b<hi); print(f'{lo}-{hi}: n={m.sum()} pixcopy(>0.95)={(pc[m]>0.95).sum()} (>0.9)={(pc[m]>0.9).sum()}')
cp=(pc>0.95)&(b>=0.90)
print('pixel copies with best>=0.90:',cp.sum())
print('by source pair', collections.Counter((ims[tr[j]]['source'][:6],ims[arg[j]]['source'][:6]) for j in sel[cp]).most_common())
print('min qwen cos among pixel copies', b[pc>0.95].min())
np.savez('/tmp/claude-0/-project-CoSiR/b74d81b8-4841-446a-8ba1-944bc40c5fd4/scratchpad/pix2.npz',sel=sel,pc=pc)
