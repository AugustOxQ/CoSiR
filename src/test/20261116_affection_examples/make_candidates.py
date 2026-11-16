import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

OUT = Path('/project/CoSiR/src/test/20261116_affection_examples')
sys.path.insert(0, '/project/CoSiR/src/test/20261102_backbones_sweep')
sys.path.insert(0, '/project/CoSiR/src/test/20261030_affection_confirmation')
import ac_data, ac_paths, ac_extract

P = Path('/data/SSD2/pre_extract/affection_local/siglip_so400m')
E = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/exports/affection/photo_siglip1/test__s0.npz')
keys = ac_data.keys('test')
with np.load(E) as f:
    ek = f['keys'].astype(str).tolist(); emotion = f['emb'].astype(np.float32)
pk = json.loads((P/'keys.json').read_text())
assert len(pk) == len(set(pk)) and len(ek) == len(set(ek))
assert set(ek) == set(keys) and len(ek) == len(keys)
lookup = {k:i for i,k in enumerate(pk)}
assert all(k in lookup for k in keys)
plain_ids = [pk[lookup[k]] for k in keys]
emo_lookup = {k:i for i,k in enumerate(ek)}
emo_ids = [ek[emo_lookup[k]] for k in keys]
assert plain_ids == emo_ids == keys  # both query and gallery use these rows
plain = np.asarray(np.load(P/'img.npy', mmap_mode='r')[[lookup[k] for k in keys]], dtype=np.float32)
emotion = emotion[[emo_lookup[k] for k in keys]]
assert plain.shape == (len(keys),1152) and emotion.shape == (len(keys),2048)
for x in (plain, emotion):
    assert np.isfinite(x).all() and (np.linalg.norm(x,axis=1)>0).all()
    x /= np.linalg.norm(x,axis=1,keepdims=True)
y = ac_data.labels('test'); tied = ac_data.ties('test'); names = ac_paths.EMOTIONS
# Exclude tied plurality labels from queries and the test-set gallery.
qs = np.flatnonzero(~tied)
def neighbours(x):
    out = np.empty((len(qs),5),dtype=np.int32)
    for start in range(0,len(qs),128):
        q = qs[start:start+128]; sim = x[q] @ x.T
        sim[:, tied] = -np.inf
        sim[np.arange(len(q)),q] = -np.inf
        for j, row in enumerate(sim):
            # Deterministic ties: key order, since keys are sorted.
            out[start+j] = np.lexsort((np.arange(len(keys)), -row))[:5]
    return out
pn, en = neighbours(plain), neighbours(emotion)
pc = ((y[pn] == y[qs,None]) & ~tied[pn]).sum(1)
ec = ((y[en] == y[qs,None]) & ~tied[en]).sum(1)
valid = np.flatnonzero((ec >= 4) & (pc <= 1))
# Panel (a): descending emotion gain, then key. No photo genre field; retain its two-per-class cap.
ranked = sorted(valid, key=lambda i: (-int(ec[i]-pc[i]), keys[qs[i]]))
chosen=[]; classes={}
for i in ranked:
    label=int(y[qs[i]])
    if classes.get(label,0)>=2: continue
    classes[label]=classes.get(label,0)+1; chosen.append(i)
    if len(chosen)==12: break
assert len(chosen)==12, f'Only {len(chosen)} candidates with two-per-class cap'
used = sorted({int(r) for i in chosen for r in [qs[i], *pn[i], *en[i]]})
used_keys = [keys[r] for r in used]
spec, rows = ac_extract.spec_for(used_keys)
images = {r: spec.load_image(int(s)).convert('RGB') for r,s in zip(used,rows)}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17)
w,h=150,116; left=145; rowh=174; margin=14
sheet=Image.new('RGB',(left+11*w+margin,48+12*rowh),'white'); draw=ImageDraw.Draw(sheet)
draw.text((left,10),'Query',font=bold,fill='black'); draw.text((left+w,10),'Plain SigLIP — nearest 5',font=bold,fill='black'); draw.text((left+6*w,10),'CoSiR photo model — nearest 5',font=bold,fill='black')
def label(r): return 'unclear (tie)' if tied[r] else names[y[r]]
records=[]
for rank,i in enumerate(chosen,1):
    q=int(qs[i]); yy=48+(rank-1)*rowh
    draw.text((8,yy+10),f'Candidate {rank}',font=bold,fill='black')
    draw.text((8,yy+36),f'plain {pc[i]}/5',font=font,fill='black'); draw.text((8,yy+58),f'CoSiR {ec[i]}/5',font=font,fill='black')
    for col,r in enumerate([q,*pn[i],*en[i]]):
        r=int(r); xx=left+col*w
        thumb=ImageOps.contain(images[r],(w-8,h),method=Image.Resampling.LANCZOS)
        sheet.paste(thumb,(xx+(w-8-thumb.width)//2,yy+(h-thumb.height)//2))
        match=col>0 and not tied[r] and y[r]==y[q]
        draw.text((xx+2,yy+h+3),('✓ ' if match else '')+label(r),font=font,fill=(0,120,45) if match else (20,20,20))
        source,name=keys[r].split('/',1)
        records.append(dict(candidate=rank,query_id=keys[q],query_label=label(q),plain_matches=int(pc[i]),cosir_matches=int(ec[i]),role='query' if col==0 else ('plain' if col<=5 else 'cosir'),neighbour_rank='' if col==0 else (col if col<=5 else col-5),photo_id=keys[r],photo_label=label(r),matches_query=bool(match),source_dataset=source,original_image_name=name))
with (OUT/'candidates.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
sheet.save(OUT/'contact_sheet.png',dpi=(150,150))
(OUT/'results.json').write_text(json.dumps(dict(test_rows=len(keys),plain_store_rows=len(pk),emotion_shape=list(emotion.shape),ids_match=True,tied=int(tied.sum()),clear_queries=len(qs),passing=len(valid),selected=len(chosen),unique_photos=len(used)),indent=2)+'\n')
