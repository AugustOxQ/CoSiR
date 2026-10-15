import sys, time, json, numpy as np
sys.path.insert(0,'/project/CoSiR-wt/r9-siglip-a2/src/test/20261202_r9_scope')
import r9_common as R, r9_recipe_groups as G
print('module', G.__file__)
RES='/project/CoSiR/src/test/20261202_r9_scope/results/recipe/'
out={}
for bb,dirn in (('clip_b32',None),('qwen3vl_emb_2b','qwen3vl_emb_2b'),('siglip2_so400m','siglip2_so400m')):
    t=time.time()
    data=R.load_data(bb); split=R.load_split(data); st=split.scorer_train
    if dirn is None:
        from r9_backbone import parts
        f=str(parts('clip_b32')['affect_km'])
    else: f=RES+dirn+'/partitions.npz'
    z=np.load(f); stored={k:z[k] for k in z.files}
    print(bb,'stored from',f,'keys',list(stored))
    print(' stored smallest: image',int(np.bincount(stored['image'],minlength=64).min()),'caption',int(np.bincount(stored['caption'],minlength=64).min()),'affect',int(np.bincount(stored['affect'],minlength=64).min()))
    frozen=G.load_frozen(); log={}
    try:
        new=G.fit_partitions(data.img_features,data.txt_features,split.groups,st,frozen,restarts=log)
    except AssertionError as e:
        print(' ASSERT',str(e)[:300]); continue
    print(' restarts log',json.dumps(log))
    for k in new:
        print('  ',k,'bit-equal' if (new[k].tobytes()==stored[k].tobytes() and new[k].dtype==stored[k].dtype) else 'DIFFERENT', 'sizes min',int(np.bincount(new[k],minlength=64).min()) if k in('image','caption') else '')
    if bb.startswith('siglip'):
        for k in ('image','caption'): print('  ',k,'sorted smallest 5',np.sort(np.bincount(new[k],minlength=64))[:5],'max',np.bincount(new[k]).max())
    print(' time',round(time.time()-t,1),'s')
