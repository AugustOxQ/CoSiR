import sys, time, numpy as np, glob
sys.path.insert(0,'.'); import extract
imgs=sorted(glob.glob('/data/PDD/wikiart_proj/wikiart/Impressionism/*.jpg'))[:40]
txt=["a dog on grass","a blue bird with a long bill","The pears are just about ripe to eat and enjoy"]*10
for n in sys.argv[1:]:
    t=time.time()
    try:
        m=extract.build(n); a=m.images(imgs); b=m.texts(txt)
        print(n,a.shape,b.shape,np.linalg.norm(a,axis=1)[:2],(a@b.T).mean(),time.time()-t,flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
    del m
    import torch; torch.cuda.empty_cache()
