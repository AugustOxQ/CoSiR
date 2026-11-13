# r29 run 2 (report alternative, not a verdict); run from rederive/. Output records/alt_told_vs_p1.json.
# Alternative check (report, "Agent choices checked"): image mAP against plain SigLIP 1 with K*'s told aspect head
# instead of the untold joined vector (r17 SC7 protocol, spec D18). Uses Codex's cached told per-query APs and its code.
import sys, os, glob, json, numpy as np
os.chdir('/project/CoSiR/src/test/20261112_siglip_only/rederive'); sys.path.insert(0, '.')
import r29rd_c1 as X
C = X.C
held, val = X.frame('held'), X.frame('val')
r14 = X.rd.read_json(X.R14_RECORDS / 'stage1_siglip1.json')['artret_picks']['frozen']['pick']
picks = X.rd.read_json('r29rd_c1_picks.json')['picks']
out = {}
for a in X.ASPECTS:
    y = held['labels'][a]['i']
    other = C.image_ap(X.plain_image(val), X.plain_image(held), y, r14, batch_size=128)[0]
    told = np.stack([np.load(glob.glob(f'r29rd_c1_cache/*/candidate_held_told_{a}_s{s}.npz')[0])['ap'] for s in range(5)])
    st = C.retrieval_interval(told, other)
    r = picks['image_map'][a]['range']
    st.update(r=r, outcome=C.outcome(st['lo'], r), plain_pick=r14, told_pick=picks['image_map'][a]['pick'])
    out[a] = st
    print(a, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}, flush=True)
json.dump(out, open('/project/CoSiR/src/test/20261112_siglip_only/records/alt_told_vs_p1.json', 'w'), indent=1, default=float)
