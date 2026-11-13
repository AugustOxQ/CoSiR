# r29 run 2: the cause check of the one C13 mismatch (records/c1_cause_check.json). Run from rederive/.
# Cause check for the one C13 mismatch (genre t2i vs P1, hi): Codex's interval code on OUR per-query APs, and on its own.
import sys, json, glob, numpy as np
sys.path.insert(0, '/project/CoSiR/src/test/20261112_siglip_only/rederive')
import os; os.chdir('/project/CoSiR/src/test/20261112_siglip_only/rederive')
import r29rd_c1 as X
C = X.C
held = X.frame('held')
other, pick = X.comparator_text('genre', 't2i', 'p1', held)
pid = held['tp']
root = '/data/SSD2/cosir_r29/20261112_siglip_only/text/siglip1_prefix/scores/art_held/caption/cstar/'
ours = np.stack([np.load(root + f'genre__s{s}__t2i__center_k10.npz')['ap__genre'] for s in range(5)])
codex = np.stack([np.load(glob.glob(f'r29rd_c1_cache/*/candidate_held_text_genre_t2i_s{s}.npz')[0])['ap'] for s in range(5)])
rec = {c['id']: c for c in json.load(open('../records/text_cells_siglip1_prefix.json'))['cells']}['artelingo/siglip1/siglip1_prefix/genre/t2i/p1']
cxc = {c['id']: c for c in json.load(open('c1_rederived.json'))['cells']}['text/genre/t2i/p1']
a = C.retrieval_interval(ours, other, pid)
b = C.retrieval_interval(codex, other, pid)
out = dict(comparator_pick=pick,
           codex_code_on_our_aps=a, our_record={k: rec[k] for k in ('value', 'd', 'lo', 'hi')},
           diff_codex_code_on_ours_vs_our_record={k: abs(a[k] - rec[k]) for k in ('value', 'd', 'lo', 'hi')},
           codex_code_on_codex_aps=b, codex_record={k: cxc[k] for k in ('value', 'd', 'lo', 'hi')},
           diff_rerun_vs_codex_record={k: abs(b[k] - cxc[k]) for k in ('value', 'd', 'lo', 'hi')})
print(json.dumps(out, indent=1, default=float))
json.dump(out, open('/project/CoSiR/src/test/20261112_siglip_only/records/c1_cause_check.json', 'w'), indent=1, default=float)
