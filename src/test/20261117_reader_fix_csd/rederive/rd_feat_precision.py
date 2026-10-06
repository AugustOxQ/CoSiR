"""Which arithmetic precision reproduces the stored bank features (rb_reader_*.npz X) and the stored seed-42 P?
Diagnostic for the ~1e-7 feature / ~5e-7 probability differences; decides nothing."""
import itertools
import numpy as np
import rd_core as K
from src.eval.aspect_episodes import AspectEpisodes


def feats(post, ep, groupings, cond, agree32, mean32, sd32):
    si, st, ci, ct, _ = ep.condition(cond)
    cols = []
    for h in groupings:
        if agree32:
            sup, con = K.pair_agreements32(post, ep, h, cond)
        else:
            sup, con = K.pair_agreements(post, ep, h, cond)
        sm = sup.mean(1) if mean32 else sup.astype(np.float64).mean(1)
        cm = con.mean(1) if mean32 else con.astype(np.float64).mean(1)
        d = sm - cm
        ss = sup.std(1, ddof=1) if sd32 else sup.astype(np.float64).std(1, ddof=1)
        cs = con.std(1, ddof=1) if sd32 else con.astype(np.float64).std(1, ddof=1)
        ai = np.argmax(post[h]["img"][si], axis=-1)
        at = np.argmax(post[h]["txt"][st], axis=-1)
        cols += [np.asarray(x, np.float64) for x in (sm, cm, d, ss, cs, (ai == at).mean(1))]
    return np.stack(cols, axis=1)


cfg, j = "A1", 0
g = K.CONFIGS[cfg]
heads = {h: np.load(K.RES / f"rb_heads_{h}.npz") for h in g}
post_cf = {h: {"img": heads[h]["img"], "txt": heads[h]["txt"]} for h in g}
z = np.load(K.RES / f"rb_bank_{cfg}_half{j}.npz")
X = np.load(K.RES / f"rb_reader_{cfg}.npz")[f"half{j}__X"]
n = len(z["anchor"])
idx = np.sort(np.random.default_rng(7).choice(n, 2000, replace=False))
ep = AspectEpisodes("x", "y", *(np.asarray(z[k])[idx] for k in
                                ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")))
for agree32, mean32, sd32 in itertools.product((False, True), repeat=3):
    if not agree32 and (mean32 or sd32):
        continue
    fa = feats(post_cf, ep, g, "a", agree32, mean32, sd32)
    fb = feats(post_cf, ep, g, "b", agree32, mean32, sd32)
    da, db = np.abs(fa - X[idx]), np.abs(fb - X[n + idx])
    percol = np.maximum(da.max(0), db.max(0))
    print(f"agree32={agree32} mean32={mean32} sd32={sd32}: max {max(da.max(), db.max()):.3g}; "
          f"equal {np.array_equal(fa, X[idx]) and np.array_equal(fb, X[n + idx])}; per stat (S,C,D,sdS,sdC,match) "
          f"{[float(percol[k::6].max()) for k in range(6)]}", flush=True)
