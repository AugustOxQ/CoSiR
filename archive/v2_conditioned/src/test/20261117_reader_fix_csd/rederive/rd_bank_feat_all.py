"""All six banks: 2,000 random episodes per half, features in aspect_deltas precision vs stored X (bit-equality)."""
import numpy as np
import rd_core as K
from src.eval.aspect_episodes import AspectEpisodes
heads = {h: np.load(K.RES / f"rb_heads_{h}.npz") for h in K.GROUPINGS}
post_cf = {h: {"img": heads[h]["img"], "txt": heads[h]["txt"]} for h in K.GROUPINGS}
for cfg in ("A1", "A0", "AR"):
    g = K.CONFIGS[cfg]
    rz = np.load(K.RES / f"rb_reader_{cfg}.npz")
    for j in (0, 1):
        z = np.load(K.RES / f"rb_bank_{cfg}_half{j}.npz")
        X = rz[f"half{j}__X"]
        n = len(z["anchor"])
        idx = np.sort(np.random.default_rng(1000 + j).choice(n, 2000, replace=False))
        ep = AspectEpisodes("x", "y", *(np.asarray(z[k])[idx] for k in
                                        ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")))
        fa, fb = K.rb_features_f32(post_cf, ep, g, "a"), K.rb_features_f32(post_cf, ep, g, "b")
        print(cfg, j, "equal", np.array_equal(fa, X[idx]) and np.array_equal(fb, X[n + idx]),
              "max", float(max(np.abs(fa - X[idx]).max(), np.abs(fb - X[n + idx]).max())), flush=True)
