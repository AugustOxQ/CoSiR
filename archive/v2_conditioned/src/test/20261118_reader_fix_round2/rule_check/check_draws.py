"""Read-only executability check of the draft's §4.4(b) draw procedure as written (A0, both halves): it runs,
terminates, and is reproducible. No features, SMD or k are computed; nothing is saved."""
import hashlib
from pathlib import Path
import numpy as np
R = Path("/project/CoSiR/src/test/20261117_reader_fix_csd/results")
H = np.load(R / "rb_halves.npz")
paint = H["painting_of_local_row"]
def draws(j, cfg="A0"):
    z = np.load(R / f"rb_bank_{cfg}_half{j}.npz")
    A = z["anchor"].astype(np.int64); N = len(A)
    Rr = H[f"local_rows_half{j}"]
    g = np.random.default_rng((21700, 21800)[j])
    U = g.random((N, 2, 4)); order = np.argsort(U, axis=2, kind="stable")
    img = Rr[g.integers(0, len(Rr), size=(N, 2, 4))]; it_i = 0
    while True:
        bad = paint[img] == paint[A][:, None, None]
        if not bad.any(): break
        img[bad] = Rr[g.integers(0, len(Rr), size=bad.sum())]; it_i += 1
    cap = Rr[g.integers(0, len(Rr), size=(N, 2, 4))]; it_c = 0
    while True:
        bad = (paint[cap] == paint[A][:, None, None]) | (paint[cap] == paint[img])
        if not bad.any(): break
        cap[bad] = Rr[g.integers(0, len(Rr), size=bad.sum())]; it_c += 1
    # paintings already in the episode (other than the anchor's) hit by a replacement row
    ep = np.concatenate([z["candidates"], z["pairs_a_img"], z["pairs_a_txt"], z["pairs_b_img"], z["pairs_b_txt"]], axis=1)
    ep_p = paint[ep]
    hit = np.zeros((N, 2, 4), bool)
    for arr in (img, cap):
        hit |= (paint[arr][..., None] == ep_p[:, None, None, :]).any(-1)
    # replacement image and caption share a group by chance? (not computed: needs labels) ; replacement rows repeated within an episode
    flat = np.concatenate([img.reshape(N, -1), cap.reshape(N, -1)], 1)
    rep = np.mean([len(np.unique(paint[r])) < flat.shape[1] for r in flat[:20000]])
    h = hashlib.sha256(order.tobytes() + img.tobytes() + cap.tobytes()).hexdigest()[:16]
    return it_i, it_c, float(hit.mean()), float(rep), h
for j in (0, 1):
    a = draws(j); b = draws(j)
    print(f"A0 half {j}: rejection rounds img {a[0]} cap {a[1]}; share of replacement slots whose image or caption painting "
          f"is already in the episode {a[2]:.5f}; share of episodes (first 20k) with a repeated painting among replacement rows {a[3]:.4f}; "
          f"reproducible {a[4] == b[4]} ({a[4]})")
