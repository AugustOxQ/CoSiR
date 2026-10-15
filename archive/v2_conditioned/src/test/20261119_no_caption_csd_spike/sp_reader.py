"""Spike: the A1c = (affect, image, csd) half-readers from round 1's A1 banks, no new banks. Writes results/sp_reader_A1c.{pkl,json}."""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20261118_reader_fix_round2"))
import r2_common as R  # noqa: E402

C, rb, rf = R.C, R.rb, R.rf
HERE = Path(__file__).resolve().parent
PARTS = ("affect", "image", "csd")
KEEP_BLOCKS = [("affect", "csd"), ("affect", "image"), ("csd", "image")]
A1 = C.CONFIGS["A1"]


def main():
    t0 = time.time()
    R.assert_inputs(["common.py", "rb_build.py", "rb_features.py", "results/rb_reader_A1.npz", "results/rb_bank_A1_half0.npz",
                     "results/rb_bank_A1_half1.npz", "results/rb_heads_affect.npz", "results/rb_heads_image.npz",
                     "results/rb_heads_csd.npz", "results/rb_halves.npz"])
    H = rb.load_halves(False)
    post = {}
    for h in PARTS:
        hp, _ = rb.load_heads(h, False, H["sha256"])
        post[h] = {"img": hp["img"], "txt": hp["txt"]}
    rz = np.load(R.r1_path("results/rb_reader_A1.npz"))
    cols = np.concatenate([np.arange(6 * A1.index(h), 6 * A1.index(h) + 6) for h in PARTS])
    halves, rec, checks = [], {"halves": {}}, {}
    for j in (0, 1):
        ep, blocks, n, _ = rb.load_bank("A1", j, False, H["sha256"])
        assert blocks == rb.BLOCKS["A1"] and n == 16384
        N = len(ep.anchor)
        # (1) features recomputed with A1c's parts on the whole A1 bank
        F = rf.both_conditions(post, PARTS, ep)
        Xfull = rz[f"half{j}__X"]
        eq_a = bool(np.array_equal(F["a"], Xfull[:N][:, cols]))
        eq_b = bool(np.array_equal(F["b"], Xfull[N:][:, cols]))
        checks[f"half{j}_recomputed_features_equal_A1_reader_features_without_caption"] = eq_a and eq_b
        # (2) keep the three blocks without caption
        keep = [i for i, b in enumerate(blocks) if b in KEEP_BLOCKS]
        assert [blocks[i] for i in keep] == KEEP_BLOCKS
        eidx = np.concatenate([np.arange(i * n, (i + 1) * n) for i in keep])
        Xa, Xb = F["a"][eidx], F["b"][eidx]
        ya, yb = rf.bank_labels(KEEP_BLOCKS, n, PARTS)
        X, y, epi = rf.stack_conditions(Xa, Xb, ya, yb)
        cnt = np.bincount(y, minlength=3)
        assert (cnt == cnt[0]).all(), cnt
        scaler, model, r, oof = rb.fit_half_reader(X, y, epi, len(eidx), 3)
        halves.append({"scaler": scaler, "model": model, "C": r["chosen_C"]})
        rec["halves"][str(j)] = {"chosen_C": r["chosen_C"], "oof_bank_accuracy": r["oof_accuracy_at_chosen_C"],
                                 "n_episodes": r["n_episodes"], "n_examples": r["n_examples"],
                                 "cv_table": [{k: v for k, v in row.items() if k != "fold_log_loss"} for row in r["cv_table"]],
                                 "class_counts": r["class_counts"], "refit_warnings": r["refit_convergence_warnings"]}
        # (3) the features of the kept episodes equal the stored X (without caption) for the same blocks
        rows = np.concatenate([eidx, N + eidx])
        checks[f"half{j}_kept_features_equal_stored"] = bool(np.array_equal(X, Xfull[rows][:, cols]))
        C.log(f"half {j}: C {r['chosen_C']}, oof acc {r['oof_accuracy_at_chosen_C']:.2f} [{time.time()-t0:.0f}s]")
    rec["checks"] = checks
    rec["groupings"] = list(PARTS)
    rec["feature_names"] = rf.feature_names(PARTS)
    print(json.dumps(rec, indent=1))
    with open(HERE / "results/sp_reader_A1c.pkl", "wb") as f:
        pickle.dump({"config": "A1c", "groupings": list(PARTS), "feature_names": rf.feature_names(PARTS), "halves": halves}, f)
    (HERE / "results/sp_reader_A1c.json").write_text(json.dumps(R.C.jsonable(rec), indent=1))


if __name__ == "__main__":
    main()
