"""Phase 1, step 2: the development value sets (rule §5 item 2) and the episode identity check (rule §6 item 4, last
bullet): build_aspect_episodes with this folder's own values restriction, on selection rows, seeds 42 (4,096 per pair)
and 9001 to 9003 (64 per pair), against the recorded per-pair episodes_sha256. Selection rows only; no held row.

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/rederive/rd_episodes.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd_common as R  # noqa: E402
import numpy as np  # noqa: E402

from src.data.artelingo_splits import EMOTION_CATCH_ALL, encode_labels  # noqa: E402
from src.data.wikiart_genre import GENRE_NAMES  # noqa: E402
from src.eval.aspect_episodes import PaintingValueIndex, episodes_sha256, validate_aspect_episodes  # noqa: E402

SEEDS = {42: 4096, 9001: 64, 9002: 64, 9003: 64}


def main():
    data, sp, labels = R.load_data()
    groups, sel = np.asarray(sp.groups), np.asarray(sp.selection)
    in_sel = np.zeros(len(groups), dtype=bool)
    in_sel[sel] = True
    V = R.dev_value_sets(labels, groups, sel)
    names = {"emotion": encode_labels(data.emotions, exclude=(EMOTION_CATCH_ALL,))[1],
             "style": encode_labels(data.art_styles)[1], "genre": list(GENRE_NAMES)}
    sizes = {k: len(v) for k, v in V.items()}
    rec = {"value_sets": {k: {"codes": V[k], "names": [names[k][c] for c in V[k]]} for k in V},
           "value_set_sizes": sizes, "value_set_sizes_ok": sizes == {"emotion": 8, "style": 23, "genre": 10},
           "pool_sel_rows": int(len(R.pool_rows(labels, sel))), "seeds": {}}
    R.log(f"value sets {sizes}")
    index = PaintingValueIndex(labels, groups)
    saved = {}
    for seed, n in SEEDS.items():
        path = R.E1 / ("baselines_seed42.json" if seed == 42 else f"smoke/baselines_seed{seed}.json")
        if seed == 42:
            R.need(path)
        recorded = json.loads(path.read_text())["episodes_sha256"]
        mine = {}
        for a, b, third in R.PAIRS:
            ep = R.build_restricted(labels, groups, sel, a, b, n, seed, third, index, (V[a], V[b]))
            validate_aspect_episodes(ep, labels, groups, index, third=third)
            if not in_sel[ep.rows()].all():
                raise AssertionError(f"seed {seed} {a}__{b}: a row outside selection")
            if len(ep.anchor) != n:
                raise AssertionError("episode count")
            mine[f"{a}__{b}"] = episodes_sha256(ep)
            if seed == 42:
                for f in R.FIELDS:
                    saved[f"{a}__{b}__{f}"] = getattr(ep, f)
        rec["seeds"][str(seed)] = {"n_per_pair": n, "sha256": mine, "recorded": recorded,
                                   "equal": {k: mine[k] == recorded[k] for k in mine}}
        R.log(f"seed {seed}: equal {rec['seeds'][str(seed)]['equal']}")
    np.savez(R.OUT / "rd_episodes_seed42.npz", pair_order=np.array(R.POOLED), **saved)
    stored = np.load(R.need(R.E1 / "episodes_seed42.npz"))
    rec["seed42_arrays_equal_episodes_seed42_npz"] = bool(
        list(stored["pair_order"]) == R.POOLED and all(np.array_equal(stored[k], saved[k]) for k in saved)
        and set(stored.files) == set(saved) | {"pair_order"})
    rec["all_equal"] = all(all(s["equal"].values()) for s in rec["seeds"].values()) and \
        rec["seed42_arrays_equal_episodes_seed42_npz"] and rec["value_set_sizes_ok"]
    rec["time"] = R.now_ams()
    R.write_json(R.RD / "rd_episodes.json", rec)
    R.log(f"all equal {rec['all_equal']}")


if __name__ == "__main__":
    main()
