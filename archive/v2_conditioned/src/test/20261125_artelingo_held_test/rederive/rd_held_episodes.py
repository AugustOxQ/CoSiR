"""Phase 2, step 1: the held rows and the held episodes (rule §5 items 1 to 3), own code.

Held rows = artelingo_splits(data).held, checked against prepare.npz (groups, split_train, scorer_train, selection)
and held_codes.npz's held_rows (only that key is read); no leakage group shared with train or val; held disjoint from
selection and scorer-train. The development value sets (rule §5 item 2) from the selection pool, each also eligible
on the held pool. Episodes: build_aspect_episodes with this folder's own values restriction (rd_common.build_restricted,
validated in phase 1 on seeds 42, 9001 to 9003), on held rows, seeds 52, 53, 54, 4,096 per pair; validated; every
member a held row; the nine per-pair SHA-256s distinct from each other and from every per-pair SHA-256 in E1's
non-smoke baselines_seed*.json. No score or metric is computed here.

  cd /project/CoSiR && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/rederive/rd_held_episodes.py \
  > src/test/20261125_artelingo_held_test/rederive/out/rd_held_episodes.log 2>&1
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd_common as R  # noqa: E402
import numpy as np  # noqa: E402

from src.data.artelingo_splits import EMOTION_CATCH_ALL, encode_labels  # noqa: E402
from src.data.wikiart_genre import GENRE_NAMES  # noqa: E402
from src.eval.aspect_episodes import (PaintingValueIndex, eligible_values, episodes_sha256,  # noqa: E402
                                      validate_aspect_episodes)

HELD_SEEDS = (52, 53, 54)
N_PER_PAIR = 4096
HELD_CODES = R.T / "20261014_stage_d_final/cache/held_codes.npz"


def held_checks(sp):
    z = np.load(R.need(R.T / "20261013_stage_d_selection/cache/prepare.npz"))
    groups = np.asarray(sp.groups)
    st, sel, val, held = (np.asarray(x) for x in (sp.scorer_train, sp.selection, sp.val, sp.held))
    train = np.sort(np.concatenate([st, sel]))
    out = {"groups_equal_prepare": bool(np.array_equal(groups, z["groups"])),
           "scorer_train_equal_prepare": bool(np.array_equal(st, z["scorer_train"])),
           "selection_equal_prepare": bool(np.array_equal(sel, z["selection"])),
           "train_equal_prepare_split_train": bool(np.array_equal(train, z["split_train"])),
           "train_val_held_partition_all_rows": bool(
               np.array_equal(np.sort(np.concatenate([train, val, held])), np.arange(len(groups))))}
    hz = np.load(HELD_CODES)  # only the held_rows key is read
    out["held_equal_held_codes"] = bool(np.array_equal(held, hz["held_rows"]))
    out["held_codes_sha256"] = R.sha_file(HELD_CODES)
    hg = set(np.unique(groups[held]).tolist())
    out["held_groups_disjoint_train"] = not (hg & set(np.unique(groups[train]).tolist()))
    out["held_groups_disjoint_val"] = not (hg & set(np.unique(groups[val]).tolist()))
    out["held_disjoint_selection"] = not np.intersect1d(held, sel).size
    out["held_disjoint_scorer_train"] = not np.intersect1d(held, st).size
    out["n_held"] = int(len(held))
    out["held_sorted"] = bool((np.diff(held) > 0).all())
    out["all_ok"] = all(v for k, v in out.items() if isinstance(v, bool))
    return out


def main():
    rec = {"time_start": R.now_ams()}
    data, sp, labels = R.load_data()
    groups, sel, held = np.asarray(sp.groups), np.asarray(sp.selection), np.asarray(sp.held)
    rec["held_checks"] = hc = held_checks(sp)
    if not hc["all_ok"]:
        raise AssertionError(f"held split checks failed: {hc}")
    R.log(f"held rows {hc['n_held']}, checks ok")
    in_held = np.zeros(len(groups), dtype=bool)
    in_held[held] = True

    # value sets (rule §5 item 2)
    V = R.dev_value_sets(labels, groups, sel)
    p1 = json.loads((R.RD / "rd_episodes.json").read_text())["value_sets"]
    names = {"emotion": encode_labels(data.emotions, exclude=(EMOTION_CATCH_ALL,))[1],
             "style": encode_labels(data.art_styles)[1], "genre": list(GENRE_NAMES)}
    sizes = {k: len(v) for k, v in V.items()}
    pool_held = R.pool_rows(labels, held)
    elig_held = {x: eligible_values(np.asarray(labels[x], dtype=np.int64), groups, pool_held, 30) for x in V}
    rec["value_sets"] = {k: {"codes": V[k], "names": [names[k][c] for c in V[k]]} for k in V}
    rec["value_sets_equal_phase1"] = all(V[k] == p1[k]["codes"] for k in V)
    rec["value_set_sizes_ok"] = sizes == {"emotion": 8, "style": 23, "genre": 10}
    rec["held_pool_rows"] = int(len(pool_held))
    rec["dev_values_eligible_on_held"] = {x: set(V[x]) <= set(elig_held[x]) for x in V}
    rec["held_eligible_outside_dev"] = {x: [names[x][c] for c in sorted(set(elig_held[x]) - set(V[x]))] for x in V}
    if not (rec["value_sets_equal_phase1"] and rec["value_set_sizes_ok"]
            and all(rec["dev_values_eligible_on_held"].values())):
        raise AssertionError(f"value sets: {rec}")
    R.log(f"value sets {sizes}; held-eligible outside dev {rec['held_eligible_outside_dev']}")

    # episodes (rule §5 item 3)
    index = PaintingValueIndex(labels, groups)
    saved, shas = {}, {}
    rec["seeds"] = {}
    for s in HELD_SEEDS:
        shas[s] = {}
        for a, b, third in R.PAIRS:
            ep = R.build_restricted(labels, groups, held, a, b, N_PER_PAIR, s, third, index, (V[a], V[b]))
            validate_aspect_episodes(ep, labels, groups, index, third=third)
            if len(ep.anchor) != N_PER_PAIR:
                raise AssertionError("episode count")
            if not in_held[ep.rows()].all():
                raise AssertionError(f"seed {s} {a}__{b}: a member outside held rows")
            la, lb = labels[a], labels[b]
            if not (np.isin(la[ep.anchor], V[a]).all() and np.isin(lb[ep.anchor], V[b]).all()):
                raise AssertionError("an anchor value outside the development set")
            if not (np.isin(la[ep.pairs_a_img], V[a]).all() and np.isin(lb[ep.pairs_b_img], V[b]).all()):
                raise AssertionError("a shared example-pair value outside the development set")
            shas[s][f"{a}__{b}"] = episodes_sha256(ep)
            for f in R.FIELDS:
                saved[f"seed{s}__{a}__{b}__{f}"] = getattr(ep, f)
        rec["seeds"][str(s)] = {"n_per_pair": N_PER_PAIR, "episodes_sha256": shas[s]}
        R.log(f"seed {s} built and validated")

    # distinctness (rule §5 item 3)
    nine = [h for s in HELD_SEEDS for h in shas[s].values()]
    recorded = {}
    for p in sorted(R.E1.glob("baselines_seed*.json")):
        for k, v in json.loads(p.read_text())["episodes_sha256"].items():
            recorded[f"{p.name}/{k}"] = v
    rec["distinct_nine"] = len(set(nine)) == 9
    rec["n_recorded_e1_hashes"] = len(recorded)
    rec["e1_files"] = sorted({k.split("/")[0] for k in recorded})
    rec["none_in_e1"] = not (set(nine) & set(recorded.values()))
    if not (rec["distinct_nine"] and rec["none_in_e1"]):
        raise AssertionError("episode hashes not distinct")
    np.savez(R.OUT / "rd_held_episodes.npz", pair_order=np.array(R.POOLED), seeds=np.array(HELD_SEEDS), **saved)
    rec["episodes_npz_sha256"] = R.sha_file(R.OUT / "rd_held_episodes.npz")
    rec["time_end"] = R.now_ams()
    R.write_json(R.RD / "rd_held_episodes.json", rec)
    R.log(f"nine hashes distinct {rec['distinct_nine']}, none in E1 ({len(recorded)} recorded) {rec['none_in_e1']}")


if __name__ == "__main__":
    main()
