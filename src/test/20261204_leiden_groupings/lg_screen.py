"""EXPLORATORY side check outside the research loop (decides nothing; seed 42 development episodes only).
Handoff: docs/superpowers/handoffs/2026-10-14-leiden-groupings-handoff.md. Two cheap screens on round 3's told-oracle
screen (src/test/20261116_grouping_step1_style/run_step1.py, reused by import, nothing there modified):

  (1') image_leiden64, caption_leiden64: run_step1's Leiden graphs (image at painting level, caption at row level, the
       k 20 kNN union graph), RBConfiguration at the resolution that gives about 64 groups after the 200-row merge
       (k-means' count). Search (agent default): r = 1.0; then doubling until K(r) >= 64; then up to 8 bisections of
       the bracket, stopping at K = 64. The evaluated r with the smallest |K - 64| wins, ties to the smaller r.
  (2)  h1: the attention-H1 student's 19 Leiden communities (train_community_post of the buddy-line snapshot), mapped
       to the scorer-train rows by painting name. The only grouping; every aspect is told to it (user, 2026-10-14).

Arms: A0 (affect, image, caption: the reference, asserted to reproduce run_step1's stated A0 margins), A3r (affect,
image_leiden64, caption_leiden64, told as run_step1's A3) and H1 (h1 alone). The H1 student saw the held paintings
(features only, no labels): its labels feed development episodes only, never a held read.

Run (from /project/CoSiR), CPU only:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261204_leiden_groupings/lg_screen.py {group,heads,eval}
    -> results/lg_<stage>.json (+ .npz); the .txt summary of eval is results/lg_eval.txt.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import leidenalg
import numpy as np
from sklearn.exceptions import ConvergenceWarning

HERE = Path(__file__).resolve().parent
STEP1 = HERE.parent / "20261116_grouping_step1_style"
sys.path.insert(0, str(STEP1))
import run_step1 as rs1  # noqa: E402

rto, rsw, df, rc, rg, n6 = rs1.rto, rs1.rsw, rs1.df, rs1.rc, rs1.rg, rs1.n6
from src.eval.aspect_metrics import CONDITIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_quick_checks import centered_term, crossfit_condition_free  # noqa: E402

RES = HERE / "results"
H1_NPZ = Path("/project/CoSiR-buddy_prototype_conditioning/src/test/20260923_artelingo_buddy_analysis/"
              "attention_h1_embedding_snapshot.npz")
STEP1_GROUP = STEP1 / "results" / "step1_group_descriptive.json"
TARGET, MAX_BISECT, R_BOUNDS = 64, 8, (1 / 64, 64.0)
NEW = ("image_leiden64", "caption_leiden64", "h1")
ARMS = {"A0": {"parts": rs1.BASE, "told": {"emotion": "affect", "style": "image", "genre": "image"}},
        "A3r": {"parts": ("affect", "image_leiden64", "caption_leiden64"),
                "told": {"emotion": "affect", "style": "image_leiden64", "genre": "image_leiden64"}},
        "H1": {"parts": ("h1",), "told": {"emotion": "h1", "style": "h1", "genre": "h1"}}}


def jdump(path, obj):
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else o.tolist()))


def compact(x):
    return np.unique(np.asarray(x, dtype=np.int64), return_inverse=True)[1].astype(np.int64)


def search(name, graph, node_of_row, X_rows):
    """K(r) = groups after merge_small at 200 rows (centroids in X_rows' space); the search of the module docstring."""
    cache, records = {}, []

    def K(r):
        if r not in cache:
            t = time.time()
            p = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=r,
                                         seed=rsw.LEIDEN_SEED)
            raw = compact(p.membership)
            local, moves = rto.merge_small(raw[node_of_row], X_rows, rs1.MIN_ROWS)
            cache[r] = (local, raw, moves)
            records.append({"r": r, "K": int(local.max()) + 1, "n_raw": int(raw.max()) + 1, "s": round(time.time() - t, 1)})
            rc.log(f"{name}: r {r:.4f} -> {int(raw.max()) + 1} communities, {records[-1]['K']} after merging "
                   f"[{records[-1]['s']:.0f}s]")
        return int(cache[r][0].max()) + 1

    r, below = 1.0, K(1.0) < TARGET
    if K(1.0) != TARGET:
        prev = r
        while K(r) != TARGET and (K(r) < TARGET) == below and R_BOUNDS[0] < r < R_BOUNDS[1]:
            prev, r = r, r * (2.0 if below else 0.5)
        lo, hi = sorted((prev, r))
        if K(lo) < TARGET < K(hi):
            for _ in range(MAX_BISECT):
                mid = 0.5 * (lo + hi)
                if K(mid) == TARGET:
                    break
                lo, hi = (mid, hi) if K(mid) < TARGET else (lo, mid)
    best = min(cache, key=lambda x: (abs(K(x) - TARGET), x))
    local, raw, moves = cache[best]
    info = {"chosen_r": best, "K": K(best), "search": records, **rs1.merged_info(raw[node_of_row], local, moves)}
    return info, local


def stage_group():
    t0 = time.time()
    data, sp, scorer_train, groups, g_st = rs1.base_data()
    out, npz = {}, {"scorer_train": scorer_train, "g_st": g_st}

    # (1') image: painting-level graph exactly as run_step1.painting_grouping (first row's unit vector per painting)
    first = rs1.first_rows(g_st)
    Xp = rc.unit(np.asarray(data.img_features[scorer_train][first], dtype=np.float32))
    graph = rsw.knn_union_graph(Xp, rs1.GRAPH_K)
    out["image_leiden64"], npz["image_leiden64"] = search("image_leiden64", graph, g_st, Xp[g_st])
    # r = 1.0 must give run_step1's stored image_leiden (same graph, Leiden call, seed and merge)
    stored = json.loads(STEP1_GROUP.read_text())["groupings"]["image_leiden"]["partition_sha256"]
    p1 =leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=1.0,
                                  seed=rsw.LEIDEN_SEED)
    l1, _ = rto.merge_small(compact(p1.membership)[g_st], Xp[g_st], rs1.MIN_ROWS)
    out["image_leiden64"]["r1_equals_step1_image_leiden"] = bool(rg.sha_array(l1) == stored)
    if not out["image_leiden64"]["r1_equals_step1_image_leiden"]:
        raise AssertionError("image graph at r = 1.0 does not reproduce run_step1's image_leiden")
    del graph, Xp, l1, p1

    # (1') caption: row-level graph on unit CLIP caption features, as run_step1.caption_grouping
    X = rc.unit(np.asarray(data.txt_features[scorer_train], dtype=np.float32))
    t = time.time()
    graph = rsw.knn_union_graph(X, rs1.GRAPH_K)
    rc.log(f"caption graph built [{time.time() - t:.0f}s]")
    out["caption_leiden64"], npz["caption_leiden64"] = search("caption_leiden64", graph, np.arange(len(X)), X)
    del graph, X

    # (2) h1: painting name -> train_community_post
    z = np.load(H1_NPZ, allow_pickle=True)
    idx = {n: i for i, n in enumerate(z["train_paintings"].astype(str))}
    names = np.asarray(data.paintings)[scorer_train].astype(str)
    missing = sorted(set(names) - set(idx))
    if missing:
        raise AssertionError(f"h1: {len(missing)} scorer-train paintings are not in the snapshot, e.g. {missing[:3]}")
    raw = np.asarray(z["train_community_post"], dtype=np.int64)[[idx[n] for n in names]]
    local = compact(raw)
    if not np.array_equal(local, local[first][g_st]):
        raise AssertionError("h1: labels are not constant within a painting")
    sizes = np.bincount(local)
    if sizes.min() < rs1.MIN_ROWS:
        raise AssertionError(f"h1: a group has {sizes.min()} rows (< {rs1.MIN_ROWS}); a merge rule is needed")
    out["h1"] = {"source": str(H1_NPZ), "source_sha256": rg.sha_file(H1_NPZ), "key": "train_community_post",
                 "snapshot_paintings": int(len(idx)), "scorer_train_paintings": int(len(set(names))),
                 "groups": rto.group_stats(local), "partition_sha256": rg.sha_array(local)}
    npz["h1"] = local
    rc.log(f"h1: {out['h1']['groups']['n_groups']} groups on {len(set(names))} scorer-train paintings "
           f"(min {sizes.min()} rows)")
    RES.mkdir(exist_ok=True)
    jdump(RES / "lg_group.json", {"groupings": out, "runtime_s": round(time.time() - t0, 1)})
    np.savez(RES / "lg_group.npz", **npz)


def stage_heads():
    t0 = time.time()
    gz = np.load(RES / "lg_group.npz")
    ctx = rg.EvalContext(df.SEED, False)
    scorer_train = np.asarray(rs1.artelingo_splits(ctx.data).scorer_train)
    if not np.array_equal(scorer_train, gz["scorer_train"]):
        raise AssertionError("scorer_train differs from the group stage's")
    sel = ctx.selection
    heads, npz = {}, {"selection": sel}
    for name in NEW:
        t = time.time()
        local = np.asarray(gz[name], dtype=np.int64)
        lab = rto.global_labels(local, scorer_train, len(ctx.groups))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            post, prov = rto.fit_one_head(ctx, lab, scorer_train, n6.HEAD_ROWS)
        prov["convergence_warnings"] = int(sum(issubclass(w.category, ConvergenceWarning) for w in caught))
        prov["n_groups"] = int(local.max()) + 1
        if prov["n_classes"] != prov["n_groups"]:
            raise AssertionError(f"{name}: a group is missing from the 60,000-row draw")
        npz[f"{name}__img"], npz[f"{name}__txt"] = post["img"][sel], post["txt"][sel]
        heads[name] = prov
        rc.log(f"{name}: heads held-out {prov['heldout_accuracy']['img']:.2f}/{prov['heldout_accuracy']['txt']:.2f} "
               f"(majority {prov['check_majority_share']:.2f}) [{time.time() - t:.0f}s]")
    jdump(RES / "lg_heads.json", {"heads": heads, "runtime_s": round(time.time() - t0, 1)})
    np.savez(RES / "lg_heads.npz", **npz)


def stage_eval():
    t0 = time.time()
    hz = np.load(RES / "lg_heads.npz")
    S = rsw.setup()                 # B = C2, stored posteriors; reproduces the stored numbers (asserted inside)
    ctx, cl = S.ctx, S.cl
    if not np.array_equal(hz["selection"], ctx.selection):
        raise AssertionError("heads were computed on another selection row set")
    inp, _, _ = rc.model_inputs(ctx, "A3", S.scorer_train, False)
    t_n1u = centered_term(inp, ctx.pooled, uniform=True)
    del inp
    post_L, _ = rto.fit_one_head(ctx, rto.global_labels(S.partition_L, S.scorer_train, len(ctx.groups)),
                                 S.scorer_train, n6.HEAD_ROWS)
    post = {"affect": post_L, "image": S.post_stored["image"], "caption": S.post_stored["caption"],
            **{n: {"img": rs1.full_post(hz, f"{n}__img", ctx), "txt": rs1.full_post(hz, f"{n}__txt", ctx)} for n in NEW}}
    arms, refs, lines = {}, {}, []
    for arm, spec in ARMS.items():
        t = time.time()
        rc.log(f"arm {arm}: groupings {spec['parts']}, told {spec['told']}")
        p = {h: post[h] for h in spec["parts"]}
        ev, arr, t6u, picked, part_idx = rs1.evaluate_general(p, spec["parts"], spec["told"], ctx, S.B, S.pB)
        Bp, _ = crossfit_condition_free(ctx.cos, t_n1u, t6u, ctx.parity)
        bar_v, bar_info = rs1.bar_margin(arr, per_anchor(Bp), ctx)
        entry = {"groupings": list(spec["parts"]), "told_mapping": spec["told"], "eval": ev, "bar": bar_info}
        if arm == "A0":             # the plumbing check: run_step1's stated A0 margins
            got = {nm: tuple(round(v, 2) for v in (ev[nm]["fusedT_vs_fusedTcf"]["r1"]["point"],
                                                   *ev[nm]["fusedT_vs_fusedTcf"]["r1"]["ci95"])) for nm in rs1.NAMES}
            if got != {nm: tuple(v) for nm, v in rs1.L_STATED.items()}:
                raise SystemExit(f"A0 does not reproduce run_step1's stated margins: {got} vs {rs1.L_STATED}")
            entry["reproduces_step1_A0"] = True
        else:
            entry["vs_A0"] = rs1.paired(arr, refs["A0"][0], bar_v, refs["A0"][1], ctx)
        if arm == "H1":             # one grouping: the reader can only pick it, so reader = told
            same = all(np.array_equal(arr["told"][j][m], arr["reader"][j][m]) for j in (0, 1) for m in METRICS)
            if not same:
                raise AssertionError("H1: reader and told terms differ with a single grouping")
        refs[arm] = (arr, bar_v)
        entry["runtime_s"] = round(time.time() - t, 1)
        arms[arm] = entry
        m = {nm: ev[nm]["fusedT_vs_fusedTcf"]["r1"] for nm in rs1.NAMES}
        line = (f"{arm}: told {rs1.ci_txt(m['told'])}  reader {rs1.ci_txt(m['reader'])}  bar {rs1.ci_txt(bar_info['r1'])}"
                f" ({bar_info['comparator']})  pick {ev['pick']['correct_share']['point']:.1f}")
        if arm != "A0":
            d = entry["vs_A0"]
            line += f"  | minus A0: told {rs1.ci_txt(d['told']['r1'])} reader {rs1.ci_txt(d['reader']['r1'])}"
        lines.append(line)
        rc.log(line + f" [{entry['runtime_s']:.0f}s]")
    result = {"n_episodes": int(ctx.n), "B": df.describe(S.pB, cl), "arms": arms, "runtime_s": round(time.time() - t0, 1)}
    jdump(RES / "lg_eval.json", result)
    (RES / "lg_eval.txt").write_text("R@1 margins over the condition-free counterpart, points [95% CI]\n"
                                     + "\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("group", "heads", "eval"))
    {"group": stage_group, "heads": stage_heads, "eval": stage_eval}[ap.parse_args().stage]()
