#!/usr/bin/env python3
"""Aggregate the raw pilot snapshots into small JSON files for the 2026-09-30 weekly figures.

The snapshots are untracked files in the percept worktree
(/project/CoSiR-buddy_prototype_conditioning, branch experiment/percept_topic_pipeline).
This script reads them once and writes only counts to
docs/reports/assets/2026-09-30_weekly/data/, which the figure script then plots.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

PERCEPT_TEST = Path("/project/CoSiR-buddy_prototype_conditioning/src/test")
BUDDY_DIR = PERCEPT_TEST / "20260923_artelingo_buddy_analysis"
PERCEPT_DIR = PERCEPT_TEST / "20260922_percept_topic_pipeline"
OUT = Path(__file__).resolve().parent / "2026-09-30_weekly" / "data"

EMOTIONS = ["amusement", "awe", "contentment", "excitement",
            "anger", "disgust", "fear", "sadness", "something else"]
GENRES = ["landscape", "portrait", "genre_painting", "religious_painting",
          "abstract_painting", "cityscape", "sketch_and_study", "still_life", "illustration"]


def crosstab(communities, labels, vocab):
    comms = sorted(set(int(c) for c in communities))
    table = {c: Counter() for c in comms}
    for c, lab in zip(communities, labels):
        if lab in vocab:
            table[int(c)][lab] += 1
    return {"communities": comms,
            "counts": [[table[c][v] for v in vocab] for c in comms]}


def knn_vote(train_emb, train_lab, query_emb, k=20, batch=512):
    """Cosine k-NN majority vote; a tie goes to the tied label met first (closest)."""
    tr = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True)
    qu = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    out = np.empty(len(qu), dtype=train_lab.dtype)
    for s in range(0, len(qu), batch):
        sims = qu[s:s + batch] @ tr.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.take_along_axis(sims, top, 1).argsort(axis=1)[:, ::-1]
        top = np.take_along_axis(top, order, 1)
        for i, row in enumerate(top):
            labs = train_lab[row]
            cnt = Counter(labs.tolist())
            best = max(cnt.values())
            out[s + i] = next(l for l in labs if cnt[l] == best)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Content-only buddy graph (CLIP image + text, K = 20), Leiden, train split.
    table = json.load(open(BUDDY_DIR / "painting_community_table.json"))
    content = {
        "emotion": crosstab([r["community_id"] for r in table],
                            [r["majority_emotion"] for r in table], EMOTIONS),
        "genre": crosstab([r["community_id"] for r in table],
                          [r["genre"] for r in table], GENRES),
    }

    # Attention-h1 student, seed 42, after training (train split, Leiden on z).
    snap = np.load(BUDDY_DIR / "attention_h1_embedding_snapshot.npz", allow_pickle=True)
    assert list(snap["train_paintings"][:50]) == [r["painting"] for r in table[:50]]
    attn = {
        "emotion": crosstab(snap["train_community_post"], snap["train_emotion"], EMOTIONS),
        "genre": crosstab(snap["train_community_post"], snap["train_genre"], GENRES),
    }

    # Held-out occupancy: buddy topics via k-NN vote (k = 20) onto the 19 train topics.
    train_comm = snap["train_community_post"].astype(int)
    held = knn_vote(snap["train_embedding_post"].astype(np.float32), train_comm,
                    snap["heldout_embedding_post"].astype(np.float32))
    n_train_topics = int(train_comm.max()) + 1
    buddy_occ = np.bincount(held, minlength=n_train_topics).tolist()

    # Held-out occupancy: PercepT paper recipe (K = 100/67, seed 42), nearest surviving centre.
    pz = np.load(PERCEPT_DIR / "percept_stage1_faithful_recipe_snapshot.npz", allow_pickle=True)
    percept_occ = np.bincount(pz["heldout_topic"].astype(int),
                              minlength=int(pz["n_surviving_clusters"])).tolist()

    data = {
        "emotions": EMOTIONS, "genres": GENRES,
        "content_only": content, "attention_h1": attn,
        "occupancy": {"buddy_knn_vote": buddy_occ, "percept_paper_recipe": percept_occ,
                      "n_heldout": int(len(held))},
    }
    (OUT / "community_composition_and_occupancy.json").write_text(json.dumps(data))
    for name, occ in (("buddy", buddy_occ), ("percept", percept_occ)):
        occ = np.array(occ)
        print(name, "topics", len(occ), "empty", int((occ == 0).sum()),
              "below 1%", int((occ < 0.01 * occ.sum()).sum()), "median", float(np.median(occ)))


STAGE_D_SEL = Path("/project/CoSiR/src/test/20261013_stage_d_selection/results")
STAGE_D_FIN = Path("/project/CoSiR/src/test/20261014_stage_d_final/results")


def prep_stage_d():
    """Copy the stage (d) aggregates the figures need (source files are gitignored)."""
    sel = json.load(open(STAGE_D_SEL / "selection_results.json"))
    post = json.load(open(STAGE_D_SEL / "posthoc_results.json"))
    fin = json.load(open(STAGE_D_FIN / "final_results.json"))
    runs = ["G1", "G2", "G3", "G4", "G5"]
    hist = {r: json.load(open(STAGE_D_SEL / f"history_{r}.json"))["history"] for r in runs}
    out = {
        "runs": runs,
        "sources": {"G1": "factor combinations", "G2": "factor combinations + swap",
                    "G3": "CLIP clusters", "G4": "CLIP clusters + swap",
                    "G5": "Block 1 communities"},
        "beta_history": {r: {"step": hist[r]["step"], "beta": hist[r]["beta"]} for r in runs},
        "decomposition": {r: {k: post["decomposition"][r][k]["pooled"]["mean"]
                              for k in ("total_vs_naive0.3", "beta_drop", "beyond_beta_drop")}
                          for r in runs},
        "mechanism_clip_cluster": post["mechanism"]["sources"]["clip_cluster"],
        "oracle_r1": {k: post["ceilings"]["recall_r1"][k]["pooled"]
                      for k in ("naive@0.3", "ceiling", "ceiling_null", "label_oracle")},
        "selection_r1": {m: sel["models"][m]["recall"]["pooled"] for m in ["naive"] + runs},
        "clip_only_r1": sel["baselines"]["clip_only"]["pooled"],
        "swap_selection": {k: post["swap"]["success"][k]
                           for k in ("naive@0.3", "naive@G3beta", "G3")},
        "criterion1": {k: v for k, v in fin["criterion1"]["by_seed"].items()},
        "criterion2_rates": fin["criterion2"]["success_rates"],
        "criterion2": {k: v["pooled"] for k, v in fin["criterion2"]["by_seed"].items()},
    }
    (OUT / "stage_d.json").write_text(json.dumps(out))
    print("stage_d.json written; G3 beta", out["beta_history"]["G3"]["beta"][-1])


H2H_RUNS = PERCEPT_TEST / "20260930_matched_h2h" / "sweep_runs.json"
V2_TEST = Path("/project/CoSiR/src/test")


def prep_h2h():
    """Final matched head-to-head: every finished sweep trial (cell, Stage 1 implementation,
    val AUC, val independent emotion AMI) and the test runs of the eight cell winners."""
    rows = json.load(open(H2H_RUNS))
    trials = []
    for r in rows:
        if r["state"] != "finished":
            continue
        sm = r["summary"]
        try:
            auc = float(sm.get("auc_primary"))
            emo = float(sm.get("ind_emo"))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(auc) and np.isfinite(emo)) or int(sm.get("k_miss") or 0):
            continue
        trials.append({"id": r["id"], "cell": r["cell"], "impl": r["config"].get("buddy_impl", "percept"),
                       "auc": auc, "emo": emo, "genre": float(sm.get("ind_genre", "nan"))})
    seeds = {}
    for path in sorted((H2H_RUNS.parent / "logs").glob("h2h-test-*.log")):
        for line in path.read_text().splitlines():
            if line.startswith("H2H_SEED "):
                row = json.loads(line[len("H2H_SEED "):])
                seeds.setdefault(row["tag"], []).append(row)
    from scipy import stats
    test = {}
    for tag, rs in seeds.items():
        auc = np.array([r["auc_primary"] for r in rs])
        emo = np.array([r["ind_emo"] for r in rs])
        half = stats.t.ppf(0.975, len(auc) - 1) * auc.std(ddof=1) / np.sqrt(len(auc))
        test[tag] = {"run_id": rs[0]["run_id"], "n": len(auc), "auc_mean": float(auc.mean()),
                     "auc_lo": float(auc.mean() - half), "auc_hi": float(auc.mean() + half),
                     "emo_mean": float(emo.mean())}
    (OUT / "h2h_final.json").write_text(json.dumps({"trials": trials, "test": test}))
    from collections import Counter
    print("h2h trials", len(trials), Counter((o["cell"], o["impl"]) for o in trials))
    print("h2h test tags", sorted(test))


def _corr_and_spectrum(codes):
    codes = np.asarray(codes, dtype=np.float64)
    std = codes.std(0)
    keep = std > 1e-9
    c = np.corrcoef(codes[:, keep], rowvar=False)
    cov = np.cov(codes, rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(cov)[::-1], 0, None)
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    return c, (ev / ev.sum()).tolist(), pr


def prep_factors():
    """Pair-code correlation and variance spectrum: collapsed R0 recipe vs repaired R3."""
    cache = V2_TEST / "20261007_naive_rule_mechanism_analysis" / "cache"
    r0 = 0.5 * (np.load(cache / "factor42_img.npy") + np.load(cache / "factor42_txt.npy"))
    r3 = np.load(V2_TEST / "20261011_factor_repair_grid" / "results" / "R3_seed42_val_pair_codes.npy")
    out = {}
    for name, codes in (("R0", r0), ("R3", r3)):
        c, spec, pr = _corr_and_spectrum(codes)
        off = np.abs(c[~np.eye(len(c), dtype=bool)])
        out[name] = {"corr": c.tolist(), "spectrum": spec, "pr": pr, "rows": int(len(codes)),
                     "max_abs_r": float(off.max()), "pairs_ge_09": int((off >= 0.9).sum() // 2)}
        print(name, "rows", len(codes), "PR", round(pr, 2), "PC1", round(spec[0], 3),
              "max|r|", round(float(off.max()), 3), "pairs>=.9", int((off >= 0.9).sum() // 2))
    diag = {}
    for d in ("D0", "D1", "D2", "D3", "D4", "D5", "D6"):
        v = json.load(open(V2_TEST / "20261009_factor_collapse_diagnosis" / "results" / f"{d}.json"))
        vals = v["values"]
        diag[d] = {"pr": min(vals["participation_ratio_img"], vals["participation_ratio_txt"]),
                   "ratio": vals["retrieval_ratio"], "max_abs_r": vals["correlation"]["max_abs"]}
    out["diagnosis"] = diag
    grid = {}
    for r in ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"):
        v = json.load(open(V2_TEST / "20261011_factor_repair_grid" / "results" / f"{r}_seed42.json"))
        grid[r] = {"passed": v["passed"], "score": v.get("selection_score"),
                   "gates_passed": v.get("gates_passed")}
    am = json.load(open(V2_TEST / "20261011_factor_repair_grid" / "results" / "amended_summary.json"))
    out["grid"] = grid
    out["amended"] = am["regated"]
    (OUT / "factors.json").write_text(json.dumps(out))
    print("diag", {k: round(v["pr"], 2) for k, v in diag.items()})


if __name__ == "__main__":
    import sys
    jobs = {"main": main, "stage_d": prep_stage_d, "h2h": prep_h2h, "factors": prep_factors}
    for name in (sys.argv[1:] or list(jobs)):
        jobs[name]()
