"""Independent C13 scoring; reads only the explicitly named NPZ/NPY inputs."""
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_info

ROOT = Path("/tmp/claude-0/-project-CoSiR/724fc430-8911-4bb5-ba46-fe27b9e6c38a/scratchpad/rederive3")
SAMPLE = Path("/project/CoSiR/src/test/20261017_prompted_aspect_views/sample.npz")
STAGE2 = Path("/project/CoSiR/src/test/20261018_percept_core/outputs/stage2")
STAGE1 = Path("/project/CoSiR/src/test/20261018_percept_core/outputs/stage1")
ASPECTS = ("emotion", "style", "genre")
ARMS = tuple(f"P{i}" for i in range(1, 6))
VERSIONS = ("clip_roberta", "qwen_qwen", "qwen_roberta")
SEEDS = (0, 1, 2)
CHECKS = []
PROBLEMS = []
START = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - START:.1f}s] {message}", flush=True)


def check(name, condition, observed=None):
    ok = bool(condition)
    CHECKS.append({"check": name, "ok": ok, "observed": observed})
    if not ok:
        PROBLEMS.append(f"{name}: {observed}")
        log(f"PROBLEM {name}: {observed}")
    return ok


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=0))}


def ami(pred, labels):
    return {
        aspect: float(adjusted_mutual_info_score(y[y >= 0], pred[y >= 0]))
        for aspect, y in labels.items()
    }


def describe(a):
    return {"shape": list(a.shape), "dtype": str(a.dtype)}


def load_labels():
    with np.load(SAMPLE, allow_pickle=False) as data:
        sample = {key: data[key] for key in (
            "sample_ids", "pid", "painting_ids", "emo", "sty", "gen",
            "first", "all_painting_ids", "subset_pid"
        )}
    for key in ("sample_ids", "pid", "painting_ids", "emo", "sty", "gen"):
        check(f"sample {key} shape", sample[key].shape == (99999,), describe(sample[key]))
    for key in ("first", "all_painting_ids"):
        check(f"sample {key} shape", sample[key].shape == (19889,), describe(sample[key]))
    check("sample subset_pid shape", sample["subset_pid"].shape == (5000,), describe(sample["subset_pid"]))
    check("sample sample_ids int64", sample["sample_ids"].dtype == np.dtype("int64"), str(sample["sample_ids"].dtype))
    for key in ("pid", "emo", "sty", "gen", "first", "subset_pid"):
        check(f"sample {key} integer", np.issubdtype(sample[key].dtype, np.integer), str(sample[key].dtype))
    pid, emo, first, subset = (sample[k] for k in ("pid", "emo", "first", "subset_pid"))
    check("sample pid range", np.all((pid >= 0) & (pid < 19889)), [int(pid.min()), int(pid.max())])
    check("sample first range", np.all((first >= 0) & (first < 99999)), [int(first.min()), int(first.max())])
    check("sample subset_pid range", np.all((subset >= 0) & (subset < 19889)), [int(subset.min()), int(subset.max())])
    check("sample subset_pid unique", np.unique(subset).size == 5000, int(np.unique(subset).size))
    check("sample emotion range", np.all((emo >= -1) & (emo <= 8)), [int(emo.min()), int(emo.max())])
    first_derived = np.full(19889, 99999, dtype=np.int64)
    np.minimum.at(first_derived, pid, np.arange(99999))
    check("sample first is actual first caption", np.array_equal(first, first_derived))
    check("sample painting names agree with pid", np.array_equal(sample["painting_ids"], sample["all_painting_ids"][pid]))
    votes = np.zeros((19889, 9), dtype=np.int64)
    valid = emo >= 0
    np.add.at(votes, (pid[valid], emo[valid]), 1)
    majority = votes.argmax(axis=1)
    majority[votes.sum(axis=1) == 0] = -1
    labels = {"emotion": majority[subset], "style": sample["sty"][first[subset]], "genre": sample["gen"][first[subset]]}
    label_info = {}
    for aspect, y in labels.items():
        ids, counts = np.unique(y[y >= 0], return_counts=True)
        label_info[aspect] = {"valid": int((y >= 0).sum()), "excluded": int((y < 0).sum()),
                              "class_counts": {str(int(i)): int(n) for i, n in zip(ids, counts)}}
        check(f"{aspect} label length", y.shape == (5000,), label_info[aspect])
    tied = (votes == votes.max(axis=1, keepdims=True)).sum(axis=1) > 1
    label_info["emotion_majority"] = {
        "tie_rule": "lowest class id", "evaluated_paintings_with_tied_nonempty_vote": int((tied[subset] & (votes[subset].sum(axis=1) > 0)).sum())
    }
    fit = np.load(ROOT / "fit_rows.npy", allow_pickle=False)
    check("fit_rows one-dimensional int64", fit.ndim == 1 and fit.dtype == np.dtype("int64"), describe(fit))
    check("fit_rows nonempty", fit.size > 0, int(fit.size))
    check("fit_rows strictly ascending", np.all(np.diff(fit) > 0))
    check("fit_rows caption range", np.all((fit >= 0) & (fit < 99999)), [int(fit.min()), int(fit.max())])
    fit_paintings = np.unique(pid[fit])
    check("fit_rows represent 13400 paintings", fit_paintings.size == 13400, int(fit_paintings.size))
    check("fit and evaluation paintings disjoint", np.intersect1d(fit_paintings, subset).size == 0,
          int(np.intersect1d(fit_paintings, subset).size))
    return labels, label_info, fit


def stage2_scores(labels):
    records = []
    absent = []
    for arm in ARMS:
        for seed in SEEDS:
            path = STAGE2 / f"{arm}__s{seed}.npz"
            if not check(f"Stage 2 file exists {path.name}", path.is_file(), str(path)):
                absent.append(str(path))
                continue
            log(f"Stage 2 {path.name}")
            with np.load(path, allow_pickle=False) as data:
                check(f"{path.name} keys", set(data.files) == {"eval_prob"}, sorted(data.files))
                x = data["eval_prob"]
            rec = {"file": str(path), "arm": arm, "training_seed": seed, **describe(x),
                   "K": int(x.shape[1]) if x.ndim == 2 else None}
            check(f"{path.name} shape", x.shape == (5000, 40), list(x.shape))
            check(f"{path.name} float32", x.dtype == np.dtype("float32"), str(x.dtype))
            finite = check(f"{path.name} finite", np.isfinite(x).all())
            rec["min"] = float(x.min())
            rec["max"] = float(x.max())
            check(f"{path.name} values in [0,1]", np.all((x >= 0) & (x <= 1)), [rec["min"], rec["max"]])
            if x.ndim == 2 and x.shape[0] == 5000 and x.shape[1] > 0 and finite:
                x = np.asarray(x, dtype=np.float32)
                rec["argmax_ami"] = ami(x.argmax(axis=1), labels)
                norms = np.linalg.norm(x, axis=1, keepdims=True)
                rec["zero_norm_rows"] = int((norms == 0).sum())
                normalized = np.asarray(x / np.maximum(norms, 1e-12), dtype=np.float32)
                per_seed = []
                for r in SEEDS:
                    pred = MiniBatchKMeans(n_clusters=20, random_state=r, n_init=3, batch_size=4096).fit_predict(normalized)
                    per_seed.append({"kmeans_seed": r, "ami": ami(pred, labels)})
                    log(f"  completed k-means seed {r}")
                rec["kmeans"] = per_seed
                rec["image_side"] = {a: stats([s["ami"][a] for s in per_seed]) for a in ASPECTS}
            records.append(rec)
    check("Stage 2 present file count", len(records) == 15, len(records))
    aggregates = {}
    for arm in ARMS:
        present = [r for r in records if r["arm"] == arm]
        scored = [r for r in present if "image_side" in r]
        if scored:
            aggregates[arm] = {"seeds_present": [r["training_seed"] for r in present],
                               "seeds_scored": [r["training_seed"] for r in scored],
                               "labels": {a: {"image_side": stats([r["image_side"][a]["mean"] for r in scored]),
                                              "argmax": stats([r["argmax_ami"][a] for r in scored])} for a in ASPECTS}}
    comparators = {}
    sets = {"established": [a for a in ARMS if a in aggregates],
            "all_three_seeds": [a for a in ARMS if a in aggregates and aggregates[a]["seeds_present"] == list(SEEDS)]}
    for case, arms in sets.items():
        comparators[case] = {}
        for aspect in ASPECTS:
            if not arms:
                continue
            highest = max(arms, key=lambda a: aggregates[a]["labels"][aspect]["image_side"]["mean"])
            highest_mean = aggregates[highest]["labels"][aspect]["image_side"]["mean"]
            eligible = [a for a in arms if highest_mean - aggregates[a]["labels"][aspect]["image_side"]["mean"] <= 0.003]
            chosen = min(eligible, key=lambda a: int(a[1:]))
            comparators[case][aspect] = {"candidate_arms": arms, "within_0_003": eligible,
                "comparator_arm": chosen, "comparator_mean": aggregates[chosen]["labels"][aspect]["image_side"]["mean"],
                "comparator_sd": aggregates[chosen]["labels"][aspect]["image_side"]["sd"],
                "highest_arm": highest, "highest_mean": highest_mean}
    return {"files": records, "absent_files": absent, "arms": aggregates, "comparators": comparators}


def stage1_scores(labels, fit):
    records, absent = [], []
    for version in VERSIONS:
        for seed in SEEDS:
            path = STAGE1 / f"{version}__s{seed}.npz"
            if not check(f"Stage 1 file exists {path.name}", path.is_file(), str(path)):
                absent.append(str(path))
                continue
            log(f"Stage 1 {path.name}")
            checks_start = len(CHECKS)
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in ("center_norms", "fit_counts", "kept", "topic", "q_paint_eval")}
            expected = {"center_norms": (100,), "fit_counts": (100,), "kept": (100,), "topic": (99999,), "q_paint_eval": (5000, 40)}
            for key, shape in expected.items():
                check(f"{path.name} {key} shape", arrays[key].shape == shape, describe(arrays[key]))
            for key in ("center_norms", "q_paint_eval"):
                check(f"{path.name} {key} floating dtype", np.issubdtype(arrays[key].dtype, np.floating), str(arrays[key].dtype))
            check(f"{path.name} fit_counts int64", arrays["fit_counts"].dtype == np.dtype("int64"), str(arrays["fit_counts"].dtype))
            check(f"{path.name} kept bool", arrays["kept"].dtype == np.dtype("bool"), str(arrays["kept"].dtype))
            check(f"{path.name} topic integer", np.issubdtype(arrays["topic"].dtype, np.integer), str(arrays["topic"].dtype))
            norms, counts, kept, topic, q = (arrays[k] for k in expected)
            check(f"{path.name} center_norms finite/nonnegative", np.all(np.isfinite(norms) & (norms >= 0)))
            check(f"{path.name} fit_counts nonnegative", np.all(counts >= 0))
            check(f"{path.name} kept sum", int(kept.sum()) == 40, int(kept.sum()))
            check(f"{path.name} fit_counts sum", int(counts.sum()) == fit.size, int(counts.sum()))
            order = np.lexsort((np.arange(100), -counts))
            own_kept = np.zeros(100, dtype=bool)
            own_kept[order[:40]] = True
            check(f"{path.name} top40 matches", np.array_equal(kept, own_kept))
            topic_valid = check(f"{path.name} topic range 0..39", np.all((topic >= 0) & (topic < 40)), [int(topic.min()), int(topic.max())])
            q_finite = check(f"{path.name} q_paint_eval finite", np.isfinite(q).all())
            coverage = np.bincount(topic[fit], minlength=40) if topic_valid else None
            rec = {"file": str(path), "version": version, "training_seed": seed,
                   "arrays": {k: describe(v) for k, v in arrays.items()}, "kept_count": int(kept.sum()),
                   "fit_counts_sum": int(counts.sum()), "top40_matches": bool(np.array_equal(kept, own_kept)),
                   "own_kept_center_indices": np.flatnonzero(own_kept).tolist(),
                   "count_40th": int(counts[order[39]]), "count_41st": int(counts[order[40]]),
                   "cut_gap": int(counts[order[39]] - counts[order[40]]),
                   "share_before_cut": float(counts[kept].sum() / fit.size),
                   "with_fit_pair": int((coverage > 0).sum()) if coverage is not None else None,
                   "with_1pct": int((coverage * 100 >= fit.size).sum()) if coverage is not None else None,
                   "topic_fit_counts": coverage.tolist() if coverage is not None else None,
                   "q_min": float(q.min()), "q_max": float(q.max()),
                   "checks_passed": all(c["ok"] for c in CHECKS[checks_start:])}
            if q_finite and q.shape == (5000, 40):
                rec["argmax_ami"] = ami(q.argmax(axis=1), labels)
            records.append(rec)
    check("Stage 1 present file count", len(records) == 9, len(records))
    aggregates = {}
    for version in VERSIONS:
        scored = [r for r in records if r["version"] == version and "argmax_ami" in r]
        if scored:
            aggregates[version] = {"seeds_scored": [r["training_seed"] for r in scored],
                                   "argmax": {a: stats([r["argmax_ami"][a] for r in scored]) for a in ASPECTS}}
    return {"files": records, "absent_files": absent, "versions": aggregates}


def summary(result):
    lines = ["# Independent CoSiR r10 C13 re-derivation, run 3", "",
             "Population standard deviations use ddof=0. Arm image-side statistics summarize each training seed's mean over the three k-means seeds. All displayed scores use six decimals; JSON retains full float precision.", "",
             "## Stage 2: per arm and label", "",
             "| Arm | Label | Seeds | Image-side mean | Image-side sd | Argmax mean | Argmax sd |",
             "|---|---|---:|---:|---:|---:|---:|"]
    s2, s1 = result["stage2"], result["stage1"]
    for arm, rec in s2["arms"].items():
        for a, scores in rec["labels"].items():
            im, arg = scores["image_side"], scores["argmax"]
            lines.append(f"| {arm} | {a} | {im['n']} | {im['mean']:.6f} | {im['sd']:.6f} | {arg['mean']:.6f} | {arg['sd']:.6f} |")
    lines += ["", "## Stage 2: per file", "", "Each label cell is the mean ± population sd over k-means seeds 0, 1, 2.", "",
              "| File | K | Emotion image-side | Style image-side | Genre image-side |", "|---|---:|---:|---:|---:|"]
    for r in s2["files"]:
        cells = [f"{r['image_side'][a]['mean']:.6f} ± {r['image_side'][a]['sd']:.6f}" if "image_side" in r else "unscored" for a in ASPECTS]
        lines.append(f"| {Path(r['file']).name} | {r['K']} | " + " | ".join(cells) + " |")
    for case, title in (("established", "Established arms"), ("all_three_seeds", "All arms with all three seeds present")):
        lines += ["", f"## Comparator: {title}", "", "Select the lowest-numbered arm within 0.003 (inclusive) of the highest mean.", "",
                  "| Aspect | Comparator | Comparator mean | Comparator sd | Highest arm | Highest mean |", "|---|---|---:|---:|---|---:|"]
        for a, r in s2["comparators"][case].items():
            lines.append(f"| {a} | {r['comparator_arm']} | {r['comparator_mean']:.6f} | {r['comparator_sd']:.6f} | {r['highest_arm']} | {r['highest_mean']:.6f} |")
    lines += ["", "## Stage 1: checks and coverage", "", f"Fit-pair denominator: {result['fit_row_count']} caption rows. Coverage uses topic[fit_rows]; the pre-cut share uses stored fit_counts. These come from separate model passes and are not compared as equal.", "",
              "| File | Checks | Kept | Fit count sum | Top 40 match | With fit pair | With ≥1% | Share before cut | 40th | 41st | Gap |",
              "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in s1["files"]:
        lines.append(f"| {Path(r['file']).name} | {'PASS' if r['checks_passed'] else 'FAIL'} | {r['kept_count']} | {r['fit_counts_sum']} | {r['top40_matches']} | {r['with_fit_pair']} | {r['with_1pct']} | {r['share_before_cut']:.6f} | {r['count_40th']} | {r['count_41st']} | {r['cut_gap']} |")
    lines += ["", "## Stage 1: argmax per version", "", "| Version | Label | Seeds | Mean AMI | Population sd |", "|---|---|---:|---:|---:|"]
    for version, r in s1["versions"].items():
        for a, st in r["argmax"].items():
            lines.append(f"| {version} | {a} | {st['n']} | {st['mean']:.6f} | {st['sd']:.6f} |")
    lines += ["", "## Stage 1: argmax per file", "", "| File | Emotion | Style | Genre |", "|---|---:|---:|---:|"]
    for r in s1["files"]:
        if "argmax_ami" in r:
            lines.append(f"| {Path(r['file']).name} | " + " | ".join(f"{r['argmax_ami'][a]:.6f}" for a in ASPECTS) + " |")
    lines += ["", "## Label counts", "", "| Label | Scored | Excluded (<0) | Class counts |", "|---|---:|---:|---|"]
    for a in ASPECTS:
        r = result["labels"][a]
        lines.append(f"| {a} | {r['valid']} | {r['excluded']} | " + ", ".join(f"{k}: {v}" for k, v in r["class_counts"].items()) + " |")
    lines += ["", f"Emotion majority ties among evaluation paintings with nonempty votes: {result['labels']['emotion_majority']['evaluated_paintings_with_tied_nonempty_vote']}; ties select the smallest class id.",
              "", "## Every check", "", "Only exact expected filenames were opened; no pending NPZ or JSON sidecar was read. File counts below refer to expected filenames present, not unrelated directory entries.", ""]
    for c in result["checks"]:
        detail = "" if c["observed"] is None else f"; observed: `{json.dumps(c['observed'], ensure_ascii=False)}`"
        lines.append(f"- {'PASS' if c['ok'] else 'FAIL'}: {c['check']}{detail}")
    lines += ["", "## Problems and absent files", ""]
    lines += [f"- {p}" for p in result["problems"]] or ["- No validation problems."]
    absent = s2["absent_files"] + s1["absent_files"]
    lines += [f"- Absent file: {p}" for p in absent] or ["- No absent files."]
    lines += ["- Nothing deleted. Nothing proposed for removal.", "", "## Reproducibility", "",
              f"Python {result['environment']['python']}; NumPy {result['environment']['numpy']}; scikit-learn {result['environment']['sklearn']}.",
              "OMP_NUM_THREADS=8; MKL_NUM_THREADS=8; PYTHONDONTWRITEBYTECODE=1; CUDA_VISIBLE_DEVICES is empty.",
              "MiniBatchKMeans(n_clusters=20, random_state=r, n_init=3, batch_size=4096), r=0,1,2; other parameters retain installed sklearn defaults. One set of partitions is shared across all labels for each matrix. AMI uses installed sklearn defaults.", ""]
    return "\n".join(lines)


def main():
    log("Loading frozen sample and fit rows")
    labels, label_info, fit = load_labels()
    result = {"rule": "CoSiR r10 C13", "run": 3, "sample_file": str(SAMPLE),
              "fit_rows_file": str(ROOT / "fit_rows.npy"), "fit_row_count": int(fit.size),
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                              "threadpools": threadpool_info(),
                              "env": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES")}},
              "labels": label_info,
              "method": {"kmeans_seeds": list(SEEDS), "kmeans_params": {"n_clusters": 20, "n_init": 3, "batch_size": 4096},
                         "normalization": "float32 x / maximum(row L2 norm, 1e-12)", "sd_ddof": 0,
                         "comparator_tolerance": 0.003, "established_arms": list(ARMS)}}
    result["stage2"] = stage2_scores(labels)
    result["stage1"] = stage1_scores(labels, fit)
    result["checks"] = CHECKS
    result["problems"] = PROBLEMS
    result["elapsed_seconds"] = time.monotonic() - START
    output = ROOT / "rederive_results.json"
    with output.open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    # The report is rendered exclusively from the saved output, preserving provenance.
    with output.open() as stream:
        saved = json.load(stream)
    (ROOT / "rederive_summary.md").write_text(summary(saved))
    log(f"COMPLETE: {len(CHECKS)} checks, {len(PROBLEMS)} problems; results and summary written")


if __name__ == "__main__":
    main()
