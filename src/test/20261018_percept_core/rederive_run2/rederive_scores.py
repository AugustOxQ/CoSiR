#!/root/miniconda3/envs/CoSiR/bin/python
"""Independent C13 derivation; reads only the explicitly authorized NPZ inputs."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    if os.environ.get(variable) != "8":
        raise RuntimeError(f"{variable} must be 8")
if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
    raise RuntimeError("PYTHONDONTWRITEBYTECODE must be 1")

import numpy as np
import scipy
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_info


ROOT = Path("/tmp/claude-0/-project-CoSiR/224c5df5-d4b9-42a6-bc93-9fe548e609f0/scratchpad/rederive2")
SAMPLE = Path("/project/CoSiR/src/test/20261017_prompted_aspect_views/sample.npz")
INPUT = Path("/project/CoSiR/src/test/20261018_percept_core/outputs/stage2")
ASPECTS = ("emotion", "style", "genre")
ARMS = tuple(f"P{i}" for i in range(1, 6))
SEEDS = (0, 1, 2)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(array)), "sd": float(np.std(array, ddof=0))}


def load_labels():
    with np.load(SAMPLE, allow_pickle=False) as sample:
        # Only these supplied data arrays are accessed; no code or result files.
        arrays = {key: sample[key] for key in (
            "sample_ids", "pid", "painting_ids", "emo", "sty", "gen",
            "first", "all_painting_ids", "subset_pid",
        )}
    for key in ("sample_ids", "pid", "painting_ids", "emo", "sty", "gen"):
        require(arrays[key].shape == (99999,), f"Unexpected sample shape: {key}")
    for key in ("first", "all_painting_ids"):
        require(arrays[key].shape == (19889,), f"Unexpected painting shape: {key}")
    require(arrays["subset_pid"].shape == (5000,), "Unexpected evaluation subset shape")
    for key in ("sample_ids", "pid", "emo", "sty", "gen", "first", "subset_pid"):
        require(np.issubdtype(arrays[key].dtype, np.integer), f"Non-integer {key}")
    require(arrays["sample_ids"].dtype == np.dtype("int64"), "sample_ids is not int64")
    pid, first, subset = arrays["pid"], arrays["first"], arrays["subset_pid"]
    require(bool(np.all((pid >= 0) & (pid < 19889))), "Caption pid out of range")
    require(bool(np.all((first >= 0) & (first < 99999))), "first index out of range")
    require(bool(np.all((subset >= 0) & (subset < 19889))), "subset_pid out of range")
    require(np.unique(subset).size == 5000, "Duplicate evaluation paintings")
    require(np.unique(arrays["sample_ids"]).size == 99999, "Duplicate sample IDs")
    calculated_first = np.full(19889, 99999, dtype=np.int64)
    np.minimum.at(calculated_first, pid, np.arange(99999, dtype=np.int64))
    require(np.array_equal(first, calculated_first), "first is not the first caption per painting")
    require(bool(np.all(arrays["painting_ids"] == arrays["all_painting_ids"][pid])),
            "Caption and painting names disagree")
    emotion = arrays["emo"]
    require(bool(np.all((emotion >= -1) & (emotion <= 8))), "Emotion out of range")
    counts = np.zeros((19889, 9), dtype=np.int64)
    valid = emotion >= 0
    np.add.at(counts, (pid[valid], emotion[valid]), 1)
    # np.argmax returns the smallest class index when counts tie.
    emotion_labels = np.argmax(counts, axis=1)
    emotion_labels[np.sum(counts, axis=1) == 0] = -1
    labels = {
        "emotion": emotion_labels[subset],
        "style": arrays["sty"][first[subset]],
        "genre": arrays["gen"][first[subset]],
    }
    checks = {
        "path": str(SAMPLE), "sha256": sha256(SAMPLE),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "sample_ids_unique": True, "subset_pid_unique": True,
        "indices_in_range": True, "first_caption_indices_verified": True,
        "painting_names_match_pid": True, "emotion_range_valid": True,
        "evaluation_emotion_majority_ties": int(np.sum(
            (np.sum(counts[subset] == np.max(counts[subset], axis=1, keepdims=True), axis=1) > 1)
            & (np.sum(counts[subset], axis=1) > 0))),
        "labels": {},
    }
    for aspect, label in labels.items():
        values, frequencies = np.unique(label, return_counts=True)
        require(bool(np.any(label >= 0)), f"No valid labels for {aspect}")
        checks["labels"][aspect] = {
            "scored": int(np.sum(label >= 0)), "excluded": int(np.sum(label < 0)),
            "class_count": int(np.unique(label[label >= 0]).size),
            "counts": {str(int(value)): int(n) for value, n in zip(values, frequencies)},
        }
    return labels, checks


def score_file(path, labels):
    started = time.monotonic()
    with np.load(path, allow_pickle=False) as archive:
        require(archive.files == ["eval_prob"], "Expected exactly one array, eval_prob")
        original = archive["eval_prob"]
    require(original.ndim == 2 and original.shape[0] == 5000 and original.shape[1] >= 2,
            f"Invalid eval_prob shape: {original.shape}")
    require(original.dtype == np.dtype("float32"), f"Expected float32, got {original.dtype}")
    require(bool(np.all(np.isfinite(original))), "Nonfinite eval_prob")
    lo, hi = float(np.min(original)), float(np.max(original))
    require(lo >= 0.0 and hi <= 1.0, f"Out of range eval_prob: {lo}, {hi}")
    probability = np.asarray(original, dtype=np.float32)
    norms = np.linalg.norm(probability, axis=1, keepdims=True)
    x = np.asarray(probability / np.maximum(norms, np.float32(1e-12)), dtype=np.float32)
    require(bool(np.all(np.isfinite(x))), "Nonfinite normalized features")
    nonzero = norms[:, 0] >= np.float32(1e-12)
    norm_error = float(np.max(np.abs(np.linalg.norm(x[nonzero], axis=1) - 1.0))) if np.any(nonzero) else 0.0
    require(norm_error <= 1e-5, f"Normalization error {norm_error}")
    masks = {aspect: label >= 0 for aspect, label in labels.items()}
    topics = np.argmax(probability, axis=1)
    result = {
        "path": str(path), "sha256": sha256(path), "K": int(x.shape[1]),
        "checks": {"shape": list(x.shape), "input_dtype": str(original.dtype),
                   "finite": True, "minimum": lo, "maximum": hi, "in_unit_interval": True,
                   "zero_norm_rows": int(np.sum(norms[:, 0] == 0)),
                   "rows_below_norm_floor": int(np.sum(~nonzero)),
                   "normalized_dtype": str(x.dtype), "max_unit_norm_error": norm_error,
                   "argmax_topic_count": int(np.unique(topics).size),
                   "argmax_tied_rows": int(np.sum(np.sum(probability == np.max(probability, axis=1, keepdims=True), axis=1) > 1))},
        "kmeans": [], "image_side": {}, "argmax": {},
    }
    for aspect in ASPECTS:
        mask = masks[aspect]
        result["argmax"][aspect] = float(adjusted_mutual_info_score(labels[aspect][mask], topics[mask]))
    for seed in SEEDS:
        pred = MiniBatchKMeans(n_clusters=20, random_state=seed, n_init=3,
                               batch_size=4096).fit_predict(x)
        ami = {aspect: float(adjusted_mutual_info_score(labels[aspect][masks[aspect]],
                                                      pred[masks[aspect]])) for aspect in ASPECTS}
        result["kmeans"].append({"random_state": seed, "ami": ami,
                                  "cluster_counts": np.bincount(pred, minlength=20).tolist()})
        print(f"{path.name}: k-means seed {seed} complete", flush=True)
    for aspect in ASPECTS:
        result["image_side"][aspect] = stats([run["ami"][aspect] for run in result["kmeans"]])
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def summarize(result):
    lines = ["# Independent CoSiR r10 / C13 clustering derivation", "",
             "All reported AMIs were computed from the authorized NPZ inputs. Population SD uses ddof=0. "
             "Image-side arm SD is across model-seed means; per-file SD is across the three k-means seeds.", "",
             "## Per arm and label", "",
             "| Arm | Label | Model seeds | Image-side mean | Image-side SD | Argmax mean | Argmax SD |",
             "|---|---|---|---:|---:|---:|---:|"]
    for arm, data in result["arms"].items():
        for aspect in ASPECTS:
            if data["model_seeds"]:
                image, argmax = data["image_side"][aspect], data["argmax"][aspect]
                numbers = f"{image['mean']:.6f} | {image['sd']:.6f} | {argmax['mean']:.6f} | {argmax['sd']:.6f}"
            else:
                numbers = "— | — | — | —"
            lines.append(f"| {arm} | {aspect} | {','.join(map(str, data['model_seeds'])) or 'none'} | {numbers} |")
    lines += ["", "## Per-file image-side scores", "", "| File | K | Emotion mean | Emotion SD | Style mean | Style SD | Genre mean | Genre SD |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, data in result["files"].items():
        numbers = " | ".join(f"{data['image_side'][aspect][stat]:.6f}" for aspect in ASPECTS for stat in ("mean", "sd"))
        lines.append(f"| {name} | {data['K']} | {numbers} |")
    lines += ["", "## Comparators", "", "Only arms with all three valid model seeds are eligible. "
              "Choose the lowest arm number within 0.003 of the highest image-side mean.", "",
              "| Aspect | Comparator | Comparator mean | Comparator SD | Highest arm(s) | Highest mean |",
              "|---|---|---:|---:|---|---:|"]
    for aspect, data in result["comparators"].items():
        if data is None:
            lines.append(f"| {aspect} | unavailable | — | — | — | — |")
        else:
            lines.append(f"| {aspect} | {data['comparator_arm']} | {data['mean']:.6f} | {data['sd']:.6f} | "
                         f"{', '.join(data['highest_arms'])} | {data['highest_mean']:.6f} |")
    lines += ["", "## Checks", "",
              f"- Expected files: 15; present: {result['checks']['present_file_count']}; absent: {len(result['absent_files'])}; "
              f"successfully scored: {len(result['files'])}; failed validation/scoring: {len(result['errors'])}.",
              "- Input paths were constructed from P1–P5 and model seeds 0–2. Only the specified NPZ files were read as scoring inputs; no pending NPZs, sidecars, external results, reports, pre-existing logs, or source code were read.",
              "- Every input was loaded with allow_pickle=False; SHA-256 hashes are stored in JSON.",
              "- Sample: six caption arrays each have 99,999 rows; two painting arrays each have 19,889 rows; subset_pid has 5,000 rows.",
              "- Integer indices/labels, int64 sample IDs, unique sample IDs and subset indices, index bounds, actual first-caption indices, caption-to-painting names, and emotion range -1–8 were checked.",
              f"- Emotion uses nonnegative-caption majority and smallest-ID tie breaking; {result['sample_checks']['evaluation_emotion_majority_ties']} evaluation paintings have a nonempty tied majority. Empty votes receive -1.",
              "- Style and genre use first[subset_pid], preserving evaluation row order. Only nonnegative labels enter AMI; all 5,000 rows enter k-means.",
              "- Each output must contain only eval_prob, have shape (5000, K), K >= 2, float32 dtype, finite values, and range [0, 1].",
              "- Normalization uses float32 x / max(row L2 norm, 1e-12); finite normalized values and unit-norm error <= 1e-5 were checked. Zero rows remain zero.",
              "- Exactly three MiniBatchKMeans partitions per file use n_clusters=20, random_state=0/1/2, n_init=3, batch_size=4096, other defaults; each partition is shared by all labels.",
              "- AMI uses sklearn default arguments. Argmax uses the original probabilities and first-maximum tie breaking. All AMIs and statistics must be finite.",
              "- Arm aggregates use present, successfully scored model seeds; comparator eligibility requires all three. SD is population SD throughout.",
              "", "### Label counts", "", "| Label | Scored | Excluded | Classes | Counts (including negative labels) |",
              "|---|---:|---:|---:|---|"]
    for aspect, data in result["sample_checks"]["labels"].items():
        lines.append(f"| {aspect} | {data['scored']} | {data['excluded']} | {data['class_count']} | {json.dumps(data['counts'], sort_keys=True)} |")
    lines += ["", "### Per-file shapes and value ranges", "", "| File | Shape | Min | Max | Zero norm rows | Argmax tied rows |",
              "|---|---|---:|---:|---:|---:|"]
    for name, data in result["files"].items():
        check = data["checks"]
        lines.append(f"| {name} | {check['shape']} | {check['minimum']!r} | {check['maximum']!r} | "
                     f"{check['zero_norm_rows']} | {check['argmax_tied_rows']} |")
    lines += ["", "### Absent files", ""]
    lines += [f"- {path}" for path in result["absent_files"]] or ["- None."]
    lines += ["", "### Problems or observations", ""]
    lines += [f"- {item}" for item in result["observations"]] or ["- None."]
    lines += ["", "### Execution", "", f"- Python: {result['environment']['python']}; NumPy: {np.__version__}; SciPy: {scipy.__version__}; scikit-learn: {sklearn.__version__}.",
              "- CPU execution; OMP_NUM_THREADS=8, MKL_NUM_THREADS=8, PYTHONDONTWRITEBYTECODE=1; CUDA_VISIBLE_DEVICES is empty.",
              f"- Elapsed seconds: {result['elapsed_seconds']!r}.", "- Nothing deleted; no removal proposed.", ""]
    return "\n".join(lines)


def main():
    started = time.monotonic()
    result = {
        "rule": "CoSiR r10 C13", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                        "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES")},
                        "threadpools": threadpool_info()},
        "configuration": {"n_clusters": 20, "random_states": list(SEEDS), "n_init": 3,
                          "batch_size": 4096, "sd_ddof": 0, "comparator_tolerance": 0.003},
        "files": {}, "absent_files": [], "errors": {}, "observations": [],
    }
    labels, result["sample_checks"] = load_labels()
    print("Sample labels derived and validated.", flush=True)
    present = 0
    for arm in ARMS:
        for seed in SEEDS:
            path = INPUT / f"{arm}__s{seed}.npz"
            if not path.is_file():
                result["absent_files"].append(str(path))
                print(f"Absent: {path.name}", flush=True)
                continue
            present += 1
            try:
                print(f"Scoring {path.name}", flush=True)
                data = score_file(path, labels)
                data["arm"], data["model_seed"] = arm, seed
                result["files"][path.name] = data
                if data["checks"]["zero_norm_rows"]:
                    result["observations"].append(f"{path.name}: {data['checks']['zero_norm_rows']} zero-norm rows.")
                if data["checks"]["argmax_tied_rows"]:
                    result["observations"].append(f"{path.name}: {data['checks']['argmax_tied_rows']} argmax-tied rows; first maximum used as specified.")
            except Exception as error:
                result["errors"][path.name] = f"{type(error).__name__}: {error}"
                traceback.print_exc()
    result["checks"] = {"expected_file_count": 15, "present_file_count": present}
    result["arms"] = {}
    for arm in ARMS:
        data = [value for value in result["files"].values() if value["arm"] == arm]
        aggregate = {"model_seeds": [value["model_seed"] for value in data],
                     "K_by_model_seed": {str(value["model_seed"]): value["K"] for value in data},
                     "image_side": {}, "argmax": {}}
        if data:
            for aspect in ASPECTS:
                aggregate["image_side"][aspect] = stats([value["image_side"][aspect]["mean"] for value in data])
                aggregate["argmax"][aspect] = stats([value["argmax"][aspect] for value in data])
        result["arms"][arm] = aggregate
        if len(set(aggregate["K_by_model_seed"].values())) > 1:
            result["observations"].append(f"{arm}: K varies across available model seeds: {aggregate['K_by_model_seed']}. This is permitted by the supplied rule.")
    eligible = [arm for arm in ARMS if result["arms"][arm]["model_seeds"] == list(SEEDS)]
    result["comparators"] = {}
    for aspect in ASPECTS:
        if not eligible:
            result["comparators"][aspect] = None
            continue
        means = {arm: result["arms"][arm]["image_side"][aspect]["mean"] for arm in eligible}
        highest = max(means.values())
        within = [arm for arm in eligible if highest - means[arm] <= 0.003]
        selected = within[0]
        result["comparators"][aspect] = {
            "eligible_arms": eligible, "arms_within_tolerance": within,
            "comparator_arm": selected, **result["arms"][selected]["image_side"][aspect],
            "highest_arm": next(arm for arm in eligible if means[arm] == highest),
            "highest_arms": [arm for arm in eligible if means[arm] == highest], "highest_mean": highest,
        }
    if result["absent_files"]:
        result["observations"].append(f"{len(result['absent_files'])} expected model files absent; see complete list above.")
    result["observations"] += [f"{name}: {error}" for name, error in result["errors"].items()]
    if not eligible:
        result["observations"].append("No arm has all three valid model seeds; comparators unavailable.")
    result["elapsed_seconds"] = time.monotonic() - started
    result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # allow_nan=False checks every stored numerical result for finite JSON values.
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (ROOT / "rederive_results.json").write_text(serialized, encoding="utf-8")
    (ROOT / "rederive_summary.md").write_text(summarize(result), encoding="utf-8")
    print(f"COMPLETE: {len(result['files'])} scored, {len(result['absent_files'])} absent, "
          f"{len(result['errors'])} errors; {result['elapsed_seconds']:.1f} seconds", flush=True)
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
