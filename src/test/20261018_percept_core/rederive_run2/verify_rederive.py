#!/root/miniconda3/envs/CoSiR/bin/python
"""Check serialized results and summary arithmetic without loading source data."""
import json
import math
from pathlib import Path

ROOT = Path("/tmp/claude-0/-project-CoSiR/224c5df5-d4b9-42a6-bc93-9fe548e609f0/scratchpad/rederive2")
data = json.loads((ROOT / "rederive_results.json").read_text(encoding="utf-8"))
summary = (ROOT / "rederive_summary.md").read_text(encoding="utf-8")
aspects = ("emotion", "style", "genre")
checked_statistics = 0


def check_stats(values, actual):
    global checked_statistics
    average = math.fsum(values) / len(values)
    sd = math.sqrt(math.fsum((value - average) ** 2 for value in values) / len(values))
    assert math.isclose(actual["mean"], average, rel_tol=1e-13, abs_tol=1e-15)
    assert math.isclose(actual["sd"], sd, rel_tol=1e-13, abs_tol=1e-15)
    checked_statistics += 1


assert not data["errors"], data["errors"]
assert len(data["files"]) + len(data["absent_files"]) == 15
assert len(data["files"]) == data["checks"]["present_file_count"]
expected = {f"P{arm}__s{seed}.npz" for arm in range(1, 6) for seed in range(3)}
assert set(data["files"]) | {Path(path).name for path in data["absent_files"]} == expected
assert not (set(data["files"]) & {Path(path).name for path in data["absent_files"]})
for name, file in data["files"].items():
    assert [run["random_state"] for run in file["kmeans"]] == [0, 1, 2]
    assert file["checks"]["shape"] == [5000, file["K"]]
    for run in file["kmeans"]:
        assert len(run["cluster_counts"]) == 20
        assert sum(run["cluster_counts"]) == 5000
        assert all(math.isfinite(value) for value in run["ami"].values())
    for aspect in aspects:
        check_stats([run["ami"][aspect] for run in file["kmeans"]], file["image_side"][aspect])
    numbers = " | ".join(f"{file['image_side'][aspect][stat]:.6f}" for aspect in aspects for stat in ("mean", "sd"))
    assert f"| {name} | {file['K']} | {numbers} |" in summary
for arm, aggregate in data["arms"].items():
    files = [file for file in data["files"].values() if file["arm"] == arm]
    assert aggregate["model_seeds"] == [file["model_seed"] for file in files]
    for aspect in aspects:
        if not files:
            assert not aggregate["image_side"] and not aggregate["argmax"]
            continue
        check_stats([file["image_side"][aspect]["mean"] for file in files], aggregate["image_side"][aspect])
        check_stats([file["argmax"][aspect] for file in files], aggregate["argmax"][aspect])
        image, argmax = aggregate["image_side"][aspect], aggregate["argmax"][aspect]
        numbers = f"{image['mean']:.6f} | {image['sd']:.6f} | {argmax['mean']:.6f} | {argmax['sd']:.6f}"
        assert f"| {arm} | {aspect} | {','.join(map(str, aggregate['model_seeds']))} | {numbers} |" in summary
for aspect in aspects:
    eligible = [arm for arm, aggregate in data["arms"].items() if aggregate["model_seeds"] == [0, 1, 2]]
    selected = data["comparators"][aspect]
    if not eligible:
        assert selected is None
        continue
    best = max(data["arms"][arm]["image_side"][aspect]["mean"] for arm in eligible)
    within = [arm for arm in eligible if best - data["arms"][arm]["image_side"][aspect]["mean"] <= 0.003]
    assert selected["comparator_arm"] == min(within, key=lambda arm: int(arm[1:]))
    assert selected["highest_mean"] == best
    assert selected["highest_arm"] in eligible
    assert data["arms"][selected["highest_arm"]]["image_side"][aspect]["mean"] == best
    assert selected["mean"] == data["arms"][selected["comparator_arm"]]["image_side"][aspect]["mean"]
    assert selected["sd"] == data["arms"][selected["comparator_arm"]]["image_side"][aspect]["sd"]
for label in data["sample_checks"]["labels"].values():
    assert label["scored"] + label["excluded"] == 5000
    assert sum(label["counts"].values()) == 5000

report = {
    "passed": True,
    "files_scored": len(data["files"]),
    "files_absent": len(data["absent_files"]),
    "kmeans_partitions_checked": sum(len(file["kmeans"]) for file in data["files"].values()),
    "per_kmeans_seed_amis_checked": sum(len(run["ami"]) for file in data["files"].values() for run in file["kmeans"]),
    "mean_sd_pairs_recomputed_with_math_fsum": checked_statistics,
    "summary_rows_match_serialized_results": True,
    "comparator_rule_verified": True,
    "label_counts_verified": True,
}
(ROOT / "rederive_verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
marker = "\n## Serialized-output verification\n"
verification_text = (
    marker + "\n"
    "- Re-read the saved JSON and checked all expected filenames are accounted for exactly once.\n"
    f"- Checked {report['kmeans_partitions_checked']} partitions each have 20 cluster-count entries summing to 5,000, and {report['per_kmeans_seed_amis_checked']} per-k-means-seed AMIs are finite.\n"
    f"- Independently recomputed {checked_statistics} mean/SD pairs using Python math.fsum and population variance.\n"
    "- Checked every numeric arm and per-file summary row matches the saved JSON rounded to six decimals.\n"
    "- Recomputed comparator eligibility, maximum means, tolerance selection, and selected-arm SDs.\n"
    "- Checked each label's scored and excluded counts and class counts sum to 5,000.\n"
    "- All verification checks passed; details are in rederive_verification.json.\n"
)
(ROOT / "rederive_summary.md").write_text(summary.split(marker)[0] + verification_text, encoding="utf-8")
print((ROOT / "rederive_verification.json").read_text(encoding="utf-8"), end="")
