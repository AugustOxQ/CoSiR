"""Apply Experiment 17.1's spec decision rule across all axis/modality results
and print the overall gate verdict for Experiment 17.2's scoping."""
import json
import os


def main():
    here = os.path.dirname(__file__)
    raw = json.load(open(os.path.join(here, "raw_clip_audit_results.json")))

    rows = []  # (dataset, axis, modality, verdict, real_auc_folded, z)
    for dataset, axes in raw.items():
        for axis_name, modalities in axes.items():
            for modality, r in modalities.items():
                rows.append((dataset, axis_name, modality, r["verdict"], r["real_auc_folded"], r["z"]))

    positives = [r for r in rows if r[3] == "positive"]
    partials = [r for r in rows if r[3] == "partial"]

    print(f"{'dataset':<14} {'axis':<28} {'modality':<6} {'verdict':<9} {'auc_folded':>10} {'z':>8}")
    for row in sorted(rows, key=lambda r: -r[5]):
        print(f"{row[0]:<14} {row[1]:<28} {row[2]:<6} {row[3]:<9} {row[4]:>10.3f} {row[5]:>8.2f}")

    if positives:
        best = max(positives, key=lambda r: r[5])
        print(f"\nGATE VERDICT: positive — strongest axis: {best[1]} ({best[0]}, {best[2]}, z={best[5]:.2f})")
    elif partials:
        best = max(partials, key=lambda r: r[5])
        print(f"\nGATE VERDICT: partial — strongest axis: {best[1]} ({best[0]}, {best[2]}, z={best[5]:.2f})")
    else:
        print("\nGATE VERDICT: null — no axis cleared its control baseline on any dataset/modality")


if __name__ == "__main__":
    main()
