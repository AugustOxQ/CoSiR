"""Stress the unchanged Attention-h1 embedding-snapshot recipe at four seeds.

The sibling snapshot script owns the architecture, training loop, losses,
fixed Adam learning rate, stopping rule, graph construction, and Leiden calls.
Each run writes its ordinary snapshot into a temporary directory; only this
pilot's Markdown report is kept in the repository.
"""

import importlib.util
import os
import sys
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(OUT_DIR, "run_attention_h1_embedding_snapshot_pilot.py")
REPORT_PATH = os.path.join(OUT_DIR, "attention_h1_baseline_seed_stress_pilot_report.md")
SEEDS = (42, 7, 123, 2024)
EMOTION_BAR = 0.1236
GENRE_BAR = 0.1954
NOISE_SCHEDULE = {
    42: (0.1306, 0.1973, 0.0488),
    7: (0.1244, 0.2583, 0.0438),
    123: (0.1334, 0.2452, 0.0487),
    2024: (0.1222, 0.2576, 0.0451),
}


def load_baseline():
    spec = importlib.util.spec_from_file_location("attention_h1_unchanged_baseline", BASELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline script: {BASELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sampled_silhouette(embeddings, communities):
    """Use the same fixed seed-42 two-stage sample as the sibling pilots."""
    idx = np.random.default_rng(42).choice(
        len(embeddings), size=min(6000, len(embeddings)), replace=False
    )
    return float(silhouette_score(
        embeddings[idx], communities[idx],
        sample_size=min(4000, len(idx)), random_state=42,
    ))


def evaluate_snapshot(path, seed):
    with np.load(path, allow_pickle=True) as snapshot:
        if int(snapshot["seed"]) != seed:
            raise RuntimeError(f"Snapshot seed mismatch for {seed}")
        communities = snapshot["heldout_community_post"]
        genres = snapshot["heldout_genre"]
        covered = genres != ""
        return {
            "seed": seed,
            "emotion": float(adjusted_mutual_info_score(
                communities, snapshot["heldout_emotion"]
            )),
            "genre": float(adjusted_mutual_info_score(
                communities[covered], genres[covered]
            )),
            "silhouette": sampled_silhouette(
                snapshot["heldout_embedding_post"], communities
            ),
            "train_communities": int(len(np.unique(snapshot["train_community_post"]))),
            "heldout_communities": int(len(np.unique(communities))),
        }


def summary(rows, key):
    values = [row[key] for row in rows]
    return np.mean(values), min(values), max(values)


def write_report(rows):
    schedule_rows = [dict(zip(("emotion", "genre", "silhouette"), NOISE_SCHEDULE[s]))
                     for s in SEEDS]
    clear = lambda r: r["emotion"] > EMOTION_BAR and r["genre"] > GENRE_BAR
    lines = [
        "# Unmodified Attention-h1 baseline: four-seed robustness check\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "Each seed runs `run_attention_h1_embedding_snapshot_pilot.py` as-is, "
        "including its Attention-h1 architecture, two InfoNCE losses, fixed "
        "Adam learning rate, stopping rule, and train/held-out Leiden passes. "
        "Only the module's seed and temporary output paths change. All four "
        "models were freshly fit. Held-out silhouette uses a seed-42 draw of "
        "at most 6,000 points followed by `silhouette_score` with "
        "`sample_size=min(4000, len(idx))`, `random_state=42`.\n\n",
        "| seed | held-out emotion AMI | held-out genre AMI | held-out silhouette "
        "| Pareto bar | train Leiden communities | held-out Leiden communities |\n",
        "|---:|---:|---:|---:|---|---:|---:|\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['emotion']:.4f} | {row['genre']:.4f} | "
            f"{row['silhouette']:.4f} | {'clears' if clear(row) else 'does not clear'} "
            f"| {row['train_communities']} | {row['heldout_communities']} |\n"
        )
    lines.extend(["\nThe held-out Pareto bar requires emotion AMI > 0.1236 "
                  "**and** genre AMI > 0.1954 (strict inequalities).\n\n",
                  "## Held-out summary\n\n",
                  "| metric | mean | min | max |\n",
                  "|---|---:|---:|---:|\n"])
    for key, label in (("emotion", "Emotion AMI"), ("genre", "Genre AMI"),
                       ("silhouette", "Silhouette")):
        mean, low, high = summary(rows, key)
        lines.append(f"| {label} | {mean:.4f} | {low:.4f} | {high:.4f} |\n")
    lines.append(f"\nPareto-bar clearance: **{sum(map(clear, rows))}/4 seeds**.\n\n")
    lines.extend(["## Comparison with noise + cosine schedule\n\n",
                  "The noise + schedule pilot's reported four-seed values are "
                  "listed alongside the untouched baseline below. Differences "
                  "are schedule minus baseline, paired by seed.\n\n",
                  "| seed | schedule emotion / genre / silhouette | "
                  "schedule − baseline emotion / genre / silhouette | schedule bar |\n",
                  "|---:|---:|---:|---|\n"])
    for row in rows:
        other = NOISE_SCHEDULE[row["seed"]]
        delta = [other[i] - row[key] for i, key in enumerate(
            ("emotion", "genre", "silhouette"))]
        lines.append(
            f"| {row['seed']} | {' / '.join(f'{x:.4f}' for x in other)} | "
            f"{' / '.join(f'{x:+.4f}' for x in delta)} | "
            f"{'clears' if clear(schedule_rows[SEEDS.index(row['seed'])]) else 'does not clear'} |\n"
        )
    lines.append("\n")
    for key, label in (("emotion", "emotion AMI"), ("genre", "genre AMI"),
                       ("silhouette", "silhouette")):
        base_mean, base_min, base_max = summary(rows, key)
        schedule_mean, _, _ = summary(schedule_rows, key)
        lines.append(
            f"- {label}: baseline mean {base_mean:.4f} (range {base_min:.4f}–"
            f"{base_max:.4f}); schedule mean {schedule_mean:.4f}; "
            f"mean difference {schedule_mean - base_mean:+.4f}.\n"
        )
    base_clear = sum(map(clear, rows))
    lines.append(
        f"- Pareto clearance: baseline {base_clear}/4; schedule 3/4. "
        + ("The schedule's 3/4 clearance also occurs without the schedule, "
           "so this count alone is within the baseline's observed seed variability.\n"
           if base_clear >= 3 else
           "The schedule clears more seeds, though four paired runs are too few "
           "to establish a reliable improvement.\n")
    )
    improved = [sum(NOISE_SCHEDULE[row["seed"]][i] > row[key]
                    for row in rows)
                for i, key in enumerate(("emotion", "genre", "silhouette"))]
    lines.append(
        f"- Per-seed increases occur in {improved[0]}/4 emotion AMIs, "
        f"{improved[1]}/4 genre AMIs, and {improved[2]}/4 silhouettes; "
        "these paired comparisons show whether gains are consistent across seeds.\n"
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as report:
        report.writelines(lines)


def main():
    baseline = load_baseline()
    rows = []
    for seed in SEEDS:
        with tempfile.TemporaryDirectory(prefix=f"attention_h1_baseline_{seed}_") as output_dir:
            baseline.SEED = seed
            baseline.REPORT_OUT_DIR = output_dir
            baseline.REPORT_PATH = os.path.join(output_dir, "snapshot_report.md")
            baseline.NPZ_PATH = os.path.join(output_dir, "snapshot.npz")
            print(f"[baseline seed {seed}] Starting unchanged snapshot fit", flush=True)
            baseline.main()
            row = evaluate_snapshot(baseline.NPZ_PATH, seed)
            rows.append(row)
            write_report(rows)
            print(f"[baseline seed {seed}] {row}", flush=True)
    print(f"Wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
