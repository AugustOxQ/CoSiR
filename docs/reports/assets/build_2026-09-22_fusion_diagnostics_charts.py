#!/usr/bin/env python3
"""Build table-driven visual diagnostics for the ArtELingo fusion investigation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / "src/test/20260923_artelingo_buddy_analysis"
OUT = Path(__file__).resolve().parent
COLORS = {
    "single-signal reference": "#4C78A8",
    "early fusion": "#F58518",
    "late fusion": "#E45756",
    "hierarchical": "#B279A2",
    "learned student": "#54A24B",
}


def parse_markdown_tables(path: Path) -> list[list[dict[str, str]]]:
    """Parse pipe-delimited markdown tables without external dependencies."""
    lines = path.read_text(encoding="utf-8").splitlines()
    tables: list[list[dict[str, str]]] = []
    index = 0
    while index + 1 < len(lines):
        if not lines[index].lstrip().startswith("|") or not lines[index + 1].lstrip().startswith("|"):
            index += 1
            continue
        header = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        separator = [cell.strip() for cell in lines[index + 1].strip().strip("|").split("|")]
        if len(header) != len(separator) or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
            index += 1
            continue
        rows: list[dict[str, str]] = []
        index += 2
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
            if len(cells) == len(header):
                rows.append(dict(zip(header, cells)))
            index += 1
        tables.append(rows)
    return tables


def find_table(tables: list[list[dict[str, str]]], *headers: str) -> list[dict[str, str]]:
    for table in tables:
        if table and all(header in table[0] for header in headers):
            return table
    raise ValueError(f"Could not find table with headers: {headers}")


def row_matching(table: list[dict[str, str]], field: str, text: str) -> dict[str, str]:
    for row in table:
        if text.casefold() in row[field].casefold():
            return row
    raise ValueError(f"Could not find {text!r} in column {field!r}")


def table_with_row(tables: list[list[dict[str, str]]], field: str, text: str) -> list[dict[str, str]]:
    for table in tables:
        if table and field in table[0] and any(text.casefold() in row[field].casefold() for row in table):
            return table
    raise ValueError(f"Could not find a table containing {text!r} in column {field!r}")


def number(value: str) -> float:
    if value.strip() in {"—", "", "-"}:
        raise ValueError("A charted value is missing")
    return float(value.replace(",", ""))


def metrics(row: dict[str, str], emotion: str = "emotion AMI", genre: str = "genre AMI") -> dict[str, float]:
    return {"emotion": number(row[emotion]), "genre": number(row[genre])}


def source_tables(root: Path) -> dict[str, list[list[dict[str, str]]]]:
    names = {
        "single": "single_modality_pilot_report.md", "early": "affect_pilot_report.md",
        "dec": "dec_pilot_v2_report.md", "late": "late_fusion_pilot_report.md",
        "hierarchical": "hierarchical_refinement_pilot_report.md", "cca": "cca_audit_pilot_report.md",
        "stage1": "learned_student_stage1_pilot_report.md", "stage2": "learned_student_stage2_pilot_report.md",
        "weight": "learned_student_weight_sweep_pilot_report.md",
    }
    return {key: parse_markdown_tables(root / "src/test/20260923_artelingo_buddy_analysis" / value) for key, value in names.items()}


def build_data(root: Path = ROOT) -> dict[str, Any]:
    tables = source_tables(root)
    single = find_table(tables["single"], "graph", "emotion_AMI", "genre_AMI")
    early = find_table(tables["early"], "affect_weight", "community_vs_emotion_AMI")
    dec = find_table(tables["dec"], "method", "emotion AMI", "genre AMI")
    late = find_table(tables["late"], "signal", "emotion AMI", "genre AMI")
    hierarchy = find_table(tables["hierarchical"], "signal", "emotion AMI", "genre AMI")
    cca = find_table(tables["cca"], "component", "held-out correlation", "null 95th percentile")
    cca_metrics = find_table(tables["cca"], "signal", "emotion AMI", "genre AMI")
    trajectory_headers = ("epoch", "content held-out recall", "affect held-out recall", "content loss", "gate mean")
    stage1_trajectory = find_table(tables["stage1"], *trajectory_headers)
    stage2_trajectory = find_table(tables["stage2"], *trajectory_headers)
    stage1_generalization = table_with_row(tables["stage1"], "signal", "Learned student — TRAIN")
    stage2_generalization = table_with_row(tables["stage2"], "signal", "Stage 2 — TRAIN")
    weight = find_table(tables["weight"], "content weight", "split", "emotion AMI", "genre AMI")

    def named(table, field, label, emotion="emotion AMI", genre="genre AMI"):
        return metrics(row_matching(table, field, label), emotion, genre)

    data = {
        "references": {
            "content-only": named(single, "graph", "Existing CLIP img+txt UNION", "emotion_AMI", "genre_AMI"),
            "GoEmotions-only Leiden": named(single, "graph", "GoEmotions-affect-only", "emotion_AMI", "genre_AMI"),
            "GoEmotions-only DEC": named(dec, "method", "DEC v2"),
            "Residual-affect-only": named(cca_metrics, "signal", "Residual-affect-only"),
        },
        "early_fusion": [{"weight": number(row["affect_weight"]), **metrics(row, "community_vs_emotion_AMI", "community_vs_genre_AMI")} for row in early],
        "late": {"union": named(late, "signal", "Late fusion — union"), "intersection": named(late, "signal", "Late fusion — intersection")},
        "hierarchical": {
            "hierarchical": named(hierarchy, "signal", "Hierarchical ("),
            "Control A": named(hierarchy, "signal", "Control A"),
            "Control B": named(hierarchy, "signal", "Control B"),
        },
        "stage1_trajectory": [{key: (number(value) if key not in {"epoch", "effective rank at 95%"} else int(number(value))) for key, value in row.items() if value != "—"} for row in stage1_trajectory],
        "stage2_trajectory": [{key: (number(value) if key not in {"epoch", "effective rank at 95%"} else int(number(value))) for key, value in row.items() if value != "—"} for row in stage2_trajectory],
        "stage1": {"train": metrics(row_matching(stage1_generalization, "signal", "TRAIN")), "held-out": metrics(row_matching(stage1_generalization, "signal", "HELD-OUT"))},
        "stage2": {"train": metrics(row_matching(stage2_generalization, "signal", "Stage 2 — TRAIN")), "held-out": metrics(row_matching(stage2_generalization, "signal", "Stage 2 — HELD-OUT"))},
        "weight_sweep": [{"weight": number(row["content weight"]), "split": row["split"], **metrics(row)} for row in weight],
        "cca": [{"component": int(number(row["component"])), "correlation": number(row["held-out correlation"]), "null_95": number(row["null 95th percentile"])} for row in cca],
    }
    return data


def style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "grid.alpha": 0.25, "figure.dpi": 150})


def save(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_pareto(data):
    fig, ax = plt.subplots(figsize=(10, 7))
    families = [("single-signal reference", data["references"].items()), ("late fusion", data["late"].items()), ("hierarchical", data["hierarchical"].items())]
    for family, values in families:
        for label, value in values:
            ax.scatter(value["emotion"], value["genre"], s=58, color=COLORS[family], label=family if label == list(values)[0][0] else None, zorder=3)
    early = data["early_fusion"]
    ax.plot([r["emotion"] for r in early], [r["genre"] for r in early], color=COLORS["early fusion"], lw=1, alpha=.7)
    ax.scatter([r["emotion"] for r in early], [r["genre"] for r in early], color=COLORS["early fusion"], label="early fusion", zorder=3)
    for stage in ("stage1", "stage2"):
        for split, value in data[stage].items():
            ax.scatter(value["emotion"], value["genre"], s=70, marker="D" if split == "held-out" else "o", color=COLORS["learned student"], label="learned student" if stage == "stage1" and split == "train" else None, zorder=4)
    for split, value in data["stage1"].items(): ax.annotate(f"Stage 1 {split}", (value["emotion"], value["genre"]), xytext=(6, 6), textcoords="offset points", fontsize=9, weight="bold")
    for split in ("train", "held-out"):
        sweep = sorted((r for r in data["weight_sweep"] if r["split"] == split), key=lambda r: r["weight"])
        ax.plot([r["emotion"] for r in sweep], [r["genre"] for r in sweep], color=COLORS["learned student"], lw=1, alpha=.55)
        ax.scatter([r["emotion"] for r in sweep], [r["genre"] for r in sweep], color=COLORS["learned student"], s=28, zorder=3)
    s1 = data["stage1"]["train"]
    ax.axvline(s1["emotion"], color=COLORS["learned student"], ls="--", lw=1, alpha=.55)
    ax.axhline(s1["genre"], color=COLORS["learned student"], ls="--", lw=1, alpha=.55)
    ax.fill_between([0, s1["emotion"]], 0, s1["genre"], color=COLORS["learned student"], alpha=.07, label="dominated by Stage 1 train")
    ax.set(xlabel="Emotion AMI", ylabel="Genre AMI", title="Fusion mechanisms on the emotion–genre trade-off frontier", xlim=(0, .16), ylim=(0, .48))
    ax.legend(fontsize=8, loc="upper left"); save(fig, "fusion_pareto_frontier.png")


def chart_method_bars(data):
    early_best = max(data["early_fusion"], key=lambda row: row["emotion"])
    methods = [("Content-only", data["references"]["content-only"]), ("GoEmotions-only", data["references"]["GoEmotions-only Leiden"]), ("Early fusion best", early_best), ("Late union", data["late"]["union"]), ("Late intersection", data["late"]["intersection"]), ("Hierarchical", data["hierarchical"]["hierarchical"]), ("Stage 1", data["stage1"]["train"]), ("Stage 2", data["stage2"]["train"])]
    fig, ax = plt.subplots(figsize=(12, 6)); x = list(range(len(methods))); width = .37
    ax.bar([i-width/2 for i in x], [v["emotion"] for _, v in methods], width, label="Emotion AMI", color="#4C78A8")
    ax.bar([i+width/2 for i in x], [v["genre"] for _, v in methods], width, label="Genre AMI", color="#F58518")
    ax.set(xticks=x, xticklabels=[name for name, _ in methods], ylabel="AMI", title="Headline train-split fusion results"); ax.tick_params(axis="x", rotation=30); ax.legend(); save(fig, "fusion_method_bars.png")


def chart_stage1_trajectory(data):
    rows = data["stage1_trajectory"]; e = [r["epoch"] for r in rows]; fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0,0].plot(e, [r["content held-out recall"] for r in rows], label="content", color="#4C78A8"); axes[0,0].plot(e, [r["affect held-out recall"] for r in rows], label="affect", color="#E45756"); axes[0,0].set(title="Held-out recall", xlabel="Epoch", ylabel="Recall"); axes[0,0].legend()
    loss_rows = [r for r in rows if "content loss" in r]; le = [r["epoch"] for r in loss_rows]; axes[0,1].plot(le, [r["content loss"] for r in loss_rows], label="content", color="#4C78A8"); axes[0,1].plot(le, [r["affect loss"] for r in loss_rows], label="affect", color="#E45756"); axes[0,1].set(title="Teacher losses", xlabel="Epoch", ylabel="Loss"); axes[0,1].legend()
    grad = [r for r in rows if "content gradient share" in r]; axes[1,0].plot([r["epoch"] for r in grad], [r["content gradient share"] for r in grad], color="#54A24B"); axes[1,0].axhline(.5, color="black", ls="--", lw=1); axes[1,0].set(title="Content gradient share", xlabel="Epoch", ylabel="Share")
    mean = [r["gate mean"] for r in rows]; std = [r["gate std"] for r in rows]; axes[1,1].plot(e, mean, color="#B279A2"); axes[1,1].fill_between(e, [m-s for m,s in zip(mean,std)], [m+s for m,s in zip(mean,std)], color="#B279A2", alpha=.2); axes[1,1].set(title="Gate mean ± standard deviation", xlabel="Epoch", ylabel="Gate value")
    save(fig, "stage1_checkpoint_trajectory.png")


def chart_stage_comparison(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for stage, label, color in [("stage1_trajectory", "Stage 1 linear", "#54A24B"), ("stage2_trajectory", "Stage 2 MLP", "#E45756")]:
        rows=data[stage]; epochs=[r["epoch"] for r in rows]; axes[0].plot(epochs,[r["content held-out recall"] for r in rows],label=label,color=color); axes[1].plot(epochs,[r["affect held-out recall"] for r in rows],label=label,color=color)
    axes[0].set(title="Content held-out recall", xlabel="Epoch", ylabel="Recall"); axes[1].set(title="Affect held-out recall", xlabel="Epoch", ylabel="Recall")
    for ax in axes: ax.legend()
    save(fig, "stage1_vs_stage2_trajectory.png")


def chart_weight_sweep(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for split, style_line in [("train", "-"), ("held-out", "--")]:
        rows=sorted((r for r in data["weight_sweep"] if r["split"]==split),key=lambda r:r["weight"])
        for ax, metric, title in [(axes[0],"emotion","Emotion AMI"),(axes[1],"genre","Genre AMI")]: ax.plot([r["weight"] for r in rows],[r[metric] for r in rows],style_line,marker="o",label=split); ax.set(title=title,xlabel="content_weight",ylabel="AMI")
    for ax in axes: ax.legend()
    save(fig, "weight_sweep_tradeoff.png")


def chart_hierarchy(data):
    methods=list(data["hierarchical"].items()); fig,ax=plt.subplots(figsize=(8,5)); x=list(range(len(methods))); w=.35
    ax.bar([i-w/2 for i in x],[v["emotion"] for _,v in methods],w,label="Emotion AMI",color="#B279A2"); ax.bar([i+w/2 for i in x],[v["genre"] for _,v in methods],w,label="Genre AMI",color="#4C78A8")
    ax.set(xticks=x,xticklabels=[n for n,_ in methods],ylabel="AMI",title="Hierarchical refinement and matched controls"); ax.legend(); ax.annotate("Control A: same split sizes, no real affect signal", xy=(1,.2014), xytext=(.15,.37), arrowprops={"arrowstyle":"->"}, fontsize=9)
    save(fig,"hierarchical_controls_bars.png")


def chart_cca(data):
    rows=data["cca"]; fig,ax=plt.subplots(figsize=(10,5)); x=[r["component"] for r in rows]; ax.bar(x,[r["correlation"] for r in rows],color="#4C78A8",label="held-out correlation"); ax.scatter(x,[r["null_95"] for r in rows],marker="_",s=230,color="#E45756",linewidths=2,label="null 95th percentile")
    ax.set(xticks=x,xlabel="Canonical component",ylabel="Correlation",title="Held-out CCA correlations exceed permutation-null thresholds"); ax.legend(); save(fig,"cca_canonical_correlations.png")


def main() -> None:
    style(); data=build_data(ROOT)
    chart_pareto(data); chart_method_bars(data); chart_stage1_trajectory(data); chart_stage_comparison(data); chart_weight_sweep(data); chart_hierarchy(data); chart_cca(data)
    print(json.dumps(data, indent=2, sort_keys=True))
    print(f"Wrote 7 PNGs to {OUT}")


if __name__ == "__main__":
    main()
