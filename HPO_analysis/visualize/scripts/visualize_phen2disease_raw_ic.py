#!/usr/bin/env python3
"""Visualize Phen2Disease-like raw IC distributions with matplotlib."""

from __future__ import annotations

import csv
import json
import os
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_JSON = BASE_DIR / "output" / "hpo_information.json"
FIG_DIR = BASE_DIR / "visualize" / "fig"
MPL_CONFIG_DIR = BASE_DIR / "visualize" / ".matplotlib_cache"

MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


BLUE = "#4E79A7"
RED = "#E15759"
GREEN = "#59A14F"
GRAY = "#6B7280"


def load_terms() -> list[dict[str, Any]]:
    with INPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["terms"].values())


def raw_ic(term: dict[str, Any]) -> float | None:
    value = term.get("phen2disease_raw", {}).get("information_content")
    if value is None:
        return None
    return float(value)


def category_values(terms: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values_by_category: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, str] = {}
    for term in terms:
        value = raw_ic(term)
        if value is None:
            continue
        categories = term.get("categories") or [{"hpo_id": "UNCATEGORIZED", "name": "Uncategorized"}]
        for category in categories:
            category_id = category["hpo_id"]
            labels[category_id] = f"{category['hpo_id']} {category['name']}"
            values_by_category[category_id].append(value)

    grouped: dict[str, dict[str, Any]] = {}
    for category_id, values in values_by_category.items():
        values = sorted(values)
        grouped[category_id] = {
            "label": labels[category_id],
            "values": values,
            "n": len(values),
            "min": values[0],
            "p05": percentile(values, 0.05),
            "q1": percentile(values, 0.25),
            "median": percentile(values, 0.50),
            "q3": percentile(values, 0.75),
            "p95": percentile(values, 0.95),
            "max": values[-1],
            "mean": statistics.fmean(values),
        }
    return grouped


def percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    weight = pos - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def save_all(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ic_distribution(values: list[float]) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(13.5, 8.0), constrained_layout=True)

    sns.histplot(values, binwidth=0.5, color=BLUE, edgecolor="white", linewidth=0.8, ax=ax)
    mean_value = statistics.fmean(values)
    median_value = statistics.median(values)
    q1 = percentile(sorted(values), 0.25)
    q3 = percentile(sorted(values), 0.75)

    ax.axvline(median_value, color=RED, linewidth=2.4, label=f"Median {median_value:.2f}")
    ax.axvline(mean_value, color=GREEN, linewidth=2.4, linestyle="--", label=f"Mean {mean_value:.2f}")
    ax.set_title("Phen2Disease-like Raw IC Distribution", loc="left", fontsize=22, pad=22, weight="bold")
    ax.set_xlabel("Phen2Disease-like raw IC = -log2(raw disease count / total raw disease IDs)", labelpad=14)
    ax.set_ylabel("HPO term count", labelpad=12)
    ax.set_xlim(left=0)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax.text(
        0.0,
        -0.20,
        f"n={len(values):,}   min={min(values):.2f}   q1={q1:.2f}   median={median_value:.2f}   q3={q3:.2f}   max={max(values):.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color=GRAY,
    )
    save_all(fig, "phen2disease_raw_ic_distribution")


def plot_ic_by_category(grouped: dict[str, dict[str, Any]]) -> None:
    rows = sorted(grouped.values(), key=lambda row: (row["median"], row["label"]))
    labels = [textwrap.fill(row["label"], width=42) for row in rows]
    values = [row["values"] for row in rows]
    y_positions = list(range(len(rows)))

    sns.set_theme(style="whitegrid", context="notebook")
    fig_height = max(10.5, len(rows) * 0.55 + 2.3)
    fig, ax = plt.subplots(figsize=(15.5, fig_height))

    violin = ax.violinplot(
        values,
        positions=y_positions,
        vert=False,
        widths=0.78,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor("#D7E8F7")
        body.set_edgecolor("#7AA6D1")
        body.set_alpha(0.85)

    box = ax.boxplot(
        values,
        positions=y_positions,
        vert=False,
        widths=0.34,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": RED, "linewidth": 2.0},
        boxprops={"facecolor": "#FFFFFF", "edgecolor": "#2F5F8F", "linewidth": 1.2},
        whiskerprops={"color": "#4B5563", "linewidth": 1.1},
        capprops={"color": "#4B5563", "linewidth": 1.1},
    )
    _ = box

    means = [row["mean"] for row in rows]
    ax.scatter(means, y_positions, s=28, color=GREEN, zorder=3, label="Mean")

    axis_max = max(row["max"] for row in rows)
    ax.set_xlim(0, axis_max + 1.2)
    label_x = axis_max + 0.42
    for y, row in zip(y_positions, rows):
        ax.text(
            label_x,
            y,
            f"n={row['n']:,}",
            va="center",
            ha="left",
            fontsize=10,
            color=GRAY,
            clip_on=False,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    fig.text(
        0.36,
        0.982,
        "Phen2Disease-like Raw IC by Top-Level HPO Category",
        ha="left",
        va="top",
        fontsize=22,
        weight="bold",
    )
    fig.text(
        0.36,
        0.952,
        "Violin = distribution, box = IQR, whiskers = 5th-95th percentile, red line = median, green dot = mean. "
        "Multi-category HPO terms are counted once in each category.",
        ha="left",
        va="top",
        fontsize=11,
        color=GRAY,
    )
    ax.set_xlabel("Phen2Disease-like raw IC", labelpad=12)
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D9DEE7", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.text(
        1.005,
        1.004,
        "terms",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=GRAY,
    )

    fig.subplots_adjust(left=0.36, right=0.91, top=0.88, bottom=0.08)
    save_all(fig, "phen2disease_raw_ic_by_category")


def write_category_summary(grouped: dict[str, dict[str, Any]]) -> None:
    rows = sorted(grouped.values(), key=lambda row: row["label"])
    output_path = FIG_DIR / "phen2disease_raw_ic_by_category_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "n", "min", "p05", "q1", "median", "q3", "p95", "max", "mean"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "category": row["label"],
                    "n": row["n"],
                    "min": row["min"],
                    "p05": row["p05"],
                    "q1": row["q1"],
                    "median": row["median"],
                    "q3": row["q3"],
                    "p95": row["p95"],
                    "max": row["max"],
                    "mean": row["mean"],
                }
            )


def main() -> None:
    terms = load_terms()
    values = sorted(value for term in terms if (value := raw_ic(term)) is not None)
    grouped = category_values(terms)
    plot_ic_distribution(values)
    plot_ic_by_category(grouped)
    write_category_summary(grouped)
    print(f"wrote {FIG_DIR / 'phen2disease_raw_ic_distribution.png'}")
    print(f"wrote {FIG_DIR / 'phen2disease_raw_ic_by_category.png'}")
    print(f"wrote {FIG_DIR / 'phen2disease_raw_ic_by_category_summary.csv'}")


if __name__ == "__main__":
    main()
