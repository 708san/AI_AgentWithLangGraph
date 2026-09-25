#!/usr/bin/env python3
"""Analyze score distributions for the exact ground-truth OMIM disease.

The saved tool responses contain full rankings.  This script memory-maps each
JSON file and scans only the ranking array, extracting the first 30 candidates
and the full-ranking entry for the exact ground-truth OMIM ID.  Consequently,
cases where the truth is outside Top30 still contribute their score to the
distribution analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import mmap
import re
import statistics
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np

from analyze_hpo_pcf_relationship import DEFAULT_PCF_DIR, DEFAULT_TSV, omim_key


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GM_DIR = PROJECT_ROOT / "tryToolsForNewData/GM"
DEFAULT_OUTPUT = PROJECT_ROOT / "tryToolsForNewData/hpo_pcf_relationship"


def iter_array_objects(mapped: mmap.mmap, array_start: int) -> Iterator[bytes]:
    """Yield first-level object bytes from a JSON array in a memory map."""
    pos = array_start + 1
    length = len(mapped)
    while pos < length:
        while pos < length and mapped[pos] in b" \t\r\n,":
            pos += 1
        if pos >= length or mapped[pos] == ord("]"):
            return
        if mapped[pos] != ord("{"):
            return
        start = pos
        depth = 0
        in_string = False
        escaped = False
        while pos < length:
            byte = mapped[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif byte == ord("\\"):
                    escaped = True
                elif byte == ord('"'):
                    in_string = False
            elif byte == ord('"'):
                in_string = True
            elif byte == ord("{"):
                depth += 1
            elif byte == ord("}"):
                depth -= 1
                if depth == 0:
                    yield bytes(mapped[start : pos + 1])
                    pos += 1
                    break
            pos += 1
        else:
            return


def candidate_omim(row: dict[str, Any]) -> str | None:
    return omim_key(row.get("omim_id") or row.get("OMIM_id") or row.get("id"))


def scan_tool(path: Path, truth: str | None, tool: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "missing_file" if not path.is_file() else "empty_or_error",
        "truth_found_full": False,
        "truth_rank": None,
        "truth_in_top30": False,
        "truth_score": None,
    }
    if tool == "GM":
        result.update({"truth_distance": None, "truth_gestalt_score": None})
    if not path.is_file():
        return result

    try:
        with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            marker = mapped.find(b'"ranking"')
            if marker < 0:
                return result
            array_start = mapped.find(b"[", marker)
            if array_start < 0:
                return result
            result["status"] = "ok" if b'"status": "ok"' in mapped[:marker] else "empty_or_error"
            for position, raw in enumerate(iter_array_objects(mapped, array_start), start=1):
                # Parse the first 30 objects for the exact same ranking used by
                # the Recall analysis.  For later objects, decode only when the
                # bytes contain the target OMIM ID.
                possible_truth = False
                if truth:
                    number = truth.split(":", 1)[1]
                    possible_truth = (
                        f'"omim_id": "{truth}"'.encode() in raw
                        or f'"omim_id":"{truth}"'.encode() in raw
                        or f'"omim_id": {number}'.encode() in raw
                        or f'"omim_id":{number}'.encode() in raw
                    )
                if position <= 30 or possible_truth:
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                else:
                    continue

                candidate = candidate_omim(row)
                if position <= 30 and candidate:
                    if position == 1:
                        result["top1_omim"] = candidate
                    if candidate == truth:
                        result["truth_in_top30"] = True

                if truth and candidate == truth and not result["truth_found_full"]:
                    result["truth_found_full"] = True
                    result["truth_rank"] = int(row.get("rank") or position)
                    result["truth_score"] = row.get("score")
                    if tool == "GM":
                        result["truth_distance"] = row.get("distance")
                        result["truth_gestalt_score"] = row.get("gestalt_score")

                # Once both the Top30 and the truth entry have been found, no
                # lower ranking object is needed.  If truth is absent, the loop
                # naturally scans to the end and records that fact.
                if position >= 30 and (not truth or result["truth_found_full"]):
                    break
            return result
    except (OSError, ValueError):
        result["status"] = "invalid_json_or_io"
        return result


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def numeric(values: list[Any]) -> list[float]:
    output = []
    for value in values:
        try:
            if value is not None and str(value) != "":
                output.append(float(value))
        except (TypeError, ValueError):
            pass
    return output


def distribution_rows(rows: list[dict[str, Any]], feature: str, label: str) -> dict[str, Any]:
    values = numeric([row.get(feature) for row in rows])
    if not values:
        return {"metric": label, "n": 0, "mean": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    q25, q75 = np.percentile(values, [25, 75])
    return {
        "metric": label,
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "q25": float(q25),
        "q75": float(q75),
        "min": min(values),
        "max": max(values),
    }


def mann_whitney(x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    if not x or not y:
        return None, None
    try:
        from scipy.stats import mannwhitneyu

        test = mannwhitneyu(x, y, alternative="two-sided")
        return float(test.statistic), float(test.pvalue)
    except Exception:
        return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--pcf-dir", type=Path, default=DEFAULT_PCF_DIR)
    parser.add_argument("--gm-dir", type=Path, default=DEFAULT_GM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.tsv.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="\t"))

    rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows, start=1):
        image_id = str(source.get("image_id", "")).strip()
        truth = next((omim_key(source.get(field)) for field in ("disease_id", "pp_disease_id", "pp_omim_full", "pp_omim") if omim_key(source.get(field))), None)
        pcf = scan_tool(args.pcf_dir / f"{image_id}.json", truth, "PCF")
        gm = scan_tool(args.gm_dir / f"{image_id}.json", truth, "GM")
        row = {
            "image_id": image_id,
            "patient_id": source.get("patient_id", ""),
            "truth_omim": truth,
            "pcf_status": pcf["status"],
            "pcf_truth_found_full": pcf["truth_found_full"],
            "pcf_truth_rank": pcf["truth_rank"],
            "pcf_truth_in_top30": pcf["truth_in_top30"],
            "pcf_truth_score": pcf["truth_score"],
            "gm_status": gm["status"],
            "gm_truth_found_full": gm["truth_found_full"],
            "gm_truth_rank": gm["truth_rank"],
            "gm_truth_in_top30": gm["truth_in_top30"],
            "gm_truth_distance": gm["truth_distance"],
            "gm_truth_score": gm["truth_score"],
            "gm_truth_gestalt_score": gm["truth_gestalt_score"],
        }
        rows.append(row)
        if index % 50 == 0 or index == len(source_rows):
            print(f"processed {index}/{len(source_rows)}", flush=True)

    case_fields = list(rows[0].keys()) if rows else []
    write_csv(args.output_dir / "truth_tool_metrics.csv", rows, case_fields)
    (args.output_dir / "truth_tool_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    distributions = [
        distribution_rows(rows, "pcf_truth_score", "PCF truth score"),
        distribution_rows(rows, "gm_truth_distance", "GM truth distance"),
        distribution_rows(rows, "gm_truth_score", "GM truth score"),
        distribution_rows(rows, "gm_truth_gestalt_score", "GM truth gestalt_score"),
    ]
    write_csv(args.output_dir / "truth_score_distribution_summary.csv", distributions, ["metric", "n", "mean", "median", "q25", "q75", "min", "max"])

    comparison_rows: list[dict[str, Any]] = []
    comparisons = [
        ("PCF", "pcf_truth_score", "pcf"),
        ("GM distance", "gm_truth_distance", "gm"),
        ("GM score", "gm_truth_score", "gm"),
        ("GM gestalt_score", "gm_truth_gestalt_score", "gm"),
    ]
    for label, feature, prefix in comparisons:
        for group_name, group_rows in (
            ("Top30", [row for row in rows if row[f"{prefix}_truth_in_top30"]]),
            ("NotTop30", [row for row in rows if not row[f"{prefix}_truth_in_top30"]]),
        ):
            comparison_rows.append({"metric": label, "group": group_name, **distribution_rows(group_rows, feature, label)})
        hit = numeric([row.get(feature) for row in rows if row[f"{prefix}_truth_in_top30"]])
        miss = numeric([row.get(feature) for row in rows if not row[f"{prefix}_truth_in_top30"]])
        u, p = mann_whitney(hit, miss)
        comparison_rows.append({"metric": label, "group": "MannWhitneyU_Top30_vs_NotTop30", "n": len(hit) + len(miss), "mean": None, "median": None, "q25": None, "q75": None, "min": None, "max": None, "mann_whitney_u": u, "p_value": p})
    write_csv(args.output_dir / "truth_score_by_recall30.csv", comparison_rows, ["metric", "group", "n", "mean", "median", "q25", "q75", "min", "max", "mann_whitney_u", "p_value"])

    # Overall truth-score distributions.
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for ax, (label, feature, color) in zip(axes, [("PCF score", "pcf_truth_score", "#377eb8"), ("GM distance", "gm_truth_distance", "#e41a1c"), ("GM score", "gm_truth_score", "#4daf4a"), ("GM gestalt_score", "gm_truth_gestalt_score", "#984ea3")]):
        values = numeric([row.get(feature) for row in rows])
        ax.hist(values, bins=25, color=color, alpha=0.8) if values else None
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Cases")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(args.output_dir / "truth_score_distributions.png", dpi=200); plt.close(fig)

    # Top30 vs NotTop30 distributions for each tool metric.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (label, feature, prefix) in zip(axes.flat, comparisons):
        hit = numeric([row.get(feature) for row in rows if row[f"{prefix}_truth_in_top30"]])
        miss = numeric([row.get(feature) for row in rows if not row[f"{prefix}_truth_in_top30"]])
        all_values = hit + miss
        if all_values:
            bins = np.linspace(min(all_values), max(all_values), 24) if min(all_values) < max(all_values) else 10
            ax.hist(hit, bins=bins, alpha=0.65, label=f"Top30 (n={len(hit)})", color="#377eb8")
            ax.hist(miss, bins=bins, alpha=0.65, label=f"Not Top30 (n={len(miss)})", color="#e41a1c")
        ax.set_title(f"{label}: recall@30 groups")
        ax.set_xlabel(label)
        ax.set_ylabel("Cases")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.output_dir / "truth_score_by_recall30.png", dpi=200); plt.close(fig)

    summary = {
        "n_cases": len(rows),
        "pcf_non_ok": sum(row["pcf_status"] != "ok" for row in rows),
        "gm_non_ok": sum(row["gm_status"] != "ok" for row in rows),
        "pcf_truth_found_full": sum(row["pcf_truth_found_full"] for row in rows),
        "gm_truth_found_full": sum(row["gm_truth_found_full"] for row in rows),
        "pcf_truth_in_top30": sum(row["pcf_truth_in_top30"] for row in rows),
        "gm_truth_in_top30": sum(row["gm_truth_in_top30"] for row in rows),
        "matching": "exact normalized OMIM ID",
        "note": "Top30 grouping uses the ranking position; scores are extracted from the full saved ranking when available.",
    }
    (args.output_dir / "truth_score_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
