#!/usr/bin/env python3
"""Evaluate score thresholds as dynamic candidate-count cutoffs.

For every saved PCF/GM ranking, this script counts candidates passing a
tool-specific threshold and checks whether the exact ground-truth OMIM also
passes it.  It compares those dynamic policies with fixed Top-k policies.
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
K_VALUES = (1, 3, 5, 10, 30)

THRESHOLDS: dict[str, tuple[float, ...]] = {
    "pcf_score": (0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 1.0),
    "gm_distance": (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10),
    "gm_score": (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
    "gm_gestalt_score": (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
}

NUMBER_RE = rb"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
FIELD_RE = {
    "score": re.compile(rb'"score"\s*:\s*(%s)' % NUMBER_RE),
    "distance": re.compile(rb'"distance"\s*:\s*(%s)' % NUMBER_RE),
    "gestalt_score": re.compile(rb'"gestalt_score"\s*:\s*(%s)' % NUMBER_RE),
}


def iter_array_objects(mapped: mmap.mmap, array_start: int) -> Iterator[bytes]:
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


def first_number(raw: bytes, field: str) -> float | None:
    match = FIELD_RE[field].search(raw)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def scan_counts(path: Path, tool: str, thresholds: tuple[float, ...]) -> dict[str, Any]:
    result = {
        "status": "missing_file" if not path.is_file() else "empty_or_error",
        "total_candidates": 0,
        "counts": {threshold: 0 for threshold in thresholds},
        "nonmonotonic": False,
    }
    if not path.is_file():
        return result
    metric = "score" if tool == "PCF" else tool
    descending = tool in {"PCF", "score", "gestalt_score"}
    try:
        with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            marker = mapped.find(b'"ranking"')
            array_start = mapped.find(b"[", marker) if marker >= 0 else -1
            if array_start < 0:
                return result
            result["status"] = "ok" if b'"status": "ok"' in mapped[:marker] else "empty_or_error"
            previous = None
            for raw in iter_array_objects(mapped, array_start):
                value = first_number(raw, metric)
                result["total_candidates"] += 1
                if value is None:
                    continue
                if previous is not None:
                    if descending and value > previous + 1e-12:
                        result["nonmonotonic"] = True
                    if not descending and value < previous - 1e-12:
                        result["nonmonotonic"] = True
                previous = value
                for threshold in thresholds:
                    passes = value >= threshold if descending else value <= threshold
                    if passes:
                        result["counts"][threshold] += 1
                # The tool rankings are intended to be sorted by this metric.
                # Once the lowest (or highest) threshold can no longer pass,
                # later ranks cannot contribute to any threshold count.  This
                # avoids parsing the duplicated low-scoring tail of the huge
                # PCF JSON files.
                if descending and value < min(thresholds):
                    break
                if not descending and value > max(thresholds):
                    break
            return result
    except (OSError, ValueError):
        result["status"] = "invalid_json_or_io"
        return result


def numeric(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        try:
            if value is not None and str(value) != "":
                out.append(float(value))
        except (TypeError, ValueError):
            pass
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summary_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None}
    return {"mean": statistics.mean(values), "median": statistics.median(values), "p95": float(np.percentile(values, 95))}


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
    with (args.output_dir / "truth_tool_metrics.csv").open(newline="", encoding="utf-8") as handle:
        truth_rows = {row["image_id"]: row for row in csv.DictReader(handle)}

    per_case: list[dict[str, Any]] = []
    aggregate: dict[str, list[dict[str, Any]]] = {metric: [] for metric in THRESHOLDS}
    monotonic = {"pcf_score": 0, "gm_distance": 0, "gm_score": 0, "gm_gestalt_score": 0}

    for index, source in enumerate(source_rows, start=1):
        image_id = str(source.get("image_id", "")).strip()
        truth = truth_rows.get(image_id, {})
        pcf = scan_counts(args.pcf_dir / f"{image_id}.json", "PCF", THRESHOLDS["pcf_score"])
        gm_distance = scan_counts(args.gm_dir / f"{image_id}.json", "distance", THRESHOLDS["gm_distance"])
        gm_score = scan_counts(args.gm_dir / f"{image_id}.json", "score", THRESHOLDS["gm_score"])
        gm_gestalt = scan_counts(args.gm_dir / f"{image_id}.json", "gestalt_score", THRESHOLDS["gm_gestalt_score"])
        scans = {"pcf_score": pcf, "gm_distance": gm_distance, "gm_score": gm_score, "gm_gestalt_score": gm_gestalt}
        for metric, scan in scans.items():
            if scan["nonmonotonic"]:
                monotonic[metric] += 1

        case = {"image_id": image_id, "patient_id": source.get("patient_id", ""), "truth_omim": truth.get("truth_omim", "")}
        for metric, scan in scans.items():
            prefix = metric
            truth_value = truth.get({"pcf_score": "pcf_truth_score", "gm_distance": "gm_truth_distance", "gm_score": "gm_truth_score", "gm_gestalt_score": "gm_truth_gestalt_score"}[metric])
            try:
                truth_value_num = float(truth_value) if truth_value not in (None, "") else None
            except ValueError:
                truth_value_num = None
            direction_high = metric in {"pcf_score", "gm_score", "gm_gestalt_score"}
            for threshold in THRESHOLDS[metric]:
                key = f"{prefix}_{threshold:g}"
                case[f"{key}_candidate_count"] = scan["counts"][threshold]
                case[f"{key}_truth_pass"] = bool(truth_value_num is not None and (truth_value_num >= threshold if direction_high else truth_value_num <= threshold))
        per_case.append(case)
        for metric in THRESHOLDS:
            aggregate[metric].append({"case": case, "truth": truth})
        if index % 25 == 0 or index == len(source_rows):
            print(f"processed {index}/{len(source_rows)}", flush=True)

    case_fields = list(per_case[0].keys()) if per_case else []
    write_csv(args.output_dir / "dynamic_cutoff_per_case.csv", per_case, case_fields)

    dynamic_rows: list[dict[str, Any]] = []
    for metric, thresholds in THRESHOLDS.items():
        direction = "higher_is_better" if metric in {"pcf_score", "gm_score", "gm_gestalt_score"} else "lower_is_better"
        for threshold in thresholds:
            counts = [int(case[f"{metric}_{threshold:g}_candidate_count"]) for case in per_case]
            hits = [bool(case[f"{metric}_{threshold:g}_truth_pass"]) for case in per_case]
            stats = summary_stats([float(count) for count in counts])
            total = sum(counts)
            hit_count = sum(hits)
            dynamic_rows.append({"method": metric, "direction": direction, "threshold": threshold, "n_cases": len(per_case), "hits": hit_count, "recall": hit_count / len(per_case) if per_case else 0.0, "total_candidates": total, "micro_precision": hit_count / total if total else 0.0, "zero_candidate_cases": sum(count == 0 for count in counts), **stats})
    write_csv(args.output_dir / "dynamic_cutoff_summary.csv", dynamic_rows, ["method", "direction", "threshold", "n_cases", "hits", "recall", "total_candidates", "micro_precision", "zero_candidate_cases", "mean", "median", "p95"])

    fixed_rows: list[dict[str, Any]] = []
    for tool, truth_prefix, ranking_size in (("PCF", "pcf", None), ("GM", "gm", None)):
        for k in K_VALUES:
            hits = sum(bool(truth.get(f"{truth_prefix}_truth_rank") and int(truth[f"{truth_prefix}_truth_rank"]) <= k) for truth in truth_rows.values())
            total = len(source_rows) * k
            fixed_rows.append({"method": f"{tool}_Top{k}", "k": k, "n_cases": len(source_rows), "hits": hits, "recall": hits / len(source_rows) if source_rows else 0.0, "mean_candidates": float(k), "median_candidates": float(k), "p95_candidates": float(k), "micro_precision": hits / total if total else 0.0})
    write_csv(args.output_dir / "fixed_topk_summary.csv", fixed_rows, ["method", "k", "n_cases", "hits", "recall", "mean_candidates", "median_candidates", "p95_candidates", "micro_precision"])

    # Recall versus candidate workload. Each panel is one metric; labels show
    # the threshold, so the useful operating points are easy to compare.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric in zip(axes.flat, THRESHOLDS):
        subset = [row for row in dynamic_rows if row["method"] == metric]
        xs = [row["mean"] for row in subset]
        ys = [row["recall"] for row in subset]
        ax.plot(xs, ys, marker="o", color="#377eb8")
        for row in subset:
            ax.annotate(f"{row['threshold']:g}", (row["mean"], row["recall"]), fontsize=7, xytext=(2, 3), textcoords="offset points")
        ax.set_xlabel("Mean candidates passing threshold")
        ax.set_ylabel("Recall")
        ax.set_title(metric)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    fig.suptitle("Dynamic score cutoffs: Recall versus candidate workload")
    fig.tight_layout(); fig.savefig(args.output_dir / "dynamic_cutoff_recall_workload.png", dpi=200); plt.close(fig)

    # Threshold curves are easier to interpret when the threshold itself is on
    # the x-axis. Workload is shown as a dashed secondary axis.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric in zip(axes.flat, THRESHOLDS):
        subset = [row for row in dynamic_rows if row["method"] == metric]
        x = [row["threshold"] for row in subset]
        ax.plot(x, [row["recall"] for row in subset], marker="o", label="Recall", color="#377eb8")
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel("Threshold"); ax.set_ylabel("Recall", color="#377eb8")
        ax2 = ax.twinx(); ax2.plot(x, [row["mean"] for row in subset], marker="x", linestyle="--", label="Mean candidates", color="#e41a1c"); ax2.set_ylabel("Mean candidates", color="#e41a1c")
        ax.set_title(metric); ax.grid(alpha=0.25)
    fig.suptitle("Threshold trade-off")
    fig.tight_layout(); fig.savefig(args.output_dir / "dynamic_cutoff_threshold_curves.png", dpi=200); plt.close(fig)

    summary = {"n_cases": len(source_rows), "dynamic_methods": list(THRESHOLDS), "nonmonotonic_case_count": monotonic, "monotonicity_scope": "scanned ranking prefix through the lowest/highest configured threshold; scan stops once no configured threshold can pass", "note": "Recall counts exact ground-truth OMIM passing the threshold; PCF truth scores missing from the full ranking are failures."}
    (args.output_dir / "dynamic_cutoff_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
