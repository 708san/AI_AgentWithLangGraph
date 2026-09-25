#!/usr/bin/env python3
"""Analyze relationships between input HPO information and PCF rankings."""

from __future__ import annotations

import argparse
import csv
import json
import mmap
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TSV = PROJECT_ROOT / "Data/NewData/phenopacket_v1.0.27_New/phenopacket_test_metadata_v0.1.27_GM_v1.1.5_with_phenopacket_data.tsv"
DEFAULT_PCF_DIR = PROJECT_ROOT / "tryToolsForNewData/PCF"
DEFAULT_IC = PROJECT_ROOT / "HPO_importance/HPO_importance.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "tryToolsForNewData/hpo_pcf_relationship"
K_VALUES = (1, 3, 5, 10, 30)
JQ_BIN = shutil.which("jq")


def omim_key(value: Any) -> str | None:
    text = str(value or "").strip()
    match = re.fullmatch(r"(?:OMIM:)?(\d+)", text, flags=re.IGNORECASE)
    return f"OMIM:{match.group(1)}" if match else None


def split_hpo(value: Any) -> list[str]:
    result = []
    seen = set()
    for item in re.split(r"[;,]", str(value or "")):
        item = item.strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def load_ic(path: Path) -> dict[str, float]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["HPO_id"]): float(row.get("information_content", 0.0)) for row in rows}


def load_ic_labels(path: Path) -> dict[str, str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["HPO_id"]): str(row.get("HPO_label", "")) for row in rows}


def _parse_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed = []
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        candidate_id = omim_key(row.get("omim_id") or row.get("OMIM_id"))
        if not candidate_id:
            continue
        raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
        matched = split_hpo(row.get("matched_hpo_id") or raw.get("matched_hpo_id"))
        parsed.append({
            "rank": int(row.get("rank") or position),
            "omim_id": candidate_id,
            "score": row.get("score"),
            "matched_hpo_id": matched,
            "count_hpo_id": row.get("count_hpo_id") or raw.get("count_hpo_id"),
        })
    return parsed


def load_pcf(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        return [], "missing_file"
    # The saved response repeats every result in ``ranking``, ``all_results``
    # and ``raw_response``.  Read only the first 30 objects from the ranking
    # array with a memory map, avoiding a 30--40 MB JSON parse per case.
    try:
        with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            marker = mapped.find(b'"ranking"')
            if marker >= 0:
                array_start = mapped.find(b"[", marker)
                if array_start >= 0:
                    rows: list[dict[str, Any]] = []
                    pos = array_start + 1
                    while len(rows) < 30 and pos < len(mapped):
                        while pos < len(mapped) and mapped[pos] in b" \t\r\n,":
                            pos += 1
                        if pos >= len(mapped) or mapped[pos] == ord("]"):
                            break
                        if mapped[pos] != ord("{"):
                            break
                        start = pos
                        depth = 0
                        in_string = False
                        escaped = False
                        while pos < len(mapped):
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
                                    rows.append(json.loads(mapped[start : pos + 1]))
                                    pos += 1
                                    break
                            pos += 1
                        else:
                            break
                    status = "ok" if b'"status": "ok"' in mapped[:marker] else "empty_or_error"
                    return _parse_candidate_rows(rows), status
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    if JQ_BIN:
        expression = (
            ".result | {status: (.status // null), ranking: "
            "[(.ranking // .all_results // .results // [])[:30][] | "
            "{rank, score, omim_id, matched_hpo_id: (.matched_hpo_id // .raw.matched_hpo_id // ''), "
            "count_hpo_id: (.count_hpo_id // .raw.count_hpo_id // null)}]}"
        )
        try:
            completed = subprocess.run([JQ_BIN, "-c", expression, str(path)], check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)
            return _parse_candidate_rows(payload.get("ranking", [])), "ok" if payload.get("status") == "ok" else "empty_or_error"
        except Exception:
            pass
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", payload) if isinstance(payload, dict) else payload
        if isinstance(result, dict):
            rows = result.get("ranking") or result.get("all_results") or []
            status = "ok" if result.get("status") == "ok" else "empty_or_error"
        else:
            rows, status = result, "ok"
        return _parse_candidate_rows(rows[:30] if isinstance(rows, list) else []), status
    except Exception as exc:
        return [], f"invalid_json:{type(exc).__name__}"


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        output = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                output[order[k]] = average
            i = j + 1
        return output
    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if len(set(rx)) > 1 and len(set(ry)) > 1 else None


def quartile_labels(values: list[float]) -> list[str]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    labels = [""] * len(values)
    names = ("Q1_low", "Q2", "Q3", "Q4_high")
    for position, index in enumerate(order):
        labels[index] = names[min(3, position * 4 // max(1, len(values)))]
    return labels


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--pcf-dir", type=Path, default=DEFAULT_PCF_DIR)
    parser.add_argument("--ic-file", type=Path, default=DEFAULT_IC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for path in (args.tsv, args.ic_file):
        if not path.is_file():
            parser.error(f"file not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ic = load_ic(args.ic_file)
    ic_labels = load_ic_labels(args.ic_file)
    with args.tsv.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="\t"))

    def one(row):
        image_id = str(row.get("image_id", "")).strip()
        present = split_hpo(row.get("pp_present_hpo"))
        truth = omim_key(row.get("disease_id") or row.get("pp_disease_id") or row.get("pp_omim_full") or row.get("pp_omim"))
        values = [ic.get(term, 0.0) for term in present]
        max_index = max(range(len(values)), key=values.__getitem__) if values else None
        max_hpo = present[max_index] if max_index is not None else ""
        max_ic = values[max_index] if max_index is not None else 0.0
        pcf, status = load_pcf(args.pcf_dir / f"{image_id}.json")
        denominator = sum(values)
        correct = next((item for item in pcf if item["omim_id"] == truth), None)
        case = {
            "image_id": image_id,
            "patient_id": row.get("patient_id", ""),
            "truth_omim": truth,
            "present_hpo_count": len(present),
            "input_ic_sum": denominator,
            "input_ic_mean": denominator / len(values) if values else 0.0,
            "input_ic_max": max_ic,
            "input_ic_max_hpo_id": max_hpo,
            "input_ic_max_hpo_label": ic_labels.get(max_hpo, ""),
            "input_ic_nonzero_count": sum(value > 0 for value in values),
            "input_ic_nonzero_mean": sum(value for value in values if value > 0) / sum(value > 0 for value in values) if any(value > 0 for value in values) else 0.0,
            "zero_ic_hpo_count": sum(value == 0 for value in values),
            "pcf_status": status,
            "correct_rank": correct["rank"] if correct else None,
            "correct_pcf_score": correct.get("score") if correct else None,
            "correct_matched_hpo_count": None,
            "correct_hpo_coverage": None,
            "correct_weighted_hpo_coverage": None,
        }
        present_set = set(present)
        input_weighted = sum(values)
        for k in K_VALUES:
            case[f"recall_at_{k}"] = bool(truth and any(item["omim_id"] == truth for item in pcf[:k]))
        if correct:
            matches = present_set & set(correct["matched_hpo_id"])
            matched_weight = sum(ic.get(term, 0.0) for term in matches)
            case["correct_matched_hpo_count"] = len(matches)
            case["correct_hpo_coverage"] = len(matches) / len(present_set) if present_set else 0.0
            case["correct_weighted_hpo_coverage"] = matched_weight / input_weighted if input_weighted else 0.0
        return case, pcf

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(one, source_rows))
    cases = [item[0] for item in results]
    present_by_image = {
        str(row.get("image_id", "")).strip(): set(split_hpo(row.get("pp_present_hpo")))
        for row in source_rows
    }
    candidate_rows = []
    for case, ranking in results:
        for candidate in ranking:
            matches = set(candidate["matched_hpo_id"]) & present_by_image.get(case["image_id"], set())
            candidate_rows.append({**case, **{f"candidate_{key}": value for key, value in candidate.items()}, "candidate_matched_hpo_count": len(matches)})

    for feature in ("input_ic_sum", "input_ic_mean", "input_ic_max"):
        labels = quartile_labels([float(case[feature]) for case in cases])
        for case, label in zip(cases, labels):
            case[f"{feature}_bin"] = label

    case_fields = list(cases[0].keys()) if cases else []
    write_csv(args.output_dir / "per_case_metrics.csv", cases, case_fields)
    write_csv(args.output_dir / "candidate_metrics_top30.csv", candidate_rows, list(candidate_rows[0].keys()) if candidate_rows else [])
    (args.output_dir / "per_case_metrics.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bin_rows = []
    for feature in ("input_ic_sum", "input_ic_mean", "input_ic_max"):
        for label in ("Q1_low", "Q2", "Q3", "Q4_high"):
            group = [case for case in cases if case[f"{feature}_bin"] == label]
            for k in K_VALUES:
                hits = sum(bool(case[f"recall_at_{k}"]) for case in group)
                bin_rows.append({"feature": feature, "bin": label, "k": k, "hits": hits, "n": len(group), "recall": hits / len(group) if group else 0.0})
    write_csv(args.output_dir / "recall_by_information_bin.csv", bin_rows, ["feature", "bin", "k", "hits", "n", "recall"])

    correlation_rows = []
    for x_feature in ("present_hpo_count", "input_ic_sum", "input_ic_mean", "input_ic_max", "correct_hpo_coverage", "correct_weighted_hpo_coverage"):
        for y_feature in ("correct_rank", "correct_pcf_score"):
            if y_feature == "correct_rank":
                # Treat a correct disease absent from the saved top-30 as a
                # censored rank of 31 so failed cases remain in this analysis.
                pairs = [
                    (float(case[x_feature]), float(case["correct_rank"] or 31))
                    for case in cases
                    if case[x_feature] is not None and case["truth_omim"]
                ]
            else:
                pairs = [(float(case[x_feature]), float(case[y_feature])) for case in cases if case[x_feature] is not None and case[y_feature] is not None]
            correlation_rows.append({"x": x_feature, "y": y_feature, "n": len(pairs), "spearman_r": spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs])})
    write_csv(args.output_dir / "correlation_summary.csv", correlation_rows, ["x", "y", "n", "spearman_r"])

    # Rank distribution and information-vs-rank scatter plots.
    ranks = [case["correct_rank"] if case["correct_rank"] is not None and case["correct_rank"] <= 30 else 31 for case in cases]
    labels = ["1", "2-3", "4-5", "6-10", "11-30", ">30/not returned"]
    intervals = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 30), (31, 31)]
    counts = [sum(lo <= r <= hi for r in ranks) for lo, hi in intervals]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, counts, color="#377eb8")
    ax.set_ylabel("Cases")
    ax.set_title("PCF Correct OMIM Rank Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(args.output_dir / "rank_distribution.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plots = [("input_ic_sum", "Correct rank (31 = >30/not returned)", "IC sum vs correct rank"), ("input_ic_mean", "Correct rank (31 = >30/not returned)", "IC mean vs correct rank"), ("input_ic_max", "Correct rank (31 = >30/not returned)", "IC max vs correct rank"), ("input_ic_sum", "Correct PCF score", "IC sum vs PCF score")]
    for ax, (x_feature, y_feature, title) in zip(axes, plots):
        xs = [float(case[x_feature]) for case in cases]
        if y_feature.startswith("Correct rank"):
            ys = [case["correct_rank"] if case["correct_rank"] is not None and case["correct_rank"] <= 30 else 31 for case in cases]
        else:
            xs = [float(case[x_feature]) for case in cases if case["correct_pcf_score"] is not None]
            ys = [float(case["correct_pcf_score"]) for case in cases if case["correct_pcf_score"] is not None]
        ax.scatter(xs, ys, s=12, alpha=0.45)
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(args.output_dir / "information_vs_pcf.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, feature in zip(axes, ("input_ic_sum", "input_ic_mean", "input_ic_max")):
        group_values = []
        group_labels = []
        for label in ("Q1_low", "Q2", "Q3", "Q4_high"):
            group = [case for case in cases if case[f"{feature}_bin"] == label]
            group_labels.append(label)
            group_values.append([sum(bool(case[f"recall_at_{k}"]) for case in group) / len(group) if group else 0.0 for k in K_VALUES])
        x = np.arange(len(K_VALUES)); width = 0.19
        for index, label in enumerate(group_labels):
            ax.bar(x + (index - 1.5) * width, group_values[index], width, label=label)
        ax.set_xticks(x, [f"@{k}" for k in K_VALUES]); ax.set_ylim(0, 1.05); ax.set_title(f"Recall by {feature} quartile"); ax.set_ylabel("Recall"); ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.output_dir / "recall_by_information_bin.png", dpi=200); plt.close(fig)

    rank_counts = {
        "1": sum(case["correct_rank"] == 1 for case in cases),
        "2-3": sum(case["correct_rank"] is not None and 2 <= case["correct_rank"] <= 3 for case in cases),
        "4-5": sum(case["correct_rank"] is not None and 4 <= case["correct_rank"] <= 5 for case in cases),
        "6-10": sum(case["correct_rank"] is not None and 6 <= case["correct_rank"] <= 10 for case in cases),
        "11-30": sum(case["correct_rank"] is not None and 11 <= case["correct_rank"] <= 30 for case in cases),
        ">30/not_returned": sum(case["correct_rank"] is None or case["correct_rank"] > 30 for case in cases),
    }
    summary = {
        "n_cases": len(cases),
        "pcf_non_ok": sum(case["pcf_status"] != "ok" for case in cases),
        "correct_in_top30": sum(case["correct_rank"] is not None and case["correct_rank"] <= 30 for case in cases),
        "rank_counts": rank_counts,
        "recall": {str(k): sum(bool(case[f"recall_at_{k}"]) for case in cases) / len(cases) if cases else 0.0 for k in K_VALUES},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
