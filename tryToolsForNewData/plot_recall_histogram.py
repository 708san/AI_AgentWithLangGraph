#!/usr/bin/env python3
"""Calculate exact OMIM Recall@k for all non-empty subsets of PCF, GM, and Zero-shot.

The denominator is every row in the input TSV. For each subset, ``union@k``
is the union of the selected tools' top-k IDs for each image.
Predicted OMIM IDs are compared after only one normalization: a numeric GM
ID and ``OMIM:<number>`` are represented as the same ``OMIM:<number>`` key.
"""

from __future__ import annotations

import argparse
import csv
import json
import mmap
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TSV = (
    PROJECT_ROOT
    / "Data/NewData/phenopacket_v1.0.27_New"
    / "phenopacket_test_metadata_v0.1.27_GM_v1.1.5_with_phenopacket_data.tsv"
)
DEFAULT_RESULT_DIR = PROJECT_ROOT / "tryToolsForNewData"
K_VALUES = (1, 3, 5, 10, 30)
MAX_RANK = max(K_VALUES)
JQ_BIN = shutil.which("jq")
OMIM_FIELD_RE = re.compile(rb'"omim_id"\s*:\s*(?:"([^"]*)"|(\d+))')


def omim_key(value: Any) -> str | None:
    """Return an exact OMIM key without using disease-name matching."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.fullmatch(r"(?:OMIM:)?(\d+)", text, flags=re.IGNORECASE)
    return f"OMIM:{match.group(1)}" if match else None


def ground_truth(row: dict[str, str]) -> str | None:
    for field in ("disease_id", "pp_disease_id", "pp_omim_full", "pp_omim"):
        key = omim_key(row.get(field))
        if key:
            return key
    return None


def load_ranking(path: Path) -> tuple[list[str], str]:
    if not path.is_file():
        return [], "missing_file"
    # The generated files duplicate each API item under ``ranking`` and
    # ``raw_response``.  Memory-map the file and stop after Recall@30 IDs;
    # this avoids parsing tens of megabytes of raw fields for every case.
    try:
        with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            ranking_start = mapped.find(b'"ranking"')
            if ranking_start >= 0:
                ids: list[str] = []
                seen: set[str] = set()
                # Do not accidentally read duplicate OMIM IDs from the raw
                # LLM/MCP trace when a ranking contains fewer than 30 items.
                ranking_end = mapped.find(b'"trace"', ranking_start)
                if ranking_end < 0:
                    ranking_end = len(mapped)
                for match in OMIM_FIELD_RE.finditer(mapped, ranking_start, ranking_end):
                    raw_value = match.group(1) if match.group(1) is not None else match.group(2)
                    if raw_value is None:
                        continue
                    value = raw_value.decode("utf-8", errors="replace")
                    key = omim_key(value)
                    if key and key not in seen:
                        ids.append(key)
                        seen.add(key)
                    if len(ids) >= MAX_RANK:
                        break
                status = "ok" if b'"status": "ok"' in mapped[:ranking_start] else "empty_or_error"
                return ids, status
    except (OSError, ValueError):
        pass
    # The saved PCF files can contain a very large raw response.  jq extracts
    # only the first 30 IDs without materializing that response in Python.
    if JQ_BIN:
        expression = (
            f"{{status: (.result.status // null), "
            f"ids: [.result.ranking[:{MAX_RANK}][]?.omim_id]}}"
        )
        try:
            completed = subprocess.run(
                [JQ_BIN, "-c", expression, str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            extracted = json.loads(completed.stdout)
            ids = []
            seen: set[str] = set()
            for value in extracted.get("ids", []):
                key = omim_key(value)
                if key and key not in seen:
                    ids.append(key)
                    seen.add(key)
            status = "ok" if extracted.get("status") == "ok" else "empty_or_error"
            return ids, status
        except Exception:
            # Fall back to the pure-Python reader for environments without a
            # compatible jq expression or for older result JSON layouts.
            pass
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], f"invalid_json:{type(exc).__name__}"
    result = payload.get("result", payload) if isinstance(payload, dict) else payload
    if isinstance(result, dict):
        rows = result.get("ranking") or result.get("all_results") or []
    elif isinstance(result, list):
        rows = result
    else:
        rows = []
    ids: list[str] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        key = omim_key(item.get("omim_id") or item.get("OMIM_id"))
        if key and key not in seen:
            ids.append(key)
            seen.add(key)
    status = "ok" if isinstance(result, dict) and result.get("status") == "ok" else "empty_or_error"
    return ids, status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()
    args.tsv = args.tsv.expanduser().resolve()
    args.result_dir = args.result_dir.expanduser().resolve()

    with args.tsv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    per_case: list[dict[str, Any]] = []
    # All non-empty subsets of the three tools.  Each subset is evaluated as
    # the union of the corresponding top-k rankings.
    methods = (
        "PCF", "GM", "ZeroShot",
        "PCF_or_GM", "PCF_or_ZeroShot", "GM_or_ZeroShot",
        "PCF_or_GM_or_ZeroShot",
    )
    counts = {method: {k: 0 for k in K_VALUES} for method in methods}
    missing = {"PCF": 0, "GM": 0, "ZeroShot": 0}
    invalid_truth = 0

    for row_index, row in enumerate(rows, start=1):
        image_id = str(row.get("image_id", "")).strip()
        truth = ground_truth(row)
        if truth is None:
            invalid_truth += 1
        pcf_ids, pcf_status = load_ranking(args.result_dir / "PCF" / f"{image_id}.json")
        gm_ids, gm_status = load_ranking(args.result_dir / "GM" / f"{image_id}.json")
        zero_ids, zero_status = load_ranking(args.result_dir / "LLM" / f"{image_id}.json")
        if pcf_status != "ok":
            missing["PCF"] += 1
        if gm_status != "ok":
            missing["GM"] += 1
        if zero_status != "ok":
            missing["ZeroShot"] += 1
        row_result = {
            "image_id": image_id,
            "patient_id": row.get("patient_id", ""),
            "truth_omim": truth,
            "pcf_status": pcf_status,
            "gm_status": gm_status,
            "zeroshot_status": zero_status,
            "pcf_ranking": pcf_ids,
            "gm_ranking": gm_ids,
            "zeroshot_ranking": zero_ids,
            "hits": {},
        }
        for k in K_VALUES:
            hit_pcf = truth is not None and truth in set(pcf_ids[:k])
            hit_gm = truth is not None and truth in set(gm_ids[:k])
            hit_zero = truth is not None and truth in set(zero_ids[:k])
            pcf_set, gm_set, zero_set = set(pcf_ids[:k]), set(gm_ids[:k]), set(zero_ids[:k])
            hit_union = truth is not None and truth in (pcf_set | gm_set)
            hit_pcf_zero = truth is not None and truth in (pcf_set | zero_set)
            hit_gm_zero = truth is not None and truth in (gm_set | zero_set)
            hit_all = truth is not None and truth in (pcf_set | gm_set | zero_set)
            row_result["hits"][str(k)] = {
                "PCF": hit_pcf, "GM": hit_gm, "ZeroShot": hit_zero,
                "PCF_or_GM": hit_union, "PCF_or_ZeroShot": hit_pcf_zero,
                "GM_or_ZeroShot": hit_gm_zero, "PCF_or_GM_or_ZeroShot": hit_all,
            }
            counts["PCF"][k] += int(hit_pcf)
            counts["GM"][k] += int(hit_gm)
            counts["ZeroShot"][k] += int(hit_zero)
            counts["PCF_or_GM"][k] += int(hit_union)
            counts["PCF_or_ZeroShot"][k] += int(hit_pcf_zero)
            counts["GM_or_ZeroShot"][k] += int(hit_gm_zero)
            counts["PCF_or_GM_or_ZeroShot"][k] += int(hit_all)
        per_case.append(row_result)
        if row_index % 50 == 0 or row_index == len(rows):
            print(f"processed {row_index}/{len(rows)}", flush=True)

    denominator = len(rows)
    rates = {
        method: {str(k): counts[method][k] / denominator if denominator else 0.0 for k in K_VALUES}
        for method in methods
    }
    summary = {
        "tsv": str(args.tsv),
        "result_dir": str(args.result_dir),
        "matching": "exact OMIM ID",
        "union_definition": "each subset is evaluated as the union of its tools' top-k IDs",
        "zeroshot_source": "LLM/{image_id}.json",
        "n_cases": denominator,
        "invalid_truth_count": invalid_truth,
        "non_ok_output_count": missing,
        "hit_counts": counts,
        "recall": rates,
        "per_case": per_case,
    }
    (args.result_dir / "recall_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (args.result_dir / "recall_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "k", "hits", "n_cases", "recall"])
        for method in methods:
            for k in K_VALUES:
                writer.writerow([method, k, counts[method][k], denominator, rates[method][str(k)]])

    x = np.arange(len(K_VALUES))
    single_methods = ("PCF", "GM", "ZeroShot")
    subset_methods = ("PCF_or_GM", "PCF_or_ZeroShot", "GM_or_ZeroShot", "PCF_or_GM_or_ZeroShot")
    width = 0.18
    fig, (ax_single, ax_subset) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    colors = {
        "PCF": "#377eb8", "GM": "#e41a1c", "ZeroShot": "#984ea3",
        "PCF_or_GM": "#4daf4a", "PCF_or_ZeroShot": "#ff7f00",
        "GM_or_ZeroShot": "#a65628", "PCF_or_GM_or_ZeroShot": "#00a6a6",
    }
    labels = {
        "PCF": "PCF", "GM": "GestaltMatcher", "ZeroShot": "GPT Zero-shot",
        "PCF_or_GM": "PCF ∪ GM", "PCF_or_ZeroShot": "PCF ∪ Zero-shot",
        "GM_or_ZeroShot": "GM ∪ Zero-shot", "PCF_or_GM_or_ZeroShot": "PCF ∪ GM ∪ Zero-shot",
    }
    def draw_panel(ax, panel_methods, title):
        offsets = np.linspace(-(len(panel_methods) - 1) * width / 2,
                              (len(panel_methods) - 1) * width / 2,
                              len(panel_methods))
        for offset, method in zip(offsets, panel_methods):
            values = [rates[method][str(k)] for k in K_VALUES]
            bars = ax.bar(x + offset, values, width, label=labels[method], color=colors[method])
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=2, fontsize=7, rotation=90)
        ax.set_xticks(x, [f"@{k}" for k in K_VALUES])
        ax.set_xlabel("Rank cutoff")
        ax.set_title(title)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", fontsize=9, frameon=True)

    draw_panel(ax_single, single_methods, "Single tools")
    draw_panel(ax_subset, subset_methods, "Tool subsets (top-k union)")
    ax_single.set_ylabel("Exact OMIM recall")
    ax_single.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    fig.suptitle(f"Exact OMIM Recall by Rank (n={denominator})", fontsize=15)
    fig.tight_layout()
    fig.savefig(args.result_dir / "recall_histogram.png", dpi=200)
    fig.savefig(args.result_dir / "recall_histogram.svg")
    plt.close(fig)

    print(f"n_cases={denominator}")
    print(f"invalid_truth_count={invalid_truth}")
    print(f"non_ok_output_count={missing}")
    for method in methods:
        print(method, " ".join(f"R@{k}={rates[method][str(k)]:.4f}" for k in K_VALUES))
    print(f"saved: {args.result_dir / 'recall_histogram.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
