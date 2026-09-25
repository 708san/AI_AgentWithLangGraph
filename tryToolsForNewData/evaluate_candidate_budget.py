#!/usr/bin/env python3
"""Evaluate candidate coverage versus candidate-count cost.

The default run evaluates fixed rank cutoffs and tool allocations.  An
optional directory of LLM decisions can be supplied; each file must contain
``pcf_k``, ``gm_k`` and ``zeroshot_k`` (or a nested ``decision`` object).
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from plot_recall_histogram import ground_truth, load_ranking

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TSV = ROOT / "Data/NewData/phenopacket_v1.0.27_New/phenopacket_test_metadata_v0.1.27_GM_v1.1.5_with_phenopacket_data.tsv"
DEFAULT_RESULT_DIR = ROOT / "tryToolsForNewData"
K_VALUES = (1, 3, 5, 10, 30)
TOOLS = ("PCF", "GM", "ZeroShot")


def ids_for(rankings: dict[str, list[str]], allocation: dict[str, int]) -> set[str]:
    out: set[str] = set()
    for tool, k in allocation.items():
        out.update(rankings.get(tool, [])[: max(0, min(30, int(k)))])
    return out


def add_result(rows: list[dict[str, Any]], name: str, allocation: dict[str, int], cases: list[dict[str, Any]]) -> None:
    hits, counts = [], []
    for case in cases:
        candidates = ids_for(case["rankings"], allocation)
        hits.append(case["truth"] is not None and case["truth"] in candidates)
        counts.append(len(candidates))
    rows.append({
        "method": name,
        "allocation": json.dumps(allocation, ensure_ascii=False, sort_keys=True),
        "coverage": float(np.mean(hits)) if hits else 0.0,
        "mean_candidates": float(np.mean(counts)) if counts else 0.0,
        "median_candidates": float(np.median(counts)) if counts else 0.0,
        "p95_candidates": float(np.percentile(counts, 95)) if counts else 0.0,
        "n_cases": len(cases),
    })


def load_llm_allocation(path: Path) -> dict[str, int] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("decision", payload) if isinstance(payload, dict) else {}
        result = {"PCF": int(value["pcf_k"]), "GM": int(value["gm_k"])}
        if any(k < 0 or k > 30 for k in result.values()):
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR / "candidate_budget_eval")
    parser.add_argument("--llm-decision-dir", type=Path, default=None, help="Backward-compatible single LLM directory")
    parser.add_argument("--llm-input-dir", type=Path, default=DEFAULT_RESULT_DIR / "llm_routing")
    parser.add_argument("--llm-tool-results-dir", type=Path, default=DEFAULT_RESULT_DIR / "llmWithToolResults_routing")
    args = parser.parse_args()
    args.tsv, args.result_dir, args.output_dir = [p.expanduser().resolve() for p in (args.tsv, args.result_dir, args.output_dir)]
    if args.llm_decision_dir:
        args.llm_decision_dir = args.llm_decision_dir.expanduser().resolve()
    args.llm_input_dir = args.llm_input_dir.expanduser().resolve()
    args.llm_tool_results_dir = args.llm_tool_results_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.tsv.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="\t"))
    cases = []
    for row in source_rows:
        image_id = str(row.get("image_id", "")).strip()
        rankings = {}
        for tool, folder in (("PCF", "PCF"), ("GM", "GM")):
            rankings[tool], _ = load_ranking(args.result_dir / folder / f"{image_id}.json")
        cases.append({"image_id": image_id, "truth": ground_truth(row), "rankings": rankings})

    results: list[dict[str, Any]] = []
    for tool in TOOLS[:2]:
        for k in K_VALUES:
            add_result(results, f"{tool}@{k}", {tool: k}, cases)
    for k in K_VALUES:
        add_result(results, f"PCF{ k }+GM{ k }", {"PCF": k, "GM": k}, cases)
    for allocation in ({"PCF": 5, "GM": 3}, {"PCF": 10, "GM": 5}):
        add_result(results, "allocation_" + "_".join(f"{k}{v}" for k, v in allocation.items()), allocation, cases)

    # Truth-aware oracle: an unattainable upper-bound reference that stops at
    # the earliest rank where any tool contains the known answer.
    oracle_ranks = []
    for case in cases:
        if case["truth"] is None:
            continue
        positions = [
            ranking.index(case["truth"]) + 1
            for ranking in case["rankings"].values()
            if case["truth"] in ranking
        ]
        if positions:
            oracle_ranks.append(min(positions))
    results.append({
        "method": "Ideal_combinations_truth_known",
        "allocation": "per-case earliest true rank",
        "coverage": len(oracle_ranks) / len(cases) if cases else 0.0,
        "mean_candidates": float(np.mean(oracle_ranks)) if oracle_ranks else 0.0,
        "median_candidates": float(np.median(oracle_ranks)) if oracle_ranks else 0.0,
        "p95_candidates": float(np.percentile(oracle_ranks, 95)) if oracle_ranks else 0.0,
        "n_cases": len(cases),
    })

    llm_dirs = []
    if args.llm_decision_dir and args.llm_decision_dir.is_dir():
        llm_dirs.append(("zeroshot-LLM", args.llm_decision_dir))
    else:
        llm_dirs.extend((("zeroshot-LLM(input-only)", args.llm_input_dir), ("zeroshot-LLM(input+tool result)", args.llm_tool_results_dir)))
    for llm_method, llm_dir in llm_dirs:
      if llm_dir.is_dir():
        by_image = {case["image_id"]: case for case in cases}
        selected = []
        for image_id, case in by_image.items():
            allocation = load_llm_allocation(llm_dir / f"{image_id}.json")
            if allocation is not None:
                selected.append((case, allocation))
        if selected:
            # Evaluate directly because LLM allocations differ per case.
            hits, counts = [], []
            for case, allocation in selected:
                candidates = ids_for(case["rankings"], allocation)
                hits.append(case["truth"] is not None and case["truth"] in candidates)
                counts.append(len(candidates))
            results.append({"method": llm_method, "allocation": "per-case", "coverage": float(np.mean(hits)), "mean_candidates": float(np.mean(counts)), "median_candidates": float(np.median(counts)), "p95_candidates": float(np.percentile(counts, 95)), "n_cases": len(selected)})

    (args.output_dir / "candidate_budget_summary.csv").write_text("", encoding="utf-8")
    with (args.output_dir / "candidate_budget_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    (args.output_dir / "candidate_budget_summary.json").write_text(json.dumps({"n_cases": len(cases), "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for prefix, label, color in (("PCF@", "PCF", "#377eb8"), ("GM@", "GM", "#e41a1c")):
        data = [r for r in results if r["method"].startswith(prefix)]
        if data:
            axes[0].plot([r["mean_candidates"] for r in data], [r["coverage"] for r in data], "o-", label=label, color=color)
    axes[0].set_title("Fixed cutoff baselines"); axes[0].set_xlabel("Mean unique candidates"); axes[0].set_ylabel("Coverage"); axes[0].grid(alpha=.25); axes[0].legend()
    subset = [r for r in results if "+" in r["method"] or r["method"].startswith("zeroshot-")]
    colors = {"zeroshot-LLM(input-only)": "#984ea3", "zeroshot-LLM(input+tool result)": "#ff7f00"}
    axes[1].scatter([r["mean_candidates"] for r in subset], [r["coverage"] for r in subset], c=[colors.get(r["method"], "#4daf4a") for r in subset])
    for r in subset: axes[1].annotate(r["method"], (r["mean_candidates"], r["coverage"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[1].set_title("Allocations and subsets"); axes[1].set_xlabel("Mean unique candidates"); axes[1].set_ylabel("Coverage"); axes[1].grid(alpha=.25)
    fig.suptitle(f"Coverage vs candidate cost (n={len(cases)})"); fig.tight_layout()
    fig.savefig(args.output_dir / "coverage_vs_candidate_cost.png", dpi=200); fig.savefig(args.output_dir / "coverage_vs_candidate_cost.svg"); plt.close(fig)
    print(f"saved: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
