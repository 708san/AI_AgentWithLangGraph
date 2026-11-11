"""Offline plots for a saved 74-case image-backed v2 run.

The script reads only the saved ``public_response.json`` files and the OldData
gold labels.  It never calls model, literature, image, or ontology APIs.

Example::

    .venv/bin/python scripts/analyze_old74_v2_results.py \
      --run-dir testCode/phenopacket_image_case/runs/all_test_gpt54_med_old74

Outputs are written below ``<run-dir>/analysis``:

* ``tool_hit_rank_stacked.png/.svg``: stacked counts by first correct rank;
* ``tool_case_correctness.png/.svg``: case-by-tool heatmap with correct rank;
* ``tool_case_correctness.csv``: the data behind the heatmap;
* ``tool_metrics.csv`` and ``analysis_summary.json``: numeric summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "testCode/phenopacket_image_case/runs/all_test_gpt54_med_old74"
DEFAULT_METADATA = PROJECT_ROOT / "Data/OldData/browser_input_74_cases.tsv"
MONDO_TOOLS = PROJECT_ROOT / "local_artifacts/evaluation/validationCode"
if str(MONDO_TOOLS) not in sys.path:
    sys.path.insert(0, str(MONDO_TOOLS))

from MondoMatcher import MondoOntologyMatcher  # noqa: E402

TOOLS: tuple[tuple[str, str], ...] = (
    ("PubCaseFinder", "pubCaseFinder"),
    ("PhenotypeSearch", "phenotypeSearchResult"),
    ("ZeroShot", "zeroShotResult"),
    ("GestaltMatcher", "GestaltMatcher"),
    ("FinalDiagnosis", "finalDiagnosis"),
)
ALL_TOOLS: tuple[tuple[str, str], ...] = TOOLS + (
    ("MergedCandidates", "mergedDiseaseCandidates"),
    ("TentativeDiagnosis", "tentativeDiagnosis"),
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _normalise_omim(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if text.startswith("OMIM:"):
        text = text[5:]
    digits = "".join(character for character in text if character.isdigit())
    return f"OMIM:{digits}" if digits else None


def _rank(value: Any, fallback: int) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return parsed if parsed > 0 else None


def _candidate_items(payload: dict[str, Any], key: str) -> list[dict[str, Any]] | None:
    value = payload.get(key)
    if isinstance(value, dict):
        value = value.get("ans")
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, dict)]


def _candidate_id(item: dict[str, Any], key: str) -> str | None:
    if key == "phenotypeSearchResult":
        disease_info = item.get("disease_info")
        if isinstance(disease_info, dict):
            return _normalise_omim(disease_info.get("OMIM_id") or disease_info.get("omim_id"))
    return _normalise_omim(
        item.get("OMIM_id")
        or item.get("omim_id")
        or item.get("candidate_id")
    )


def _first_match_ranks(
    payload: dict[str, Any],
    key: str,
    gold: str,
    matcher: MondoOntologyMatcher,
) -> tuple[int | None, int | None]:
    items = _candidate_items(payload, key)
    if items is None:
        return None, None
    exact: list[int] = []
    close_only: list[int] = []
    for index, item in enumerate(items, start=1):
        item_rank = _rank(item.get("rank") or item.get("best_rank"), index)
        candidate_id = _candidate_id(item, key)
        if item_rank is None or candidate_id is None:
            continue
        judgment = matcher.judge(gold, candidate_id)
        if judgment == "MATCH":
            exact.append(item_rank)
        elif judgment == "CLOSE MATCH":
            close_only.append(item_rank)
    return min(exact, default=None), min(close_only, default=None)


def _load_cases(metadata_path: Path, run_dir: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            patient_id = str(row.get("patient_id") or "").strip()
            gold = _normalise_omim(row.get("correct_omim"))
            if not patient_id or not gold:
                continue
            response_path = run_dir / f"patient_{patient_id}" / "results" / "public_response.json"
            payload = None
            if response_path.is_file():
                try:
                    payload = _read_json(response_path)
                except (OSError, json.JSONDecodeError, ValueError):
                    payload = None
            cases.append({
                "patient_id": patient_id,
                "sex": row.get("gender", ""),
                "correct_omim": gold,
                "correct_disease": row.get("correct_disease", ""),
                "payload": payload,
            })
    return cases


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_plots(output_dir: Path, cases: list[dict[str, Any]], matrix: list[dict[str, Any]]) -> None:
    with tempfile.TemporaryDirectory(prefix="old74-mpl-", dir="/tmp") as cache_dir:
        os.environ.setdefault("MPLCONFIGDIR", cache_dir)
        os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = [label for label, _ in TOOLS]
        colors = ["#3b82b6", "#9a8065", "#3aaa43", "#df3d3d", "#9a72c4"]

        # Grouped bars by rank threshold, with exact MATCH solid and CLOSE-only
        # results hatched. This mirrors the historical evaluation plot.
        ranks = range(1, 6)
        x_base = np.arange(len(ranks))
        width = 0.14
        fig, ax = plt.subplots(figsize=(18, 9))
        for tool_index, label in enumerate(labels):
            x_pos = x_base + (tool_index - (len(labels) - 1) / 2) * width
            exact_values = []
            close_values = []
            for threshold in ranks:
                exact_values.append(sum(
                    item["match"][label] is not None and item["match"][label] <= threshold
                    for item in matrix
                ) / len(cases))
                close_values.append(sum(
                    item["match"][label] is None
                    and item["close"][label] is not None
                    and item["close"][label] <= threshold
                    for item in matrix
                ) / len(cases))
            ax.bar(x_pos, exact_values, width, color=colors[tool_index], label=label, edgecolor="white")
            ax.bar(x_pos, close_values, width, bottom=exact_values, color=colors[tool_index], alpha=0.38, hatch="//", edgecolor="white")
            for x, exact, close in zip(x_pos, exact_values, close_values):
                total = exact + close
                if total:
                    ax.text(x, total + 0.006, f"{total * 100:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
                if exact:
                    ax.text(x, exact / 2, f"{exact * 100:.1f}%", ha="center", va="center", color="white", rotation=90, fontsize=7)
                if close:
                    ax.text(x, exact + close / 2, f"{close * 100:.1f}%", ha="center", va="center", rotation=90, fontsize=7)
        ax.set_xticks(x_base, [f"Rank {rank}" for rank in ranks], fontweight="bold")
        ax.set_ylabel("Percentage of cases")
        ax.set_ylim(0, 0.82)
        ax.set_title("OldData v2: correct disease rank by tool (M=exact, hatched=CLOSE-only)")
        ax.grid(axis="y", linestyle=":", alpha=0.65)
        ax.legend(title="Tools", loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.subplots_adjust(left=0.06, right=0.86, top=0.94, bottom=0.12)
        fig.savefig(output_dir / "tool_hit_rank_stacked.png", dpi=220, bbox_inches="tight")
        fig.savefig(output_dir / "tool_hit_rank_stacked.svg", bbox_inches="tight")
        plt.close(fig)

        # Case-by-tool table/heatmap. Sort high-consensus cases first.
        matrix = sorted(matrix, key=lambda item: (-item["exact_tool_count"], -item["hit_tool_count"], int(item["patient_id"])))
        heat = np.zeros((len(matrix), len(labels)))
        for row_index, item in enumerate(matrix):
            for col_index, label in enumerate(labels):
                if item["match"][label] is not None:
                    heat[row_index, col_index] = 2
                elif item["close"][label] is not None:
                    heat[row_index, col_index] = 1
        fig_height = max(10, len(matrix) * 0.22)
        fig = plt.figure(figsize=(18, fig_height))
        grid = fig.add_gridspec(1, 2, width_ratios=[2.8, 3.5], wspace=0.02)
        table_ax = fig.add_subplot(grid[0, 0])
        heat_ax = fig.add_subplot(grid[0, 1])
        table_ax.axis("off")
        table_data = [[
            item["patient_id"], item["sex"], item["correct_disease"][:34], str(item["hpo_count"]),
            f"{item['exact_tool_count']}/{len(labels)}",
        ] for item in matrix]
        table = table_ax.table(
            cellText=table_data,
            loc="center", cellLoc="left", colWidths=[0.12, 0.10, 0.53, 0.10, 0.12],
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table_ax.text(0.06, 1.012, "patient", transform=table_ax.transAxes, fontsize=7, fontweight="bold", va="bottom")
        table_ax.text(0.19, 1.012, "sex", transform=table_ax.transAxes, fontsize=7, fontweight="bold", va="bottom")
        table_ax.text(0.30, 1.012, "disease", transform=table_ax.transAxes, fontsize=7, fontweight="bold", va="bottom")
        table_ax.text(0.84, 1.012, "HPO", transform=table_ax.transAxes, fontsize=7, fontweight="bold", va="bottom")
        table_ax.text(0.94, 1.012, "exact", transform=table_ax.transAxes, fontsize=7, fontweight="bold", va="bottom")
        for (row, _), cell in table.get_celld().items():
            if row % 2 == 1:
                cell.set_facecolor("#f5f5f5")
        image = heat_ax.imshow(heat, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#d9d9d9", "#f4a261", "#377fc4"]), vmin=0, vmax=2)
        heat_ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        heat_ax.set_yticks(range(len(matrix)), [])
        heat_ax.set_title("Correct disease rank by tool (M=exact, C=close-only)")
        heat_ax.set_xlabel("Tool")
        for row_index, item in enumerate(matrix):
            for col_index, label in enumerate(labels):
                match_rank = item["match"][label]
                close_rank = item["close"][label]
                text = f"M{match_rank}" if match_rank is not None else f"C{close_rank}" if close_rank is not None else "-"
                heat_ax.text(col_index, row_index, text, ha="center", va="center", fontsize=5, color="white" if match_rank is not None else "black")
        heat_ax.legend(handles=[
            matplotlib.patches.Patch(color="#377fc4", label="Exact disease found"),
            matplotlib.patches.Patch(color="#f4a261", label="Close disease only"),
            matplotlib.patches.Patch(color="#d9d9d9", label="No exact/close rank"),
        ], loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.suptitle("OldData v2: case-level tool correctness", y=0.998, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.92, 0.985])
        fig.savefig(output_dir / "tool_case_correctness.png", dpi=220, bbox_inches="tight")
        fig.savefig(output_dir / "tool_case_correctness.svg", bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot saved OldData v2 tool results without API calls.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    output_dir = (args.output_dir or run_dir / "analysis").expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata TSV not found: {metadata_path}")

    cases = _load_cases(metadata_path, run_dir)
    labels = [label for label, _ in TOOLS]
    matcher = MondoOntologyMatcher(str(MONDO_TOOLS / "mondo-base.json"), child_threshold=50)
    matrix: list[dict[str, Any]] = []
    for case in cases:
        payload = case["payload"]
        item: dict[str, Any] = {
            "patient_id": case["patient_id"],
            "sex": case["sex"],
            "correct_omim": case["correct_omim"],
            "correct_disease": case["correct_disease"],
            "response_available": {},
            "match": {},
            "close": {},
            "exact_tool_count": 0,
            "hit_tool_count": 0,
            "hpo_count": len(payload.get("hpoList", [])) if payload else 0,
        }
        for label, key in ALL_TOOLS:
            item["response_available"][label] = (
                payload is not None and _candidate_items(payload, key) is not None
            )
            match_rank, close_rank = (
                _first_match_ranks(payload, key, case["correct_omim"], matcher)
                if payload else (None, None)
            )
            item["match"][label] = match_rank
            item["close"][label] = close_rank
            if label in labels:
                item["exact_tool_count"] += match_rank is not None
                item["hit_tool_count"] += match_rank is not None or close_rank is not None
        matrix.append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    for item in matrix:
        row = {
            "patient_id": item["patient_id"],
            "sex": item["sex"],
            "correct_omim": item["correct_omim"],
            "correct_disease": item["correct_disease"],
            "hpo_count": item["hpo_count"],
        }
        for label in labels:
            match_rank = item["match"][label]
            close_rank = item["close"][label]
            row[label] = f"M{match_rank}" if match_rank is not None else f"C{close_rank}" if close_rank is not None else "-"
        comparison_rows.append(row)
    _write_csv(
        output_dir / "tool_case_correctness.csv",
        comparison_rows,
        ["patient_id", "sex", "correct_omim", "correct_disease", "hpo_count", *labels],
    )

    metrics = []
    for label in labels:
        match_ranks = [item["match"][label] for item in matrix]
        close_ranks = [item["close"][label] for item in matrix]
        response_available = [item["response_available"][label] for item in matrix]
        metrics.append({
            "tool": label,
            "cases": len(matrix),
            "response_cases": sum(response_available),
            "exact_cases": sum(rank is not None for rank in match_ranks),
            "close_only_cases": sum(match is None and close is not None for match, close in zip(match_ranks, close_ranks)),
            "top1_exact": sum(isinstance(rank, int) and rank <= 1 for rank in match_ranks),
            "top3_exact": sum(isinstance(rank, int) and rank <= 3 for rank in match_ranks),
            "top5_exact": sum(isinstance(rank, int) and rank <= 5 for rank in match_ranks),
            "top1_total": sum((match is not None and match <= 1) or (match is None and close is not None and close <= 1) for match, close in zip(match_ranks, close_ranks)),
            "top3_total": sum((match is not None and match <= 3) or (match is None and close is not None and close <= 3) for match, close in zip(match_ranks, close_ranks)),
            "top5_total": sum((match is not None and match <= 5) or (match is None and close is not None and close <= 5) for match, close in zip(match_ranks, close_ranks)),
            "no_exact_match": sum(response and match is None for response, match in zip(response_available, match_ranks)),
            "no_response": sum(not response for response in response_available),
        })
    _write_csv(output_dir / "tool_metrics.csv", metrics, list(metrics[0]))
    summary = {
        "run_dir": str(run_dir),
        "metadata": str(metadata_path),
        "case_count": len(cases),
        "cases_with_saved_response": sum(case["payload"] is not None for case in cases),
        "tools": metrics,
        "outputs": [
            str(output_dir / name)
            for name in (
                "tool_hit_rank_stacked.png",
                "tool_hit_rank_stacked.svg",
                "tool_case_correctness.png",
                "tool_case_correctness.svg",
                "tool_case_correctness.csv",
                "tool_metrics.csv",
            )
        ],
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _make_plots(output_dir, cases, matrix)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
