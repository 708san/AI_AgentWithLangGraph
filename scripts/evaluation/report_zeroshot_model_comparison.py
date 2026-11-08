"""Summarize and plot a saved Zero-shot model-comparison run.

By default this is an offline analysis tool. ``--resolve-missing-ids`` opts in
to ZebraSeek's Azure-embedding-backed disease-name resolver for candidates
without a locally known OMIM identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_RUN_DIR = Path(
    "testCode/phenopacket_image_case/runs/zeroshot_model_reasoning_comparison_74"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONDO_TOOLS = PROJECT_ROOT / "local_artifacts/evaluation/validationCode"
if str(MONDO_TOOLS) not in sys.path:
    sys.path.insert(0, str(MONDO_TOOLS))

from MondoMatcher import MondoOntologyMatcher  # noqa: E402
from agent.tools.candidate_metadata import (  # noqa: E402
    canonical_omim_id,
    known_disease_aliases,
    known_disease_name,
    names_compatible,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _optional_rank(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _normalize_omim(value: Any) -> str | None:
    return canonical_omim_id(value)


def _git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _percent(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100 if denominator else 0.0


class ZebraSeekZeroShotResolver:
    """Mirror normalize_zeroshot_results identity resolution and rank handling."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        self.cache = loaded if isinstance(loaded, dict) else {}
        self._disease_normalize = None

    def _lookup_by_name(self, cleaned_name: str) -> dict[str, Any]:
        cache_key = cleaned_name.upper()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            return cached if isinstance(cached, dict) else {}

        if self._disease_normalize is None:
            # Import lazily: this runtime module loads the local FAISS index and
            # Azure embedding client. It is needed only for unresolved names.
            from agent.tools.diseaseNormalize import disease_normalize

            self._disease_normalize = disease_normalize
        try:
            omim_id, label, similarity = self._disease_normalize(cache_key)
            result = {
                "omim_id": canonical_omim_id(omim_id),
                "disease_name": str(label or ""),
                "similarity": float(similarity),
                "lookup_error": None,
            }
        except Exception as exc:
            result = {
                "omim_id": None,
                "disease_name": "",
                "similarity": None,
                "lookup_error": f"{type(exc).__name__}: {exc}",
            }
        self.cache[cache_key] = result
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)
        return result

    @staticmethod
    def _clean_name(value: Any) -> str:
        text = str(value or "").strip().strip("*")
        return re.sub(r"\s*\(.*\)\s*", "", text).strip()

    def normalize(self, predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return ZebraSeek-normalized candidates and a per-candidate audit."""
        normalized: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        seen_candidate_keys: set[str] = set()

        for model_order, raw in enumerate(predictions, start=1):
            if not isinstance(raw, dict):
                continue
            original_name = raw.get("disease_name") or ""
            original_id = raw.get("OMIM_id") or raw.get("omim_id")
            cleaned_name = self._clean_name(original_name)
            supplied_id = canonical_omim_id(original_id)
            canonical_name = known_disease_name(supplied_id)
            resolved_id = supplied_id
            resolved_name = cleaned_name
            similarity = None
            lookup_error = None

            if supplied_id and canonical_name:
                resolved_name = canonical_name
                status = (
                    "match"
                    if cleaned_name.upper() == canonical_name.upper()
                    else "normalized"
                    if names_compatible(
                        cleaned_name,
                        canonical_name,
                        known_disease_aliases(supplied_id),
                    )
                    else "mismatch_id_authoritative"
                )
            elif cleaned_name:
                lookup = self._lookup_by_name(cleaned_name)
                lookup_id = canonical_omim_id(lookup.get("omim_id"))
                lookup_name = str(lookup.get("disease_name") or "")
                similarity = lookup.get("similarity")
                lookup_error = lookup.get("lookup_error")
                if lookup_id and lookup_name and names_compatible(
                    cleaned_name,
                    lookup_name,
                    known_disease_aliases(lookup_id),
                ):
                    resolved_id = lookup_id
                    resolved_name = lookup_name
                    status = "normalized_from_name"
                else:
                    resolved_id = None
                    status = "unconfirmed"
            else:
                resolved_id = None
                status = "unconfirmed"

            duplicate_key = resolved_id or cleaned_name.upper()
            duplicate = duplicate_key in seen_candidate_keys
            excluded = status == "unconfirmed"
            display_rank = None
            if not excluded:
                # Match normalize_zeroshot_results: retain model order, drop
                # unconfirmed identities, and assign gap-free display ranks.
                display_rank = len(normalized) + 1
                resolved_candidate = dict(raw)
                resolved_candidate["OMIM_id"] = resolved_id
                resolved_candidate["disease_name"] = resolved_name
                resolved_candidate["rank"] = display_rank
                normalized.append(resolved_candidate)
                seen_candidate_keys.add(duplicate_key)

            audit.append({
                "model_order": model_order,
                "model_rank": raw.get("rank"),
                "raw_disease_name": original_name,
                "cleaned_disease_name": cleaned_name,
                "supplied_omim_id": original_id,
                "resolved_omim_id": resolved_id,
                "resolved_disease_name": resolved_name,
                "status": status,
                "duplicate": duplicate,
                "excluded": excluded,
                "display_rank": display_rank,
                "embedding_similarity": similarity,
                "lookup_error": lookup_error,
            })

        return normalized, audit


def _condition_metrics(
    run_dir: Path,
    rows: list[dict[str, str]],
    model: str,
    effort: str,
    *,
    matcher: MondoOntologyMatcher,
    resolver: ZebraSeekZeroShotResolver | None = None,
    resolution_audit: list[dict[str, Any]] | None = None,
    case_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    key = f"{model.replace('-', '_')}__{effort}"
    saved_exact_ranks = [
        _optional_rank(row.get(f"{key}_exact_gold_rank")) for row in rows
    ]
    saved_close_ranks = [
        _optional_rank(row.get(f"{key}_exact_or_close_gold_rank")) for row in rows
    ]
    completed = 0
    durations: list[float] = []
    rank_sequence_ok = 0
    prediction_count = 0
    unresolved_count = 0
    resolved_from_name_count = 0
    duplicate_omim_cases = 0
    prediction_cases = 0
    exact_ranks: list[int | None] = []
    close_ranks: list[int | None] = []

    for case_index, row in enumerate(rows):
        case_dir = run_dir / f"patient_{row['patient_id']}" / model / effort
        status = _read_json(case_dir / "status.json") if (case_dir / "status.json").is_file() else {}
        if status.get("status") == "completed":
            completed += 1
        duration = status.get("duration_seconds")
        if isinstance(duration, (int, float)):
            durations.append(float(duration))

        try:
            predictions = json.loads(row.get(f"{key}_predictions_json", "[]"))
        except json.JSONDecodeError:
            predictions = []
        if not isinstance(predictions, list):
            predictions = []
        prediction_cases += 1
        prediction_count += len(predictions)
        raw_missing = sum(
            _normalize_omim(candidate.get("OMIM_id") or candidate.get("omim_id")) is None
            for candidate in predictions
            if isinstance(candidate, dict)
        )
        omim_ids = []
        gold = _normalize_omim(row.get("correct_omim"))

        if resolver is not None:
            normalized, audit = resolver.normalize(predictions)
            if resolution_audit is not None:
                for audit_item in audit:
                    resolution_audit.append({
                        "patient_id": row["patient_id"],
                        "model": model,
                        "reasoning_effort": effort,
                        **audit_item,
                    })
            unresolved_count += sum(item["excluded"] for item in audit)
            resolved_from_name_count += sum(
                item["status"] == "normalized_from_name" for item in audit
            )
            ranks = [item["rank"] for item in normalized]
            exact_candidate_ranks = []
            close_candidate_ranks = []
            candidate_judgments: dict[int, str] = {}
            for candidate in normalized:
                omim_id = _normalize_omim(candidate.get("OMIM_id") or candidate.get("omim_id"))
                if not omim_id or not gold:
                    continue
                judgment = matcher.judge(gold, omim_id)
                candidate_judgments[int(candidate["rank"])] = judgment
                if judgment == "MATCH":
                    exact_candidate_ranks.append(candidate["rank"])
                    close_candidate_ranks.append(candidate["rank"])
                elif judgment == "CLOSE MATCH":
                    close_candidate_ranks.append(candidate["rank"])
            exact_ranks.append(min(exact_candidate_ranks, default=None))
            close_ranks.append(min(close_candidate_ranks, default=None))
            case_match_rank = exact_ranks[-1]
            case_close_rank = close_ranks[-1]
        else:
            normalized = predictions
            candidate_judgments = {}
            ranks = []
            for index, candidate in enumerate(predictions, start=1):
                rank = candidate.get("rank", index) if isinstance(candidate, dict) else None
                try:
                    ranks.append(int(rank))
                except (TypeError, ValueError):
                    ranks.append(None)
                omim_id = _normalize_omim(candidate.get("OMIM_id") or candidate.get("omim_id")) if isinstance(candidate, dict) else None
                if omim_id:
                    omim_ids.append(omim_id)
                    if gold:
                        candidate_rank = ranks[-1] if ranks and ranks[-1] is not None else len(ranks)
                        candidate_judgments[candidate_rank] = matcher.judge(gold, omim_id)
            exact_ranks = saved_exact_ranks
            close_ranks = saved_close_ranks
            unresolved_count += raw_missing
            case_match_rank = saved_exact_ranks[case_index]
            case_close_rank = saved_close_ranks[case_index]

        if case_records is not None:
            top_candidates = []
            for index, candidate in enumerate(normalized, start=1):
                if not isinstance(candidate, dict):
                    continue
                candidate_rank = candidate.get("rank", index)
                try:
                    candidate_rank = int(candidate_rank)
                except (TypeError, ValueError):
                    candidate_rank = index
                candidate_id = _normalize_omim(
                    candidate.get("OMIM_id") or candidate.get("omim_id")
                )
                candidate_name = str(candidate.get("disease_name") or "").strip()
                top_candidates.append(
                    f"{candidate_rank}: {candidate_name or '(名称なし)'}"
                    f" [{candidate_id or 'OMIMなし'}]"
                    f" ({candidate_judgments.get(candidate_rank, 'NO MATCH')})"
                )
                if len(top_candidates) == 5:
                    break
            case_records.append({
                "patient_id": row["patient_id"],
                "correct_omim": gold or "",
                "correct_disease_name": known_disease_name(gold) or "",
                "model": model,
                "reasoning_effort": effort,
                "match_rank": case_match_rank if case_match_rank is not None else "",
                "close_match_rank": case_close_rank if case_close_rank is not None else "",
                "top5_candidates": " | ".join(top_candidates),
            })

        if ranks and sorted(ranks) == list(range(1, len(ranks) + 1)):
            rank_sequence_ok += 1
        if resolver is not None:
            omim_ids = [
                _normalize_omim(candidate.get("OMIM_id") or candidate.get("omim_id"))
                for candidate in normalized
                if isinstance(candidate, dict)
            ]
            omim_ids = [omim_id for omim_id in omim_ids if omim_id]
        if len(omim_ids) != len(set(omim_ids)):
            duplicate_omim_cases += 1

    denominator = len(rows)
    hit_exact = {
        top_k: sum(rank is not None and rank <= top_k for rank in exact_ranks)
        for top_k in range(1, 6)
    }
    hit_close = {
        top_k: sum(rank is not None and rank <= top_k for rank in close_ranks)
        for top_k in range(1, 6)
    }
    reciprocal_ranks = [1 / rank if rank is not None and rank > 0 else 0 for rank in exact_ranks]
    valid_exact_ranks = [rank for rank in exact_ranks if rank is not None and rank > 0]
    return {
        "model": model,
        "reasoning_effort": effort,
        "cases": denominator,
        "completed_cases": completed,
        "hit_exact": hit_exact,
        "hit_exact_pct": {k: _percent(v, denominator) for k, v in hit_exact.items()},
        "hit_close": hit_close,
        "hit_close_pct": {k: _percent(v, denominator) for k, v in hit_close.items()},
        "mrr_exact": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "mean_correct_rank": statistics.mean(valid_exact_ranks) if valid_exact_ranks else None,
        "mean_seconds": statistics.mean(durations) if durations else None,
        "median_seconds": statistics.median(durations) if durations else None,
        "p90_seconds": _percentile(durations, 0.90),
        "max_seconds": max(durations) if durations else None,
        "rank_sequence_ok_cases": rank_sequence_ok,
        "rank_sequence_ok_pct": _percent(rank_sequence_ok, prediction_cases),
        "predictions_per_case": prediction_count / prediction_cases if prediction_cases else 0,
        "unconfirmed_candidates": unresolved_count,
        "unconfirmed_pct": _percent(unresolved_count, prediction_count),
        "resolved_from_name_candidates": resolved_from_name_count,
        "resolution_mode": "zebraseek_runtime" if resolver is not None else "saved_ids_only",
        "cases_with_duplicate_omim": duplicate_omim_cases,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _write_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    fields = [
        "model", "reasoning_effort", "cases", "completed_cases",
        "hit1_exact", "hit3_exact", "hit5_exact",
        "hit1_close", "hit3_close", "hit5_close", "mrr_exact",
        "mean_seconds", "median_seconds", "p90_seconds", "max_seconds",
        "rank_sequence_ok_cases", "rank_sequence_ok_pct", "predictions_per_case",
        "resolved_from_name_candidates", "unconfirmed_candidates", "unconfirmed_pct",
        "resolution_mode", "cases_with_duplicate_omim",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in metrics:
            writer.writerow({
                "model": item["model"],
                "reasoning_effort": item["reasoning_effort"],
                "cases": item["cases"],
                "completed_cases": item["completed_cases"],
                "hit1_exact": item["hit_exact"][1],
                "hit3_exact": item["hit_exact"][3],
                "hit5_exact": item["hit_exact"][5],
                "hit1_close": item["hit_close"][1],
                "hit3_close": item["hit_close"][3],
                "hit5_close": item["hit_close"][5],
                "mrr_exact": f"{item['mrr_exact']:.5f}",
                "mean_seconds": f"{item['mean_seconds']:.3f}" if item["mean_seconds"] is not None else "",
                "median_seconds": f"{item['median_seconds']:.3f}" if item["median_seconds"] is not None else "",
                "p90_seconds": f"{item['p90_seconds']:.3f}" if item["p90_seconds"] is not None else "",
                "max_seconds": f"{item['max_seconds']:.3f}" if item["max_seconds"] is not None else "",
                "rank_sequence_ok_cases": item["rank_sequence_ok_cases"],
                "rank_sequence_ok_pct": f"{item['rank_sequence_ok_pct']:.2f}",
                "predictions_per_case": f"{item['predictions_per_case']:.2f}",
                "resolved_from_name_candidates": item["resolved_from_name_candidates"],
                "unconfirmed_candidates": item["unconfirmed_candidates"],
                "unconfirmed_pct": f"{item['unconfirmed_pct']:.2f}",
                "resolution_mode": item["resolution_mode"],
                "cases_with_duplicate_omim": item["cases_with_duplicate_omim"],
            })


def _write_case_comparison_csv(path: Path, case_records: list[dict[str, Any]]) -> None:
    condition_columns: list[str] = []
    cases: dict[str, dict[str, Any]] = {}
    for item in case_records:
        condition = f"{item['model']}_{item['reasoning_effort']}"
        if condition not in condition_columns:
            condition_columns.append(condition)
        patient_id = str(item["patient_id"])
        case = cases.setdefault(patient_id, {
            "patient_id": patient_id,
            "correct_omim": item["correct_omim"],
            "correct_disease_name": item["correct_disease_name"],
        })
        match_rank = item["match_rank"]
        close_rank = item["close_match_rank"]
        case[condition] = (
            f"M{match_rank}" if match_rank else f"C{close_rank}" if close_rank else "-"
        )
        case.setdefault("sex", item.get("sex", ""))
        case.setdefault("age", item.get("age", ""))
        case.setdefault("gene", item.get("gene", ""))
        case.setdefault("hpo_summary", item.get("hpo_summary", ""))
    fields = [
        "patient_id", "sex", "age", "gene", "correct_disease_name",
        "hpo_summary", "correct_omim", *condition_columns,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for patient_id in sorted(cases, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
            writer.writerow(cases[patient_id])


def _write_case_comparison_plot(path: Path, case_records: list[dict[str, Any]], rank_threshold: int = 5) -> None:
    with tempfile.TemporaryDirectory(prefix="zeroshot-cases-mpl-") as mpl_cache:
        os.environ.setdefault("MPLCONFIGDIR", mpl_cache)
        os.environ.setdefault("XDG_CACHE_HOME", mpl_cache)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Patch

        condition_keys = list(dict.fromkeys(
            f"{item['model']}_{item['reasoning_effort']}" for item in case_records
        ))
        cases: dict[str, dict[str, Any]] = {}
        for item in case_records:
            patient_id = str(item["patient_id"])
            case = cases.setdefault(patient_id, {**item, "condition_results": {}})
            case["condition_results"][f"{item['model']}_{item['reasoning_effort']}"] = item

        def status(item: dict[str, Any]) -> tuple[int, str, int]:
            match_rank = _optional_rank(str(item.get("match_rank", "")))
            close_rank = _optional_rank(str(item.get("close_match_rank", "")))
            if match_rank is not None and match_rank <= rank_threshold:
                return 2, f"M{match_rank}", match_rank
            if close_rank is not None and close_rank <= rank_threshold:
                return 1, f"C{close_rank}", 500 + close_rank
            return 0, "-", 999

        ordered_cases = sorted(
            cases.values(),
            key=lambda case: (
                -sum(status(case["condition_results"].get(key, {}))[0] == 2 for key in condition_keys),
                -sum(status(case["condition_results"].get(key, {}))[0] == 1 for key in condition_keys),
                case["patient_id"],
            ),
        )
        matrix = np.array([
            [status(case["condition_results"].get(key, {}))[0] for key in condition_keys]
            for case in ordered_cases
        ])
        colors = ["#d9d9d9", "#f6b26b", "#2f7ed8"]
        fig_height = max(12, len(ordered_cases) * 0.30)
        fig = plt.figure(figsize=(21, fig_height))
        grid = GridSpec(1, 2, width_ratios=[8.8, max(5.8, len(condition_keys) * 1.0)], wspace=0.015, figure=fig)
        ax_meta = fig.add_subplot(grid[0, 0])
        ax = fig.add_subplot(grid[0, 1], sharey=ax_meta)
        ax.imshow(matrix, aspect="auto", cmap=ListedColormap(colors), vmin=0, vmax=2)
        n_cases = len(ordered_cases)
        y_min, y_max = -1.35, n_cases - 0.5
        ax_meta.set_xlim(0, 1)
        ax_meta.set_ylim(y_max, y_min)
        ax_meta.axis("off")
        columns = [
            ("patient", 0.012, "left"), ("sex", 0.095, "left"), ("age", 0.145, "left"),
            ("gene", 0.215, "left"), ("disease", 0.315, "left"),
            ("HPO", 0.845, "left"), ("exact", 0.985, "right"),
        ]
        for label, x, ha in columns:
            ax_meta.text(x, -1.05, label, ha=ha, va="bottom", fontsize=7, fontweight="bold")
        for y in np.arange(-0.5, n_cases, 1):
            ax_meta.hlines(y, 0, 1, color="#ededed", linewidth=0.55)
        for x in [0.08, 0.135, 0.205, 0.305, 0.835, 0.93]:
            ax_meta.vlines(x, -0.5, y_max, color="#dddddd", linewidth=0.55)

        for y, case in enumerate(ordered_cases):
            exact_count = 0
            for x, key in enumerate(condition_keys):
                code, label, _ = status(case["condition_results"].get(key, {}))
                exact_count += code == 2
                ax.text(x, y, label, ha="center", va="center", fontsize=7, color="white" if code == 2 else "black", fontweight="bold")
            ax_meta.text(0.012, y, str(case["patient_id"]), ha="left", va="center", fontsize=6.2)
            ax_meta.text(0.095, y, str(case.get("sex") or "-"), ha="left", va="center", fontsize=6.2)
            ax_meta.text(0.145, y, str(case.get("age") or "-"), ha="left", va="center", fontsize=6.2)
            ax_meta.text(0.215, y, str(case.get("gene") or "-"), ha="left", va="center", fontsize=6.2)
            ax_meta.text(0.315, y, str(case.get("correct_disease_name") or "-")[:48], ha="left", va="center", fontsize=6.2)
            ax_meta.text(0.845, y, str(case.get("hpo_summary") or "-"), ha="left", va="center", fontsize=6.2)
            ax_meta.text(0.985, y, f"{exact_count}/{len(condition_keys)}", ha="right", va="center", fontsize=6.2)

        labels = [f"{key.replace('_', ' / ')}" for key in condition_keys]
        ax.set_xticks(np.arange(len(condition_keys)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks([])
        ax.set_xlim(-0.5, len(condition_keys) - 0.5)
        ax.set_ylim(y_max, y_min)
        ax.set_xlabel("Model / reasoning effort")
        fig.suptitle(
            "Zero-shot correct disease rank by model and reasoning effort "
            "(M=match rank, C=close-only rank)",
            y=0.978, fontsize=13, fontweight="bold",
        )
        for y in np.arange(-0.5, n_cases, 1):
            ax.hlines(y, -0.5, len(condition_keys) - 0.5, color="white", linewidth=0.55)
        for x in np.arange(-0.5, len(condition_keys), 1):
            ax.vlines(x, -0.5, y_max, color="white", linewidth=0.55)
        summaries = []
        for key in condition_keys:
            result_items = [case["condition_results"].get(key, {}) for case in ordered_cases]
            exact = sum(status(item)[0] == 2 for item in result_items)
            close = sum(status(item)[0] == 1 for item in result_items)
            summaries.append(f"{key}: M {exact}/{n_cases}, C {close}/{n_cases}, total {(exact + close) / max(n_cases, 1) * 100:.1f}%")
        fig.text(0.5, 0.995, f"Rank <= {rank_threshold} summary: " + " | ".join(summaries), ha="center", va="top", fontsize=7, color="#333333")
        ax.legend(handles=[
            Patch(facecolor=colors[2], label="Exact disease found"),
            Patch(facecolor=colors[1], label="Close disease only"),
            Patch(facecolor=colors[0], label="No exact/close rank"),
        ], loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.subplots_adjust(left=0.015, right=0.895, top=0.955, bottom=0.065, wspace=0.015)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=220)
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)


def _write_stacked_plot(path: Path, metrics: list[dict[str, Any]]) -> None:
    with tempfile.TemporaryDirectory(prefix="zeroshot-mpl-") as mpl_cache:
        os.environ.setdefault("MPLCONFIGDIR", mpl_cache)
        os.environ.setdefault("XDG_CACHE_HOME", mpl_cache)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        colors = ["#1f77b4", "#8c564b", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]
        top_ks = [1, 2, 3, 4, 5]
        x_base = np.arange(len(top_ks))
        width = 0.13
        fig, ax = plt.subplots(figsize=(16, 8.5))
        max_total = 0.0
        for index, item in enumerate(metrics):
            offset = (index - (len(metrics) - 1) / 2) * width
            x_pos = x_base + offset
            exact = [item["hit_exact_pct"][k] / 100 for k in top_ks]
            close_only = [
                (item["hit_close_pct"][k] - item["hit_exact_pct"][k]) / 100
                for k in top_ks
            ]
            totals = [a + b for a, b in zip(exact, close_only)]
            max_total = max(max_total, *totals)
            label = f"{item['model']} / {item['reasoning_effort']}"
            ax.bar(
                x_pos, exact, width, color=colors[index], label=label,
                edgecolor="white", linewidth=0.5,
            )
            ax.bar(
                x_pos, close_only, width, bottom=exact, color=colors[index],
                alpha=0.38, hatch="//", edgecolor="white", linewidth=0.5,
            )
            for x, total in zip(x_pos, totals):
                if total > 0:
                    ax.text(x, total + 0.006, f"{total * 100:.1f}%", ha="center", va="bottom", fontsize=7)

        ax.set_ylim(0, min(1.05, max(0.5, max_total + 0.12)))
        ax.set_xticks(x_base)
        ax.set_xticklabels([f"Top-{k}" for k in top_ks], fontsize=13, fontweight="bold")
        ax.set_ylabel(f"Percentage of cases (N={metrics[0]['cases']})")
        ax.set_title("Zero-shot MATCH and MONDO CLOSE MATCH by rank threshold", pad=32)
        fig.text(
            0.5, 0.94,
            "Solid = ZebraSeek/MONDO MATCH; hatched = CLOSE MATCH only. Each bar is one model / reasoning condition.",
            ha="center", va="center", fontsize=10,
        )
        ax.legend(loc="upper left", title="Model / reasoning", bbox_to_anchor=(1, 1))
        ax.grid(axis="y", linestyle=":", alpha=0.65)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.2)
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)


def _write_metrics_table(path: Path, metrics: list[dict[str, Any]]) -> None:
    with tempfile.TemporaryDirectory(prefix="zeroshot-table-mpl-") as mpl_cache:
        os.environ.setdefault("MPLCONFIGDIR", mpl_cache)
        os.environ.setdefault("XDG_CACHE_HOME", mpl_cache)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        columns = [
            "Model", "Effort", "MATCH H@1", "MATCH H@3", "MATCH H@5",
            "CLOSE H@5", "MRR", "Mean s", "Median s", "P90 s",
            "Full ranks", "Resolved/name", "Unconfirmed",
        ]
        cells = []
        for item in metrics:
            cells.append([
                item["model"], item["reasoning_effort"],
                f"{item['hit_exact'][1]}/74 ({item['hit_exact_pct'][1]:.1f}%)",
                f"{item['hit_exact'][3]}/74 ({item['hit_exact_pct'][3]:.1f}%)",
                f"{item['hit_exact'][5]}/74 ({item['hit_exact_pct'][5]:.1f}%)",
                f"{item['hit_close'][5]}/74 ({item['hit_close_pct'][5]:.1f}%)",
                f"{item['mrr_exact']:.3f}",
                f"{item['mean_seconds']:.1f}",
                f"{item['median_seconds']:.1f}",
                f"{item['p90_seconds']:.1f}",
                f"{item['rank_sequence_ok_cases']}/74",
                str(item["resolved_from_name_candidates"]),
                f"{item['unconfirmed_candidates']} ({item['unconfirmed_pct']:.1f}%)",
            ])

        fig, ax = plt.subplots(figsize=(19, 4.5))
        ax.axis("off")
        ax.set_title("Zero-shot comparison summary (74 cases)", loc="left", fontsize=16, fontweight="bold", pad=18)
        table = ax.table(
            cellText=cells,
            colLabels=columns,
            loc="center",
            cellLoc="center",
            colWidths=[0.07, 0.06, 0.09, 0.09, 0.09, 0.09, 0.05, 0.06, 0.06, 0.06, 0.07, 0.08, 0.12],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.3)
        table.scale(1.0, 1.65)
        for (row, _column), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#263238")
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#F2F6FA" if row % 2 else "#FFFFFF")
        fig.text(
            0.01, 0.025,
            "CLOSE H@5 includes MATCH. Full ranks = normalized candidate ranks are continuous. Resolved/name = IDs assigned by ZebraSeek name normalization; Unconfirmed = excluded candidates.",
            fontsize=8, color="#555555",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)


def _write_report(
    path: Path,
    run_dir: Path,
    config: dict[str, Any],
    run_status: dict[str, Any],
    metrics: list[dict[str, Any]],
    resolution_enabled: bool,
) -> None:
    exact5_best = max(metrics, key=lambda item: item["hit_exact"][5])
    close5_best = max(metrics, key=lambda item: item["hit_close"][5])
    exact1_best = max(metrics, key=lambda item: item["hit_exact"][1])
    exact1_tied = [
        item for item in metrics
        if item["hit_exact"][1] == exact1_best["hit_exact"][1]
    ]
    exact1_summary = ", ".join(
        f"{item['model']} / {item['reasoning_effort']}" for item in exact1_tied
    )
    fastest = min(
        (item for item in metrics if item["mean_seconds"] is not None),
        key=lambda item: item["mean_seconds"],
    )
    slowest = max(
        (item for item in metrics if item["mean_seconds"] is not None),
        key=lambda item: item["mean_seconds"],
    )

    lines = [
        "# Zero-shot model/reasoning comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Run identity",
        "",
        f"- Result directory: `{run_dir}`",
        f"- Dataset: `{config.get('metadata_source', 'unknown')}`",
        f"- Dataset SHA-256: `{config.get('metadata_sha256', 'unknown')}`",
        f"- Cases / conditions: {config.get('cases')} / {len(config.get('conditions', []))}",
        f"- Models / reasoning: {', '.join(config.get('models', []))} / {', '.join(config.get('reasoning_efforts', []))}",
        f"- Output token limit: {config.get('max_tokens')}; request timeout: {config.get('request_timeout_seconds')}s; concurrency: {config.get('max_parallel_conditions')}",
        f"- Prompt source: `{config.get('prompt_source')}`; Zero-shot only: {config.get('only_zero_shot')}",
        f"- Run status: {run_status.get('status')}; completed cases: {run_status.get('cases')}; incomplete: {len(run_status.get('incomplete_conditions', []))}",
        f"- Code revision at report time: `{_git_value('rev-parse', '--short', 'HEAD')}` on `{_git_value('branch', '--show-current')}`; working tree: {'dirty' if _git_value('status', '--porcelain') else 'clean'}",
        f"- ID resolution: {'ZebraSeek runtime name normalization enabled (Azure embeddings for uncached unresolved names)' if resolution_enabled else 'saved IDs only; no embedding API calls'}",
        f"- Reproduction command: `.venv/bin/python -m scripts.evaluation.report_zeroshot_model_comparison --run-dir {run_dir}{' --resolve-missing-ids' if resolution_enabled else ''}`",
        "",
        "## Metrics",
        "",
        "Hit@k counts cases where the gold OMIM appears at rank k or better. MATCH means the ZebraSeek/MONDO matcher returns MATCH (same OMIM or same MONDO term); CLOSE MATCH uses its parent/child or qualifying sibling rule. MRR uses MATCH ranks; misses contribute zero. Latency statistics use saved per-condition elapsed seconds. When name resolution is enabled, candidate IDs, filtering, and continuous rank reassignment mirror ZebraSeek's ZeroShot normalization before matching.",
        "",
        "| Model | Effort | MATCH H@1 | MATCH H@3 | MATCH H@5 | MATCH-or-CLOSE H@1 | MATCH-or-CLOSE H@3 | MATCH-or-CLOSE H@5 | MATCH MRR | Mean s | Median s | P90 s | Max s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        lines.append(
            f"| {item['model']} | {item['reasoning_effort']} | "
            + " | ".join(f"{item['hit_exact'][k]}/74 ({item['hit_exact_pct'][k]:.1f}%)" for k in (1, 3, 5))
            + " | "
            + " | ".join(f"{item['hit_close'][k]}/74 ({item['hit_close_pct'][k]:.1f}%)" for k in (1, 3, 5))
            + f" | {item['mrr_exact']:.3f} | {item['mean_seconds']:.1f} | {item['median_seconds']:.1f} | {item['p90_seconds']:.1f} | {item['max_seconds']:.1f} |"
        )

    lines.extend([
        "",
        "## Findings",
        "",
        f"- Highest MATCH Hit@1: {exact1_best['hit_exact'][1]}/{exact1_best['cases']} ({exact1_summary}).",
        f"- Best MATCH Hit@5: {exact5_best['model']} / {exact5_best['reasoning_effort']} at {exact5_best['hit_exact'][5]}/{exact5_best['cases']} ({exact5_best['hit_exact_pct'][5]:.1f}%).",
        f"- Best MATCH-or-CLOSE Hit@5: {close5_best['model']} / {close5_best['reasoning_effort']} at {close5_best['hit_close'][5]}/{close5_best['cases']} ({close5_best['hit_close_pct'][5]:.1f}%).",
        f"- Fastest mean response: {fastest['model']} / {fastest['reasoning_effort']} at {fastest['mean_seconds']:.1f}s/case; slowest: {slowest['model']} / {slowest['reasoning_effort']} at {slowest['mean_seconds']:.1f}s/case.",
        "- Raising reasoning effort did not consistently improve ranking. Compare the condition-level values rather than assuming higher effort is better.",
        "",
        "## Output integrity",
        "",
        "| Model | Effort | Completed | Continuous rank lists | Mean candidates/case | Resolved from name | Unconfirmed / excluded | Duplicate OMIM cases |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for item in metrics:
        lines.append(
            f"| {item['model']} | {item['reasoning_effort']} | {item['completed_cases']}/{item['cases']} | "
            f"{item['rank_sequence_ok_cases']}/74 ({item['rank_sequence_ok_pct']:.1f}%) | "
            f"{item['predictions_per_case']:.2f} | {item['resolved_from_name_candidates']} | "
            f"{item['unconfirmed_candidates']} ({item['unconfirmed_pct']:.1f}%) | "
            f"{item['cases_with_duplicate_omim']} |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "This is a retrospective ranking comparison on 74 legacy cases using the saved correct-OMIM labels. MONDO close matches are a separate ontology-based measure and are not exact diagnoses. Results do not establish clinical validity or generalize beyond this dataset. The experiment compares model/reasoning settings with a shared prompt, but service-side model deployments may change over time.",
        "",
        "## Files",
        "",
        "- `stacked_hit_close_topn.png` / `.svg`: cumulative Top-1 to Top-5 MATCH and CLOSE MATCH-only results by condition.",
        "- `condition_metrics_table.png` / `.svg`: accuracy, latency, rank integrity, name-resolved IDs, and unconfirmed candidates.",
        "- `condition_metrics.csv`: all condition-level metrics and output-integrity counts.",
        "- `case_comparison.png` / `.svg`: case-by-condition heatmap with patient context, gold disease, and M/C rank labels.",
        "- `case_comparison.csv`: one row per patient, with context and one M/C rank column for each model/effort condition.",
        "- `zeroshot_id_resolution_audit.csv`: per-candidate identity/rank decisions when name resolution is enabled.",
        "- `comparison_report.md`: this report.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--resolve-missing-ids",
        action="store_true",
        help=(
            "Mirror ZebraSeek ZeroShot ID normalization. This may call the configured "
            "Azure embedding endpoint for each uncached disease name that needs resolution."
        ),
    )
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    default_output_name = "analysis_zebraseek_resolved" if args.resolve_missing_ids else "analysis"
    output_dir = (args.output_dir or (run_dir / default_output_name)).expanduser().resolve()

    config = _read_json(run_dir / "run_config.json")
    run_status = _read_json(run_dir / "run_status.json")
    rows = _read_csv(run_dir / "comparison_summary.csv")
    if run_status.get("status") not in {"completed", "completed_with_errors"}:
        raise ValueError(f"Run is not finished: {run_status.get('status')}")
    if len(rows) != config.get("cases"):
        raise ValueError(f"Expected {config.get('cases')} comparison rows; found {len(rows)}")

    matcher = MondoOntologyMatcher(str(MONDO_TOOLS / "mondo-base.json"), child_threshold=50)
    resolver = (
        ZebraSeekZeroShotResolver(output_dir / "zeroshot_id_resolution_cache.json")
        if args.resolve_missing_ids
        else None
    )
    resolution_audit: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    metrics = [
        _condition_metrics(
            run_dir,
            rows,
            model,
            effort,
            matcher=matcher,
            resolver=resolver,
            resolution_audit=resolution_audit,
            case_records=case_records,
        )
        for model in config["models"]
        for effort in config["reasoning_efforts"]
    ]
    context_path = PROJECT_ROOT / "Data/OldData/phenopacket_test_metadata_old_74_cases_with_phenopacket_data.tsv"
    patient_context: dict[str, dict[str, str]] = {}
    if context_path.is_file():
        for context in _read_tsv(context_path):
            patient_id = str(context.get("patient_id", "")).strip()
            if not patient_id:
                continue
            years = str(context.get("age_year", "")).strip()
            months = str(context.get("age_month", "")).strip()
            age_parts = []
            if years and years not in {"0", "0.0"}:
                age_parts.append(f"{years}y")
            if months and months not in {"0", "0.0"}:
                age_parts.append(f"{months}m")
            patient_context[patient_id] = {
                "sex": str(context.get("gender", "")).strip()[:1].upper(),
                "age": "".join(age_parts),
                "gene": str(context.get("gene", "")).strip(),
                "correct_disease_name": str(context.get("disease", "")).strip(),
            }
    for item in case_records:
        context = patient_context.get(str(item["patient_id"]), {})
        item.update(context)
        case_dir = run_dir / f"patient_{item['patient_id']}"
        input_path = case_dir / "input.json"
        input_snapshot = _read_json(input_path) if input_path.is_file() else {}
        present_count = len(input_snapshot.get("hpo_ids", []))
        absent_count = len(input_snapshot.get("absent_hpo_ids", []))
        item["hpo_summary"] = f"P{present_count}/A{absent_count}"
        if not item.get("correct_disease_name"):
            item["correct_disease_name"] = known_disease_name(item.get("correct_omim")) or ""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_csv(output_dir / "condition_metrics.csv", metrics)
    _write_case_comparison_csv(output_dir / "case_comparison.csv", case_records)
    _write_case_comparison_plot(output_dir / "case_comparison", case_records)
    if resolver is not None:
        audit_path = output_dir / "zeroshot_id_resolution_audit.csv"
        audit_fields = [
            "patient_id", "model", "reasoning_effort", "model_order", "model_rank",
            "raw_disease_name", "cleaned_disease_name", "supplied_omim_id",
            "resolved_omim_id", "resolved_disease_name", "status", "duplicate",
            "excluded", "display_rank", "embedding_similarity", "lookup_error",
        ]
        with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=audit_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(resolution_audit)
    _write_stacked_plot(output_dir / "stacked_hit_close_topn", metrics)
    _write_metrics_table(output_dir / "condition_metrics_table", metrics)
    _write_report(
        output_dir / "comparison_report.md",
        run_dir,
        config,
        run_status,
        metrics,
        resolution_enabled=resolver is not None,
    )
    print(f"Report: {output_dir / 'comparison_report.md'}")
    print(f"Plot:   {output_dir / 'stacked_hit_close_topn.png'}")
    print(f"Table:  {output_dir / 'condition_metrics_table.png'}")
    print(f"CSV:    {output_dir / 'condition_metrics.csv'}")
    print(f"Cases:  {output_dir / 'case_comparison.csv'}")
    print(f"Case plot: {output_dir / 'case_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
