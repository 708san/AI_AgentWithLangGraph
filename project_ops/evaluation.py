"""Offline registration and comparison of existing evaluation reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from project_ops.run_management import attach_analysis


ANALYSIS_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation report must be a JSON object: {path}")
    if not isinstance(value.get("summary"), dict):
        raise ValueError(f"Evaluation report has no summary object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize_report(report: dict[str, Any], source_path: Path) -> dict[str, Any]:
    cases = report.get("cases")
    summary = report.get("summary", {})
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_type": "evaluation_metrics",
        "created_at": _now(),
        "source_report": str(source_path.expanduser().resolve()),
        "metadata": report.get("metadata", {}),
        "case_count": len(cases) if isinstance(cases, list) else None,
        "stage_count": len(summary),
        "summary": summary,
    }


def _stage_rate(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if not isinstance(value, dict):
        return None
    try:
        return float(value["rate"])
    except (KeyError, TypeError, ValueError):
        return None


def compare_reports(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_summary = candidate.get("summary", {})
    baseline_summary = baseline.get("summary", {})
    stages = sorted(set(candidate_summary) & set(baseline_summary))
    stage_comparisons: dict[str, Any] = {}
    for stage in stages:
        current = candidate_summary[stage]
        previous = baseline_summary[stage]
        current_top_k = current.get("top_k", {}) if isinstance(current, dict) else {}
        previous_top_k = previous.get("top_k", {}) if isinstance(previous, dict) else {}
        ks = sorted(set(current_top_k) & set(previous_top_k), key=str)
        top_k = {}
        for k in ks:
            current_rate = _stage_rate(current_top_k, k)
            previous_rate = _stage_rate(previous_top_k, k)
            top_k[k] = {
                "candidate_rate": current_rate,
                "baseline_rate": previous_rate,
                "rate_delta": (
                    current_rate - previous_rate
                    if current_rate is not None and previous_rate is not None
                    else None
                ),
            }
        candidate_mrr = current.get("mrr") if isinstance(current, dict) else None
        baseline_mrr = previous.get("mrr") if isinstance(previous, dict) else None
        stage_comparisons[stage] = {
            "candidate_mrr": candidate_mrr,
            "baseline_mrr": baseline_mrr,
            "mrr_delta": (
                candidate_mrr - baseline_mrr
                if isinstance(candidate_mrr, (int, float)) and isinstance(baseline_mrr, (int, float))
                else None
            ),
            "top_k": top_k,
        }

    candidate_cases = {
        str(item.get("case_id")): item
        for item in candidate.get("cases", [])
        if isinstance(item, dict) and item.get("case_id") is not None
    }
    baseline_cases = {
        str(item.get("case_id")): item
        for item in baseline.get("cases", [])
        if isinstance(item, dict) and item.get("case_id") is not None
    }
    case_changes: dict[str, dict[str, list[str]]] = {}
    for stage in stages:
        gained: list[str] = []
        lost: list[str] = []
        for case_id in sorted(set(candidate_cases) & set(baseline_cases)):
            current_stage = (candidate_cases[case_id].get("stages") or {}).get(stage, {})
            previous_stage = (baseline_cases[case_id].get("stages") or {}).get(stage, {})
            current_hits = current_stage.get("hits", {}) if isinstance(current_stage, dict) else {}
            previous_hits = previous_stage.get("hits", {}) if isinstance(previous_stage, dict) else {}
            for k in sorted(set(current_hits) & set(previous_hits), key=str):
                if current_hits[k] and not previous_hits[k]:
                    gained.append(f"{case_id}@{k}")
                elif previous_hits[k] and not current_hits[k]:
                    lost.append(f"{case_id}@{k}")
        case_changes[stage] = {"gained": gained, "lost": lost}

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_type": "evaluation_comparison",
        "created_at": _now(),
        "candidate_report": candidate.get("metadata", {}),
        "baseline_report": baseline.get("metadata", {}),
        "stages": stage_comparisons,
        "case_changes": case_changes,
    }


def analyze_run(
    run_dir: Path,
    report_path: Path,
    *,
    baseline_report_path: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    report = _load_report(report_path)
    metrics = summarize_report(report, report_path)
    analysis_dir = run_dir / "analysis"
    metrics_path = analysis_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    result: dict[str, Any] = {
        "run_id": run_dir.name,
        "metrics": metrics,
        "metrics_path": str(metrics_path),
    }
    analysis_record: dict[str, Any] = {
        "kind": "evaluation_metrics",
        "source_report": str(report_path),
        "path": str(metrics_path),
        "case_count": metrics.get("case_count"),
    }
    if baseline_report_path:
        baseline_report_path = baseline_report_path.expanduser().resolve()
        baseline = _load_report(baseline_report_path)
        comparison = compare_reports(report, baseline)
        comparison_path = analysis_dir / "comparison.json"
        _write_json(comparison_path, comparison)
        result["comparison"] = comparison
        result["comparison_path"] = str(comparison_path)
        analysis_record["comparison_path"] = str(comparison_path)
        analysis_record["baseline_report"] = str(baseline_report_path)
    attach_analysis(run_dir, analysis_record, registry_path=registry_path)
    return result

