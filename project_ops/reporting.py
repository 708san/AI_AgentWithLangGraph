"""Generate dependency-free project and Run reports from saved artifacts."""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _flatten_metrics(
    run: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    summary = metrics.get("summary", {})
    for stage, stage_metrics in summary.items():
        if not isinstance(stage_metrics, dict):
            continue
        top_k = stage_metrics.get("top_k", {})
        rows.append({
            "run_id": run.get("run_id"),
            "stage": stage,
            "mrr": stage_metrics.get("mrr"),
            "top_k": json.dumps(top_k, ensure_ascii=False, sort_keys=True),
            "candidate_count": stage_metrics.get("candidate_count"),
        })
    return rows


def _run_rows(registry: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs = registry.get("runs", []) if isinstance(registry, dict) else []
    run_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        outputs = run.get("outputs", []) if isinstance(run.get("outputs", []), list) else []
        output_statuses = Counter(
            str(item.get("validation_status"))
            for item in outputs
            if isinstance(item, dict) and item.get("validation_status")
        )
        analysis = run.get("analysis", []) if isinstance(run.get("analysis", []), list) else []
        metrics_files = [
            Path(item["path"])
            for item in analysis
            if isinstance(item, dict)
            and item.get("kind") == "evaluation_metrics"
            and item.get("path")
        ]
        metrics_loaded = []
        for metrics_path in metrics_files:
            value = _read_json(metrics_path, None)
            if isinstance(value, dict):
                metrics_loaded.append(value)
                stage_rows.extend(_flatten_metrics(run, value))
        model = run.get("model") if isinstance(run.get("model"), dict) else {}
        dataset = run.get("dataset") if isinstance(run.get("dataset"), dict) else {}
        run_rows.append({
            "run_id": run.get("run_id"),
            "status": run.get("status"),
            "kind": run.get("kind"),
            "purpose": run.get("purpose"),
            "created_at": run.get("created_at"),
            "updated_at": run.get("updated_at"),
            "model": model.get("name"),
            "dataset": dataset.get("dataset_id"),
            "dataset_version": dataset.get("version"),
            "branch": (run.get("git") or {}).get("branch"),
            "commit": (run.get("git") or {}).get("commit"),
            "output_validation": dict(output_statuses),
            "metrics_count": len(metrics_loaded),
            "run_dir": run.get("run_dir"),
        })
    return run_rows, stage_rows


def _bar_chart(title: str, counts: dict[str, int]) -> str:
    width = 680
    height = max(120, 56 + len(counts) * 34)
    maximum = max(counts.values(), default=1)
    rows = [f'<text x="10" y="20" class="chart-title">{html.escape(title)}</text>']
    for index, (label, count) in enumerate(sorted(counts.items())):
        y = 38 + index * 30
        bar_width = int(470 * count / maximum) if maximum else 0
        rows.append(
            f'<text x="10" y="{y + 14}">{html.escape(label)}</text>'
            f'<rect x="150" y="{y}" width="{bar_width}" height="18" fill="#4f46e5"/>'
            f'<text x="{160 + bar_width}" y="{y + 14}">{count}</text>'
        )
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(rows)}</svg>'


def _html_report(overview: dict[str, Any]) -> str:
    run_rows = overview["runs"]
    status_counts = overview["run_summary"]["status_counts"]
    rows = []
    for row in run_rows:
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row.get(key) or ''))}</td>"
                for key in ("run_id", "status", "kind", "model", "dataset", "created_at", "metrics_count")
            )
            + "</tr>"
        )
    stage_rows = []
    for row in overview["stage_metrics"]:
        stage_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row.get(key) or ''))}</td>"
                for key in ("run_id", "stage", "mrr", "candidate_count")
            )
            + "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Project overview</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #172033; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #d9deea; padding: .45rem; text-align: left; font-size: .9rem; }}
th {{ background: #eef1f8; }}
svg {{ width: 680px; max-width: 100%; border: 1px solid #d9deea; padding: .5rem; }}
.chart-title {{ font-weight: 700; }}
.cards {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
.card {{ background: #f4f6fb; padding: 1rem; min-width: 10rem; }}
</style></head><body>
<h1>Project overview</h1>
<p>Generated: {html.escape(overview['generated_at'])}</p>
<div class="cards">
<div class="card"><strong>Catalog files</strong><br>{overview['catalog_summary'].get('file_count', '')}</div>
<div class="card"><strong>Runs</strong><br>{overview['run_summary']['total']}</div>
<div class="card"><strong>Evaluation stages</strong><br>{len(overview['stage_metrics'])}</div>
<div class="card"><strong>Execution profiles</strong><br>{len(overview.get('profiles', []))}</div>
</div>
{_bar_chart('Run status', status_counts)}
<h2>Runs</h2>
<table><thead><tr>{''.join(f'<th>{key}</th>' for key in ('run_id', 'status', 'kind', 'model', 'dataset', 'created_at', 'metrics_count'))}</tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan="7">No registered runs</td></tr>'}</tbody></table>
<h2>Stage metrics</h2>
<table><thead><tr><th>run_id</th><th>stage</th><th>mrr</th><th>candidate_count</th></tr></thead>
<tbody>{''.join(stage_rows) or '<tr><td colspan="4">No evaluation metrics</td></tr>'}</tbody></table>
<h2>Execution profiles</h2>
<ul>{''.join(f"<li><code>{html.escape(str(profile.get('profile_id') or ''))}</code> — {html.escape(str(profile.get('purpose') or ''))}</li>" for profile in overview.get('profiles', [])) or '<li>No profiles registered</li>'}</ul>
</body></html>
"""


def generate_project_report(
    *,
    catalog_path: Path,
    registry_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    catalog = _read_json(catalog_path.expanduser().resolve(), {})
    registry = _read_json(registry_path.expanduser().resolve(), {})
    repository_root = Path(str(catalog.get("repository_root") or catalog_path.parent.parent.parent))
    profile_payload = _read_json(repository_root / "configs/execution_profiles.json", {})
    profiles = profile_payload.get("profiles", []) if isinstance(profile_payload, dict) else []
    profiles = [item for item in profiles if isinstance(item, dict)]
    run_rows, stage_rows = _run_rows(registry)
    status_counts = dict(Counter(str(row.get("status")) for row in run_rows))
    kind_counts = dict(Counter(str(row.get("kind")) for row in run_rows))
    overview = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "project_overview",
        "generated_at": _now(),
        "catalog_path": str(catalog_path.expanduser().resolve()),
        "registry_path": str(registry_path.expanduser().resolve()),
        "catalog_summary": catalog.get("summary", {}),
        "run_summary": {
            "total": len(run_rows),
            "status_counts": status_counts,
            "kind_counts": kind_counts,
        },
        "runs": run_rows,
        "stage_metrics": stage_rows,
        "profiles": profiles,
    }
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / "project_overview.json"
    html_path = output_dir / "project_overview.html"
    runs_csv_path = output_dir / "runs.csv"
    metrics_csv_path = output_dir / "stage_metrics.csv"
    _write_json(overview_path, overview)
    html_path.write_text(_html_report(overview), encoding="utf-8")
    with runs_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["run_id", "status", "kind", "model", "dataset", "dataset_version", "branch", "commit", "created_at", "metrics_count", "run_dir"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in run_rows)
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["run_id", "stage", "mrr", "top_k", "candidate_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in stage_rows)
    return {
        "output_dir": str(output_dir),
        "project_overview": str(overview_path),
        "html": str(html_path),
        "runs_csv": str(runs_csv_path),
        "stage_metrics_csv": str(metrics_csv_path),
        "run_count": len(run_rows),
        "stage_metric_count": len(stage_rows),
    }

