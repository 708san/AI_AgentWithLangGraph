"""Import existing local execution artifacts into the Run registry."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from project_ops.output_validation import (
    validate_external_response,
    validation_filename,
    write_validation_report,
)
from project_ops.run_management import (
    attach_output_record,
    attach_output_validation,
    initialize_run,
    update_run_status,
)


RUN_MARKERS = {
    "execution_summary.json",
    "run_config.json",
    "run_info.json",
    "summary.json",
    "case_manifest.json",
    "evaluation.json",
}
OUTPUT_NAMES = {
    "external_response.json": "external_response",
    "public_response.json": "public_response",
    "evaluation.json": "evaluation_report",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "legacy"


def _run_id_for(source_root: Path, source_dir: Path) -> str:
    relative = source_dir.relative_to(source_root).as_posix() if source_dir != source_root else "."
    readable = _safe_slug(f"{source_root.name}__{relative}")
    digest = hashlib.sha1(str(source_dir.resolve()).encode("utf-8")).hexdigest()[:10]
    return _safe_slug(f"legacy__{readable}__{digest}")


def _is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if any((path / marker).is_file() for marker in RUN_MARKERS):
        return True
    return any(
        (path / name).is_file() or (path / "results" / name).is_file()
        for name in ("external_response.json", "public_response.json")
    )


def discover_legacy_runs(source_root: Path) -> list[Path]:
    source_root = source_root.expanduser().resolve()
    if _is_run_dir(source_root):
        return [source_root]
    return sorted(path for path in source_root.iterdir() if _is_run_dir(path))


def _source_status(source_dir: Path) -> str:
    for filename in ("execution_summary.json", "summary.json", "run_info.json", "run_config.json"):
        payload = _read_json(source_dir / filename)
        status = str(payload.get("status") or "").lower()
        if status in {"completed", "failed", "interrupted", "running", "skipped"}:
            return status
    if list(source_dir.rglob("external_response.json")) or list(source_dir.rglob("public_response.json")):
        return "imported"
    return "failed"


def _conditions(source_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload: dict[str, Any] = {}
    for filename in ("run_config.json", "run_info.json", "execution_summary.json", "summary.json"):
        payload.update(_read_json(source_dir / filename))
    dataset = {}
    model = {}
    pipeline = {}
    for key in ("dataset_id", "metadata", "metadata_source", "image_dir"):
        if payload.get(key) is not None:
            dataset[key] = payload[key]
    for key in ("model", "model_name"):
        if payload.get(key) is not None:
            model["name" if key == "model_name" else key] = payload[key]
    for key in ("evidence_mode", "max_flow_iterations", "use_phenobrain", "use_absent_hpo", "pubmed_max_workers", "evidence_max_workers"):
        if payload.get(key) is not None:
            pipeline[key] = payload[key]
    metadata = {
        "legacy_import": True,
        "source_conditions": payload,
    }
    return dataset, model, pipeline, metadata


def import_legacy_run(
    *,
    root: Path,
    source_root: Path,
    source_dir: Path,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    source_dir = source_dir.expanduser().resolve()
    run_id = _run_id_for(source_root, source_dir)
    run_dir = root / "artifacts" / "runs" / run_id
    if run_dir.exists():
        return {"run_id": run_id, "source_dir": str(source_dir), "status": "already_imported"}
    dataset, model, pipeline, metadata = _conditions(source_dir)
    metadata.update({
        "source_dir": str(source_dir),
        "source_root": str(source_root),
        "source_relative_path": str(source_dir.relative_to(source_root)),
    })
    manifest = initialize_run(
        root=root,
        run_id=run_id,
        kind="legacy_import",
        command=["legacy-artifact-import", str(source_dir)],
        purpose="Imported existing execution artifact without rerunning the pipeline",
        entrypoint=None,
        dataset=dataset,
        model=model,
        pipeline=pipeline,
        metadata=metadata,
        registry_path=registry_path,
    )
    update_run_status(run_dir, "imported", registry_path=registry_path)

    output_records = []
    for output_path in sorted(source_dir.rglob("*.json")):
        output_kind = OUTPUT_NAMES.get(output_path.name)
        if not output_kind:
            continue
        case_id = None
        for parent in output_path.parents:
            if parent == source_dir:
                break
            if parent.name.startswith("patient_"):
                case_id = parent.name
                break
        if output_kind in {"external_response", "public_response"}:
            report = validate_external_response(output_path, output_kind=output_kind)
            report["run_id"] = run_id
            if case_id:
                report["case_id"] = case_id
            report_path = run_dir / "validation" / validation_filename(output_kind, case_id)
            write_validation_report(report_path, report)
            attach_output_validation(
                run_dir,
                {
                    "kind": output_kind,
                    "case_id": case_id,
                    "path": str(output_path),
                    "schema_version": report.get("schema_version"),
                    "sha256": report.get("sha256"),
                    "validation_status": report.get("status"),
                    "validation_report": str(report_path),
                    "source": "legacy_import",
                },
                registry_path=registry_path,
            )
            output_records.append(report)
        else:
            attach_output_record(
                run_dir,
                {
                    "kind": output_kind,
                    "case_id": case_id,
                    "path": str(output_path),
                    "validation_status": "not_validated",
                    "source": "legacy_import",
                },
                registry_path=registry_path,
            )
            output_records.append({"path": str(output_path), "status": "not_validated"})
    return {
        "run_id": run_id,
        "source_dir": str(source_dir),
        "status": "imported",
        "output_count": len(output_records),
        "validation_statuses": [item.get("status") for item in output_records],
    }


def import_legacy_root(
    *,
    root: Path,
    source_root: Path,
    registry_path: Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    source_root = source_root.expanduser().resolve()
    runs = discover_legacy_runs(source_root)
    if limit is not None:
        runs = runs[:limit]
    return [
        import_legacy_run(
            root=root,
            source_root=source_root,
            source_dir=source_dir,
            registry_path=registry_path,
        )
        for source_dir in runs
    ]

