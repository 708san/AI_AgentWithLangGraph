"""Run manifest and registry management for reproducible local experiments."""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = 1
REGISTRY_SCHEMA_VERSION = 1
RUN_STATUSES = {"initialized", "running", "completed", "failed", "interrupted", "skipped", "imported"}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SECRET_ENV_NAMES = {
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GESTALT_API_PASS",
    "GESTALT_API_USER",
    "PUBMED_API_KEY",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_context(root: Path) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain", "--untracked-files=all") or ""
    return {
        "branch": _git(root, "branch", "--show-current"),
        "commit": _git(root, "rev-parse", "HEAD"),
        "working_tree_dirty": bool(status),
        "status_entries": len(status.splitlines()) if status else 0,
    }


def environment_context() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "environment_presence": {
            name: bool(os.environ.get(name))
            for name in sorted(SECRET_ENV_NAMES)
        },
    }


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must contain only letters, numbers, '.', '_', and '-' "
            "and must not start with punctuation"
        )
    return run_id


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _normalise_optional(value: Any) -> Any:
    if value in (None, "", [], {}):
        return None
    return value


def create_run_manifest(
    *,
    root: Path,
    run_id: str,
    kind: str,
    command: Iterable[str],
    purpose: str | None = None,
    entrypoint: str | None = None,
    dataset: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    pipeline: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    run_id = _validate_run_id(str(run_id))
    command_list = [str(item) for item in command]
    now = utc_now()
    run_dir = (run_dir or root / "artifacts" / "runs" / run_id).expanduser().resolve()
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_type": "run",
        "run_id": run_id,
        "status": "initialized",
        "created_at": now,
        "updated_at": now,
        "repository_root": str(root),
        "run_dir": str(run_dir),
        "kind": str(kind),
        "purpose": _normalise_optional(purpose),
        "entrypoint": _normalise_optional(entrypoint),
        "command": command_list,
        "command_text": shlex.join(command_list),
        "git": git_context(root),
        "environment": environment_context(),
        "dataset": dataset or {},
        "model": model or {},
        "pipeline": pipeline or {},
        "metadata": metadata or {},
        "artifacts": {
            "manifest": str(run_dir / "run_manifest.json"),
            "command": str(run_dir / "command.txt"),
            "cases": str(run_dir / "cases"),
            "report": str(run_dir / "report"),
        },
    }
    return manifest


def _registry_entry(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "kind": manifest.get("kind"),
        "purpose": manifest.get("purpose"),
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "repository_root": manifest.get("repository_root"),
        "run_dir": manifest.get("run_dir"),
        "git": manifest.get("git", {}),
        "dataset": manifest.get("dataset", {}),
        "model": manifest.get("model", {}),
        "pipeline": manifest.get("pipeline", {}),
        "entrypoint": manifest.get("entrypoint"),
        "outputs": manifest.get("outputs", []),
        "analysis": manifest.get("analysis", []),
    }


def _read_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": REGISTRY_SCHEMA_VERSION, "registry_type": "run", "runs": []}
    value = _read_json(path)
    runs = value.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"Registry runs must be a list: {path}")
    return value


def register_run(manifest: dict[str, Any], registry_path: Path) -> dict[str, Any]:
    registry_path = registry_path.expanduser().resolve()
    registry = _read_registry(registry_path)
    entries = [item for item in registry["runs"] if item.get("run_id") != manifest.get("run_id")]
    entries.append(_registry_entry(manifest))
    entries.sort(key=lambda item: (str(item.get("created_at") or ""), str(item.get("run_id") or "")), reverse=True)
    registry.update({
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "registry_type": "run",
        "updated_at": utc_now(),
        "runs": entries,
    })
    _atomic_write_json(registry_path, registry)
    return registry["runs"][0] if entries else {}


def initialize_run(
    *,
    root: Path,
    run_id: str,
    kind: str,
    command: Iterable[str],
    registry_path: Path | None = None,
    purpose: str | None = None,
    entrypoint: str | None = None,
    dataset: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    pipeline: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    run_dir = root / "artifacts" / "runs" / _validate_run_id(str(run_id))
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    manifest = create_run_manifest(
        root=root,
        run_id=run_id,
        kind=kind,
        command=command,
        purpose=purpose,
        entrypoint=entrypoint,
        dataset=dataset,
        model=model,
        pipeline=pipeline,
        metadata=metadata,
        run_dir=run_dir,
    )
    for directory in (run_dir, run_dir / "cases", run_dir / "report"):
        directory.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(run_dir / "run_manifest.json", manifest)
    (run_dir / "command.txt").write_text(manifest["command_text"] + "\n", encoding="utf-8")
    registry_path = registry_path or root / "local_artifacts/catalog/run_registry.json"
    register_run(manifest, registry_path)
    return manifest


def update_run_status(
    run_dir: Path,
    status: str,
    *,
    registry_path: Path | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if status not in RUN_STATUSES:
        raise ValueError(f"Unsupported run status: {status}")
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    now = utc_now()
    manifest.update({"status": status, "updated_at": now})
    if status == "running" and not manifest.get("started_at"):
        manifest["started_at"] = now
    if status in {"completed", "failed", "interrupted", "skipped"}:
        manifest.setdefault("started_at", manifest.get("created_at", now))
        manifest["finished_at"] = now
    if error:
        manifest["error"] = error
    _atomic_write_json(manifest_path, manifest)
    registry_path = registry_path or Path(manifest["repository_root"]) / "local_artifacts/catalog/run_registry.json"
    register_run(manifest, registry_path)
    return manifest


def record_process_result(
    run_dir: Path,
    *,
    return_code: int | None,
    stdout_path: Path,
    stderr_path: Path,
    duration_seconds: float,
    error: str | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["execution"] = {
        "return_code": return_code,
        "duration_seconds": duration_seconds,
        "stdout": str(stdout_path.expanduser().resolve()),
        "stderr": str(stderr_path.expanduser().resolve()),
        "error": error,
    }
    manifest["updated_at"] = utc_now()
    _atomic_write_json(manifest_path, manifest)
    registry_path = registry_path or Path(manifest["repository_root"]) / "local_artifacts/catalog/run_registry.json"
    register_run(manifest, registry_path)
    return manifest


def attach_output_record(
    run_dir: Path,
    output_record: dict[str, Any],
    *,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Attach one output record to the authoritative Run manifest."""
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    outputs = manifest.setdefault("outputs", [])
    if not isinstance(outputs, list):
        raise ValueError("Run manifest outputs must be a list")
    output_key = (output_record.get("kind"), output_record.get("path"), output_record.get("case_id"))
    outputs[:] = [
        item for item in outputs
        if (item.get("kind"), item.get("path"), item.get("case_id")) != output_key
    ]
    outputs.append(output_record)
    manifest["outputs"] = outputs
    manifest.setdefault("artifacts", {})["validation"] = str(run_dir / "validation")
    manifest["updated_at"] = utc_now()
    _atomic_write_json(manifest_path, manifest)
    registry_path = registry_path or Path(manifest["repository_root"]) / "local_artifacts/catalog/run_registry.json"
    register_run(manifest, registry_path)
    return manifest


def attach_output_validation(
    run_dir: Path,
    output_record: dict[str, Any],
    *,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Attach one validated output record to the authoritative Run manifest."""
    return attach_output_record(run_dir, output_record, registry_path=registry_path)


def attach_analysis(
    run_dir: Path,
    analysis_record: dict[str, Any],
    *,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Attach offline metrics/comparison evidence to the authoritative Run."""
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    analysis = manifest.setdefault("analysis", [])
    if not isinstance(analysis, list):
        raise ValueError("Run manifest analysis must be a list")
    key = (analysis_record.get("kind"), analysis_record.get("path"))
    analysis[:] = [
        item for item in analysis
        if (item.get("kind"), item.get("path")) != key
    ]
    analysis.append(analysis_record)
    manifest["analysis"] = analysis
    manifest.setdefault("artifacts", {})["analysis"] = str(run_dir / "analysis")
    manifest["updated_at"] = utc_now()
    _atomic_write_json(manifest_path, manifest)
    registry_path = registry_path or Path(manifest["repository_root"]) / "local_artifacts/catalog/run_registry.json"
    register_run(manifest, registry_path)
    return manifest


def list_runs(registry_path: Path, status: str | None = None) -> list[dict[str, Any]]:
    registry = _read_registry(registry_path.expanduser().resolve())
    runs = registry["runs"]
    if status:
        if status not in RUN_STATUSES:
            raise ValueError(f"Unsupported run status: {status}")
        runs = [item for item in runs if item.get("status") == status]
    return runs

