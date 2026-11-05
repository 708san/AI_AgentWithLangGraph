"""Offline compatibility checks for two execution artifact directories.

The comparison deliberately treats ``external_response.json`` as a versioned
delivery artifact rather than as a permanently fixed schema.  A fixed fixture
comparison can therefore detect behavior changes while still allowing a future
external-response schema version to be introduced explicitly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from project_ops.output_validation import validate_external_response
from project_ops.run_management import attach_analysis


COMPATIBILITY_SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_CONTRACT = {
    "required": ["external_response"],
    "optional": ["public_response", "state"],
}
ARTIFACT_FILENAMES = {
    "external_response": "external_response.json",
    "public_response": "public_response.json",
    "state": "state.json",
}
# These values describe a run location or timing, not diagnosis behavior.  In
# particular, imagePath differs when the same fixture is copied to a temp dir.
VOLATILE_KEYS = {
    "agent_log",
    "artifact_dir",
    "duration_seconds",
    "finished_at",
    "imagePath",
    "pipeline_wall_clock_seconds",
    "profiled_node_seconds",
    "runtime_log",
    "started_at",
    "traceback",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise_for_comparison(value: Any) -> Any:
    """Remove run-location metadata while preserving response content."""
    if isinstance(value, dict):
        return {
            str(key): _normalise_for_comparison(item)
            for key, item in value.items()
            if str(key) not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_normalise_for_comparison(item) for item in value]
    return value


def _diff_values(left: Any, right: Any, path: str = "$", limit: int = 100) -> list[str]:
    if limit <= 0:
        return []
    if type(left) is not type(right):
        return [f"{path}: type changed from {type(left).__name__} to {type(right).__name__}"]
    if isinstance(left, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}.{key}"
            if key not in left:
                differences.append(f"{child}: added")
            elif key not in right:
                differences.append(f"{child}: removed")
            else:
                differences.extend(_diff_values(left[key], right[key], child, limit - len(differences)))
            if len(differences) >= limit:
                return differences[:limit]
        return differences
    if isinstance(left, list):
        differences = []
        if len(left) != len(right):
            differences.append(f"{path}: length changed from {len(left)} to {len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(_diff_values(left_item, right_item, f"{path}[{index}]", limit - len(differences)))
            if len(differences) >= limit:
                return differences[:limit]
        return differences
    if left != right:
        return [f"{path}: value changed"]
    return []


def _find_artifact(root: Path, filename: str) -> tuple[Path | None, list[str]]:
    candidates = sorted(path for path in root.rglob(filename) if path.is_file())
    if not candidates:
        return None, []
    candidates.sort(key=lambda path: (len(path.relative_to(root).parts), str(path)))
    return candidates[0], [str(path) for path in candidates]


def _artifact_contract(profile: dict[str, Any] | None) -> dict[str, list[str]]:
    if not profile:
        return dict(DEFAULT_ARTIFACT_CONTRACT)
    configured = profile.get("compatibility")
    if not isinstance(configured, dict):
        return dict(DEFAULT_ARTIFACT_CONTRACT)
    return {
        "required": [str(item) for item in configured.get("required_artifacts", [])],
        "optional": [str(item) for item in configured.get("optional_artifacts", [])],
    }


def _compare_artifact(kind: str, baseline: Path | None, candidate: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": kind,
        "baseline": str(baseline) if baseline else None,
        "candidate": str(candidate) if candidate else None,
        "status": "missing",
        "differences": [],
    }
    if baseline is None or candidate is None:
        result["status"] = "missing"
        return result
    if kind == "external_response":
        result["baseline_validation"] = validate_external_response(baseline, output_kind=kind)
        result["candidate_validation"] = validate_external_response(candidate, output_kind=kind)
        if result["baseline_validation"]["status"] == "failed" or result["candidate_validation"]["status"] == "failed":
            result["status"] = "invalid"
            return result
    try:
        baseline_value = _normalise_for_comparison(_load_json(baseline))
        candidate_value = _normalise_for_comparison(_load_json(candidate))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        result["status"] = "invalid"
        result["differences"] = [f"could not load JSON: {type(exc).__name__}: {exc}"]
        return result
    differences = _diff_values(baseline_value, candidate_value)
    result["status"] = "equivalent" if not differences else "changed"
    result["differences"] = differences
    return result


def compare_artifact_directories(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    profile: dict[str, Any] | None = None,
    report_path: Path | None = None,
    attach_run_dir: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Compare two saved execution directories without rerunning any API."""
    baseline_dir = baseline_dir.expanduser().resolve()
    candidate_dir = candidate_dir.expanduser().resolve()
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"Baseline artifact directory does not exist: {baseline_dir}")
    if not candidate_dir.is_dir():
        raise FileNotFoundError(f"Candidate artifact directory does not exist: {candidate_dir}")

    contract = _artifact_contract(profile)
    kinds = list(dict.fromkeys(contract["required"] + contract["optional"]))
    artifacts: dict[str, Any] = {}
    missing_required: list[str] = []
    changed: list[str] = []
    invalid: list[str] = []
    ambiguous: dict[str, Any] = {}
    for kind in kinds:
        filename = ARTIFACT_FILENAMES.get(kind, f"{kind}.json")
        baseline_path, baseline_candidates = _find_artifact(baseline_dir, filename)
        candidate_path, candidate_candidates = _find_artifact(candidate_dir, filename)
        item = _compare_artifact(kind, baseline_path, candidate_path)
        if len(baseline_candidates) > 1 or len(candidate_candidates) > 1:
            ambiguous[kind] = {
                "baseline_candidates": baseline_candidates,
                "candidate_candidates": candidate_candidates,
            }
        artifacts[kind] = item
        if kind in contract["required"] and (baseline_path is None or candidate_path is None):
            missing_required.append(kind)
        if item["status"] == "changed":
            changed.append(kind)
        if item["status"] == "invalid":
            invalid.append(kind)

    if invalid:
        status = "invalid"
    elif missing_required:
        status = "incomplete"
    elif changed:
        status = "changed"
    else:
        status = "equivalent"
    result: dict[str, Any] = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "analysis_type": "run_artifact_compatibility",
        "created_at": _now(),
        "profile_id": profile.get("profile_id") if profile else None,
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "status": status,
        "contract": contract,
        "artifacts": artifacts,
        "summary": {
            "required_missing": missing_required,
            "changed": changed,
            "invalid": invalid,
            "ambiguous_artifacts": sorted(ambiguous),
        },
    }
    if ambiguous:
        result["ambiguous_artifacts"] = ambiguous
    if report_path:
        report_path = report_path.expanduser().resolve()
        _write_json(report_path, result)
        result["report_path"] = str(report_path)
    if attach_run_dir:
        attach_analysis(
            attach_run_dir,
            {
                "kind": "run_artifact_compatibility",
                "path": str(report_path) if report_path else None,
                "status": status,
                "baseline_dir": str(baseline_dir),
                "candidate_dir": str(candidate_dir),
            },
            registry_path=registry_path,
        )
    return result
