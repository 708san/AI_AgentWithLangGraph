"""Schema-flexible validation and fingerprinting for external output files."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALIDATOR_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_from_payload(payload: dict[str, Any]) -> str | None:
    for key in ("output_schema_version", "schema_version", "external_response_version"):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    nested = payload.get("external_response")
    if isinstance(nested, dict):
        for key in ("output_schema_version", "schema_version", "external_response_version"):
            value = nested.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def _diagnosis_blocks(payload: dict[str, Any]) -> list[tuple[str, Any]]:
    blocks: list[tuple[str, Any]] = []
    sources = [("root", payload)]
    nested = payload.get("external_response")
    if isinstance(nested, dict):
        sources.append(("external_response", nested))
    for location, source in sources:
        if "finalDiagnosis" in source:
            value = source["finalDiagnosis"]
            if isinstance(value, dict) and isinstance(value.get("finalDiagnosis"), dict):
                value = value["finalDiagnosis"]
            blocks.append((f"{location}.finalDiagnosis", value))
    return blocks


def _validate_diagnosis_block(location: str, value: Any) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(value, dict):
        return [f"{location} must be an object"], warnings
    if "ans" not in value:
        warnings.append(f"{location}.ans is absent; output shape may be version-specific")
        return errors, warnings
    answers = value["ans"]
    if not isinstance(answers, list):
        return [f"{location}.ans must be an array"], warnings
    ranks: list[int] = []
    for index, item in enumerate(answers):
        item_location = f"{location}.ans[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_location} must be an object")
            continue
        if "rank" not in item:
            warnings.append(f"{item_location}.rank is absent")
            continue
        rank = item["rank"]
        if type(rank) is not int or rank < 1:
            errors.append(f"{item_location}.rank must be a positive integer")
        else:
            ranks.append(rank)
    if len(ranks) != len(set(ranks)):
        errors.append(f"{location}.ans contains duplicate ranks")
    if ranks and sorted(ranks) != list(range(1, len(ranks) + 1)):
        warnings.append(f"{location}.ans ranks are not a contiguous 1..N sequence")
    if "references" in value and not isinstance(value["references"], list):
        errors.append(f"{location}.references must be an array when present")
    return errors, warnings


def validate_external_response(
    response_path: Path,
    *,
    schema_version: str | None = None,
    output_kind: str = "external_response",
) -> dict[str, Any]:
    """Validate structural invariants without assuming one permanent schema."""
    response_path = response_path.expanduser().resolve()
    result: dict[str, Any] = {
        "validator_version": VALIDATOR_VERSION,
        "validated_at": _now(),
        "output_kind": output_kind,
        "path": str(response_path),
        "exists": response_path.is_file(),
        "schema_version": schema_version,
        "shape_profile": "unknown",
        "errors": [],
        "warnings": [],
    }
    if not response_path.is_file():
        result["errors"] = [f"Output file does not exist: {response_path}"]
        result["status"] = "failed"
        return result

    result["size_bytes"] = response_path.stat().st_size
    result["sha256"] = _sha256(response_path)
    try:
        payload = json.loads(response_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        result["errors"] = [f"Output is not valid UTF-8 JSON: {type(exc).__name__}: {exc}"]
        result["status"] = "failed"
        return result
    if not isinstance(payload, dict):
        result["errors"] = ["Output root must be a JSON object"]
        result["status"] = "failed"
        return result

    detected_version = schema_version or _version_from_payload(payload)
    result["schema_version"] = detected_version or "unversioned"
    if not detected_version:
        result["warnings"].append(
            "No output schema version was declared; pass --schema-version for managed runs"
        )

    if "external_response" in payload:
        if not isinstance(payload["external_response"], dict):
            result["errors"].append("external_response must be an object when present")
        else:
            result["shape_profile"] = "node_wrapped"
    elif "finalDiagnosis" in payload:
        result["shape_profile"] = "direct"
    else:
        result["warnings"].append(
            "No recognized finalDiagnosis or external_response block; recorded as unknown shape"
        )

    blocks = _diagnosis_blocks(payload)
    if not blocks:
        result["warnings"].append("No finalDiagnosis block was found")
    seen_locations: set[str] = set()
    for location, value in blocks:
        if location in seen_locations:
            continue
        seen_locations.add(location)
        errors, warnings = _validate_diagnosis_block(location, value)
        result["errors"].extend(errors)
        result["warnings"].extend(warnings)

    if result["errors"]:
        result["status"] = "failed"
    elif result["warnings"]:
        result["status"] = "partial"
    else:
        result["status"] = "passed"
    result["summary"] = {
        "error_count": len(result["errors"]),
        "warning_count": len(result["warnings"]),
    }
    return result


def validation_filename(output_kind: str, case_id: str | None = None) -> str:
    raw = output_kind + (f"__{case_id}" if case_id else "")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._") or "output"
    return f"{safe}.json"


def write_validation_report(path: Path, report: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

