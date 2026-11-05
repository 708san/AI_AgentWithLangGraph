"""Execute existing repository commands under Run management."""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from project_ops.output_validation import (
    validate_external_response,
    validation_filename,
    write_validation_report,
)
from project_ops.run_management import (
    attach_output_validation,
    initialize_run,
    record_process_result,
    update_run_status,
)


def parse_output_spec(value: str) -> tuple[str, str, str | None]:
    if "=" not in value:
        raise ValueError("Output specification must be KIND=PATH")
    kind_part, path = value.split("=", 1)
    kind_part = kind_part.strip()
    case_id = None
    if kind_part.endswith("]") and "[" in kind_part:
        kind_part, case_id = kind_part[:-1].rsplit("[", 1)
        kind_part = kind_part.strip()
        case_id = case_id.strip() or None
    if not kind_part or not path.strip():
        raise ValueError("Output specification must contain non-empty KIND and PATH")
    return kind_part, path.strip(), case_id


def run_managed_command(
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
    output_specs: Iterable[tuple[str, str] | tuple[str, str, str | None]] = (),
    registry_path: Path | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    command_list = [str(item) for item in command]
    manifest = initialize_run(
        root=root,
        run_id=run_id,
        kind=kind,
        command=command_list,
        registry_path=registry_path,
        purpose=purpose,
        entrypoint=entrypoint,
        dataset=dataset,
        model=model,
        pipeline=pipeline,
        metadata=metadata,
    )
    run_dir = Path(manifest["run_dir"])
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    update_run_status(run_dir, "running", registry_path=registry_path)
    started = time.monotonic()
    return_code: int | None = None
    process_error: str | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.run(
                command_list,
                cwd=root,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
            return_code = process.returncode
    except OSError as exc:
        process_error = f"{type(exc).__name__}: {exc}"

    record_process_result(
        run_dir,
        return_code=return_code,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        duration_seconds=round(time.monotonic() - started, 3),
        error=process_error,
        registry_path=registry_path,
    )
    final_status = "completed" if return_code == 0 and process_error is None else "failed"
    update_run_status(
        run_dir,
        final_status,
        registry_path=registry_path,
        error=process_error or (f"Command exited with status {return_code}" if return_code else None),
    )

    validation_results = []
    for output_spec in output_specs:
        if len(output_spec) == 2:
            output_kind, output_value = output_spec
            case_id = None
        else:
            output_kind, output_value, case_id = output_spec
        response_path = Path(output_value).expanduser()
        if not response_path.is_absolute():
            response_path = root / response_path
        report = validate_external_response(response_path, output_kind=output_kind)
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
                "path": str(response_path.resolve()),
                "schema_version": report.get("schema_version"),
                "sha256": report.get("sha256"),
                "validation_status": report.get("status"),
                "validation_report": str(report_path),
            },
            registry_path=registry_path,
        )
        validation_results.append(report)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": final_status,
        "return_code": return_code,
        "command": shlex.join(command_list),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "validation": validation_results,
    }

