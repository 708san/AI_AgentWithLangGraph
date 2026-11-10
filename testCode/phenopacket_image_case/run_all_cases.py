"""Run every unique patient in the image-backed Phenopacket metadata.

This is a separate batch path from ``run_one_case.py``.  The one-case path
keeps its timestamped directories; this path stores one stable directory per
patient under ``runs/all_test/patient_<id>/`` and runs cases sequentially.

Usage::

    .venv/bin/python testCode/phenopacket_image_case/run_all_cases.py \
      --pubmed-max-workers 2 \
      --evidence-max-workers 3

Completed cases are skipped when their input fingerprint and execution
configuration match the existing ``run_config.json`` and the output contract
passed. Failed or incomplete cases have prior generated artifacts cleared
before retrying in the same patient folder. An interrupted batch can be
started again with the same command; completed patients are skipped. Each
case also receives ``results/phase_results.json`` and
``node_results/<patient_id>/<node>.json`` snapshots, refreshed after every
node, so a Ctrl-C leaves completed phases inspectable.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_old_phenopacket_patient import (  # noqa: E402
    _metadata_id,
    _sha256,
    find_image,
    row_to_phenopacket,
)
from testCode.phenopacket_image_case.run_one_case import (  # noqa: E402
    DEFAULT_IMAGE_DIR,
    DEFAULT_METADATA,
    DEFAULT_OUTPUT_ROOT,
    EVIDENCE_MODE,
    _positive_int,
)


DEFAULT_ALL_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "all_test"
RUN_ONE = Path(__file__).with_name("run_one_case.py")
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_CASES: set[str] = set()
_ACTIVE_PROCESSES: dict[str, subprocess.Popen] = {}
_STOP_REQUESTED = threading.Event()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_rows(metadata_path: Path) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    by_patient: dict[str, list[dict[str, str]]] = {}
    order: list[str] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Metadata TSV has no header: {metadata_path}")
        for raw_row in reader:
            row = {key: value or "" for key, value in raw_row.items()}
            patient_id = _metadata_id(row)
            if not patient_id:
                continue
            if patient_id not in by_patient:
                order.append(patient_id)
            by_patient.setdefault(patient_id, []).append(row)
    return order, by_patient


def _safe_case_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown"))
    return component.strip("._") or "unknown"


def _row_image_token(row: dict[str, str], occurrence: int) -> str:
    """Use the image identity to distinguish duplicate patient rows."""
    image_id = (row.get("image_id") or row.get("imageId") or "").strip()
    filename = (row.get("image_filename") or row.get("filename") or "").strip()
    return _safe_case_component(image_id or Path(filename).stem or f"row_{occurrence}")


def _case_entries(
    patient_order: list[str],
    rows_by_patient: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    """Expand metadata rows into independently runnable image-backed cases."""
    entries: list[dict[str, Any]] = []
    for patient_id in patient_order:
        rows = rows_by_patient[patient_id]
        used_keys: set[str] = set()
        for occurrence, row in enumerate(rows):
            image_token = _row_image_token(row, occurrence)
            if len(rows) == 1:
                case_key = f"patient_{_safe_case_component(patient_id)}"
            else:
                case_key = f"patient_{_safe_case_component(patient_id)}__image_{image_token}"
                if case_key in used_keys:
                    case_key = f"{case_key}__row_{occurrence}"
            used_keys.add(case_key)
            entries.append({
                "case_key": case_key,
                "patient_id": str(patient_id),
                "metadata_occurrence": occurrence,
                "row": row,
            })
    return entries


def _expected_config(
    *,
    case_key: str,
    patient_id: str,
    metadata_occurrence: int,
    metadata_path: Path,
    image_dir: Path,
    row: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_path = find_image(row, str(image_dir))
    return {
        "runner": "phenopacket_image_case_v2",
        "case_key": case_key,
        "patient_id": str(patient_id),
        "metadata_source": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "image_dir": str(image_dir),
        "image_sha256": _sha256(image_path),
        "metadata_occurrence": metadata_occurrence,
        "model": args.model,
        "zero_shot_model": args.zero_shot_model,
        "reasoning_effort": "medium",
        "zero_shot_reasoning_effort": "high",
        "default_max_completion_tokens": 40000,
        "max_flow_iterations": args.max_flow_iterations,
        "use_phenobrain": args.use_phenobrain,
        "use_absent_hpo": args.use_absent_hpo,
        "pubmed_max_workers": args.pubmed_max_workers,
        "evidence_max_workers": args.evidence_max_workers,
        "evidence_mode": EVIDENCE_MODE,
    }


def _case_is_complete(case_dir: Path, expected_config: dict[str, Any]) -> bool:
    config = _read_json(case_dir / "run_config.json")
    summary = _read_json(case_dir / "execution_summary.json")
    contract = _read_json(case_dir / "results" / "output_contract.json")
    phase_results = _read_json(case_dir / "results" / "phase_results.json")
    return bool(
        config == expected_config
        and summary
        and summary.get("status") == "completed"
        and contract
        and contract.get("passed") is True
        # Runs created before the phase manifest was introduced remain
        # resumable/skip-able; the next --force run upgrades their artifacts.
        and (phase_results is None or phase_results.get("status") == "completed")
        and (case_dir / "results" / "public_response.json").is_file()
        and (case_dir / "results" / "finalDiagnosis.json").is_file()
    )


def _case_is_managed(case_dir: Path, patient_id: str) -> bool:
    config = _read_json(case_dir / "run_config.json")
    summary = _read_json(case_dir / "execution_summary.json")
    return bool(
        (config and config.get("runner") == "phenopacket_image_case_v2"
         and str(config.get("patient_id")) == str(patient_id))
        or (summary and str(summary.get("patient_id")) == str(patient_id))
    )


def _prepare_case_for_run(
    case_dir: Path,
    *,
    output_root: Path,
    patient_id: str,
    expected_config: dict[str, Any],
    case_key: str | None = None,
) -> None:
    """Clear this runner's known artifacts before starting a fresh attempt.

    Refuse to clean an unrecognized non-empty directory. This guards against a
    typo in ``--output-root`` or a user-created ``patient_<id>`` directory.
    """
    resolved_root = output_root.resolve()
    resolved_case = case_dir.resolve()
    expected_name = case_key or f"patient_{patient_id}"
    if resolved_case.parent != resolved_root or case_dir.name != expected_name:
        raise ValueError(f"Refusing to reset case directory outside output root: {case_dir}")

    if case_dir.exists():
        if not case_dir.is_dir() or case_dir.is_symlink():
            raise ValueError(f"Case output path is not a regular directory: {case_dir}")
        if any(case_dir.iterdir()) and not _case_is_managed(case_dir, patient_id):
            raise ValueError(
                f"Refusing to reset unrecognized non-empty case directory: {case_dir}"
            )

        for name in ("input", "logs", "results", "node_results"):
            artifact_dir = case_dir / name
            if artifact_dir.is_symlink():
                raise ValueError(f"Refusing to reset symlinked artifact directory: {artifact_dir}")
            if artifact_dir.is_dir():
                shutil.rmtree(artifact_dir)
            elif artifact_dir.exists():
                raise ValueError(f"Expected an artifact directory at: {artifact_dir}")

        for name in (
            "run_info.json",
            "run_config.json",
            "execution_summary.json",
            "launcher.log",
        ):
            artifact_file = case_dir / name
            if artifact_file.is_symlink():
                raise ValueError(f"Refusing to reset symlinked artifact file: {artifact_file}")
            if artifact_file.is_file():
                artifact_file.unlink()
            elif artifact_file.exists():
                raise ValueError(f"Expected a regular artifact file at: {artifact_file}")

    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(case_dir / "run_config.json", expected_config)
    _write_json(
        case_dir / "execution_summary.json",
        {
            "status": "running",
            "patient_id": str(patient_id),
            "run_dir": str(case_dir),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run image-backed Phenopacket cases with stable output folders."
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_ALL_OUTPUT_ROOT))
    parser.add_argument("--model", default="gpt-5-4")
    parser.add_argument("--zero-shot-model", default="gpt-5-4")
    parser.add_argument("--max-flow-iterations", type=int, default=0)
    parser.add_argument("--use-phenobrain", action="store_true")
    parser.add_argument("--use-absent-hpo", action="store_true")
    parser.add_argument(
        "--pubmed-max-workers",
        type=_positive_int,
        default=2,
        help="Maximum concurrent candidate literature searches per patient. Defaults to 2.",
    )
    parser.add_argument(
        "--evidence-max-workers",
        type=_positive_int,
        default=3,
        help="Maximum concurrent candidate Evidence extractions per patient. Defaults to 3.",
    )
    parser.add_argument(
        "--max-parallel-cases",
        type=_positive_int,
        default=2,
        help="Maximum cases running at the same time. Defaults to 2.",
    )
    parser.add_argument(
        "--patient-id",
        dest="patient_ids",
        action="append",
        help="Run only this patient ID; may be supplied more than once.",
    )
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all selected old-case rows/images and call no external APIs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when the stable patient folder already has a completed result.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip matching completed folders and continue after Ctrl-C (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing completed folders and rebuild every selected case.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed patient instead of continuing.",
    )
    return parser


def _run_one_case(
    *,
    patient_id: str,
    case_dir: Path,
    args: argparse.Namespace,
    case_key: str | None = None,
    metadata_occurrence: int = 0,
) -> int:
    case_key = case_key or f"patient_{patient_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = case_dir / "launcher.log"
    command = [
        sys.executable,
        str(RUN_ONE),
        "--patient-id",
        str(patient_id),
        "--metadata",
        args.metadata,
        "--image-dir",
        args.image_dir,
        "--output-root",
        args.output_root,
        "--run-dir",
        str(case_dir),
        "--metadata-occurrence",
        str(metadata_occurrence),
        "--case-key",
        case_key,
        "--model",
        args.model,
        "--zero-shot-model",
        args.zero_shot_model,
        "--max-flow-iterations",
        str(args.max_flow_iterations),
        "--pubmed-max-workers",
        str(args.pubmed_max_workers),
        "--evidence-max-workers",
        str(args.evidence_max_workers),
    ]
    if args.use_phenobrain:
        command.append("--use-phenobrain")
    if args.use_absent_hpo:
        command.append("--use-absent-hpo")

    with launcher_log.open("w", encoding="utf-8") as handle:
        with _ACTIVE_LOCK:
            if _STOP_REQUESTED.is_set():
                raise InterruptedError("Batch shutdown requested before case launch")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            _ACTIVE_PROCESSES[case_key] = process
        try:
            return process.wait()
        except KeyboardInterrupt:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE_PROCESSES.pop(case_key, None)


def _run_case_entry(
    entry: dict[str, Any],
    *,
    args: argparse.Namespace,
    metadata_path: Path,
    image_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    patient_id = entry["patient_id"]
    case_key = entry["case_key"]
    case_dir = output_root / case_key
    row = entry["row"]
    started = time.monotonic()
    with _ACTIVE_LOCK:
        _ACTIVE_CASES.add(case_key)
    try:
        expected_config = _expected_config(
            case_key=case_key,
            patient_id=patient_id,
            metadata_occurrence=entry["metadata_occurrence"],
            metadata_path=metadata_path,
            image_dir=image_dir,
            row=row,
            args=args,
        )
        if args.resume and not args.force and _case_is_complete(case_dir, expected_config):
            return {
                "patient_id": patient_id,
                "case_key": case_key,
                "status": "skipped",
                "reason": "matching_completed_result_exists",
                "run_dir": str(case_dir),
                "duration_seconds": 0.0,
            }

        _prepare_case_for_run(
            case_dir,
            output_root=output_root,
            patient_id=patient_id,
            expected_config=expected_config,
            case_key=case_key,
        )
        return_code = _run_one_case(
            patient_id=patient_id,
            case_key=case_key,
            metadata_occurrence=entry["metadata_occurrence"],
            case_dir=case_dir,
            args=args,
        )
        summary = _read_json(case_dir / "execution_summary.json") or {}
        status = "completed" if return_code == 0 and summary.get("status") == "completed" else "failed"
        return {
            "patient_id": patient_id,
            "case_key": case_key,
            "status": status,
            "return_code": return_code,
            "run_dir": str(case_dir),
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": summary.get("error") if status == "failed" else None,
        }
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_CASES.discard(case_key)


def _terminate_active_processes(timeout_seconds: float = 10.0) -> set[str]:
    """Stop child runners on Ctrl-C and return their case keys."""
    with _ACTIVE_LOCK:
        active = dict(_ACTIVE_PROCESSES)
    for process in active.values():
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_seconds
    for process in active.values():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return set(active)


def main() -> int:
    args = _build_parser().parse_args()
    metadata_path = Path(args.metadata).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata TSV not found: {metadata_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    patient_order, rows_by_patient = _load_rows(metadata_path)
    if args.patient_ids:
        requested = set(str(value) for value in args.patient_ids)
        patient_order = [patient_id for patient_id in patient_order if patient_id in requested]
    case_entries = _case_entries(patient_order, rows_by_patient)
    if args.limit is not None:
        case_entries = case_entries[: args.limit]
    if not case_entries:
        raise ValueError("No patient IDs matched the requested metadata/filter.")

    if args.dry_run:
        for entry in case_entries:
            find_image(entry["row"], str(image_dir))
        print(
            f"Validated {len(case_entries)} old cases, metadata={metadata_path}, "
            f"image_dir={image_dir}; no APIs called."
        )
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "case_manifest.json",
        {
            "schema_version": 1,
            "aggregation_key": "patient_id",
            "cases": [
                {
                    "case_key": entry["case_key"],
                    "patient_id": entry["patient_id"],
                    "metadata_occurrence": entry["metadata_occurrence"],
                    "image_id": entry["row"].get("image_id") or entry["row"].get("imageId"),
                    "image_filename": entry["row"].get("image_filename") or entry["row"].get("filename"),
                    "image_sha256": _sha256(find_image(entry["row"], str(image_dir))),
                }
                for entry in case_entries
            ],
        },
    )
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    interrupted = False
    next_index = 0
    active_futures = {}
    stop_scheduling = False
    executor = ThreadPoolExecutor(max_workers=args.max_parallel_cases)

    def submit_case(entry):
        future = executor.submit(
            _run_case_entry,
            entry,
            args=args,
            metadata_path=metadata_path,
            image_dir=image_dir,
            output_root=output_root,
        )
        active_futures[future] = entry

    try:
        while next_index < len(case_entries) and len(active_futures) < args.max_parallel_cases:
            submit_case(case_entries[next_index])
            next_index += 1

        while active_futures:
            done, _ = wait(active_futures, return_when=FIRST_COMPLETED)
            for future in done:
                entry = active_futures.pop(future)
                case_dir = output_root / entry["case_key"]
                try:
                    record = future.result()
                except Exception as exc:
                    error = {
                        "status": "failed",
                        "case_key": entry["case_key"],
                        "patient_id": entry["patient_id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    safe_case_path = (
                        not case_dir.is_symlink()
                        and case_dir.resolve().parent == output_root
                        and case_dir.name == entry["case_key"]
                    )
                    if safe_case_path and (
                        not case_dir.exists()
                        or not any(case_dir.iterdir())
                        or _case_is_managed(case_dir, entry["patient_id"])
                    ):
                        case_dir.mkdir(parents=True, exist_ok=True)
                        _write_json(case_dir / "execution_summary.json", error)
                    record = {
                        "patient_id": entry["patient_id"],
                        "case_key": entry["case_key"],
                        "status": "failed",
                        "run_dir": str(case_dir),
                        "duration_seconds": None,
                        "error": str(exc),
                    }

                records.append(record)
                print(
                    f"[{len(records)}/{len(case_entries)}] "
                    f"patient={record['patient_id']} case={record['case_key']} "
                    f"{record['status']}"
                )
                if record["status"] == "failed" and args.stop_on_error:
                    stop_scheduling = True

                if not stop_scheduling and next_index < len(case_entries):
                    submit_case(case_entries[next_index])
                    next_index += 1

            _write_json(
                output_root / "summary.json",
                {
                    "status": "running",
                    "started_at": started_at,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": str(metadata_path),
                    "image_dir": str(image_dir),
                    "model_config": {
                        "model": args.model,
                        "reasoning_effort": "medium",
                        "zero_shot_model": args.zero_shot_model,
                        "zero_shot_reasoning_effort": "high",
                        "default_max_completion_tokens": 40000,
                    },
                    "patient_count": len(patient_order),
                    "case_count": len(case_entries),
                    "max_parallel_cases": args.max_parallel_cases,
                    "processed_count": len(records),
                    "records": records,
                },
            )
    except KeyboardInterrupt:
        interrupted = True
        with _ACTIVE_LOCK:
            _STOP_REQUESTED.set()
            active_case_keys = set(_ACTIVE_CASES)
        process_case_keys = _terminate_active_processes()
        active_case_keys.update(process_case_keys)
        for future in active_futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

        existing_keys = {record["case_key"] for record in records}
        for entry in case_entries:
            case_key = entry["case_key"]
            if case_key in existing_keys:
                continue
            case_dir = output_root / case_key
            if case_key in active_case_keys:
                interrupted_at = datetime.now(timezone.utc).isoformat()
                _write_json(case_dir / "execution_summary.json", {
                    "status": "interrupted",
                    "patient_id": entry["patient_id"],
                    "case_key": case_key,
                    "run_dir": str(case_dir),
                    "interrupted_at": interrupted_at,
                })
                phase_path = case_dir / "results" / "phase_results.json"
                phase_results = _read_json(phase_path) or {}
                phase_results.update({
                    "status": "interrupted",
                    "interrupted_at": interrupted_at,
                })
                if phase_results:
                    _write_json(phase_path, phase_results)
                records.append({
                    "patient_id": entry["patient_id"],
                    "case_key": case_key,
                    "status": "interrupted",
                    "run_dir": str(case_dir),
                    "duration_seconds": None,
                })
            else:
                expected_config = _expected_config(
                    case_key=case_key,
                    patient_id=entry["patient_id"],
                    metadata_occurrence=entry["metadata_occurrence"],
                    metadata_path=metadata_path,
                    image_dir=image_dir,
                    row=entry["row"],
                    args=args,
                )
                is_complete = _case_is_complete(case_dir, expected_config)
                records.append({
                    "patient_id": entry["patient_id"],
                    "case_key": case_key,
                    "status": "completed" if is_complete else "pending",
                    "run_dir": str(case_dir),
                    "duration_seconds": 0.0 if is_complete else None,
                })
        print("Interrupted; active case processes stopped. Rerun with --resume to continue.")
    else:
        executor.shutdown(wait=True)

    failed = sum(record["status"] == "failed" for record in records)
    final_summary = {
        "status": (
            "interrupted" if interrupted
            else "completed" if failed == 0 and len(records) == len(case_entries)
            else "failed"
        ),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "metadata": str(metadata_path),
        "image_dir": str(image_dir),
        "model_config": {
            "model": args.model,
            "reasoning_effort": "medium",
            "zero_shot_model": args.zero_shot_model,
            "zero_shot_reasoning_effort": "high",
            "default_max_completion_tokens": 40000,
        },
        "output_root": str(output_root),
        "patient_count": len(patient_order),
        "case_count": len(case_entries),
        "max_parallel_cases": args.max_parallel_cases,
        "processed_count": sum(record["status"] != "pending" for record in records),
        "completed_count": sum(record["status"] == "completed" for record in records),
        "skipped_count": sum(record["status"] == "skipped" for record in records),
        "failed_count": failed,
        "interrupted_count": sum(record["status"] == "interrupted" for record in records),
        "pending_count": sum(record["status"] == "pending" for record in records),
        "records": records,
    }
    _write_json(output_root / "summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    if interrupted:
        return 130
    return 1 if failed or len(records) != len(case_entries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
