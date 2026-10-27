"""Run every unique patient in the image-backed Phenopacket metadata.

This is a separate batch path from ``run_one_case.py``.  The one-case path
keeps its timestamped directories; this path stores one stable directory per
patient under ``runs/all_test/patient_<id>/`` and runs cases sequentially.

Usage::

    .venv/bin/python testCode/phenopacket_image_case/run_all_cases.py \
      --pubmed-max-workers 2 \
      --evidence-max-workers 2

Completed cases are skipped when their input fingerprint and execution
configuration match the existing ``run_config.json`` and the output contract
passed. Failed or incomplete cases have prior generated artifacts cleared
before retrying in the same patient folder. An interrupted batch can be
started again with the same command; completed patients are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
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


def _expected_config(
    *,
    patient_id: str,
    metadata_path: Path,
    image_dir: Path,
    row: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_path = find_image(row, str(image_dir))
    return {
        "runner": "phenopacket_image_case_v2",
        "patient_id": str(patient_id),
        "metadata_source": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "image_dir": str(image_dir),
        "image_sha256": _sha256(image_path),
        "metadata_occurrence": 0,
        "model": args.model,
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
    return bool(
        config == expected_config
        and summary
        and summary.get("status") == "completed"
        and contract
        and contract.get("passed") is True
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
) -> None:
    """Clear this runner's known artifacts before starting a fresh attempt.

    Refuse to clean an unrecognized non-empty directory. This guards against a
    typo in ``--output-root`` or a user-created ``patient_<id>`` directory.
    """
    resolved_root = output_root.resolve()
    resolved_case = case_dir.resolve()
    if resolved_case.parent != resolved_root or case_dir.name != f"patient_{patient_id}":
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
        description="Run all unique Phenopacket patients sequentially with stable output folders."
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_ALL_OUTPUT_ROOT))
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-flow-iterations", type=int, default=1)
    parser.add_argument("--use-phenobrain", action="store_true")
    parser.add_argument("--use-absent-hpo", action="store_true")
    parser.add_argument(
        "--pubmed-max-workers",
        type=_positive_int,
        default=2,
        help="Maximum concurrent candidate literature searches per patient.",
    )
    parser.add_argument(
        "--evidence-max-workers",
        type=_positive_int,
        default=2,
        help="Maximum concurrent candidate Evidence extractions per patient.",
    )
    parser.add_argument(
        "--patient-id",
        dest="patient_ids",
        action="append",
        help="Run only this patient ID; may be supplied more than once.",
    )
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when the stable patient folder already has a completed result.",
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
) -> int:
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
        "0",
        "--model",
        args.model,
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
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
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
    if args.limit is not None:
        patient_order = patient_order[: args.limit]
    if not patient_order:
        raise ValueError("No patient IDs matched the requested metadata/filter.")

    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    interrupted = False

    for index, patient_id in enumerate(patient_order, 1):
        case_dir = output_root / f"patient_{patient_id}"
        row = rows_by_patient[patient_id][0]
        case_started = time.monotonic()
        print(f"[{index}/{len(patient_order)}] patient={patient_id}")
        try:
            expected_config = _expected_config(
                patient_id=patient_id,
                metadata_path=metadata_path,
                image_dir=image_dir,
                row=row,
                args=args,
            )
            if not args.force and _case_is_complete(case_dir, expected_config):
                record = {
                    "patient_id": patient_id,
                    "status": "skipped",
                    "reason": "matching_completed_result_exists",
                    "run_dir": str(case_dir),
                    "duration_seconds": 0.0,
                }
                records.append(record)
                print("  skipped: matching completed result exists")
            else:
                _prepare_case_for_run(
                    case_dir,
                    output_root=output_root,
                    patient_id=patient_id,
                    expected_config=expected_config,
                )
                return_code = _run_one_case(
                    patient_id=patient_id,
                    case_dir=case_dir,
                    args=args,
                )
                summary = _read_json(case_dir / "execution_summary.json") or {}
                status = "completed" if return_code == 0 and summary.get("status") == "completed" else "failed"
                record = {
                    "patient_id": patient_id,
                    "status": status,
                    "return_code": return_code,
                    "run_dir": str(case_dir),
                    "duration_seconds": round(time.monotonic() - case_started, 3),
                    "error": summary.get("error") if status == "failed" else None,
                }
                records.append(record)
                print(f"  {status}: {record['duration_seconds']:.3f}s")
                if status == "failed" and args.stop_on_error:
                    break
        except KeyboardInterrupt:
            interrupted = True
            _write_json(
                case_dir / "execution_summary.json",
                {
                    "status": "interrupted",
                    "patient_id": str(patient_id),
                    "run_dir": str(case_dir),
                    "interrupted_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            records.append({
                "patient_id": patient_id,
                "status": "interrupted",
                "run_dir": str(case_dir),
                "duration_seconds": round(time.monotonic() - case_started, 3),
            })
            print(f"  interrupted: patient={patient_id}")
        except Exception as exc:
            error = {
                "status": "failed",
                "patient_id": patient_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            safe_case_path = (
                not case_dir.is_symlink()
                and case_dir.resolve().parent == output_root
                and case_dir.name == f"patient_{patient_id}"
            )
            if safe_case_path and (
                not case_dir.exists()
                or not any(case_dir.iterdir())
                or _case_is_managed(case_dir, patient_id)
            ):
                case_dir.mkdir(parents=True, exist_ok=True)
                _write_json(case_dir / "execution_summary.json", error)
            else:
                print(f"  preserving unrecognized case directory: {case_dir}")
            record = {
                "patient_id": patient_id,
                "status": "failed",
                "run_dir": str(case_dir),
                "duration_seconds": round(time.monotonic() - case_started, 3),
                "error": str(exc),
            }
            records.append(record)
            print(f"  failed: {exc}")
            if args.stop_on_error:
                break

        _write_json(
            output_root / "summary.json",
            {
                "status": "interrupted" if interrupted else "running",
                "started_at": started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "metadata": str(metadata_path),
                "image_dir": str(image_dir),
                "patient_count": len(patient_order),
                "processed_count": len(records),
                "records": records,
            },
        )
        if interrupted:
            break

    failed = sum(record["status"] == "failed" for record in records)
    final_summary = {
        "status": (
            "interrupted" if interrupted
            else "completed" if failed == 0 and len(records) == len(patient_order)
            else "failed"
        ),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "metadata": str(metadata_path),
        "image_dir": str(image_dir),
        "output_root": str(output_root),
        "patient_count": len(patient_order),
        "processed_count": len(records),
        "completed_count": sum(record["status"] == "completed" for record in records),
        "skipped_count": sum(record["status"] == "skipped" for record in records),
        "failed_count": failed,
        "interrupted_count": sum(record["status"] == "interrupted" for record in records),
        "records": records,
    }
    _write_json(output_root / "summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    if interrupted:
        return 130
    return 1 if failed or len(records) != len(patient_order) else 0


if __name__ == "__main__":
    raise SystemExit(main())
