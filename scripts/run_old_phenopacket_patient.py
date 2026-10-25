"""Run a legacy Phenopacket case with its facial image and preserve the run.

The old evaluation data is distributed as a TSV plus a separate image store,
not as one JSON Phenopacket file.  This adapter selects a patient row,
materializes the Phenopacket fields needed by the agent, and then delegates to
``scripts.run_from_phenopacket``.  Each invocation gets its own run directory
containing the input snapshot, image copy, node results, agent log, stdout /
stderr capture, and an execution summary.

The default case is patient 276, which is recorded as image 359 in the
repository's Phenopacket metadata.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_CANDIDATES = (
    PROJECT_ROOT
    / "Data/NewData/phenopacket_v1.0.27_New/phenopacket_test_metadata_v0.1.27_GM_v1.1.5_with_phenopacket_data.tsv",
    PROJECT_ROOT / "local_artifacts/evaluation/sampleData/filtered_with_phenopacket_new.tsv",
    PROJECT_ROOT / "Data/OldData/phenopacket_test_metadata_old_74_cases_with_phenopacket_data.tsv",
)
DEFAULT_IMAGE_DIRS = (
    PROJECT_ROOT / "Data/NewData/phenopacket_v1.0.27_New/test_images",
    PROJECT_ROOT / "local_artifacts/evaluation/sampleData/PhenoPacketStore_25072025",
    PROJECT_ROOT / "Data/OldData/test_images",
)


def _first_nonempty(row: dict[str, str], names: Iterable[str], default: str = "") -> str:
    for name in names:
        value = (row.get(name) or "").strip()
        if value:
            return value
    return default


def _hpo_ids(value: str) -> list[str]:
    """Extract HPO IDs from both raw Phenopacket TSV and browser TSV values."""
    return list(dict.fromkeys(re.findall(r"HP:\d{7}", value or "")))


def _normalise_sex(value: str) -> str:
    value = (value or "").strip().lower()
    return {
        "male": "MALE",
        "female": "FEMALE",
        "unknown": "UNKNOWN_SEX",
        "unknown_sex": "UNKNOWN_SEX",
    }.get(value, value.upper() if value else "UNKNOWN_SEX")


def _metadata_id(row: dict[str, str]) -> str:
    return _first_nonempty(row, ("patient_id", "subject_id", "id"))


def find_metadata_file(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Metadata TSV not found: {path}")
        return path

    for path in DEFAULT_METADATA_CANDIDATES:
        if path.is_file():
            return path
    candidates = ", ".join(str(path) for path in DEFAULT_METADATA_CANDIDATES)
    raise FileNotFoundError(f"No metadata TSV found. Checked: {candidates}")


def load_patient_row(
    metadata_path: Path,
    patient_id: str,
    occurrence: int | None = None,
) -> dict[str, str]:
    """Load one patient row, optionally selecting a duplicate-row occurrence.

    The default remains strict for the legacy runner.  Batch image-case tests
    can explicitly select the first row for a patient whose TSV contains
    multiple images for the same patient ID.
    """
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Metadata TSV has no header: {metadata_path}")
        matches = [row for row in reader if _metadata_id(row) == str(patient_id)]

    if not matches:
        raise LookupError(f"Patient ID {patient_id} was not found in {metadata_path}")
    if occurrence is not None:
        if occurrence < 0 or occurrence >= len(matches):
            raise IndexError(
                f"Patient ID {patient_id} occurrence {occurrence} is out of range; "
                f"found {len(matches)} rows in {metadata_path}."
            )
        return {key: value or "" for key, value in matches[occurrence].items()}
    if len(matches) > 1:
        raise ValueError(
            f"Patient ID {patient_id} occurs {len(matches)} times in {metadata_path}; "
            "use a metadata file with unique patient IDs."
        )
    return {key: value or "" for key, value in matches[0].items()}


def row_to_phenopacket(row: dict[str, str]) -> dict[str, Any]:
    """Convert a legacy metadata row to the subset consumed by the pipeline."""
    patient_id = _metadata_id(row)
    if not patient_id:
        raise ValueError("The selected metadata row has no patient ID")

    present_value = _first_nonempty(
        row,
        ("pp_present_hpo", "present_features", "hpo_ids_to_paste"),
    )
    absent_value = _first_nonempty(
        row,
        ("pp_absent_hpo", "absent_features", "absent_hpo_ids"),
    )

    features = [
        {"type": {"id": hpo_id}, "excluded": False}
        for hpo_id in _hpo_ids(present_value)
    ]
    features.extend(
        {"type": {"id": hpo_id}, "excluded": True}
        for hpo_id in _hpo_ids(absent_value)
    )

    return {
        "id": _first_nonempty(row, ("phenopacket_id", "PhenoPacket ID"), f"patient_{patient_id}"),
        "subject": {
            "id": str(patient_id),
            "sex": _normalise_sex(_first_nonempty(row, ("gender", "sex"))),
        },
        "phenotypicFeatures": features,
        "metaData": {
            "phenopacketSchemaVersion": "0.1",
            "createdBy": "run_old_phenopacket_patient.py",
            "externalReferences": [
                {
                    "id": _first_nonempty(row, ("PMID", "pmid")),
                    "reference": _first_nonempty(row, ("filename_phenopacket", "PhenoPacket ID")),
                }
            ],
        },
    }


def _candidate_image_names(row: dict[str, str]) -> list[str]:
    image_id = _first_nonempty(row, ("image_id", "imageId"))
    filename = _first_nonempty(row, ("filename", "image_filename", "image_file"))
    names: list[str] = []
    for value in (filename, image_id):
        if not value:
            continue
        name = Path(value).name
        if name not in names:
            names.append(name)
        if not Path(name).suffix:
            for extension in (".png", ".jpg", ".jpeg", ".JPG", ".JPEG"):
                if f"{name}{extension}" not in names:
                    names.append(f"{name}{extension}")
    return names


def find_image(row: dict[str, str], explicit_dir: str | None) -> Path:
    directories = [Path(explicit_dir).expanduser().resolve()] if explicit_dir else []
    directories.extend(path for path in DEFAULT_IMAGE_DIRS if path not in directories)
    names = _candidate_image_names(row)

    for directory in directories:
        if not directory.is_dir():
            continue
        for name in names:
            direct = directory / name
            if direct.is_file():
                return direct
        # Keep extension matching case-insensitive for old image stores.
        by_lower_name = {path.name.lower(): path for path in directory.iterdir() if path.is_file()}
        for name in names:
            found = by_lower_name.get(name.lower())
            if found:
                return found

    checked = ", ".join(str(path) for path in directories)
    raise FileNotFoundError(
        f"No face image found for {names or ['<no image_id>']}. Checked: {checked}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Tee:
    def __init__(self, original: TextIO, capture: TextIO):
        self.original = original
        self.capture = capture

    def write(self, value: str) -> int:
        self.original.write(value)
        self.capture.write(value)
        self.flush()
        return len(value)

    def flush(self) -> None:
        self.original.flush()
        self.capture.flush()

    def isatty(self) -> bool:
        return self.original.isatty()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _new_run_dir(base_dir: Path, patient_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base_dir / f"patient_{patient_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def prepare_run(
    patient_id: str,
    metadata_path: Path,
    image_dir: str | None,
    base_output_dir: Path,
    copy_image: bool,
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    row = load_patient_row(metadata_path, patient_id)
    phenopacket = row_to_phenopacket(row)
    source_image = find_image(row, image_dir)
    run_dir = _new_run_dir(base_output_dir, patient_id)
    input_dir = run_dir / "input"
    input_dir.mkdir()

    phenopacket_path = input_dir / "phenopacket.json"
    metadata_row_path = input_dir / "metadata_row.json"
    _write_json(phenopacket_path, phenopacket)
    _write_json(metadata_row_path, row)

    image_for_run = source_image
    if copy_image:
        image_for_run = input_dir / f"face_image{source_image.suffix.lower()}"
        shutil.copy2(source_image, image_for_run)

    run_info = {
        "patient_id": str(patient_id),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_source": str(metadata_path),
        "image_source": str(source_image),
        "image_used": str(image_for_run),
        "image_sha256": _sha256(source_image),
        "phenopacket": str(phenopacket_path),
        "metadata_row": str(metadata_row_path),
        "present_hpo_count": sum(not feature.get("excluded", False) for feature in phenopacket["phenotypicFeatures"]),
        "absent_hpo_count": sum(feature.get("excluded", False) for feature in phenopacket["phenotypicFeatures"]),
    }
    _write_json(run_dir / "run_info.json", run_info)
    return run_dir, phenopacket, row


def execute(args: argparse.Namespace) -> int:
    metadata_path = find_metadata_file(args.metadata)
    base_output_dir = Path(args.output_base).expanduser().resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    run_dir, phenopacket, row = prepare_run(
        patient_id=str(args.patient_id),
        metadata_path=metadata_path,
        image_dir=args.image_dir,
        base_output_dir=base_output_dir,
        copy_image=not args.no_copy_image,
    )

    process_log_path = run_dir / "process.log"
    summary: dict[str, Any] = {
        "status": "prepared",
        "patient_id": str(args.patient_id),
        "run_dir": str(run_dir),
        "metadata_source": str(metadata_path),
        "phenopacket_id": phenopacket.get("id"),
        "image_id": _first_nonempty(row, ("image_id", "imageId")),
        "model": args.model,
        "dry_run": bool(args.dry_run),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
    }
    _write_json(run_dir / "execution_summary.json", summary)

    started = time.monotonic()
    with process_log_path.open("w", encoding="utf-8") as capture:
        tee_out = _Tee(sys.stdout, capture)
        tee_err = _Tee(sys.stderr, capture)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
            print(f"run_dir={run_dir}")
            print(f"metadata={metadata_path}")
            print(f"image={run_info['image_used']}")
            try:
                if args.dry_run:
                    print("dry-run: pipeline was not invoked")
                    exit_code = 0
                else:
                    if str(PROJECT_ROOT) not in sys.path:
                        sys.path.insert(0, str(PROJECT_ROOT))
                    from scripts.run_from_phenopacket import run_pipeline_from_phenopacket

                    exit_code = run_pipeline_from_phenopacket(
                        str(run_dir / "input/phenopacket.json"),
                        args.model,
                        image_path_arg=run_info["image_used"],
                        output_mode="file",
                        use_phenobrain=args.use_phenobrain,
                        output_dir=str(run_dir),
                    )
            except Exception as exc:  # Preserve the traceback in process.log and summary.
                print(f"execution failed: {exc}", file=sys.stderr)
                import traceback

                traceback.print_exc()
                exit_code = 1

    summary.update(
        {
            "status": "succeeded" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "process_log": str(process_log_path),
            "agent_logs": str(run_dir / "logs"),
            "node_results": str(run_dir / "node_results"),
            "result": str(run_dir / f"{args.patient_id}.json"),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json(run_dir / "execution_summary.json", summary)
    print(f"saved_run={run_dir}")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a legacy Phenopacket patient with its face image and save all run artifacts."
    )
    parser.add_argument("--patient-id", default="276", help="Patient ID to select (default: 276).")
    parser.add_argument("--metadata", help="Legacy metadata TSV. Defaults to the repository's patient-276 metadata.")
    parser.add_argument("--image-dir", help="Directory containing the face image; auto-detected by default.")
    parser.add_argument(
        "--output-base",
        default=str(PROJECT_ROOT / "run_outputs" / "legacy_phenopacket"),
        help="Directory under which a timestamped run directory is created.",
    )
    parser.add_argument("--model", default="gpt-4o", choices=["gpt-4o", "gpt-5-1", "gpt-5-2"])
    parser.add_argument("--use-phenobrain", action="store_true", help="Enable PhenoBrain explicitly.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare and validate artifacts without calling the agent.")
    parser.add_argument("--no-copy-image", action="store_true", help="Use the source image in place instead of copying it into the run.")
    return parser


if __name__ == "__main__":
    raise SystemExit(execute(build_parser().parse_args()))
