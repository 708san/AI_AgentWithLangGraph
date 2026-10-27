"""Run one image-backed Phenopacket case and preserve inspectable artifacts.

This reusable runner is tracked; generated inputs, images, and run artifacts
below ``testCode/`` remain ignored by git. The default case is patient 276 / image 359.

Usage::

    # The harness always exercises the evidence-grounded v2 flow.
    python testCode/phenopacket_image_case/run_one_case.py

Each run creates an ignored directory containing:

    input/phenopacket.json       materialized Phenopacket input
    input/metadata_row.json      source TSV row
    input/face_image.*           copied image used by GestaltMatcher
    logs/agent.log               node results/prompts from pipeline logging
    logs/runtime.log             stdout/stderr from the execution
    results/public_response.json frontend-facing serialized State (pipeline-native)
    results/relevant_results.json compact frontend-compatible tool/diagnosis/evidence view
    results/finalDiagnosis.json  established external-compatible final diagnosis (pipeline-native)
    results/reflection.json      established external-compatible reflection
    results/tentativeDiagnosis.json
    results/mergedDiseaseCandidates.json
    results/zeroShotResult.json
    results/PhenotypeSearchResult.json
    results/GestaltMatcher.json
    results/pubCaseFinder.json
    results/sources.json           source ledger (pipeline-native)
    results/evidence.json          extracted evidence with traceable IDs (pipeline-native)
    results/references.json        numbered references joined to evidence (pipeline-native)
    results/external_response.json all external-compatible node outputs (pipeline-native)
    results/output_contract.json  checks for the v2/frontend output contract
    results/timing_summary.json   node, pipeline, and total elapsed time
    node_results/<patient_id>.json accumulated internal State and per-node snapshots
    execution_summary.json       paths, input identity, and execution status
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_METADATA = (
    PROJECT_ROOT
    / "Data/NewData/phenopacket_v1.0.27_New"
    / "phenopacket_test_metadata_v0.1.27_GM_v1.1.5_with_phenopacket_data.tsv"
)
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "Data/NewData/phenopacket_v1.0.27_New/test_images"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "testCode/phenopacket_image_case/runs"
EVIDENCE_MODE = "v2"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if key != "llm"
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _build_timing_summary(profiler: Any, pipeline_seconds: float, total_seconds: float) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    profiled_seconds = 0.0
    for node_name, timings in sorted(
        profiler.timings.items(),
        key=lambda item: sum(item[1]),
        reverse=True,
    ):
        values = [float(value) for value in timings]
        node_total = sum(values)
        profiled_seconds += node_total
        nodes[node_name] = {
            "count": len(values),
            "total_seconds": round(node_total, 3),
            "average_seconds": round(node_total / len(values), 3) if values else 0.0,
            "min_seconds": round(min(values), 3) if values else 0.0,
            "max_seconds": round(max(values), 3) if values else 0.0,
        }
    return {
        "pipeline_wall_clock_seconds": round(pipeline_seconds, 3),
        "profiled_node_seconds": round(profiled_seconds, 3),
        "total_script_seconds": round(total_seconds, 3),
        "nodes": nodes,
    }


def _print_timing_summary(timing: dict[str, Any]) -> None:
    print("\n=== 実行時間サマリー ===")
    print(f"パイプライン経過時間: {timing['pipeline_wall_clock_seconds']:.3f}秒")
    print(f"スクリプト全体の経過時間: {timing['total_script_seconds']:.3f}秒")
    print("ノード別:")
    for node_name, values in timing["nodes"].items():
        print(
            f"  {node_name}: {values['total_seconds']:.3f}秒 "
            f"(実行回数 {values['count']}, 平均 {values['average_seconds']:.3f}秒)"
        )
    print(
        "※ノード別時間の合計は、並列実行されたノードを重複して数えるため、"
        "パイプライン経過時間より長くなる場合があります。"
    )


def _new_run_dir(output_root: Path, patient_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"patient_{patient_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _existing_or_new_run_dir(
    output_root: Path,
    patient_id: str,
    exact_run_dir: str | None,
) -> Path:
    if exact_run_dir:
        run_dir = Path(exact_run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    return _new_run_dir(output_root, patient_id)


def _validate_v2_public_response(response: dict[str, Any]) -> dict[str, Any]:
    """Check the output contract that the frontend-facing test must exercise."""
    errors: list[str] = []
    warnings: list[str] = []
    if response.get("evidence_mode") != EVIDENCE_MODE:
        errors.append(f"evidence_mode must be {EVIDENCE_MODE!r}")

    for candidate in response.get("mergedDiseaseCandidates", []) or []:
        for ranking in candidate.get("tool_rankings", []) or []:
            if ranking.get("rank") is None:
                errors.append(
                    f"Missing tool rank: {candidate.get('disease_name', 'unknown')} / "
                    f"{ranking.get('tool', 'unknown')}"
                )
        omim_id = candidate.get("OMIM_id")
        if omim_id and not re.fullmatch(r"OMIM:\d+", str(omim_id)):
            errors.append(f"Non-canonical candidate OMIM: {omim_id}")

    # tentativeDiagnosis is an intermediate candidate snapshot.  It is kept
    # for auditability, but the external long-form/citation contract applies
    # to finalDiagnosis, which is the consumer-facing explanation.
    global_reference_records = response.get("references")
    if not isinstance(global_reference_records, list):
        errors.append("top-level references must be an array")
        global_reference_records = []
    if any(not isinstance(record, dict) for record in global_reference_records):
        errors.append("top-level references must contain dictionaries")
    global_reference_numbers = {
        record.get("number")
        for record in global_reference_records
        if isinstance(record, dict)
    }

    for diagnosis_key in ("tentativeDiagnosis", "finalDiagnosis"):
        diagnosis = response.get(diagnosis_key) or {}
        reference_records = diagnosis.get("references", []) or []
        reference_numbers = {
            record.get("number")
            for record in reference_records
            if isinstance(record, dict)
        }
        if any(not isinstance(record, dict) for record in reference_records):
            errors.append(f"{diagnosis_key}.references must be dictionaries")
        if not reference_numbers.issubset(global_reference_numbers):
            errors.append(f"{diagnosis_key}.references are missing from top-level references")
        if "PhenotypeSearch" in str(diagnosis.get("reference", "")):
            errors.append(f"PhenotypeSearch leaked into {diagnosis_key}.reference")
        for item in diagnosis.get("ans", []) or []:
            omim_id = item.get("OMIM_id")
            if omim_id and not re.fullmatch(r"OMIM:\d+", str(omim_id)):
                errors.append(f"Non-canonical diagnosis OMIM: {omim_id}")
            numbers = item.get("citation_numbers", []) or []
            if not set(numbers).issubset(reference_numbers):
                errors.append(f"Reference number mismatch in {diagnosis_key}")
            if diagnosis_key == "finalDiagnosis" and item.get("literature_status") == "supported" and not numbers:
                errors.append(
                    f"Literature-supported {diagnosis_key} item has no inline citations: "
                    f"{item.get('candidate_id')}"
                )
            for description_key in ("description", "description_en", "description_ja"):
                description = str(item.get(description_key, ""))
                if diagnosis_key == "finalDiagnosis" and len(description.strip()) < 120:
                    warnings.append(
                        f"Description is too short in {diagnosis_key}.{description_key}: "
                        f"{item.get('candidate_id')}"
                    )
                cited = {int(value) for value in re.findall(r"\[(\d+)\]", description)}
                if not cited.issubset(global_reference_numbers):
                    errors.append(f"Citation mismatch in {diagnosis_key}.{description_key}")

    reflection = response.get("reflection") or {}
    for item in reflection.get("ans", []) or []:
        numbers = item.get("citation_numbers", []) or []
        if not set(numbers).issubset(global_reference_numbers):
            errors.append("Reference number mismatch in reflection")
        for text_key in ("diagnosis_analysis_en", "diagnosis_analysis_ja"):
            cited = {int(value) for value in re.findall(r"\[(\d+)\]", str(item.get(text_key, "")))}
            if not cited.issubset(global_reference_numbers):
                errors.append(f"Citation mismatch in reflection.{text_key}")

    if response.get("validation_status") not in {"passed", "partial", "failed"}:
        errors.append("validation_status is missing or invalid")
    return {"passed": not errors, "errors": errors, "warnings": warnings}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exactly one image-backed Phenopacket case and save all artifacts."
    )
    parser.add_argument("--patient-id", default="276")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Use an exact output directory instead of a timestamped directory.",
    )
    parser.add_argument(
        "--metadata-occurrence",
        type=int,
        default=None,
        help="Select a zero-based duplicate-row occurrence for batch runs.",
    )
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-flow-iterations", type=int, default=1)
    parser.add_argument("--use-phenobrain", action="store_true")
    parser.add_argument("--use-absent-hpo", action="store_true")
    parser.add_argument(
        "--pubmed-max-workers",
        type=_positive_int,
        default=2,
        help="Maximum concurrent candidate literature searches. Defaults to 2.",
    )
    parser.add_argument(
        "--evidence-max-workers",
        type=_positive_int,
        default=2,
        help="Maximum concurrent candidate Evidence extractions. Defaults to 2.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    started = time.monotonic()
    started_at = datetime.now(timezone.utc)
    run_dir: Path | None = None
    previous_result_dir = os.environ.get("AGENT_RESULT_DIR")
    previous_node_result_dir = os.environ.get("AGENT_NODE_RESULT_DIR")

    try:
        # Importing this adapter keeps the legacy TSV-to-Phenopacket mapping in
        # one place and lets this test use the same image resolution as existing
        # old-case runs.
        from scripts.run_old_phenopacket_patient import (
            _sha256,
            find_image,
            load_patient_row,
            row_to_phenopacket,
        )

        metadata_path = Path(args.metadata).expanduser().resolve()
        image_dir = Path(args.image_dir).expanduser().resolve()
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata TSV not found: {metadata_path}")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        row = load_patient_row(
            metadata_path,
            str(args.patient_id),
            occurrence=args.metadata_occurrence,
        )
        phenopacket = row_to_phenopacket(row)
        source_image = find_image(row, str(image_dir))
        run_dir = _existing_or_new_run_dir(
            Path(args.output_root).expanduser().resolve(),
            str(args.patient_id),
            args.run_dir,
        )

        input_dir = run_dir / "input"
        logs_dir = run_dir / "logs"
        results_dir = run_dir / "results"
        node_results_dir = run_dir / "node_results"
        for directory in (input_dir, logs_dir, results_dir, node_results_dir):
            directory.mkdir(parents=True, exist_ok=True)

        phenopacket_path = input_dir / "phenopacket.json"
        metadata_row_path = input_dir / "metadata_row.json"
        copied_image = input_dir / f"face_image{source_image.suffix.lower()}"
        _write_json(phenopacket_path, phenopacket)
        _write_json(metadata_row_path, row)
        shutil.copy2(source_image, copied_image)

        run_info = {
            "patient_id": str(args.patient_id),
            "phenopacket_id": phenopacket.get("id"),
            "metadata_occurrence": args.metadata_occurrence,
            "metadata_source": str(metadata_path),
            "metadata_sha256": _sha256(metadata_path),
            "image_source": str(source_image),
            "image_used": str(copied_image),
            "image_sha256": _sha256(source_image),
            "present_hpo_count": sum(
                not feature.get("excluded", False)
                for feature in phenopacket.get("phenotypicFeatures", [])
            ),
            "absent_hpo_count": sum(
                feature.get("excluded", False)
                for feature in phenopacket.get("phenotypicFeatures", [])
            ),
        }
        _write_json(run_dir / "run_info.json", run_info)

        run_config = {
            "runner": "phenopacket_image_case_v2",
            "patient_id": str(args.patient_id),
            "metadata_source": str(metadata_path),
            "metadata_sha256": run_info["metadata_sha256"],
            "image_dir": str(image_dir),
            "image_sha256": run_info["image_sha256"],
            "metadata_occurrence": args.metadata_occurrence,
            "model": args.model,
            "max_flow_iterations": args.max_flow_iterations,
            "use_phenobrain": args.use_phenobrain,
            "use_absent_hpo": args.use_absent_hpo,
            "pubmed_max_workers": args.pubmed_max_workers,
            "evidence_max_workers": args.evidence_max_workers,
            "evidence_mode": EVIDENCE_MODE,
        }
        _write_json(run_dir / "run_config.json", run_config)

        # The pipeline's saver owns public/external artifacts; keep its
        # cumulative per-node State in a separate folder for audit/debugging.
        os.environ["AGENT_RESULT_DIR"] = str(results_dir)
        os.environ["AGENT_NODE_RESULT_DIR"] = str(node_results_dir)
        # The worker limits are read at module import time, so set them before
        # importing the pipeline inside the redirected execution block.
        os.environ["PUBMED_MAX_WORKERS"] = str(args.pubmed_max_workers)
        os.environ["EVIDENCE_MAX_WORKERS"] = str(args.evidence_max_workers)

        runtime_log_path = logs_dir / "runtime.log"
        pipeline_started = time.monotonic()
        with runtime_log_path.open("w", encoding="utf-8") as runtime_log:
            with contextlib.redirect_stdout(runtime_log), contextlib.redirect_stderr(runtime_log):
                # Import after AGENT_RESULT_DIR is set so save_result writes
                # into this isolated run directory.
                from agent.agent_pipeline import RareDiseaseDiagnosisPipeline
                from agent.utils.profiler import profiler

                profiler.reset()

                pipeline = RareDiseaseDiagnosisPipeline(
                    model_name=args.model,
                    enable_log=True,
                    log_filename="agent.log",
                    log_dir=str(logs_dir),
                )
                state = pipeline.run(
                    hpo_list=[
                        feature["type"]["id"]
                        for feature in phenopacket.get("phenotypicFeatures", [])
                        if not feature.get("excluded", False)
                    ],
                    absent_hpo_list=[
                        feature["type"]["id"]
                        for feature in phenopacket.get("phenotypicFeatures", [])
                        if feature.get("excluded", False)
                    ],
                    image_path=str(copied_image),
                    onset="Unknown",
                    sex=phenopacket.get("subject", {}).get("sex", "UNKNOWN_SEX"),
                    patient_id=str(args.patient_id),
                    use_absentHPO=args.use_absent_hpo,
                    use_phenobrain=args.use_phenobrain,
                    public_response=False,
                    evidence_mode=EVIDENCE_MODE,
                    max_flow_iterations=args.max_flow_iterations,
                )
        timing = _build_timing_summary(
            profiler,
            time.monotonic() - pipeline_started,
            time.monotonic() - started,
        )
        _write_json(results_dir / "timing_summary.json", timing)

        # These files are written incrementally by the pipeline's native
        # result saver. Read them back for the runner-specific contract check
        # and compact summary; do not serialize or rewrite them here.
        public_response_path = results_dir / "public_response.json"
        external_response_path = results_dir / "external_response.json"
        if not public_response_path.is_file():
            raise FileNotFoundError(
                f"Pipeline did not emit its public response: {public_response_path}"
            )
        if not external_response_path.is_file():
            raise FileNotFoundError(
                f"Pipeline did not emit its external response: {external_response_path}"
            )
        public_response = json.loads(public_response_path.read_text(encoding="utf-8"))
        output_contract = _validate_v2_public_response(public_response)
        relevant_results = {
            "input": run_info,
            "tool_results": {
                "pubCaseFinder": public_response.get("pubCaseFinder", []),
                "GestaltMatcher": public_response.get("GestaltMatcher", []),
                "zeroShotResult": public_response.get("zeroShotResult"),
                "phenoBrain": public_response.get("phenoBrain", []),
                "phenotypeSearchResult": public_response.get("phenotypeSearchResult"),
                "mergedDiseaseCandidates": public_response.get("mergedDiseaseCandidates", []),
            },
            "diagnosis_results": {
                "tentativeDiagnosis": public_response.get("tentativeDiagnosis"),
                "reflection": public_response.get("reflection"),
                "finalDiagnosis": public_response.get("finalDiagnosis"),
            },
            "evidence_results": {
                "sources": public_response.get("sources", []),
                "evidence": public_response.get("evidence", []),
                "validation_status": public_response.get("validation_status"),
                "validation_errors": public_response.get("validation_errors", []),
            },
        }

        _write_json(results_dir / "relevant_results.json", relevant_results)
        _write_json(results_dir / "output_contract.json", output_contract)

        # Always leave the frontend-facing JSON behind, even when the contract
        # check fails.  This makes the failed artifact inspectable instead of
        # hiding the actual response behind the assertion.
        if not output_contract["passed"]:
            raise AssertionError(
                "V2 frontend output contract failed: "
                + "; ".join(output_contract["errors"])
            )

        summary = {
            "status": "completed",
            "patient_id": str(args.patient_id),
            "model": args.model,
            "evidence_mode": EVIDENCE_MODE,
            "worker_config": {
                "pubmed_max_workers": args.pubmed_max_workers,
                "evidence_max_workers": args.evidence_max_workers,
            },
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "pipeline_wall_clock_seconds": timing["pipeline_wall_clock_seconds"],
            "profiled_node_seconds": timing["profiled_node_seconds"],
            "run_dir": str(run_dir),
            "files": {
                "input_phenopacket": str(phenopacket_path),
                "input_image": str(copied_image),
                "agent_log": str(logs_dir / "agent.log"),
                "runtime_log": str(runtime_log_path),
                "state": str(node_results_dir / f"{args.patient_id}.json"),
                "public_response": str(public_response_path),
                "external_response": str(external_response_path),
                "relevant_results": str(results_dir / "relevant_results.json"),
                "final_diagnosis": str(results_dir / "finalDiagnosis.json"),
                "output_contract": str(results_dir / "output_contract.json"),
                "timing_summary": str(results_dir / "timing_summary.json"),
                "node_results": str(node_results_dir),
            },
        }
        _write_json(run_dir / "execution_summary.json", summary)
        _print_timing_summary(timing)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        if run_dir is None:
            run_dir = Path(args.output_root).expanduser().resolve() / f"failed_{int(time.time())}"
            run_dir.mkdir(parents=True, exist_ok=True)
        partial_node_results: dict[str, Any] = {}
        node_result_path = run_dir / "node_results" / f"{args.patient_id}.json"
        if node_result_path.is_file():
            try:
                partial_node_results = json.loads(node_result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                partial_node_results = {}
        _write_json(
            run_dir / "results" / "partial_results.json",
            {
                "execution_status": "failed",
                "patient_id": str(args.patient_id),
                "partial_node_results": partial_node_results,
            },
        )
        error_log = run_dir / "logs" / "runtime.log"
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with error_log.open("a", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        summary = {
            "status": "failed",
            "patient_id": str(args.patient_id),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "run_dir": str(run_dir),
            "duration_seconds": round(time.monotonic() - started, 3),
            "runtime_log": str(error_log),
            "partial_results": str(run_dir / "results" / "partial_results.json"),
            "node_results": str(node_result_path),
        }
        _write_json(run_dir / "execution_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    finally:
        if previous_result_dir is None:
            os.environ.pop("AGENT_RESULT_DIR", None)
        else:
            os.environ["AGENT_RESULT_DIR"] = previous_result_dir
        if previous_node_result_dir is None:
            os.environ.pop("AGENT_NODE_RESULT_DIR", None)
        else:
            os.environ["AGENT_NODE_RESULT_DIR"] = previous_node_result_dir


if __name__ == "__main__":
    raise SystemExit(main())
