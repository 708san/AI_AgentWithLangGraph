"""Run exactly one case through the evidence-grounded v2 pipeline.

Usage example:
    ./.venv/bin/python scripts/run_single_v2_case.py
    ./.venv/bin/python scripts/run_single_v2_case.py --input case.json

The script is intentionally a thin execution harness. It does not alter the
diagnostic flow; it only creates an isolated artifact directory and saves the
input, state, public response, node snapshots, and logs for inspection.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CASE = {
    "patient_id": "single-case-smoke-001",
    "hpo_list": [
        "HP:0001252",
        "HP:0001263",
        "HP:0001290",
    ],
    "absent_hpo_list": [],
    "onset": "childhood",
    "sex": "unknown",
}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items() if key != "llm"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


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


def _load_case(path: str | None, phenopacket_path: str | None = None) -> dict[str, Any]:
    if path and phenopacket_path:
        raise ValueError("Use either --input or --phenopacket, not both.")
    if phenopacket_path:
        source_path = Path(phenopacket_path)
        phenopacket = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(phenopacket, dict):
            raise ValueError("Phenopacket must be a JSON object.")
        subject = phenopacket.get("subject") or {}
        patient_id = subject.get("id") or phenopacket.get("id")
        features = phenopacket.get("phenotypicFeatures") or []
        present_hpo: list[str] = []
        absent_hpo: list[str] = []
        for feature in features:
            if not isinstance(feature, dict):
                continue
            term = feature.get("type") or {}
            hpo_id = term.get("id") if isinstance(term, dict) else None
            if not hpo_id:
                continue
            if feature.get("excluded", False):
                absent_hpo.append(str(hpo_id))
            else:
                present_hpo.append(str(hpo_id))
        return {
            "patient_id": str(patient_id) if patient_id is not None else "",
            "hpo_list": present_hpo,
            "absent_hpo_list": absent_hpo,
            "onset": None,
            "sex": str(subject.get("sex", "unknown")).lower(),
            "phenopacket_id": phenopacket.get("id"),
            "phenopacket_path": str(source_path),
        }
    if not path:
        return dict(DEFAULT_CASE)
    case_path = Path(path)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    if not isinstance(case, dict):
        raise ValueError("Input case must be a JSON object.")
    return case


def _validate_case(case: dict[str, Any]) -> None:
    if not case.get("patient_id"):
        raise ValueError("Input case requires a non-empty patient_id.")
    hpo_list = case.get("hpo_list")
    if not isinstance(hpo_list, list) or not hpo_list:
        raise ValueError("Input case requires a non-empty hpo_list array.")
    if not all(isinstance(item, str) and item.strip() for item in hpo_list):
        raise ValueError("Every hpo_list item must be a non-empty string.")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one v2 rare-disease diagnosis case.")
    parser.add_argument(
        "--input",
        help="JSON case file. If omitted, the built-in single-case smoke fixture is used.",
    )
    parser.add_argument(
        "--phenopacket",
        help="Phenopacket JSON file. Its subject and phenotypicFeatures are converted to one case.",
    )
    parser.add_argument(
        "--output-dir",
        help="Artifact directory. Defaults to local_artifacts/single_case_runs/<timestamp>.",
    )
    parser.add_argument("--model", default="gpt-4o", help="LLM model/deployment name.")
    parser.add_argument(
        "--evidence-mode",
        choices=("v2", "legacy"),
        default="v2",
        help="Execution flow. Defaults to the evidence-grounded v2 flow.",
    )
    parser.add_argument(
        "--max-flow-iterations",
        type=int,
        default=1,
        help="Maximum literature retry iterations after Reflection.",
    )
    parser.add_argument(
        "--image-path",
        help="Optional facial image path for GestaltMatcher.",
    )
    parser.add_argument(
        "--use-phenobrain",
        action="store_true",
        help="Enable the optional PhenoBrain API call.",
    )
    parser.add_argument(
        "--use-absent-hpo",
        action="store_true",
        help="Include absent_hpo_list in the LLM prompts.",
    )
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
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    try:
        case = _load_case(args.input, args.phenopacket)
        _validate_case(case)
        if args.max_flow_iterations < 0:
            raise ValueError("--max-flow-iterations must be non-negative.")

        timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path(args.output_dir or f"local_artifacts/single_case_runs/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = output_dir / "log"
        node_results_dir = output_dir / "node_results"
        log_dir.mkdir(parents=True, exist_ok=True)
        node_results_dir.mkdir(parents=True, exist_ok=True)

        image_path = args.image_path or case.get("image_path")
        copied_image_path = None
        if image_path:
            source_image = Path(image_path)
            if not source_image.is_file():
                raise FileNotFoundError(f"Image file not found: {source_image}")
            input_dir = output_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            copied_image = input_dir / f"face_image{source_image.suffix.lower()}"
            shutil.copy2(source_image, copied_image)
            copied_image_path = str(copied_image)

        case_artifact = dict(case)
        if copied_image_path:
            case_artifact["image_path"] = copied_image_path
        _write_json(output_dir / "input_case.json", case_artifact)
        # Keep the tool-native aggregate patient JSON in node_results/, as in
        # the original runner, while placing public/external handoff snapshots
        # beside the run summary for easy discovery.
        os.environ["AGENT_RESULT_DIR"] = str(output_dir)
        os.environ["AGENT_NODE_RESULT_DIR"] = str(node_results_dir)
        # Set these before importing the pipeline: the worker limits are read
        # from the environment when the literature/evidence modules load.
        os.environ["PUBMED_MAX_WORKERS"] = str(args.pubmed_max_workers)
        os.environ["EVIDENCE_MAX_WORKERS"] = str(args.evidence_max_workers)

        # Import after the artifact path is ready so node save_result decorators
        # use this run's isolated result directory.
        from agent.agent_pipeline import RareDiseaseDiagnosisPipeline
        from agent.utils.response_serializer import serialize_public_state
        from agent.utils.profiler import profiler

        profiler.reset()

        pipeline = RareDiseaseDiagnosisPipeline(
            model_name=args.model,
            enable_log=True,
            log_filename="agent.log",
            log_dir=str(log_dir),
        )

        runtime_log_path = output_dir / "runtime.log"
        pipeline_started = time.monotonic()
        with runtime_log_path.open("w", encoding="utf-8") as runtime_log:
            with contextlib.redirect_stdout(runtime_log), contextlib.redirect_stderr(runtime_log):
                result = pipeline.run(
                    hpo_list=case["hpo_list"],
                    image_path=copied_image_path or image_path,
                    absent_hpo_list=case.get("absent_hpo_list", []),
                    onset=case.get("onset"),
                    sex=case.get("sex"),
                    patient_id=case["patient_id"],
                    use_absentHPO=args.use_absent_hpo or bool(case.get("use_absent_hpo", False)),
                    use_phenobrain=args.use_phenobrain or bool(case.get("use_phenobrain", False)),
                    public_response=False,
                    evidence_mode=args.evidence_mode,
                    max_flow_iterations=args.max_flow_iterations,
                )
        pipeline_seconds = time.monotonic() - pipeline_started
        timing = _build_timing_summary(profiler, pipeline_seconds, time.monotonic() - started)
        _write_json(output_dir / "timing_summary.json", timing)

        _write_json(output_dir / "state.json", result)
        _write_json(output_dir / "public_response.json", serialize_public_state(result))

        summary = {
            "status": "completed",
            "patient_id": case["patient_id"],
            "evidence_mode": args.evidence_mode,
            "worker_config": {
                "pubmed_max_workers": args.pubmed_max_workers,
                "evidence_max_workers": args.evidence_max_workers,
            },
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "pipeline_wall_clock_seconds": timing["pipeline_wall_clock_seconds"],
            "profiled_node_seconds": timing["profiled_node_seconds"],
            "artifact_dir": str(output_dir),
            "files": {
                "input": str(output_dir / "input_case.json"),
                "state": str(output_dir / "state.json"),
                "public_response": str(output_dir / "public_response.json"),
                "external_response": str(output_dir / "external_response.json"),
                "tool_native_patient_result": str(node_results_dir / f"{case['patient_id']}.json"),
                "agent_log": str(log_dir / "agent.log"),
                "runtime_log": str(runtime_log_path),
                "timing_summary": str(output_dir / "timing_summary.json"),
                "node_results": str(node_results_dir),
            },
        }
        _write_json(output_dir / "execution_summary.json", summary)
        _print_timing_summary(timing)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        output_dir = Path(args.output_dir or f"local_artifacts/single_case_runs/{started_at.strftime('%Y%m%dT%H%M%SZ')}")
        output_dir.mkdir(parents=True, exist_ok=True)
        case_value = locals().get("case", {})
        patient_id = case_value.get("patient_id") if isinstance(case_value, dict) else None
        partial_node_results = {}
        if patient_id:
            node_result_path = output_dir / "node_results" / f"{patient_id}.json"
            if node_result_path.exists():
                try:
                    partial_node_results = json.loads(node_result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    partial_node_results = {}
        _write_json(
            output_dir / "state.json",
            {
                "execution_status": "failed",
                "patient_id": patient_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "partial_node_results": partial_node_results,
            },
        )
        error_summary = {
            "status": "failed",
            "patient_id": patient_id,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "artifact_dir": str(output_dir),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "files": {
                "input": str(output_dir / "input_case.json"),
                "state": str(output_dir / "state.json"),
                "agent_log": str(output_dir / "log" / "agent.log"),
                "runtime_log": str(output_dir / "runtime.log"),
                "node_results": str(output_dir / "node_results"),
            },
        }
        _write_json(output_dir / "execution_summary.json", error_summary)
        print(json.dumps(error_summary, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
