"""Run one real case through the operational streaming adapter.

This is a local verification runner. It reuses the same legacy metadata/image
materialization as ``run_one_case.py`` and writes every streamed node update so
the stream can be inspected without connecting to the worker database.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_METADATA = (
    PROJECT_ROOT
    / "Data/NewData/phenopacket_v1.0.27_New"
    / "phenopacket_test_metadata_v0.1.27_GM_v1.1.5_with_phenopacket_data.tsv"
)
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "Data/NewData/phenopacket_v1.0.27_New/test_images"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "testCode/phenopacket_image_case/runs/stream"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(value), ensure_ascii=False) + "\n")
        handle.flush()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real case through agent_adapter.run_stream()."
    )
    parser.add_argument("--patient-id", default="276")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--metadata-occurrence", type=int, default=None)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-flow-iterations", type=int, default=1)
    parser.add_argument("--use-phenobrain", action="store_true")
    parser.add_argument("--use-absent-hpo", action="store_true")
    parser.add_argument("--pubmed-max-workers", type=int, default=2)
    parser.add_argument("--evidence-max-workers", type=int, default=2)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    previous_result_dir = os.environ.get("AGENT_RESULT_DIR")
    previous_node_result_dir = os.environ.get("AGENT_NODE_RESULT_DIR")

    try:
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
        if args.run_dir:
            run_dir = Path(args.run_dir).expanduser().resolve()
        else:
            timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
            run_dir = (
                Path(args.output_root).expanduser().resolve()
                / f"patient_{args.patient_id}_{timestamp}"
            )
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=False)

        input_dir = run_dir / "input"
        results_dir = run_dir / "results"
        node_results_dir = run_dir / "node_results"
        stream_dir = run_dir / "stream_outputs"
        logs_dir = run_dir / "logs"
        for directory in (input_dir, results_dir, node_results_dir, stream_dir, logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        copied_image = input_dir / f"face_image{source_image.suffix.lower()}"
        _write_json(input_dir / "phenopacket.json", phenopacket)
        _write_json(input_dir / "metadata_row.json", row)
        shutil.copy2(source_image, copied_image)

        present_hpo_ids = [
            feature["type"]["id"]
            for feature in phenopacket.get("phenotypicFeatures", [])
            if not feature.get("excluded", False)
        ]
        absent_hpo_ids = [
            feature["type"]["id"]
            for feature in phenopacket.get("phenotypicFeatures", [])
            if feature.get("excluded", False)
        ]
        _write_json(
            run_dir / "run_config.json",
            {
                "runner": "agent_adapter_stream_case_v1",
                "patient_id": str(args.patient_id),
                "metadata_source": str(metadata_path),
                "metadata_sha256": _sha256(metadata_path),
                "image_source": str(source_image),
                "image_sha256": _sha256(source_image),
                "model": args.model,
                "evidence_mode": "v2",
                "max_flow_iterations": args.max_flow_iterations,
            },
        )

        os.environ["AGENT_RESULT_DIR"] = str(results_dir)
        os.environ["AGENT_NODE_RESULT_DIR"] = str(node_results_dir)
        os.environ["PUBMED_MAX_WORKERS"] = str(args.pubmed_max_workers)
        os.environ["EVIDENCE_MAX_WORKERS"] = str(args.evidence_max_workers)

        runtime_log_path = logs_dir / "runtime.log"
        stream_events_path = run_dir / "stream_events.jsonl"
        accumulated_updates: dict[str, Any] = {}
        node_update_history: dict[str, list[Any]] = {}
        sequence = 0

        with runtime_log_path.open("w", encoding="utf-8") as runtime_log:
            with contextlib.redirect_stdout(runtime_log), contextlib.redirect_stderr(runtime_log):
                from agent import agent_adapter

                agent_result: dict[str, Any] = {"evidence_mode": "v2"}
                for node_name, node_update in agent_adapter.run_stream(
                    hpo_ids=present_hpo_ids,
                    absent_hpo_ids=absent_hpo_ids,
                    patient_id=str(args.patient_id),
                    sex=phenopacket.get("subject", {}).get("sex", "UNKNOWN_SEX"),
                    onset="Unknown",
                    model_name=args.model,
                    enable_log=True,
                    use_absent_hpo=args.use_absent_hpo,
                    image_path=str(copied_image),
                    evidence_mode="v2",
                    max_flow_iterations=args.max_flow_iterations,
                ):
                    serializable_update = agent_adapter.make_json_serializable(node_update)
                    agent_result.update(node_update)
                    accumulated_updates.update(serializable_update)
                    node_update_history.setdefault(node_name, []).append(serializable_update)
                    event = {
                        "sequence": sequence,
                        "node_name": node_name,
                        "node_update": serializable_update,
                    }
                    _write_json(
                        stream_dir / f"{sequence:03d}_{node_name}.json",
                        event,
                    )
                    _append_jsonl(stream_events_path, event)
                    step_type = agent_adapter.get_step_type_for_node(node_name)
                    if step_type:
                        _write_json(
                            results_dir / "external_steps" / f"{step_type}.json",
                            agent_adapter.build_step_output_json(step_type, agent_result),
                        )
                    _write_json(
                        results_dir / "stream_state.json",
                        {
                            "patient_id": str(args.patient_id),
                            "evidence_mode": "v2",
                            "updates": accumulated_updates,
                            "node_update_history": node_update_history,
                        },
                    )
                    sequence += 1

        summary = {
            "status": "completed",
            "patient_id": str(args.patient_id),
            "event_count": sequence,
            "run_dir": str(run_dir),
            "stream_events": str(stream_events_path),
            "stream_outputs": str(stream_dir),
            "node_results": str(node_results_dir),
            "runtime_log": str(runtime_log_path),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        _write_json(run_dir / "execution_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"stream run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
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
