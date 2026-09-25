#!/usr/bin/env python3
"""Run ZebraSeek sequentially for the NewData TSV with resumable case bundles.

The runner keeps the existing LangGraph pipeline, node profiler, and pipeline
logger.  Each TSV row is treated as a case because one patient can have more
than one image.  The image ID is therefore the stable case key.

Default behavior is resumable: a case with a completed ``status.json`` and the
same configuration is skipped.  Use ``--overwrite`` to rerun selected cases.
Use ``--limit 1`` or ``--image-ids`` for a small test run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TSV = ROOT / "Data/NewData/phenopacket_v1.0.27_New/phenopacket_test_metadata_v0.1.27_GM_v1.1.5_with_phenopacket_data.tsv"
DEFAULT_IMAGE_DIR = ROOT / "Data/NewData/phenopacket_v1.0.27_New/test_images"
DEFAULT_OUTPUT_DIR = ROOT / "Improved_test/NewData_zebraseek_gpt5-2"


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return jsonable(dump(mode="json"))
        except TypeError:
            return jsonable(dump())
    dump_dict = getattr(value, "dict", None)
    if callable(dump_dict):
        return jsonable(dump_dict())
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_list(values: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        value = value.strip()
        if not value:
            continue
        if value.startswith("["):
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError("JSON list option must contain a list")
            result.extend(str(item).strip() for item in parsed if str(item).strip())
        else:
            result.extend(item.strip() for item in value.split(",") if item.strip())
    return list(dict.fromkeys(result))


def split_ids(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").replace(",", ";").split(";") if item.strip()]


def resolve_image(image_dir: Path, image_id: str) -> Path | None:
    matches = sorted(image_dir.glob(f"{image_id}.*"))
    return matches[0] if matches else None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    image_ids = [str(row.get("image_id", "")).strip() for row in rows]
    duplicates = sorted({image_id for image_id in image_ids if image_id and image_ids.count(image_id) > 1})
    if duplicates:
        raise ValueError(f"image_id must be unique for resumable case bundles: {duplicates[:5]}")
    return rows


def patient_input(row: dict[str, str], image_dir: Path) -> dict[str, Any]:
    image_id = str(row.get("image_id", "")).strip()
    image_path = resolve_image(image_dir, image_id)
    age = row.get("age_note") or ""
    if not age and row.get("age_year"):
        age = f"{row.get('age_year')} years"
        if row.get("age_month"):
            age += f" {row.get('age_month')} months"
    return {
        "patient_id": str(row.get("patient_id", "unknown")),
        "image_id": image_id,
        "present_hpo_ids": split_ids(row.get("pp_present_hpo")),
        "absent_hpo_ids": split_ids(row.get("pp_absent_hpo")),
        "sex": row.get("gender") or "unknown",
        "onset": age or "unknown",
        "image_path": str(image_path) if image_path else None,
    }


class Timing:
    def __init__(self) -> None:
        self.started_at = utc_now()
        self.started = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def record(self, stage: str, started: float, *, status: str = "ok", details: dict | None = None) -> None:
        event = {
            "stage": stage,
            "status": status,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "finished_at": utc_now(),
        }
        if details:
            event["details"] = jsonable(details)
        with self.lock:
            self.events.append(event)


class Tee(io.TextIOBase):
    """Write pipeline output to the terminal and a per-case console log."""

    def __init__(self, original, log_handle):
        self.original = original
        self.log_handle = log_handle

    def write(self, value):
        self.original.write(value)
        self.original.flush()
        self.log_handle.write(value)
        self.log_handle.flush()
        return len(value)

    def flush(self):
        self.original.flush()
        self.log_handle.flush()


def install_timing_hooks(timing: Timing):
    """Instrument the fixed ZebraSeek helpers without changing their behavior."""
    try:
        from agent.tools import zebraseek_flow as flow
    except ImportError:
        return None, {}

    originals = {}
    labels = (
        ("_run_initial_tool", "initial_tool"),
        ("_record_initial_results", "initial_tools_barrier"),
        ("_search_candidate", "candidate_search"),
        ("_reflection_for_candidate", "reflection"),
        ("_rerank", "rerank"),
        ("_gene_annotations", "gene_annotation"),
    )
    for name, label in labels:
        original = getattr(flow, name, None)
        if original is None:
            continue
        originals[name] = original

        def timed(*args, _original=original, _label=label, **kwargs):
            started = time.perf_counter()
            status = "ok"
            value = None
            try:
                value = _original(*args, **kwargs)
                if isinstance(value, dict) and value.get("status") in {"error", "skipped", "not_applicable"}:
                    status = value["status"]
                elif value in (None, [], {}):
                    status = "empty"
                return value
            except Exception:
                status = "error"
                raise
            finally:
                details: dict[str, Any] = {}
                if _label == "initial_tool" and args:
                    details["tool"] = args[0]
                if _label in {"candidate_search", "reflection", "gene_annotation"} and args and isinstance(args[0], dict):
                    details["candidate_id"] = args[0].get("candidate_id", "")
                timing.record(_label, started, status=status, details=details)

        setattr(flow, name, timed)
    return flow, originals


def restore_timing_hooks(flow, originals) -> None:
    if flow is not None:
        for name, original in originals.items():
            setattr(flow, name, original)


def config_for(args) -> dict[str, Any]:
    return {
        "model": args.model,
        "ranking_mode": args.ranking_mode,
        "use_togomcp": not args.no_togomcp,
        "use_phenobrain": not args.no_phenobrain,
        "use_llm": True,
        "workflow": "zebraseek",
    }


def config_hash(config: dict[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if key != "config_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def clean_case_dir(case_dir: Path) -> None:
    if case_dir.exists():
        for child in case_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    case_dir.mkdir(parents=True, exist_ok=True)


def pipeline_result_files(case_dir: Path, result: dict[str, Any]) -> None:
    write_json(case_dir / "final_state.json", result)
    counts = {
        "initial_tools": len(result.get("tool_response_records", [])),
        "candidates": len(result.get("candidate_pool", [])),
        "search_sessions": len(result.get("search_sessions", [])),
        "tool_calls": len(result.get("tool_call_records", [])),
        "prompts": len(result.get("prompt_records", [])),
        "sources": len(result.get("source_records", [])),
        "evidence": len(result.get("evidence_records", [])),
        "reflection": len(result.get("reflection_assessments", [])),
        "final_candidates": len(result.get("final_candidates", [])),
        "gene_annotations": len(result.get("gene_annotations", [])),
    }
    write_json(case_dir / "state_overview.json", {"top_level_keys": sorted(result.keys()), "counts": counts})
    for key, filename in {
        "prompt_records": "prompts.json",
        "procedure_trace": "procedure.json",
        "tool_response_records": "tool_response_records.json",
        "source_records": "source_records.json",
        "evidence_records": "evidence_records.json",
        "candidate_pool": "candidate_records.json",
        "search_sessions": "search_sessions.json",
        "tool_call_records": "tool_call_records.json",
        "reflection_assessments": "reflection_assessments.json",
        "final_candidates": "final_candidates.json",
        "gene_annotations": "gene_annotations.json",
        "final_output": "final_output.json",
    }.items():
        write_json(case_dir / filename, result.get(key, []))
    (case_dir / "procedure.md").write_text(
        "# ZebraSeek procedure\n\n"
        "1. Input normalization\n"
        "2. Parallel initial rankers and Top5 candidate union\n"
        "3. Disease identity normalization\n"
        "4. Fixed candidate research with source and evidence traceability\n"
        "5. Reflection (`correct` / `incorrect` / `uncertain`)\n"
        "6. LLM or tool-average reranking\n"
        "7. Final candidate gene annotation\n\n"
        "See `final_state.json` for the complete state, `prompts.json` for prompts, `procedure.json` for actual bindings, `pipeline_logs/pipeline.log` for the existing logger output, and `node_profile.*` for the existing node profiler.\n",
        encoding="utf-8",
    )


def run_case(case_index: int, row: dict[str, str], args, output_root: Path, run_config: dict[str, Any]) -> dict[str, Any]:
    image_id = str(row.get("image_id", "")).strip()
    patient_id = str(row.get("patient_id", "unknown")).strip()
    case_key = f"{case_index:04d}_image_{image_id}_patient_{patient_id}"
    case_dir = output_root / "cases" / case_key
    clean_case_dir(case_dir)
    started = Timing()
    input_data = patient_input(row, args.image_dir)
    write_json(case_dir / "input.json", {"case_index": case_index, "source_row": row, "patient_input": input_data})
    write_json(case_dir / "status.json", {"status": "running", "case_key": case_key, "config": run_config, "config_hash": config_hash(run_config), "started_at": started.started_at})
    (case_dir / "run.log").write_text(f"started_at={started.started_at}\ncase_index={case_index}\nimage_id={image_id}\npatient_id={patient_id}\n", encoding="utf-8")
    flow = originals = None
    result = None
    error = None
    node_profiler = None
    try:
        from agent.agent_pipeline import RareDiseaseDiagnosisPipeline
        from agent.utils.profiler import profiler as node_profiler

        import inspect
        if "workflow" not in inspect.signature(RareDiseaseDiagnosisPipeline).parameters:
            raise RuntimeError("The checked-out agent pipeline does not contain the ZebraSeek workflow. Checkout the ImproveSearch branch (or merge its ZebraSeek implementation) before running this script.")

        node_profiler.reset()
        case_log_dir = case_dir / "pipeline_logs"
        os.environ["AGENT_RESULT_DIR"] = str(case_dir / "legacy_node_results")
        flow, originals = install_timing_hooks(started)
        pipeline = RareDiseaseDiagnosisPipeline(
            model_name=args.model,
            enable_log=True,
            log_filename="pipeline.log",
            log_dir=str(case_log_dir),
            ranking_mode=args.ranking_mode,
            use_togomcp=not args.no_togomcp,
            use_phenobrain=not args.no_phenobrain,
            workflow="zebraseek",
        )
        pipeline_started = time.perf_counter()
        console_path = case_dir / "console.log"
        with console_path.open("w", encoding="utf-8") as console_handle, redirect_stdout(Tee(sys.__stdout__, console_handle)), redirect_stderr(Tee(sys.__stderr__, console_handle)):
            result = pipeline.run(
                present_hpo_ids=input_data["present_hpo_ids"],
                absent_hpo_ids=input_data["absent_hpo_ids"],
                image_path=input_data["image_path"],
                onset=input_data["onset"],
                sex=input_data["sex"],
                patient_id=patient_id,
                use_phenobrain=not args.no_phenobrain,
                use_togomcp=not args.no_togomcp,
                ranking_mode=args.ranking_mode,
                verbose=False,
            )
        started.record("pipeline_total", pipeline_started, details={"top_level_keys": sorted(result.keys()) if isinstance(result, dict) else []})
        if not isinstance(result, dict):
            raise TypeError(f"Pipeline returned {type(result).__name__}, expected dict")
        pipeline_result_files(case_dir, result)
        status = "ok"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        write_json(case_dir / "error.json", {"error": error})
        status = "error"
    finally:
        restore_timing_hooks(flow, originals)
        finished = utc_now()
        timing_payload = {"started_at": started.started_at, "finished_at": finished, "total_elapsed_ms": round((time.perf_counter() - started.started) * 1000, 2), "events": started.events}
        write_json(case_dir / "timing.json", timing_payload)
        if node_profiler is not None:
            profile_data = {
                name: {"count": len(times), "total_seconds": round(sum(times), 6), "average_seconds": round(sum(times) / len(times), 6) if times else 0.0, "min_seconds": round(min(times), 6) if times else 0.0, "max_seconds": round(max(times), 6) if times else 0.0}
                for name, times in sorted(node_profiler.timings.items())
            }
            write_json(case_dir / "node_profile.json", profile_data)
            (case_dir / "node_profile.txt").write_text(node_profiler.get_summary() + "\n", encoding="utf-8")
        with (case_dir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(f"finished_at={finished}\nstatus={status}\n")
            if error:
                handle.write(f"error={error}\n")
        write_json(case_dir / "status.json", {"status": status, "case_key": case_key, "config": run_config, "config_hash": config_hash(run_config), "started_at": started.started_at, "finished_at": finished, "error": error})
    return {"case_index": case_index, "case_key": case_key, "image_id": image_id, "patient_id": patient_id, "status": status, "error": error, "case_dir": str(case_dir), "elapsed_ms": jsonable((time.perf_counter() - started.started) * 1000)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patient-ids", nargs="+", help="Patient IDs separated by spaces, commas, or one JSON list.")
    parser.add_argument("--image-ids", nargs="+", help="Image IDs separated by spaces, commas, or one JSON list.")
    parser.add_argument("--limit", type=int, help="Run only the first N selected cases; use --limit 1 for a smoke test.")
    parser.add_argument("--overwrite", action="store_true", help="Rerun completed cases and replace their case bundles.")
    parser.add_argument("--model", choices=("gpt-4o", "gpt-5-1", "gpt-5-2"), default="gpt-5-2")
    parser.add_argument("--ranking-mode", choices=("llm", "tool_average"), default="llm")
    parser.add_argument("--no-togomcp", action="store_true")
    parser.add_argument("--no-phenobrain", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List selected cases and input paths without calling ZebraSeek.")
    args = parser.parse_args()
    args.tsv = args.tsv.expanduser().resolve()
    args.image_dir = args.image_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.tsv.is_file():
        parser.error(f"TSV not found: {args.tsv}")
    rows = load_rows(args.tsv)
    patient_ids = set(parse_list(args.patient_ids))
    image_ids = set(parse_list(args.image_ids))
    selected = [(index, row) for index, row in enumerate(rows, start=1) if (not patient_ids or row.get("patient_id") in patient_ids) and (not image_ids or row.get("image_id") in image_ids)]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        selected = selected[: args.limit]
    if not selected:
        parser.error("No cases matched the filters")
    print(f"Selected cases: {len(selected)}")
    for index, row in selected:
        input_data = patient_input(row, args.image_dir)
        print(f"- case_index={index} image_id={row.get('image_id')} patient_id={row.get('patient_id')} HPO+={len(input_data['present_hpo_ids'])} HPO-={len(input_data['absent_hpo_ids'])} image={input_data['image_path']}")
    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = config_for(args)
    run_config["tsv"] = str(args.tsv)
    run_config["image_dir"] = str(args.image_dir)
    run_config["config_hash"] = config_hash(run_config)
    manifest_path = args.output_dir / "run_manifest.json"
    previous = {}
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("cases", {})
        except json.JSONDecodeError:
            previous = {}
    manifest_cases = dict(previous)
    results: list[dict[str, Any]] = []
    for position, (index, row) in enumerate(selected, start=1):
        image_id = str(row.get("image_id", "")).strip()
        patient_id = str(row.get("patient_id", "unknown")).strip()
        case_key = f"{index:04d}_image_{image_id}_patient_{patient_id}"
        case_dir = args.output_dir / "cases" / case_key
        status_path = case_dir / "status.json"
        if not args.overwrite and status_path.is_file():
            try:
                status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                status_payload = {}
            if status_payload.get("status") == "ok" and status_payload.get("config_hash") == config_hash(run_config):
                result = {"case_index": index, "case_key": case_key, "image_id": image_id, "patient_id": patient_id, "status": "skipped_existing", "case_dir": str(case_dir)}
                results.append(result)
                manifest_cases[case_key] = result
                print(f"[{position}/{len(selected)}] skip completed {case_key}", flush=True)
                continue
        print(f"[{position}/{len(selected)}] run {case_key}", flush=True)
        result = run_case(index, row, args, args.output_dir, run_config)
        results.append(result)
        manifest_cases[case_key] = result
        write_json(manifest_path, {"created_at": utc_now(), "config": run_config, "cases": manifest_cases, "last_completed_position": position})
        if result["status"] == "error":
            print(f"  ERROR: {result['error']}", flush=True)
    write_json(manifest_path, {"created_at": utc_now(), "config": run_config, "cases": manifest_cases, "last_completed_position": len(selected)})
    write_json(args.output_dir / "run_summary.json", {"created_at": utc_now(), "config": run_config, "selected_count": len(selected), "results": results})
    errors = sum(item.get("status") == "error" for item in results)
    print(f"Saved run summary: {args.output_dir / 'run_summary.json'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
