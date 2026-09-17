#!/usr/bin/env python3
"""Run the ZebraSeek flow for one or more OldData patient IDs.

Examples::

    .venv/bin/python scripts/test_patient11721_improved.py --patient-ids 11721
    .venv/bin/python scripts/test_patient11721_improved.py --patient-ids 11721 272 --ranking-mode tool_average
    .venv/bin/python scripts/test_patient11721_improved.py --patient-ids '["11721", "272"]'

Each patient gets an audit bundle below ``Improved_test/patient_<id>/``. The
bundle includes the final state, prompts, fixed procedure, existing pipeline
log, original node-profiler output, per-stage timing, tool responses, sources,
Evidence and call records.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TSV_PATH = ROOT / "Data/OldData/phenopacket_test_metadata_old_74_cases_with_phenopacket_data.tsv"
DEFAULT_IMAGE_DIR = ROOT / "Data/OldData/test_images"


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return jsonable(dump(mode="json"))
        except TypeError:
            return jsonable(dump())
    if hasattr(value, "dict") and callable(value.dict):
        return jsonable(value.dict())
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def parse_patient_ids(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        value = value.strip()
        if value.startswith("["):
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError("--patient-ids JSON value must be a list")
            result.extend(str(item) for item in parsed)
        else:
            result.extend(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("At least one patient ID is required")
    return list(dict.fromkeys(result))


def load_patient(patient_id: str) -> dict[str, str]:
    with TSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        matches = [row for row in rows if row.get("patient_id") == patient_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for patient_id={patient_id}, found {len(matches)}")
    return matches[0]


class Timing:
    def __init__(self) -> None:
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def record(self, stage: str, started: float, *, status: str = "ok", details: dict | None = None) -> None:
        event = {
            "stage": stage,
            "status": status,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        if details:
            event["details"] = jsonable(details)
        with self.lock:
            self.events.append(event)


def install_timing_hooks(timing: Timing):
    """Instrument the flow without changing production behavior."""
    from agent.tools import zebraseek_flow as flow

    originals = {}

    def wrap(name: str, label: str):
        original = getattr(flow, name)
        originals[name] = original

        def timed(*args, **kwargs):
            started = time.perf_counter()
            value = None
            status = "ok"
            try:
                value = original(*args, **kwargs)
                if isinstance(value, dict) and value.get("status") in {"error", "skipped", "not_applicable"}:
                    status = value["status"]
                elif value in (None, [], {}):
                    status = "empty"
                return value
            except BaseException:
                status = "error"
                raise
            finally:
                details: dict[str, Any] = {}
                if label == "initial_tool" and args:
                    details["tool"] = args[0]
                    if isinstance(value, dict):
                        details["response_status"] = value.get("status", "ok")
                elif label in {"candidate_search", "reflection", "gene_annotation"} and args:
                    details["candidate_id"] = args[0].get("candidate_id") if isinstance(args[0], dict) else ""
                timing.record(label, started, status=status, details=details)

        setattr(flow, name, timed)

    for name, label in (
        ("_run_initial_tool", "initial_tool"),
        ("_record_initial_results", "initial_tools_barrier"),
        ("_search_candidate", "candidate_search"),
        ("_reflection_for_candidate", "reflection"),
        ("_rerank", "rerank"),
        ("_gene_annotations", "gene_annotation"),
    ):
        wrap(name, label)
    return flow, originals


def restore_timing_hooks(flow, originals) -> None:
    for name, original in originals.items():
        setattr(flow, name, original)


def clean_case_dir(case_dir: Path) -> None:
    if case_dir.exists():
        for child in case_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    case_dir.mkdir(parents=True, exist_ok=True)


def run_one(patient_id: str, args, output_root: Path) -> int:
    case_dir = output_root / f"patient_{patient_id}"
    clean_case_dir(case_dir)
    timing = Timing()
    # Reuse the project's original node profiler in addition to the detailed
    # wall-clock events collected below.
    from agent.utils.profiler import profiler as node_profiler
    node_profiler.reset()
    log_lines = [f"started_at={timing.started_at}", f"patient_id={patient_id}", f"tsv={TSV_PATH}"]
    flow = originals = None
    result: dict[str, Any] | None = None
    error: str | None = None
    try:
        row = load_patient(patient_id)
        present = [item for item in row.get("pp_present_hpo", "").split(";") if item]
        absent = [item for item in row.get("pp_absent_hpo", "").split(";") if item]
        image_path = DEFAULT_IMAGE_DIR / f"{row.get('image_id')}.jpg"
        patient_input = {
            "patient_id": row["patient_id"],
            "present_hpo_ids": present,
            "absent_hpo_ids": absent,
            "sex": row.get("gender") or "unknown",
            "onset": row.get("age_note") or (f"{row.get('age_year')} years" if row.get("age_year") else "unknown"),
            "image_path": str(image_path) if image_path.exists() else None,
        }
        write_json(case_dir / "input.json", {"source_row": row, "patient_input": patient_input})
        log_lines.extend([f"present_hpo_count={len(present)}", f"absent_hpo_count={len(absent)}", f"image_path={patient_input['image_path']}"])

        from agent.agent_pipeline import RareDiseaseDiagnosisPipeline

        flow, originals = install_timing_hooks(timing)
        pipeline = RareDiseaseDiagnosisPipeline(
            model_name=args.model,
            enable_log=True,
            log_filename="pipeline.log",
            log_dir=str(case_dir / "pipeline_logs"),
            ranking_mode=args.ranking_mode,
            use_togomcp=not args.no_togomcp,
            llm=False if args.no_llm else None,
        )
        started_run = time.perf_counter()
        result = pipeline.run(
            present_hpo_ids=present,
            absent_hpo_ids=absent,
            image_path=patient_input["image_path"],
            sex=patient_input["sex"],
            onset=patient_input["onset"],
            patient_id=patient_id,
            ranking_mode=args.ranking_mode,
            use_togomcp=not args.no_togomcp,
            use_phenobrain=not args.no_phenobrain,
            verbose=False,
        )
        timing.record("pipeline_total", started_run, details={"candidate_count": len(result.get("candidate_pool", []))})
        log_lines.extend([
            f"candidate_count={len(result.get('candidate_pool', []))}",
            f"final_count={len(result.get('final_candidates', []))}",
            f"source_count={len(result.get('source_records', []))}",
            f"evidence_count={len(result.get('evidence_records', []))}",
            f"tool_call_count={len(result.get('tool_call_records', []))}",
            f"prompt_count={len(result.get('prompt_records', []))}",
        ])
        for candidate in result.get("final_candidates", []):
            reflection = candidate.get("reflection") or {}
            log_lines.append("final_candidate=" + json.dumps({
                "rank": candidate.get("rank"),
                "candidate_id": candidate.get("candidate_id"),
                "disease_name": candidate.get("disease_name"),
                "judgment": reflection.get("judgment"),
                "gene_count": len(candidate.get("genes", [])),
            }, ensure_ascii=False))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        log_lines.append(f"error={error}")
        write_json(case_dir / "error.json", {"error": error})
    finally:
        if flow is not None and originals is not None:
            restore_timing_hooks(flow, originals)
        finished_at = datetime.now(timezone.utc).isoformat()
        total_elapsed_ms = round((time.perf_counter() - timing.started) * 1000, 2)
        write_json(case_dir / "timing.json", {"started_at": timing.started_at, "finished_at": finished_at, "total_elapsed_ms": total_elapsed_ms, "events": timing.events})
        profile_data = {
            name: {
                "count": len(times),
                "total_seconds": round(sum(times), 6),
                "average_seconds": round(sum(times) / len(times), 6) if times else 0.0,
                "min_seconds": round(min(times), 6) if times else 0.0,
                "max_seconds": round(max(times), 6) if times else 0.0,
            }
            for name, times in sorted(node_profiler.timings.items())
        }
        write_json(case_dir / "node_profile.json", profile_data)
        (case_dir / "node_profile.txt").write_text(node_profiler.get_summary() + "\n", encoding="utf-8")
        log_lines.extend([f"finished_at={finished_at}", f"total_elapsed_ms={total_elapsed_ms}"])
        (case_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if result is not None:
        write_json(case_dir / "final_state.json", result)
        write_json(case_dir / "state_overview.json", {
            "top_level_keys": sorted(result.keys()),
            "counts": {
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
            },
        })
        (case_dir / "procedure.md").write_text(
            "# ZebraSeek procedure\n\n"
            "1. Patient input normalization\n"
            "2. Parallel initial rankers: PubCaseFinder, GestaltMatcher, VectorSearch, PhenoBrain, ZeroShot\n"
            "3. Top5 union and full-response retention\n"
            "4. Per-candidate fixed route: resolve_identity -> present HPO / absent HPO / case reports / PubMed (parallel) -> contradiction -> expansion\n"
            "5. One-hop candidates receive the same route without expansion\n"
            "6. Reflection (`correct` / `incorrect` / `uncertain`)\n"
            "7. LLM or tool-average reranking\n"
            "8. Post-ranking causal candidate gene lookup\n\n"
            "See `final_state.json` for the complete state, `prompts.json` for LLM prompts, `procedure.json` for actual bindings/calls, `pipeline_logs/pipeline.log` for the existing pipeline logger output, and `node_profile.txt`/`node_profile.json` for the original node profiler.\n",
            encoding="utf-8",
        )
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
        }.items():
            write_json(case_dir / filename, result.get(key, []))
    return 1 if error else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-ids", nargs="+", required=True, help="Patient IDs separated by spaces, commas, or one JSON list.")
    parser.add_argument("--ranking-mode", choices=("llm", "tool_average"), default="llm")
    parser.add_argument("--model", default="gpt-4o", choices=("gpt-4o", "gpt-5-1", "gpt-5-2"))
    parser.add_argument("--no-togomcp", action="store_true", help="Disable TogoMCP verification.")
    parser.add_argument("--no-llm", action="store_true", help="Skip ZeroShot/Reflection and use deterministic ranking fallback.")
    parser.add_argument("--no-phenobrain", action="store_true", help="Skip the PhenoBrain initial ranker.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Improved_test")
    args = parser.parse_args()
    output_root = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    statuses = [run_one(patient_id, args, output_root) for patient_id in parse_patient_ids(args.patient_ids)]
    write_json(output_root / "run_summary.json", {
        "patient_ids": parse_patient_ids(args.patient_ids),
        "statuses": statuses,
        "options": {
            "ranking_mode": args.ranking_mode,
            "model": args.model,
            "use_togomcp": not args.no_togomcp,
            "use_llm": not args.no_llm,
            "use_phenobrain": not args.no_phenobrain,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return 1 if any(statuses) else 0


if __name__ == "__main__":
    sys.exit(main())
