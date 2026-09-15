"""Run the five image-inclusive benchmark cases and save actual vs expected output."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CASE_FILE = PROJECT_ROOT / "data" / "five_cases.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "run_outputs" / "five_cases"


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def resolve_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_case(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    input_data = case.get("input", {})
    if not input_data.get("present_hpo_list"):
        errors.append("present_hpo_list is empty")
    image_path = resolve_path(input_data.get("image_path"))
    if image_path is None or not image_path.is_file():
        errors.append(f"image not found: {input_data.get('image_path')}")
    return errors


def run_case(pipeline: RareDiseaseDiagnosisPipeline, case: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_from_phenopacket import format_final_diagnosis

    input_data = case["input"]
    image_path = resolve_path(input_data.get("image_path"))
    started = time.perf_counter()
    state = pipeline.run(
        hpo_list=input_data["present_hpo_list"],
        absent_hpo_list=input_data.get("absent_hpo_list", []),
        image_path=str(image_path) if image_path else None,
        onset=input_data.get("onset"),
        sex=input_data.get("sex"),
        patient_id=case.get("patient_id"),
        use_phenobrain=False,
        verbose=False,
    )
    return {
        "case_id": case["case_id"],
        "patient_id": case.get("patient_id"),
        "input": input_data,
        "expected_output": case.get("expected_output", {}),
        "actual_output": format_final_diagnosis(state.get("finalDiagnosis")),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", choices=["gpt-4o", "gpt-5-1", "gpt-5-2"], default="gpt-5-2")
    parser.add_argument("--case-ids", nargs="+", help="Patient IDs to run; default is all five cases.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print inputs without calling the agent.")
    args = parser.parse_args()

    cases = load_cases(args.case_file)
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [case for case in cases if case.get("patient_id") in wanted or case.get("case_id") in wanted]
        if not cases:
            raise SystemExit(f"No matching cases for: {', '.join(args.case_ids)}")

    for case in cases:
        errors = validate_case(case)
        if errors:
            raise SystemExit(f"{case['case_id']}: " + "; ".join(errors))

    print(f"Cases: {len(cases)}")
    for case in cases:
        input_data = case["input"]
        print(f"- {case['case_id']}: HPO+={len(input_data['present_hpo_list'])}, HPO-={len(input_data.get('absent_hpo_list', []))}, image={input_data['image_path']}")

    if args.dry_run:
        return 0

    from agent.agent_pipeline import RareDiseaseDiagnosisPipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    node_result_dir = args.output_dir / "node_results"
    node_result_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AGENT_RESULT_DIR"] = str(node_result_dir)

    pipeline = RareDiseaseDiagnosisPipeline(
        model_name=args.model,
        enable_log=True,
        log_filename=f"five_cases_{args.model}.log",
        log_dir=str(args.output_dir / "logs"),
    )

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] running {case['case_id']} ...")
        try:
            result = run_case(pipeline, case)
            result["status"] = "ok"
        except Exception as exc:  # keep the other cases runnable
            result = {
                "case_id": case["case_id"],
                "patient_id": case.get("patient_id"),
                "input": case["input"],
                "expected_output": case.get("expected_output", {}),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  ERROR: {result['error']}")
        results.append(result)
        with (args.output_dir / f"{case['case_id']}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)

    summary = {
        "model": args.model,
        "case_file": str(args.case_file),
        "results": results,
    }
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output_dir / 'results.json'}")
    return 0 if all(result.get("status") == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
