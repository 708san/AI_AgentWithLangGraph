"""Run isolated evaluation experiments and save every requested stage."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.evaluation.evaluate import ROOT, evaluate, load_cases
from scripts.evaluation.trace import EvaluationTrace


def serialize(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items() if key != "llm"}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(serialize(value), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def resolve_image(value, benchmark, image_root=None):
    if not value:
        return None
    path = Path(value)
    roots = [Path(image_root)] if image_root else [Path(benchmark).parent, Path(benchmark).parent.parent, ROOT]
    candidates = [path] if path.is_absolute() else [root / path for root in roots]
    found = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if len(found) != 1:
        raise ValueError(f"Image must resolve uniquely: {value}; use --image-root")
    return str(found.pop())


def prepare(cases, args, mode):
    inputs = []
    for case in cases:
        data = case.get("input", {})
        present = data.get("present_hpo_list")
        absent = data.get("absent_hpo_list", [])
        if not isinstance(present, list) or not present or not isinstance(absent, list):
            raise ValueError(f"Invalid HPO lists: {case['case_id']}")
        if any(not isinstance(term, str) or not term.startswith("HP:") for term in present + absent):
            raise ValueError(f"Invalid HPO ID: {case['case_id']}")
        inputs.append({
            "hpo_list": present,
            "absent_hpo_list": absent,
            "onset": data.get("onset"),
            "sex": data.get("sex"),
            "patient_id": case["case_id"],
            "use_absentHPO": args.use_absent_hpo,
            "image_path": resolve_image(data.get("image_path"), args.benchmark, args.image_root)
                if mode == "tentative" and not args.no_images else None,
        })
    return inputs


def zero_shot_executor(model):
    """Run only Zero-shot and normalization; never pass expected labels to the model."""
    from agent.llm.azure_llm_instance import get_llm_instance
    from agent.tools.ZeroShot import createZeroshot
    from agent.tools.diseaseNormalize import normalize_zeroshot_results
    from agent.tools.make_HPOdic import make_hpo_dic

    llm = get_llm_instance(model)

    def execute(inputs, record, checkpoint):
        state = {
            "hpoDict": make_hpo_dic(inputs["hpo_list"], None),
            "absentHpoDict": make_hpo_dic(inputs["absent_hpo_list"], None),
            "onset": inputs.get("onset"),
            "sex": inputs.get("sex"),
            "use_absentHPO": inputs.get("use_absentHPO", False),
            "llm": llm,
        }
        result, prompt = createZeroshot(state)
        record["zeroShotRaw"] = deepcopy(serialize(result))
        record.setdefault("prompts", {})["zeroShot"] = prompt
        checkpoint()
        if result is None:
            raise ValueError("Zero-shot returned no output")
        state["zeroShotResult"] = result
        normalized = normalize_zeroshot_results(state)
        record["zeroShotResult"] = serialize(normalized)
        record["zeroShotValidation"] = serialize(state.get("zeroShotValidation", []))
        checkpoint()

    return execute


def tentative_executor(model, *, log_dir=None):
    """Run the current v2 flow and checkpoint internal stage snapshots."""
    from agent.agent_pipeline import RareDiseaseDiagnosisPipeline

    pipeline = RareDiseaseDiagnosisPipeline(
        model_name=model,
        enable_log=log_dir is not None,
        log_dir=str(log_dir) if log_dir else None,
    )

    def execute(inputs, record, checkpoint):
        if log_dir is not None:
            pipeline.log_dir = str(Path(log_dir) / f"repeat_{record['repeat']:03d}")
            pipeline.log_filename = f"{inputs['patient_id']}.log"
            pipeline.logfile_path = pipeline._get_logfile_path()
            pipeline._write_graph_ascii_to_log()
        state = pipeline.run(**inputs, public_response=False, evidence_mode="v2")
        for key in (
            "zeroShotRaw", "zeroShotResult", "zeroShotValidation",
            "mergedDiseaseCandidates", "tentativeRaw", "tentativeDiagnosis",
            "tentativeValidation", "finalDiagnosis", "validation_status", "validation_errors",
        ):
            if key in state:
                record[key] = serialize(state[key])
        record["final_state"] = serialize(state)
        checkpoint()
        if state.get("tentativeDiagnosis") is None:
            raise ValueError("Tentative diagnosis returned no output")

    return execute


def main(mode, argv=None):
    parser = argparse.ArgumentParser(description=f"Run {mode} evaluation.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--model", choices=["gpt-4o", "gpt-5-1", "gpt-5-2"], default="gpt-4o")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--use-absent-hpo", action="store_true")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--enable-log", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1 or any(k < 1 for k in args.ks):
        parser.error("repeats and ks must be positive")
    if args.enable_log and mode != "tentative":
        parser.error("--enable-log is supported by tentative mode only")
    try:
        cases = load_cases(args.benchmark)
        inputs = prepare(cases, args, mode)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(f"Validated {len(cases)} cases, {args.repeats} repeat(s), mode={mode}; no APIs called.")
        return 0

    directory = args.output_dir.resolve()
    if directory.exists():
        parser.error(f"Output directory already exists: {directory}")
    directory.mkdir(parents=True)
    log_directory = ROOT / "log" / directory.name if args.enable_log else None
    if log_directory:
        log_directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "mode": mode,
        "model": args.model,
        "python": platform.python_version(),
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "repeats": args.repeats,
        "use_absent_hpo": args.use_absent_hpo,
        "no_images": args.no_images,
        "enable_log": args.enable_log,
        "source_hashes": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "agent").rglob("*.py")
        },
    }
    save(directory / "metadata.json", metadata)
    stages = ["zeroShotRaw", "zeroShotResult"] if mode == "zero-shot" else [
        "zeroShotRaw", "zeroShotResult", "mergedDiseaseCandidates",
        "tentativeRaw", "tentativeDiagnosis", "finalDiagnosis",
    ]
    failed = False
    old_result_dir = os.environ.get("AGENT_RESULT_DIR")
    try:
        try:
            execute = zero_shot_executor(args.model) if mode == "zero-shot" else tentative_executor(args.model, log_dir=log_directory)
        except Exception as exc:
            save(directory / "initialization_error.json", {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            return 1
        for repeat in range(1, args.repeats + 1):
            run_dir = directory / f"repeat_{repeat:03d}"
            predictions = run_dir / "predictions"
            predictions.mkdir(parents=True)
            os.environ["AGENT_RESULT_DIR"] = str(run_dir / "node_results")
            for case, data in zip(cases, inputs):
                record = {
                    "case_id": case["case_id"], "patient_id": case.get("patient_id"),
                    "input": data, "model": args.model, "repeat": repeat, "status": "running",
                }
                path = predictions / f"{case['case_id']}.json"
                checkpoint = lambda: save(path, record)
                checkpoint()
                started = time.perf_counter()
                try:
                    with EvaluationTrace(run_dir / "traces" / f"{case['case_id']}.jsonl"):
                        execute(data, record, checkpoint)
                    record["status"] = "ok"
                except Exception as exc:
                    record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    failed = True
                record["elapsed_seconds"] = time.perf_counter() - started
                checkpoint()
                print(f"repeat={repeat} case={case['case_id']} status={record['status']}")
            report = evaluate(cases, predictions, stages, sorted(set(args.ks)))
            report["metadata"] = metadata
            save(run_dir / "evaluation.json", report)
    finally:
        if old_result_dir is None:
            os.environ.pop("AGENT_RESULT_DIR", None)
        else:
            os.environ["AGENT_RESULT_DIR"] = old_result_dir
    print(f"Saved: {directory}")
    return 1 if failed else 0
