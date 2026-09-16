"""Benchmark execution; provider imports are delayed until after validation."""
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
from threading import Lock

from scripts.evaluation.evaluate import ROOT, evaluate, load_cases
from scripts.evaluation.trace import EvaluationTrace
from scripts.evaluation.usage import DEFAULT_PRICING, UsageCapture, load_pricing, summarize, format_cost


def serialize(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: serialize(v) for k, v in value.items() if k != "llm"}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    return value


def save(path, value):
    # Atomic replacement inside a newly allocated experiment directory.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(serialize(value), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def resolve_image(value, benchmark, image_root=None):
    if not value:
        return None
    path = Path(value)
    roots = [image_root] if image_root else [benchmark.parent, benchmark.parent.parent, ROOT]
    candidates = [path] if path.is_absolute() else [root / path for root in roots]
    found = {p.resolve() for p in candidates if p.is_file()}
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
        if any(not isinstance(h, str) or not h.startswith("HP:") for h in present + absent):
            raise ValueError(f"Invalid HPO ID: {case['case_id']}")
        inputs.append({"hpo_list": present, "absent_hpo_list": absent,
            "onset": data.get("onset"), "sex": data.get("sex"),
            "patient_id": case["case_id"], "use_absentHPO": args.use_absent_hpo,
            "image_path": resolve_image(data.get("image_path"), args.benchmark, args.image_root)
                if mode == "tentative" and not args.no_images else None})
    return inputs


def zero_shot_executor(model, reasoning_effort=None):
    # Do not import agent.nodes or agent_pipeline: they initialize other tools.
    from agent.llm.azure_llm_instance import get_llm_instance
    from agent.tools.ZeroShot import createZeroshot
    from agent.tools.make_HPOdic import make_hpo_dic
    from agent.tools import diseaseNormalize
    llm = get_llm_instance(model)
    if reasoning_effort is not None:
        # This wrapper is newly created for this evaluation. Production defaults
        # and the tentative executor are not modified.
        llm.llm = llm.llm.model_copy(update={"extra_body": {
            **(llm.llm.extra_body or {}), "reasoning_effort": reasoning_effort}})

    def infer(inputs, record, checkpoint):
        state = {"hpoDict": make_hpo_dic(inputs["hpo_list"], None),
                 "absentHpoDict": make_hpo_dic(inputs["absent_hpo_list"], None),
                 "onset": inputs["onset"], "sex": inputs["sex"],
                 "use_absentHPO": inputs["use_absentHPO"], "llm": llm}
        if not any(state["hpoDict"].values()):
            raise ValueError("No present HPO labels found")
        result, prompt = createZeroshot(state)
        # Production normalization mutates candidates in place. Freeze the raw
        # output and checkpoint it before normalization, including failure cases.
        record["zeroShotRaw"] = deepcopy(serialize(result))
        record["prompts"] = {"zeroShot": prompt}
        record["effective_input"] = serialize(state)
        checkpoint()
        if result is None:
            raise ValueError("Zero-shot returned no output")
        state["zeroShotResult"] = result
        normalized = diseaseNormalize.normalize_zeroshot_results(state)
        record["zeroShotResult"] = serialize(normalized)
        checkpoint()

    def execute(inputs, record, checkpoint):
        record["llm_settings"] = {"model_alias": model, "deployment": llm.deployment_name,
            "api_version": llm.api_version, "extra_body": llm.llm.extra_body,
            "max_tokens": llm.llm.max_tokens}
        with UsageCapture([("zero_shot", llm.llm.root_client),
                           ("normalization", diseaseNormalize.client)], record, checkpoint):
            infer(inputs, record, checkpoint)
    return execute


def tentative_executor(model):
    from langgraph.graph import StateGraph, END
    from agent.agent_pipeline import RareDiseaseDiagnosisPipeline, NODE_DEFINITIONS, EDGES
    from agent.state.state_types import State

    # Derive the prefix from the production graph, preserving its parallel joins.
    excluded = {"diseaseSearchNode", "reflectionNode", "finalDiagnosisNode", "diseaseNormalizeForFinalNode"}
    class TentativePipeline(RareDiseaseDiagnosisPipeline):
        def _build_graph(self):
            builder = StateGraph(State)
            for name, function in NODE_DEFINITIONS:
                if name in excluded:
                    continue
                def wrapped(state, fn=function, node=name):
                    result = fn(state)
                    self.capture(node, result)
                    return result
                builder.add_node(name, wrapped)
            for source, target in EDGES:
                sources = source if isinstance(source, list) else [source]
                if target in excluded or any(s in excluded for s in sources):
                    continue
                builder.add_edge(source, target)
            builder.add_edge("diseaseNormalizeNode", END)
            return builder.compile()

    pipeline = TentativePipeline(model_name=model, enable_log=False)
    lock = Lock()

    def execute(inputs, record, checkpoint):
        def capture(node, result):
            with lock:
                snapshot = serialize(result)
                record.setdefault("node_outputs", {})[node] = snapshot
                if isinstance(snapshot, dict):
                    for key, value in snapshot.items():
                        if key != "prompt":
                            record[key] = value
                    if "prompt" in snapshot:
                        record.setdefault("prompts", {})[node] = snapshot["prompt"]
                    if node == "createZeroShotNode":
                        record["zeroShotRaw"] = snapshot.get("zeroShotResult")
                    if node == "createDiagnosisNode":
                        record["tentativeRaw"] = snapshot.get("tentativeDiagnosis")
                checkpoint()
        pipeline.capture = capture
        state = pipeline.run(**inputs)
        record["final_state"] = serialize(state)
        checkpoint()
        if state.get("tentativeDiagnosis") is None:
            raise ValueError("Tentative diagnosis returned no output")
    return execute


def main(mode, argv=None):
    parser = argparse.ArgumentParser(description=f"Run {mode} inference, save and score (external API calls unless --dry-run).")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--model", choices=["gpt-4o", "gpt-5-1", "gpt-5-2"], default="gpt-4o")
    parser.add_argument("--output-dir", type=Path, help="New experiment directory; existing paths are refused")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--use-absent-hpo", action="store_true")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate case inputs without importing providers or calling APIs")
    if mode == "zero-shot":
        parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"],
                            help="Override reasoning for this Zero-shot evaluation only")
        parser.add_argument("--pricing-file", type=Path, default=DEFAULT_PRICING,
                            help="Retail rates per million tokens; snapshotted in metadata")
    args = parser.parse_args(argv)
    if args.repeats < 1 or any(k < 1 for k in args.ks):
        parser.error("repeats and ks must be positive")
    try:
        pricing = load_pricing(args.pricing_file) if mode == "zero-shot" else None
        if mode == "zero-shot" and args.reasoning_effort is not None:
            if args.model == "gpt-4o" or (args.model == "gpt-5-1" and args.reasoning_effort == "xhigh"):
                raise ValueError("Selected model does not support this reasoning-effort option")
        cases = load_cases(args.benchmark)
        inputs = prepare(cases, args, mode)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(f"Validated {len(cases)} cases, {args.repeats} repeat(s), mode={mode}; no APIs called. Model/data dependencies are not checked.")
        return 0
    directory = (args.output_dir or ROOT / "local_artifacts/evaluation_results" / (mode + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))).resolve()
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output directory already exists: {directory}")
    metadata = {"mode": mode, "model": args.model, "python": platform.python_version(),
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "benchmark": str(args.benchmark.resolve()), "repeats": args.repeats,
        "use_absent_hpo": args.use_absent_hpo, "no_images": args.no_images,
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (ROOT / "agent").rglob("*.py")}}
    if mode == "zero-shot":
        metadata.update(pricing=pricing, reasoning_effort_override=args.reasoning_effort,
            evaluation_source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in Path(__file__).parent.glob("*.py")})
    save(directory / "metadata.json", metadata)
    if mode == "zero-shot":
        print(f"Pricing ({pricing['currency']}): {pricing.get('assumptions', 'User-supplied rates')}")
    old_result_dir = os.environ.get("AGENT_RESULT_DIR")
    stages = ["zeroShotRaw", "zeroShotResult"] if mode == "zero-shot" else ["zeroShotRaw", "zeroShotResult", "tentativeRaw", "tentativeDiagnosis"]
    failed = False
    usage_records = []
    try:
        try:
            if mode == "zero-shot":
                execute = zero_shot_executor(args.model, args.reasoning_effort) if args.reasoning_effort is not None else zero_shot_executor(args.model)
            else:
                execute = tentative_executor(args.model)
        except Exception as exc:
            save(directory / "initialization_error.json", {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"Initialization failed: {type(exc).__name__}: {exc}")
            return 1
        for repeat in range(1, args.repeats + 1):
            repeat_records = []
            run_dir = directory / f"repeat_{repeat:03d}"
            predictions = run_dir / "predictions"
            predictions.mkdir(parents=True)
            os.environ["AGENT_RESULT_DIR"] = str(run_dir / "node_results")
            for case, data in zip(cases, inputs):
                record = {"case_id": case["case_id"], "patient_id": case.get("patient_id"),
                          "input": data, "model": args.model, "repeat": repeat, "status": "running"}
                path = predictions / f"{case['case_id']}.json"
                checkpoint = lambda: save(path, record)
                checkpoint()
                start = time.perf_counter()
                try:
                    # Only whitelisted patient inputs reach inference; expected_output is never passed.
                    with EvaluationTrace(run_dir / "traces" / f"{case['case_id']}.jsonl"):
                        execute(data, record, checkpoint)
                    record["status"] = "ok"
                except Exception as exc:
                    record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    failed = True
                finally:
                    if record["status"] == "running":
                        record["status"] = "interrupted"
                    record["elapsed_seconds"] = time.perf_counter() - start
                    if mode == "zero-shot":
                        record["cost"] = summarize([record], pricing)
                        usage_records.append(record)
                        repeat_records.append(record)
                        save(run_dir / "cost_summary.json", summarize(repeat_records, pricing))
                        save(directory / "cost_summary.json", summarize(usage_records, pricing))
                    checkpoint()
                print(f"repeat={repeat} case={case['case_id']} status={record['status']}")
            report = evaluate(cases, predictions, stages, sorted(set(args.ks)))
            report["metadata"] = metadata
            save(run_dir / "evaluation.json", report)
            print(json.dumps(report["summary"], ensure_ascii=False))
            if mode == "zero-shot":
                print(format_cost(summarize(repeat_records, pricing)))
    finally:
        if old_result_dir is None:
            os.environ.pop("AGENT_RESULT_DIR", None)
        else:
            os.environ["AGENT_RESULT_DIR"] = old_result_dir
    print(f"Saved: {directory}")
    if mode == "zero-shot":
        print(format_cost(summarize(usage_records, pricing)))
    return 1 if failed else 0
