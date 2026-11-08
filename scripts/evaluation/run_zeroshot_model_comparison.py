"""Compare the current Zero-shot agent across models and reasoning efforts.

The default experiment is 74 legacy cases x 2 models x 3 reasoning efforts:
``gpt-5-2/gpt-5-4`` and ``none/medium/high``.  Only the Zero-shot node is
called; the rest of the diagnosis graph and all external search tools are
intentionally excluded.

Every condition is saved independently under ``patient_<id>/<model>/<effort>``.
Completed conditions are skipped on the next invocation, while failed or
interrupted conditions are retried.  This makes Ctrl-C and transient API
failures resumable without losing completed results.

Example::

    .venv/bin/python -m scripts.evaluation.run_zeroshot_model_comparison

Use ``--output-root`` to choose a stable experiment directory.  The default
directory is stable so rerunning the command resumes the same experiment.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools.ZeroShot import build_zeroshot_prompt, createZeroshot  # noqa: E402
from agent.tools.make_HPOdic import make_hpo_dic  # noqa: E402
from agent.llm.azure_llm_instance import get_llm_instance  # noqa: E402
from scripts.run_old_phenopacket_patient import (  # noqa: E402
    _metadata_id,
    _normalise_sex,
    _sha256,
    row_to_phenopacket,
)

MONDO_TOOLS = PROJECT_ROOT / "local_artifacts/evaluation/validationCode"
if str(MONDO_TOOLS) not in sys.path:
    sys.path.insert(0, str(MONDO_TOOLS))
from MondoMatcher import MondoOntologyMatcher  # noqa: E402


DEFAULT_METADATA = PROJECT_ROOT / "Data/OldData/browser_input_74_cases.tsv"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "testCode/phenopacket_image_case/runs/zeroshot_model_reasoning_comparison_74"
)
MODELS = ("gpt-5-2", "gpt-5-4")
REASONING_EFFORTS = ("none", "medium", "high")
SCHEMA_VERSION = 1
TOP_K_VALUES = (1, 3, 5)
ZERO_SHOT_MAX_TOKENS = 30_000


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_cases(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {key: value or "" for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
        ]
    if len(rows) != 74:
        raise ValueError(f"Expected 74 metadata rows, found {len(rows)}")
    ids = [_metadata_id(row) for row in rows]
    if any(not patient_id for patient_id in ids) or len(set(ids)) != 74:
        raise ValueError("Expected 74 rows with unique, non-empty patient IDs")
    return rows


def _case_state(row: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    packet = row_to_phenopacket(row)
    features = packet.get("phenotypicFeatures", [])
    present_ids = [
        feature["type"]["id"]
        for feature in features
        if not feature.get("excluded", False)
    ]
    absent_ids = [
        feature["type"]["id"]
        for feature in features
        if feature.get("excluded", False)
    ]
    state = {
        "hpoList": present_ids,
        "hpoDict": make_hpo_dic(present_ids, None),
        "absentHpoList": absent_ids,
        "absentHpoDict": make_hpo_dic(absent_ids, None),
        # Match the current agent's default Zero-shot input configuration.
        "use_absentHPO": False,
        "onset": "Unknown",
        "sex": _normalise_sex(packet.get("subject", {}).get("sex", "")),
    }
    return packet, state


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _condition_key(model: str, effort: str) -> str:
    return f"{model.replace('-', '_')}__{effort}"


def _condition_dir(case_dir: Path, model: str, effort: str) -> Path:
    return case_dir / model / effort


def _condition_is_complete(
    condition_dir: Path,
    *,
    input_fingerprint: str,
    prompt_sha256: str,
    model: str,
    effort: str,
) -> bool:
    status = _read_json(condition_dir / "status.json")
    return bool(
        status
        and status.get("status") == "completed"
        and status.get("input_fingerprint") == input_fingerprint
        and status.get("prompt_sha256") == prompt_sha256
        and status.get("model") == model
        and status.get("reasoning_effort") == effort
        and status.get("max_tokens") == ZERO_SHOT_MAX_TOKENS
        and (condition_dir / "zero_shot.json").is_file()
        and (condition_dir / "reasoned_output.json").is_file()
    )


def _write_condition_status(
    condition_dir: Path,
    *,
    status: str,
    model: str,
    effort: str,
    input_fingerprint: str,
    prompt_sha256: str,
    started_at: str | None = None,
    **extra: Any,
) -> None:
    payload = {
        "status": status,
        "model": model,
        "reasoning_effort": effort,
        "max_tokens": ZERO_SHOT_MAX_TOKENS,
        "input_fingerprint": input_fingerprint,
        "prompt_sha256": prompt_sha256,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    _write_json(condition_dir / "status.json", payload)


def _write_condition_timing(
    condition_dir: Path,
    *,
    model: str,
    effort: str,
    started_at: str,
    finished_at: str | None,
    duration_seconds: float,
    status: str,
) -> None:
    _write_json(
        condition_dir / "timing.json",
        {
            "model": model,
            "reasoning_effort": effort,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
        },
    )


def _make_llm(model: str, effort: str, timeout_seconds: float):
    """Build the Zero-shot comparison client with its configured token budget."""
    wrapper = get_llm_instance(model, reasoning_effort=effort)
    wrapper.llm = wrapper.get_temp_llm_with_max_tokens(
        ZERO_SHOT_MAX_TOKENS,
        timeout_seconds=timeout_seconds,
    )
    return wrapper


def _run_condition(
    *,
    condition_dir: Path,
    llm: Any,
    state: dict[str, Any],
    expected_prompt: str,
    input_fingerprint: str,
    prompt_sha256: str,
    model: str,
    effort: str,
) -> None:
    condition_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    _write_condition_status(
        condition_dir,
        status="running",
        model=model,
        effort=effort,
        input_fingerprint=input_fingerprint,
        prompt_sha256=prompt_sha256,
        started_at=started_at,
    )
    reasoning_payload: dict[str, Any] = {}
    stop_heartbeat = threading.Event()

    def report_elapsed() -> None:
        while not stop_heartbeat.wait(30):
            elapsed = time.monotonic() - started
            print(
                f"[ZeroShot:{model}/{effort}] elapsed={elapsed:.0f}s",
                flush=True,
            )

    heartbeat = threading.Thread(
        target=report_elapsed,
        name=f"zeroshot-timer-{model}-{effort}",
        daemon=True,
    )
    heartbeat.start()
    try:
        result, prompt = createZeroshot(
            {**state, "llm": llm},
            reasoning_sink=reasoning_payload.update,
        )
        if prompt != expected_prompt:
            raise RuntimeError("ZeroShot prompt changed between snapshot and execution")
        if result is None:
            raise RuntimeError("ZeroShot returned no result")
        _write_json(condition_dir / "zero_shot.json", result.model_dump())
        _write_json(condition_dir / "reasoned_output.json", reasoning_payload)
        finished_at = datetime.now(timezone.utc).isoformat()
        duration_seconds = round(time.monotonic() - started, 3)
        _write_condition_status(
            condition_dir,
            status="completed",
            model=model,
            effort=effort,
            input_fingerprint=input_fingerprint,
            prompt_sha256=prompt_sha256,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
        )
        _write_condition_timing(
            condition_dir,
            model=model,
            effort=effort,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            status="completed",
        )
    except KeyboardInterrupt:
        finished_at = datetime.now(timezone.utc).isoformat()
        duration_seconds = round(time.monotonic() - started, 3)
        _write_condition_status(
            condition_dir,
            status="interrupted",
            model=model,
            effort=effort,
            input_fingerprint=input_fingerprint,
            prompt_sha256=prompt_sha256,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
        )
        _write_condition_timing(
            condition_dir,
            model=model,
            effort=effort,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            status="interrupted",
        )
        raise
    except Exception as exc:
        finished_at = datetime.now(timezone.utc).isoformat()
        duration_seconds = round(time.monotonic() - started, 3)
        _write_condition_status(
            condition_dir,
            status="failed",
            model=model,
            effort=effort,
            input_fingerprint=input_fingerprint,
            prompt_sha256=prompt_sha256,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        _write_condition_timing(
            condition_dir,
            model=model,
            effort=effort,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            status="failed",
        )
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=1)


def _run_condition_job(
    *,
    condition_dir: Path,
    llm: Any | None,
    request_timeout_seconds: float,
    state: dict[str, Any],
    expected_prompt: str,
    input_fingerprint: str,
    prompt_sha256: str,
    model: str,
    effort: str,
) -> dict[str, Any]:
    """Run one condition from the global work queue."""
    try:
        if llm is None:
            llm = _make_llm(model, effort, request_timeout_seconds)
        _run_condition(
            condition_dir=condition_dir,
            llm=llm,
            state=state,
            expected_prompt=expected_prompt,
            input_fingerprint=input_fingerprint,
            prompt_sha256=prompt_sha256,
            model=model,
            effort=effort,
        )
    except Exception as exc:
        # _run_condition normally records model errors itself. This catches
        # unexpected worker/setup errors without losing resumable state.
        _write_condition_status(
            condition_dir,
            status="failed",
            model=model,
            effort=effort,
            input_fingerprint=input_fingerprint,
            prompt_sha256=prompt_sha256,
            error_type=type(exc).__name__,
            error=str(exc),
        )
    return _read_json(condition_dir / "status.json") or {
        "status": "unknown",
        "model": model,
        "reasoning_effort": effort,
    }


def _normalize_omim(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"", "NONE", "N/A"}:
        return None
    digits = "".join(character for character in text if character.isdigit())
    return f"OMIM:{digits}" if digits else None


def _ranks_for_gold(
    result_path: Path,
    correct_omim: str,
    matcher: MondoOntologyMatcher,
) -> dict[str, int | None]:
    payload = _read_json(result_path) or {}
    gold = _normalize_omim(correct_omim)
    exact: list[int] = []
    exact_or_close: list[int] = []
    for index, candidate in enumerate(payload.get("ans", []), start=1):
        predicted = _normalize_omim(candidate.get("OMIM_id"))
        try:
            rank = int(candidate.get("rank", index))
        except (TypeError, ValueError):
            continue
        judgment = matcher.judge(gold, predicted)
        if judgment == "MATCH":
            exact.append(rank)
            exact_or_close.append(rank)
        elif judgment == "CLOSE MATCH":
            exact_or_close.append(rank)
    return {
        "exact_rank": min(exact, default=None),
        "exact_or_close_rank": min(exact_or_close, default=None),
    }


def _condition_predictions(condition_dir: Path) -> list[dict[str, Any]]:
    return (_read_json(condition_dir / "zero_shot.json") or {}).get("ans", [])


def _write_summaries(
    output_root: Path,
    rows: list[dict[str, str]],
    conditions: list[tuple[str, str]],
    matcher: MondoOntologyMatcher,
) -> None:
    summary_path = output_root / "comparison_summary.csv"
    fields = ["patient_id", "correct_omim", "prompt_sha256"]
    for model, effort in conditions:
        key = _condition_key(model, effort)
        fields.extend(
            [
                f"{key}_status",
                f"{key}_exact_gold_rank",
                f"{key}_exact_or_close_gold_rank",
                f"{key}_predictions_json",
            ]
        )

    records: list[dict[str, Any]] = []
    for row in rows:
        patient_id = _metadata_id(row)
        case_dir = output_root / f"patient_{patient_id}"
        record: dict[str, Any] = {
            "patient_id": patient_id,
            "correct_omim": row.get("correct_omim", ""),
            "prompt_sha256": "",
        }
        for model, effort in conditions:
            key = _condition_key(model, effort)
            condition_dir = _condition_dir(case_dir, model, effort)
            status = _read_json(condition_dir / "status.json") or {}
            ranks = _ranks_for_gold(
                condition_dir / "zero_shot.json",
                row.get("correct_omim", ""),
                matcher,
            )
            record[f"{key}_status"] = status.get("status", "not_started")
            record[f"{key}_exact_gold_rank"] = ranks["exact_rank"]
            record[f"{key}_exact_or_close_gold_rank"] = ranks[
                "exact_or_close_rank"
            ]
            record[f"{key}_predictions_json"] = json.dumps(
                _condition_predictions(condition_dir), ensure_ascii=False
            )
            record["prompt_sha256"] = record["prompt_sha256"] or status.get(
                "prompt_sha256", ""
            )
        records.append(record)

    temporary = summary_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(summary_path)

    metrics: list[dict[str, Any]] = []
    for model, effort in conditions:
        key = _condition_key(model, effort)
        for top_k in TOP_K_VALUES:
            exact_cases = sum(
                bool(record[f"{key}_exact_gold_rank"])
                and int(record[f"{key}_exact_gold_rank"]) <= top_k
                for record in records
            )
            close_cases = sum(
                bool(record[f"{key}_exact_or_close_gold_rank"])
                and int(record[f"{key}_exact_or_close_gold_rank"]) <= top_k
                for record in records
            )
            metrics.append(
                {
                    "model": model,
                    "reasoning_effort": effort,
                    "condition": key,
                    "top_k": top_k,
                    "exact_cases": exact_cases,
                    "exact_rate": round(exact_cases / len(records), 5),
                    "exact_or_close_cases": close_cases,
                    "exact_or_close_rate": round(close_cases / len(records), 5),
                }
            )
    metrics_path = output_root / "evaluation_summary.csv"
    metrics_temporary = metrics_path.with_suffix(".csv.tmp")
    with metrics_temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "reasoning_effort",
                "condition",
                "top_k",
                "exact_cases",
                "exact_rate",
                "exact_or_close_cases",
                "exact_or_close_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics)
    metrics_temporary.replace(metrics_path)
    _write_json(output_root / "evaluation_summary.json", {"metrics": metrics})


def _write_timing_summary(
    output_root: Path,
    rows: list[dict[str, str]],
    conditions: list[tuple[str, str]],
) -> None:
    condition_summaries = []
    for model, effort in conditions:
        durations: list[float] = []
        completed = 0
        failed = 0
        interrupted = 0
        for row in rows:
            status = _read_json(
                _condition_dir(output_root / f"patient_{_metadata_id(row)}", model, effort)
                / "status.json"
            ) or {}
            duration = status.get("duration_seconds")
            if isinstance(duration, (int, float)):
                durations.append(float(duration))
            if status.get("status") == "completed":
                completed += 1
            elif status.get("status") == "failed":
                failed += 1
            elif status.get("status") == "interrupted":
                interrupted += 1
        condition_summaries.append(
            {
                "model": model,
                "reasoning_effort": effort,
                "condition": _condition_key(model, effort),
                "total_cases": len(rows),
                "completed_cases": completed,
                "failed_cases": failed,
                "interrupted_cases": interrupted,
                "measured_cases": len(durations),
                "total_seconds": round(sum(durations), 3),
                "average_seconds": round(sum(durations) / len(durations), 3)
                if durations
                else None,
                "min_seconds": round(min(durations), 3) if durations else None,
                "max_seconds": round(max(durations), 3) if durations else None,
            }
        )
    _write_json(
        output_root / "timing_summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "conditions": condition_summaries,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current Zero-shot output for all 74 legacy cases."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
        help="Models to compare. Defaults to gpt-5-2 and gpt-5-4.",
    )
    parser.add_argument(
        "--reasoning-efforts",
        nargs="+",
        choices=REASONING_EFFORTS,
        default=list(REASONING_EFFORTS),
        help="Reasoning efforts to compare. Defaults to none, medium, high.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=900,
        help="Azure request timeout per condition.",
    )
    parser.add_argument(
        "--max-parallel-conditions",
        type=int,
        default=6,
        help="Maximum Zero-shot conditions running globally across patients. Defaults to 6.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional smoke-test limit; the default runs all 74 cases.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed conditions instead of resuming them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize all inputs/prompts and configuration without calling APIs.",
    )
    args = parser.parse_args()
    if args.request_timeout_seconds <= 0:
        parser.error("--request-timeout-seconds must be greater than zero")
    if args.max_parallel_conditions < 1:
        parser.error("--max-parallel-conditions must be at least one")
    if args.max_cases is not None and not 1 <= args.max_cases <= 74:
        parser.error("--max-cases must be between 1 and 74")
    return args


def main() -> int:
    args = _parse_args()
    metadata_path = args.metadata.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    load_dotenv(PROJECT_ROOT / ".env")

    all_rows = _load_cases(metadata_path)
    rows = all_rows[: args.max_cases] if args.max_cases else all_rows
    models = list(dict.fromkeys(args.models))
    efforts = list(dict.fromkeys(args.reasoning_efforts))
    conditions = [(model, effort) for model in models for effort in efforts]
    metadata_sha256 = _sha256(metadata_path)
    existing_config = _read_json(output_root / "run_config.json")
    expected_config = {
        "runner": "zeroshot_model_reasoning_comparison",
        "schema_version": SCHEMA_VERSION,
        "metadata_source": str(metadata_path),
        "metadata_sha256": metadata_sha256,
        "models": models,
        "reasoning_efforts": efforts,
        "conditions": [
            {"model": model, "reasoning_effort": effort}
            for model, effort in conditions
        ],
        "request_timeout_seconds": args.request_timeout_seconds,
        "max_tokens": ZERO_SHOT_MAX_TOKENS,
        "prompt_source": "agent/tools/ZeroShot.py:build_zeroshot_prompt",
        "only_zero_shot": True,
        "cases": len(rows),
        "max_parallel_conditions": args.max_parallel_conditions,
    }
    if existing_config:
        for key in (
            "metadata_sha256",
            "models",
            "reasoning_efforts",
            "conditions",
            "prompt_source",
        ):
            if existing_config.get(key) != expected_config.get(key):
                raise ValueError(
                    f"Existing run configuration differs at {key!r}: {output_root}. "
                    "Choose another --output-root for a new experiment."
                )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "run_config.json", expected_config)
    matcher = MondoOntologyMatcher(
        str(MONDO_TOOLS / "mondo-base.json"), child_threshold=50
    )

    run_status = {
        "status": "running",
        "models": models,
        "reasoning_efforts": efforts,
        "conditions": len(conditions),
        "cases": len(rows),
        "max_parallel_conditions": args.max_parallel_conditions,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_root / "run_status.json", run_status)
    attempted = 0
    pending_tasks: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows, start=1):
            patient_id = _metadata_id(row)
            packet, state = _case_state(row)
            prompt = build_zeroshot_prompt(state)
            if prompt is None:
                raise ValueError(f"No ZeroShot prompt for patient {patient_id}")
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            input_snapshot = {
                "phenopacket_id": packet.get("id"),
                "patient_id": patient_id,
                "hpo_ids": state["hpoList"],
                "hpo_labels": list(state["hpoDict"].values()),
                "absent_hpo_ids": state["absentHpoList"],
                "absent_hpo_labels": list(state["absentHpoDict"].values()),
                "use_absentHPO": False,
                "sex": state["sex"],
                "onset": state["onset"],
                "correct_omim": row.get("correct_omim", ""),
            }
            input_fingerprint = _fingerprint(input_snapshot)
            case_dir = output_root / f"patient_{patient_id}"
            case_dir.mkdir(parents=True, exist_ok=True)
            _write_json(case_dir / "input.json", input_snapshot)
            (case_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            if args.dry_run:
                continue

            for model, effort in conditions:
                condition_dir = _condition_dir(case_dir, model, effort)
                if not args.force and _condition_is_complete(
                    condition_dir,
                    input_fingerprint=input_fingerprint,
                    prompt_sha256=prompt_sha256,
                    model=model,
                    effort=effort,
                ):
                    print(
                        f"[{index}/{len(rows)}] patient {patient_id} "
                        f"{model}/{effort} skipped (completed)",
                        flush=True,
                    )
                    continue
                pending_tasks.append(
                    {
                        "case_index": index,
                        "patient_id": patient_id,
                        "model": model,
                        "effort": effort,
                        "condition_dir": condition_dir,
                        "state": state,
                        "prompt": prompt,
                        "input_fingerprint": input_fingerprint,
                        "prompt_sha256": prompt_sha256,
                    }
                )

        attempted = len(pending_tasks)
        if args.dry_run:
            _write_summaries(output_root, rows, conditions, matcher)
            _write_timing_summary(output_root, rows, conditions)
        elif pending_tasks:
            max_workers = min(args.max_parallel_conditions, len(pending_tasks))
            print(
                f"Queued {len(pending_tasks)} incomplete conditions across "
                f"{len(rows)} patients; running continuously with "
                f"{max_workers} workers.",
                flush=True,
            )
            executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="zeroshot-global",
            )
            futures = {
                executor.submit(
                    _run_condition_job,
                    condition_dir=task["condition_dir"],
                    llm=None,
                    request_timeout_seconds=args.request_timeout_seconds,
                    state=task["state"],
                    expected_prompt=task["prompt"],
                    input_fingerprint=task["input_fingerprint"],
                    prompt_sha256=task["prompt_sha256"],
                    model=task["model"],
                    effort=task["effort"],
                ): task
                for task in pending_tasks
            }
            try:
                for future in as_completed(futures):
                    task = futures[future]
                    status = future.result()
                    duration = status.get("duration_seconds")
                    duration_text = (
                        f"{float(duration):.1f}s"
                        if isinstance(duration, (int, float))
                        else "n/a"
                    )
                    print(
                        f"[{task['case_index']}/{len(rows)}] patient "
                        f"{task['patient_id']} {task['model']}/{task['effort']} "
                        f"{status.get('status', 'unknown')} ({duration_text})",
                        flush=True,
                    )
                    run_status.update(
                        {
                            "attempted_conditions": attempted,
                            "completed_tasks": sum(
                                1
                                for completed_future in futures
                                if completed_future.done()
                            ),
                            "current_patient_id": task["patient_id"],
                            "current_condition": _condition_key(
                                task["model"], task["effort"]
                            ),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    _write_json(output_root / "run_status.json", run_status)
            except KeyboardInterrupt:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
            _write_summaries(output_root, rows, conditions, matcher)
            _write_timing_summary(output_root, rows, conditions)
    except KeyboardInterrupt:
        _write_summaries(output_root, rows, conditions, matcher)
        _write_timing_summary(output_root, rows, conditions)
        _write_json(
            output_root / "run_status.json",
            {
                **run_status,
                "status": "interrupted",
                "attempted_conditions": attempted,
                "interrupted_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        print("Interrupted; completed conditions are saved and can be resumed.")
        return 130

    if args.dry_run:
        _write_json(
            output_root / "run_status.json",
            {**run_status, "status": "dry_run", "completed_at": datetime.now(timezone.utc).isoformat()},
        )
        return 0

    _write_summaries(output_root, rows, conditions, matcher)
    _write_timing_summary(output_root, rows, conditions)
    incomplete = []
    for row in rows:
        patient_id = _metadata_id(row)
        for model, effort in conditions:
            status = _read_json(
                _condition_dir(output_root / f"patient_{patient_id}", model, effort)
                / "status.json"
            ) or {}
            if status.get("status") != "completed":
                incomplete.append(
                    {
                        "patient_id": patient_id,
                        "model": model,
                        "reasoning_effort": effort,
                        "status": status.get("status", "not_started"),
                    }
                )
    final_status = "completed" if not incomplete else "completed_with_errors"
    _write_json(
        output_root / "run_status.json",
        {
            **run_status,
            "status": final_status,
            "attempted_conditions": attempted,
            "incomplete_conditions": incomplete,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return 0 if not incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
