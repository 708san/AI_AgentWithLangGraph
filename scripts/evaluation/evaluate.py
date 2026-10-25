"""Score saved pipeline stages against benchmark OMIM labels.

This module never calls an external provider. It preserves the original stage
outputs in the report so normalization, omissions, duplicates, and identity
errors can be reviewed after scoring.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGES = (
    "zeroShotRaw",
    "zeroShotResult",
    "mergedDiseaseCandidates",
    "tentativeRaw",
    "tentativeDiagnosis",
    "finalDiagnosis",
)


def omim_id(value):
    match = re.fullmatch(r"(?:OMIM\s*:\s*)?(\d{6})", str(value or "").strip(), re.I)
    return f"OMIM:{match.group(1)}" if match else None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_cases(path):
    payload = read_json(path)
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark must contain a nonempty 'cases' list")
    seen = set()
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("Each case needs a unique, nonempty string case_id")
        seen.add(case_id)
        expected = (case.get("expected_output") or {}).get("omim_id")
        if not omim_id(expected):
            raise ValueError(f"Missing or invalid expected_output.omim_id: {case_id}")
    return cases


def _stage_items(output, stage):
    if output is None:
        return None
    if stage == "mergedDiseaseCandidates":
        if isinstance(output, dict):
            return output.get("mergedDiseaseCandidates", output.get("candidates"))
        return output if isinstance(output, list) else None
    if isinstance(output, dict):
        return output.get("ans")
    return None


def _candidate_id(item):
    if not isinstance(item, dict):
        return None
    return omim_id(item.get("OMIM_id") or item.get("omim_id"))


def score_stage(output, expected, ks, stage="diagnosis"):
    items = _stage_items(output, stage)
    row = {
        "status": "ok",
        "candidate_count": 0,
        "invalid_omim_count": 0,
        "duplicate_omim_count": 0,
        "correct_rank": None,
        "reciprocal_rank": 0.0,
        "hits": {str(k): False for k in ks},
    }
    if items is None:
        row["status"] = "missing_stage"
        return row
    if not isinstance(items, list):
        row["status"] = "invalid_output"
        return row
    row["candidate_count"] = len(items)
    if not items:
        row["status"] = "empty"
        return row

    ranks = []
    ids = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            row["status"] = "invalid_candidate"
            return row
        raw_rank = item.get("rank")
        if stage == "mergedDiseaseCandidates" and raw_rank is None:
            raw_rank = item.get("best_rank") or index
        rank = raw_rank
        if type(rank) is not int or rank < 1:
            row["status"] = "invalid_rank"
            return row
        ranks.append(rank)
        ids.append(_candidate_id(item))
    if len(set(ranks)) != len(ranks):
        row["status"] = "duplicate_rank"
        return row

    row["invalid_omim_count"] = ids.count(None)
    valid_ids = [value for value in ids if value]
    row["duplicate_omim_count"] = len(valid_ids) - len(set(valid_ids))
    matches = [rank for rank, value in zip(ranks, ids) if value == expected]
    if matches:
        rank = min(matches)
        row.update(correct_rank=rank, reciprocal_rank=1 / rank)
        row["hits"] = {str(k): rank <= k for k in ks}
    return row


def load_prediction(directory, case):
    paths = {Path(directory) / f"{case['case_id']}.json"}
    if case.get("patient_id") is not None:
        paths.add(Path(directory) / f"{case['patient_id']}.json")
    found = sorted(path for path in paths if path.is_file())
    if not found:
        return None, None, "missing_file"
    if len(found) > 1:
        return None, None, "ambiguous_files"
    path = found[0]
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object")
        for key in ("case_id", "patient_id"):
            if key in payload and key in case and str(payload[key]) != str(case[key]):
                raise ValueError(f"Mismatched {key}")
        return payload, {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}, None
    except (ValueError, UnicodeError) as exc:
        return None, {"path": str(path.resolve())}, f"invalid_file: {exc}"


def evaluate(cases, directory, stages, ks, bare_stage=None):
    rows = []
    for case in cases:
        payload, source, error = load_prediction(directory, case)
        expected = omim_id(case["expected_output"]["omim_id"])
        row = {
            "case_id": case["case_id"],
            "patient_id": case.get("patient_id"),
            "expected_omim_id": expected,
            "expected_output": copy.deepcopy(case["expected_output"]),
            "source": source,
            "source_payload": copy.deepcopy(payload),
            "stages": {},
        }
        for stage in stages:
            output = payload if payload is not None and bare_stage == stage else (payload or {}).get(stage) if payload else None
            score = score_stage(output, expected, ks, stage)
            score["output"] = copy.deepcopy(output)
            if error:
                score["status"] = error
            elif payload and payload.get("status") == "error" and output is None:
                score["status"] = "run_error"
            row["stages"][stage] = score
        rows.append(row)

    summary = {}
    for stage in stages:
        scores = [row["stages"][stage] for row in rows]
        statuses = {}
        for score in scores:
            statuses[score["status"]] = statuses.get(score["status"], 0) + 1
        summary[stage] = {
            "total_cases": len(cases),
            "status_counts": statuses,
            "mrr": sum(score["reciprocal_rank"] for score in scores) / len(cases),
            "top_k": {
                str(k): {
                    "hits": sum(score["hits"][str(k)] for score in scores),
                    "rate": sum(score["hits"][str(k)] for score in scores) / len(cases),
                }
                for k in ks
            },
        }
    return {"summary": summary, "cases": rows}


def compare(current, baseline, stages, ks):
    result = {}
    for stage in stages:
        changes = {}
        for k in ks:
            gained, lost = [], []
            for new, old in zip(current["cases"], baseline["cases"]):
                n = new["stages"][stage]["hits"][str(k)]
                o = old["stages"][stage]["hits"][str(k)]
                if n and not o:
                    gained.append(new["case_id"])
                if o and not n:
                    lost.append(new["case_id"])
            changes[str(k)] = {
                "gained": gained,
                "lost": lost,
                "rate_delta": current["summary"][stage]["top_k"][str(k)]["rate"] - baseline["summary"][stage]["top_k"][str(k)]["rate"],
            }
        result[stage] = {
            "top_k": changes,
            "mrr_delta": current["summary"][stage]["mrr"] - baseline["summary"][stage]["mrr"],
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=["zeroShotResult", "tentativeDiagnosis", "finalDiagnosis"])
    parser.add_argument("--bare-stage", choices=STAGES)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(k < 1 for k in args.ks):
        parser.error("--ks must be positive integers")
    cases = load_cases(args.cases)
    stages = [args.bare_stage] if args.bare_stage else list(dict.fromkeys(args.stages))
    ks = sorted(set(args.ks))
    report = evaluate(cases, args.predictions_dir, stages, ks, args.bare_stage)
    report["metadata"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(args.cases.resolve()),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "predictions_dir": str(args.predictions_dir.resolve()),
        "matching": "Exact OMIM ID; rank field preserved; all benchmark cases in denominator",
    }
    if args.baseline_dir:
        baseline = evaluate(cases, args.baseline_dir, stages, ks, args.bare_stage)
        report["baseline"] = baseline
        report["comparison"] = compare(report, baseline, stages, ks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for stage, metrics in report["summary"].items():
        print(stage, metrics)
    print(f"Report: {args.output}")
    bad = any(score["status"] not in ("ok", "empty") for row in report["cases"] for score in row["stages"].values())
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
