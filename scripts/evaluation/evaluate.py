"""Score saved Zero-shot and tentative predictions against benchmark OMIM labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGES = ("zeroShotRaw", "zeroShotResult", "tentativeRaw", "tentativeDiagnosis")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def omim_id(value):
    """Accept only an OMIM number or explicitly OMIM-prefixed number."""
    match = re.fullmatch(r"(?:OMIM\s*:\s*)?(\d{6})", str(value).strip(), re.I)
    return f"OMIM:{match.group(1)}" if match else None


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
        for identifier in (case_id, case.get("patient_id")):
            if identifier is not None and (not re.fullmatch(r"[\w.-]+", str(identifier)) or str(identifier) in (".", "..")):
                raise ValueError(f"Unsafe filename identifier for {case_id}")
        label = case.get("expected_output", {})
        if not omim_id(label.get("omim_id")):
            raise ValueError(f"Missing or invalid expected_output.omim_id: {case_id}")
    return cases


def load_prediction(directory, case):
    """Accept case_id.json or patient_id.json; refuse ambiguous matches."""
    paths = {directory / f"{case['case_id']}.json"}
    if case.get("patient_id") is not None:
        paths.add(directory / f"{case['patient_id']}.json")
    found = sorted(p for p in paths if p.is_file())
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


def score_stage(output, expected, ks):
    row = {"status": "ok", "candidate_count": 0, "invalid_omim_count": 0,
           "duplicate_omim_count": 0, "correct_rank": None,
           "reciprocal_rank": 0.0, "hits": {str(k): False for k in ks}}
    if output is None:
        row["status"] = "missing_stage"
        return row
    if not isinstance(output, dict) or not isinstance(output.get("ans"), list):
        row["status"] = "invalid_output"
        return row
    candidates = output["ans"]
    row["candidate_count"] = len(candidates)
    if not candidates:
        row["status"] = "empty"
        return row
    ranks = []
    for candidate in candidates:
        rank = candidate.get("rank") if isinstance(candidate, dict) else None
        if type(rank) is not int or rank < 1:
            row["status"] = "invalid_rank"
            return row
        ranks.append(rank)
    if len(set(ranks)) != len(ranks):
        row["status"] = "duplicate_rank"
        return row
    ids = []
    for candidate in candidates:
        upper = omim_id(candidate.get("OMIM_id"))
        lower = omim_id(candidate.get("omim_id"))
        if upper and lower and upper != lower:
            row["status"] = "conflicting_omim_ids"
            return row
        ids.append(upper or lower)
    row["invalid_omim_count"] = ids.count(None)
    valid_ids = [value for value in ids if value]
    row["duplicate_omim_count"] = len(valid_ids) - len(set(valid_ids))
    matches = [rank for rank, value in zip(ranks, ids) if value == expected]
    if matches:
        rank = min(matches)
        row.update(correct_rank=rank, reciprocal_rank=1 / rank)
        row["hits"] = {str(k): rank <= k for k in ks}
    return row


def evaluate(cases, directory, stages, ks, bare_stage=None):
    rows = []
    for case in cases:
        payload, source, error = load_prediction(directory, case)
        expected = omim_id(case["expected_output"]["omim_id"])
        row = {"case_id": case["case_id"], "patient_id": case.get("patient_id"),
               "expected_omim_id": expected,
               "expected_output": copy.deepcopy(case["expected_output"]),
               "source": source,
               "source_payload": copy.deepcopy(payload), "stages": {}}
        for stage in stages:
            output = None
            if payload is not None:
                output = payload if bare_stage == stage else payload.get(stage)
            score = score_stage(output, expected, ks)
            # Preserve original ranks, wording and all extra fields independently
            # of scoring, including outputs rejected by score_stage validation.
            score["output"] = copy.deepcopy(output)
            if error:
                score["status"] = error
            elif payload.get("status") == "error" and output is None:
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
            "total_cases": len(cases), "status_counts": statuses,
            "mrr": sum(s["reciprocal_rank"] for s in scores) / len(cases),
            "top_k": {str(k): {
                "hits": sum(s["hits"][str(k)] for s in scores),
                "rate": sum(s["hits"][str(k)] for s in scores) / len(cases),
            } for k in ks},
        }
    return {"summary": summary, "cases": rows}


def compare(current, baseline, stages, ks):
    comparison = {}
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
            changes[str(k)] = {"gained": gained, "lost": lost,
                "rate_delta": current["summary"][stage]["top_k"][str(k)]["rate"] - baseline["summary"][stage]["top_k"][str(k)]["rate"]}
        comparison[stage] = {"top_k": changes,
            "mrr_delta": current["summary"][stage]["mrr"] - baseline["summary"][stage]["mrr"]}
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "local_artifacts/five_case_benchmark/data/five_cases.json")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=["zeroShotResult", "tentativeDiagnosis"])
    parser.add_argument("--bare-stage", choices=STAGES, help="Explicit stage for files containing only {ans: [...]} (no automatic inference)")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if any(k <= 0 for k in args.ks):
        parser.error("--ks must be positive integers")
    for directory in (args.predictions_dir, args.baseline_dir):
        if directory is not None and not directory.is_dir():
            parser.error(f"Directory does not exist: {directory}")
    try:
        cases = load_cases(args.cases)
    except (OSError, ValueError, AttributeError) as exc:
        parser.error(str(exc))
    ks = sorted(set(args.ks))
    stages = [args.bare_stage] if args.bare_stage else list(dict.fromkeys(args.stages))
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
    output = args.output or ROOT / "local_artifacts/evaluation_results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    except FileExistsError:
        parser.error(f"Refusing to overwrite existing output: {output}")
    for stage, metrics in report["summary"].items():
        print(stage)
        for k, values in metrics["top_k"].items():
            print(f"  Hit@{k}: {values['hits']}/{metrics['total_cases']} ({values['rate']:.1%})")
        print(f"  MRR: {metrics['mrr']:.4f}; statuses: {metrics['status_counts']}")
        if args.baseline_dir:
            print(f"  Change vs baseline: {report['comparison'][stage]}")
    print(f"Report: {output}")
    bad = any(s["status"] not in ("ok", "empty") for row in report["cases"] for s in row["stages"].values())
    if args.baseline_dir:
        bad |= any(s["status"] not in ("ok", "empty") for row in baseline["cases"] for s in row["stages"].values())
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
