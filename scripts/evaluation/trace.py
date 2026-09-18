"""Read-only observation of production frames, enabled only by evaluation.

No wrappers replace production functions; no arguments, return values or State
fields are changed. Exceptions in the observer never escape into inference.
"""
from __future__ import annotations

import json
import linecache
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
    "ZeroShot.py": {"createZeroshot"},
    "diagnosis.py": {"createDiagnosis", "parse_diagnosis_text", "record_diagnosis_attempt"},
    "diseaseNormalize.py": {"normalize_zeroshot_results", "disease_normalize",
        "diseaseNormalizeForDiagnosis", "normalize_pcf_results", "normalize_gestalt_results"},
    "rankingMerge.py": {"merge_ranked_disease_candidates", "_add_candidate"},
}
STATE_FIELDS = {"hpoDict", "absentHpoDict", "onset", "sex", "use_absentHPO",
    "zeroShotResult", "pubCaseFinder", "GestaltMatcher", "phenotypeSearchResult",
    "phenoBrain", "mergedDiseaseCandidates", "webresources", "tentativeDiagnosis"}
FIELDS = {"prompt", "content", "text", "Diagnosis", "diag", "disease_name",
    "omim_id", "tool_ranking", "cleaned_name", "disease_name_upper",
    "omim_label", "sim", "existing_omim_num", "unique_omim_ids", "block",
    "normalized_omim_id", "key", "candidate", "case_blocks"}


def snapshot(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(k): snapshot(v) for k, v in value.items() if k != "llm"}
    if isinstance(value, (list, tuple, set)):
        return [snapshot(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"unserialized_type": type(value).__name__}


class EvaluationTrace:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.counter = 0
        self.calls = {}
        self.errors = 0
        self.handle = None
        self.enabled = False
        self.selected = {}

    def emit(self, item):
        try:
            encoded = json.dumps(item, ensure_ascii=False)
            with self.lock:
                self.handle.write(encoded + "\n")
                self.handle.flush()
        except Exception:
            self.errors += 1

    def observe(self, frame, event, arg):
        try:
            name = frame.f_code.co_name
            code = frame.f_code
            if code not in self.selected:
                path = Path(code.co_filename)
                self.selected[code] = (name in TARGETS.get(path.name, set())
                                       and path.resolve().parent == ROOT / "agent/tools")
            if not self.selected[code]:
                return None
            if event == "call":
                with self.lock:
                    self.counter += 1
                    self.calls[id(frame)] = self.counter
            values = frame.f_locals
            capture = event in ("call", "return", "exception")
            label = event
            if event == "line":
                source = linecache.getline(code.co_filename, frame.f_lineno).strip()
                # Capture before mutation/selection, using the actual values used
                # by production. Record the predicate without reapplying it.
                if source.startswith("if sim >="):
                    capture, label = True, "normalization_decision_input"
                elif source.startswith("if rank_match and"):
                    capture, label = True, "parse_block_check"
            if not capture:
                return self.observe
            data = {k: snapshot(values[k]) for k in FIELDS if k in values}
            if name == "record_diagnosis_attempt":
                data.update({k: snapshot(values[k]) for k in
                             ("attempt", "input_candidates", "output", "validation") if k in values})
            if event == "call" and isinstance(values.get("state"), dict):
                data["input_state"] = {k: snapshot(v) for k, v in values["state"].items() if k in STATE_FIELDS}
            if event == "return":
                data["returned"] = snapshot(arg)
            if label == "parse_block_check":
                data["matched_fields"] = {k: values.get(k) is not None for k in ("rank_match", "disease_match", "omim_match", "desc_match")}
            response = values.get("response")
            if response is not None:
                data["llm_response"] = {k: snapshot(getattr(response, k, None)) for k in ("content", "id", "response_metadata", "usage_metadata")}
            if event == "exception":
                data["exception"] = {"type": arg[0].__name__, "message": str(arg[1])}
            if name in {"createDiagnosis", "record_diagnosis_attempt"} and values.get("parsing_error") is not None:
                error = values["parsing_error"]
                data["parsing_error"] = {"type": type(error).__name__, "message": str(error)}
            # A parser call is recorded before any parsing: retain its caller's
            # prompt and response metadata even when parsing subsequently fails.
            if name == "parse_diagnosis_text" and event == "call":
                caller = frame.f_back.f_locals
                data["prompt"] = snapshot(caller.get("prompt"))
                response = caller.get("response")
                if response is not None:
                    data["response_metadata"] = snapshot(getattr(response, "response_metadata", None))
            self.emit({"call_id": self.calls.get(id(frame)), "thread": threading.get_ident(),
                       "function": name, "event": label, "line": frame.f_lineno, "data": data})
            if event == "return":
                self.calls.pop(id(frame), None)
        except Exception:
            self.errors += 1
        return self.observe

    def __enter__(self):
        self.previous = sys.gettrace()
        self.previous_thread = getattr(threading, "gettrace", lambda: getattr(threading, "_trace_hook", None))()
        if self.previous or self.previous_thread:
            # Do not displace a debugger/coverage tracing hook.
            print("[evaluation trace] Disabled: an existing debugger/trace hook is active.")
            return self
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("x", encoding="utf-8")
            self.emit({"event": "trace_start", "schema_version": 1})
            threading.settrace(self.observe)
            sys.settrace(self.observe)
            self.enabled = True
        except Exception as exc:
            sys.settrace(self.previous)
            threading.settrace(self.previous_thread)
            self.errors += 1
            print(f"[evaluation trace] Could not start: {type(exc).__name__}")
        return self

    def __exit__(self, *exc):
        if self.enabled:
            sys.settrace(self.previous)
            threading.settrace(self.previous_thread)
            self.emit({"event": "trace_end", "recording_errors": self.errors})
        if self.handle:
            try:
                self.handle.close()
            except Exception:
                self.errors += 1
        if self.errors:
            print(f"[evaluation trace] Recording errors: {self.errors}; trace may be incomplete.")
        return False
