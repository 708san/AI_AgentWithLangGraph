"""Read-only JSONL observation for evaluation runs."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
    "ZeroShot.py": {"createZeroshot"},
    "diseaseNormalize.py": {"normalize_zeroshot_results", "disease_normalize"},
    "diagnosis.py": {"createDiagnosisV2", "validate_v2_candidate_output"},
    "finalDiagnosis.py": {"createFinalDiagnosisV2"},
}


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
        self.handle = None
        self.lock = threading.Lock()
        self.previous = None
        self.previous_thread = None
        self.enabled = False
        self.errors = 0

    def _trace(self, frame, event, arg):
        try:
            path = Path(frame.f_code.co_filename)
            if path.resolve().parent != ROOT / "agent/tools" or frame.f_code.co_name not in TARGETS.get(path.name, set()):
                return self._trace
            if event not in {"call", "return", "exception"}:
                return self._trace
            data = {key: snapshot(value) for key, value in frame.f_locals.items() if key != "llm"}
            if event == "return":
                data["returned"] = snapshot(arg)
            elif event == "exception":
                data["exception"] = {"type": arg[0].__name__, "message": str(arg[1])}
            with self.lock:
                self.handle.write(json.dumps({"function": frame.f_code.co_name, "event": event, "line": frame.f_lineno, "data": data}, ensure_ascii=False) + "\n")
                self.handle.flush()
        except Exception:
            self.errors += 1
        return self._trace

    def __enter__(self):
        self.previous = sys.gettrace()
        self.previous_thread = getattr(threading, "gettrace", lambda: None)()
        if self.previous or self.previous_thread:
            return self
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("x", encoding="utf-8")
            self.handle.write(json.dumps({"event": "trace_start", "schema_version": 1}) + "\n")
            sys.settrace(self._trace)
            threading.settrace(self._trace)
            self.enabled = True
        except Exception:
            self.errors += 1
        return self

    def __exit__(self, *exc):
        if self.enabled:
            sys.settrace(self.previous)
            threading.settrace(self.previous_thread)
            self.handle.write(json.dumps({"event": "trace_end", "recording_errors": self.errors}) + "\n")
        if self.handle:
            self.handle.close()
        return False
