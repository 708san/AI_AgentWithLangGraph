"""Observe actual production function bodies without initializing API clients."""
import ast
import json
import re
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.evaluation.trace import EvaluationTrace, ROOT, snapshot


class Output:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return snapshot(vars(self))


def load_functions(filename, names, namespace):
    path = ROOT / "agent/tools" / filename
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class TraceTests(unittest.TestCase):
    def parser(self):
        return load_functions("diagnosis.py", {"parse_diagnosis_text"},
            {"re": re, "DiagnosisOutput": Output, "DiagnosisFormat": Output})["parse_diagnosis_text"]

    def test_parser_keeps_existing_silent_drop_and_records_original(self):
        parse = self.parser()
        text = "===CASE_START===RANK::1\nDISEASE::A\nOMIM::123456\nDESCRIPTION::support===CASE_END==="
        text += "===CASE_START===RANK:: 2\nDISEASE::B\nDESCRIPTION::lost===CASE_END==="
        expected = snapshot(parse(text))
        previous = sys.gettrace()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            with EvaluationTrace(path):
                with ThreadPoolExecutor(max_workers=1) as pool:
                    actual = snapshot(pool.submit(parse, text).result())
            events = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual["ans"]), 1)
        self.assertIs(sys.gettrace(), previous)
        self.assertEqual(next(e for e in events if e["event"] == "call")["data"]["text"], text)
        checks = [e for e in events if e["event"] == "parse_block_check"]
        self.assertEqual([e["data"]["matched_fields"]["rank_match"] for e in checks], [True, False])

    def test_normalization_preserves_mutation_and_records_decision(self):
        normalize = load_functions("diseaseNormalize.py", {"normalize_zeroshot_results"},
            {"re": re, "State": dict, "Optional": __import__("typing").Optional,
             "ZeroShotOutput": Output, "disease_normalize": lambda name: ("OMIM:123456", "A", .8)})["normalize_zeroshot_results"]
        def state():
            return {"zeroShotResult": Output(ans=[Output(disease_name="A (old)", OMIM_id="999999"),
                                                    Output(disease_name="B", OMIM_id=None)])}
        expected = snapshot(normalize(state()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            with EvaluationTrace(path):
                actual = snapshot(normalize(state()))
            events = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(actual, expected)
        decisions = [e["data"] for e in events if e["event"] == "normalization_decision_input"]
        self.assertEqual(decisions[0]["diag"]["OMIM_id"], "999999")
        self.assertEqual(decisions[0]["disease_name_upper"], "A")
        self.assertEqual(decisions[1]["unique_omim_ids"], ["OMIM:123456"])

    def test_recording_failure_does_not_replace_pipeline_exception(self):
        parse = self.parser()
        with tempfile.TemporaryDirectory() as tmp:
            with EvaluationTrace(Path(tmp) / "trace.jsonl") as trace:
                trace.handle.close()
                with self.assertRaises(TypeError):
                    parse(None)
            self.assertGreater(trace.errors, 0)


if __name__ == "__main__":
    unittest.main()
