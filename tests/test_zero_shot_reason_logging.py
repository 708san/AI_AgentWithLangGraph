"""Exercise real inference, node, graph and logger bodies without external services."""
import ast
from contextlib import redirect_stdout
import datetime
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import ValidationError

from agent.state.state_types import State, ZeroShotOutput, ZeroShotReasonedOutput
from agent.tools.ZeroShot import createZeroshot
from agent.utils.logger import log_node_result
from scripts.evaluation import runner


ROOT = Path(__file__).resolve().parents[1]
REASON = "Supplied seizures and developmental delay support this candidate. 発症年齢は不明。"


def generated_output():
    return ZeroShotReasonedOutput(ans=[
        {"disease_name": "Original disease", "rank": 1, "OMIM_id": "123456", "selection_reason": REASON},
        {"disease_name": "Other disease", "rank": 2, "OMIM_id": None, "selection_reason": "Supported by the supplied seizures."},
    ])


def load_definition(relative_path, symbol, namespace):
    # As in test_evaluation_trace: avoid importing unrelated API clients/indexes,
    # but execute the actual production function/class (not a copied substitute).
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == symbol]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[symbol]


class ZeroShotReasonLoggingTests(unittest.TestCase):
    def llm(self):
        llm = Mock()
        llm.invoke_with_content_filter_retry.return_value = generated_output()
        return llm

    def state(self, llm):
        return {"hpoDict": {"HP:0001250": "Seizures"}, "absentHpoDict": {"HP:0000252": "Microcephaly"},
                "use_absentHPO": True, "onset": "Unknown", "sex": "Unknown", "llm": llm}

    def test_reason_required_and_pipeline_output_is_separate_snapshot(self):
        for value in (None, ""):
            data = generated_output().model_dump()
            if value is None:
                del data["ans"][0]["selection_reason"]
            else:
                data["ans"][0]["selection_reason"] = value
            with self.assertRaises(ValidationError):
                ZeroShotReasonedOutput.model_validate(data)
        llm, saved = self.llm(), []
        output, prompt = createZeroshot(self.state(llm), reasoning_sink=saved.append)
        llm.get_structured_llm.assert_called_once_with(ZeroShotReasonedOutput)
        self.assertIn("selection_reason", prompt)
        self.assertIn("Microcephaly", prompt)
        self.assertIs(type(output), ZeroShotOutput)
        self.assertNotIn("selection_reason", output.model_dump_json())
        self.assertIsNone(output.ans[1].OMIM_id)
        output.ans[0].disease_name = "Normalized name"
        output.ans[0].OMIM_id = "OMIM:654321"
        output.ans.pop()
        self.assertEqual(saved[0], generated_output().model_dump())
        llm.invoke_with_content_filter_retry.assert_called_once()

    def test_missing_inputs_empty_output_and_provider_failure(self):
        sink = Mock()
        self.assertEqual(createZeroshot({}, reasoning_sink=sink), (None, None))
        sink.assert_not_called()
        llm = self.llm()
        llm.invoke_with_content_filter_retry.return_value = None
        output, _ = createZeroshot(self.state(llm), reasoning_sink=sink)
        self.assertIsNone(output)
        sink.assert_not_called()
        llm.invoke_with_content_filter_retry.side_effect = RuntimeError("provider failed")
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            createZeroshot(self.state(llm), reasoning_sink=sink)
        sink.assert_not_called()

    def pipeline_fixture(self, llm):
        node = load_definition("agent/nodes.py", "createZeroShotNode", {
            "State": State, "createZeroshot": createZeroshot, "profile_node": lambda f: f})

        def normalize(state):
            self.assertNotIn("zeroShotReasoning", state)
            self.assertNotIn("selection_reason", state["zeroShotResult"].model_dump_json())
            state["zeroShotResult"].ans[0].disease_name = "Normalized name"
            return {"zeroShotResult": state["zeroShotResult"]}

        nodes = [
            ("BeginningOfFlowNode", lambda state: {"depth": 1, "hpoDict": {"HP:0001250": "Seizures"}}),
            ("createZeroShotNode", node), ("NormalizeZeroShotNode", normalize),
            ("createDiagnosisNode", lambda state: {"tentativeDiagnosis": {"ans": []}}),
            ("diseaseNormalizeNode", lambda state: {}), ("reflectionNode", lambda state: {}),
            ("finalDiagnosisNode", lambda state: {}),
        ]
        names = [START] + [name for name, fn in nodes]
        # reflection -> final is conditional in the real pipeline builder.
        edges = [(a, b) for a, b in zip(names, names[1:]) if a != "reflectionNode"] + [("finalDiagnosisNode", END)]
        cls = load_definition("agent/agent_pipeline.py", "RareDiseaseDiagnosisPipeline", {
            "os": os, "datetime": datetime, "StateGraph": StateGraph, "State": State,
            "NODE_DEFINITIONS": nodes, "EDGES": edges, "log_node_result": log_node_result,
            "get_llm_instance": lambda model: llm})
        return cls, nodes, edges

    def test_real_graph_logs_reasons_only_when_enabled_and_never_passes_them_on(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                llm = self.llm()
                cls, _, _ = self.pipeline_fixture(llm)
                path = Path(tmp) / "run.log"
                pipeline = cls(enable_log=enabled, log_dir=tmp, log_filename=path.name)
                with redirect_stdout(io.StringIO()):
                    result = pipeline.run(["HP:0001250"], patient_id="test")
                self.assertNotIn("zeroShotReasoning", result)
                self.assertNotIn("selection_reason", result["zeroShotResult"].model_dump_json())
                self.assertEqual(result["zeroShotResult"].ans[0].disease_name, "Normalized name")
                self.assertEqual(path.exists(), enabled)
                if enabled:
                    text = path.read_text()
                    self.assertIn("Zero-shot Selection Reasons (before normalization)", text)
                    self.assertIn(REASON, text)
                    start = text.index("----- Zero-shot Selection Reasons (before normalization) -----")
                    stop = text.index("----- End Zero-shot Selection Reasons -----", start)
                    section = text[start:stop]
                    self.assertIn("Original disease", section)
                    self.assertNotIn("Normalized name", section)
                llm.invoke_with_content_filter_retry.assert_called_once()

    def test_cached_zero_shot_is_not_generated_or_given_fabricated_reasons(self):
        llm = self.llm()
        _, nodes, _ = self.pipeline_fixture(llm)
        cached = ZeroShotOutput(ans=[])
        state = {**self.state(llm), "zeroShotResult": cached}
        with redirect_stdout(io.StringIO()):
            result = dict(nodes)["createZeroShotNode"](state)
        self.assertEqual(result, {"zeroShotResult": cached})
        llm.invoke_with_content_filter_retry.assert_not_called()

    def test_evaluation_prefix_keeps_stage_and_prompt_without_log_only_reasons(self):
        cls, nodes, edges = self.pipeline_fixture(self.llm())
        module = SimpleNamespace(RareDiseaseDiagnosisPipeline=cls, NODE_DEFINITIONS=nodes, EDGES=edges)
        with patch.dict("sys.modules", {"agent.agent_pipeline": module}):
            execute = runner.tentative_executor("gpt-5-2")
        record = {}
        with redirect_stdout(io.StringIO()):
            execute(dict(hpo_list=["HP:0001250"], patient_id="test"), record, lambda: None)
        self.assertEqual(record["zeroShotRaw"]["ans"][0]["disease_name"], "Original disease")
        self.assertEqual(record["zeroShotResult"]["ans"][0]["disease_name"], "Normalized name")
        self.assertIn("selection_reason", record["prompts"]["createZeroShotNode"])
        self.assertNotIn("zeroShotReasoning", record)
        self.assertNotIn(REASON, json.dumps(record, ensure_ascii=False))

    def test_real_sdk_requests_reason_field_without_network(self):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
                "model": "gpt-5.2-2025-12-11", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": generated_output().model_dump_json()}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})

        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            model = AzureChatOpenAI(azure_endpoint="https://example.test/", api_key="test",
                                   azure_deployment="test", api_version="2025-04-01-preview", http_client=http)
            llm = SimpleNamespace(get_structured_llm=lambda schema: model.with_structured_output(schema),
                invoke_with_content_filter_retry=lambda runnable, messages, **kw: runnable.invoke(messages))
            saved = []
            output, _ = createZeroshot(self.state(llm), reasoning_sink=saved.append)
        self.assertEqual(len(requests), 1)
        schema = requests[0]["response_format"]["json_schema"]["schema"]
        candidate = schema["$defs"]["ZeroShotReasonedCandidate"]
        self.assertIn("selection_reason", candidate["required"])
        self.assertEqual(saved[0]["ans"][0]["selection_reason"], REASON)
        self.assertNotIn("selection_reason", output.model_dump_json())

    def test_evaluation_runtime_logs_separate_repeats_and_keep_reasons_out_of_predictions(self):
        cls, nodes, edges = self.pipeline_fixture(self.llm())
        module = SimpleNamespace(RareDiseaseDiagnosisPipeline=cls, NODE_DEFINITIONS=nodes, EDGES=edges)
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "log"
            with patch.dict("sys.modules", {"agent.agent_pipeline": module}):
                execute = runner.tentative_executor("gpt-5-2", log_dir=log_dir)
            for repeat in (1, 2):
                record = {"repeat": repeat}
                with redirect_stdout(io.StringIO()):
                    execute(dict(hpo_list=["HP:0001250"], patient_id="test"), record, lambda: None)
                log = log_dir / f"repeat_{repeat:03d}" / "test.log"
                self.assertIn(REASON, log.read_text())
                self.assertNotIn(REASON, json.dumps(record, ensure_ascii=False))
                self.assertNotIn("zeroShotReasoning", record["final_state"])
            self.assertTrue((log_dir / "repeat_001/test.log").is_file())


if __name__ == "__main__":
    unittest.main()
