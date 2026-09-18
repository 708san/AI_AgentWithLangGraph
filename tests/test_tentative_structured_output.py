import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI

from agent.state.state_types import TentativeDiagnosisOutput as DiagnosisOutput, TentativeDiagnosisCandidate as DiagnosisFormat
from agent.tools.diagnosis import createDiagnosis
from scripts.evaluation.trace import EvaluationTrace


class TentativeStructuredTests(unittest.TestCase):
    def output(self):
        return DiagnosisOutput(ans=[DiagnosisFormat(candidate_id="candidate_0001", rank=1, disease_name="Example",
            OMIM_id=None, description="Supported by ZeroShot rank 1.")], reference=None)

    def state(self, payload):
        wrapper = Mock()
        wrapper.invoke_with_content_filter_retry.return_value = payload
        return {"llm": wrapper, "hpoDict": {"HP:0001250": "Seizure"},
                "absentHpoDict": {"HP:0000252": "Microcephaly"},
                "mergedDiseaseCandidates": [{"disease_name": "Example", "OMIM_id": None,
                    "tool_rankings": [{"tool": "ZeroShot", "rank": 1}]}]}

    def test_both_templates_and_absent_branches_use_schema(self):
        for gestalt in ([], [{"syndrome_name": "Example"}]):
            for absent in (False, True):
                output = self.output()
                state = self.state({"raw": AIMessage(content="{}"), "parsed": output, "parsing_error": None})
                state.update(GestaltMatcher=gestalt, use_absentHPO=absent)
                with patch("agent.tools.diagnosis.parse_diagnosis_text", side_effect=AssertionError("legacy parser used")):
                    actual, prompt = createDiagnosis(state)
                self.assertIs(actual, output)
                state["llm"].llm.with_structured_output.assert_called_once_with(
                    DiagnosisOutput, method="json_schema", strict=True, include_raw=True)
                self.assertNotIn("===CASE_START===", prompt)
                self.assertNotIn("Do NOT use JSON", prompt)
                self.assertIn("max 2 sentences", prompt)
                self.assertEqual("Microcephaly" in prompt, absent)
                self.assertEqual("Facial image analysis (GestaltMatcher) is not available" in prompt, not bool(gestalt))

    def test_parser_failure_retains_raw_response_in_trace(self):
        state = self.state({"raw": AIMessage(content="broken JSON"), "parsed": None,
                            "parsing_error": ValueError("invalid JSON")})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            with self.assertLogs("agent.tools.diagnosis", level="ERROR"):
                with EvaluationTrace(path):
                    with self.assertRaisesRegex(ValueError, "parsing failed") as caught:
                        createDiagnosis(state)
            events = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(str(caught.exception.__cause__), "invalid JSON")
        failure = next(e for e in events if e.get("event") == "exception")
        self.assertEqual(failure["data"]["llm_response"]["content"], "broken JSON")
        self.assertEqual(failure["data"]["parsing_error"]["message"], "invalid JSON")
        self.assertIn("prompt", failure["data"])
        self.assertEqual(events[-1]["recording_errors"], 0)

    def test_missing_and_refused_outputs_raise(self):
        payloads = [
            {"raw": AIMessage(content="{}"), "parsed": None, "parsing_error": None},
            {"raw": AIMessage(content="", additional_kwargs={"refusal": "refused"}),
             "parsed": None, "parsing_error": None},
        ]
        for payload in payloads:
            with self.assertLogs("agent.tools.diagnosis", level="ERROR"):
                with self.assertRaises(ValueError):
                    createDiagnosis(self.state(payload))

    def test_provider_error_is_not_replaced(self):
        state = self.state(None)
        error = RuntimeError("provider failed")
        state["llm"].invoke_with_content_filter_retry.side_effect = error
        with self.assertLogs("agent.tools.diagnosis", level="ERROR"):
            with self.assertRaises(RuntimeError) as caught:
                createDiagnosis(state)
        self.assertIs(caught.exception, error)

    def test_real_langchain_sdk_request_and_parsed_result_without_network(self):
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
                "model": "gpt-5.2-2025-12-11", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": self.output().model_dump_json()}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})
        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            model = AzureChatOpenAI(azure_endpoint="https://example.test/", api_key="test",
                azure_deployment="test", api_version="2025-04-01-preview", http_client=http)
            state = self.state(None)
            state["llm"] = SimpleNamespace(llm=model,
                invoke_with_content_filter_retry=lambda runnable, messages, **kw: runnable.invoke(messages))
            result, _ = createDiagnosis(state)
        self.assertEqual(result, self.output())
        self.assertEqual(len(requests), 1)
        spec = requests[0]["response_format"]
        self.assertEqual(spec["type"], "json_schema")
        self.assertTrue(spec["json_schema"]["strict"])
        schema = spec["json_schema"]["schema"]
        self.assertEqual(set(schema["required"]), {"ans", "reference"})
        self.assertEqual(set(schema["$defs"]["TentativeDiagnosisCandidate"]["required"]),
                         {"candidate_id", "disease_name", "OMIM_id", "description", "rank"})

    def test_retry_once_retains_empty_final_and_logs_both_attempts(self):
        first = self.output()
        first.ans[0].candidate_id = "unknown"
        last = DiagnosisOutput(ans=[])
        state = self.state(None)
        state["llm"].invoke_with_content_filter_retry.side_effect = [
            {"raw": AIMessage(content=output.model_dump_json()), "parsed": output, "parsing_error": None}
            for output in (first, last)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            with EvaluationTrace(path):
                result, prompt = createDiagnosis(state)
            events = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertIs(result, last)
        self.assertEqual(state["llm"].invoke_with_content_filter_retry.call_count, 2)
        logs = [e["data"] for e in events if e.get("function") == "record_diagnosis_attempt" and e["event"] == "call"]
        self.assertEqual([e["attempt"] for e in logs], [1, 2])
        self.assertEqual(logs[0]["output"]["ans"][0]["candidate_id"], "unknown")
        self.assertEqual(logs[1]["output"]["ans"], [])
        self.assertEqual(logs[1]["validation"]["missing_ids"], ["candidate_0001"])
        self.assertIn(logs[0]["prompt"], prompt)
        self.assertIn(first.model_dump_json(), prompt)
        self.assertEqual(events[-1]["recording_errors"], 0)

    def test_name_and_omim_changes_do_not_trigger_retry(self):
        output = self.output()
        output.ans[0].disease_name = "Changed name"
        output.ans[0].OMIM_id = "999999"
        state = self.state({"raw": AIMessage(content="{}"), "parsed": output, "parsing_error": None})
        result, _ = createDiagnosis(state)
        self.assertIs(result, output)
        self.assertEqual(result.ans[0].OMIM_id, "999999")
        self.assertEqual(state["llm"].invoke_with_content_filter_retry.call_count, 1)

    def test_duplicate_ids_retry_and_recovery(self):
        first = self.output()
        first.ans.append(first.ans[0].model_copy())
        final = self.output()
        state = self.state(None)
        state["llm"].invoke_with_content_filter_retry.side_effect = [
            {"raw": AIMessage(content="{}"), "parsed": output, "parsing_error": None}
            for output in (first, final)]
        result, prompt = createDiagnosis(state)
        self.assertIs(result, final)
        self.assertIn('"duplicate_ids": ["candidate_0001"]', prompt)
        self.assertEqual(state["llm"].invoke_with_content_filter_retry.call_count, 2)

    def test_partial_final_is_returned_without_filling_or_overwriting(self):
        first = DiagnosisOutput(ans=[])
        final = self.output()
        final.ans[0].disease_name = "Different name"
        state = self.state(None)
        state["mergedDiseaseCandidates"].append({"disease_name": "Second", "OMIM_id": "654321"})
        state["llm"].invoke_with_content_filter_retry.side_effect = [
            {"raw": AIMessage(content="{}"), "parsed": output, "parsing_error": None}
            for output in (first, final)]
        result, prompt = createDiagnosis(state)
        self.assertIs(result, final)
        self.assertEqual(len(result.ans), 1)
        self.assertEqual(result.ans[0].disease_name, "Different name")
        self.assertIn('"candidate_id": "candidate_0002"', prompt)
        self.assertEqual(state["llm"].invoke_with_content_filter_retry.call_count, 2)


if __name__ == "__main__":
    unittest.main()
