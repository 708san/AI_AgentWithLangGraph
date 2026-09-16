import json
import unittest
from copy import deepcopy
from types import SimpleNamespace

import httpx
from openai import OpenAI, LengthFinishReasonError
from pydantic import BaseModel

from scripts.evaluation.usage import UsageCapture, summarize, load_pricing, DEFAULT_PRICING


class UsageTests(unittest.TestCase):
    def completion(self, finish="stop"):
        return {"id": "test-response", "object": "chat.completion", "created": 1,
            "model": "gpt-5.2-2025-12-11",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": '{"answer": "ok"}'}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500,
                      "prompt_tokens_details": {"cached_tokens": 800},
                      "completion_tokens_details": {"reasoning_tokens": 400}}}

    def test_actual_sdk_parsing_request_and_hooks_unchanged(self):
        class Output(BaseModel):
            answer: str
        for finish in ("stop", "length"):
            requests, prior_hooks = [], []
            def respond(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200, json=self.completion(finish))
            http = httpx.Client(transport=httpx.MockTransport(respond),
                                event_hooks={"response": [lambda r: prior_hooks.append(r.status_code)]})
            client = OpenAI(api_key="test-key", http_client=http)
            hooks_before = {k: list(v) for k, v in http.event_hooks.items()}
            record, checkpoints = {}, []
            # The same SDK call without observation has identical behavior and
            # serialized request arguments, including its structured schema.
            try:
                client.chat.completions.parse(model="gpt-5.2", messages=[], response_format=Output)
            except LengthFinishReasonError:
                self.assertEqual(finish, "length")
            with UsageCapture([("zero_shot", client)], record, lambda: checkpoints.append(deepcopy(record))):
                if finish == "length":
                    with self.assertRaises(LengthFinishReasonError):
                        client.chat.completions.parse(model="gpt-5.2", messages=[], response_format=Output)
                else:
                    response = client.chat.completions.parse(model="gpt-5.2", messages=[], response_format=Output)
                    self.assertEqual(response.choices[0].message.parsed.answer, "ok")
            self.assertEqual(http.event_hooks, hooks_before)
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0], requests[1])
            self.assertEqual(requests[0]["model"], "gpt-5.2")
            self.assertEqual(prior_hooks, [200, 200])
            self.assertEqual(record["api_usage"][0]["usage"]["completion_tokens"], 500)
            self.assertNotIn("test-key", json.dumps(record))
            self.assertNotIn('"content"', json.dumps(record))
            self.assertTrue(checkpoints)
            client.close()

    def test_cache_discount_reasoning_not_double_counted_and_embedding(self):
        price = load_pricing(DEFAULT_PRICING)
        rows = [{"stage": "zero_shot", **self.completion()},
                {"stage": "normalization", "model": "text-embedding-3-large",
                 "usage": {"prompt_tokens": 100, "total_tokens": 100}}]
        report = summarize([{"api_usage": rows}], price)
        expected = (200 * 1.75 + 800 * .175 + 500 * 14 + 100 * .158) / 1_000_000
        self.assertAlmostEqual(report["estimated_cost"], expected)
        self.assertEqual(report["stages"]["zero_shot"]["reasoning_tokens"], 400)
        self.assertAlmostEqual(report["known_cost_without_cache_discount"],
                               (1000 * 1.75 + 500 * 14 + 100 * .158) / 1_000_000)

    def test_missing_usage_and_model_mismatch_are_not_free(self):
        price = load_pricing(DEFAULT_PRICING)
        for bad in ({"stage": "zero_shot", "usage": None},
                    {"stage": "zero_shot", **self.completion(), "model": "gpt-5.2-pro"}):
            report = summarize([{"api_usage": [bad]}], price)
            self.assertIsNone(report["estimated_cost"])
            self.assertEqual(report["stages"]["zero_shot"]["unpriced_attempts"], 1)
        self.assertIsNone(summarize([{}], price)["estimated_cost"])

    def test_transport_error_keeps_attempt_and_restores_hooks(self):
        def fail(request):
            raise httpx.ConnectError("mock connection failure", request=request)
        http = httpx.Client(transport=httpx.MockTransport(fail))
        client = SimpleNamespace(_client=http)
        record = {}
        with self.assertRaises(httpx.ConnectError):
            with UsageCapture([("normalization", client)], record, lambda: None):
                http.post("https://example.test/embeddings")
        self.assertEqual(record["api_usage"][0]["status"], "no_response")
        self.assertEqual(http.event_hooks, {"request": [], "response": []})
        http.close()

    def test_sdk_retry_observes_each_attempt(self):
        count = 0
        def respond(request):
            nonlocal count
            count += 1
            if count == 1:
                return httpx.Response(429, headers={"retry-after-ms": "1"},
                                      json={"error": {"message": "rate limited"}})
            return httpx.Response(200, json=self.completion())
        http = httpx.Client(transport=httpx.MockTransport(respond))
        client = OpenAI(api_key="test", http_client=http, max_retries=1)
        record = {}
        with UsageCapture([("zero_shot", client)], record, lambda: None):
            client.chat.completions.create(model="gpt-5.2", messages=[])
        self.assertEqual([r["http_status"] for r in record["api_usage"]], [429, 200])
        report = summarize([record], load_pricing(DEFAULT_PRICING))
        self.assertEqual(report["stages"]["zero_shot"]["output_tokens"], 500)
        self.assertEqual(report["stages"]["zero_shot"]["unpriced_attempts"], 1)
        self.assertIsNone(report["estimated_cost"])
        self.assertGreater(report["known_cost"], 0)
        client.close()


if __name__ == "__main__":
    unittest.main()
