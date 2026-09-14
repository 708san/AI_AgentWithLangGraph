import unittest
from unittest.mock import Mock, patch

import requests

from agent.tools.phenobrain_api import call_phenobrain
from agent.tools.rankingMerge import merge_ranked_disease_candidates


class MockResponse:
    def __init__(self, payload, status_code=200, json_error=None):
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


class TestPhenoBrainApi(unittest.TestCase):
    @patch("agent.tools.phenobrain_api.requests.request")
    def test_success_returns_normalized_results(self, mock_request):
        mock_request.side_effect = [
            MockResponse({"TASK_ID": "task-1"}),
            MockResponse({"state": "SUCCESS", "result": [{"CODE": "RD:8366", "SCORE": 0.998956}]}),
            MockResponse(
                [
                    {
                        "CODE": "RD:8366",
                        "ENG_NAME": "Example disease",
                        "SOURCE_CODES": ["OMIM:123456", "OMIM:654321", "ORPHA:123"],
                    }
                ]
            ),
        ]

        results = call_phenobrain(["HP:0001913"], poll_interval=0, max_poll_seconds=1)

        self.assertEqual(
            results,
            [
                {
                    "disease_name": "Example disease",
                    "omim_id": "OMIM:123456",
                    "orpha_id": "ORPHA:123",
                    "source_codes": ["OMIM:123456", "OMIM:654321", "ORPHA:123"],
                    "rd_id": "RD:8366",
                    "rank": 1,
                    "score": 0.998956,
                }
            ],
        )
        predict_call = mock_request.call_args_list[0]
        self.assertEqual(predict_call.kwargs["params"][0], ("model", "Ensemble"))
        self.assertIn(("hpoList[]", "HP:0001913"), predict_call.kwargs["params"])

    @patch("agent.tools.phenobrain_api.requests.request")
    def test_predict_timeout_returns_empty_list(self, mock_request):
        mock_request.side_effect = requests.Timeout("timed out")

        self.assertEqual(call_phenobrain(["HP:0001913"], max_poll_seconds=1), [])

    @patch("agent.tools.phenobrain_api.requests.request")
    def test_missing_task_id_returns_empty_list(self, mock_request):
        mock_request.return_value = MockResponse({"NO_TASK_ID": "missing"})

        self.assertEqual(call_phenobrain(["HP:0001913"], max_poll_seconds=1), [])

    @patch("agent.tools.phenobrain_api.time.sleep", return_value=None)
    @patch("agent.tools.phenobrain_api.time.monotonic")
    @patch("agent.tools.phenobrain_api.requests.request")
    def test_polling_timeout_returns_empty_list(self, mock_request, mock_monotonic, _mock_sleep):
        mock_request.side_effect = [
            MockResponse({"TASK_ID": "task-1"}),
            MockResponse({"state": "MODEL_PREDICT"}),
            MockResponse({"state": "MODEL_PREDICT"}),
        ]
        mock_monotonic.side_effect = [0, 0, 0.5, 0.5, 1.5]

        self.assertEqual(
            call_phenobrain(["HP:0001913"], poll_interval=0, max_poll_seconds=1),
            [],
        )

    @patch("agent.tools.phenobrain_api.requests.request")
    def test_disease_detail_failure_returns_empty_list(self, mock_request):
        mock_request.side_effect = [
            MockResponse({"TASK_ID": "task-1"}),
            MockResponse({"state": "SUCCESS", "result": [{"CODE": "RD:8366", "SCORE": 0.998956}]}),
            requests.ConnectionError("detail failed"),
        ]

        self.assertEqual(call_phenobrain(["HP:0001913"], max_poll_seconds=1), [])

    @patch("agent.tools.phenobrain_api.requests.request")
    def test_json_parse_error_returns_empty_list(self, mock_request):
        mock_request.return_value = MockResponse({}, json_error=ValueError("bad json"))

        self.assertEqual(call_phenobrain(["HP:0001913"], max_poll_seconds=1), [])

    def test_ranking_merge_combines_phenobrain_by_omim_id(self):
        state = {
            "pubCaseFinder": [
                {
                    "disease_name": "Acroosteolysis dominant type",
                    "omim_id": "OMIM:102500",
                    "rank": 3,
                    "score": 0.7,
                    "description": "PCF result",
                }
            ],
            "zeroShotResult": None,
            "GestaltMatcher": [],
            "phenotypeSearchResult": None,
            "phenoBrain": [
                {
                    "disease_name": "Acroosteolysis dominant type",
                    "omim_id": "OMIM:102500",
                    "orpha_id": "ORPHA:955",
                    "rd_id": "RD:8366",
                    "rank": 1,
                    "score": 0.9989560835133189,
                }
            ],
        }

        merged = merge_ranked_disease_candidates(state)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["OMIM_id"], "OMIM:102500")
        self.assertEqual(merged[0]["consensus_count"], 2)
        self.assertEqual(merged[0]["best_rank"], 1)
        self.assertEqual(
            {ranking["tool"] for ranking in merged[0]["tool_rankings"]},
            {"PubCaseFinder", "PhenoBrain"},
        )
        self.assertIn("RD=RD:8366", merged[0]["tool_rankings"][1]["note"])


if __name__ == "__main__":
    unittest.main()
