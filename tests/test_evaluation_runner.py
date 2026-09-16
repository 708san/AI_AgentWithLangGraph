import json
import tempfile
import unittest
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from scripts.evaluation import runner


class RunnerTests(unittest.TestCase):
    def fixture(self, root):
        benchmark = root / "cases.json"
        benchmark.write_text(json.dumps({"cases": [{"case_id": "a", "patient_id": "1",
            "input": {"present_hpo_list": ["HP:0001250"]},
            "expected_output": {"omim_id": "123456"}}]}))
        return benchmark

    def test_dry_run_never_initializes_providers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            for mode, factory in [("zero-shot", "zero_shot_executor"), ("tentative", "tentative_executor")]:
                with patch.object(runner, factory) as mocked:
                    self.assertEqual(runner.main(mode, ["--benchmark", str(path), "--dry-run"]), 0)
                    mocked.assert_not_called()

    def test_repeats_save_and_score_without_label_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.fixture(root)
            output = root / "run"
            calls = []
            def execute(inputs, record, checkpoint):
                self.assertNotIn("expected_output", inputs)
                self.assertNotIn("expected_output", record)
                calls.append(inputs)
                record["zeroShotRaw"] = {"ans": [{"rank": len(calls), "OMIM_id": "123456"}]}
                record["zeroShotResult"] = {"ans": [{"rank": len(calls), "OMIM_id": "OMIM:123456"}]}
                record["prompts"] = {"zeroShot": "test prompt"}
                record["api_usage"] = [{"stage": "zero_shot", "model": "gpt-5.2",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                              "prompt_tokens_details": {"cached_tokens": 0},
                              "completion_tokens_details": {"reasoning_tokens": 5}}}]
                checkpoint()
            with patch.object(runner, "zero_shot_executor", return_value=execute):
                self.assertEqual(runner.main("zero-shot", ["--benchmark", str(path), "--output-dir", str(output), "--repeats", "2"]), 0)
            self.assertEqual(len(calls), 2)
            reports = [json.loads((output / f"repeat_{n:03d}/evaluation.json").read_text()) for n in (1, 2)]
            self.assertEqual(reports[0]["summary"]["zeroShotRaw"]["top_k"]["1"]["hits"], 1)
            self.assertEqual(reports[1]["summary"]["zeroShotRaw"]["top_k"]["1"]["hits"], 0)
            self.assertEqual(reports[0]["summary"]["zeroShotResult"]["top_k"]["1"]["hits"], 1)
            costs = json.loads((output / "cost_summary.json").read_text())
            first_cost = json.loads((output / "repeat_001/cost_summary.json").read_text())
            self.assertEqual(costs["records"], 2)
            self.assertEqual(costs["stages"]["zero_shot"]["reasoning_tokens"], 10)
            self.assertAlmostEqual(costs["estimated_cost"], first_cost["estimated_cost"] * 2)

    def test_zero_shot_normalization_preserves_raw_even_on_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                raw = {"ans": [{"rank": 1, "disease_name": "Original", "OMIM_id": None}]}
                def normalize(state):
                    # Simulate production's in-place mutation, then optional error.
                    result = state["zeroShotResult"]
                    result["ans"][0].update(disease_name="Normalized", OMIM_id="OMIM:123456")
                    if fail:
                        raise RuntimeError("normalization failed")
                    return result
                modules = {
                    "agent.llm.azure_llm_instance": SimpleNamespace(get_llm_instance=lambda model: SimpleNamespace(
                        deployment_name="test", api_version="test",
                        llm=SimpleNamespace(extra_body={"reasoning_effort": "none"}, max_tokens=None,
                            root_client=SimpleNamespace(_client=SimpleNamespace(event_hooks={}))))),
                    "agent.tools.ZeroShot": SimpleNamespace(createZeroshot=lambda state: (raw, "prompt")),
                    "agent.tools.make_HPOdic": SimpleNamespace(make_hpo_dic=lambda ids, _: dict.fromkeys(ids, "label")),
                    "agent.tools.diseaseNormalize": SimpleNamespace(normalize_zeroshot_results=normalize,
                        client=SimpleNamespace(_client=SimpleNamespace(event_hooks={}))),
                }
                record, checkpoints = {}, []
                with patch.dict("sys.modules", modules):
                    execute = runner.zero_shot_executor("gpt-5-2")
                    inputs = dict(hpo_list=["HP:0001250"], absent_hpo_list=[], onset=None,
                                  sex=None, use_absentHPO=False)
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "normalization failed"):
                            execute(inputs, record, lambda: checkpoints.append(deepcopy(record)))
                    else:
                        execute(inputs, record, lambda: checkpoints.append(deepcopy(record)))
                self.assertIsNone(record["zeroShotRaw"]["ans"][0]["OMIM_id"])
                self.assertEqual(record["zeroShotRaw"]["ans"][0]["disease_name"], "Original")
                self.assertNotIn("zeroShotResult", checkpoints[0])
                self.assertEqual(len(checkpoints), 2 if fail else 3)  # usage exit checkpoint
                if fail:
                    self.assertNotIn("zeroShotResult", record)
                else:
                    self.assertEqual(record["zeroShotResult"]["ans"][0]["OMIM_id"], "OMIM:123456")

    def test_partial_output_survives_case_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.fixture(root)
            def execute(inputs, record, checkpoint):
                record["zeroShotRaw"] = {"ans": []}
                checkpoint()
                raise RuntimeError("mock failure")
            with patch.object(runner, "tentative_executor", return_value=execute):
                code = runner.main("tentative", ["--benchmark", str(path), "--output-dir", str(root / "out")])
            self.assertEqual(code, 1)
            record = json.loads((root / "out/repeat_001/predictions/a.json").read_text())
            self.assertEqual(record["zeroShotRaw"], {"ans": []})
            self.assertEqual(record["status"], "error")

    def test_packaged_image_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "images").mkdir()
            image = root / "images/a.jpg"
            image.write_bytes(b"test")
            self.assertEqual(runner.resolve_image("images/a.jpg", root / "data/cases.json"), str(image.resolve()))


if __name__ == "__main__":
    unittest.main()
