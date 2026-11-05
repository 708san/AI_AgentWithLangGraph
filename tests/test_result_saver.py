import unittest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from agent.utils.result_saver import save_node_result, save_result
from agent.utils.profiler import profile_node


class ResultSaverTests(unittest.TestCase):
    def test_decorated_node_marks_itself_as_persisting(self):
        @profile_node
        @save_result("exampleNode")
        def node(state):
            return {"value": state["value"]}

        self.assertTrue(node._saves_result)
        with patch("agent.utils.result_saver.save_node_result") as save:
            result = node({"value": 1})

        self.assertEqual(result, {"value": 1})
        save.assert_called_once_with("exampleNode", result, {"value": 1})

    def test_node_and_phase_audit_artifacts_are_written_incrementally(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "results"
            node_dir = Path(tmp) / "node_results"
            state = {
                "patient_id": "patient/42",
                "evidence_mode": "v2",
                "sources": [],
                "evidence": [],
                "pubCaseFinder": [],
            }
            with patch.dict(
                os.environ,
                {
                    "AGENT_RESULT_DIR": str(result_dir),
                    "AGENT_NODE_RESULT_DIR": str(node_dir),
                },
                clear=False,
            ):
                save_node_result("PCFnode", {"pubCaseFinder": [{"rank": 1}]}, state)

            phase_results = json.loads((result_dir / "phase_results.json").read_text())
            self.assertEqual(phase_results["status"], "in_progress")
            self.assertEqual(phase_results["completed_nodes"], ["PCFnode"])
            self.assertEqual(
                phase_results["tool_results"]["pubCaseFinder"],
                [{"rank": 1}],
            )
            self.assertTrue((node_dir / "patient_42" / "PCFnode.json").is_file())
            self.assertTrue((node_dir / "patient_42.json").is_file())


if __name__ == "__main__":
    unittest.main()
