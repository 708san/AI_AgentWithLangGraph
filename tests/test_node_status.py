import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.utils.result_saver import save_result


class NodeStatusTests(unittest.TestCase):
    def test_decorated_node_records_completed_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENT_RESULT_DIR": tmp}, clear=False):
                @save_result("exampleNode")
                def example_node(_state):
                    return {"value": 1}

                example_node({})
            payload = json.loads((Path(tmp) / "node_status.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["nodes"]["exampleNode"]["status"], "completed")
            self.assertIsNotNone(payload["nodes"]["exampleNode"]["started_at"])
            self.assertIsNotNone(payload["nodes"]["exampleNode"]["finished_at"])

    def test_decorated_node_records_failure_and_reraises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENT_RESULT_DIR": tmp}, clear=False):
                @save_result("failingNode")
                def failing_node(_state):
                    raise RuntimeError("expected failure")

                with self.assertRaisesRegex(RuntimeError, "expected failure"):
                    failing_node({})
            payload = json.loads((Path(tmp) / "node_status.json").read_text(encoding="utf-8"))
            node = payload["nodes"]["failingNode"]
            self.assertEqual(node["status"], "failed")
            self.assertEqual(node["error_type"], "RuntimeError")
            self.assertIn("expected failure", node["error_message"])


if __name__ == "__main__":
    unittest.main()
