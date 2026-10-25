import unittest
from unittest.mock import patch

from agent.utils.result_saver import save_result
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


if __name__ == "__main__":
    unittest.main()
