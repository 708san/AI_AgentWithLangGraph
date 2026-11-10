import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from agent.llm.azure_llm_instance import get_llm_instance
from agent.llm.llm_wrapper import AzureOpenAIWrapper


class TestAzureLlmInstance(unittest.TestCase):
    def test_import_does_not_require_gpt4o_environment(self):
        env = os.environ.copy()
        for suffix in ("ENDPOINT", "API_KEY", "DEPLOYMENT_NAME", "API_VERSION"):
            env[f"AZURE_OPENAI_4o_{suffix}"] = ""

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from agent.llm.azure_llm_instance import get_llm_instance\n"
                    "print('IMPORT_OK')"
                ),
            ],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IMPORT_OK", result.stdout)

    def test_explicit_gpt4o_call_requires_gpt4o_environment(self):
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_4o_ENDPOINT": "",
                "AZURE_OPENAI_4o_API_KEY": "",
                "AZURE_OPENAI_4o_DEPLOYMENT_NAME": "",
                "AZURE_OPENAI_4o_API_VERSION": "",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Environment variables for model 'gpt-4o' are not fully set",
            ):
                get_llm_instance("gpt-4o")

    @patch("agent.llm.azure_llm_instance.AzureOpenAIWrapper")
    def test_gpt52_uses_only_gpt52_environment(self, mock_wrapper):
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_4o_ENDPOINT": "",
                "AZURE_OPENAI_4o_API_KEY": "",
                "AZURE_OPENAI_4o_DEPLOYMENT_NAME": "",
                "AZURE_OPENAI_4o_API_VERSION": "",
                "AZURE_OPENAI_5-2_ENDPOINT": "https://example.invalid",
                "AZURE_OPENAI_5-2_API_KEY": "test-key",
                "AZURE_OPENAI_5-2_DEPLOYMENT_NAME": "test-deployment",
                "AZURE_OPENAI_5-2_API_VERSION": "2025-01-01",
            },
            clear=False,
        ):
            get_llm_instance("gpt-5-2")

        mock_wrapper.assert_called_once_with(
            model_name="gpt-5-2",
            azure_endpoint="https://example.invalid",
            api_key="test-key",
            deployment_name="test-deployment",
            api_version="2025-01-01",
            reasoning_effort="medium",
        )

    @patch("agent.llm.azure_llm_instance.AzureOpenAIWrapper")
    def test_gpt54_uses_gpt54_environment_and_requested_effort(self, mock_wrapper):
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_5-4_ENDPOINT": "https://example.invalid",
                "AZURE_OPENAI_5-4_API_KEY": "test-key",
                "AZURE_OPENAI_5-4_DEPLOYMENT_NAME": "test-deployment",
                "AZURE_OPENAI_5-4_API_VERSION": "2025-01-01",
            },
            clear=False,
        ):
            get_llm_instance("gpt-5-4", reasoning_effort="medium")

        mock_wrapper.assert_called_once_with(
            model_name="gpt-5-4",
            azure_endpoint="https://example.invalid",
            api_key="test-key",
            deployment_name="test-deployment",
            api_version="2025-01-01",
            reasoning_effort="medium",
        )

    def test_gpt54_rejects_unsupported_reasoning_effort(self):
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_5-4_ENDPOINT": "https://example.invalid",
                "AZURE_OPENAI_5-4_API_KEY": "test-key",
                "AZURE_OPENAI_5-4_DEPLOYMENT_NAME": "test-deployment",
                "AZURE_OPENAI_5-4_API_VERSION": "2025-01-01",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "Unsupported reasoning_effort"):
                get_llm_instance("gpt-5-4", reasoning_effort="invalid")

    @patch("agent.llm.llm_wrapper.AzureChatOpenAI")
    def test_gpt54_uses_responses_api_reasoning_payload(self, mock_chat_openai):
        AzureOpenAIWrapper(
            model_name="gpt-5-4",
            azure_endpoint="https://example.invalid",
            api_key="test-key",
            deployment_name="test-deployment",
            api_version="2025-01-01",
            reasoning_effort="medium",
        )

        params = mock_chat_openai.call_args.kwargs
        self.assertTrue(params["use_responses_api"])
        self.assertEqual(params["output_version"], "responses/v1")
        self.assertEqual(params["reasoning"], {"effort": "medium"})
        self.assertEqual(params["verbosity"], "medium")
        self.assertEqual(params["max_tokens"], 40000)


if __name__ == "__main__":
    unittest.main()
