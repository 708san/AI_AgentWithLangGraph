import os

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import AzureChatOpenAI
from typing import Any, Dict, Optional


CONTENT_FILTER_MARKERS = [
    "content_filter",
    "content filter",
    "responsibleaipolicyviolation",
    "filtered due to the prompt",
]


def is_content_filter_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in CONTENT_FILTER_MARKERS)


class AzureOpenAIWrapper:
    def __init__(self, model_name, azure_endpoint, api_key, deployment_name, api_version):
        # 設定を保持（再構築時に使用）
        self.model_name = model_name
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version
        
        # デフォルトのトークン数
        self.default_max_tokens = 8192 if model_name == 'gpt-4o' else 15000

        self.rate_limiter = InMemoryRateLimiter(
            requests_per_second=float(os.getenv("AZURE_LLM_REQUESTS_PER_SECOND", "0.25")),
            check_every_n_seconds=float(os.getenv("AZURE_LLM_RATE_CHECK_SECONDS", "0.5")),
            max_bucket_size=float(os.getenv("AZURE_LLM_MAX_BUCKET_SIZE", "1")),
        )
        
        # 初期LLMインスタンスを作成
        self.llm = self._create_llm(self.default_max_tokens)
    
    def _create_llm(
        self,
        max_completion_tokens: int,
        timeout_seconds: Optional[float] = None,
    ) -> AzureChatOpenAI:
        """
        指定されたトークン数でLLMインスタンスを作成
        
        Args:
            max_completion_tokens: トークン数の上限
        
        Returns:
            AzureChatOpenAI: 新しいLLMインスタンス
        """
        llm_params: Dict[str, Any] = {
            "azure_endpoint": self.azure_endpoint,
            "api_key": self.api_key,
            "deployment_name": self.deployment_name,
            "api_version": self.api_version,
            "rate_limiter": self.rate_limiter,
        }
        if timeout_seconds is not None:
            llm_params["timeout"] = timeout_seconds
        
        if self.model_name == 'gpt-4o':
            llm_params['temperature'] = 0.0
            llm_params['max_tokens'] = max_completion_tokens
        elif self.model_name in ['gpt-5-1', 'gpt-5-2']:
            llm_params['extra_body'] = {
                "max_completion_tokens": max_completion_tokens,
                "verbosity": "medium",
                "reasoning_effort": "none"
            }
        
        return AzureChatOpenAI(**llm_params)
    
    def get_temp_llm_with_max_tokens(
        self,
        max_completion_tokens: int,
        timeout_seconds: Optional[float] = None,
    ) -> AzureChatOpenAI:
        """
        一時的なLLMインスタンスを作成（元のインスタンスは変更しない）
        
        Args:
            max_completion_tokens: トークン数の上限
        
        Returns:
            AzureChatOpenAI: 新しいLLMインスタンス
        """
        return self._create_llm(max_completion_tokens, timeout_seconds=timeout_seconds)

    def get_structured_llm(self, output_schema):
        """構造化出力用のLLMを取得"""
        return self.llm.with_structured_output(output_schema)

    def invoke_with_content_filter_retry(
        self,
        runnable: Any,
        input_data: Any,
        context: str = "LLM",
        retry_count: int = 1,
    ):
        """content filter に引っ掛かった場合だけ、同じ呼び出しを指定回数再試行する。"""
        for attempt in range(retry_count + 1):
            try:
                return runnable.invoke(input_data)
            except Exception as e:
                if not is_content_filter_error(e) or attempt >= retry_count:
                    raise
                print(
                    f"[{context}] Content filter triggered. "
                    f"Retrying once ({attempt + 1}/{retry_count})."
                )
    
    def generate(self, prompt: str) -> str:
        """通常のテキスト生成用のメソッド"""
        return self.invoke_with_content_filter_retry(self.llm, prompt, context="Generate")
