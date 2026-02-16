from langchain_openai import AzureChatOpenAI

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
        
        # 初期LLMインスタンスを作成
        self.llm = self._create_llm(self.default_max_tokens)
    
    def _create_llm(self, max_completion_tokens):
        """
        指定されたトークン数でLLMインスタンスを作成
        
        Args:
            max_completion_tokens: トークン数の上限
        
        Returns:
            AzureChatOpenAI: 新しいLLMインスタンス
        """
        llm_params = {
            "azure_endpoint": self.azure_endpoint,
            "api_key": self.api_key,
            "deployment_name": self.deployment_name,
            "api_version": self.api_version,
        }
        
        if self.model_name == 'gpt-4o':
            llm_params['temperature'] = 0.0
            llm_params['max_tokens'] = max_completion_tokens
        elif self.model_name in ['gpt-5-1', 'gpt-5-2']:
            llm_params['model_kwargs'] = {
                "extra_body": {
                    "max_completion_tokens": max_completion_tokens,
                    "verbosity": "medium",
                    "reasoning_effort": "none"
                }
            }
        
        return AzureChatOpenAI(**llm_params)
    
    def get_temp_llm_with_max_tokens(self, max_completion_tokens):
        """
        一時的なLLMインスタンスを作成（元のインスタンスは変更しない）
        
        Args:
            max_completion_tokens: トークン数の上限
        
        Returns:
            AzureChatOpenAI: 新しいLLMインスタンス
        """
        return self._create_llm(max_completion_tokens)

    def get_structured_llm(self, output_schema):
        """構造化出力用のLLMを取得"""
        return self.llm.with_structured_output(output_schema)
    
    def generate(self, prompt: str) -> str:
        """通常のテキスト生成用のメソッド"""
        return self.llm.invoke(prompt)