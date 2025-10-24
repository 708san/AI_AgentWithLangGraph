"""Azure OpenAI LLMラッパークラス"""
from langchain_openai import AzureChatOpenAI
from typing import Any


class AzureOpenAIWrapper:
    """Azure OpenAI LLMのラッパークラス"""
    
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        deployment_name: str,
        api_version: str,
        temperature: float = 0.0,
    ):
        """
        初期化
        
        Args:
            azure_endpoint: Azure エンドポイント
            api_key: APIキー
            deployment_name: デプロイ名
            api_version: APIバージョン
            temperature: 温度パラメータ（デフォルト：0.0）
        """
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            deployment_name=deployment_name,
            api_version=api_version,
            temperature=temperature,
        )
        self._temperature = temperature
        self._deployment_name = deployment_name

    def get_structured_llm(self, output_schema: Any) -> Any:
        """
        構造化出力用のLLMを取得
        
        Args:
            output_schema: 出力スキーマ
        
        Returns:
            構造化出力対応のLLM
        """
        return self.llm.with_structured_output(output_schema)
    
    def generate(self, prompt: str) -> str:
        """
        通常のテキスト生成（要約など）用のメソッド
        
        Args:
            prompt: 入力プロンプト
        
        Returns:
            生成されたテキスト
        """
        return self.llm.invoke(prompt)
    
    def set_temperature(self, temperature: float) -> None:
        """
        温度パラメータを動的に変更
        
        Args:
            temperature: 新しい温度パラメータ
        """
        self.llm.temperature = temperature
        self._temperature = temperature
    
    def get_temperature(self) -> float:
        """
        現在の温度パラメータを取得
        
        Returns:
            現在の温度パラメータ
        """
        return self._temperature
    
    def get_deployment_name(self) -> str:
        """
        デプロイ名を取得
        
        Returns:
            デプロイ名
        """
        return self._deployment_name