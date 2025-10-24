"""LLM設定クラス"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """LLM設定を管理するクラス"""
    
    azure_endpoint: str
    api_key: str
    deployment_name: str
    api_version: str
    temperature: float = 0.0
    cache_key: Optional[str] = None
    
    @classmethod
    def from_env(cls, cache_key: Optional[str] = None) -> 'LLMConfig':
        """
        環境変数からLLM設定を作成
        
        Args:
            cache_key: キャッシュキー（デフォルト：None）
        
        Returns:
            LLMConfig インスタンス
        """
        import os
        return cls(
            azure_endpoint=os.getenv("AZURE_OPENAI_4o_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_4o_API_KEY", ""),
            deployment_name=os.getenv("AZURE_OPENAI_4o_DEPLOYMENT_NAME", ""),
            api_version=os.getenv("AZURE_OPENAI_4o_API_VERSION", ""),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            cache_key=cache_key,
        )
    
    @classmethod
    def from_dict(cls, config_dict: dict, cache_key: Optional[str] = None) -> 'LLMConfig':
        """
        辞書からLLM設定を作成
        
        Args:
            config_dict: 設定辞書
            cache_key: キャッシュキー（デフォルト：None）
        
        Returns:
            LLMConfig インスタンス
        """
        return cls(
            azure_endpoint=config_dict.get("azure_endpoint", ""),
            api_key=config_dict.get("api_key", ""),
            deployment_name=config_dict.get("deployment_name", ""),
            api_version=config_dict.get("api_version", ""),
            temperature=config_dict.get("temperature", 0.0),
            cache_key=cache_key,
        )
