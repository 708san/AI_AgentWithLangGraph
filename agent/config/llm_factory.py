"""LLMインスタンスファクトリークラス"""
from typing import Optional, Dict, Any
from .llm_config import LLMConfig


class LLMFactory:
    """LLMインスタンスを一元管理するシングルトンファクトリークラス"""
    
    _instance: Optional['LLMFactory'] = None
    _llm_cache: Dict[str, Any] = {}
    
    def __new__(cls):
        """シングルトンパターンの実装"""
        if cls._instance is None:
            cls._instance = super(LLMFactory, cls).__new__(cls)
        return cls._instance
    
    def create_azure_openai(
        self,
        llm_config: LLMConfig,
    ) -> 'AzureOpenAIWrapper':
        """
        Azure OpenAI LLMを作成または取得
        
        Args:
            llm_config: LLM設定
        
        Returns:
            AzureOpenAIWrapper インスタンス
        """
        # キャッシュキーがある場合はキャッシュから取得を試みる
        if llm_config.cache_key and llm_config.cache_key in self._llm_cache:
            return self._llm_cache[llm_config.cache_key]
        
        from agent.llm.llm_wrapper import AzureOpenAIWrapper
        
        wrapper = AzureOpenAIWrapper(
            azure_endpoint=llm_config.azure_endpoint,
            api_key=llm_config.api_key,
            deployment_name=llm_config.deployment_name,
            api_version=llm_config.api_version,
            temperature=llm_config.temperature,
        )
        
        # キャッシュに保存
        if llm_config.cache_key:
            self._llm_cache[llm_config.cache_key] = wrapper
        
        return wrapper
    
    def get_cached_llm(self, cache_key: str) -> Optional['AzureOpenAIWrapper']:
        """
        キャッシュからLLMを取得
        
        Args:
            cache_key: キャッシュキー
        
        Returns:
            キャッシュされたLLMインスタンス、存在しない場合はNone
        """
        return self._llm_cache.get(cache_key)
    
    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._llm_cache.clear()
