import os

from dotenv import load_dotenv

# カレントディレクトリの.envファイルを読み込む
load_dotenv()

from .llm_wrapper import AzureOpenAIWrapper


MODEL_ENV_PREFIX = {
    "gpt-4o": "AZURE_OPENAI_4o",
    "gpt-5-1": "AZURE_OPENAI_5-1",
    "gpt-5-2": "AZURE_OPENAI_5-2",
}


def _get_model_env_values(model_name: str):
    prefix = MODEL_ENV_PREFIX.get(model_name)
    if not prefix:
        raise ValueError(f"Unsupported model_name: {model_name}")

    endpoint = os.environ.get(f"{prefix}_ENDPOINT")
    api_key = os.environ.get(f"{prefix}_API_KEY")
    deployment_name = os.environ.get(f"{prefix}_DEPLOYMENT_NAME")
    api_version = os.environ.get(f"{prefix}_API_VERSION")
    return endpoint, api_key, deployment_name, api_version

def get_llm_instance(model_name: str = 'gpt-4o'):
    """
    指定されたモデル名に基づいてAzureOpenAIWrapperのインスタンスを生成して返す。
    """
    endpoint, api_key, deployment_name, api_version = _get_model_env_values(model_name)

    if not all([endpoint, api_key, deployment_name, api_version]):
        raise ValueError(f"Environment variables for model '{model_name}' are not fully set.")

    return AzureOpenAIWrapper(
        model_name=model_name,
        azure_endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment_name,
        api_version=api_version
    )

# デフォルトのインスタンス（後方互換性のため、あるいは単体テスト用）
# ただし、新しい設計では直接この変数をインポートしないことが推奨される
azure_llm = get_llm_instance('gpt-4o')