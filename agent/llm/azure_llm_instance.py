import os

from dotenv import load_dotenv

# カレントディレクトリの.envファイルを読み込む
load_dotenv()

from .llm_wrapper import AzureOpenAIWrapper

def get_llm_instance(model_name: str = 'gpt-4o'):
    """
    指定されたモデル名に基づいてAzureOpenAIWrapperのインスタンスを生成して返す。
    """
    if model_name == 'gpt-4o':
        endpoint = os.environ.get("AZURE_OPENAI_4o_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_4o_API_KEY")
        deployment_name = os.environ.get("AZURE_OPENAI_4o_DEPLOYMENT_NAME")
        api_version = os.environ.get("AZURE_OPENAI_4o_API_VERSION")
    elif model_name == 'gpt-5':
        # GPT-5用の環境変数を設定（.envファイルに追記が必要）
        endpoint = os.environ.get("AZURE_OPENAI_5_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_5_API_KEY")
        deployment_name = os.environ.get("AZURE_OPENAI_5_DEPLOYMENT_NAME")
        api_version = os.environ.get("AZURE_OPENAI_5_API_VERSION")
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

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