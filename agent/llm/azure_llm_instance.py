import os
from .llm_wrapper import AzureOpenAIWrapper

azure_llm = AzureOpenAIWrapper(
    azure_endpoint=os.environ["AZURE_OPENAI_4o_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_4o_API_KEY"],
    deployment_name=os.environ["AZURE_OPENAI_4o_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_4o_API_VERSION"]
)