from langchain_openai import AzureChatOpenAI

class AzureOpenAIWrapper:
    def __init__(self, azure_endpoint, api_key, deployment_name, api_version):
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            deployment_name=deployment_name,
            api_version=api_version,
            temperature=0.2,
        )

    def get_structured_llm(self, output_schema):
        return self.llm.with_structured_output(output_schema)
    
    def generate(self, prompt: str) -> str:
        """
        通常のテキスト生成（要約など）用のメソッド
        """
        return self.llm.invoke(prompt)