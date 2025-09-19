from langchain.schema import HumanMessage
from ..state.state_types import ZeroShotOutput
from ..llm.prompt import prompt_dict, build_prompt
from ..llm.azure_llm_instance import azure_llm


def createZeroshot(hpo_dict, absent_hpo_dict=None):
    """
    hpo_dictとabsent_hpo_dictを使ってZero-Shot診断プロンプトを作成し、LLMに投げる
    """
    if not hpo_dict:
        return None, None

    present_hpo = ", ".join([v for k, v in hpo_dict.items() if v])
    absent_hpo = ", ".join([v for k, v in (absent_hpo_dict or {}).items()]) if absent_hpo_dict else ""

    prompt = build_prompt(
        prompt_dict["zero-shot-diagnosis-prompt"],
        {
            "present_hpo": present_hpo,
            "absent_hpo": absent_hpo
        }
    )

    # structured_llmを使う場合
    structured_llm = azure_llm.get_structured_llm(ZeroShotOutput)
    messages = [HumanMessage(content=prompt)]
    result = structured_llm.invoke(messages)
    return result, prompt