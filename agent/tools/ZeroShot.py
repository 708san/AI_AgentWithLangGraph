from langchain.schema import HumanMessage
from ..state.state_types import ZeroShotOutput, State
from ..llm.prompt import prompt_dict, build_prompt


def createZeroshot(state: State):
    """
    hpo_dictとabsent_hpo_dictを使ってZero-Shot診断プロンプトを作成し、LLMに投げる
    """
    hpo_dict = state.get("hpoDict", {})
    absent_hpo_dict = state.get("absentHpoDict", {})
    onset = state.get("onset")
    sex = state.get("sex")
    llm = state.get("llm")

    if not hpo_dict or not llm:
        return None, None

    present_hpo = ", ".join([v for k, v in hpo_dict.items() if v])
    absent_hpo = ", ".join([v for k, v in (absent_hpo_dict or {}).items()]) if absent_hpo_dict else ""

    prompt = build_prompt(
        prompt_dict["zero-shot-diagnosis-prompt"],
        {
            "present_hpo": present_hpo,
            "absent_hpo": absent_hpo,
            "onset": onset if onset else "Unknown",
            "sex": sex if sex else "Unknown"
        }
    )

    # structured_llmを使う場合
    structured_llm = llm.get_structured_llm(ZeroShotOutput)
    messages = [HumanMessage(content=prompt)]
    result = structured_llm.invoke(messages)
    return result, prompt