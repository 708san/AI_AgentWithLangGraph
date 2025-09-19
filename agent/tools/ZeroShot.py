from langchain.schema import HumanMessage
from ..state.state_types import ZeroShotOutput
from ..llm.prompt import prompt_dict, build_prompt
from ..llm.azure_llm_instance import azure_llm


def createZeroshot(hpo_dict):
    if hpo_dict:
        info_type = "HPO terms"
        patient_info = ", ".join([f"{v}" for k, v in hpo_dict.items()])
    else:
        return None, None
    prompt_template = prompt_dict["zero-shot-diagnosis-prompt"]
    inputs = {
        "info_type": info_type,
        "patient_info": patient_info
    }
    structured_llm = azure_llm.get_structured_llm(ZeroShotOutput)
    prompt = build_prompt(prompt_template, inputs)
    messages = [HumanMessage(content=prompt)]
    result = structured_llm.invoke(messages)
    return result, prompt