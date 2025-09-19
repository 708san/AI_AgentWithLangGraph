from langchain.schema import HumanMessage
from ..state.state_types import ReflectionFormat
from ..llm.prompt import prompt_dict, build_prompt
from ..llm.azure_llm_instance import azure_llm


def format_disease_knowledge(info_list, disease_name):
    """
    InformationItemのリストから、rankに該当するものだけをプロンプト用に整形
    """
    if not info_list:
        return "No disease knowledge available."
    lines = []
    for i, item in enumerate(info_list, 1):
        if item.get("disease_name") == disease_name:
            line = f"[{i}] {item.get('title', '')}\nURL: {item.get('url', '')}\n{item.get('content', '')}\n"
            lines.append(line)
    if not lines:
        return "No disease knowledge available for this rank."
    return "\n".join(lines)

def create_reflection(hpo_dict, diagnosis_to_judge, disease_knowledge_list, absent_hpo_dict=None):
    prompt_template = prompt_dict["reflection_prompt"]
    diagnosis_name = diagnosis_to_judge.disease_name
    description = diagnosis_to_judge.description
    rank = diagnosis_to_judge.rank
    disease_name = diagnosis_to_judge.disease_name

    disease_knowledge_str = format_disease_knowledge(disease_knowledge_list, disease_name) if disease_knowledge_list is not None else ""

    present_hpo = ", ".join([v for k, v in hpo_dict.items()]) if hpo_dict else ""
    absent_hpo = ", ".join([v for k, v in (absent_hpo_dict or {}).items()]) if absent_hpo_dict else ""

    inputs = {
        "present_hpo": present_hpo,
        "absent_hpo": absent_hpo,
        "diagnosis_to_judge": f"{diagnosis_name} (Rank: {rank})\nDescription: {description}",
        "disease_knowledge": disease_knowledge_str
    }
    structured_llm = azure_llm.get_structured_llm(ReflectionFormat)
    prompt = build_prompt(prompt_template, inputs)
    
    print("reflection prompt\n")
    print(prompt)
    print("\n")
    
    messages = [HumanMessage(content=prompt)]
    result = structured_llm.invoke(messages)
    if isinstance(result, dict):
        return ReflectionFormat(**result), prompt
    return result, prompt