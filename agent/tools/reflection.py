from langchain.schema import HumanMessage
from state.state_types import ReflectionFormat
from llm.prompt import prompt_dict, build_prompt
from llm.azure_llm_instance import azure_llm

def create_reflection(patient_info, diagnosis_to_judge):#, similar_case_detailed, disease_knowledge):
    prompt_template = prompt_dict["reflection_prompt"]
    diagnosis_name = diagnosis_to_judge.disease_name
    description = diagnosis_to_judge.description
    rank = diagnosis_to_judge.rank
    inputs = {
        "patient_info": patient_info,
        "diagnosis_to_judge": f"{diagnosis_name} (Rank: {rank})\nDescription: {description}",
        #"similar_case_detailed": similar_case_detailed,
        #"disease_knowledge": disease_knowledge
    }
    structured_llm = azure_llm.get_structured_llm(ReflectionFormat)
    prompt = build_prompt(prompt_template, inputs)
    messages = [HumanMessage(content=prompt)]
    result = structured_llm.invoke(messages)
    if isinstance(result, dict):
        return ReflectionFormat(**result)
    return result