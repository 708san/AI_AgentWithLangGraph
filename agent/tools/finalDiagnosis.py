from langchain.schema import HumanMessage
from typing_extensions import Optional
from state.state_types import State, DiagnosisOutput
from llm.prompt import prompt_dict, build_prompt
from llm.azure_llm_instance import azure_llm

def createFinalDiagnosis(state: State) -> Optional[DiagnosisOutput]:
    """
    Generate FinalDiagnosis using State, prompt, and DiagnosisOutput
    """
   
    patient_info = state.get("clinicalText", "")
    similar_case_detailed = state.get("memory", "")
    tentative_result = state.get("tentativeDiagnosis", None)
    judgements = state.get("reflection", None)

    # Format tentativeDiagnosis (DiagnosisOutput type or None) as string, including reference
    if tentative_result is not None and hasattr(tentative_result, "ans"):
        tentative_result_str = "\n".join([
            f"{i+1}. {item.disease_name} (Rank: {item.rank})\nDescription: {item.description}\nReference: {getattr(item, 'reference', '')}"
            for i, item in enumerate(tentative_result.ans)
        ])
        
        if hasattr(tentative_result, "reference") and tentative_result.reference:
            tentative_result_str += f"\n[DiagnosisOutput Reference]: {tentative_result.reference}"
    else:
        tentative_result_str = ""

    # Format reflection (ReflectionOutput type or None) as string, including reference
    if judgements is not None and hasattr(judgements, "ans"):
        judgements_str = "\n".join([
            f"{i+1}. {getattr(item, 'disease_name', '')}\nCorrectness: {getattr(item, 'Correctness', '')}\nPatientSummary: {getattr(item, 'PatientSummary', '')}\nDiagnosisAnalysis: {getattr(item, 'DiagnosisAnalysis', '')}\nReference: {getattr(item, 'reference', '')}"
            for i, item in enumerate(judgements.ans)
        ])
       
        if hasattr(judgements, "reference") and judgements.reference:
            judgements_str += f"\n[ReflectionOutput Reference]: {judgements.reference}"
    else:
        judgements_str = ""

    # If memory (similar_case_detailed) is a list, join as string
    if isinstance(similar_case_detailed, list):
        similar_case_detailed_str = "\n".join([str(item) for item in similar_case_detailed])
    else:
        similar_case_detailed_str = str(similar_case_detailed)

    prompt_template = prompt_dict["final_diagnosis_prompt"]
    inputs = {
        "patient_info": patient_info,
        "similar_case_detailed": similar_case_detailed_str,
        "tentative_result": tentative_result_str,
        "judgements": judgements_str
    }
    prompt = build_prompt(prompt_template, inputs)
    messages = [HumanMessage(content=prompt)]
    structured_llm = azure_llm.get_structured_llm(DiagnosisOutput)
    return structured_llm.invoke(messages)