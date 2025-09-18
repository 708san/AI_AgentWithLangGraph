from langchain.schema import HumanMessage
from typing_extensions import List, Optional
from ..state.state_types import PCFres, DiagnosisOutput
from ..llm.prompt import prompt_dict, build_prompt
from ..llm.azure_llm_instance import azure_llm

def createDiagnosis(hpo_dict: dict[str,str], pubCaseFinder: List[PCFres], zeroShotResult, gestaltMatcherResult) -> Optional[DiagnosisOutput]:
    top_str = "\n".join(
        [f"{i+1}. {item['omim_disease_name_en']} (score: {item['score']}) - {item['description']}" for i, item in enumerate(pubCaseFinder)]
    )
    zeroShotResult_str = ""
    if zeroShotResult and hasattr(zeroShotResult, "ans"):
        zeroShotResult_str = "\n".join([
            f"{i+1}. {item.disease_name} (rank: {item.rank})"
            for i, item in enumerate(zeroShotResult.ans)
        ])
    # --- GestaltMatcherの結果を整形 ---
    gestaltMatcherResult_str = ""
    if gestaltMatcherResult:
        gestaltMatcherResult_str = "\n".join([
            f"{i+1}. Disease Name: OMIM{item.get('omim_id', '')}:{item.get('syndrome_name', '')}), "
            f"Score: {item.get('gestalt_score', '')}"
            for i, item in enumerate(gestaltMatcherResult)
        ])
    # ---------------------------------

    inputs = {
        "hpo_list": ", ".join([f"{k}:{v}" for k, v in hpo_dict.items()]),
        "pcf_result": top_str,
        "zeroShotResult": zeroShotResult_str,
        "gestaltMatcherResult": gestaltMatcherResult_str  
    }
    prompt_template = prompt_dict["diagnosis_prompt"] 
    structured_llm = azure_llm.get_structured_llm(DiagnosisOutput)

    prompt = build_prompt(prompt_template, inputs)

   
        
    messages = [HumanMessage(content=prompt)]
    return structured_llm.invoke(messages)