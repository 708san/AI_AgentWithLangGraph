import json
from typing import List, Optional
from langchain.schema import HumanMessage
from ..state.state_types import State, DiagnosisOutput, ZeroShotOutput, PCFres, PhenotypeSearchFormat
from ..llm.prompt import prompt_dict
from ..llm.azure_llm_instance import azure_llm # get_structured_llmを直接使うためにインポート

def createDiagnosis(state: State) -> Optional[DiagnosisOutput]:
    """
    Integrates multiple information sources (PCF, ZeroShot, GestaltMatcher, PhenotypeSearch) 
    to generate a tentative diagnosis.
    """
    hpo_list = state.get("hpo_list", [])
    absent_hpo_list = list(state.get("absentHpoDict", {}).values())
    onset = state.get("onset", "Unknown")
    sex = state.get("sex", "Unknown")
    pcf_results = state.get("pubCaseFinder", [])
    zeroshot_results = state.get("zeroShotResult")
    gestalt_matcher_results = state.get("GestaltMatcher", [])
    web_search_results = state.get("webresources", [])
    phenotype_search_results = state.get("phenotypeSearchResult", [])

    # --- Format each result into a string for the prompt ---
    
    # PCF results
    pcf_text = "\n".join([f"{i+1}. {res.get('disease_name', res.get('omim_disease_name_en', 'N/A'))} (score: {res.get('score', 0):.3f}) - {res.get('description', '')}" for i, res in enumerate(pcf_results)]) if pcf_results else "No results from PubCaseFinder."

    # Zero-shot results
    zeroshot_text = "\n".join([f"{i+1}. {res.disease_name} (rank: {res.rank})" for i, res in enumerate(zeroshot_results.ans)]) if zeroshot_results and zeroshot_results.ans else "No results from Zero-Shot Diagnosis."

    # GestaltMatcher results
    gm_text = "\n".join([f"{i+1}. {res.get('syndrome_name', 'N/A')}) (Similarity score: {res.get('score', 0):.3f})" for i, res in enumerate(gestalt_matcher_results)]) if gestalt_matcher_results else "No results from GestaltMatcher."

    # Web search results
    web_text = "\n".join([f"- {res.get('title', 'No Title')}: {res.get('content', 'No Content')}" for res in web_search_results]) if web_search_results else "No relevant web search results found."
    # Phenotype search results
    phenotype_search_text = "\n".join([f"{i+1}. {res.disease_info.disease_name} (OMIM: {res.disease_info.OMIM_id}, Similarity score: {res.similarity_score:.3f})" for i, res in enumerate(phenotype_search_results)]) if phenotype_search_results else "No results from Phenotype Similarity Search."

    # Assemble the prompt
    prompt = prompt_dict["diagnosis_prompt"].format(
        hpo_list=", ".join(hpo_list),
        absent_hpo_list=", ".join(absent_hpo_list),
        onset=onset,
        sex=sex,
        pcf_results=pcf_text,
        zeroshot_results=zeroshot_text,
        gestalt_matcher_results=gm_text,
        phenotype_search_results=phenotype_search_text,
        web_search_results=web_text
    )

    # --- Query the LLM to get the diagnosis result ---
    # Get a structured LLM instance that outputs in the DiagnosisOutput format
    structured_llm = azure_llm.get_structured_llm(DiagnosisOutput)

    # Create the message payload for the LLM
    messages = [HumanMessage(content=prompt)]
    
    # Invoke the LLM and get the structured result
    diagnosis_json = structured_llm.invoke(messages)
    
    if diagnosis_json:
        return (diagnosis_json, prompt)
    
    return None