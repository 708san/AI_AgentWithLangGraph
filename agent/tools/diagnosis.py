import json
from typing import List, Optional
import re
from langchain.schema import HumanMessage
from ..state.state_types import State, DiagnosisOutput,DiagnosisFormat, ZeroShotOutput, PCFres, PhenotypeSearchFormat
from ..llm.prompt import prompt_dict

def parse_diagnosis_text(text: str) -> DiagnosisOutput:
    """
    LLMのテキスト出力をパースしてDiagnosisOutputオブジェクトに変換する。
    """
    cases = []
    # Extract cases
    case_blocks = re.findall(r"===CASE_START===(.*?)===CASE_END===", text, re.DOTALL)
    
    for block in case_blocks:
        rank_match = re.search(r"RANK::(\d+)", block)
        disease_match = re.search(r"DISEASE::(.*)", block)
        omim_match = re.search(r"OMIM::(.*)", block)
        desc_match = re.search(r"DESCRIPTION::(.*)", block, re.DOTALL)
        
        if rank_match and disease_match and desc_match:
            rank = int(rank_match.group(1).strip())
            disease = disease_match.group(1).strip()
            omim = omim_match.group(1).strip() if omim_match else None
            desc = desc_match.group(1).strip()
            
            # Clean up OMIM if it's "None" or empty or "N/A"
            if omim and (omim.lower() == "none" or omim.lower() == "n/a" or not omim):
                omim = None
                
            cases.append(DiagnosisFormat(
                rank=rank,
                disease_name=disease,
                OMIM_id=omim,
                description=desc
            ))
            
    # Extract references
    ref_match = re.search(r"===REFERENCES_START===(.*?)===REFERENCES_END===", text, re.DOTALL)
    references = ref_match.group(1).strip() if ref_match else None
    
    return DiagnosisOutput(ans=cases, reference=references)

def createDiagnosis(state: State) -> Optional[DiagnosisOutput]:
    """
    Integrates multiple information sources (PCF, ZeroShot, GestaltMatcher, PhenotypeSearch) 
    to generate a tentative diagnosis.
    """
    hpo_list = list(state.get("hpoDict", {}).values())
    absent_hpo_list = list(state.get("absentHpoDict", {}).values())
    onset = state.get("onset", "Unknown")
    sex = state.get("sex", "Unknown")
    pcf_results = state.get("pubCaseFinder", [])
    zeroshot_results = state.get("zeroShotResult")
    gestalt_matcher_results = state.get("GestaltMatcher", [])
    web_search_results = state.get("webresources", [])
    phenotype_search_results = state.get("phenotypeSearchResult", [])
    llm = state.get("llm")

    if not llm:
        print("LLM instance not found in state.")
        return None, None

    # --- Format each result into a string for the prompt ---
    
    # PCF results
    pcf_text = "\n".join([f"{i+1}. {res.get('disease_name', res.get('omim_disease_name_en', 'N/A'))} (score: {res.get('score', 0):.3f}) - {res.get('description', '')}" for i, res in enumerate(pcf_results)]) if pcf_results else "No results from PubCaseFinder."

    # Zero-shot results
    zeroshot_text = "\n".join([f"{i+1}. {res.disease_name} (rank: {res.rank})" for i, res in enumerate(zeroshot_results.ans)]) if zeroshot_results and zeroshot_results.ans else "No results from Zero-Shot Diagnosis."

    # GestaltMatcher results - デバッグ出力追加
    
    has_gestalt = gestalt_matcher_results and len(gestalt_matcher_results) > 0
    
    
    if has_gestalt:
        gm_text = "\n".join([f"{i+1}. {res.get('syndrome_name', 'N/A')} (Similarity score: {res.get('score', 0):.3f})" for i, res in enumerate(gestalt_matcher_results)])
        print(f"[DEBUG] 使用するプロンプト: diagnosis_prompt (GestaltMatcher有り)")
    else:
        gm_text = None
        print(f"[DEBUG] 使用するプロンプト: diagnosis_prompt_no_gestalt (GestaltMatcher無し)")

    # Web search results
    web_text = "\n".join([f"- {res.get('title', 'No Title')}: {res.get('content', 'No Content')}" for res in web_search_results]) if web_search_results else "No relevant web search results found."
    
    # Phenotype search results
    phenotype_lines = []
    if phenotype_search_results:
        for i, res in enumerate(phenotype_search_results):
            disease_info = res.disease_info
            definition_text = disease_info.definition or "not provided"
            phenotype_list_text = ", ".join(disease_info.phenotype) if disease_info.phenotype else "not provided"
            
            line = (
                f"{i+1}. {disease_info.disease_name} (OMIM: {disease_info.OMIM_id}, Similarity score: {res.similarity_score:.3f})\n"
                f"   - Definition: {definition_text}\n"
                f"   - Typical Phenotypes: {phenotype_list_text}"
            )
            phenotype_lines.append(line)
        phenotype_search_text = "\n".join(phenotype_lines)
    else:
        phenotype_search_text = "No results from Phenotype Similarity Search."

    # GestaltMatcherの結果があるかどうかで異なるプロンプトを使用
    if has_gestalt:
        prompt_template = prompt_dict["diagnosis_prompt"]
        prompt = prompt_template.format(
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
    else:
        # GestaltMatcher情報がない場合のプロンプト
        prompt_template = prompt_dict["diagnosis_prompt_no_gestalt"]
        prompt = prompt_template.format(
            hpo_list=", ".join(hpo_list),
            absent_hpo_list=", ".join(absent_hpo_list),
            onset=onset,
            sex=sex,
            pcf_results=pcf_text,
            zeroshot_results=zeroshot_text,
            phenotype_search_results=phenotype_search_text,
            web_search_results=web_text
        )

    # --- Query the LLM to get the diagnosis result ---
    messages = [HumanMessage(content=prompt)]
    
    response = llm.llm.invoke(messages)
    content = response.content
    """
    print("\n[DEBUG] createDiagnosis Raw Output:")
    print(content)
    print("[DEBUG] End of Raw Output\n")
    """
    
    diagnosis_output = parse_diagnosis_text(content)

    
    if diagnosis_output and diagnosis_output.ans:
        return (diagnosis_output, prompt)
    
    return None, None