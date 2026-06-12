from typing import Optional
import re
from langchain.schema import HumanMessage
from ..state.state_types import State, DiagnosisOutput, DiagnosisFormat
from ..llm.prompt import prompt_dict, build_prompt

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
    use_absent_hpo = state.get("use_absentHPO", False)
    absent_hpo_list = (
        [value for value in state.get("absentHpoDict", {}).values() if value]
        if use_absent_hpo
        else []
    )
    onset = state.get("onset", "Unknown")
    sex = state.get("sex", "Unknown")
    gestalt_matcher_results = state.get("GestaltMatcher", [])
    web_search_results = state.get("webresources", [])
    merged_candidates = state.get("mergedDiseaseCandidates", [])
    llm = state.get("llm")

    if not llm:
        print("LLM instance not found in state.")
        return None, None

    has_gestalt = gestalt_matcher_results and len(gestalt_matcher_results) > 0

    candidate_lines = []
    for index, candidate in enumerate(merged_candidates, 1):
        tool_parts = []
        for ranking in candidate.get("tool_rankings", []):
            rank_text = f"rank {ranking.get('rank')}" if ranking.get("rank") is not None else "rank N/A"
            score = ranking.get("score")
            score_text = f", score {score:.3f}" if isinstance(score, (int, float)) else ""
            matched_hpo = ranking.get("matched_hpo_id")
            matched_text = f", matched HPO: {matched_hpo}" if matched_hpo else ""
            note = ranking.get("note")
            note_text = f", note: {note}" if note else ""
            tool_parts.append(
                f"{ranking.get('tool', 'UnknownTool')} ({rank_text}{score_text}{matched_text}{note_text})"
            )
        candidate_lines.append(
            f"{index}. {candidate.get('disease_name', 'N/A')} "
            f"(OMIM: {candidate.get('OMIM_id') or 'N/A'}, "
            f"supported by {candidate.get('consensus_count', 0)} tool(s), "
            f"best tool rank: {candidate.get('best_rank', 'N/A')})\n"
            f"   Tool rankings: {'; '.join(tool_parts) if tool_parts else 'No tool ranking details.'}"
        )

    merged_candidate_text = "\n".join(candidate_lines) if candidate_lines else "No merged disease candidates."

    if has_gestalt:
        print(f"[DEBUG] 使用するプロンプト: diagnosis_prompt (GestaltMatcher有り)")
    else:
        print(f"[DEBUG] 使用するプロンプト: diagnosis_prompt_no_gestalt (GestaltMatcher無し)")

    # Web search results
    web_text = "\n".join([
        f"- {res.get('title', 'No Title')}: {res.get('content') or res.get('snippet', 'No Content')}"
        for res in web_search_results
    ]) if web_search_results else "No relevant web search results found."

    # GestaltMatcherの結果があるかどうかで異なるプロンプトを使用
    if has_gestalt:
        prompt_template = prompt_dict["diagnosis_prompt"]
    else:
        # GestaltMatcher情報がない場合のプロンプト
        prompt_template = prompt_dict["diagnosis_prompt_no_gestalt"]

    prompt = build_prompt(
        prompt_template,
        {
            "hpo_list": ", ".join(hpo_list),
            "absent_hpo_list": ", ".join(absent_hpo_list),
            "use_absentHPO": use_absent_hpo,
            "onset": onset,
            "sex": sex,
            "merged_candidate_results": merged_candidate_text,
            "web_search_results": web_text,
        },
    )

    # --- Query the LLM to get the diagnosis result ---
    messages = [HumanMessage(content=prompt)]
    
    response = llm.invoke_with_content_filter_retry(
        llm.llm,
        messages,
        context="Diagnosis",
    )
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
