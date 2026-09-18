from typing import Optional
import logging
import json
from collections import Counter
import re
from langchain.schema import HumanMessage
from ..state.state_types import State, DiagnosisOutput, DiagnosisFormat, TentativeDiagnosisOutput
from ..llm.prompt import prompt_dict, build_prompt

logger = logging.getLogger(__name__)


def validate_candidate_ids(input_candidates, output):
    """Check missing, unexpected and duplicate IDs within one candidate-ranking call.

    A match only proves ID coverage. Names, OMIM IDs, ranks and clinical validity
    are not compared, and the output is never filtered or rewritten here.
    """
    expected = [item["candidate_id"] for item in input_candidates]
    counts = Counter(item.candidate_id for item in output.ans)
    missing = [key for key in expected if key not in counts]
    unexpected = [key for key in counts if key not in set(expected)]
    duplicates = [key for key, count in counts.items() if count > 1]
    return {"matched": not (missing or unexpected or duplicates),
            "missing_ids": missing, "unexpected_ids": unexpected, "duplicate_ids": duplicates}


def record_diagnosis_attempt(attempt, prompt, input_candidates, output, response, validation, parsing_error):
    """EvaluationTrace records these locals immediately, before normalization.

    Keep clinical payloads out of console logs; evaluation JSONL stores both
    attempts, the original ID/name/OMIM mapping and the returned values.
    """
    if validation is not None:
        logger.log(logging.INFO if validation["matched"] else logging.WARNING,
                   "Tentative diagnosis attempt %s candidate validation: %s",
                   attempt, validation)


def regeneration_prompt(original_prompt, output, validation, input_candidates):
    missing = [item for item in input_candidates if item["candidate_id"] in validation["missing_ids"]]
    return original_prompt + "\n\n## Previous output\n" + output.model_dump_json() + (
        "\n\n## Validation results\n" + json.dumps({**validation, "missing_candidates": missing}, ensure_ascii=False) +
        "\n\n## Regeneration instructions\n"
        "- Return a complete replacement containing every original input candidate exactly once. "
        "Do not return only missing candidates. Copy candidate_id exactly; remove duplicate and unexpected IDs.\n"
        "- Follow the same structured output schema, including disease_name and OMIM_id.\n"
        "- This is a completeness check, not evidence of clinical likelihood. Do not promote a candidate because it was missing.\n"
        "- Rank all candidates using the original patient information and evidence; reassess relative ranks where necessary.\n"
        "- Treat the previous output as a draft, not an additional evidence source.\n"
        "- Do not invent supporting findings to justify inclusion.\n"
        "- Return the complete structured output only, without an explanation of the corrections."
    )

def parse_diagnosis_text(text: str) -> DiagnosisOutput:
    """
    旧形式の保存済みテキストをDiagnosisOutputへ変換する。新規推論では使用しない。
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

def createDiagnosis(state: State) -> tuple[Optional[DiagnosisOutput], Optional[str]]:
    """
    Integrates multiple information sources (PCF, ZeroShot, GestaltMatcher, PhenotypeSearch) 
    to generate a structured tentative diagnosis. Returns (output, prompt).
    Retry candidate-ID mismatches once; retain the last parsed result even if
    incomplete or empty. Invalid/refused responses still raise parsing errors.

    Both GestaltMatcher prompt variants use TentativeDiagnosisOutput with strict
    JSON schema. IDs are local to this call and stable across its two attempts.
    Names and OMIM IDs are logged, not checked against or replaced by input values.
    record_diagnosis_attempt exposes each attempt to evaluation tracing; the return
    tuple contains only the final parsed output and its prompt. Normalization is
    a later step. Content-filter/provider retries are separate from regeneration.
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
    phenobrain_results = state.get("phenoBrain", [])
    web_search_results = state.get("webresources", [])
    merged_candidates = state.get("mergedDiseaseCandidates", [])
    llm = state.get("llm")

    if not llm:
        print("LLM instance not found in state.")
        return None, None

    has_gestalt = gestalt_matcher_results and len(gestalt_matcher_results) > 0
    merged_candidate_sources = ["PubCaseFinder", "Zero-Shot Diagnosis"]
    if has_gestalt:
        merged_candidate_sources.append("GestaltMatcher")
    if phenobrain_results:
        merged_candidate_sources.append("PhenoBrain")
    merged_candidate_sources.append("Phenotype Similarity Search")

    candidate_lines = []
    input_candidates = []
    for index, candidate in enumerate(merged_candidates, 1):
        candidate_id = f"candidate_{index:04d}"
        input_candidates.append({"candidate_id": candidate_id,
            "disease_name": candidate.get("disease_name"), "OMIM_id": candidate.get("OMIM_id")})
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
            f"{index}. [candidate_id: {candidate_id}] {candidate.get('disease_name', 'N/A')} "
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
            "merged_candidate_sources": ", ".join(merged_candidate_sources),
            "merged_candidate_results": merged_candidate_text,
            "web_search_results": web_text,
        },
    )

    # --- Query the LLM to get the diagnosis result ---
    original_prompt = prompt
    structured_llm = llm.llm.with_structured_output(
        TentativeDiagnosisOutput, method="json_schema", strict=True, include_raw=True,
    )
    try:
        for attempt in (1, 2):
            response = parsing_error = diagnosis_output = validation = None
            try:
                structured_response = llm.invoke_with_content_filter_retry(
                    structured_llm, [HumanMessage(content=prompt)], context="Diagnosis",
                )
                response = structured_response["raw"]
                parsing_error = structured_response["parsing_error"]
                diagnosis_output = structured_response["parsed"]
                if parsing_error is not None:
                    raise ValueError("Tentative diagnosis structured output parsing failed") from parsing_error
                if response is not None and response.additional_kwargs.get("refusal"):
                    raise ValueError("Tentative diagnosis request was refused")
                if not isinstance(diagnosis_output, TentativeDiagnosisOutput):
                    raise ValueError("Tentative diagnosis returned no structured output")
                validation = validate_candidate_ids(input_candidates, diagnosis_output)
            except Exception as exc:
                record_diagnosis_attempt(attempt, prompt, input_candidates, diagnosis_output,
                                         response, validation, parsing_error or exc)
                raise
            record_diagnosis_attempt(attempt, prompt, input_candidates, diagnosis_output,
                                     response, validation, None)
            if validation["matched"] or attempt == 2:
                if not validation["matched"]:
                    logger.warning("Tentative diagnosis candidate mismatch remains; retaining final output")
                return diagnosis_output, prompt
            prompt = regeneration_prompt(original_prompt, diagnosis_output, validation, input_candidates)
    except Exception as exc:
        # Clinical response content belongs in local evaluation artifacts, not
        # console logs. Preserve the exception and its cause for the caller.
        logger.error("Tentative diagnosis structured output failed (%s)", type(exc).__name__)
        raise
