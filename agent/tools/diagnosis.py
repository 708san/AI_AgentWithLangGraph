from typing import Optional
import re
from langchain.schema import HumanMessage
from ..state.state_types import State, DiagnosisOutput, DiagnosisFormat
from ..llm.prompt import prompt_dict, build_prompt
from ..utils.response_serializer import omim_number


_REFERENCE_SOURCE_ALIASES = (
    ("pubcasefinder", ("pubcasefinder", "pcf")),
    ("zeroshot", ("zeroshot", "zero-shot")),
    ("phenobrain", ("phenobrain",)),
    ("phenotypesearch", ("phenotypesearch", "phenotype search", "phenotype similarity search")),
    ("web", ("web search", "researchgate", "syndactyly")),
)
_URL_PATTERN = re.compile(r"https?://[^\s)]+")
_MISSING_URL_PATTERN = re.compile(r"\s*URL:\s*(?:Not provided|N/A)\.?", re.IGNORECASE)


def _parse_reference_entries(reference_text: Optional[str]) -> dict[int, str]:
    """Parse the existing numbered reference block without another model call."""
    if not reference_text:
        return {}

    entry_pattern = re.compile(
        r"(?ms)^\s*(?:[-*]\s*)?(?:\[(\d+)\]|(\d+)[.)])\s*(.*?)(?=^\s*(?:[-*]\s*)?(?:\[\d+\]|\d+[.)])\s|\Z)"
    )
    entries = {}
    for match in entry_pattern.finditer(str(reference_text)):
        reference_id = match.group(1) or match.group(2)
        entries[int(reference_id)] = match.group(3).strip()
    return entries


def _reference_ids_from_description(description: str, reference_entries: dict[int, str]) -> list[int]:
    """Resolve explicit citations, with a deterministic tool-name fallback."""
    cited_ids = [int(value) for value in re.findall(r"\[(\d+)\]", description or "")]
    if cited_ids:
        return cited_ids

    description_lower = (description or "").lower()
    inferred_ids = []
    for _, aliases in _REFERENCE_SOURCE_ALIASES:
        if not any(alias in description_lower for alias in aliases):
            continue
        for reference_id, reference_text in reference_entries.items():
            reference_lower = reference_text.lower()
            if any(alias in reference_lower for alias in aliases):
                inferred_ids.append(reference_id)
    return list(dict.fromkeys(inferred_ids))


def _record_value(record, key: str) -> str:
    if isinstance(record, dict):
        return str(record.get(key) or "")
    return str(getattr(record, key, "") or "")


def _url_for_reference(reference: str, source_records) -> Optional[str]:
    """Find the original search URL for a reference by its source title."""
    if _URL_PATTERN.search(reference or ""):
        return None

    reference_lower = (reference or "").lower()
    for record in source_records or []:
        url = _record_value(record, "url")
        title = _record_value(record, "title")
        disease_name = _record_value(record, "disease_name")
        if not url or not re.match(r"https?://", url):
            continue
        for label in (title, disease_name):
            label_lower = re.sub(r"\s+", " ", label.lower()).strip()
            if label_lower and len(label_lower) >= 8 and label_lower in reference_lower:
                return url
    return None


def _enrich_reference(reference: str, source_records) -> str:
    reference = _MISSING_URL_PATTERN.sub("", reference or "").strip()
    url = _url_for_reference(reference, source_records)
    if not url:
        return reference
    return f"{reference.rstrip()} URL: {url}"


def _enrich_reference_block(reference_text: Optional[str], source_records) -> Optional[str]:
    if not reference_text:
        return reference_text
    entries = _parse_reference_entries(reference_text)
    if not entries:
        return reference_text
    return "\n".join(
        f"{reference_id}. {_enrich_reference(reference, source_records)}"
        for reference_id, reference in entries.items()
    )


def attach_diagnosis_references(diagnosis_output: DiagnosisOutput, source_records=None) -> DiagnosisOutput:
    """Attach cited/tool references and source URLs to each diagnosis item."""
    if not diagnosis_output or not getattr(diagnosis_output, "ans", None):
        return diagnosis_output

    reference_entries = _parse_reference_entries(diagnosis_output.reference)
    enriched_entries = {
        reference_id: _enrich_reference(reference, source_records)
        for reference_id, reference in reference_entries.items()
    }
    diagnosis_output.reference = _enrich_reference_block(diagnosis_output.reference, source_records)
    for diagnosis in diagnosis_output.ans:
        reference_ids = _reference_ids_from_description(diagnosis.description or "", reference_entries)
        mapped = [enriched_entries[reference_id] for reference_id in reference_ids if reference_id in enriched_entries]
        existing = []
        for reference in list(getattr(diagnosis, "reference", []) or []):
            numeric_match = re.fullmatch(r"\s*\[?(\d+)\]?\.?\s*", str(reference))
            if numeric_match and int(numeric_match.group(1)) in enriched_entries:
                existing.append(enriched_entries[int(numeric_match.group(1))])
            else:
                existing.append(_enrich_reference(str(reference), source_records))
        diagnosis.reference = list(
            dict.fromkeys(_enrich_reference(reference, source_records) for reference in existing + mapped)
        )
    return diagnosis_output


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
    
    return attach_diagnosis_references(DiagnosisOutput(ans=cases, reference=references))

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
            f"(OMIM: {omim_number(candidate.get('OMIM_id')) or 'N/A'}, "
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
        f"- {res.get('title', 'No Title')}\n"
        f"  URL: {res.get('url', 'N/A')}\n"
        f"  {res.get('content') or res.get('snippet', 'No Content')}"
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
    diagnosis_output = attach_diagnosis_references(
        diagnosis_output,
        source_records=(web_search_results or []) + (state.get("memory", []) or []),
    )

    
    if diagnosis_output and diagnosis_output.ans:
        return (diagnosis_output, prompt)
    
    return None, None
