from collections import Counter
import json
from typing import Optional
import re
from langchain.schema import HumanMessage
from ..state.state_types import (
    State,
    DiagnosisOutput,
    DiagnosisFormat,
    DiagnosisOutputV2,
    DiagnosisFormatV2,
    EvidenceItem,
    MergedDiseaseCandidateV2,
    SourceRecord,
)
from ..llm.prompt import prompt_dict, build_prompt
from ..utils.response_serializer import omim_number
from .candidate_metadata import canonical_omim_id, enrich_candidates, names_compatible


_REFERENCE_SOURCE_ALIASES = (
    ("pubcasefinder", ("pubcasefinder", "pcf")),
    ("zeroshot", ("zeroshot", "zero-shot")),
    ("phenobrain", ("phenobrain",)),
    ("phenotypesearch", ("phenotypesearch", "phenotype search", "phenotype similarity search")),
    ("web", ("web search", "researchgate", "syndactyly")),
)
_URL_PATTERN = re.compile(r"https?://[^\s)]+")
_MISSING_URL_PATTERN = re.compile(r"\s*URL:\s*(?:Not provided|N/A)\.?", re.IGNORECASE)
_PUBMED_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?(?:pubmed\.ncbi\.nlm\.nih\.gov|pmc\.ncbi\.nlm\.nih\.gov)/",
    re.IGNORECASE,
)


def is_pubmed_reference(reference: str) -> bool:
    """Return whether a legacy reference is an external PubMed/PMC citation."""
    value = str(reference or "")
    if "phenotypesearch" in value.lower():
        return False
    return bool(_PUBMED_URL_PATTERN.search(value))


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
    cited_ids: list[int] = []
    for group in re.findall(r"\[((?:\d+\s*,\s*)*\d+)\]", description or ""):
        cited_ids.extend(int(value.strip()) for value in group.split(","))
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


def _filter_and_renumber_references(reference_text: Optional[str], source_records):
    """Keep the old numbered-string format while retaining only PubMed/PMC items."""
    entries = _parse_reference_entries(reference_text)
    if not entries:
        return {}, {}, reference_text

    enriched = {
        reference_id: _enrich_reference(reference, source_records)
        for reference_id, reference in entries.items()
    }
    allowed = {
        reference_id: reference
        for reference_id, reference in enriched.items()
        if is_pubmed_reference(reference)
    }
    old_to_new = {
        old_id: new_id
        for new_id, old_id in enumerate(sorted(allowed), 1)
    }
    filtered = {
        old_to_new[old_id]: reference
        for old_id, reference in allowed.items()
    }
    block = "\n".join(
        f"{reference_id}. {filtered[reference_id]}"
        for reference_id in sorted(filtered)
    )
    return filtered, old_to_new, block


def _rewrite_citations(description: str, old_to_new: dict[int, int]) -> str:
    def replace(match):
        mapped = [
            old_to_new[int(value.strip())]
            for value in match.group(1).split(",")
            if int(value.strip()) in old_to_new
        ]
        return f"[{','.join(str(value) for value in mapped)}]" if mapped else ""

    return re.sub(r"\[((?:\d+\s*,\s*)*\d+)\]", replace, description or "")


def attach_diagnosis_references(diagnosis_output: DiagnosisOutput, source_records=None) -> DiagnosisOutput:
    """Attach PubMed/PMC references without changing the legacy string format."""
    if not diagnosis_output or not getattr(diagnosis_output, "ans", None):
        return diagnosis_output

    enriched_entries, old_to_new, reference_block = _filter_and_renumber_references(
        diagnosis_output.reference,
        source_records,
    )
    diagnosis_output.reference = reference_block
    for diagnosis in diagnosis_output.ans:
        diagnosis.description = _rewrite_citations(diagnosis.description or "", old_to_new)
        original_ids = _reference_ids_from_description(
            diagnosis.description or "",
            _parse_reference_entries(diagnosis_output.reference),
        )
        # Descriptions have already been renumbered, so these IDs are current IDs.
        reference_ids = original_ids
        mapped = [enriched_entries[reference_id] for reference_id in reference_ids if reference_id in enriched_entries]
        existing = []
        for reference in list(getattr(diagnosis, "reference", []) or []):
            numeric_match = re.fullmatch(r"\s*\[?(\d+)\]?\.?\s*", str(reference))
            if numeric_match and int(numeric_match.group(1)) in enriched_entries:
                existing.append(enriched_entries[int(numeric_match.group(1))])
            elif is_pubmed_reference(_enrich_reference(str(reference), source_records)):
                existing.append(_enrich_reference(str(reference), source_records))
        diagnosis.reference = list(
            dict.fromkeys(reference for reference in existing + mapped if is_pubmed_reference(reference))
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
            # The legacy rank is display order. Tool ranks are stored separately.
            rank = len(cases) + 1
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


def _v2_status(candidate: MergedDiseaseCandidateV2, evidence: list[EvidenceItem], sources: list[SourceRecord]) -> str:
    candidate_evidence = [item for item in evidence if item.candidate_id == candidate.candidate_id]
    source_map = {source.source_id: source for source in sources}
    if any(item.support == "contradicts" for item in candidate_evidence) and not any(
        item.support == "supports" for item in candidate_evidence
    ):
        return "contradictory"
    if any(
        item.support == "supports"
        and source_map.get(item.source_id)
        and source_map[item.source_id].source_type == "literature"
        for item in candidate_evidence
    ):
        return "supported"
    if candidate_evidence:
        return "insufficient"
    return "not_retrieved"


def _v2_reason(candidate: MergedDiseaseCandidateV2, evidence: list[EvidenceItem], status: str) -> tuple[str, str]:
    candidate_evidence = [item for item in evidence if item.candidate_id == candidate.candidate_id]
    if candidate_evidence:
        claims_en = "; ".join(item.claim_en for item in candidate_evidence[:4])
        claims_ja = "；".join(item.claim_ja for item in candidate_evidence[:4])
    else:
        claims_en = "No validated literature evidence was retrieved for this candidate."
        claims_ja = "この候補を支持する検証済みの文献根拠は取得できませんでした。"
    suffix_en = {
        "supported": " Literature evidence supports the candidate, but this is not a confirmed diagnosis.",
        "insufficient": " The candidate remains possible, but literature support is insufficient.",
        "contradictory": " Retrieved evidence contains a clinically important contradiction.",
        "not_retrieved": " Literature evidence was not retrieved; this candidate must not be described as literature-supported.",
    }[status]
    suffix_ja = {
        "supported": " 文献根拠はこの候補を支持しますが、確定診断ではありません。",
        "insufficient": " 候補としては残りますが、文献支持は不十分です。",
        "contradictory": " 取得した文献根拠には臨床的に重要な矛盾が含まれます。",
        "not_retrieved": " 文献根拠を取得できていないため、文献支持済みとは表現できません。",
    }[status]
    return claims_en + suffix_en, claims_ja + suffix_ja


def validate_v2_candidate_output(output, candidates: list[MergedDiseaseCandidateV2]) -> dict:
    """Validate identity fields without deciding clinical correctness.

    Tentative diagnosis must retain every merged candidate. Unknown/duplicate
    IDs, omitted candidates, rank corruption, OMIM mismatch, and clear
    disease-name mismatch are generation errors and trigger one regeneration.
    """
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    items = list(getattr(output, "ans", []) or []) if output is not None else []
    errors: list[str] = []
    ids = [getattr(item, "candidate_id", None) for item in items]
    counts = Counter(ids)
    unknown = [item_id for item_id in counts if item_id not in candidate_map]
    duplicate = [item_id for item_id, count in counts.items() if count > 1]
    omitted = [candidate.candidate_id for candidate in candidates if candidate.candidate_id not in counts]
    if not items:
        errors.append("Tentative diagnosis returned no candidates.")
    if unknown:
        errors.append(f"Unknown candidate_id values: {unknown}")
    if duplicate:
        errors.append(f"Duplicate candidate_id values: {duplicate}")
    if omitted:
        errors.append(f"Tentative diagnosis omitted merged candidates: {omitted}")

    ranks = [getattr(item, "rank", None) for item in items]
    if ranks != list(range(1, len(ranks) + 1)):
        errors.append("Tentative diagnosis ranks must be contiguous and start at 1.")

    identity_errors = []
    for item in items:
        candidate = candidate_map.get(getattr(item, "candidate_id", None))
        if candidate is None:
            continue
        expected_omim = canonical_omim_id(candidate.OMIM_id)
        actual_omim = canonical_omim_id(getattr(item, "OMIM_id", None))
        if expected_omim != actual_omim:
            identity_errors.append(
                f"{candidate.candidate_id}: OMIM mismatch (expected {expected_omim}, got {actual_omim})."
            )
        if not names_compatible(getattr(item, "disease_name_en", ""), candidate.disease_name):
            identity_errors.append(
                f"{candidate.candidate_id}: disease_name_en conflicts with the authoritative candidate name."
            )
    errors.extend(identity_errors)
    return {
        "matched": not errors,
        "status": "accepted" if not errors else "invalid",
        "errors": errors,
        "unknown_ids": unknown,
        "duplicate_ids": duplicate,
        "omitted_candidate_ids": omitted,
        "candidate_count": len(items),
    }


def _v2_regeneration_prompt(original_prompt: str, output, validation: dict) -> str:
    previous = output.model_dump_json() if hasattr(output, "model_dump_json") else json.dumps(output, ensure_ascii=False)
    return (
        f"{original_prompt}\n\n"
        "## Previous tentative output\n"
        f"{previous}\n\n"
        "## Identity validation errors\n"
        f"{json.dumps(validation, ensure_ascii=False)}\n\n"
        "## Regeneration instructions\n"
        "Return every supplied candidate exactly once, using only existing candidate_id values. "
        "Do not omit a candidate or add one outside the input. Assign ranks 1..N, copy the authoritative OMIM_id exactly, "
        "and use the authoritative disease identity rather than inventing a name or subtype. "
        "Return the structured output only."
    )


def createDiagnosisV2(state: State) -> tuple[DiagnosisOutputV2 | None, str, dict]:
    """Generate a bilingual tentative diagnosis with one identity-safe retry."""
    llm = state.get("llm")
    candidates = enrich_candidates(state.get("mergedDiseaseCandidates", []) or [])
    evidence = list(state.get("evidence", []) or [])
    sources = list(state.get("sources", []) or [])
    if not candidates:
        return DiagnosisOutputV2(ans=[], evidence=evidence), "", {
            "status": "empty_input", "attempts": [], "errors": []
        }

    def fallback() -> DiagnosisOutputV2:
        answers = []
        for rank, candidate in enumerate(candidates, 1):
            status = _v2_status(candidate, evidence, sources)
            description_en, description_ja = _v2_reason(candidate, evidence, status)
            answers.append(
                DiagnosisFormatV2(
                    candidate_id=candidate.candidate_id,
                    disease_name_en=candidate.disease_name,
                    disease_name_ja=candidate.disease_name,
                    OMIM_id=canonical_omim_id(candidate.OMIM_id),
                    rank=rank,
                    description_en=description_en,
                    description_ja=description_ja,
                    evidence_ids=[item.evidence_id for item in evidence if item.candidate_id == candidate.candidate_id],
                    literature_status=status,
                )
            )
        return DiagnosisOutputV2(ans=answers, evidence=evidence)

    if not llm:
        return fallback(), "", {"status": "fallback_generation_error", "attempts": [], "errors": ["LLM instance not found."]}

    evidence_text = []
    for item in evidence:
        evidence_text.append(
            f"{item.evidence_id} | candidate={item.candidate_id} | type={item.evidence_type} | "
            f"support={item.support} | confidence={item.confidence} | EN={item.claim_en} | JA={item.claim_ja} | "
            f"excerpt={item.excerpt}"
        )
    candidate_text = []
    for index, candidate in enumerate(candidates, 1):
        supports = "; ".join(
            f"{support.tool}:rank={support.rank},score={support.score}"
            for support in candidate.tool_supports
        )
        candidate_text.append(
            f"candidate_order={index}; candidate_id={candidate.candidate_id}; disease={candidate.disease_name}; OMIM={candidate.OMIM_id or 'N/A'}; "
            f"tools={supports or 'none'}"
        )
    prompt = f"""You are a clinical genetics assistant generating a bilingual differential diagnosis.
Use ONLY the supplied candidate list and validated Evidence items. Do not cite diagnostic tools as literature.
The CANDIDATES list is the authoritative, already-merged candidate list; do not split, duplicate, omit, or invent candidates.
Return every supplied candidate exactly once, including candidates with no retrieved literature Evidence. If N candidates are supplied, return N items with ranks exactly 1, 2, ..., N.
Rank candidates by overall clinical compatibility. The rank field is the tentative differential rank, not a PubCaseFinder, GestaltMatcher, ZeroShot, PhenoBrain, or PhenotypeSearch rank.
Tool ranks and scores are auxiliary signals for deciding order only; NEVER copy a tool rank into the final rank field, and NEVER reuse a rank.
Use the candidate order shown above as the default ordering; change it only when the supplied phenotype and literature Evidence provide a clear reason.
Use exactly the supplied candidate_id and evidence_id values.
English and Japanese descriptions must express the same claims and evidence IDs. Never invent a reference, PMID, gene, phenotype, or mechanism.
Descriptions must state concrete patient features that match the supplied Evidence, not only say that the candidate is supported.
Each description should be a concise but substantive multi-sentence explanation: (1) identify the most relevant patient findings, (2) state the specific disease features reported in the supplied Evidence, (3) explain why the patient findings do or do not correspond to those features, and (4) state important limitations and the conclusion. Do not reduce the explanation to "matches" or "is supported."
When Evidence contains limitations, distinguish explicitly absent findings from findings that were not reported.
Do not present general medical knowledge as literature evidence; use only the supplied Evidence IDs for literature claims.
Set literature_status to supported only when a literature Evidence item supports the candidate; use insufficient or not_retrieved otherwise.

PATIENT_PRESENT_HPO: {', '.join(state.get('hpoDict', {}).values())}
PATIENT_ABSENT_HPO: {', '.join(state.get('absentHpoDict', {}).values()) if state.get('use_absentHPO') else 'N/A'}
CANDIDATES:\n{chr(10).join(candidate_text)}
EVIDENCE:\n{chr(10).join(evidence_text) or 'No validated Evidence'}"""
    structured_llm = llm.get_structured_llm(DiagnosisOutputV2)
    validation = {"status": "not_run", "attempts": [], "errors": []}
    current_prompt = prompt
    for attempt in (1, 2):
        try:
            result = llm.invoke_with_content_filter_retry(
                structured_llm,
                [HumanMessage(content=current_prompt)],
                context=f"DiagnosisV2:attempt_{attempt}",
            )
        except Exception as exc:
            validation["attempts"].append({"attempt": attempt, "status": "generation_error", "error_type": type(exc).__name__})
            validation["errors"].append(f"generation_error:{type(exc).__name__}")
            validation["status"] = "fallback_generation_error"
            return fallback(), current_prompt, validation

        attempt_validation = validate_v2_candidate_output(result, candidates)
        validation["attempts"].append({
            "attempt": attempt,
            "status": "accepted" if attempt_validation["matched"] else "candidate_mismatch",
            "validation": attempt_validation,
            "output": result.model_dump() if hasattr(result, "model_dump") else None,
        })
        if attempt_validation["matched"]:
            allowed_candidates = {candidate.candidate_id: candidate for candidate in candidates}
            allowed_evidence = {item.evidence_id for item in evidence}
            accepted = result.model_copy(deep=True)
            for item in accepted.ans:
                authoritative = allowed_candidates[item.candidate_id]
                item.OMIM_id = canonical_omim_id(authoritative.OMIM_id)
                item.evidence_ids = [value for value in item.evidence_ids if value in allowed_evidence]
                if not item.evidence_ids:
                    item.literature_status = "not_retrieved"
            validation["status"] = "accepted" if attempt == 1 else "regenerated"
            return DiagnosisOutputV2(ans=accepted.ans, evidence=evidence), current_prompt, validation
        validation["errors"] = attempt_validation["errors"]
        if attempt == 1:
            current_prompt = _v2_regeneration_prompt(prompt, result, attempt_validation)

    validation["status"] = "fallback_after_regeneration"
    return fallback(), current_prompt, validation
