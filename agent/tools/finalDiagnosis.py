import os
import re
import time
import traceback
from langchain.schema import HumanMessage
from typing_extensions import Optional
from ..state.state_types import (
    State,
    DiagnosisOutput,
    DiagnosisOutputV2,
    DiagnosisFormatV2,
    EvidenceItem,
    FinalDiagnosisGenerationOutput,
    ReflectionOutputV2,
)
from ..llm.prompt import prompt_dict, build_prompt
from ..llm.llm_wrapper import is_content_filter_error
from ..utils.response_serializer import omim_number
from .diagnosis import attach_diagnosis_references
from .candidate_metadata import canonical_omim_id, enrich_candidates
from .evidence_validation import validate_v2_identity_fields


INLINE_EVIDENCE_MARKER = re.compile(r"\[\[EVIDENCE:([^\]]+)\]\]")


def _candidate_evidence(candidate_id: str, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    return [item for item in evidence if item.candidate_id == candidate_id]


def _evidence_status(items: list[EvidenceItem], candidate_sources=()) -> str:
    if any(item.support == "supports" for item in items):
        return "supported"
    if any(item.support == "contradicts" for item in items):
        return "contradictory"
    # No usable claim after retrieving a candidate-matched source is different
    # from retrieving no source in this run. Neither status is a negative
    # clinical finding.
    return "insufficient" if items or list(candidate_sources) else "not_retrieved"


def _fallback_description(
    disease_name: str,
    patient_features: list[str],
    evidence: list[EvidenceItem],
) -> tuple[str, str]:
    """Build an auditable long-form fallback when final generation fails."""
    feature_text = ", ".join(patient_features[:8]) or "the supplied phenotype"
    evidence_en = []
    evidence_ja = []
    for item in evidence[:5]:
        marker = f"[[EVIDENCE:{item.evidence_id}]]"
        evidence_en.append(f"{item.claim_en} {marker}")
        evidence_ja.append(f"{item.claim_ja} {marker}")

    if evidence_en:
        joined_en = " ".join(evidence_en)
        joined_ja = " ".join(evidence_ja)
        description_en = (
            f"The patient presents with {feature_text}. The retrieved literature for "
            f"{disease_name} reports the following disease-specific findings: {joined_en} "
            "These findings provide literature-based support for phenotypic similarity, "
            "but they do not by themselves establish a molecular diagnosis or prove that "
            "all disease features are present in this patient."
        )
        description_ja = (
            f"患者には{feature_text}が認められます。{disease_name}について取得された文献では、"
            f"次の疾患特異的所見が報告されています。{joined_ja} "
            "これらは表現型の類似性を文献に基づいて支持しますが、これだけで分子診断を確定したり、"
            "疾患の全所見が患者に存在すると判断したりすることはできません。"
        )
        return description_en, description_ja

    return (
        f"No validated literature Evidence is available for {disease_name} in this run. "
        "This does not establish that no relevant literature exists; the candidate remains a differential hypothesis based on the supplied phenotype and diagnostic-tool signals.",
        f"この実行では{disease_name}について検証済みの文献Evidenceを確認できませんでした。"
        "これは関連文献が存在しないことを意味しません。候補は、提供された表現型と診断ツールの情報に基づく鑑別仮説として扱います。",
    )


def _apply_evidence_contract(
    item: DiagnosisFormatV2,
    evidence: list[EvidenceItem],
    patient_features: list[str],
    sources=(),
) -> DiagnosisFormatV2:
    """Keep final descriptions, Evidence IDs, and inline markers consistent."""
    candidate_items = _candidate_evidence(item.candidate_id, evidence)
    candidate_sources = [
        source for source in sources
        if source.source_type == "literature"
        and item.candidate_id in (source.candidate_ids or [])
    ]
    allowed = {evidence_item.evidence_id for evidence_item in candidate_items}

    def keep_valid_markers(text: str) -> str:
        return INLINE_EVIDENCE_MARKER.sub(
            lambda match: match.group(0) if match.group(1) in allowed else "",
            text or "",
        )

    item.description_en = keep_valid_markers(item.description_en)
    item.description_ja = keep_valid_markers(item.description_ja)
    marker_ids = {
        marker_id
        for text in (item.description_en, item.description_ja)
        for marker_id in INLINE_EVIDENCE_MARKER.findall(text or "")
        if marker_id in allowed
    }
    requested_ids = [evidence_id for evidence_id in item.evidence_ids if evidence_id in allowed]
    evidence_ids = list(dict.fromkeys(requested_ids + sorted(marker_ids)))

    # Models occasionally return valid evidence_ids but omit the inline marker.
    # Append only the missing markers; normal model output keeps markers next
    # to the relevant sentence.
    if candidate_items and not evidence_ids:
        evidence_ids = [evidence_item.evidence_id for evidence_item in candidate_items]
    if evidence_ids:
        selected_items = [
            evidence_item
            for evidence_item in candidate_items
            if evidence_item.evidence_id in evidence_ids
        ]
        for evidence_id in evidence_ids:
            marker = f"[[EVIDENCE:{evidence_id}]]"
            if marker not in item.description_en:
                item.description_en = f"{item.description_en.rstrip()} {marker}".strip()
            if marker not in item.description_ja:
                item.description_ja = f"{item.description_ja.rstrip()} {marker}".strip()
        item.literature_status = _evidence_status(
            selected_items or candidate_items,
            candidate_sources,
        )
    else:
        item.literature_status = _evidence_status(candidate_items, candidate_sources)

    item.evidence_ids = evidence_ids
    return item


def _is_content_filter_error(error: Exception) -> bool:
    return is_content_filter_error(error)


def _is_retryable_final_error(error: Exception) -> bool:
    if _is_content_filter_error(error):
        return False
    message = str(error).lower()
    markers = [
        "timeout",
        "timed out",
        "rate limit",
        "too many requests",
        "temporarily unavailable",
        "connection",
        "server error",
        "service unavailable",
        "gateway",
    ]
    return any(marker in message for marker in markers)


def _truncate_text(text: str, limit: int = 500) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _final_request_timeout_seconds() -> float:
    return float(os.getenv("FINAL_DIAGNOSIS_REQUEST_TIMEOUT_SECONDS", "240"))


def _final_retry_wait_seconds() -> float:
    return float(os.getenv("FINAL_DIAGNOSIS_RETRY_WAIT_SECONDS", "30"))


def _format_tentative_diagnosis(tentative_result, compact: bool = False) -> str:
    if tentative_result is None or not hasattr(tentative_result, "ans"):
        return ""

    lines = []
    for i, item in enumerate(tentative_result.ans):
        if compact:
            lines.append(
                f"{i+1}. {item.disease_name} (Rank: {item.rank}, OMIM: {omim_number(item.OMIM_id) or 'N/A'})"
            )
        else:
            lines.append(
                f"{i+1}. {item.disease_name} (Rank: {item.rank})\n"
                f"Description: {item.description}\n"
                f"References: {'; '.join(getattr(item, 'reference', []) or [])}"
            )

    tentative_result_str = "\n".join(lines)
    if not compact and hasattr(tentative_result, "reference") and tentative_result.reference:
        tentative_result_str += f"\n[DiagnosisOutput Reference]: {tentative_result.reference}"
    return tentative_result_str


def _format_reflection(judgements, compact: bool = False) -> str:
    if judgements is None or not hasattr(judgements, "ans"):
        return ""

    judgements_list = []
    for i, item in enumerate(judgements.ans):
        if compact:
            judgement_entry = (
                f"{i+1}. {getattr(item, 'disease_name', '')}\n"
                f"Correctness: {getattr(item, 'Correctness', '')}\n"
                f"PatientSummary: {_truncate_text(getattr(item, 'PatientSummary', ''), 300)}\n"
                f"DiagnosisAnalysis: {_truncate_text(getattr(item, 'DiagnosisAnalysis', ''), 500)}"
            )
        else:
            refs = getattr(item, 'references', [])
            refs_str = "\n".join([f"  - {r}" for r in refs]) if refs else "No specific references listed."
            judgement_entry = (
                f"{i+1}. {getattr(item, 'disease_name', '')}\n"
                f"Correctness: {getattr(item, 'Correctness', '')}\n"
                f"PatientSummary: {getattr(item, 'PatientSummary', '')}\n"
                f"DiagnosisAnalysis: {getattr(item, 'DiagnosisAnalysis', '')}\n"
                f"References:\n{refs_str}"
            )
        judgements_list.append(judgement_entry)

    judgements_str = "\n\n".join(judgements_list)
    if not compact and hasattr(judgements, "reference") and judgements.reference:
        judgements_str += f"\n\n[ReflectionOutput Reference]: {judgements.reference}"
    return judgements_str


def _build_final_prompt(
    present_hpo,
    absent_hpo,
    use_absent_hpo,
    onset,
    sex,
    similar_case_detailed_str,
    tentative_result_str,
    judgements_str,
) -> str:
    prompt_template = prompt_dict["final_diagnosis_prompt"]
    inputs = {
        "present_hpo": present_hpo,
        "absent_hpo": absent_hpo,
        "use_absentHPO": use_absent_hpo,
        "onset": onset if onset else "Unknown",
        "sex": sex if sex else "Unknown",
        "similar_case_detailed": similar_case_detailed_str,
        "tentative_result": tentative_result_str,
        "judgements": judgements_str,
    }
    return build_prompt(prompt_template, inputs)


def _invoke_final_with_retry(llm, prompt: str, attempt_name: str):
    messages = [HumanMessage(content=prompt)]
    retry_wait_seconds = _final_retry_wait_seconds()
    retry_count = 0
    while True:
        try:
            temp_llm = llm.get_temp_llm_with_max_tokens(
                llm.default_max_tokens,
                timeout_seconds=_final_request_timeout_seconds(),
            )
            structured_llm = temp_llm.with_structured_output(DiagnosisOutput)
            return llm.invoke_with_content_filter_retry(
                structured_llm,
                messages,
                context=f"FinalDiagnosis:{attempt_name}",
            )
        except Exception as e:
            if not _is_retryable_final_error(e):
                raise
            retry_count += 1
            print(
                f"[FinalDiagnosis] Retryable Azure error on {attempt_name} prompt "
                f"({type(e).__name__}: {e}). "
                f"Retry #{retry_count} after {retry_wait_seconds:.1f}s."
            )
            time.sleep(retry_wait_seconds)


def createFinalDiagnosis(state: State) -> Optional[DiagnosisOutput]:
    """
    Generate FinalDiagnosis using State, prompt, and DiagnosisOutput
    """
   
    hpo_dict = state.get("hpoDict", {})
    absent_hpo_dict = state.get("absentHpoDict", {}) 
    use_absent_hpo = state.get("use_absentHPO", False)
    similar_case_detailed = state.get("clinicalText", "")
    tentative_result = state.get("tentativeDiagnosis", None)
    judgements = state.get("reflection", None)
    onset=state.get("onset", "Unknown")
    sex=state.get("sex", "Unknown")
    llm = state.get("llm")

    if not llm:
        print("LLM instance not found in state.")
        return None, None

    present_hpo = ", ".join([v for k, v in hpo_dict.items()]) if hpo_dict else ""
    absent_hpo = (
        ", ".join([v for k, v in (absent_hpo_dict or {}).items() if v])
        if use_absent_hpo and absent_hpo_dict
        else ""
    )

    # If memory (similar_case_detailed) is a list, join as string
    if isinstance(similar_case_detailed, list):
        similar_case_detailed_str = "\n".join([str(item) for item in similar_case_detailed])
    else:
        similar_case_detailed_str = str(similar_case_detailed)

    attempts = [
        (
            "full",
            _format_tentative_diagnosis(tentative_result, compact=False),
            _format_reflection(judgements, compact=False),
            similar_case_detailed_str,
        ),
        (
            "compact_without_raw_references",
            _format_tentative_diagnosis(tentative_result, compact=True),
            _format_reflection(judgements, compact=True),
            "",
        ),
        (
            "minimal_tentative_only",
            _format_tentative_diagnosis(tentative_result, compact=True),
            "Reflection details were omitted because a previous final-diagnosis prompt was blocked by the content filter.",
            "",
        ),
    ]

    last_prompt = ""
    for attempt_name, tentative_text, judgement_text, similar_case_text in attempts:
        prompt = _build_final_prompt(
            present_hpo=present_hpo,
            absent_hpo=absent_hpo,
            use_absent_hpo=use_absent_hpo,
            onset=onset,
            sex=sex,
            similar_case_detailed_str=similar_case_text,
            tentative_result_str=tentative_text,
            judgements_str=judgement_text,
        )
        last_prompt = prompt
        try:
            if attempt_name != "full":
                print(f"[FinalDiagnosis] Retrying with {attempt_name} prompt.")
            result = attach_diagnosis_references(
                _invoke_final_with_retry(llm, prompt, attempt_name),
                source_records=(state.get("webresources", []) or []) + (state.get("memory", []) or []),
            )
            return result, prompt
        except Exception as e:
            if _is_content_filter_error(e) and attempt_name != attempts[-1][0]:
                print(f"[FinalDiagnosis] Content filter triggered on {attempt_name} prompt. Retrying with less detail.")
                continue
            if _is_content_filter_error(e):
                print("[FinalDiagnosis] Content filter triggered on all retry prompts. Falling back to tentative diagnosis.")
                if tentative_result is not None and hasattr(tentative_result, "ans"):
                    tentative_result.reference = (
                        (tentative_result.reference or "")
                        + "\n[Fallback] Final diagnosis LLM call was blocked by Azure content filter; "
                        "tentative diagnosis was used as finalDiagnosis."
                    ).strip()
                    return tentative_result, prompt
                return DiagnosisOutput(
                    ans=[],
                    reference="Final diagnosis LLM call was blocked by Azure content filter.",
                ), prompt
            raise

    return None, last_prompt


def createFinalDiagnosisV2(state: State) -> tuple[DiagnosisOutputV2 | None, str]:
    """Generate the final bilingual diagnosis from tentative output and Evidence."""
    tentative = state.get("tentativeDiagnosis")
    reflection = state.get("reflection")
    evidence = list(state.get("evidence", []) or [])
    sources = list(state.get("sources", []) or [])
    llm = state.get("llm")
    if tentative is None or not hasattr(tentative, "ans"):
        return DiagnosisOutputV2(ans=[], evidence=evidence), ""

    tentative_items = [item for item in tentative.ans if isinstance(item, DiagnosisFormatV2)]
    if not tentative_items:
        return DiagnosisOutputV2(ans=[], evidence=evidence), ""

    patient_features = list((state.get("hpoDict", {}) or {}).values())

    def fallback() -> DiagnosisOutputV2:
        answers = []
        # The internal tentative differential may now contain every merged
        # candidate; the public final differential remains capped at five.
        for original in tentative_items[:5]:
            item = original.model_copy(deep=True)
            candidate_items = _candidate_evidence(item.candidate_id, evidence)
            item.description_en, item.description_ja = _fallback_description(
                item.disease_name_en,
                patient_features,
                candidate_items,
            )
            item = _apply_evidence_contract(item, evidence, patient_features, sources)
            answers.append(item)
        return DiagnosisOutputV2(ans=answers, evidence=evidence)

    if not llm:
        return fallback(), ""

    evidence_text = "\n".join(
        f"{item.evidence_id} | candidate={item.candidate_id} | "
        f"source={next((source.identifier or source.title for source in sources if source.source_id == item.source_id), 'unknown')} | "
        f"support={item.support} | "
        f"confidence={item.confidence} | EN={item.claim_en} | JA={item.claim_ja} | "
        f"excerpt={item.excerpt}"
        for item in evidence
    ) or "No validated Evidence"
    reflection_text = "\n".join(
        f"{item.candidate_id} | correctness={item.correctness} | status={item.literature_status} | "
        f"EN={item.diagnosis_analysis_en} | JA={item.diagnosis_analysis_ja} | evidence={item.evidence_ids}"
        for item in (getattr(reflection, "ans", []) or [])
        if isinstance(item, object)
    ) or "No Reflection output"
    tentative_text = "\n".join(
        f"initial_tentative_order={index} | {item.candidate_id} | {item.disease_name_en}"
        for index, item in enumerate(tentative_items, 1)
    )
    merged_candidates = enrich_candidates(state.get("mergedDiseaseCandidates", []) or [])
    candidates_by_id = {candidate.candidate_id: candidate for candidate in merged_candidates}
    tool_support_text = "\n".join(
        f"{item.candidate_id} | consensus_count={getattr(candidates_by_id.get(item.candidate_id), 'consensus_count', 0)} | "
        f"best_tool_rank={getattr(candidates_by_id.get(item.candidate_id), 'best_rank', 'unknown')} | "
        f"tools={', '.join(f'{support.tool}:rank={support.rank},score={support.score}' for support in getattr(candidates_by_id.get(item.candidate_id), 'tool_supports', [])) or 'none'} | "
        f"candidate_phenotypes={', '.join(getattr(candidates_by_id.get(item.candidate_id), 'phenotype_labels', [])[:20]) or 'none'}"
        for item in tentative_items
    )
    prompt = f"""You are the final clinical-genetics reviewer.
Produce a bilingual final differential diagnosis by integrating patient phenotype fit, the diagnostic-tool signals, Reflection, and validated literature Evidence.
Rank candidates primarily by the specificity and overall compatibility of the supplied patient findings with the candidate phenotype profile; use diagnostic-tool rank and cross-tool consensus as supporting prioritization signals, then use Reflection to resolve inclusion or uncertainty.
Literature retrieval success or the number of retrieved papers must not determine candidate rank. Missing or weakly retrieved literature is neutral, not evidence against a candidate. A candidate with correctness=true remains eligible for the top five even when literature_status is not_retrieved; exclude or strongly deprioritize it only for a stated clinical incompatibility or decisive, identity-matched contradictory Evidence.
Every diagnosis must use an existing candidate_id and only existing evidence_ids. Do not introduce disease-specific claims unsupported by the supplied Evidence or candidate phenotype profile.
A candidate may be described as literature-supported only when a validated literature Evidence item supports it. Tool rankings are not citations.
If no usable literature Evidence was retrieved, say that no candidate-matched literature Evidence was available in this run; do not imply that no literature exists. Explain the candidate's phenotype/tool rationale without presenting it as a literature claim.
Keep the English and Japanese descriptions semantically aligned and use the same evidence_ids.
Explain the concrete patient features in relation to the supplied literature claims and identify important uncertainty.
Write a substantive multi-sentence explanation for each diagnosis. Connect patient findings to the supplied candidate phenotype profile and, where available, specific literature claims; explain the clinical interpretation and remaining uncertainty. Do not return only a conclusion such as "supported" or "not supported".
For every literature-grounded statement, place one or more exact inline markers immediately after the supported statement using the format [[EVIDENCE:EVD-...]]. Use only supplied evidence_id values. Use the same evidence markers in the English and Japanese descriptions. Do not use numeric [1] markers; the serializer will convert the internal markers to reference numbers.
Do not invent absent findings or treat an unreported feature as absent.
Do not claim a mechanism, inheritance pattern, or phenotype from general model knowledge unless it is present in the supplied Evidence.
Return at most five diagnoses. The order of the returned items is the final differential order, and their rank fields MUST be exactly 1, 2, ..., N with no gaps or duplicates.
Rank 1 is the most likely final diagnosis. Do not copy any rank from a diagnostic tool or from an earlier provisional output; tool/provisional ranks are not final ranks.
Use tentative order as one prioritization signal, together with phenotype compatibility and tool support. Do not change order merely because one candidate has more literature than another.

PATIENT_HPO: {', '.join((state.get('hpoDict', {}) or {}).values())}
PATIENT_ABSENT_HPO: {', '.join((state.get('absentHpoDict', {}) or {}).values()) if state.get('use_absentHPO') else 'N/A'}
The tentative order is the initial phenotype/tool differential created before literature retrieval; its order is useful, but it is not evidence of literature coverage.
TENTATIVE:\n{tentative_text}
TOOL_AND_CANDIDATE_PHENOTYPE_SUPPORT:\n{tool_support_text}
REFLECTION:\n{reflection_text}
EVIDENCE:\n{evidence_text}"""
    try:
        result = llm.get_structured_llm(FinalDiagnosisGenerationOutput).invoke(
            [HumanMessage(content=prompt)]
        )
        # Validate the generated identity against the authoritative merged
        # candidates, not against DiagnosisFormatV2.  The latter uses
        # ``disease_name_en`` while the validator intentionally expects the
        # canonical candidate's ``disease_name``.
        authoritative_candidates = {
            candidate.candidate_id: candidate
            for candidate in enrich_candidates(
                state.get("mergedDiseaseCandidates", []) or []
            )
        }
        for item in tentative_items:
            authoritative_candidates.setdefault(
                item.candidate_id,
                enrich_candidates([{
                    "candidate_id": item.candidate_id,
                    "disease_name": item.disease_name_en,
                    "OMIM_id": item.OMIM_id,
                }])[0],
            )
        identity_errors = validate_v2_identity_fields(
            result, list(authoritative_candidates.values())
        )
        if len(getattr(result, "ans", []) or []) > 5:
            identity_errors.append("Final diagnosis returned more than five candidates.")
        if identity_errors:
            safe = fallback()
            safe.validation_status = "partial"
            safe.validation_errors = [
                "Final diagnosis generation failed identity validation.",
                *identity_errors,
            ]
            print(f"[FinalDiagnosisV2] identity validation failed: {identity_errors}")
            return safe, prompt
        answers = result.model_copy(deep=True).ans
        for item in answers:
            item.OMIM_id = canonical_omim_id(
                authoritative_candidates[item.candidate_id].OMIM_id
            )
            item = _apply_evidence_contract(item, evidence, patient_features, sources)
        return DiagnosisOutputV2(ans=answers, evidence=evidence), prompt
    except Exception as exc:
        print(f"[FinalDiagnosisV2] structured generation failed: {exc}")
        traceback.print_exc()
        return fallback(), prompt
