import os
import re
import time
from langchain.schema import HumanMessage
from openai import LengthFinishReasonError
from ..state.state_types import (
    ReflectionFormat,
    ReflectionOutputV2,
    ReflectionFormatV2,
    EvidenceAssessment,
    State,
)
from .candidate_metadata import enrich_candidates
from .diagnosis import is_pubmed_reference
from ..llm.prompt import prompt_dict, build_prompt
from ..llm.llm_wrapper import is_content_filter_error


def _reflection_token_limits() -> list[int]:
    raw_limits = os.getenv("REFLECTION_TOKEN_LIMITS", "12000,16000,20000")
    try:
        limits = [int(value.strip()) for value in raw_limits.split(",") if value.strip()]
    except ValueError:
        limits = []
    return limits or [12000, 16000, 20000]


def _reflection_request_timeout_seconds() -> float:
    return float(os.getenv("REFLECTION_REQUEST_TIMEOUT_SECONDS", "180"))


def _reflection_retry_wait_seconds() -> float:
    return float(os.getenv("REFLECTION_RETRY_WAIT_SECONDS", "20"))


def _is_retryable_reflection_error(error: Exception) -> bool:
    if is_content_filter_error(error):
        return False
    message = str(error).lower()
    retryable_markers = [
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
    return any(marker in message for marker in retryable_markers)


def _invoke_reflection_with_retry(llm, structured_llm, messages, diagnosis_name: str):
    retry_wait_seconds = _reflection_retry_wait_seconds()
    retry_count = 0
    while True:
        try:
            return llm.invoke_with_content_filter_retry(
                structured_llm,
                messages,
                context=f"Reflection:{diagnosis_name}",
            )
        except Exception as e:
            if not _is_retryable_reflection_error(e):
                raise
            retry_count += 1
            print(
                f"[Reflection] Retryable Azure error for {diagnosis_name} "
                f"({type(e).__name__}: {e}). "
                f"Retry #{retry_count} after {retry_wait_seconds:.1f}s."
            )
            time.sleep(retry_wait_seconds)


def format_disease_knowledge(info_list, disease_name):
    """
    InformationItemのリストから、rankに該当するものだけをプロンプト用に整形
    """
    if not info_list:
        return "No disease knowledge available."
    lines = []
    for i, item in enumerate(info_list, 1):
        if item.get("disease_name") == disease_name:
            line = f"[{i}] {item.get('title', '')}\nURL: {item.get('url', '')}\n{item.get('content', '')}\n"
            lines.append(line)
    if not lines:
        return "No disease knowledge available for this rank."
    return "\n".join(lines)


def _normalize_omim_id(value) -> str | None:
    if not value:
        return None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    return f"OMIM:{match.group(0)}"


def _normalize_name(value: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", str(value or ""))
    value = re.sub(r"[^A-Za-z0-9]+", " ", value.upper())
    return re.sub(r"\s+", " ", value).strip()


def _find_merged_candidate(state: State, diagnosis_to_judge):
    diagnosis_omim = _normalize_omim_id(getattr(diagnosis_to_judge, "OMIM_id", None))
    diagnosis_name = _normalize_name(getattr(diagnosis_to_judge, "disease_name", ""))
    merged_candidates = state.get("mergedDiseaseCandidates", []) or []

    for candidate in merged_candidates:
        candidate_omim = _normalize_omim_id(candidate.get("OMIM_id"))
        if diagnosis_omim and candidate_omim == diagnosis_omim:
            return candidate
        if not diagnosis_omim and _normalize_name(candidate.get("disease_name", "")) == diagnosis_name:
            return candidate
    return None


def _format_tool_support(state: State, diagnosis_to_judge) -> str:
    matched_candidate = _find_merged_candidate(state, diagnosis_to_judge)

    lines = [
        f"TentativeDiagnosis rank: {getattr(diagnosis_to_judge, 'rank', 'N/A')}",
    ]
    if not matched_candidate:
        lines.append("Merged tool ranking details: not available for this candidate.")
        return "\n".join(lines)

    lines.append(
        "Merged candidate summary: "
        f"consensus_count={matched_candidate.get('consensus_count', 'N/A')}, "
        f"best_tool_rank={matched_candidate.get('best_rank', 'N/A')}"
    )
    for ranking in matched_candidate.get("tool_rankings", []) or []:
        score = ranking.get("score")
        score_text = f", score={score:.3f}" if isinstance(score, (int, float)) else ""
        note = str(ranking.get("note", "") or "").replace("\n", " ").strip()
        if len(note) > 180:
            note = note[:177].rstrip() + "..."
        note_text = f", note={note}" if note else ""
        lines.append(
            f"- {ranking.get('tool', 'UnknownTool')}: "
            f"rank={ranking.get('rank', 'N/A')}{score_text}{note_text}"
        )
    return "\n".join(lines)


def _has_strong_tool_support(state: State, diagnosis_to_judge) -> bool:
    candidate = _find_merged_candidate(state, diagnosis_to_judge)
    tentative_rank = getattr(diagnosis_to_judge, "rank", None)
    if not candidate:
        return isinstance(tentative_rank, int) and tentative_rank <= 1

    for ranking in candidate.get("tool_rankings", []) or []:
        tool = str(ranking.get("tool", "")).lower()
        rank = ranking.get("rank")
        try:
            rank = int(rank)
        except (TypeError, ValueError):
            continue

        if tool == "pubcasefinder" and rank <= 1:
            return True
        if tool == "phenotypesearch" and rank <= 3:
            return True
        if tool == "gestaltmatcher" and rank <= 2:
            return True

    return isinstance(tentative_rank, int) and tentative_rank <= 1


def _looks_like_literature_gap_false(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "literature mismatch",
        "retrieval mismatch",
        "disease-identity mismatch",
        "literature/disease-identity mismatch",
        "provided medical literature does not",
        "medical literature does not",
        "does not describe",
        "does not actually describe",
        "not describe",
        "insufficient literature",
        "lack of disease-specific",
        "lacks disease-specific",
        "no disease-specific",
        "disease-specific literature",
        "different disease entities",
        "different neurodevelopmental disorders",
    ]
    return any(marker in lowered for marker in markers)


def _has_decisive_contradiction(text: str) -> bool:
    lowered = text.lower()
    negated_markers = [
        "no decisive contradiction",
        "no explicit contradiction",
        "not a decisive contradiction",
        "without decisive contradiction",
    ]
    if any(marker in lowered for marker in negated_markers):
        return False

    markers = [
        "decisive contradiction",
        "clearly incompatible",
        "directly contradict",
        "explicitly absent findings directly contradict",
        "defining phenotype is largely incompatible",
        "core phenotype is clearly incompatible",
    ]
    return any(marker in lowered for marker in markers)


def _apply_tool_supported_reflection_override(result: ReflectionFormat, state: State, diagnosis_to_judge) -> ReflectionFormat:
    if getattr(result, "Correctness", None) is not False:
        return result

    analysis = str(getattr(result, "DiagnosisAnalysis", "") or "")
    if not _looks_like_literature_gap_false(analysis):
        return result
    if not _has_strong_tool_support(state, diagnosis_to_judge):
        return result
    if _has_decisive_contradiction(analysis):
        return result

    result.Correctness = True
    result.DiagnosisAnalysis = (
        analysis.rstrip()
        + "\n\n[Tool-supported override] Correctness was changed from False to True because "
        "the negative judgement was driven mainly by insufficient or mismatched literature, "
        "while this candidate has strong phenotype-based diagnostic tool support and no decisive "
        "clinical contradiction was identified."
    )
    print(f"[Reflection] Tool-supported override applied for {getattr(diagnosis_to_judge, 'disease_name', '')}.")
    return result


def create_reflection(state: State, diagnosis_to_judge):
    prompt_template = prompt_dict["reflection_prompt"]
    
    hpo_dict = state.get("hpoDict", {})
    absent_hpo_dict = state.get("absentHpoDict", {})
    use_absent_hpo = state.get("use_absentHPO", False)
    disease_knowledge_list = state.get("memory", [])
    onset = state.get("onset")
    sex = state.get("sex")
    llm = state.get("llm")

    if not llm:
        print("LLM instance not found in state.")
        return None, None

    diagnosis_name = diagnosis_to_judge.disease_name
    description = diagnosis_to_judge.description
    rank = diagnosis_to_judge.rank
    omim_id = getattr(diagnosis_to_judge, "OMIM_id", None) or "N/A"
    disease_name = diagnosis_to_judge.disease_name

    disease_knowledge_str = format_disease_knowledge(disease_knowledge_list, disease_name) if disease_knowledge_list is not None else ""

    present_hpo = ", ".join([v for k, v in hpo_dict.items()]) if hpo_dict else ""
    absent_hpo = (
        ", ".join([v for k, v in (absent_hpo_dict or {}).items() if v])
        if use_absent_hpo and absent_hpo_dict
        else ""
    )

    inputs = {
        "present_hpo": present_hpo,
        "absent_hpo": absent_hpo,
        "use_absentHPO": use_absent_hpo,
        "onset": onset if onset else "Unknown",
        "sex": sex if sex else "Unknown",
        "diagnosis_to_judge": f"{diagnosis_name} (Rank: {rank})\nOMIM/MONDO identifier: {omim_id}\nDescription: {description}",
        "tool_support": _format_tool_support(state, diagnosis_to_judge),
        "disease_knowledge": disease_knowledge_str
    }
    
    prompt = build_prompt(prompt_template, inputs)
    messages = [HumanMessage(content=prompt)]
    
    # トークン数を段階的に増やして再試行する。
    # 既存ログでは 25,000 での長さ上限到達がなく、出力も数千文字程度のため、
    # gpt-5 系の内部推論分を見込んでも過剰になりにくい範囲へ抑える。
    token_limits = _reflection_token_limits()
    
    for attempt, max_tokens in enumerate(token_limits, 1):
        try:
            print(f"[Reflection] 試行 {attempt}/{len(token_limits)}: max_completion_tokens={max_tokens}")
            
            # 一時的なLLMインスタンスを作成（元のllmは変更しない）
            temp_llm = llm.get_temp_llm_with_max_tokens(
                max_tokens,
                timeout_seconds=_reflection_request_timeout_seconds(),
            )
            structured_llm = temp_llm.with_structured_output(ReflectionFormat)
            
            # 推論実行
            result = _invoke_reflection_with_retry(llm, structured_llm, messages, diagnosis_name)
            
            print(f"[Reflection] 成功 (max_completion_tokens={max_tokens})")
            
            if isinstance(result, dict):
                reflection_result = ReflectionFormat(**result)
            else:
                reflection_result = result
            reflection_result.references = [
                reference
                for reference in reflection_result.references
                if is_pubmed_reference(reference)
            ]
            reflection_result = _apply_tool_supported_reflection_override(
                reflection_result,
                state,
                diagnosis_to_judge,
            )
            return reflection_result, prompt
            
        except LengthFinishReasonError as e:
            print(f"[Reflection] トークン上限到達 (max_completion_tokens={max_tokens})")
            if attempt < len(token_limits):
                print(f"  -> より大きなトークン数で再試行します")
                continue
            else:
                print(f"  -> 最大試行回数に達しました。デフォルト結果を返します。")
                # 最大試行回数に達した場合、保守的な結果を返す
                fallback_result = ReflectionFormat(
                    disease_name=diagnosis_name,
                    Correctness=False,
                    PatientSummary=f"Token limit exceeded for reflection. Present HPO: {present_hpo[:100]}...",
                    DiagnosisAnalysis=f"Reflection could not be completed due to token limit after {len(token_limits)} attempts with max_tokens up to {token_limits[-1]}.",
                    references=[]
                )
                return fallback_result, prompt
                
        except Exception as e:
            print(f"[ERROR] Reflection failed for {diagnosis_name}: {type(e).__name__}: {e}")
            # その他のエラーの場合も保守的な結果を返す
            fallback_result = ReflectionFormat(
                disease_name=diagnosis_name,
                Correctness=False,
                PatientSummary=f"Error during reflection: {present_hpo[:100]}...",
                DiagnosisAnalysis=f"Reflection failed with error: {str(e)}",
                references=[]
            )
            return fallback_result, prompt


def create_reflection_v2(state: State) -> tuple[ReflectionOutputV2, str]:
    """Reflect on candidates using Evidence IDs, with bilingual output."""
    candidates = enrich_candidates(state.get("mergedDiseaseCandidates", []) or [])
    evidence = list(state.get("evidence", []) or [])
    sources = {source.source_id: source for source in state.get("sources", []) or []}
    present_hpo = ", ".join((state.get("hpoDict", {}) or {}).values())
    llm = state.get("llm")

    def direct_profile_overlap(candidate) -> bool:
        def normalize(value: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()

        patient_terms = {normalize(value) for value in (state.get("hpoDict", {}) or {}).values()}
        patient_terms.discard("")
        disease_terms = {
            normalize(value) for value in (candidate.phenotype_labels or [])
        }
        disease_terms.discard("")
        return bool(patient_terms & disease_terms)

    def status_for(candidate):
        items = [item for item in evidence if item.candidate_id == candidate.candidate_id]
        candidate_sources = [
            source for source in sources.values()
            if source.source_type == "literature"
            and candidate.candidate_id in (source.candidate_ids or [])
        ]
        if any(item.support == "contradicts" for item in items) and not any(item.support == "supports" for item in items):
            return "contradictory"
        if any(
            item.support == "supports" and sources.get(item.source_id) and sources[item.source_id].source_type == "literature"
            for item in items
        ):
            return "supported"
        # ``not_retrieved`` describes this run's retrieval outcome, not the
        # existence of literature.  If a candidate-matched source was fetched
        # but yielded no validated Evidence, retrieval succeeded but the
        # available evidence is insufficient.
        return "insufficient" if items or candidate_sources else "not_retrieved"

    def fallback() -> ReflectionOutputV2:
        answers = []
        for candidate in candidates:
            items = [item for item in evidence if item.candidate_id == candidate.candidate_id]
            status = status_for(candidate)
            assessments = [
                EvidenceAssessment(
                    evidence_id=item.evidence_id,
                    relevance="high" if item.evidence_type == "phenotype_overlap" else "low",
                    supports_patient=(
                        "yes" if item.evidence_type == "phenotype_overlap" and item.support == "supports"
                        else "no" if item.evidence_type == "phenotype_mismatch" or item.support == "contradicts"
                        else "unknown"
                    ),
                    reason_en=item.claim_en,
                    reason_ja=item.claim_ja,
                )
                for item in items
            ]
            answers.append(
                ReflectionFormatV2(
                    candidate_id=candidate.candidate_id,
                    disease_name_en=candidate.disease_name,
                    disease_name_ja=candidate.disease_name,
                    # Without an LLM judgment, do not turn retrieval/tool
                    # presence into an automatic positive reflection.
                    correctness=(
                        status != "contradictory"
                        and direct_profile_overlap(candidate)
                    ),
                    patient_summary_en=f"Patient features supplied to the flow: {present_hpo or 'not provided'}.",
                    patient_summary_ja=f"フローに入力された患者所見: {present_hpo or '未提供'}。",
                    diagnosis_analysis_en=(
                        "The candidate is supported by validated literature evidence."
                        if status == "supported"
                        else "The candidate cannot be considered literature-supported yet."
                    ),
                    diagnosis_analysis_ja=(
                        "この候補は検証済みの文献Evidenceにより支持されています。"
                        if status == "supported"
                        else "この候補は現時点では文献支持済みとは判断できません。"
                    ),
                    evidence_assessments=assessments,
                    # Only attach evidence that can support or contradict the
                    # reflection and that comes from a citable literature
                    # source.  Unknown retrieval snippets remain available in
                    # evidence_assessments, but are not exposed as references.
                    evidence_ids=[
                        item.evidence_id
                        for item in items
                        if item.support in {"supports", "contradicts"}
                        and sources.get(item.source_id)
                        and sources[item.source_id].source_type == "literature"
                        and sources[item.source_id].citation_allowed
                    ],
                    literature_status=status,
                )
            )
        return ReflectionOutputV2(ans=answers)

    if not llm or not candidates:
        return fallback(), ""

    evidence_text = "\n".join(
        f"{item.evidence_id} candidate={item.candidate_id} support={item.support} confidence={item.confidence} "
        f"EN={item.claim_en} JA={item.claim_ja} excerpt={item.excerpt}"
        for item in evidence
    ) or "No validated Evidence"
    candidate_text = "\n".join(
        f"{candidate.candidate_id}: {candidate.disease_name}; "
        f"OMIM={candidate.OMIM_id or 'unknown'}; "
        f"consensus_count={candidate.consensus_count}; best_rank={candidate.best_rank}; "
        f"tools={', '.join(f'{support.tool}:rank={support.rank},score={support.score}' for support in candidate.tool_supports) or 'none'}; "
        f"database_phenotypes={', '.join(candidate.phenotype_labels[:20]) or 'none'}"
        for candidate in candidates
    )
    prompt = f"""You are reviewing a rare-disease differential diagnosis.
Judge whether each candidate should remain plausible in the differential, not whether it is confirmed.
`correctness` is a binary retain-in-the-differential decision, not a literature-retrieval flag and not a claim of confirmed diagnosis. Set it true only when the patient has a meaningful, candidate-specific phenotype match (one distinctive match or multiple matching findings) and no decisive contradiction. Set it false when the profile shows only generic overlap, a clear phenotype mismatch, a decisive contradiction, or tool ranking is the only positive signal. A candidate returned by a diagnostic tool is a hypothesis to assess, not proof of fit. Do not mark every candidate true by default; compare candidates against the same patient findings and distinguish strong, specific matches from broad neurodevelopmental overlap.
Missing or weakly retrieved literature is uncertainty, never evidence against a candidate by itself. Do not set correctness=false solely because no source or Evidence was retrieved; use the phenotype profile and explicitly supplied absent findings to judge plausibility.
An explicitly absent finding may count against a candidate only when it is supplied in the absent-phenotype section; unreported findings are unknown.
Tool rankings, consensus, and database phenotypes are auxiliary candidate-evaluation information, not literature citations.
For every supplied candidate return ReflectionFormatV2 with English and Japanese fields.
Do not drop a candidate solely because literature is insufficient; use `insufficient` or `not_retrieved` when appropriate. `not_retrieved` means no candidate-matched source was obtained in this run, not that no literature exists.
Use only supplied candidate_id and evidence_id values. Keep evidence_ids identical across the bilingual analysis.
Literature status must be supported only when a literature Evidence item supports the candidate. For each EvidenceAssessment, `supports_patient=yes` only when the claim describes a phenotype actually present in the patient; a disease-identity statement, mechanism, or a different/unique finding is not a patient match. Use `no` for a clear mismatch and `unknown` when the claim does not permit comparison.
Explain in prose how the patient features relate to the disease claims in the Evidence. Do not treat an article about the candidate as support for every patient feature or for the candidate's overall fit.
Do not create a separate match, impact, or limitation object; state uncertainty in the analysis text.
Do not use a general medical-knowledge statement as if it came from a cited literature source.

PATIENT_FEATURES: {present_hpo or 'not provided'}
PATIENT_ABSENT_FEATURES: {', '.join((state.get('absentHpoDict', {}) or {}).values()) if state.get('use_absentHPO') else 'not provided'}
CANDIDATES:\n{candidate_text}
EVIDENCE:\n{evidence_text}"""
    try:
        result = llm.get_structured_llm(ReflectionOutputV2).invoke([HumanMessage(content=prompt)])
        allowed_candidates = {candidate.candidate_id for candidate in candidates}
        allowed_evidence = {item.evidence_id for item in evidence}
        fallback_by_candidate = {
            item.candidate_id: item
            for item in fallback().ans
        }
        answers_by_candidate = {}
        for item in getattr(result, "ans", []) or []:
            if item.candidate_id not in allowed_candidates:
                continue
            candidate_evidence = {
                evidence_item.evidence_id
                for evidence_item in evidence
                if evidence_item.candidate_id == item.candidate_id
            }
            item.evidence_ids = [
                value for value in item.evidence_ids
                if value in allowed_evidence and value in candidate_evidence
            ]
            item.evidence_assessments = [
                assessment
                for assessment in item.evidence_assessments
                if assessment.evidence_id in candidate_evidence
            ]
            answers_by_candidate[item.candidate_id] = item

        # Reflection must cover every merged candidate.  If the model omits a
        # candidate, keep the flow auditable by adding a deterministic
        # insufficient/not_retrieved assessment instead of silently dropping it.
        answers = [
            answers_by_candidate.get(candidate.candidate_id)
            or fallback_by_candidate[candidate.candidate_id]
            for candidate in candidates
        ]
        return ReflectionOutputV2(ans=answers), prompt
    except Exception as exc:
        print(f"[ReflectionV2] structured generation failed: {exc}")
        return fallback(), prompt
