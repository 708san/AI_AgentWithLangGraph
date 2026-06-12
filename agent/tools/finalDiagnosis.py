import os
import time
from langchain.schema import HumanMessage
from typing_extensions import Optional
from ..state.state_types import State, DiagnosisOutput
from ..llm.prompt import prompt_dict, build_prompt
from ..llm.llm_wrapper import is_content_filter_error


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
                f"{i+1}. {item.disease_name} (Rank: {item.rank}, OMIM: {item.OMIM_id or 'N/A'})"
            )
        else:
            lines.append(
                f"{i+1}. {item.disease_name} (Rank: {item.rank})\n"
                f"Description: {item.description}\n"
                f"Reference: {getattr(item, 'reference', '')}"
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
            result = _invoke_final_with_retry(llm, prompt, attempt_name)
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
