import os
import time
from langchain.schema import HumanMessage
from openai import LengthFinishReasonError
from ..state.state_types import ReflectionFormat, State
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
        "diagnosis_to_judge": f"{diagnosis_name} (Rank: {rank})\nDescription: {description}",
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
                return ReflectionFormat(**result), prompt
            return result, prompt
            
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
