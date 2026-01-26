from langchain.schema import HumanMessage
from openai import LengthFinishReasonError
from ..state.state_types import ReflectionFormat, State
from ..llm.prompt import prompt_dict, build_prompt


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
    absent_hpo = ", ".join([v for k, v in (absent_hpo_dict or {}).items()]) if absent_hpo_dict else ""

    inputs = {
        "present_hpo": present_hpo,
        "absent_hpo": absent_hpo,
        "onset": onset if onset else "Unknown",
        "sex": sex if sex else "Unknown",
        "diagnosis_to_judge": f"{diagnosis_name} (Rank: {rank})\nDescription: {description}",
        "disease_knowledge": disease_knowledge_str
    }
    
    prompt = build_prompt(prompt_template, inputs)
    messages = [HumanMessage(content=prompt)]
    
    # トークン数を段階的に増やして再試行
    token_limits = [25000, 35000, 50000]
    
    for attempt, max_tokens in enumerate(token_limits, 1):
        try:
            print(f"[Reflection] 試行 {attempt}/{len(token_limits)}: max_completion_tokens={max_tokens}")
            
            # 動的にmax_completion_tokensを設定したLLMを取得
            from langchain_openai import AzureChatOpenAI
            
            # 元のLLMの設定を取得
            base_llm = llm.llm
            
            # 新しいmax_completion_tokensでLLMを再構築
            llm_params = {
                "azure_endpoint": base_llm._client._base_url.host,
                "api_key": base_llm.openai_api_key,
                "deployment_name": base_llm.deployment_name,
                "api_version": base_llm.openai_api_version,
                "model_kwargs": {
                    "extra_body": {
                        "max_completion_tokens": max_tokens,
                        "verbosity": "medium",
                        "reasoning_effort": "none"
                    }
                }
            }
            
            temp_llm = AzureChatOpenAI(**llm_params)
            structured_llm = temp_llm.with_structured_output(ReflectionFormat)
            
            result = structured_llm.invoke(messages)
            
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