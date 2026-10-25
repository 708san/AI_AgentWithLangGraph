from langchain.schema import HumanMessage
from ..state.state_types import ZeroShotFormat, ZeroShotOutput, ZeroShotReasonedOutput, State
from ..llm.prompt import prompt_dict, build_prompt


def createZeroshot(state: State, *, reasoning_sink=None):
    """
    Generate reasoned Zero-shot candidates and return a reason-free snapshot.

    ``reasoning_sink`` receives the independent model output for audit logging;
    the returned ``ZeroShotOutput`` intentionally excludes ``selection_reason``.
    Explicitly absent HPO terms are included only when ``use_absentHPO=True``.
    """
    hpo_dict = state.get("hpoDict", {})
    absent_hpo_dict = state.get("absentHpoDict", {})
    use_absent_hpo = state.get("use_absentHPO", False)
    onset = state.get("onset")
    sex = state.get("sex")
    llm = state.get("llm")

    if not hpo_dict or not llm:
        return None, None

    present_hpo = ", ".join([v for k, v in hpo_dict.items() if v])
    absent_hpo = (
        ", ".join([v for k, v in (absent_hpo_dict or {}).items() if v])
        if use_absent_hpo and absent_hpo_dict
        else ""
    )

    prompt = build_prompt(
        prompt_dict["zero-shot-diagnosis-prompt"],
        {
            "present_hpo": present_hpo,
            "absent_hpo": absent_hpo,
            "use_absentHPO": use_absent_hpo,
            "onset": onset if onset else "Unknown",
            "sex": sex if sex else "Unknown"
        }
    )

    # structured_llmを使う場合
    # Reasons are generated in a separate schema and copied to the optional
    # sink before any downstream normalization. They never enter State.
    structured_llm = llm.get_structured_llm(ZeroShotReasonedOutput)
    messages = [HumanMessage(content=prompt)]
    generated = llm.invoke_with_content_filter_retry(
        structured_llm,
        messages,
        context="ZeroShot",
    )
    if generated is None:
        return None, prompt

    if reasoning_sink is not None:
        reasoning_sink(generated.model_dump())
    result = ZeroShotOutput(ans=[
        ZeroShotFormat(
            disease_name=item.disease_name,
            rank=item.rank,
            OMIM_id=item.OMIM_id,
        )
        for item in generated.ans
    ])
    return result, prompt
