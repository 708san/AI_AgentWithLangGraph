from langchain.schema import HumanMessage
from ..state.state_types import ZeroShotFormat, ZeroShotOutput, ZeroShotReasonedOutput, State
from ..llm.prompt import prompt_dict, build_prompt


def build_zeroshot_prompt(state: State) -> str | None:
    """Build the current Zero-shot prompt from the supplied agent state."""
    hpo_dict = state.get("hpoDict", {})
    if not hpo_dict:
        return None
    absent_hpo_dict = state.get("absentHpoDict", {})
    use_absent_hpo = state.get("use_absentHPO", False)
    onset = state.get("onset")
    sex = state.get("sex")
    present_hpo = ", ".join([value for value in hpo_dict.values() if value])
    absent_hpo = (
        ", ".join([value for value in (absent_hpo_dict or {}).values() if value])
        if use_absent_hpo and absent_hpo_dict
        else ""
    )
    return build_prompt(
        prompt_dict["zero-shot-diagnosis-prompt"],
        {
            "present_hpo": present_hpo,
            "absent_hpo": absent_hpo,
            "use_absentHPO": use_absent_hpo,
            "onset": onset if onset else "Unknown",
            "sex": sex if sex else "Unknown",
        },
    )


def createZeroshot(state: State, *, reasoning_sink=None):
    """
    Generate reasoned Zero-shot candidates and return a reason-free snapshot.

    ``reasoning_sink`` receives the independent model output for audit logging;
    the returned ``ZeroShotOutput`` intentionally excludes ``selection_reason``.
    Explicitly absent HPO terms are included only when ``use_absentHPO=True``.
    """
    # Zero-shot can use a separate reasoning profile from the rest of the graph.
    llm = state.get("zero_shot_llm") or state.get("llm")

    prompt = build_zeroshot_prompt(state)
    if prompt is None or not llm:
        return None, prompt

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
