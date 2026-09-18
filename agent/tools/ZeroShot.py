from langchain.schema import HumanMessage
from ..state.state_types import ZeroShotFormat, ZeroShotOutput, ZeroShotReasonedOutput, State
from ..llm.prompt import prompt_dict, build_prompt


def createZeroshot(state: State, *, reasoning_sink=None):
    """
    Generate reasoned candidates, returning (ZeroShotOutput, prompt) without reasons.

    Uses present HPO labels, onset and sex; explicit absent HPOs are included only
    when use_absentHPO=True. The LLM always generates ZeroShotReasonedOutput, even
    without a reasoning_sink. If supplied, the callback receives an independent
    model_dump snapshot before normalization; it must not forward reasons to State.
    Returned candidates are new three-field objects, so downstream in-place
    normalization cannot alter the reason snapshot. This function does not normalize.

    Missing HPOs/LLM return (None, None); a None LLM result returns (None, prompt).
    Provider, parsing and callback exceptions propagate. Cache reuse is handled
    by createZeroShotNode, not here. See scripts/evaluation/README.md for persistence.
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
    structured_llm = llm.get_structured_llm(ZeroShotReasonedOutput)
    messages = [HumanMessage(content=prompt)]
    generated = llm.invoke_with_content_filter_retry(
        structured_llm,
        messages,
        context="ZeroShot",
    )
    if generated is None:
        return None, prompt

    # Construct new objects: normalization mutates candidates in place. Neither
    # the pipeline output nor its later mutations may carry/change logged reasons.
    result = ZeroShotOutput(ans=[
        ZeroShotFormat(disease_name=item.disease_name, rank=item.rank, OMIM_id=item.OMIM_id)
        for item in generated.ans
    ])
    if reasoning_sink is not None:
        reasoning_sink(generated.model_dump())
    return result, prompt
