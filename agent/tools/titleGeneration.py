"""Patient title generation utilities.

This module is intentionally independent from the LangGraph wiring so it can be
called from a graph node, an API endpoint, or a batch script.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


DEFAULT_MAX_FEATURES = 12
DEFAULT_MAX_TITLE_CHARS = 80
DEFAULT_MAX_COMPLETION_TOKENS = 120


class PatientTitleFeature(BaseModel):
    id: str = ""
    label: str = ""


class PatientTitleInput(BaseModel):
    patient_id: Optional[str] = None
    sex: Optional[str] = None
    age_iso8601duration: Optional[str] = None
    onset: Optional[str] = None
    present_features: list[PatientTitleFeature] = Field(default_factory=list)
    absent_features: list[PatientTitleFeature] = Field(default_factory=list)
    image_exists: Optional[bool] = None
    clinical_text: Optional[str] = None
    max_features: int = DEFAULT_MAX_FEATURES
    max_title_chars: int = DEFAULT_MAX_TITLE_CHARS


class PatientTitleOutput(BaseModel):
    title: str
    status: Literal["success", "fallback", "skipped", "error"]
    used_llm: bool = False
    error_message: Optional[str] = None
    prompt: Optional[str] = None


def sex_to_jp(sex: Optional[str]) -> str:
    value = (sex or "").upper()
    if value == "MALE":
        return "男性"
    if value == "FEMALE":
        return "女性"
    if value in {"UNKNOWN", "UNKNOWN_SEX", ""}:
        return "不明"
    if value == "OTHER_SEX":
        return "その他"
    return str(sex or "不明")


def normalize_one_line(text: str, max_chars: int = DEFAULT_MAX_TITLE_CHARS) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    normalized = normalized.strip("\"'「」")
    if max_chars > 0 and len(normalized) > max_chars:
        return normalized[: max_chars - 1].rstrip() + "…"
    return normalized


def _feature_label(feature: PatientTitleFeature) -> str:
    return feature.label or feature.id


def fallback_patient_title(input_data: PatientTitleInput) -> str:
    patient_id = input_data.patient_id or "unknown"
    sex_jp = sex_to_jp(input_data.sex)
    image_text = "不明"
    if input_data.image_exists is True:
        image_text = "あり"
    elif input_data.image_exists is False:
        image_text = "なし"
    title = (
        f"患者{patient_id}（{sex_jp}）: "
        f"所見{len(input_data.present_features)}件／"
        f"除外{len(input_data.absent_features)}件・画像{image_text}"
    )
    return normalize_one_line(title, input_data.max_title_chars)


def build_patient_title_prompt(input_data: PatientTitleInput) -> str:
    present_labels = [
        _feature_label(feature)
        for feature in input_data.present_features[: input_data.max_features]
    ]
    absent_labels = [
        _feature_label(feature)
        for feature in input_data.absent_features[: input_data.max_features]
    ]

    patient_id = input_data.patient_id or "不明"
    sex_jp = sex_to_jp(input_data.sex)
    age_text = input_data.age_iso8601duration or "不明"
    onset_text = input_data.onset or "不明"
    if input_data.image_exists is True:
        image_text = "あり"
    elif input_data.image_exists is False:
        image_text = "なし"
    else:
        image_text = "不明"
    clinical_text = input_data.clinical_text or "なし"

    return f"""あなたは臨床遺伝の専門家です。以下の患者情報を「1行の日本語タイトル」として要約してください。

制約:
- 1行のみ（改行禁止）
- {input_data.max_title_chars}文字以内を目安
- できるだけ具体的（重要所見は最大3つまで）
- 不明な情報は無理に補わない
- 疾患名は入力に明示されていない限り推測しない
- 出力はタイトルだけ（前置き・箇条書き・説明不要）

患者ID: {patient_id}
性別: {sex_jp}
年齢(最終受診時): {age_text}
発症時期: {onset_text}
画像: {image_text}
所見(あり)件数: {len(input_data.present_features)}
所見(なし/除外)件数: {len(input_data.absent_features)}
主要所見(あり): {", ".join(present_labels) if present_labels else "なし"}
主要所見(なし/除外): {", ".join(absent_labels) if absent_labels else "なし"}
補足テキスト: {clinical_text}
"""


def _invoke_llm(
    llm: Any,
    prompt: str,
    max_completion_tokens: int,
    timeout_seconds: Optional[float],
) -> str:
    runnable = llm
    if hasattr(llm, "get_temp_llm_with_max_tokens"):
        runnable = llm.get_temp_llm_with_max_tokens(
            max_completion_tokens,
            timeout_seconds=timeout_seconds,
        )

    if hasattr(llm, "invoke_with_content_filter_retry"):
        msg = llm.invoke_with_content_filter_retry(
            runnable,
            prompt,
            context="PatientTitle",
            retry_count=1,
        )
    elif hasattr(runnable, "invoke"):
        msg = runnable.invoke(prompt)
    elif callable(runnable):
        msg = runnable(prompt)
    else:
        raise TypeError("llm must be callable or expose invoke().")

    return msg.content if hasattr(msg, "content") else str(msg)


def _load_llm(model_name: str) -> Any:
    from agent.llm.azure_llm_instance import get_llm_instance

    return get_llm_instance(model_name)


def generate_patient_title(
    input_data: PatientTitleInput | dict[str, Any],
    llm: Any = None,
    model_name: str = "gpt-4o",
    use_llm: bool = True,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    timeout_seconds: Optional[float] = 20.0,
    include_prompt: bool = False,
) -> PatientTitleOutput:
    """Generate a Japanese one-line patient title.

    The function is fail-soft by design: LLM errors return a fallback title
    instead of raising, so callers can safely use it near the end of a job.
    """
    parsed_input = (
        input_data
        if isinstance(input_data, PatientTitleInput)
        else PatientTitleInput(**input_data)
    )
    fallback = fallback_patient_title(parsed_input)

    if not parsed_input.present_features and not parsed_input.clinical_text:
        return PatientTitleOutput(title=fallback, status="skipped", used_llm=False)

    prompt = build_patient_title_prompt(parsed_input)
    if not use_llm:
        return PatientTitleOutput(
            title=fallback,
            status="fallback",
            used_llm=False,
            prompt=prompt if include_prompt else None,
        )

    try:
        resolved_llm = llm or _load_llm(model_name)
        raw_title = _invoke_llm(
            resolved_llm,
            prompt,
            max_completion_tokens=max_completion_tokens,
            timeout_seconds=timeout_seconds,
        )
        title = normalize_one_line(raw_title, parsed_input.max_title_chars)
        if not title:
            return PatientTitleOutput(
                title=fallback,
                status="fallback",
                used_llm=True,
                error_message="LLM returned an empty title.",
                prompt=prompt if include_prompt else None,
            )
        return PatientTitleOutput(
            title=title,
            status="success",
            used_llm=True,
            prompt=prompt if include_prompt else None,
        )
    except Exception as exc:
        return PatientTitleOutput(
            title=fallback,
            status="error",
            used_llm=False,
            error_message=f"{type(exc).__name__}: {exc}",
            prompt=prompt if include_prompt else None,
        )


def _features_from_dict(hpo_dict: Optional[dict[str, str]]) -> list[PatientTitleFeature]:
    return [
        PatientTitleFeature(id=str(hpo_id), label=str(label or ""))
        for hpo_id, label in (hpo_dict or {}).items()
    ]


def _features_from_ids(hpo_ids: Optional[list[str]]) -> list[PatientTitleFeature]:
    return [PatientTitleFeature(id=str(hpo_id), label="") for hpo_id in (hpo_ids or [])]


def patient_title_input_from_state(state: dict[str, Any]) -> PatientTitleInput:
    image_path = state.get("imagePath")
    present_features = _features_from_dict(state.get("hpoDict")) or _features_from_ids(state.get("hpoList"))
    absent_features = _features_from_dict(state.get("absentHpoDict")) or _features_from_ids(state.get("absentHpoList"))
    return PatientTitleInput(
        patient_id=state.get("patient_id"),
        sex=state.get("sex"),
        age_iso8601duration=state.get("age_iso8601duration") or state.get("age"),
        onset=state.get("onset"),
        present_features=present_features,
        absent_features=absent_features,
        image_exists=bool(image_path) if image_path is not None else None,
        clinical_text=state.get("clinicalText"),
    )


def generate_patient_title_from_state(
    state: dict[str, Any],
    use_llm: bool = True,
    model_name: str = "gpt-4o",
    include_prompt: bool = False,
) -> PatientTitleOutput:
    return generate_patient_title(
        patient_title_input_from_state(state),
        llm=state.get("llm"),
        model_name=model_name,
        use_llm=use_llm,
        include_prompt=include_prompt,
    )


def patient_title_input_from_phenopacket(
    phenopacket: dict[str, Any],
    image_exists: Optional[bool] = None,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> PatientTitleInput:
    subject = phenopacket.get("subject") or {}
    age = ((subject.get("timeAtLastEncounter") or {}).get("age") or {}).get(
        "iso8601duration"
    )
    present_features: list[PatientTitleFeature] = []
    absent_features: list[PatientTitleFeature] = []

    for feature in phenopacket.get("phenotypicFeatures", []) or []:
        term = feature.get("type") or {}
        title_feature = PatientTitleFeature(
            id=str(term.get("id") or ""),
            label=str(term.get("label") or ""),
        )
        if not title_feature.id and not title_feature.label:
            continue
        if feature.get("excluded", False):
            absent_features.append(title_feature)
        else:
            present_features.append(title_feature)

    return PatientTitleInput(
        patient_id=str(subject.get("id") or phenopacket.get("id") or ""),
        sex=subject.get("sex"),
        age_iso8601duration=str(age) if age else None,
        present_features=present_features,
        absent_features=absent_features,
        image_exists=image_exists,
        max_features=max_features,
    )
