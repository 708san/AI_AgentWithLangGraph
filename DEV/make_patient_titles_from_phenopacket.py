#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# プロジェクトルートを import パスに追加（DEV/ からでも agent/ を読めるようにする）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from agent.llm.azure_llm_instance import get_llm_instance  # noqa: E402


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def iter_json_files(root_dir: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".json"):
                paths.append(os.path.join(dirpath, name))
    paths.sort()
    return paths


def sex_to_jp(sex: str) -> str:
    s = (sex or "").upper()
    if s == "MALE":
        return "男性"
    if s == "FEMALE":
        return "女性"
    if s in {"UNKNOWN", "UNKNOWN_SEX"}:
        return "不明"
    if s == "OTHER_SEX":
        return "その他"
    return "不明"


def extract_present_absent_features(phenopacket: Dict[str, Any]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    present: List[Dict[str, str]] = []
    absent: List[Dict[str, str]] = []

    for feature in phenopacket.get("phenotypicFeatures", []) or []:
        type_obj = feature.get("type") or {}
        hpo_id = type_obj.get("id") or ""
        label = type_obj.get("label") or ""
        if not hpo_id and not label:
            continue

        item = {"id": str(hpo_id), "label": str(label)}
        if feature.get("excluded", False):
            absent.append(item)
        else:
            present.append(item)

    return present, absent


def extract_age_iso8601(phenopacket: Dict[str, Any]) -> Optional[str]:
    subject = phenopacket.get("subject") or {}
    tlae = subject.get("timeAtLastEncounter") or {}
    age = tlae.get("age") or {}
    iso = age.get("iso8601duration")
    return str(iso) if iso else None


def extract_onset_labels(phenopacket: Dict[str, Any], limit: int = 3) -> List[str]:
    labels: List[str] = []
    for feature in phenopacket.get("phenotypicFeatures", []) or []:
        onset = feature.get("onset") or {}
        oc = onset.get("ontologyClass") or {}
        label = oc.get("label")
        if label:
            labels.append(str(label))
        if len(labels) >= limit:
            break
    # 重複除去（順序保持）
    seen = set()
    uniq: List[str] = []
    for x in labels:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def has_image(image_dir: Optional[str], image_key: str) -> bool:
    if not image_dir:
        return False
    key = str(image_key)
    for ext in IMAGE_EXTS:
        if os.path.exists(os.path.join(image_dir, key + ext)):
            return True
    hits = glob.glob(os.path.join(image_dir, key + ".*"))
    return len(hits) > 0


def normalize_one_line(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t


def build_patient_summary_prompt_jp(
    patient_id: str,
    sex_jp: str,
    age_iso8601: Optional[str],
    onset_labels: List[str],
    present: List[Dict[str, str]],
    absent: List[Dict[str, str]],
    image_exists: bool,
    max_features: int,
) -> str:
    present_labels = [x.get("label") or x.get("id") for x in present][:max_features]
    absent_labels = [x.get("label") or x.get("id") for x in absent][:max_features]

    age_txt = age_iso8601 or "不明"
    onset_txt = "、".join(onset_labels) if onset_labels else "不明"
    img_txt = "あり" if image_exists else "なし"

    return f"""あなたは臨床遺伝の専門家です。以下の患者情報を「1行の日本語タイトル」として要約してください。

制約:
- 1行のみ（改行禁止）
- 80文字以内を目安
- できるだけ具体的（重要所見は最大3つまで）
- 不明な情報は無理に補わない
- 出力はタイトルだけ（前置き・箇条書き・説明不要）

患者ID: {patient_id}
性別: {sex_jp}
年齢(最終受診時): {age_txt}
発症時期(所見から抽出): {onset_txt}
画像: {img_txt}
所見(あり)件数: {len(present)}
所見(なし/除外)件数: {len(absent)}
主要所見(あり): {", ".join(present_labels) if present_labels else "なし"}
主要所見(なし/除外): {", ".join(absent_labels) if absent_labels else "なし"}
"""


def generate_title_with_llm(model_name: str, prompt: str, max_completion_tokens: int = 120) -> str:
    llm_wrapper = get_llm_instance(model_name)
    temp_llm = llm_wrapper.get_temp_llm_with_max_tokens(max_completion_tokens)
    msg = temp_llm.invoke(prompt)
    content = msg.content if hasattr(msg, "content") else str(msg)
    return normalize_one_line(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Japanese patient-title TSV from Phenopacket JSON files (LLM-based).")
    parser.add_argument("--phenopacket_dir", required=True, help="Root dir that contains Phenopacket JSON files.")
    parser.add_argument("--image_dir", default=None, help="Dir that contains patient images (optional).")
    parser.add_argument("--out_tsv", required=True, help="Output TSV path.")
    parser.add_argument("--model", type=str, default="gpt-4o", choices=["gpt-4o", "gpt-5-1", "gpt-5-2"], help="Azure OpenAI model name.")
    parser.add_argument("--max_features", type=int, default=12, help="Max number of present/absent features to show to the LLM.")
    parser.add_argument("--dry_run", action="store_true", help="Do not call LLM; output rule-based title instead.")
    args = parser.parse_args()

    json_paths = iter_json_files(args.phenopacket_dir)
    if not json_paths:
        raise SystemExit(f"No .json files found under: {args.phenopacket_dir}")

    with open(args.out_tsv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "patient_id",
                "title",
                "sex",
                "age_iso8601duration",
                "present_hpo_count",
                "absent_hpo_count",
                "image_exists",
                "phenopacket_path",
            ],
            delimiter="\t",
        )
        writer.writeheader()

        for path in json_paths:
            with open(path, "r", encoding="utf-8") as f_in:
                phenopacket = json.load(f_in)

            subject = phenopacket.get("subject", {}) or {}
            patient_id = str(subject.get("id") or phenopacket.get("id") or os.path.basename(path))
            sex_raw = str(subject.get("sex") or "UNKNOWN_SEX")
            sex_jp = sex_to_jp(sex_raw)

            age_iso = extract_age_iso8601(phenopacket)
            onset_labels = extract_onset_labels(phenopacket, limit=3)
            present, absent = extract_present_absent_features(phenopacket)

            image_ok = has_image(args.image_dir, patient_id)

            prompt = build_patient_summary_prompt_jp(
                patient_id=patient_id,
                sex_jp=sex_jp,
                age_iso8601=age_iso,
                onset_labels=onset_labels,
                present=present,
                absent=absent,
                image_exists=image_ok,
                max_features=args.max_features,
            )

            if args.dry_run:
                title = f"患者{patient_id}（{sex_jp}）: 所見{len(present)}件／除外{len(absent)}件・画像{'あり' if image_ok else 'なし'}"
            else:
                try:
                    title = generate_title_with_llm(args.model, prompt, max_completion_tokens=120)
                except Exception as e:
                    # 失敗時はルールベースにフォールバック
                    title = f"患者{patient_id}（{sex_jp}）: 所見{len(present)}件／除外{len(absent)}件・画像{'あり' if image_ok else 'なし'}"
                    title = title + f"（要約失敗: {type(e).__name__}）"

            writer.writerow(
                {
                    "patient_id": patient_id,
                    "title": title,
                    "sex": sex_raw,
                    "age_iso8601duration": age_iso or "",
                    "present_hpo_count": str(len(present)),
                    "absent_hpo_count": str(len(absent)),
                    "image_exists": "1" if image_ok else "0",
                    "phenopacket_path": path,
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())