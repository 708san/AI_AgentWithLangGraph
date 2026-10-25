import os
import numpy as np
import faiss
import json
import re
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import Optional
from ..state.state_types import State,ZeroShotOutput
from .candidate_metadata import (
    canonical_omim_id,
    known_disease_aliases,
    known_disease_name,
    names_compatible,
)

load_dotenv()

# 設定
tenant = "dbcls"
region = "japaneast"
model = "text-embedding-3-large"
deployment_name = f"{region}-{model}"
endpoint = f"https://{tenant}-{region}.openai.azure.com/"
api_key = os.getenv(f"AZURE_{tenant.upper()}_{region.upper()}")
if not api_key:
    raise RuntimeError(f"AZURE_{tenant.upper()}_{region.upper()} is not set in .env")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-05-01-preview"
)

# インデックスとマッピングファイルのパス
INDEX_BASE = os.path.join(os.path.dirname(__file__), "../data/DataForOmimMapping/DataForOmimMapping")
INDEX_BIN = INDEX_BASE + ".bin"
INDEX_JSON = INDEX_BASE + ".json"
OMIM_MAPPING_JSON = os.path.join(os.path.dirname(__file__), "../data/DataForOmimMapping/omim_mapping.json")

def extract_omim_number(omim_id_str: any) -> Optional[str]:
    """OMIM ID文字列から数字部分のみを抽出する"""
    if not omim_id_str or not isinstance(omim_id_str, (str, int)):
        return None
    
    match = re.search(r'\d+', str(omim_id_str))
    return match.group(0) if match else None

# インデックスとマッピングのロード
faiss_index = faiss.read_index(INDEX_BIN)
with open(INDEX_JSON, encoding="utf-8") as f:
    index_map = json.load(f)
with open(OMIM_MAPPING_JSON, encoding="utf-8") as f:
    original_omim_mapping = json.load(f)

# 数字IDをキーとする新しい検索用マッピングを作成
omim_mapping_by_number = {
    extract_omim_number(key): value 
    for key, value in original_omim_mapping.items() 
    if extract_omim_number(key)
}

def normalize_pcf_results(state: State) -> list:
    """
    Stateを受け取り、その中のPCFの結果リストを正規化する。
    """
    pcf_results = state.get("pubCaseFinder", [])
    for result in pcf_results:
        omim_id_num = extract_omim_number(result.get("omim_id"))
        if omim_id_num:
            result["omim_id"] = canonical_omim_id(omim_id_num)
        if omim_id_num and omim_id_num in omim_mapping_by_number:
            result["disease_name"] = omim_mapping_by_number[omim_id_num]
    return pcf_results


def normalize_gestalt_results(state: State) -> list:
    """
    Stateを受け取り、その中のGestaltMatcherの結果リストを正規化する。
    """
    gestalt_results = state.get("GestaltMatcher", [])
    for result in gestalt_results:
        omim_id_num = extract_omim_number(result.get("omim_id"))
        if omim_id_num:
            result["omim_id"] = canonical_omim_id(omim_id_num)
        if omim_id_num and omim_id_num in omim_mapping_by_number:
            # GestaltMatcherの出力キーに合わせて 'syndrome_name' を更新
            result["syndrome_name"] = omim_mapping_by_number[omim_id_num]
    return gestalt_results

def normalize_zeroshot_results(state: State) -> Optional[ZeroShotOutput]:
    """
    Stateを受け取り、その中のZeroShotOutputを正規化し、OMIM IDを付与し、重複を排除する。
    """
    zeroshot_output = state.get("zeroShotResult")
    if not zeroshot_output or not zeroshot_output.ans:
        return zeroshot_output

    unique_omim_ids = set()
    normalized_ans = []
    validation = []

    for diag in zeroshot_output.ans:
        original_name = diag.disease_name
        original_omim = diag.OMIM_id
        cleaned_name = diag.disease_name.strip().strip('*')
        cleaned_name = re.sub(r'\s*\(.*\)\s*', '', cleaned_name).strip()
        disease_name_upper = cleaned_name.upper()
        supplied_id = canonical_omim_id(original_omim)
        known_label = known_disease_name(supplied_id)
        status = "unconfirmed"
        reason = "No local disease identity could be confirmed."
        resolved_id = supplied_id
        resolved_label = cleaned_name

        if supplied_id and known_label:
            if names_compatible(cleaned_name, known_label, known_disease_aliases(supplied_id)):
                status = "match" if cleaned_name.upper() == known_label.upper() else "normalized"
                reason = "Disease name and OMIM identifier are compatible."
                resolved_label = known_label
            else:
                status = "mismatch"
                reason = f"OMIM identifier maps to {known_label!r}, not the generated disease name."
                # Do not replace an unrelated disease merely because an
                # embedding happens to return a nearby label.
                resolved_id = None
        elif not supplied_id and cleaned_name:
            try:
                omim_id, omim_label, sim = disease_normalize(disease_name_upper)
            except Exception as exc:
                omim_id, omim_label, sim = None, None, 0.0
                reason = f"Normalization lookup failed: {type(exc).__name__}."
            candidate_id = canonical_omim_id(omim_id)
            if candidate_id and omim_label and names_compatible(
                cleaned_name,
                omim_label,
                known_disease_aliases(candidate_id),
            ):
                resolved_id = candidate_id
                resolved_label = omim_label
                status = "normalized"
                reason = f"Unique local normalization match (similarity={sim:.2f})."
            else:
                status = "unconfirmed"
                reason = "No compatible local name/OMIM match was confirmed; candidate retained without an assigned OMIM ID."
        elif supplied_id:
            reason = "OMIM identifier is not present in the local database; candidate retained as unconfirmed."

        duplicate_key = resolved_id or cleaned_name.upper()
        if status != "mismatch" and duplicate_key in unique_omim_ids:
            validation.append({
                "disease_name": original_name,
                "OMIM_id": original_omim,
                "status": "duplicate",
                "reason": "The normalized disease identity was already emitted by another candidate.",
                "excluded": True,
                "exclusion_status": "excluded",
            })
            continue

        excluded = status == "mismatch"
        validation.append({
            "disease_name": original_name,
            "OMIM_id": original_omim,
            "normalized_disease_name": resolved_label,
            "normalized_OMIM_id": resolved_id,
            "status": status,
            "reason": reason,
            "excluded": excluded,
            "exclusion_status": "excluded" if excluded else None,
        })
        if excluded:
            continue
        diag.disease_name = resolved_label
        diag.OMIM_id = resolved_id
        normalized_ans.append(diag)
        unique_omim_ids.add(duplicate_key)

    state["zeroShotValidation"] = validation
    zeroshot_output.ans = normalized_ans
    return zeroshot_output

def disease_normalize(disease_name: str):
    """
    疾患名をembeddingし、FAISSインデックスで最も類似するOMIM IDと正規化病名を返す。
    """
    # 疾患名をembedding
    response = client.embeddings.create(
        model=deployment_name,
        input=[disease_name]
    )
    query_embedding = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    
    # 類似度最大のインデックスを取得
    distance, indices = faiss_index.search(query_embedding, 1)
    idx = indices[0][0]
    sim = float(distance[0][0])  # コサイン類似度
    
    omim_id_from_index = index_map["omim_ids"][idx]
    label_from_index = index_map["labels"][idx]
    
    # omim_mapping.jsonから正式病名を取得
    omim_id_num = extract_omim_number(omim_id_from_index)
    omim_label = omim_mapping_by_number.get(omim_id_num, label_from_index)
    
    return omim_id_from_index, omim_label, sim

def diseaseNormalizeForDiagnosis(Diagnosis):
    """
    tentativeDiagnosis: DiagnosisOutput
    病名に互換性のあるOMIM IDと正規化病名を付与する。
    埋め込み類似度による候補除外は行わず、互換性を確認できない候補は
    OMIM IDを未確定のまま保持する。既存OMIM IDは優先する。
    """
    filtered_ans = []
    if not hasattr(Diagnosis, "ans"):
        return Diagnosis
        
    for diag in Diagnosis.ans:
        existing_omim_num = extract_omim_number(getattr(diag, "OMIM_id", None))
        if existing_omim_num:
            diag.OMIM_id = canonical_omim_id(existing_omim_num)
            diag.disease_name = omim_mapping_by_number.get(
                existing_omim_num,
                diag.disease_name.strip().strip("*")
            )
            filtered_ans.append(diag)
            continue

        disease_name_upper = diag.disease_name.upper()
        omim_id, omim_label, sim = disease_normalize(disease_name_upper)

        candidate_id = canonical_omim_id(omim_id)
        if candidate_id and omim_label and names_compatible(
            diag.disease_name,
            omim_label,
            known_disease_aliases(candidate_id),
        ):
            diag.OMIM_id = candidate_id
            diag.disease_name = omim_label
            filtered_ans.append(diag)
        else:
            diag.OMIM_id = None
            filtered_ans.append(diag)
            print(
                f"Retained {diag.disease_name} without an OMIM ID: "
                f"no compatible local name match (embedding similarity={sim:.2f})"
            )
    Diagnosis.ans = filtered_ans
    return Diagnosis
