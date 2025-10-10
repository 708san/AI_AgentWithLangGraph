import os
import numpy as np
import faiss
import json
import re
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import Optional
from ..state.state_types import State,ZeroShotOutput

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

    for diag in zeroshot_output.ans:
        disease_name_upper = diag.disease_name.upper()
        omim_id, omim_label, sim = disease_normalize(disease_name_upper)

        # 類似度が高い場合のみ採用し、OMIM IDがユニークであることを確認
        if sim >= 0.75 and omim_id not in unique_omim_ids:
            diag.OMIM_id = omim_id
            diag.disease_name = omim_label
            normalized_ans.append(diag)
            unique_omim_ids.add(omim_id)
        else:
            print(f"Filtered out {diag.disease_name} (OMIM: {omim_id}) due to low similarity ({sim:.2f}) or duplication.")
    
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
    各診断候補にOMIM idと正規化病名を付与し、類似度0.75未満は棄却
    """
    filtered_ans = []
    if not hasattr(Diagnosis, "ans"):
        return Diagnosis
        
    for diag in Diagnosis.ans:
        disease_name_upper = diag.disease_name.upper()
        omim_id, omim_label, sim = disease_normalize(disease_name_upper)

        if sim >= 0.75:
            diag.OMIM_id = omim_id
            diag.disease_name = omim_label
            filtered_ans.append(diag)
        else:
            print(f"Filtered out {diag.disease_name} due to low similarity ({sim:.2f})")
    Diagnosis.ans = filtered_ans
    return Diagnosis