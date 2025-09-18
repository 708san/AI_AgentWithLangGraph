import os
import numpy as np
import faiss
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

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

# インデックスとマッピングのロード
faiss_index = faiss.read_index(INDEX_BIN)
with open(INDEX_JSON, encoding="utf-8") as f:
    index_map = json.load(f)
with open(OMIM_MAPPING_JSON, encoding="utf-8") as f:
    omim_mapping = json.load(f)


def disease_normalize(disease_name: str):
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
    omim_id = index_map["omim_ids"][idx]
    label = index_map["labels"][idx]
    # omim_mapping.jsonから正式病名を取得
    omim_label = omim_mapping.get(omim_id, label)
    return omim_id, omim_label, sim

def diseaseNormalizeForDiagnosis(Diagnosis):
    """
    tentativeDiagnosis: DiagnosisOutput
    各診断候補にOMIM idと正規化病名を付与し、類似度0.8未満は棄却
    """
    filtered_ans = []
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