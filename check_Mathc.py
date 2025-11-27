import os
import csv
import json
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv

# --- .envファイルから環境変数を読み込み ---
load_dotenv()

# --- 定数設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_CSV = os.path.join(BASE_DIR, "escape", "result_analyze", "patient_omim_label.csv")
TOOL_RESULTS_DIR = os.path.join(BASE_DIR, "escape", "res_4o")
RANKED_RESULTS_DIR = os.path.join(BASE_DIR, "ranked_results_structured")
OUTPUT_CSV = os.path.join(BASE_DIR, "evaluation_summary_with_similarity.csv")

# --- Azure OpenAIの設定 ---
try:
    tenant = "dbcls"
    region = "japaneast"
    model = "text-embedding-3-large"
    deployment_name = f"{region}-{model}"
    endpoint = f"https://{tenant}-{region}.openai.azure.com/"
    api_key = os.getenv(f"AZURE_{tenant.upper()}_{region.upper()}")
    if not api_key:
        raise KeyError(f"AZURE_{tenant.upper()}_{region.upper()}")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-05-01-preview"
    )
except KeyError as e:
    print(f"エラー: 環境変数 {e} が設定されていません。.envファイルを確認してください。")
    exit(1)

# --- ベクトル関連のヘルパー関数 ---
def get_embedding(text: str) -> np.ndarray:
    """テキストをベクトル化する。大文字/小文字を統一する処理を追加。"""
    # 大文字/小文字を統一し、不要な文字を除去
    cleaned_text = text.strip("*").strip().lower()
    if not cleaned_text:
        raise ValueError("Cannot get embedding for an empty string.")
    response = client.embeddings.create(model=deployment_name, input=[cleaned_text])
    return np.array(response.data[0].embedding, dtype='float32')

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """コサイン類似度を計算する"""
    norm_vec1 = vec1 / np.linalg.norm(vec1)
    norm_vec2 = vec2 / np.linalg.norm(vec2)
    return float(np.dot(norm_vec1, norm_vec2))

# --- 評価用のコア関数 (以前の仕様に戻す) ---
def evaluate_list(item_list: list, correct_omim_id: str, correct_label_emb: np.ndarray, omim_key: str, name_key: str) -> (str, str):
    """
    リストを評価し、(評価結果, ランク) のタプルを返す。
    以前の仕様（Match優先、なければSimilar）に戻す。
    """
    if not item_list or not isinstance(item_list, list):
        return "Miss", "N/A"

    correct_omim_num = correct_omim_id.replace("OMIM:", "").strip()

    # --- ステップ1: OMIM IDによる完全一致をリスト全体で試みる ---
    for i, item in enumerate(item_list):
        if not isinstance(item, dict): continue
        
        omim_value_raw = item
        for key in omim_key.split('.'):
            omim_value_raw = omim_value_raw.get(key) if isinstance(omim_value_raw, dict) else None
        
        if omim_value_raw:
            item_omim_num = str(omim_value_raw).replace("OMIM:", "").strip()
            if item_omim_num == correct_omim_num:
                rank = str(item.get("rank", i + 1))
                return "Match", rank # Matchが見つかったら即座に終了

    # --- ステップ2: Matchがなかった場合のみ、類似度による一致を試みる ---
    max_sim = -1.0
    best_rank_for_similar = "N/A"
    for i, item in enumerate(item_list):
        if not isinstance(item, dict): continue

        name_value_raw = item
        for key in name_key.split('.'):
            name_value_raw = name_value_raw.get(key) if isinstance(name_value_raw, dict) else None

        if name_value_raw:
            try:
                item_emb = get_embedding(name_value_raw)
                sim = cosine_similarity(correct_label_emb, item_emb)
                
                # 類似度がしきい値(0.7)を超え、かつこれまでで最も高い場合、候補を更新
                if sim >= 0.75 and sim > max_sim:
                    max_sim = sim
                    best_rank_for_similar = str(item.get("rank", i + 1))
            except Exception:
                continue
    
    if max_sim != -1.0:
        return "Similar", best_rank_for_similar

    # MatchもSimilarも見つからなかった場合
    return "Miss", "N/A"

def main():
    # 正解データを読み込む
    ground_truth = {}
    try:
        with open(GROUND_TRUTH_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row['patient_id'].strip()
                if patient_id:
                    ground_truth[patient_id] = {
                        "omim_id": row['omim_id'].strip(),
                        "label": row['label'].strip()
                    }
    except FileNotFoundError:
        print(f"エラー: 正解ファイルが見つかりません: {GROUND_TRUTH_CSV}")
        return

    # 結果を書き込むCSVファイルを準備
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out)
        header = [
            "patient_id", "correct_omim_id",
            "GestaltMatcher_res", "GestaltMatcher_rank",
            "PCFNode_res", "PCFNode_rank",
            "DiseaseSearch_res", "DiseaseSearch_rank",
            "ZeroShot_res", "ZeroShot_rank",
            "GPT4o_Ranked_res", "GPT4o_Ranked_rank"
        ]
        writer.writerow(header)

        for patient_id, truth_data in ground_truth.items():
            print(f"--- Processing Patient ID: {patient_id} ---")
            correct_omim_id = truth_data["omim_id"]
            correct_label = truth_data["label"]
            
            try:
                # get_embedding内で小文字に変換される
                correct_label_emb = get_embedding(correct_label)
            except Exception as e:
                print(f"Error getting embedding for ground truth label '{correct_label}': {e}")
                continue

            tool_json_path = os.path.join(TOOL_RESULTS_DIR, f"{patient_id}.json")
            ranked_json_path = os.path.join(RANKED_RESULTS_DIR, f"ranked_{patient_id}.json")

            results = {}

            if os.path.exists(tool_json_path):
                with open(tool_json_path, 'r', encoding='utf-8') as f:
                    tool_data = json.load(f)
                
                results["GestaltMatcher"] = evaluate_list(tool_data.get("GestaltMatcherNode"), correct_omim_id, correct_label_emb, 'omim_id', 'syndrome_name')
                results["PCFNode"] = evaluate_list(tool_data.get("PCFNode"), correct_omim_id, correct_label_emb, 'omim_id', 'omim_disease_name_en')
                results["DiseaseSearch"] = evaluate_list(tool_data.get("DiseaseSearchWithHPONode"), correct_omim_id, correct_label_emb, 'disease_info.OMIM_id', 'disease_info.disease_name')
                results["ZeroShot"] = evaluate_list(tool_data.get("NormalizeZeroShotNode", {}).get("ans"), correct_omim_id, correct_label_emb, 'OMIM_id', 'disease_name')
            else:
                results["GestaltMatcher"] = ("File Missing", "N/A")
                results["PCFNode"] = ("File Missing", "N/A")
                results["DiseaseSearch"] = ("File Missing", "N/A")
                results["ZeroShot"] = ("File Missing", "N/A")

            if os.path.exists(ranked_json_path):
                with open(ranked_json_path, 'r', encoding='utf-8') as f:
                    ranked_data = json.load(f)
                results["GPT4o_Ranked"] = evaluate_list(ranked_data.get("ranked_list_from_gpt4o"), correct_omim_id, correct_label_emb, 'OMIM_id', 'disease_name')
            else:
                results["GPT4o_Ranked"] = ("File Missing", "N/A")

            row_to_write = [patient_id, correct_omim_id]
            for key in ["GestaltMatcher", "PCFNode", "DiseaseSearch", "ZeroShot", "GPT4o_Ranked"]:
                row_to_write.extend(results[key])
            writer.writerow(row_to_write)
    
    print(f"\n処理が完了しました。結果は {OUTPUT_CSV} に保存されました。")

if __name__ == "__main__":
    main()