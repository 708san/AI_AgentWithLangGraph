import os
import json
from openai import AzureOpenAI, APIError
from dotenv import load_dotenv
from typing import Dict, Any, List

# --- .envファイルから環境変数を読み込み ---
load_dotenv()

# --- Azure OpenAI クライアントの設定 ---
try:
    AZURE_DEPLOYMENT_NAME = os.environ["AZURE_OPENAI_4o_DEPLOYMENT_NAME"]
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_4o_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_4o_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_4o_API_VERSION"],
    )
except KeyError as e:
    print(f"エラー: 環境変数 {e} が設定されていません。.envファイルを確認してください。")
    exit(1)

# --- 定数設定 ---
INPUT_DIR = "/Users/yoshikuwa-n/Downloads/WorkForBioHackathon/AI_AgentWithLangGraph/escape/res_4o"
OUTPUT_DIR = os.path.join(os.getcwd(), "ranked_results_json")

# --- プロンプトテンプレート (プレーンテキスト出力を指示するように変更) ---
RANKING_PROMPT_TEMPLATE = """You are a senior clinical geneticist. Your task is to synthesize findings from the provided analytical tool reports into a single, ranked list of potential diagnoses.

Create a master list of all unique disease candidates from all reports. Then, rank them from most to least plausible based on the strength of the evidence (consensus across tools, high scores/ranks).

Your final output must be a numbered list. Each line should contain only the rank, disease name, and OMIM ID. Do not include any reasoning or other text.

**Example Output:**
1. DISEASE NAME 1 (OMIM:XXXXXX)
2. DISEASE NAME 2 (OMIM:YYYYYY)
3. DISEASE NAME 3 (OMIM:ZZZZZZ)

---
INPUT CONTEXT

II. Analytical Tool Reports

PubCaseFinder Report (Phenotype-based):
{pcf_results}

Zero-Shot Diagnosis Report (Generative AI-based):
{zeroshot_results}

GestaltMatcher Report (Facial Dysmorphology-based):
{gestalt_matcher_results}

Phenotype Similarity Search Report (Vector-based):
{phenotype_search_results}

III. Supporting Literature
Web Search Results (Literature/Case Reports):
{web_search_results}
"""

# --- データ整形用のヘルパー関数 (変更なし) ---
def format_pcf(data: List[Dict[str, Any]]) -> str:
    if not data: return "No results."
    lines = [f"- {item.get('omim_disease_name_en', 'Unknown')} (Score: {item.get('score', 0):.3f})" for item in data]
    return "\n".join(lines)

def format_zeroshot(data: Dict[str, Any]) -> str:
    if not data or "ans" not in data or not data["ans"]: return "No results."
    lines = [f"- {item.get('disease_name', 'Unknown')} (Rank: {item.get('rank', 'N/A')})" for item in data["ans"]]
    return "\n".join(lines)

def format_gestalt(data: List[Dict[str, Any]]) -> str:
    if not data: return "No results."
    lines = [f"- {item.get('syndrome_name', 'Unknown')} (Score: {item.get('score', 0):.3f})" for item in data]
    return "\n".join(lines)

def format_phenotype_search(data: List[Dict[str, Any]]) -> str:
    if not data: return "No results."
    lines = []
    for item in data:
        info = item.get("disease_info", {})
        lines.append(f"- {info.get('disease_name', 'Unknown')} (OMIM: {info.get('OMIM_id', 'N/A')}, Score: {item.get('similarity_score', 0):.3f})")
    return "\n".join(lines)

def format_web_search(data: List[Dict[str, Any]]) -> str:
    if not data: return "No results."
    lines = [f"- Title: {item.get('title', 'No Title')}\n  Snippet: {item.get('snippet', 'No snippet.')}" for item in data]
    return "\n".join(lines)

# --- メイン処理 ---
def process_files_in_directory(input_dir: str, output_dir: str):
    """
    ディレクトリ内のすべてのJSONファイルを処理し、Azure OpenAIでランキングを生成し、結果を個別のJSONファイルに保存する。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])

    for filename in json_files:
        filepath = os.path.join(input_dir, filename)
        print(f"\n--- Processing file: {filename} ---")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            context = {
                "pcf_results": format_pcf(data.get("PCFNode", [])),
                "zeroshot_results": format_zeroshot(data.get("NormalizeZeroShotNode", {})),
                "gestalt_matcher_results": format_gestalt(data.get("GestaltMatcherNode", [])),
                "phenotype_search_results": format_phenotype_search(data.get("DiseaseSearchWithHPONode", [])),
                "web_search_results": format_web_search(data.get("HPOwebSearchNode", []))
            }

            final_prompt = RANKING_PROMPT_TEMPLATE.format(**context)

            print("Querying Azure OpenAI (gpt-4o) for ranking...")
            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[
                    # システムメッセージを一般的なものに戻す
                    {"role": "system", "content": "You are a helpful assistant specialized in clinical genetics."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.1,
                # response_format を削除
            )

            # モデルからのテキスト応答をそのまま取得
            ranked_list_content = response.choices[0].message.content

            output_data = {
                "source_file": filename,
                "prompt_sent_to_api": final_prompt,
                "ranked_list_from_gpt4o": ranked_list_content # テキストをそのまま保存
            }
            
            output_filepath = os.path.join(output_dir, f"ranked_{filename}")
            with open(output_filepath, 'w', encoding='utf-8') as f_out:
                json.dump(output_data, f_out, indent=2, ensure_ascii=False)
            
            print(f"Result successfully saved to: {output_filepath}")

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filename}")
        except APIError as e:
            print(f"!!! Azure OpenAI API Error occurred with {filename} !!!")
            print(f"    Status code: {e.status_code}")
            print(f"    Response: {e.response}")
            print(f"    Body: {e.body}")
        except Exception as e:
            print(f"An unexpected error occurred with {filename}: {e}")

if __name__ == "__main__":
    process_files_in_directory(INPUT_DIR, OUTPUT_DIR)