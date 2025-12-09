import os
import json
import re

# --- 定数設定 ---
# 入力ディレクトリ
INPUT_DIR = os.path.join(os.getcwd(), "ranked_results_json")
# 出力ディレクトリ
OUTPUT_DIR = os.path.join(os.getcwd(), "ranked_results_structured")

def parse_ranked_list(text_list: str) -> list:
    """
    モデルが生成したテキストのランキングリストを、オブジェクトのリストに変換する。
    より堅牢な正規表現とフォールバックロジックを持つ改良版。
    """
    structured_list = []
    
    # --- 改良版の正規表現パターン ---
    # 1. OMIM IDのコロンの後にスペースがあっても許容する (\s*)
    # 2. OMIM IDが "Not available" のような文字列でも許容する ([\w\s]+)
    pattern = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*\((OMIM:\s*[\w\s]+)\)\s*$", re.MULTILINE)

    lines = text_list.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if match:
            # 正規表現がマッチした場合
            rank = int(match.group(1))
            disease_name = match.group(2).strip()
            omim_id = match.group(3).strip()
            
            structured_list.append({
                "rank": rank,
                "disease_name": disease_name,
                "OMIM_id": omim_id
            })
        else:
            # --- フォールバック処理 ---
            # 正規表現がマッチしなかった場合、単純な分割を試みる
            try:
                # ランク部分を抽出
                rank_match = re.match(r"^\s*(\d+)", line)
                rank = int(rank_match.group(1)) if rank_match else -1
                
                # OMIM ID部分を括弧ごと抽出
                omim_part_match = re.search(r"\((OMIM:.*?)\)", line)
                if omim_part_match:
                    omim_id = omim_part_match.group(1).strip()
                    # 病名部分はランクとOMIM IDの間と仮定
                    content_part = line[rank_match.end():omim_part_match.start()].strip()
                    disease_name = re.sub(r"^\.\s*", "", content_part).strip()
                    
                    structured_list.append({
                        "rank": rank,
                        "disease_name": disease_name,
                        "OMIM_id": omim_id
                    })
                else:
                    # OMIM IDが見つからない場合は、ランク以降をすべて病名とみなす
                    content_part = line[rank_match.end():].strip()
                    disease_name = re.sub(r"^\.\s*", "", content_part).strip()
                    structured_list.append({
                        "rank": rank,
                        "disease_name": disease_name,
                        "OMIM_id": "N/A"
                    })
            except Exception:
                # パースに失敗した行はスキップ
                print(f"Warning: Could not parse line: '{line}'")
                continue

    return structured_list

def process_and_restructure_files(input_dir: str, output_dir: str):
    """
    入力ディレクトリ内のJSONファイルを処理し、構造化して出力ディレクトリに保存する。
    """
    if not os.path.exists(input_dir):
        print(f"エラー: 入力ディレクトリ '{input_dir}' が見つかりません。")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])

    for filename in json_files:
        input_filepath = os.path.join(input_dir, filename)
        output_filepath = os.path.join(output_dir, filename)
        
        print(f"--- Processing: {filename} ---")

        try:
            with open(input_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            ranked_text = data.get("ranked_list_from_gpt4o")

            if isinstance(ranked_text, str) and ranked_text.strip():
                structured_data = parse_ranked_list(ranked_text)
                data["ranked_list_from_gpt4o"] = structured_data
            else:
                # 既に処理済みか、中身が空の場合は何もしない
                if not isinstance(ranked_text, list):
                     data["ranked_list_from_gpt4o"] = [] # 空の文字列だった場合は空リストに変換
                print(f"Skipping parsing for {filename}: 'ranked_list_from_gpt4o' is not a processable string.")


            with open(output_filepath, 'w', encoding='utf-8') as f_out:
                json.dump(data, f_out, indent=2, ensure_ascii=False)
            
            print(f"Successfully restructured and saved to: {output_filepath}")

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filename}")
        except Exception as e:
            print(f"An unexpected error occurred with {filename}: {e}")

if __name__ == "__main__":
    process_and_restructure_files(INPUT_DIR, OUTPUT_DIR)