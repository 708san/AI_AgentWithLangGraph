import os
import json
import glob
import pandas as pd
from mondoMatcher import MondoOntologyMatcher

# パス設定
JSON_DIR = './testCode/res_5'
TSV_PATH = './sampleData/ValidationDataWithoutDupli_newest.tsv'
MONDO_JSON = './validationCode/mondo-base.json'
OUTPUT_CSV = 'MONDO_BASE_match.csv'

# 各ツールの評価対象となるキーの定義
TOOL_CONFIG = {
    'PubCaseFinder': {'json_key': 'pubCaseFinder'},
    'PhenotypeSearch': {'json_key': 'phenotypeSearchResult'},
    'ZeroShot': {'json_key': 'zeroShotResult'},
    'GestaltMatcher': {'json_key': 'GestaltMatcher'},
    'TentativeDiagnosis': {'json_key': 'tentativeDiagnosis'},
    'FinalDiagnosis': {'json_key': 'finalDiagnosis'}
}

def normalize_omim(val):
    """どんな形式のOMIM IDも 'OMIM:数字' に統一する"""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip().upper()
    if not s or s == 'NONE' or s == 'N/A':
        return None
    # 数字だけ、または 'OMIM:123' の形にする
    if s.startswith('OMIM:'):
        return s
    # 数字以外の文字（記号など）を除去して数字部分だけ取り出す（123456など）
    import re
    nums = re.findall(r'\d+', s)
    if nums:
        return f"OMIM:{nums[0]}"
    return None

def extract_omim_from_item(item):
    """
    1つの要素（辞書）からOMIM IDを執念深く探す
    """
    if not isinstance(item, dict):
        return None
    
    # 探索対象のキー候補
    possible_keys = ['omim_id', 'OMIM_id', 'omimId', 'omim']
    for k in possible_keys:
        if k in item:
            return normalize_omim(item[k])
    
    # phenotypeSearchResult のように disease_info 内にある場合
    if 'disease_info' in item and isinstance(item['disease_info'], dict):
        sub_info = item['disease_info']
        for k in possible_keys:
            if k in sub_info:
                return normalize_omim(sub_info[k])
                
    return None

def evaluate_tool(data, tool_name, correct_omim_norm, matcher):
    """
    ツールごとのリストを走査し、ランクと判定を返す
    """
    config = TOOL_CONFIG[tool_name]
    raw_val = data.get(config['json_key'])
    
    # zeroShotResult や finalDiagnosis は {"ans": [...]} という構造
    if isinstance(raw_val, dict) and 'ans' in raw_val:
        items = raw_val['ans']
    elif isinstance(raw_val, list):
        items = raw_val
    else:
        return "Miss", "N/A"

    found_matches = []

    for i, item in enumerate(items):
        pred_omim = extract_omim_from_item(item)
        if not pred_omim or not correct_omim_norm:
            continue
            
        # mondoMatcherで判定
        result = matcher.judge_relationship(correct_omim_norm, pred_omim)
        
        # ランクはJSON内の'rank'キーを優先、なければインデックス+1
        current_rank = item.get('rank', i + 1)
        
        if result == "MATCH":
            found_matches.append(("Match", current_rank))
        elif result == "CLOSE MATCH":
            found_matches.append(("Similar", current_rank))

    if not found_matches:
        return "Miss", "N/A"

    # 最も高い順位(数値が小さいもの)を返す
    found_matches.sort(key=lambda x: x[1] if isinstance(x[1], (int, float)) else 999)
    return found_matches[0]

def main():
    if not os.path.exists(MONDO_JSON):
        print(f"Error: {MONDO_JSON} not found.")
        return
    matcher = MondoOntologyMatcher(MONDO_JSON)

    # 正解データ読み込み
    df_tsv = pd.read_table(TSV_PATH)
    # omim_idsを正規化して保持
    df_tsv['correct_omim_norm'] = df_tsv['omim_ids'].apply(normalize_omim)
    truth_map = df_tsv.set_index(df_tsv['patient_id'].astype(str))['correct_omim_norm'].to_dict()

    all_results = []
    json_paths = glob.glob(os.path.join(JSON_DIR, "*.json"))
    
    for path in sorted(json_paths):
        p_id = os.path.basename(path).replace('.json', '')
        if p_id not in truth_map:
            continue

        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except:
                continue

        correct_omim_norm = truth_map[p_id]
        row = {'patient_id': p_id}

        for tool_name in TOOL_CONFIG.keys():
            res, rank = evaluate_tool(data, tool_name, correct_omim_norm, matcher)
            row[f"{tool_name}_res"] = res
            row[f"{tool_name}_rank"] = rank

        all_results.append(row)

    # 出力
    output_df = pd.DataFrame(all_results)
    cols = ['patient_id']
    for tool in TOOL_CONFIG.keys():
        cols.extend([f"{tool}_res", f"{tool}_rank"])
    
    output_df[cols].to_csv(OUTPUT_CSV, index=False)
    print(f"Success! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()