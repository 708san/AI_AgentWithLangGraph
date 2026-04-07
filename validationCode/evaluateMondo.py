import os
import json
import glob
import pandas as pd
import re
from MondoMatcher import MondoOntologyMatcher

# パス設定
JSON_DIR = './testCode/res_5'
TSV_PATH = './sampleData/ValidationDataWithoutDupli_newest.tsv'
MONDO_JSON = './validationCode/mondo-base.json'
OUTPUT_CSV = 'MONDO_BASE_match.csv'

# ツールの定義と、JSON内でのキー候補（揺れに対応）
TOOL_CONFIG = {
    'PubCaseFinder': ['pubCaseFinder', 'pubcasefinder'],
    'PhenotypeSearch': ['phenotypeSearchResult', 'phenotypesearch'],
    'ZeroShot': ['zeroShotResult', 'zeroShot', 'zeroshot'],
    'GestaltMatcher': ['GestaltMatcher', 'gestaltmatcher'],
    'TentativeDiagnosis': ['tentativeDiagnosis', 'tentativediagnosis', 'tentative_diag'],
    'FinalDiagnosis': ['finalDiagnosis', 'finaldiagnosis', 'final_diag']
}

def normalize_omim(val):
    """どんな形式のOMIM IDも 'OMIM:数字' に統一する"""
    if val is None or pd.isna(val) or val == "":
        return None
    s = str(val).strip().upper()
    if s in ['NONE', 'N/A', '']:
        return None
    # 数字部分だけを抽出
    nums = re.findall(r'\d+', s)
    if nums:
        return f"OMIM:{nums[0]}"
    return None

def extract_omim(item):
    """
    辞書からOMIM IDを抽出。数値型やネスト構造、キーの揺れに対応。
    """
    if not isinstance(item, dict):
        return None
    
    # 探索対象のキー（大文字小文字を問わずチェック）
    target_keys = ['omim_id', 'omim', 'omimid', 'omim_ps']
    
    # 直接の階層をチェック
    for k, v in item.items():
        if k.lower() in target_keys:
            return normalize_omim(v)
    
    # disease_info などのサブ階層をチェック
    if 'disease_info' in item:
        return extract_omim(item['disease_info'])
    
    return None

def get_candidates(data, tool_name):
    """
    ツール名に対応するキーを探し、リストを抽出する。
    {"ans": [...]} 形式と直のリスト形式の両方に対応。
    """
    possible_keys = TOOL_CONFIG.get(tool_name, [tool_name])
    
    for key in possible_keys:
        if key in data:
            val = data[key]
            # {"ans": [...]} 形式の場合
            if isinstance(val, dict) and 'ans' in val:
                return val['ans']
            # リスト形式の場合
            if isinstance(val, list):
                return val
    return []

def main():
    if not os.path.exists(MONDO_JSON):
        print(f"Error: {MONDO_JSON} not found.")
        return
    
    matcher = MondoOntologyMatcher(MONDO_JSON, child_threshold=4)
    
    # 正解データの読み込み
    df_tsv = pd.read_table(TSV_PATH)
    truth_map = df_tsv.set_index(df_tsv['patient_id'].astype(str))['omim_ids'].apply(normalize_omim).to_dict()

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

        correct_omim = truth_map[p_id]
        row = {'patient_id': p_id}

        for tool_name in TOOL_CONFIG.keys():
            candidates = get_candidates(data, tool_name)
            
            match_rank = None
            close_rank = None

            for i, cand in enumerate(candidates):
                pred_omim = extract_omim(cand)
                if not pred_omim or not correct_omim:
                    continue
                
                judgment = matcher.judge(correct_omim, pred_omim)
                # JSON内に 'rank' があれば採用、なければ index+1
                rank = cand.get('rank', i + 1)
                
                if judgment == "MATCH":
                    if match_rank is None: match_rank = rank
                    if close_rank is None: close_rank = rank
                elif judgment == "CLOSE MATCH":
                    if close_rank is None: close_rank = rank
                
                # すでに両方見つかっている場合、かつこれ以上順位が上がらない(i=0)ならブレイク可能ですが、
                # rank指定がある可能性を考慮して最後まで回すか、最小値を保持します。

            row[f"{tool_name}_Match_rank"] = match_rank
            row[f"{tool_name}_Close_rank"] = close_rank

        all_results.append(row)

    # 出力
    output_df = pd.DataFrame(all_results)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Success! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()