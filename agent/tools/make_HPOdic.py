import json
from typing import List, Dict
import os


def make_hpo_dic(hpo_list: List[str], mapping_path: str) -> Dict[str, str]:
    """
    hpo_list: HPO IDのリスト
    mapping_path: phenotype_mapping.jsonのパス
    戻り値: {HPO_ID: ターム名} の辞書
    """
    # mapping_pathを絶対パスに変換
    base_dir = os.path.dirname(os.path.abspath(__file__))
    abs_mapping_path = os.path.join(base_dir, "..", "data", "phenotype_mapping.json")
    with open(abs_mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {hpo_id: mapping.get(hpo_id, "") for hpo_id in hpo_list}