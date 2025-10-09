import os
import json
from functools import wraps

def save_result(node_name):
    def decorator(func):
        @wraps(func)
        def wrapper(state):
            # 元のノード関数を実行
            result_data = func(state)
            
            # 結果のキーを特定 (e.g., "pubCaseFinder", "zeroShotResult")
            # result_dataは{"key": value}の形式を想定
            if not result_data or not isinstance(result_data, dict):
                return result_data

            # 最初のキーを結果のキーとみなす（より堅牢な方法も検討可能）
            result_key = next(iter(result_data))
            result_value = result_data[result_key]
            patient_id = state.get("patient_id", "unknown")

            # ファイル保存ロジック
            res_dir = "res"
            os.makedirs(res_dir, exist_ok=True)
            out_path = os.path.join(res_dir, f"{patient_id}.json")

            try:
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    with open(out_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    data = {}
            except json.JSONDecodeError:
                data = {}

            # Pydanticモデルやリストをdictに変換
            if hasattr(result_value, "dict"):
                data[node_name] = result_value.dict()
            elif isinstance(result_value, list):
                data[node_name] = [r.dict() if hasattr(r, "dict") else r for r in result_value]
            else:
                data[node_name] = result_value

            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return result_data
        return wrapper
    return decorator
