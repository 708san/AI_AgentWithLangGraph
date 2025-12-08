import os
import json
import fcntl  # 追加: 排他制御用
from functools import wraps
from pydantic import BaseModel
import copy

def _convert_pydantic_objects(obj):
    """
    データ構造（辞書やリスト）を再帰的にスキャンし、
    Pydanticオブジェクトを辞書に変換する。
    """
    if isinstance(obj, dict):
        return {k: _convert_pydantic_objects(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_pydantic_objects(elem) for elem in obj]
    elif isinstance(obj, BaseModel):
        return obj.model_dump()
    else:
        return obj

def save_result(node_name):
    def decorator(func):
        @wraps(func)
        def wrapper(state):
            # 1. 元のノード関数を実行し、Pydanticオブジェクトを含む結果を取得
            result_data = func(state)
            
            if not result_data or not isinstance(result_data, dict):
                # 変更がない、または期待する形式でない場合はそのまま返す
                return result_data

            # --- ファイル保存ロジック ---
            # 副作用が state に影響しないようにする
            try:
                patient_id = state.get("patient_id", "unknown")
                res_dir = "res"
                os.makedirs(res_dir, exist_ok=True)
                out_path = os.path.join(res_dir, f"{patient_id}.json")

                # ファイルが存在しない場合は空のJSONを作成しておく
                if not os.path.exists(out_path):
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump({}, f)

                # 排他制御付きで読み書きを行う
                with open(out_path, 'r+', encoding='utf-8') as f:
                    # 排他ロックを取得 (ブロッキング)
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        content = f.read()
                        if content.strip():
                            try:
                                data_for_saving = json.loads(content)
                            except json.JSONDecodeError:
                                # JSONが壊れている場合は空からやり直す（またはログ出力）
                                print(f"[WARNING] JSON decode error in {out_path}. Overwriting.")
                                data_for_saving = {}
                        else:
                            data_for_saving = {}

                        # 今回の結果をマージ
                        data_for_saving.update(result_data)

                        # ファイル保存用に、Pydanticオブジェクトを再帰的に辞書へ変換
                        serializable_data = _convert_pydantic_objects(data_for_saving)

                        # ファイルの先頭に戻って書き込む
                        f.seek(0)
                        f.truncate()
                        json.dump(serializable_data, f, ensure_ascii=False, indent=2)
                        
                    finally:
                        # ロック解放
                        fcntl.flock(f, fcntl.LOCK_UN)
            
            except Exception as e:
                print(f"[ERROR in result_saver for node '{node_name}']: {e}")
                # ファイル保存に失敗しても、エージェントの実行は継続させる

            # 2. LangGraphには、元の変更されていない結果を返す
            # これにより、state内のオブジェクトはPydanticオブジェクトのまま維持される
            return result_data
        return wrapper
    return decorator