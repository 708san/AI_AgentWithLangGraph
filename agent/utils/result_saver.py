import os
import json
import fcntl  # 追加: 排他制御用
from functools import wraps
from pydantic import BaseModel
import copy
from typing import Any
from agent.utils.response_serializer import serialize_node_result

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

def save_node_result(node_name: str, result_data: Any, state: dict[str, Any]) -> None:
    """Persist both the accumulated state fields and a structured node result."""
    if not result_data or not isinstance(result_data, dict):
        return
    try:
        patient_id = state.get("patient_id", "unknown")
        res_dir = os.environ.get("AGENT_RESULT_DIR", "res")
        os.makedirs(res_dir, exist_ok=True)
        out_path = os.path.join(res_dir, f"{patient_id}.json")

        if not os.path.exists(out_path):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

        with open(out_path, "r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                content = f.read()
                if content.strip():
                    try:
                        data_for_saving = json.loads(content)
                    except json.JSONDecodeError:
                        print(f"[WARNING] JSON decode error in {out_path}. Overwriting.")
                        data_for_saving = {}
                else:
                    data_for_saving = {}

                # Keep the existing accumulated fields for diagnostics and
                # add one structured, per-node projection for consumers.
                data_for_saving.update(result_data)
                node_results = data_for_saving.setdefault("node_results", {})
                if not isinstance(node_results, dict):
                    node_results = {}
                    data_for_saving["node_results"] = node_results
                node_results[node_name] = serialize_node_result(node_name, result_data, state)

                serializable_data = _convert_pydantic_objects(data_for_saving)
                f.seek(0)
                f.truncate()
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"[ERROR in result_saver for node '{node_name}']: {e}")


def save_result(node_name):
    def decorator(func):
        @wraps(func)
        def wrapper(state):
            result_data = func(state)
            save_node_result(node_name, result_data, state)
            return result_data
        return wrapper
    return decorator
