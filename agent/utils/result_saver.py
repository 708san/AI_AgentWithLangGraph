import os
import json
import fcntl  # 追加: 排他制御用
from functools import wraps
from pydantic import BaseModel
import copy
from typing import Any
from agent.utils.response_serializer import (
    serialize_external_compatibility,
    serialize_node_result,
    serialize_public_state,
)

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
    """Persist node output and refresh accumulated, externally consumable JSONs."""
    if not result_data or not isinstance(result_data, dict):
        return
    try:
        patient_id = state.get("patient_id", "unknown")
        res_dir = os.environ.get("AGENT_RESULT_DIR", "res")
        node_result_dir = os.environ.get("AGENT_NODE_RESULT_DIR", res_dir)
        os.makedirs(res_dir, exist_ok=True)
        os.makedirs(node_result_dir, exist_ok=True)
        out_path = os.path.join(node_result_dir, f"{patient_id}.json")

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

                # Rebuild the latest accumulated view from prior node outputs
                # and this node's update. Graph nodes may run in parallel, so
                # the lock protects both this merge and all emitted snapshots.
                state_update = result_data.get("result") if isinstance(result_data.get("result"), dict) else result_data
                data_for_saving.update(state_update)
                node_results = data_for_saving.setdefault("node_results", {})
                if not isinstance(node_results, dict):
                    node_results = {}
                    data_for_saving["node_results"] = node_results
                node_results[node_name] = serialize_node_result(node_name, result_data, state)

                # The graph input carries patient data and settings that are
                # not necessarily returned by any one node. Use it as the
                # baseline, then let the persisted cumulative node results win.
                accumulated_state = {
                    key: value for key, value in state.items()
                    if key not in {"node_results", "llm"}
                }
                accumulated_state.update({
                    key: value for key, value in data_for_saving.items()
                    if key not in {"node_results", "llm"}
                })
                if state.get("evidence_mode"):
                    accumulated_state["evidence_mode"] = state["evidence_mode"]
                public_state = serialize_public_state(accumulated_state)
                external_outputs = serialize_external_compatibility(accumulated_state)

                serializable_data = _convert_pydantic_objects(data_for_saving)
                f.seek(0)
                f.truncate()
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
                f.flush()

                # The decorator-driven runner always leaves the established
                # node-wrapped files and the canonical traceable response in
                # AGENT_RESULT_DIR, updating them as each node completes.
                _write_json(os.path.join(res_dir, "public_response.json"), public_state)
                for artifact_name, artifact in external_outputs.items():
                    if artifact_name == "external_response":
                        filename = "external_response.json"
                    else:
                        filename = f"{artifact_name}.json"
                    _write_json(os.path.join(res_dir, filename), artifact)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"[ERROR in result_saver for node '{node_name}']: {e}")


def _write_json(path: str, value: Any) -> None:
    """Atomically replace one generated JSON snapshot."""
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(_convert_pydantic_objects(value), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def save_result(node_name):
    def decorator(func):
        @wraps(func)
        def wrapper(state):
            result_data = func(state)
            save_node_result(node_name, result_data, state)
            return result_data
        wrapper._saves_result = True
        return wrapper
    return decorator
