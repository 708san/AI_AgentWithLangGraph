import re
import time
from typing import Any, Optional

import requests


# The public docs list http://, but POST /disease-list-detail currently only
# works reliably when called over HTTPS directly.
PHENOBRAIN_BASE_URL = "https://www.phenobrain.cs.tsinghua.edu.cn"


def _log_error(message: str) -> None:
    print(f"[PhenoBrain] {message}")


def _normalize_source_codes(source_codes: Any) -> list[str]:
    if source_codes is None:
        return []
    if isinstance(source_codes, str):
        return [code for code in re.split(r"[,;]\s*", source_codes) if code]
    if isinstance(source_codes, list):
        return [str(code) for code in source_codes if code]
    return [str(source_codes)]


def _first_code(source_codes: list[str], prefix: str) -> Optional[str]:
    for code in source_codes:
        if code.upper().startswith(prefix):
            return code
    return None


def _extract_detail_map(details: Any) -> dict[str, dict[str, Any]]:
    if isinstance(details, dict):
        for key in ("result", "data", "diseaseList", "diseases"):
            if isinstance(details.get(key), list):
                details = details[key]
                break
        else:
            if details.get("CODE"):
                details = [details]

    detail_map: dict[str, dict[str, Any]] = {}
    if isinstance(details, dict):
        for rd_id, item in details.items():
            if isinstance(item, dict):
                detail_map[str(rd_id)] = item
        return detail_map

    if not isinstance(details, list):
        return detail_map

    for item in details:
        if not isinstance(item, dict):
            continue
        rd_id = item.get("CODE") or item.get("RD_ID") or item.get("rd_id")
        if rd_id:
            detail_map[str(rd_id)] = item
    return detail_map


def _request_json(method: str, url: str, *, request_timeout: float, **kwargs):
    response = requests.request(method, url, timeout=request_timeout, **kwargs)
    response.raise_for_status()
    return response.json()


def _predict(
    hpo_list: list[str],
    model: str,
    topk: int,
    request_timeout: float,
) -> Optional[str]:
    params = [("model", model)]
    params.extend(("hpoList[]", hpo) for hpo in hpo_list)
    params.append(("topk", topk))

    data = _request_json(
        "GET",
        f"{PHENOBRAIN_BASE_URL}/predict",
        request_timeout=request_timeout,
        params=params,
    )
    task_id = data.get("TASK_ID") if isinstance(data, dict) else None
    if not task_id:
        _log_error(f"predict response did not contain TASK_ID: {data}")
        return None
    return str(task_id)


def _poll_prediction(
    task_id: str,
    request_timeout: float,
    poll_interval: float,
    max_poll_seconds: float,
) -> Optional[list[dict[str, Any]]]:
    deadline = time.monotonic() + max_poll_seconds

    while time.monotonic() < deadline:
        data = _request_json(
            "GET",
            f"{PHENOBRAIN_BASE_URL}/query-predict-result",
            request_timeout=request_timeout,
            params={"taskId": task_id},
        )
        if not isinstance(data, dict):
            _log_error(f"polling response was not an object: {data}")
            return None

        state = data.get("state")
        if state == "SUCCESS":
            result = data.get("result")
            if isinstance(result, list):
                return result
            _log_error(f"SUCCESS response did not contain result list: {data}")
            return None
        if state not in {"MODEL_INIT", "MODEL_PREDICT"}:
            _log_error(f"PhenoBrain returned error state: {state}")
            return None

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval, remaining))

    _log_error(f"polling timed out after {max_poll_seconds} seconds")
    return None


def _fetch_disease_details(
    rd_ids: list[str],
    request_timeout: float,
) -> Optional[dict[str, dict[str, Any]]]:
    if not rd_ids:
        return {}
    data = _request_json(
        "POST",
        f"{PHENOBRAIN_BASE_URL}/disease-list-detail",
        request_timeout=request_timeout,
        json={"diseaseList": rd_ids},
    )
    return _extract_detail_map(data)


def _format_results(
    predictions: list[dict[str, Any]],
    details_by_rd_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for index, item in enumerate(predictions, 1):
        if not isinstance(item, dict):
            continue
        rd_id = item.get("CODE") or item.get("RD_ID") or item.get("rd_id")
        if not rd_id:
            continue

        rd_id = str(rd_id)
        detail = details_by_rd_id.get(rd_id, {})
        source_codes = _normalize_source_codes(detail.get("SOURCE_CODES"))
        disease_name = detail.get("ENG_NAME") or item.get("ENG_NAME") or ""

        results.append(
            {
                "disease_name": disease_name,
                "omim_id": _first_code(source_codes, "OMIM:"),
                "orpha_id": _first_code(source_codes, "ORPHA:"),
                "source_codes": source_codes,
                "rd_id": rd_id,
                "rank": index,
                "score": item.get("SCORE"),
            }
        )
    return results


def call_phenobrain(
    hpo_list,
    model="Ensemble",
    topk=5,
    request_timeout=30,
    poll_interval=1.0,
    max_poll_seconds=60,
):
    hpo_list = [str(hpo) for hpo in (hpo_list or []) if hpo]
    if not hpo_list:
        return []

    try:
        task_id = _predict(hpo_list, model, topk, request_timeout)
        if not task_id:
            return []

        predictions = _poll_prediction(
            task_id,
            request_timeout=request_timeout,
            poll_interval=poll_interval,
            max_poll_seconds=max_poll_seconds,
        )
        if predictions is None:
            return []

        rd_ids = [
            str(item.get("CODE") or item.get("RD_ID") or item.get("rd_id"))
            for item in predictions
            if isinstance(item, dict) and (item.get("CODE") or item.get("RD_ID") or item.get("rd_id"))
        ]
        details_by_rd_id = _fetch_disease_details(rd_ids, request_timeout)
        if details_by_rd_id is None:
            return []

        return _format_results(predictions, details_by_rd_id)
    except requests.exceptions.RequestException as exc:
        _log_error(f"request failed: {exc}")
    except ValueError as exc:
        _log_error(f"failed to parse JSON response: {exc}")
    except Exception as exc:
        _log_error(f"unexpected failure: {exc}")
    return []
