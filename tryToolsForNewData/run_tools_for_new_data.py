#!/usr/bin/env python3
"""Run PubCaseFinder and GestaltMatcher for every row in the new-data TSV.

The request format and endpoints mirror ``agent/tools/pcf_api.py`` and
``agent/tools/gestaltMathcher.py``.  Unlike the normal pipeline helpers, this
script keeps the complete ranking returned by each API so it can be compared
later.  One JSON file is written per ``image_id`` under ``PCF`` and ``GM``.

Example:
    python tryToolsForNewData/run_tools_for_new_data.py
    python tryToolsForNewData/run_tools_for_new_data.py --limit 3
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TSV = (
    PROJECT_ROOT
    / "Data/NewData/phenopacket_v1.0.27_New"
    / "phenopacket_test_metadata_v0.1.27_GM_v1.1.5_with_phenopacket_data.tsv"
)
DEFAULT_IMAGE_DIR = DEFAULT_TSV.parent / "test_images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tryToolsForNewData"

PCF_URL = "https://pubcasefinder.dbcls.jp/api/pcf_get_ranked_list"
GM_URL = "https://staging-pubcasefinder.dbcls.jp/gm_endpoint/predict"
MAX_DISTANCE = 1.3


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _request_with_retries(
    method: str,
    url: str,
    *,
    max_retries: int,
    timeout: float,
    **kwargs: Any,
) -> tuple[Any | None, dict[str, Any] | None]:
    errors: list[str] = []
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response, None
        except Exception as exc:  # keep one failed row from stopping the batch
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
    return None, {"errors": errors}


def _present_hpo_ids(row: dict[str, str]) -> list[str]:
    value = row.get("pp_present_hpo") or row.get("present_hpo") or ""
    result: list[str] = []
    seen: set[str] = set()
    for hpo_id in value.split(";"):
        hpo_id = hpo_id.strip()
        if hpo_id and hpo_id not in seen:
            result.append(hpo_id)
            seen.add(hpo_id)
    return result


def _find_image(image_dir: Path, image_id: str) -> Path | None:
    direct = image_dir / image_id
    if direct.is_file():
        return direct
    matches = sorted(image_dir.glob(f"{image_id}.*"))
    return matches[0] if matches else None


def _pcf(row: dict[str, str], *, retries: int, timeout: float) -> dict[str, Any]:
    hpo_ids = _present_hpo_ids(row)
    request_info = {
        "method": "GET",
        "url": PCF_URL,
        "params": {"target": "omim", "format": "json", "hpo_id": ",".join(hpo_ids)},
        "hpo_ids": hpo_ids,
    }
    started = time.perf_counter()
    response, error = _request_with_retries(
        "GET",
        PCF_URL,
        max_retries=retries,
        timeout=timeout,
        params=request_info["params"],
    )
    if error:
        return {"status": "error", "request": request_info, "error": error, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}
    try:
        raw = response.json()
        if not isinstance(raw, list):
            raise ValueError(f"expected a list, got {type(raw).__name__}")
        ranking = []
        for rank, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                continue
            ranking.append({
                "rank": rank,
                "omim_disease_name_en": item.get("omim_disease_name_en", ""),
                "description": item.get("description", ""),
                "score": item.get("score"),
                "omim_id": item.get("id", ""),
                "raw": item,
            })
        return {
            "status": "ok",
            "request": request_info,
            "ranking": ranking,
            "all_results": ranking,
            "raw_response": raw,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except Exception as exc:
        return {"status": "error", "request": request_info, "error": {"errors": [f"response parse: {type(exc).__name__}: {exc}"]}, "response_text": response.text[:2000], "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}


def _gm(row: dict[str, str], image_path: Path, *, retries: int, timeout: float) -> dict[str, Any]:
    username = os.environ.get("GESTALT_API_USER")
    password = os.environ.get("GESTALT_API_PASS")
    request_info = {"method": "POST", "url": GM_URL, "image_path": str(image_path), "payload": {"img": "<base64 image>"}}
    if not username or not password:
        return {"status": "error", "request": request_info, "error": "GESTALT_API_USER and GESTALT_API_PASS are required"}
    started = time.perf_counter()
    try:
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    except Exception as exc:
        return {"status": "error", "request": request_info, "error": f"image read failed: {type(exc).__name__}: {exc}"}
    response, error = _request_with_retries(
        "POST",
        GM_URL,
        max_retries=retries,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
        json={"img": encoded},
        auth=(username, password),
    )
    if error:
        return {"status": "error", "request": request_info, "error": error, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}
    try:
        raw = response.json()
        syndromes = raw.get("suggested_syndromes_list", []) if isinstance(raw, dict) else []
        ranking = []
        for rank, item in enumerate(syndromes, start=1):
            if not isinstance(item, dict):
                continue
            result = dict(item)
            distance = result.get("distance") or result.get("gestalt_score")
            result["score"] = (MAX_DISTANCE - float(distance)) / MAX_DISTANCE if distance is not None else 0.0
            result["rank"] = rank
            ranking.append(result)
        return {
            "status": "ok",
            "request": request_info,
            "ranking": ranking,
            "all_results": ranking,
            "raw_response": raw,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except Exception as exc:
        return {"status": "error", "request": request_info, "error": {"errors": [f"response parse: {type(exc).__name__}: {exc}"]}, "response_text": response.text[:2000], "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", type=int, default=0, help="0-based row offset")
    parser.add_argument("--limit", type=int, default=None, help="maximum number of rows")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing JSON files")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    args.tsv = args.tsv.expanduser().resolve()
    args.image_dir = args.image_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.tsv.is_file():
        parser.error(f"TSV not found: {args.tsv}")
    if not args.image_dir.is_dir():
        parser.error(f"image directory not found: {args.image_dir}")

    with args.tsv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = rows[args.start : args.start + args.limit if args.limit is not None else None]
    pcf_dir = args.output_dir / "PCF"
    gm_dir = args.output_dir / "GM"
    pcf_dir.mkdir(parents=True, exist_ok=True)
    gm_dir.mkdir(parents=True, exist_ok=True)
    print(f"rows={len(selected)} output={args.output_dir}", flush=True)

    for index, row in enumerate(selected, start=args.start + 1):
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            print(f"[{index}] skipped: image_id is empty", flush=True)
            continue
        image_path = _find_image(args.image_dir, image_id)
        pcf_path = pcf_dir / f"{image_id}.json"
        gm_path = gm_dir / f"{image_id}.json"
        if not args.overwrite and pcf_path.exists() and gm_path.exists():
            print(f"[{index}] {image_id}: skipped (both outputs exist)", flush=True)
            continue
        metadata = {"image_id": image_id, "patient_id": row.get("patient_id", ""), "tsv_row": row}
        pcf_result = None
        if args.overwrite or not pcf_path.exists():
            pcf_result = _pcf(row, retries=args.retries, timeout=args.timeout)
            _json_dump(pcf_path, {**metadata, "tool": "PubCaseFinder", "result": pcf_result})
        if image_path is None:
            gm_result = {"status": "error", "error": f"image not found for image_id={image_id}", "image_dir": str(args.image_dir)}
        elif args.overwrite or not gm_path.exists():
            gm_result = _gm(row, image_path, retries=args.retries, timeout=args.timeout)
        else:
            gm_result = None
        if args.overwrite or not gm_path.exists():
            _json_dump(gm_path, {**metadata, "image_path": str(image_path) if image_path else None, "tool": "GestaltMatcher", "result": gm_result})
        print(f"[{index}/{len(rows)}] {image_id}: PCF={pcf_result and pcf_result.get('status')} GM={gm_result and gm_result.get('status')}", flush=True)
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
