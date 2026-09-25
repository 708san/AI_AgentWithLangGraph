#!/usr/bin/env python3
"""Call PubCaseFinder once and print the response summary.

No files are created or modified.  The request format follows
``run_tools_for_new_data.py`` and ``agent/tools/pcf_api.py``.

Example::

    .venv/bin/python tryToolsForNewData/call_pcf_once.py \
        HP:0001263 HP:0001252 HP:0001508
"""

from __future__ import annotations

import argparse
import json
import time

import requests


PCF_URL = "https://pubcasefinder.dbcls.jp/api/pcf_get_ranked_list"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hpo_ids", nargs="+", help="HPO IDs, for example HP:0001263 HP:0001252")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    hpo_ids = []
    for value in args.hpo_ids:
        hpo_ids.extend(item.strip() for item in value.replace(",", ";").split(";") if item.strip())
    hpo_ids = list(dict.fromkeys(hpo_ids))
    params = {
        "target": "omim",
        "format": "json",
        "hpo_id": ",".join(hpo_ids),
    }

    print("request_url:", PCF_URL)
    print("hpo_ids:", ",".join(hpo_ids))

    for attempt in range(1, args.retries + 1):
        started = time.perf_counter()
        try:
            response = requests.get(PCF_URL, params=params, timeout=args.timeout)
            elapsed = time.perf_counter() - started
            print("resolved_url:", response.url)
            print("status_code:", response.status_code)
            print("elapsed_sec:", round(elapsed, 2))
            response.raise_for_status()

            raw = response.json()
            if not isinstance(raw, list):
                print("response_type:", type(raw).__name__)
                print(json.dumps(raw, ensure_ascii=False, indent=2)[:2000])
                return 1

            print("response_type: list")
            print("result_count:", len(raw))
            print("top5:")
            for rank, item in enumerate(raw[:5], start=1):
                if not isinstance(item, dict):
                    print(f"  {rank}. {item}")
                    continue
                print(json.dumps({
                    "rank": rank,
                    "omim_id": item.get("id", ""),
                    "disease_name": item.get("omim_disease_name_en", ""),
                    "score": item.get("score"),
                }, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(f"attempt {attempt}/{args.retries} failed: {type(exc).__name__}: {exc}")
            if attempt < args.retries:
                time.sleep(2 ** (attempt - 1))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
