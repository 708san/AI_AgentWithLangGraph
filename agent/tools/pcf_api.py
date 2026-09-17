
import os
import requests
import time

def callingPCF(hpo_list, depth, max_retries=3, return_full=False):
    hpo_ids = ",".join(hpo_list)
    url = f"https://pubcasefinder.dbcls.jp/api/pcf_get_ranked_list?target=omim&format=json&hpo_id={hpo_ids}"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            data = response.json()
            all_results = []
            for rank, item in enumerate(data, 1):
                all_results.append({
                    "omim_disease_name_en": item.get("omim_disease_name_en", ""),
                    "description": item.get("description", ""),
                    "score": item.get("score", None),
                    "omim_id": item.get("id", ""),
                    "rank": rank,
                })
            if return_full:
                return {
                    "top5": all_results[:5],
                    "all": all_results,
                    "raw": data,
                    "request": {
                        "url": url,
                        "hpo_ids": list(hpo_list),
                        "depth": depth,
                    },
                }
            return all_results[:5]
        except Exception as e:
            print(f"[PhenotypeAnalyzer] PubCaseFinder API失敗 (試行 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数バックオフ: 1秒, 2秒, 4秒
                print(f"  {wait_time}秒後にリトライします...")
                time.sleep(wait_time)
            else:
                print("  最大リトライ回数に達しました。空のリストを返します。")
                return []
