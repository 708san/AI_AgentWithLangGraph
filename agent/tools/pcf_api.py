import requests

def callingPCF(hpo_list):
    hpo_ids = ",".join(hpo_list)
    url = f"https://pubcasefinder.dbcls.jp/api/pcf_get_ranked_list?target=omim&format=json&hpo_id={hpo_ids}"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        top5 = []
        for item in data[:5]:
            top5.append({
                "omim_disease_name_en": item.get("omim_disease_name_en", ""),
                "description": item.get("description", ""),
                "score": item.get("score", None)
            })
        return top5
    except Exception as e:
        print(f"[PhenotypeAnalyzer] PubCaseFinder API失敗: {e}")
        return []