
import requests

def callingPCF(hpo_list , depth):
    hpo_ids = ",".join(hpo_list)
    url = f"https://pubcasefinder.dbcls.jp/api/pcf_get_ranked_list?target=omim&format=json&hpo_id={hpo_ids}"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        top = []
        for item in data[:5]:
            top.append({
                "omim_disease_name_en": item.get("omim_disease_name_en", ""),
                "description": item.get("description", ""),
                "score": item.get("score", None),
                "omim_id": item.get("id", "")
            })
        return top
    except Exception as e:
        print(f"[PhenotypeAnalyzer] PubCaseFinder API失敗: {e}")
        return []
"""

from ..mcp_client_instance import mcp_client # ステップ1で作成したクライアントをインポート

def callingPCF(hpo_list):
    if not mcp_client:
        print("[PhenotypeAnalyzer] MCP client is not available.")
        return []

    hpo_ids = ",".join(hpo_list)
    
    try:
        # MCPクライアント経由でツールを呼び出す
        # "pcf_get_ranked_list" はMCPサーバ側で定義されたツール名を指定
        data = mcp_client.invoke(
            "pcf_get_ranked_list", # ツール名
            target="omim",         # ツールに渡す引数
            format="json",
            hpo_id=hpo_ids
        )
        
        # MCPサーバからのレスポンスを整形する処理はそのまま
        top5 = []
        for item in data[:5]:
            top5.append({
                "omim_disease_name_en": item.get("omim_disease_name_en", ""),
                "description": item.get("description", ""),
                "score": item.get("score", None)
            })
        return top5
    except Exception as e:
        print(f"[PhenotypeAnalyzer] MCP Tool call failed: {e}")
        return []
"""