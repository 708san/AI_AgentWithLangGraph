import base64
import requests
import json
import os
from dotenv import load_dotenv

def call_gestalt_matcher_api(image_path: str, depth: int):
    """
    画像ファイルのパスを受け取り、GestaltMatcher APIを叩いて
    suggested_genes_listの上位(depth+2)件だけをリストで返す関数。
    認証情報は環境変数 GESTALT_API_USER, GESTALT_API_PASS から取得

    Args:
        image_path (str): 画像ファイルのパス
        depth (int): 返す件数を決めるための基準

    Returns:
        list: suggested_genes_listの上位(depth+2)件
    """
    load_dotenv()
    api_url = "https://dev-pubcasefinder.dbcls.jp/gm_endpoint/predict"
    username = os.environ.get("GESTALT_API_USER")
    password = os.environ.get("GESTALT_API_PASS")
    if not username or not password:
        raise ValueError("環境変数 GESTALT_API_USER または GESTALT_API_PASS が設定されていません。")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    headers = {"Content-Type": "application/json"}
    payload = {"img": img_b64}

    response = requests.post(
        api_url,
        headers=headers,
        data=json.dumps(payload),
        auth=(username, password)
    )

    response.raise_for_status()
    result = response.json()
    syndromes = result.get("suggested_syndromes_list", [])
    return syndromes[:depth + 4]