# This tool is responsible for searching information using disease names.
from typing import List, Dict, Any
from langchain_community.retrievers import PubMedRetriever, WikipediaRetriever
from state.state_types import State, InformationItem 


def diseaseSearchForDiagnosis(state: State) -> Dict[str, List[InformationItem]]:
    """
    暫定診断リストの各疾患について知識検索を実行し、重複を避けながらStateのmemoryに結果を追加する。
    """
    print("🔬 知識検索を開始します...")
    
    # Stateから必要な情報を取得
    tentativeDiagnosis = state.get("tentativeDiagnosis")
    search_depth = state.get("depth", 1)
    
    # 既存のmemoryと、そこに含まれるURLのセットを取得
    memory = state.get("memory", [])
    retrieved_urls = {item['url'] for item in memory}

    if not tentativeDiagnosis or not hasattr(tentativeDiagnosis, "ans"):
        print("暫定診断が見つからないため、検索をスキップします。")
        # 変更がない場合でも、現在のmemoryを返すのが安全
        return {"memory": memory}

    disease_names = [diag.disease_name for diag in tentativeDiagnosis.ans]
    if not disease_names:
        print("検索対象の疾患名がないため、スキップします。")
        return {"memory": memory}

    print(f"  - 検索深度: {search_depth}, 対象疾患: {disease_names}")

    # --- Wikipediaからの情報取得 ---
    try:
        # 検索深度に応じて取得するドキュメント数を変更 (例: depth * 1)
        wiki_retriever = WikipediaRetriever(top_k_results=search_depth * 5, doc_content_chars_max=1500)
        for name in disease_names:
            print(f"    - [Wikipedia] 「{name}」を検索中...")
            wiki_docs = wiki_retriever.invoke(name)
            
            for doc in wiki_docs:
                url = doc.metadata.get("source", "N/A")
                if url not in retrieved_urls:
                    print(f"      - 新規情報を追加: {url}")
                    memory.append({
                        "title": doc.metadata.get("title", name),
                        "url": url,
                        "content": f"[Source: Wikipedia] {doc.page_content}",
                        "disease_name": name
                    })
                    retrieved_urls.add(url)
    except Exception as e:
        print(f"    - [Wikipedia] 検索でエラーが発生しました: {e}")

    # --- PubMedからの情報取得 ---
    try:
        # 検索深度に応じて取得する論文数を変更 (例: depth * 2)
        pubmed_retriever = PubMedRetriever(top_k_results=search_depth * 2)
        for name in disease_names:
            print(f"    - [PubMed] 「{name}」を検索中...")
            pubmed_docs = pubmed_retriever.invoke(name)

            for doc in pubmed_docs:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{doc.metadata['uid']}/"
                if url not in retrieved_urls:
                    print(f"      - 新規情報を追加: {url}")
                    memory.append({
                        "title": doc.metadata.get("Title", name),
                        "url": url,
                        "content": f"[Source: PubMed] {doc.page_content}",
                        "disease_name": name  
                    })
                    retrieved_urls.add(url)
    except Exception as e:
        print(f"    - [PubMed] 検索でエラーが発生しました: {e}")
            
    print("✅ 知識検索が完了しました。")
    
    # 更新されたmemoryをStateに返す
    return {"memory": memory}