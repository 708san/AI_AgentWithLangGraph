from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.retrievers import PubMedRetriever, WikipediaRetriever
from ..state.state_types import State, InformationItem
from ..llm.llm_wrapper import AzureOpenAIWrapper
import os
import time
import random


DISEASE_SEARCH_MAX_WORKERS = int(os.getenv("DISEASE_SEARCH_MAX_WORKERS", "4"))


def summarize_text(text: str, llm: AzureOpenAIWrapper) -> str:
    """入力テキストを要約する関数"""
    try:
        prompt = """
You are an expert clinical geneticist and a diagnostician. Your critical task is to analyze a medical text and convert it into a high-yield, structured summary designed specifically for differential diagnosis. Your output must not only list symptoms but also highlight features that distinguish the condition from its clinical mimics.

Instructions:

From the text I provide, generate a summary strictly following these rules:

1. Information to Extract (Include ONLY these):

Disease: The name of the syndrome or disorder.

Genetics: The causative gene(s) and inheritance pattern. If not specified, state "Not specified".

Key Phenotypes: A concise, bulleted list of the core clinical features and symptoms.

Differentiating Features: This is the most critical section. Extract features that are particularly useful for distinguishing this syndrome from others. This includes:

Hallmark signs: Features that are highly characteristic or pathognomonic.

Key negative findings: Symptoms typically ABSENT in this condition but present in similar ones (e.g., "Absence of hyperphagia").

Unique constellations: A specific combination of symptoms that points strongly to this diagnosis.

2. Information to Exclude (Strictly Omit):

Patient case histories, family origins, or demographic details.

Treatment, management, or therapeutic strategies.

Research methodology, study populations, or author details.

Prognosis, mortality, or prevalence statistics.

General background information that isn't a clinical feature.

3. Output Format (Use this exact structure):

Disease: [Name of the disease]
Genetics: [Gene(s), Inheritance pattern]
Key Phenotypes:

[Bulleted list of core clinical features]

[Example: Intellectual disability]

[Example: Craniofacial dysmorphism]
Differentiating Features:

Hallmark(s): [List highly specific or unique signs.]

Key Negative Finding(s): [List what is typically absent, e.g., "Absence of..."]

Unique Constellation: [Describe a diagnostically powerful combination of symptoms.]

Now, process the following text:

""" + text
        summary_msg = llm.generate(prompt)
        summary = summary_msg.content if hasattr(summary_msg, "content") else str(summary_msg)
        return summary.strip()
    except Exception as e:
        print(f"要約時にエラー: {e}")
        return text


def search_single_disease_wikipedia(disease_name: str, search_depth: int, llm: AzureOpenAIWrapper) -> List[Dict[str, Any]]:
    """
    1つの疾患についてWikipediaを検索する（並列実行用）
    retrieved_urlsチェックは呼び出し側で行うため、ここでは全結果を返す
    """
    results = []
    try:
        wiki_retriever = WikipediaRetriever(top_k_results=search_depth * 1, doc_content_chars_max=2000)
        print(f"    - [Wikipedia] 「{disease_name}」を検索中...")
        wiki_docs = wiki_retriever.invoke(disease_name)

        for doc in wiki_docs:
            url = doc.metadata.get("source", "N/A")
            summary = summarize_text(doc.page_content, llm)
            results.append({
                "title": doc.metadata.get("title", disease_name),
                "url": url,
                "content": f"[Source: Wikipedia] {summary}",
                "disease_name": disease_name
            })
    except Exception as e:
        print(f"    - [Wikipedia] 「{disease_name}」の検索でエラー: {e}")
    
    return results


def search_single_disease_pubmed(disease_name: str, search_depth: int, llm: AzureOpenAIWrapper) -> List[Dict[str, Any]]:
    """
    1つの疾患についてPubMedを検索する（並列実行用）
    429エラー時はリトライする
    """
    results = []
    max_retries = 3
    base_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            pubmed_retriever = PubMedRetriever(top_k_results=search_depth * 3, doc_content_chars_max=3000)
            print(f"    - [PubMed] 「{disease_name}」を検索中...")
            pubmed_docs = pubmed_retriever.invoke(disease_name)

            for doc in pubmed_docs:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{doc.metadata['uid']}/"
                summary = summarize_text(doc.page_content, llm)
                results.append({
                    "title": doc.metadata.get("Title", disease_name),
                    "url": url,
                    "content": f"[Source: PubMed] {summary}",
                    "disease_name": disease_name
                })
            
            return results
            
        except Exception as e:
            error_msg = str(e)
            
            if "429" in error_msg or "Too Many Requests" in error_msg:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"    - [PubMed] 「{disease_name}」でレート制限エラー (429)")
                    print(f"      -> {delay:.1f}秒待機してリトライします...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"    - [PubMed] 「{disease_name}」で最大リトライ回数到達。スキップします。")
                    return []
            else:
                print(f"    - [PubMed] 「{disease_name}」の検索でエラー: {e}")
                return []
    
    return results


def diseaseSearchForDiagnosis(state: State) -> Dict[str, List[InformationItem]]:
    """
    暫定診断リストの各疾患について知識検索を並列実行し、重複を避けながらStateのmemoryに結果を追加する。
    """
    print("🔬 知識検索を開始します（並列処理）...")
    start_time = time.time()

    # Stateから必要な情報を取得
    llm = state.get("llm")
    tentativeDiagnosis = state.get("tentativeDiagnosis")
    search_depth = state.get("depth", 1)

    if not llm:
        print("LLMインスタンスがstate内に見つかりません。検索をスキップします。")
        return {"memory": state.get("memory", [])}

    # 既存のmemoryと、そこに含まれるURLのセットを取得
    memory = state.get("memory", [])
    retrieved_urls = {item['url'] for item in memory}

    if not tentativeDiagnosis or not hasattr(tentativeDiagnosis, "ans"):
        print("暫定診断が見つからないため、検索をスキップします。")
        return {"memory": memory}

    disease_names = [diag.disease_name for diag in tentativeDiagnosis.ans]
    if not disease_names:
        print("検索対象の疾患名がないため、スキップします。")
        return {"memory": memory}

    print(f"  - 検索深度: {search_depth}, 対象疾患: {disease_names}")

    # --- 並列実行の準備 ---
    max_workers = min(len(disease_names) * 2, DISEASE_SEARCH_MAX_WORKERS)
    print(f"  - 知識検索 max_workers: {max_workers}")
    
    # 並列実行で取得した全結果を一時保存
    all_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 全ての検索タスクを投入
        futures = {}
        
        # Wikipedia検索タスク
        for disease_name in disease_names:
            future = executor.submit(
                search_single_disease_wikipedia,
                disease_name,
                search_depth,
                llm
            )
            futures[future] = ('wikipedia', disease_name)
        
        # PubMed検索タスク
        for disease_name in disease_names:
            future = executor.submit(
                search_single_disease_pubmed,
                disease_name,
                search_depth,
                llm
            )
            futures[future] = ('pubmed', disease_name)
        
        # 結果を収集（as_completedで完了順に処理）
        completed_count = 0
        total_tasks = len(futures)
        
        for future in as_completed(futures):
            source, disease_name = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                
                completed_count += 1
                print(f"  進捗: {completed_count}/{total_tasks} 完了 ({source}: {disease_name}, {len(results)}件)")
                
            except Exception as e:
                print(f"    - [{source}] 「{disease_name}」の処理でエラー: {e}")
                completed_count += 1

    # --- スレッドプール終了後に重複チェックして追加（スレッドセーフ） ---
    new_items_count = 0
    for item in all_results:
        if item['url'] not in retrieved_urls:
            memory.append(item)
            retrieved_urls.add(item['url'])
            new_items_count += 1

    elapsed_time = time.time() - start_time
    print(f"✅ 知識検索が完了しました（{elapsed_time:.2f}秒, {new_items_count}件の新規情報を追加）")

    return {"memory": memory}
