from typing import List, Dict, Any
from langchain_community.retrievers import PubMedRetriever, WikipediaRetriever
from ..state.state_types import State, InformationItem
from ..llm.azure_llm_instance import azure_llm


def summarize_text(text: str) -> str:
    """
    入力テキストを要約する関数（azure_llmを利用）
    """
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
        summary_msg = azure_llm.generate(prompt)
        summary = summary_msg.content if hasattr(summary_msg, "content") else str(summary_msg)
        return summary.strip()
    except Exception as e:
        print(f"要約時にエラー: {e}")
        return text  

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
        wiki_retriever = WikipediaRetriever(top_k_results=search_depth * 3, doc_content_chars_max=3000)
        for name in disease_names:
            print(f"    - [Wikipedia] 「{name}」を検索中...")
            wiki_docs = wiki_retriever.invoke(name)
            
            for doc in wiki_docs:
                url = doc.metadata.get("source", "N/A")
                if url not in retrieved_urls:
                    print(f"      - 新規情報を追加: {url}")
                    summary = summarize_text(doc.page_content)
                    memory.append({
                        "title": doc.metadata.get("title", name),
                        "url": url,
                        "content": f"[Source: Wikipedia] {summary}",
                        "disease_name": name
                    })
                    retrieved_urls.add(url)
    except Exception as e:
        print(f"    - [Wikipedia] 検索でエラーが発生しました: {e}")

    # --- PubMedからの情報取得 ---
    try:
        pubmed_retriever = PubMedRetriever(top_k_results=search_depth * 3, doc_content_chars_max=3000)
        for name in disease_names:
            print(f"    - [PubMed] 「{name}」を検索中...")
            pubmed_docs = pubmed_retriever.invoke(name)

            for doc in pubmed_docs:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{doc.metadata['uid']}/"
                if url not in retrieved_urls:
                    print(f"      - 新規情報を追加: {url}")
                    summary = summarize_text(doc.page_content)
                    memory.append({
                        "title": doc.metadata.get("Title", name),
                        "url": url,
                        "content": f"[Source: PubMed] {summary}",
                        "disease_name": name  
                    })
                    retrieved_urls.add(url)
    except Exception as e:
        print(f"    - [PubMed] 検索でエラーが発生しました: {e}")
            
    print("✅ 知識検索が完了しました。")
    
    return {"memory": memory}
    