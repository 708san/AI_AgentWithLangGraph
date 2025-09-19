from ..state.state_types import State, webresource
from typing import List
from ddgs import DDGS
from ..llm.azure_llm_instance import azure_llm

webresearch_prompt_dict = {
   "generate_query_prompt": """You are a medical research assistant specializing in clinical genetics and bioinformatics. Your task is to generate effective DDGS(DuckDuckGo Search) queries to identify potential syndromes or genetic disorders based on a provided list of Human Phenotype Ontology (HPO) terms.

Instructions:

1. Review all provided HPO terms.
2. Strategically group symptoms and create several queries by combining different key features. Do not simply list all symptoms in one query.
3. Prioritize the most distinctive and clinically significant features (e.g., craniofacial anomalies, overgrowth features) in your combinations.
4. Include at least one query focusing on a logical subgroup of symptoms (e.g., endocrine/hormonal issues or neurodevelopmental features).
5. Use medical search terms such as "syndrome", "disorder", "genetic differential diagnosis", or "association" to focus results on medical literature and databases.
6. Output exactly 2 unique, concise, and ready-to-use DDGS(DuckDuckGo Search) query strings as a numbered list.

HPO term list:
{hpo_terms}

Follow the instructions and generate 2 search queries in English.
""",
   "summarize_results_prompt":  """ROLE:
Act as a medical doctor summarizing a clinical article for a colleague.

TASK:
Read the provided article text and summarize it into a single, concise paragraph.

INSTRUCTIONS:

Primary Focus: The summary must concentrate on the core message, specifically the connection between phenotypes (clinical features) and their related diseases or syndromes.

Content Selection: Extract only the most critical information and key findings. Ignore background details, methodology, or tangential points.

Conditional Check: Before summarizing, first determine if the article is medically related. If the content is not about medicine, biology, or clinical science, you must ignore all other rules and output the exact phrase: not a medical-related page.

Output Format: If the article is medical, the final output must be a single paragraph only.

ARTICLE TO SUMMARIZE:

{article_text}
"""
}

def extract_hpo_labels(hpo_dict: dict) -> List[str]:
    return list(hpo_dict.values())

def generate_queries(hpo_labels: List[str]) -> List[str]:
    prompt = webresearch_prompt_dict["generate_query_prompt"].format(hpo_terms=', '.join(hpo_labels))
    # Azure OpenAI でクエリ生成
    queries_msg = azure_llm.generate(prompt)
    # 返り値が文字列の場合は分割、リストの場合はそのまま
    if isinstance(queries_msg, str):
        # 1. ... 2. ... の形式を想定して分割
        queries = [q.strip().lstrip("0123456789. ") for q in queries_msg.split('\n') if q.strip()]
        queries = [q for q in queries if q]  # 空要素除去
        return queries[:2]
    elif hasattr(queries_msg, "content"):
        # content属性がある場合
        lines = queries_msg.content.split('\n')
        queries = [q.strip().lstrip("0123456789. ") for q in lines if q.strip()]
        queries = [q for q in queries if q]
        return queries[:2]
    else:
        return list(queries_msg)[:2]

def summarize_content(article_text: str) -> str:
    prompt = webresearch_prompt_dict["summarize_results_prompt"].format(article_text=article_text)
    summary_msg = azure_llm.generate(prompt)
    summary = summary_msg.content if hasattr(summary_msg, "content") else str(summary_msg)
    return summary.strip()

def search_hpo_terms(state: State) -> List[webresource]:
    hpo_labels = extract_hpo_labels(state["hpoDict"])
    queries = generate_queries(hpo_labels)
    new_webresources = []
    existing_urls = {w["url"] for w in state.get("webresources", [])}
    with DDGS() as ddgs:
        for query in queries:
            results = list(ddgs.text(query, max_results=2))
            for result in results:
                url = result.get("href") or result.get("url")
                title = result.get("title") or ""
                snippet = result.get("body") or result.get("snippet") or ""
                if not url or url in existing_urls:
                    continue
                summary = summarize_content(snippet)
                if summary.lower().startswith("not a medical-related page"):
                    continue
                new_webresources.append(webresource(
                    title=title,
                    url=url,
                    snippet=summary
                ))
                existing_urls.add(url)
    return new_webresources