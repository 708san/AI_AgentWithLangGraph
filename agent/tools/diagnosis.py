from langchain.schema import HumanMessage
from typing_extensions import List, Optional
from ..state.state_types import PCFres, DiagnosisOutput
from ..llm.prompt import prompt_dict, build_prompt
from ..llm.azure_llm_instance import azure_llm

def format_webresources(webresources: list) -> str:
    """
    Webリソースリストを診断プロンプト用のテキストに整形
    """
    if not webresources:
        return "No relevant web search results found."
    lines = []
    for i, res in enumerate(webresources, 1):
        title = res.get("title", "")
        url = res.get("url", "")
        snippet = res.get("snippet", "")
        lines.append(f"{i}. {title}\nURL: {url}\nSummary: {snippet}")
    return "\n".join(lines)


def createDiagnosis(hpo_dict: dict[str,str], pubCaseFinder: List[PCFres], zeroShotResult, gestaltMatcherResult, webresources=None, absent_hpo_dict=None) -> Optional[DiagnosisOutput]:
    top_str = "\n".join(
        [f"{i+1}. {item['omim_disease_name_en']} (score: {item['score']}) - {item['description']}" for i, item in enumerate(pubCaseFinder)]
    )
    zeroShotResult_str = ""
    if zeroShotResult and hasattr(zeroShotResult, "ans"):
        zeroShotResult_str = "\n".join([
            f"{i+1}. {item.disease_name} (rank: {item.rank})"
            for i, item in enumerate(zeroShotResult.ans)
        ])
    # --- GestaltMatcherの結果を整形 ---
    gestaltMatcherResult_str = ""
    if gestaltMatcherResult:
        gestaltMatcherResult_str = "\n".join([
            f"{i+1}. Disease Name: OMIM{item.get('omim_id', '')}:{item.get('syndrome_name', '')}), "
            f"Distance: The distance from reference case of the disease is {item.get('distance', '')} in feature space."
            for i, item in enumerate(gestaltMatcherResult)
        ])
    # --- Webリソースの整形 ---
    webresources_str = format_webresources(webresources) if webresources is not None else "No web search results provided."
    # ---------------------------------

    inputs = {
        "hpo_list": ", ".join([v for k, v in hpo_dict.items()]),
        "absent_hpo_list": ", ".join([v for k, v in (absent_hpo_dict or {}).items()]),
        "pcf_result": top_str,
        "zeroShotResult": zeroShotResult_str,
        "gestaltMatcherResult": gestaltMatcherResult_str,
        "web_search_results": webresources_str
    }
    # プロンプトテンプレートにweb_search_resultsを追加する必要あり
    prompt_template = (
        prompt_dict["diagnosis_prompt"] +
        "\n**5. Web Search Results (Literature/Case Reports):**\n{web_search_results}\n"
    )
    structured_llm = azure_llm.get_structured_llm(DiagnosisOutput)

    prompt = build_prompt(prompt_template, inputs)

    messages = [HumanMessage(content=prompt)]
    result = structured_llm.invoke(messages)
    return result, prompt