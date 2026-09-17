import os

from .state.state_types import State, ReflectionOutput
from .tools.pcf_api import callingPCF
from .tools.diagnosis import createDiagnosis
from .tools.ZeroShot import createZeroshot
from .tools.make_HPOdic import make_hpo_dic
from .tools.reflection import create_reflection
from .tools.diseaseSearch import diseaseSearchForDiagnosis
from .tools.diseaseNormalize import diseaseNormalizeForDiagnosis, normalize_pcf_results, normalize_gestalt_results, normalize_zeroshot_results
from .tools.finalDiagnosis import createFinalDiagnosis
from .tools.gestaltMathcher import call_gestalt_matcher_api
from .tools.HPOwebReserch import search_hpo_terms
from .tools.embeddingSearchWithHPO import embedding_search_with_hpo
from .tools.rankingMerge import merge_ranked_disease_candidates
from .tools.phenobrain_api import call_phenobrain
from .tools.togomcp_search import (
    discover_diseases_with_togomcp,
    admit_discovered_candidates,
    research_candidates_with_togomcp,
)

from .utils.result_saver import save_result
from .utils.profiler import profile_node

from concurrent.futures import ThreadPoolExecutor, as_completed


REFLECTION_MAX_WORKERS = int(os.getenv("REFLECTION_MAX_WORKERS", "6"))
TOOL_FULL_RESULT_LIMIT = int(os.getenv("TOOL_FULL_RESULT_LIMIT", "100"))


def _append_tool_response_record(state: State, record: dict) -> list[dict]:
    # Initial tool nodes run in parallel.  Return only this node's delta and
    # let the State reducer concatenate records without write conflicts.
    return [record]


def _empty_reflection_output() -> ReflectionOutput:
    return ReflectionOutput(ans=[])

@profile_node
def BeginningOfFlowNode(state: State):
    print("BeginningOfFlowNode called")
    state["depth"] += 1
    print(f"Current depth: {state['depth']}")
    # reset Diagnosis and Reflection when starting a new flow
    
    return {"depth": state["depth"], "tentativeDiagnosis": None, "reflection": None}

@profile_node
@save_result("HPOwebSearchNode")
def HPOwebSearchNode(state: State):
    print("HPOwebSearchNode called")
    try:
        webresources = search_hpo_terms(state)
        
        # 既存のwebresourcesとマージ（重複排除はsearch_hpo_terms内で実施済み想定）
        state["webresources"] = state.get("webresources", []) + webresources
        return {"webresources": state["webresources"]}
    except Exception as e:
        print(f"Error in HPOwebSearchNode: {e}")
        return {"webresources": state.get("webresources", [])}
    
@profile_node
@save_result("DiseaseSearchWithHPONode")
def DiseaseSearchWithHPONode(state: State):
    """
    Searches for diseases with similar phenotypes based on the patient's HPO list using embedding vector search.
    """
    print("DiseaseSearchWithHPONode called")

    # Following the design principle of passing the state to tool functions.
    # It is expected that the search_phenotypes_by_embedding function will:
    # 1. Extract the hpo_list from the state.
    # 2. Execute the search.
    # 3. Return the results as a List[PhenotypeSearchFormat].
    response = embedding_search_with_hpo(state, return_metadata=True)

    if isinstance(response, dict):
        search_results = response.get("top5", [])
        tool_record = {
            "tool": "PhenotypeSearch",
            "request": response.get("request", {}),
            "response": response.get("raw", {}),
            "top5": [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in response.get("top5", [])
            ],
            "all": [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in response.get("all", [])
            ],
            "rank_status": "full_response",
        }
    else:
        search_results = response
        tool_record = None
    
    if not search_results:
        print("Phenotype search returned no results.")
        return {}
        
    result = {"phenotypeSearchResult": search_results}
    if tool_record is not None:
        result["tool_response_records"] = _append_tool_response_record(state, tool_record)
    return result


@profile_node
def PCFnode(state: State):
    print("PCFnode called")
    depth = state.get("depth", 0)
    hpo_list = state["hpoList"]
    if not hpo_list:
        return {"pubCaseFinder": []}
    response = callingPCF(hpo_list, depth, return_full=True)
    if isinstance(response, dict):
        return {
            "pubCaseFinder": response.get("top5", []),
            "tool_response_records": _append_tool_response_record(
                state,
                {
                    "tool": "PubCaseFinder",
                    "request": response.get("request", {}),
                    "response": response.get("raw", {}),
                    "top5": response.get("top5", []),
                    "all": response.get("all", []),
                    "rank_status": "full_response",
                },
            ),
        }
    return {"pubCaseFinder": response or []}

@profile_node
@save_result("PhenoBrainNode")
def PhenoBrainNode(state: State):
    print("PhenoBrainNode called")
    if not state.get("use_phenobrain", False):
        return {"phenoBrain": []}
    hpo_list = state.get("hpoList", [])
    if not hpo_list:
        return {"phenoBrain": []}
    response = call_phenobrain(
        hpo_list,
        topk=TOOL_FULL_RESULT_LIMIT,
        return_metadata=True,
    )
    if isinstance(response, dict):
        return {
            "phenoBrain": response.get("top5", []),
            "tool_response_records": _append_tool_response_record(
                state,
                {
                    "tool": "PhenoBrain",
                    "request": response.get("request", {}),
                    "response": response.get("raw", {}),
                    "top5": response.get("top5", []),
                    "all": response.get("all", []),
                    "rank_status": "full_response",
                },
            ),
        }
    return {"phenoBrain": response or []}

@profile_node
@save_result("NormalizePCFNode")
def NormalizePCFNode(state: State):
    """PCFの結果に含まれる病名をOMIM IDに基づいて正規化する"""
    print("NormalizePCFNode called")
    normalized_results = normalize_pcf_results(state)
    if not normalized_results:
        return {}
    return {"pubCaseFinder": normalized_results}

@profile_node
def GestaltMatcherNode(state: State):
    print("GestaltMatcherNode called")
    image_path = state.get("imagePath", None)
    depth = state.get("depth", 0)
    if not image_path:
        print("No image path provided.")
        return {"GestaltMatcher": []}
    try:
        response = call_gestalt_matcher_api(image_path, depth, return_full=True)
        if isinstance(response, dict):
            gestalt_results = response.get("top5", [])
            tool_record = {
                "tool": "GestaltMatcher",
                "request": response.get("request", {}),
                "response": response.get("raw", {}),
                "top5": response.get("top5", []),
                "all": response.get("all", []),
                "rank_status": "full_response",
            }
        else:
            gestalt_results = response or []
            tool_record = None
        syndrome_list = []
        for res in gestalt_results:
            syndrome_list.append({
                "subject_id": res.get("subject_id", ""),
                "syndrome_name": res.get("syndrome_name", ""),
                "omim_id": res.get("omim_id", ""),
                "image_id": res.get("image_id", ""),
                "score": res.get("score")
            })
        result = {"GestaltMatcher": syndrome_list}
        if tool_record is not None:
            result["tool_response_records"] = _append_tool_response_record(state, tool_record)
        return result
    except Exception as e:
        print(f"Error calling GestaltMatcher API: {e}")
        return {"GestaltMatcher": []}
    
@profile_node    
@save_result("NormalizeGestaltMatcherNode")
def NormalizeGestaltMatcherNode(state: State):
    """GestaltMatcherの結果に含まれる病名をOMIM IDに基づいて正規化する"""
    print("NormalizeGestaltMatcherNode called")
    normalized_results = normalize_gestalt_results(state)
    if not normalized_results:
        return {}
    return {"GestaltMatcher": normalized_results}

@profile_node
def createHPODictNode(state: State):
    print("createHPODictNode called")
    hpo_list = state.get("hpoList", [])
    hpo_dict = make_hpo_dic(hpo_list, None)
    return {"hpoDict": hpo_dict}

@profile_node
def createAbsentHPODictNode(state: State):
    print("createAbsentHPODictNode called")
    absent_hpo_list = state.get("absentHpoList", [])
    absent_hpo_dict = make_hpo_dic(absent_hpo_list, None)
    return {"absentHpoDict": absent_hpo_dict}

@profile_node
def createZeroShotNode(state: State):
    print("createZeroShotNode called")
    hpo_dict = state.get("hpoDict", {})
    if state.get("zeroShotResult") is not None:
        return {"zeroShotResult": state["zeroShotResult"]}
    if hpo_dict:
        # createZeroshotが(result, prompt)を返すように修正
        result, prompt = createZeroshot(state)
        if result:
            # promptはstateに保存しないので、ここでは返さない
            return {
                "zeroShotResult": result,
                "prompt": prompt,
                "tool_response_records": _append_tool_response_record(
                    state,
                    {
                        "tool": "ZeroShot",
                        "request": {"prompt": prompt},
                        "response": result.model_dump() if hasattr(result, "model_dump") else result,
                        "top5": [
                            item.model_dump() if hasattr(item, "model_dump") else item
                            for item in (getattr(result, "ans", []) or [])[:5]
                        ],
                        "all": [
                            item.model_dump() if hasattr(item, "model_dump") else item
                            for item in (getattr(result, "ans", []) or [])
                        ],
                        "rank_status": "llm_ranked_response",
                    },
                ),
            }
    return {"zeroShotResult": None}

@profile_node
@save_result("NormalizeZeroShotNode")
def NormalizeZeroShotNode(state: State):
    """ZeroShotの結果に含まれる病名を正規化し、重複を排除する"""
    print("NormalizeZeroShotNode called")
    # stateから値を取り出すのではなく、stateをそのまま渡す
    normalized_result = normalize_zeroshot_results(state)
    if not normalized_result:
        return {}
    # 既存のキー 'zeroShotResult' を上書きする
    return {"zeroShotResult": normalized_result}

@profile_node
@save_result("mergeCandidateResultsNode")
def mergeCandidateResultsNode(state: State):
    """各ツールの順位付き疾患候補を疾患単位に統合する"""
    print("mergeCandidateResultsNode called")
    merged_candidates = merge_ranked_disease_candidates(state)
    return {"mergedDiseaseCandidates": merged_candidates}


@profile_node
@save_result("TogoMCPDiscoveryNode")
def TogoMCPDiscoveryNode(state: State):
    """Discover additional phenotype-linked diseases through TogoMCP."""
    print("TogoMCPDiscoveryNode called")
    return discover_diseases_with_togomcp(state)


@profile_node
@save_result("TogoMCPCandidateAdmissionNode")
def TogoMCPCandidateAdmissionNode(state: State):
    """Promote only reviewed TogoMCP discoveries into the candidate pool."""
    print("TogoMCPCandidateAdmissionNode called")
    return admit_discovered_candidates(state)


@profile_node
@save_result("TogoMCPResearchNode")
def TogoMCPResearchNode(state: State):
    """Collect candidate-specific external evidence before reflection."""
    print("TogoMCPResearchNode called")
    return research_candidates_with_togomcp(state)

@profile_node
@save_result("createDiagnosisNode")
def createDiagnosisNode(state: State):
    """
    Gathers all preliminary reports and generates a tentative diagnosis by synthesizing them.
    """
    print("createDiagnosisNode called")
    
    # The createDiagnosis function now takes the entire state as input.
    # It will handle extracting all necessary information internally.
    result, prompt = createDiagnosis(state)
    
    if result:
        return {"tentativeDiagnosis": result, "prompt": prompt}
    
    return {}


@profile_node
@save_result("diseaseNormalizeNode")
def diseaseNormalizeNode(state: State):
    print("diseaseNormalizeNode called")
    tentativeDiagnosis = state.get("tentativeDiagnosis", None)
    if tentativeDiagnosis is not None:
        normalizedDiagnosis = diseaseNormalizeForDiagnosis(tentativeDiagnosis)
        return {"tentativeDiagnosis": normalizedDiagnosis}
    return {"tentativeDiagnosis": None}

@profile_node
def diseaseSearchNode(state: State):
    print("diseaseSearchNode called")
    
    return diseaseSearchForDiagnosis(state)

@profile_node
@save_result("reflectionNode")
def reflectionNode(state: State):
    print("reflectionNode called")
    tentativeDiagnosis = state.get("tentativeDiagnosis")
    hpo_dict = state.get("hpoDict")
    
    if tentativeDiagnosis and hpo_dict and hasattr(tentativeDiagnosis, 'ans'):
        diagnosis_to_judge_lis = tentativeDiagnosis.ans
        if not diagnosis_to_judge_lis:
            return {"reflection": _empty_reflection_output()}

        reflection_result_list = []
        prompts = []
        
        # 並列実行関数
        def process_single_reflection(diagnosis_to_judge):
            try:
                reflection_result, prompt = create_reflection(state, diagnosis_to_judge)
                return reflection_result, prompt
            except Exception as e:
                print(f"[ERROR] Reflection failed for {diagnosis_to_judge.disease_name}: {e}")
                return None, None
        
        max_workers = min(len(diagnosis_to_judge_lis), REFLECTION_MAX_WORKERS)
        print(f"[Reflection] max_workers={max_workers}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_diagnosis = {
                executor.submit(process_single_reflection, diagnosis): diagnosis 
                for diagnosis in diagnosis_to_judge_lis
            }

            for future in as_completed(future_to_diagnosis):
                diagnosis = future_to_diagnosis[future]
                try:
                    reflection_result, prompt = future.result()
                    if reflection_result:
                        reflection_result_list.append(reflection_result)
                        prompts.append(prompt)
                except Exception as e:
                    print(f"[ERROR] Future exception for {diagnosis.disease_name}: {e}")
        
        if not reflection_result_list:
            return {"reflection": _empty_reflection_output()}
        
        reflection_output = ReflectionOutput(ans=reflection_result_list)
        return {"reflection": reflection_output, "prompt": prompts}
    
    return {"reflection": _empty_reflection_output()}


@profile_node
@save_result("finalDiagnosisNode")
def finalDiagnosisNode(state: State):
    print("finalDiagnosisNode called")
    finalDiagnosis, prompt = createFinalDiagnosis(state)
    return {"finalDiagnosis": finalDiagnosis, "prompt": prompt}

@profile_node
@save_result("diseaseNormalizeForFinalNode")
def diseaseNormalizeForFinalNode(state: State):
    print("diseaseNormalizeForFinalNode called")
    finalDiagnosis = state.get("finalDiagnosis", None)
    if finalDiagnosis is not None:
        normalizedDiagnosis = diseaseNormalizeForDiagnosis(finalDiagnosis)
        return {"finalDiagnosis": normalizedDiagnosis}
    return {"finalDiagnosis": None}


"""
@save_result("reflectionNode")
def reflectionNode(state: State):
    print("reflectionNode called")
    tentativeDiagnosis = state.get("tentativeDiagnosis")
    hpo_dict = state.get("hpoDict")
    if tentativeDiagnosis and hpo_dict:
        diagnosis_to_judge_lis = tentativeDiagnosis.ans
        reflection_result_list = []
        prompts = []
        for diagnosis_to_judge in diagnosis_to_judge_lis:
            reflection_result, prompt = create_reflection(
                state, diagnosis_to_judge
            )
            reflection_result_list.append(reflection_result)
            prompts.append(prompt)
        print(type(reflection_result_list[0]))
        return {"reflection": {"ans": reflection_result_list}, "prompt": prompts}
    return {"reflection": None}
"""
