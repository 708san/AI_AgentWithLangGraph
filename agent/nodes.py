from typing_extensions import List, Optional
from .state.state_types import State, PCFres, DiagnosisOutput, ReflectionOutput
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

from .utils.result_saver import save_result

import os
import json

def BeginningOfFlowNode(state: State):
    print("BeginningOfFlowNode called")
    state["depth"] += 1
    print(f"Current depth: {state['depth']}")
    # reset Diagnosis and Reflection when starting a new flow
    
    return {"depth": state["depth"], "tentativeDiagnosis": None, "reflection": None}

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
    search_results = embedding_search_with_hpo(state)
    
    if not search_results:
        print("Phenotype search returned no results.")
        return {}
        
    return {"phenotypeSearchResult": search_results}


@save_result("PCFNode")
def PCFnode(state: State):
    print("PCFnode called")
    depth = state.get("depth", 0)
    hpo_list = state["hpoList"]
    if not hpo_list:
        return {"pubCaseFinder": []}
    result = callingPCF(hpo_list, depth)
    return {"pubCaseFinder": result}



def NormalizePCFNode(state: State):
    """PCFの結果に含まれる病名をOMIM IDに基づいて正規化する"""
    print("NormalizePCFNode called")
    normalized_results = normalize_pcf_results(state)
    if not normalized_results:
        return {}
    return {"pubCaseFinder": normalized_results}

@save_result("GestaltMatcherNode")
def GestaltMatcherNode(state: State):
    print("GestaltMatcherNode called")
    image_path = state.get("imagePath", None)
    depth = state.get("depth", 0)
    if not image_path:
        print("No image path provided.")
        return {"GestaltMatcher": []}
    try:
        gestalt_results = call_gestalt_matcher_api(image_path, depth)
        syndrome_list = []
        for res in gestalt_results:
            syndrome_list.append({
                "subject_id": res.get("subject_id", ""),
                "syndrome_name": res.get("syndrome_name", ""),
                "omim_id": res.get("omim_id", ""),
                "image_id": res.get("image_id", ""),
                "score": res.get("score")
            })
        return {"GestaltMatcher": syndrome_list}
    except Exception as e:
        print(f"Error calling GestaltMatcher API: {e}")
        return {"GestaltMatcher": []}
    

def NormalizeGestaltMatcherNode(state: State):
    """GestaltMatcherの結果に含まれる病名をOMIM IDに基づいて正規化する"""
    print("NormalizeGestaltMatcherNode called")
    normalized_results = normalize_gestalt_results(state)
    if not normalized_results:
        return {}
    return {"GestaltMatcher": normalized_results}

def createHPODictNode(state: State):
    print("createHPODictNode called")
    hpo_list = state.get("hpoList", [])
    hpo_dict = make_hpo_dic(hpo_list, None)
    return {"hpoDict": hpo_dict}

def createAbsentHPODictNode(state: State):
    print("createAbsentHPODictNode called")
    absent_hpo_list = state.get("absentHpoList", [])
    absent_hpo_dict = make_hpo_dic(absent_hpo_list, None)
    return {"absentHpoDict": absent_hpo_dict}


def createZeroShotNode(state: State):
    print("createZeroShotNode called")
    hpo_dict = state.get("hpoDict", {})
    absent_hpo_dict = state.get("absentHpoDict", {})
    if state.get("zeroShotResult") is not None:
        return {"zeroShotResult": state["zeroShotResult"]}
    if hpo_dict:
        # createZeroshotが(result, prompt)を返すように修正
        result, prompt = createZeroshot(hpo_dict, absent_hpo_dict=absent_hpo_dict,onset=state.get("onset", "Unknown"),sex=state.get("sex", "Unknown"))
        if result:
            # promptはstateに保存しないので、ここでは返さない
            return {"zeroShotResult": result, "prompt": prompt}
    return {"zeroShotResult": None}


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



@save_result("diseaseNormalizeNode")
def diseaseNormalizeNode(state: State):
    print("diseaseNormalizeNode called")
    tentativeDiagnosis = state.get("tentativeDiagnosis", None)
    if tentativeDiagnosis is not None:
        normalizedDiagnosis = diseaseNormalizeForDiagnosis(tentativeDiagnosis)
        return {"tentativeDiagnosis": normalizedDiagnosis}
    return {"tentativeDiagnosis": None}


def diseaseSearchNode(state: State):
    print("diseaseSearchNode called")
    
    return diseaseSearchForDiagnosis(state)


@save_result("reflectionNode")
def reflectionNode(state: State):
    print("reflectionNode called")
    tentativeDiagnosis = state.get("tentativeDiagnosis", None)
    hpo_dict = state.get("hpoDict", {})
    absent_hpo_dict = state.get("absentHpoDict", {})
    disease_knowledge = state.get("memory", [])
    patient_id = state.get("patient_id", "unknown")

    if tentativeDiagnosis and hpo_dict:
        diagnosis_to_judge_lis = tentativeDiagnosis.ans
        reflection_result_list = []
        prompts = []
        for diagnosis_to_judge in diagnosis_to_judge_lis:
            reflection_result, prompt = create_reflection(
                hpo_dict, diagnosis_to_judge, disease_knowledge,
                absent_hpo_dict=absent_hpo_dict,
                onset=state.get("onset", "Unknown"),
                sex=state.get("sex", "Unknown")
            )
            reflection_result_list.append(reflection_result)
            prompts.append(prompt)
        print(type(reflection_result_list[0]))
        return {"reflection": ReflectionOutput(ans=reflection_result_list), "prompt": "\n---\n".join(prompts)}
    return {"reflection": None}

@save_result("finalDiagnosisNode")
def finalDiagnosisNode(state: State):
    print("finalDiagnosisNode called")
    finalDiagnosis, prompt = createFinalDiagnosis(state)
    return {"finalDiagnosis": finalDiagnosis, "prompt": prompt}


@save_result("diseaseNormalizeForFinalNode")
def diseaseNormalizeForFinalNode(state: State):
    print("diseaseNormalizeForFinalNode called")
    finalDiagnosis = state.get("finalDiagnosis", None)
    if finalDiagnosis is not None:
        normalizedDiagnosis = diseaseNormalizeForDiagnosis(finalDiagnosis)
        return {"finalDiagnosis": normalizedDiagnosis}
    return {"finalDiagnosis": None}