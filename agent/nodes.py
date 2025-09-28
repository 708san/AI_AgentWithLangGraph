from typing_extensions import List, Optional
from .state.state_types import State, PCFres, DiagnosisOutput, ReflectionOutput
from .tools.pcf_api import callingPCF
from .tools.diagnosis import createDiagnosis
from .tools.ZeroShot import createZeroshot
from .tools.make_HPOdic import make_hpo_dic
from .tools.reflection import create_reflection
from .tools.diseaseSearch import diseaseSearchForDiagnosis
from .tools.diseaseNormalize import diseaseNormalizeForDiagnosis
from .tools.finalDiagnosis import createFinalDiagnosis
from .tools.gestaltMathcher import call_gestalt_matcher_api
from .tools.HPOwebReserch import search_hpo_terms

from agent.llm.prompt import prompt_dict

import os
import json

def save_node_result(node_name, result, patient_id):
    res_dir = os.path.join(os.path.dirname(__file__), '../res')
    os.makedirs(res_dir, exist_ok=True)
    out_path = os.path.join(res_dir, f"{patient_id}.json")
    # 既存ファイルがあれば読み込んで追記
    if os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}
    # Pydanticモデルやクラスインスタンスならdict化
    if hasattr(result, "dict"):
        data[node_name] = result.dict()
    elif isinstance(result, list):
        # リストの場合は各要素をdict化
        data[node_name] = [r.dict() if hasattr(r, "dict") else r for r in result]
    else:
        data[node_name] = result
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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
    
def BeginningOfFlowNode(state: State):
    print("BeginningOfFlowNode called")
    state["depth"] += 1
    print(f"Current depth: {state['depth']}")
    # reset Diagnosis and Reflection when starting a new flow
    
    return {"depth": state["depth"], "tentativeDiagnosis": None, "reflection": None}


def PCFnode(state: State):
    print("PCFnode called")
    depth = state.get("depth", 0)
    hpo_list = state["hpoList"]
    patient_id = state.get("patient_id", "unknown")
    if not hpo_list:
        result = []
        save_node_result("PCFnode", result, patient_id)
        return {"pubCaseFinder": result}
    result = callingPCF(hpo_list, depth)
    save_node_result("PCFnode", result, patient_id)
    return {"pubCaseFinder": result}

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
        patient_id = state.get("patient_id", "unknown")
        save_node_result("GestaltMatcherNode", syndrome_list, patient_id)
        return {"GestaltMatcher": syndrome_list}
    except Exception as e:
        print(f"Error calling GestaltMatcher API: {e}")
        return {"GestaltMatcher": []}

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
    patient_id = state.get("patient_id", "unknown")
    hpo_dict = state.get("hpoDict", {})
    absent_hpo_dict = state.get("absentHpoDict", {})
    if state.get("zeroShotResult") is not None:
        return {"zeroShotResult": state["zeroShotResult"]}
    if hpo_dict:
        # createZeroshotが(result, prompt)を返すように修正
        result, prompt = createZeroshot(hpo_dict, absent_hpo_dict=absent_hpo_dict,onset=state.get("onset", "Unknown"),sex=state.get("sex", "Unknown"))
        if result:
            save_node_result("createZeroShotNode", result, patient_id)
            return {"result": {"zeroShotResult": result}, "prompt": prompt}
    return {"result": {"zeroShotResult": None}}



def createDiagnosisNode(state: State):
    # To integrate streamss from both ZeroShot and PCF before diagnosis
    print("DiagnosisNode called")
    hpo_dict = state.get("hpoDict", {})
    absent_hpo_dict = state.get("absentHpoDict", {})
    pubCaseFinder = state.get("pubCaseFinder", [])
    zeroShotResult = state.get("zeroShotResult", None)
    gestaltMatcherResult = state.get("GestaltMatcher", None)
    webresources = state.get("webresources", [])


    if hpo_dict and pubCaseFinder:
        result, prompt = createDiagnosis(hpo_dict, pubCaseFinder, zeroShotResult, gestaltMatcherResult, webresources, absent_hpo_dict=absent_hpo_dict, onset=state.get("onset", "Unknown"),
    sex=state.get("sex", "Unknown"))
        return {"result": {"tentativeDiagnosis": result}, "prompt": prompt}
    return {"result": {"tentativeDiagnosis": None}}





def diseaseNormalizeNode(state: State):
    print("diseaseNormalizeNode called")
    tentativeDiagnosis = state.get("tentativeDiagnosis", None)
    patient_id = state.get("patient_id", "unknown")
    if tentativeDiagnosis is not None:
        normalizedDiagnosis = diseaseNormalizeForDiagnosis(tentativeDiagnosis)
        save_node_result("diseaseNormalizeNode", normalizedDiagnosis, patient_id)
        return {"tentativeDiagnosis": normalizedDiagnosis}
    save_node_result("diseaseNormalizeNode", None, patient_id)
    return {"tentativeDiagnosis": None}


def dieaseSearchNode(state: State):
    print("diseaseSearchNode called")
    
    return diseaseSearchForDiagnosis(state)



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
        save_node_result("reflectionNode", [r.dict() if hasattr(r, "dict") else r for r in reflection_result_list], patient_id)
        return {"result": {"reflection": ReflectionOutput(ans=reflection_result_list)}, "prompt": "\n---\n".join(prompts)}
    save_node_result("reflectionNode", None, patient_id)
    return {"result": {"reflection": None}}

def finalDiagnosisNode(state: State):
    print("finalDiagnosisNode called")
    finalDiagnosis, prompt = createFinalDiagnosis(state)
    return {"result": {"finalDiagnosis": finalDiagnosis}, "prompt": prompt}



def diseaseNormalizeForFinalNode(state: State):
    print("diseaseNormalizeForFinalNode called")
    finalDiagnosis = state.get("finalDiagnosis", None)
    patient_id = state.get("patient_id", "unknown")
    if finalDiagnosis is not None:
        normalizedDiagnosis = diseaseNormalizeForDiagnosis(finalDiagnosis)
        save_node_result("diseaseNormalizeForFinalNode", normalizedDiagnosis, patient_id)
        return {"finalDiagnosis": normalizedDiagnosis}
    save_node_result("diseaseNormalizeForFinalNode", None, patient_id)
    return {"finalDiagnosis": None}