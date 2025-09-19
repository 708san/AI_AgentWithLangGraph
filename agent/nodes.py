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
    if not hpo_list:
        return {"pubCaseFinder": []}
    return {"pubCaseFinder": callingPCF(hpo_list, depth)}

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
                "distance": res.get("distance"),
                "image_id": res.get("image_id", ""),
                "gestalt_score": res.get("gestalt_score")
            })
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
    hpo_dict = state.get("hpoDict", {})
    absent_hpo_dict = state.get("absentHpoDict", {})
    if state.get("zeroShotResult") is not None:
        return {"zeroShotResult": state["zeroShotResult"]}
    if hpo_dict:
        # createZeroshotが(result, prompt)を返すように修正
        result, prompt = createZeroshot(hpo_dict, absent_hpo_dict=absent_hpo_dict)
        if result:
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
        result, prompt = createDiagnosis(hpo_dict, pubCaseFinder, zeroShotResult, gestaltMatcherResult, webresources, absent_hpo_dict=absent_hpo_dict)
        return {"result": {"tentativeDiagnosis": result}, "prompt": prompt}
    return {"result": {"tentativeDiagnosis": None}}



def diseaseNormalizeNode(state: State):
    print("diseaseNormalizeNode called")
    tentativeDiagnosis = state.get("tentativeDiagnosis", None)
    if tentativeDiagnosis is not None:
        normalizedDiagnosis = diseaseNormalizeForDiagnosis(tentativeDiagnosis)
        return {"tentativeDiagnosis": normalizedDiagnosis}
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

    if tentativeDiagnosis and hpo_dict:
        diagnosis_to_judge_lis = tentativeDiagnosis.ans
        reflection_result_list = []
        prompts = []
        for diagnosis_to_judge in diagnosis_to_judge_lis:
            reflection_result, prompt = create_reflection(hpo_dict, diagnosis_to_judge, disease_knowledge, absent_hpo_dict=absent_hpo_dict)
            reflection_result_list.append(reflection_result)
            prompts.append(prompt)
        print(type(reflection_result_list[0]))
        return {"result": {"reflection": ReflectionOutput(ans=reflection_result_list)}, "prompt": "\n---\n".join(prompts)}
    return {"result": {"reflection": None}}

def finalDiagnosisNode(state: State):
    print("finalDiagnosisNode called")
    finalDiagnosis, prompt = createFinalDiagnosis(state)
    return {"result": {"finalDiagnosis": finalDiagnosis}, "prompt": prompt}

def diseaseNormalizeForFinalNode(state: State):
    print("diseaseNormalizeForFinalNode called")
    finalDiagnosis = state.get("finalDiagnosis", None)
    if finalDiagnosis is not None:
        normalizedDiagnosis = diseaseNormalizeForDiagnosis(finalDiagnosis)
        return {"finalDiagnosis": normalizedDiagnosis}
    return {"finalDiagnosis": None}