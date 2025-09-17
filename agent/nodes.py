from typing_extensions import List, Optional
from state.state_types import State, PCFres, DiagnosisOutput,ReflectionOutput
from tools.pcf_api import callingPCF
from tools.diagnosis import createDiagnosis
from tools.ZeroShot import createZeroshot
from tools.make_HPOdic import make_hpo_dic
from tools.reflection import create_reflection
from tools.diseaseSearch import diseaseSearchForDiagnosis
from tools.diseaseNormalize import diseaseNormalizeForDiagnosis
from tools.finalDiagnosis import createFinalDiagnosis

def BeginningOfFlowNode(state: State):
    print("BeginningOfFlowNode called")
    state["depth"] += 1
    print(f"Current depth: {state['depth']}")
    # reset Diagnosis and Reflection when starting a new flow
    
    return {"depth": state["depth"], "tentativeDiagnosis": None, "reflection": None}

def PCFnode(state: State):
    print("PCFnode called")
    
    hpo_list = state["hpoList"]
    if not hpo_list:
        return {"pubCaseFinder": []}
    return {"pubCaseFinder": callingPCF(hpo_list)}

def createHPODictNode(state: State):
    print("createHPODictNode called")
    hpo_list = state.get("hpoList", [])
    hpo_dict = make_hpo_dic(hpo_list, None)
    return {"hpoDict": hpo_dict}


def createZeroShotNode(state: State):
    print("createZeroShotNode called")
    hpo_dict = state.get("hpoDict", {})
    if state.get("zeroShotResult") is not None:
        return {"zeroShotResult": state["zeroShotResult"]}
    if hpo_dict:
        zeroShotResult = createZeroshot(hpo_dict)
        if zeroShotResult:
            return {"zeroShotResult": zeroShotResult}
    return {"zeroShotResult": None}



def createDiagnosisNode(state: State):
    # To integrate streamss from both ZeroShot and PCF before diagnosis
    print("DiagnosisNode called")
    hpo_dict = state.get("hpoDict", {})
    pubCaseFinder = state.get("pubCaseFinder", [])
    zeroShotResult = state.get("zeroShotResult", None)

    if hpo_dict and pubCaseFinder:
        tentativeDiagnosis = createDiagnosis(hpo_dict, pubCaseFinder, zeroShotResult)
        return {"tentativeDiagnosis": tentativeDiagnosis}
    return {"tentativeDiagnosis": None}

def diseaseNormalizeNode(state: State):
    print("diseaseNormalizeNode called")
    tentativeDiagnosis = state.get("tentativeDiagnosis", None)
    # Placeholder for disease normalization logic
    return 

def dieaseSearchNode(state: State):
    print("diseaseSearchNode called")
    # Placeholder for disease search logic
    return

def reflectionNode(state: State):
    print("reflectionNode called")
    tentativeDiagnosis = state.get("tentativeDiagnosis", None)
    hpo_dict = state.get("hpoDict", {})
    if tentativeDiagnosis and hpo_dict:
        diagnosis_to_judge_lis = tentativeDiagnosis.ans
        reflection_result_list = []
        for diagnosis_to_judge in diagnosis_to_judge_lis:
            reflection_result = create_reflection(hpo_dict, diagnosis_to_judge)
            reflection_result_list.append(reflection_result)
        print(type(reflection_result_list[0]))
        return {"reflection": ReflectionOutput(ans=reflection_result_list)}
    return {"reflection": None}

def finalDiagnosisNode(state: State):
    print("finalDiagnosisNode called")
    finalDiagnosis = createFinalDiagnosis(state)
    return {"finalDiagnosis": finalDiagnosis}