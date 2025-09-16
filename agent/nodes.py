from typing_extensions import List, Optional
from state.state_types import State, PCFres, DiagnosisOutput
from tools.pcf_api import callingPCF
from tools.diagnosis import createDiagnosis
from tools.ZeroShot import createZeroshot
from tools.make_HPOdic import make_hpo_dic


def PCFnode(state: State):
    hpo_list = state["hpoList"]
    if not hpo_list:
        return {"pubCaseFinder": []}
    return {"pubCaseFinder": callingPCF(hpo_list)}

def createDiagnosisNode(state: State):
    hpo_dict = state.get("hpoDict", {})
    pubCaseFinder = state.get("pubCaseFinder", [])
    zeroShotResult = state.get("zeroShotResult", None)

    if hpo_dict and pubCaseFinder:
        finalDiagnosis = createDiagnosis(hpo_dict, pubCaseFinder, zeroShotResult)
        return {"finalDiagnosis": finalDiagnosis}
    return {"finalDiagnosis": None}

def createZeroShotNode(state: State):
    hpo_dict = state.get("hpoDict", {})
    if hpo_dict:
        zeroShotResult = createZeroshot(hpo_dict)
        if zeroShotResult:
            return {"zeroShotResult": zeroShotResult}
    return {"zeroShotResult": None}

def createHPODictNode(state: State):
    hpo_list = state.get("hpoList", [])
    hpo_dict = make_hpo_dic(hpo_list, None)
    return {"hpoDict": hpo_dict}