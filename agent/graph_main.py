from langgraph.graph import StateGraph, START, END
from state.state_types import State
from nodes import PCFnode, createDiagnosisNode, createZeroShotNode, createHPODictNode,diseaseNormalizeNode,dieaseSearchNode,reflectionNode,BeginningOfFlowNode,finalDiagnosisNode,GestaltMatcherNode



graph_builder = StateGraph(State)
graph_builder.add_node("BeginningOfFlowNode", BeginningOfFlowNode)
graph_builder.add_node("createZeroShotNode", createZeroShotNode)
graph_builder.add_node("PCFnode", PCFnode)
graph_builder.add_node("GestaltMatcherNode", GestaltMatcherNode)
graph_builder.add_node("createHPODictNode", createHPODictNode)
graph_builder.add_node("createDiagnosisNode", createDiagnosisNode)
graph_builder.add_node("diseaseNormalizeNode", diseaseNormalizeNode)
graph_builder.add_node("diseaseSearchNode", dieaseSearchNode)  # Placeholder for disease search node
graph_builder.add_node("reflectionNode", reflectionNode)
graph_builder.add_node("finalDiagnosisNode", finalDiagnosisNode)

def controll_join_node(state: State):
    if state.get("diagnosis_flag") < 2:
        return "GoToEnd"
    return "DoDiagnosis"

def after_reflection_edge(state: State):
    if state.get("depth", 0) > 2:
        print("depth limit reached, force to finalDiagnosisNode")
        return "ProceedToFinalDiagnosisNode"
    reflection = state.get("reflection")
    # if reflection is None or not reflection.ans:
    if not reflection or not hasattr(reflection, "ans") or not reflection.ans:
        return "ReturnToBeginningNode"
    # if all not getattr(ans, "Correction", False) for ans in reflection.ans:
    if all(not getattr(ans, "Correctness", False) for ans in reflection.ans):
        print("think again.")
        return "ReturnToBeginningNode"
    # Go to finalDiagnosisNode instead of END
    return "ProceedToFinalDiagnosisNode"

graph_builder.add_edge(START, "BeginningOfFlowNode")
graph_builder.add_edge( "BeginningOfFlowNode", "PCFnode")
graph_builder.add_edge( "BeginningOfFlowNode", "createHPODictNode")
graph_builder.add_edge( "BeginningOfFlowNode", "GestaltMatcherNode")
graph_builder.add_edge("createHPODictNode", "createZeroShotNode")
graph_builder.add_edge(["createZeroShotNode", "PCFnode", "GestaltMatcherNode"], "createDiagnosisNode")
graph_builder.add_edge("createDiagnosisNode", "diseaseNormalizeNode")
graph_builder.add_edge("diseaseNormalizeNode", "diseaseSearchNode")
graph_builder.add_edge("diseaseSearchNode", "reflectionNode")
graph_builder.add_conditional_edges("reflectionNode", after_reflection_edge,path_map={
    "ReturnToBeginningNode": "BeginningOfFlowNode",
    "ProceedToFinalDiagnosisNode": "finalDiagnosisNode"})
graph_builder.add_edge("finalDiagnosisNode", END)


if __name__ == "__main__":
    input_hpo_list = [
    "HP:0000054", "HP:0000286", "HP:0000297", "HP:0000965", "HP:0001263",
    "HP:0001513", "HP:0002265", "HP:0002342", "HP:0030820"
]
    image_path = "/Users/yoshikuwa-n/Downloads/WorkForBioHackathon/AI_AgentWithLangGraph/sampleData/PhenoPacketStore_25072025/20001.jpg"
    initial_state = {
        #defalut depth is 0 (and in beggining node, depth will be increased to 1)
        "depth": 0,
        "clinicalText": None,
        "hpoList": input_hpo_list,
        "imagePath": image_path,
        "pubCaseFinder": [],
        "GestaltMatcher": None,
        "hpoDict": {},
        "zeroShotResult": None,
        "memory": [],
        "tentativeDiagnosis": None,
        "reflection": None
    }
    graph = graph_builder.compile()
    try:
        dot = graph.get_graph().draw_ascii()
        print("=== LangGraph フロー図 (Mermaid記法) ===")
        print(dot)
    except Exception as e:
        print("グラフ可視化に失敗しました:", e)
    
    result = graph.invoke(initial_state)

    

    print("=== result of reflection ===")
    reflection = result.get("reflection", None)
    if reflection is None:
        print("No reflection result.")
    elif hasattr(reflection, "ans"):
        for i, ans in enumerate(reflection.ans, 1):
            print(f"--- Reflection {i} ---")
            print(f"Diagnosis: {getattr(ans, 'disease_name', '')}")
            print(f"Correctness: {getattr(ans, 'Correctness', '')}")
            print(f"Patient Summary:\n{getattr(ans, 'PatientSummary', '')}")
            print(f"Diagnosis Analysis:\n{getattr(ans, 'DiagnosisAnalysis', '')}")
            print(f"Reference:\n{getattr(ans, 'reference', '')}")
            print("-" * 40)
    else:
        print(reflection)
    print("\n")

    print("=== result of finalDiagnosis ===")
    final_diag = result.get("finalDiagnosis", None)
    if final_diag is None:
        print("No final diagnosis.")
    elif hasattr(final_diag, "ans"):
        for i, diag in enumerate(final_diag.ans, 1):
            print(f"Rank {i}: {diag.disease_name}")
            print(f"  Description: {diag.description}")
            print(f"  Reference: {getattr(final_diag, 'reference', '')}")
            print("-" * 40)
    else:
        print(final_diag)
    print("\n")
    # check each result..
    """
    print("=== HPO ===")
    print(result["hpoDict"])
    print("\n")
    print(result["hpoList"])
    print("\n")
    print("=== results of gestaltMatcher ===")
    print(result["GestaltMatcher"])
    print("\n")
    print("=== result of PCF ===")
    print(result["pubCaseFinder"])
    print("\n")
    print("=== result of ZeroShot ===")
    print(result["zeroShotResult"])
    print("\n")
    print("=== result of diseaseSearch ===")
    print(result["memory"])
    
    print("=== result of tentativeDiagnosis ===")
    print(result["tentativeDiagnosis"])
    print("\n")
    print("=== result of reflection ===")
    print(result["reflection"])
    print("\n")
    print("=== result of finalDiagnosis ===")
    print(result.get("finalDiagnosis", None))
    print("\n")
    """