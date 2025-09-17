from langgraph.graph import StateGraph, START, END
from state.state_types import State
from nodes import PCFnode, createDiagnosisNode, createZeroShotNode, createHPODictNode,diseaseNormalizeNode,dieaseSearchNode,reflectionNode,BeginningOfFlowNode,finalDiagnosisNode



graph_builder = StateGraph(State)
graph_builder.add_node("BeginningOfFlowNode", BeginningOfFlowNode)
graph_builder.add_node("createZeroShotNode", createZeroShotNode)
graph_builder.add_node("PCFnode", PCFnode)
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
graph_builder.add_edge("createHPODictNode", "createZeroShotNode")
graph_builder.add_edge(["createZeroShotNode", "PCFnode"], "createDiagnosisNode")
graph_builder.add_edge("createDiagnosisNode", "diseaseNormalizeNode")
graph_builder.add_edge("diseaseNormalizeNode", "diseaseSearchNode")
graph_builder.add_edge("diseaseSearchNode", "reflectionNode")
graph_builder.add_conditional_edges("reflectionNode", after_reflection_edge,path_map={
    "ReturnToBeginningNode": "BeginningOfFlowNode",
    "ProceedToFinalDiagnosisNode": "finalDiagnosisNode"})
graph_builder.add_edge("finalDiagnosisNode", END)


if __name__ == "__main__":
    input_hpo_list = ["HP:0001250", "HP:0004322"]
    initial_state = {
        #defalut depth is 0 (and in beggining node, depth will be increased to 1)
        "depth": 0,
        "clinicalText": None,
        "hpoList": input_hpo_list,
        "pubCaseFinder": [],
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

    print("=== HPO ===")
    print(result["hpoDict"])
    print("\n")
    print(result["hpoList"])
    print("\n")
    print("=== result of PCF ===")
    print(result["pubCaseFinder"])
    print("\n")
    print("=== result of ZeroShot ===")
    print(result["zeroShotResult"])
    print("\n")
    print("=== result of tentativeDiagnosis ===")
    print(result["tentativeDiagnosis"])
    print("\n")
    print("=== result of reflection ===")
    print(result["reflection"])
    print("\n")
    print("=== result of finalDiagnosis ===")
    print(result.get("finalDiagnosis", None))
    print("\n")