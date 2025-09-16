from langgraph.graph import StateGraph, START, END
from state.state_types import State
from nodes import PCFnode, createDiagnosisNode, createZeroShotNode, createHPODictNode

graph_builder = StateGraph(State)
graph_builder.add_node("createZeroShotNode", createZeroShotNode)
graph_builder.add_node("PCFnode", PCFnode)
graph_builder.add_node("createDiagnosisNode", createDiagnosisNode)
graph_builder.add_node("createHPODictNode", createHPODictNode)
graph_builder.add_edge(START, "PCFnode")
graph_builder.add_edge(START, "createHPODictNode")
graph_builder.add_edge("createHPODictNode", "createZeroShotNode")
graph_builder.add_edge("createZeroShotNode", "createDiagnosisNode")
graph_builder.add_edge("PCFnode", "createDiagnosisNode")
graph_builder.add_edge("createDiagnosisNode", END)

if __name__ == "__main__":
    input_hpo_list = ["HP:0001250", "HP:0004322"]
    initial_state = {
        "clinicalTest": None,
        "hpoList": input_hpo_list,
        "pubCaseFinder": [],
        "hpoDict": {},
        "zeroShotResult": None,
        "history": [],
        "finalDiagnosis": None
    }
    graph = graph_builder.compile()
    try:
        dot = graph.get_graph().draw_ascii()
        print("=== LangGraph フロー図 (Mermaid記法) ===")
        print(dot)
    except Exception as e:
        print("グラフ可視化に失敗しました:", e)
    result = graph.invoke(initial_state)
    print("=== 診断結果 ===")
    print(result["finalDiagnosis"])