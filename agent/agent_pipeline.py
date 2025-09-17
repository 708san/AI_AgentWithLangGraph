from langgraph.graph import StateGraph, START, END
from agent.state.state_types import State
from agent.nodes import (
    PCFnode, createDiagnosisNode, createZeroShotNode, createHPODictNode,
    diseaseNormalizeNode, dieaseSearchNode, reflectionNode,
    BeginningOfFlowNode, finalDiagnosisNode, GestaltMatcherNode
)


class RareDiseaseDiagnosisPipeline:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("BeginningOfFlowNode", BeginningOfFlowNode)
        graph_builder.add_node("createZeroShotNode", createZeroShotNode)
        graph_builder.add_node("PCFnode", PCFnode)
        graph_builder.add_node("GestaltMatcherNode", GestaltMatcherNode)
        graph_builder.add_node("createHPODictNode", createHPODictNode)
        graph_builder.add_node("createDiagnosisNode", createDiagnosisNode)
        graph_builder.add_node("diseaseNormalizeNode", diseaseNormalizeNode)
        graph_builder.add_node("diseaseSearchNode", dieaseSearchNode)
        graph_builder.add_node("reflectionNode", reflectionNode)
        graph_builder.add_node("finalDiagnosisNode", finalDiagnosisNode)

        def after_reflection_edge(state: State):
            if state.get("depth", 0) > 2:
                print("depth limit reached, force to finalDiagnosisNode")
                return "ProceedToFinalDiagnosisNode"
            reflection = state.get("reflection")
            if not reflection or not hasattr(reflection, "ans") or not reflection.ans:
                return "ReturnToBeginningNode"
            if all(not getattr(ans, "Correctness", False) for ans in reflection.ans):
                print("think again.")
                return "ReturnToBeginningNode"
            return "ProceedToFinalDiagnosisNode"

        graph_builder.add_edge(START, "BeginningOfFlowNode")
        graph_builder.add_edge("BeginningOfFlowNode", "PCFnode")
        graph_builder.add_edge("BeginningOfFlowNode", "createHPODictNode")
        graph_builder.add_edge("BeginningOfFlowNode", "GestaltMatcherNode")
        graph_builder.add_edge("createHPODictNode", "createZeroShotNode")
        graph_builder.add_edge(["createZeroShotNode", "PCFnode", "GestaltMatcherNode"], "createDiagnosisNode")
        graph_builder.add_edge("createDiagnosisNode", "diseaseNormalizeNode")
        graph_builder.add_edge("diseaseNormalizeNode", "diseaseSearchNode")
        graph_builder.add_edge("diseaseSearchNode", "reflectionNode")
        graph_builder.add_conditional_edges(
            "reflectionNode", after_reflection_edge, path_map={
                "ReturnToBeginningNode": "BeginningOfFlowNode",
                "ProceedToFinalDiagnosisNode": "finalDiagnosisNode"
            }
        )
        graph_builder.add_edge("finalDiagnosisNode", END)
        return graph_builder.compile()

    def run(self, hpo_list, image_path=None, verbose=True):
        initial_state = {
            "depth": 0,
            "clinicalText": None,
            "hpoList": hpo_list,
            "imagePath": image_path,
            "pubCaseFinder": [],
            "GestaltMatcher": None,
            "hpoDict": {},
            "zeroShotResult": None,
            "memory": [],
            "tentativeDiagnosis": None,
            "reflection": None
        }
        result = self.graph.invoke(initial_state)
        if verbose:
            self.pretty_print(result)
        return result

    def pretty_print(self, result):
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
                # referencesはList[str]なので整形して出力
                references = getattr(ans, 'references', [])
                if references:
                    print("References:")
                    for ref in references:
                        print(f"  - {ref}")
                else:
                    print("References: None")
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

