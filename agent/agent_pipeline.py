import os
import datetime
import json
from langgraph.graph import StateGraph, START, END
from agent.state.state_types import State, ZeroShotOutput, DiagnosisOutput, ReflectionOutput

from agent.nodes import (
    PCFnode, createDiagnosisNode, createZeroShotNode, createHPODictNode,
    diseaseNormalizeNode, dieaseSearchNode, reflectionNode,
    BeginningOfFlowNode, finalDiagnosisNode, GestaltMatcherNode
)

class RareDiseaseDiagnosisPipeline:
    def __init__(self, enable_log=False):
        self.graph = self._build_graph()
        self.enable_log = enable_log
        self.logfile_path = None
        if self.enable_log:
            self.logfile_path = self._get_logfile_path()

    def _get_logfile_path(self):
        log_dir = os.path.join(os.getcwd(), "log")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(log_dir, f"agent_log_{timestamp}.log")

    def _log(self, node_name, result):
        if not self.enable_log or not self.logfile_path:
            return
        with open(self.logfile_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== {node_name} ===\n")
            try:
                # ZeroShotOutput, DiagnosisOutput, ReflectionOutputはpydanticモデル
                if isinstance(result, ZeroShotOutput) or isinstance(result, DiagnosisOutput) or isinstance(result, ReflectionOutput):
                    f.write(result.model_dump_json(indent=2, ensure_ascii=False))
                elif hasattr(result, "dict"):
                    f.write(json.dumps(result.dict(), ensure_ascii=False, indent=2))
                elif isinstance(result, dict):
                    # dictの中にpydanticモデルが入っている場合も考慮
                    def default(o):
                        if hasattr(o, "model_dump"):
                            return o.model_dump()
                        if hasattr(o, "dict"):
                            return o.dict()
                        return str(o)
                    f.write(json.dumps(result, ensure_ascii=False, indent=2, default=default))
                elif isinstance(result, list):
                    for item in result:
                        if hasattr(item, "model_dump_json"):
                            f.write(item.model_dump_json(indent=2, ensure_ascii=False) + "\n")
                        elif hasattr(item, "dict"):
                            f.write(json.dumps(item.dict(), ensure_ascii=False, indent=2) + "\n")
                        else:
                            f.write(str(item) + "\n")
                else:
                    f.write(str(result))
            except Exception as e:
                f.write(f"ログ整形エラー: {e}\n")
            f.write("\n")


    def _build_graph(self):
        graph_builder = StateGraph(State)
        # ラップして各ノードの結果をログに記録
        def wrap_node(node_func, node_name):
            def wrapped(state):
                result = node_func(state)
                self._log(node_name, result)
                return result
            return wrapped

        graph_builder.add_node("BeginningOfFlowNode", wrap_node(BeginningOfFlowNode, "BeginningOfFlowNode"))
        graph_builder.add_node("createZeroShotNode", wrap_node(createZeroShotNode, "createZeroShotNode"))
        graph_builder.add_node("PCFnode", wrap_node(PCFnode, "PCFnode"))
        graph_builder.add_node("GestaltMatcherNode", wrap_node(GestaltMatcherNode, "GestaltMatcherNode"))
        graph_builder.add_node("createHPODictNode", wrap_node(createHPODictNode, "createHPODictNode"))
        graph_builder.add_node("createDiagnosisNode", wrap_node(createDiagnosisNode, "createDiagnosisNode"))
        graph_builder.add_node("diseaseNormalizeNode", wrap_node(diseaseNormalizeNode, "diseaseNormalizeNode"))
        graph_builder.add_node("diseaseSearchNode", wrap_node(dieaseSearchNode, "diseaseSearchNode"))
        graph_builder.add_node("reflectionNode", wrap_node(reflectionNode, "reflectionNode"))
        graph_builder.add_node("finalDiagnosisNode", wrap_node(finalDiagnosisNode, "finalDiagnosisNode"))

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
