import os
import datetime
import json
from langgraph.graph import StateGraph, START, END
from agent.state.state_types import State, ZeroShotOutput, DiagnosisOutput, ReflectionOutput
from agent.utils.logger import log_node_result
from agent.llm.azure_llm_instance import get_llm_instance

from agent.nodes import (
    PCFnode, createDiagnosisNode, createZeroShotNode, createHPODictNode,createAbsentHPODictNode, 
    diseaseNormalizeNode, diseaseSearchNode, reflectionNode,
    BeginningOfFlowNode, finalDiagnosisNode, GestaltMatcherNode,
    diseaseNormalizeForFinalNode, HPOwebSearchNode,
    NormalizePCFNode, NormalizeGestaltMatcherNode, NormalizeZeroShotNode, DiseaseSearchWithHPONode
)

class RareDiseaseDiagnosisPipeline:
    def __init__(self, model_name: str = 'gpt-4o', enable_log=False, log_filename=None):
        self.graph = self._build_graph()
        self.enable_log = enable_log
        self.logfile_path = None
        self.log_filename = log_filename
        if self.enable_log:
            self.logfile_path = self._get_logfile_path()
            self._write_graph_ascii_to_log()
        
        self.llm = get_llm_instance(model_name)
            
    def _get_logfile_path(self):
        log_dir = os.path.join(os.getcwd(), "log")
        os.makedirs(log_dir, exist_ok=True)
        if self.log_filename:
            return os.path.join(log_dir, self.log_filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(log_dir, f"agent_log_{timestamp}.log")
    
    def _write_graph_ascii_to_log(self):
        # エージェントフロー図をASCIIでlogファイルの先頭に出力
        try:
            ascii_graph = self.graph.get_graph().draw_ascii()
        except Exception as e:
            ascii_graph = f"[Failed to draw graph: {e}]"
        with open(self.logfile_path, "w", encoding="utf-8") as f:
            f.write("=== Agent Flow Graph ===\n")
            f.write(ascii_graph)
            f.write("\n\n")

    def _log(self, node_name, result):
        if not self.enable_log:
            return
        log_node_result(self.logfile_path, node_name, result)

    def _build_graph(self):
        graph_builder = StateGraph(State)
        # ラップして各ノードの結果をログに記録
        def wrap_node(node_func, node_name):
            def wrapped(state):
                result = node_func(state)
                self._log(node_name, result)
                # プロンプト付きdictの場合はresult["result"]を返す
                if isinstance(result, dict) and "result" in result:
                    return result["result"]
                return result
            return wrapped

        graph_builder.add_node("BeginningOfFlowNode", wrap_node(BeginningOfFlowNode, "BeginningOfFlowNode"))
        graph_builder.add_node("createZeroShotNode", wrap_node(createZeroShotNode, "createZeroShotNode"))
        graph_builder.add_node("PCFnode", wrap_node(PCFnode, "PCFnode"))
        graph_builder.add_node("GestaltMatcherNode", wrap_node(GestaltMatcherNode, "GestaltMatcherNode"))
        graph_builder.add_node("NormalizeZeroShotNode", wrap_node(NormalizeZeroShotNode, "NormalizeZeroShotNode"))
        graph_builder.add_node("NormalizePCFNode", wrap_node(NormalizePCFNode, "NormalizePCFNode"))
        graph_builder.add_node("NormalizeGestaltMatcherNode", wrap_node(NormalizeGestaltMatcherNode, "NormalizeGestaltMatcherNode"))
        graph_builder.add_node("createHPODictNode", wrap_node(createHPODictNode, "createHPODictNode"))
        graph_builder.add_node("createAbsentHPODictNode", wrap_node(createAbsentHPODictNode, "createAbsentHPODictNode"))
        graph_builder.add_node("HPOwebSearchNode", wrap_node(HPOwebSearchNode, "HPOwebSearchNode"))
        graph_builder.add_node("DiseaseSearchWithHPONode", wrap_node(DiseaseSearchWithHPONode, "DiseaseSearchWithHPONode"))
        graph_builder.add_node("createDiagnosisNode", wrap_node(createDiagnosisNode, "createDiagnosisNode"))
        graph_builder.add_node("diseaseNormalizeNode", wrap_node(diseaseNormalizeNode, "diseaseNormalizeNode"))
        graph_builder.add_node("diseaseSearchNode", wrap_node(diseaseSearchNode, "diseaseSearchNode"))
        graph_builder.add_node("reflectionNode", wrap_node(reflectionNode, "reflectionNode"))
        graph_builder.add_node("finalDiagnosisNode", wrap_node(finalDiagnosisNode, "finalDiagnosisNode"))
        graph_builder.add_node("diseaseNormalizeForFinalNode", wrap_node(diseaseNormalizeForFinalNode, "diseaseNormalizeForFinalNode"))
        
        def after_reflection_edge(state: State):
            print("\n--- Running after_reflection_edge ---")

            # 1. depthのチェック
            depth = state.get("depth", 0)
            print(f"Current depth: {depth}")
            if depth > 0:
                print("Depth limit reached, forcing to finalDiagnosisNode.")
                return "ProceedToFinalDiagnosisNode"

            # 2. reflectionオブジェクトの存在と内容を確認
            reflection = state.get("reflection")
            print(f"Type of reflection object: {type(reflection)}")
            if not reflection or not hasattr(reflection, "ans") or not reflection.ans:
                print("Reflection object is missing, empty, or has no 'ans'. Returning to beginning.")
                return "ReturnToBeginningNode"
            
            # 3. reflection.ans の中身と、各要素のCorrectnessの型と値を調べる
            correctness_values_for_any = []
            print("Inspecting items in reflection.ans:")
            for i, ans_item in enumerate(reflection.ans):
                disease_name = getattr(ans_item, "disease_name", "Unknown Disease")
                correctness_val = getattr(ans_item, "Correctness", "N/A")
                
                # any()で評価する実際の値を取得
                bool_val = getattr(ans_item, "Correctness", False)
                correctness_values_for_any.append(bool_val)
                
                

            # 4. any()の評価結果を確認
            should_proceed = any(correctness_values_for_any)
            print(f"\nList of boolean values for 'any()': {correctness_values_for_any}")
            print(f"Result of 'any(correctness_values_for_any)': {should_proceed}")

            if should_proceed:
                print("Decision: Proceeding to final diagnosis.")
                print("--- End of after_reflection_edge ---\n")
                return "ProceedToFinalDiagnosisNode"
            else:
                print("Decision: All 'Correctness' are False or missing. Looping back.")
                print("--- End of after_reflection_edge ---\n")
                return "ReturnToBeginningNode"

        graph_builder.add_edge(START, "BeginningOfFlowNode")
        graph_builder.add_edge("BeginningOfFlowNode", "PCFnode")
        graph_builder.add_edge("PCFnode", "NormalizePCFNode")
        graph_builder.add_edge("BeginningOfFlowNode", "createHPODictNode")
        graph_builder.add_edge("BeginningOfFlowNode", "GestaltMatcherNode")
        graph_builder.add_edge("GestaltMatcherNode", "NormalizeGestaltMatcherNode")
        graph_builder.add_edge("BeginningOfFlowNode", "createAbsentHPODictNode")
        graph_builder.add_edge(["createHPODictNode","createAbsentHPODictNode"], "createZeroShotNode")
        graph_builder.add_edge("createZeroShotNode", "NormalizeZeroShotNode")
        graph_builder.add_edge("createHPODictNode", "HPOwebSearchNode")
        graph_builder.add_edge("createHPODictNode", "DiseaseSearchWithHPONode")
        graph_builder.add_edge(["NormalizeZeroShotNode", "NormalizePCFNode", "NormalizeGestaltMatcherNode", "HPOwebSearchNode", "DiseaseSearchWithHPONode"], "createDiagnosisNode")
        graph_builder.add_edge("createDiagnosisNode", "diseaseNormalizeNode")
        ###中断用
        #graph_builder.add_edge("diseaseNormalizeNode", END)
        ###
        
        graph_builder.add_edge("diseaseNormalizeNode", "diseaseSearchNode")
        graph_builder.add_edge("diseaseSearchNode", "reflectionNode")
        graph_builder.add_conditional_edges(
            "reflectionNode", after_reflection_edge, path_map={
                "ReturnToBeginningNode": "BeginningOfFlowNode",
                "ProceedToFinalDiagnosisNode": "finalDiagnosisNode"
            }
        )
        graph_builder.add_edge("finalDiagnosisNode", "diseaseNormalizeForFinalNode")
        graph_builder.add_edge("diseaseNormalizeForFinalNode", END)
        
        return graph_builder.compile()

    def run(self, hpo_list, image_path=None, verbose=False, absent_hpo_list=None, onset=None, sex=None, patient_id=None):
        initial_state = {
            "depth": 0,
            "clinicalText": None,
            "hpoList": hpo_list,
            "absentHpoList": absent_hpo_list or [],
            "imagePath": image_path,
            "pubCaseFinder": [],
            "GestaltMatcher": None,
            "hpoDict": {},
            "zeroShotResult": None,
            "memory": [],
            "tentativeDiagnosis": None,
            "reflection": None,
            "onset": onset if onset else "Unknown",
            "sex": sex if sex else "Unknown",
            "patient_id": patient_id if patient_id else "unknown",
            "llm": self.llm,
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


"""
        def after_reflection_edge(state: State):
            if state.get("depth", 0) > 2:
                print("depth limit reached, force to finalDiagnosisNode")
                return "ProceedToFinalDiagnosisNode"
            reflection = state.get("reflection")
            if not reflection or not hasattr(reflection, "ans") or not reflection.ans:
                return "ReturnToBeginningNode"
            if any(getattr(ans, "Correctness", False) for ans in reflection.ans):
                return "ProceedToFinalDiagnosisNode"
            print("think again.")
            return "ReturnToBeginningNode"
"""