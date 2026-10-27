import os
import datetime
from langgraph.graph import StateGraph, START, END
from agent.state.state_types import State
from agent.utils.logger import log_node_result
from agent.utils.result_saver import save_node_result
from agent.utils.hpo_importance_filter import filter_hpo_by_importance
from agent.llm.azure_llm_instance import get_llm_instance
from agent.utils.response_serializer import serialize_public_state

from agent.nodes import (
    PCFnode, createDiagnosisNode, createZeroShotNode, createHPODictNode,createAbsentHPODictNode, 
    diseaseNormalizeNode, diseaseSearchNode, reflectionNode,
    BeginningOfFlowNode, finalDiagnosisNode, GestaltMatcherNode,
    diseaseNormalizeForFinalNode, HPOwebSearchNode,
    NormalizePCFNode, NormalizeGestaltMatcherNode, NormalizeZeroShotNode, DiseaseSearchWithHPONode,
    PhenoBrainNode, mergeCandidateResultsNode,
    candidateMetadataNode, literatureSearchNode, evidenceExtractionNode,
    diagnosisOutputValidationNode,
)
from agent.state.state_types import ReflectionOutputV2


NODE_DEFINITIONS = [
    ("BeginningOfFlowNode", BeginningOfFlowNode),
    ("createZeroShotNode", createZeroShotNode),
    ("PCFnode", PCFnode),
    ("PhenoBrainNode", PhenoBrainNode),
    ("GestaltMatcherNode", GestaltMatcherNode),
    ("NormalizeZeroShotNode", NormalizeZeroShotNode),
    ("NormalizePCFNode", NormalizePCFNode),
    ("NormalizeGestaltMatcherNode", NormalizeGestaltMatcherNode),
    ("createHPODictNode", createHPODictNode),
    ("createAbsentHPODictNode", createAbsentHPODictNode),
    ("HPOwebSearchNode", HPOwebSearchNode),
    ("DiseaseSearchWithHPONode", DiseaseSearchWithHPONode),
    ("mergeCandidateResultsNode", mergeCandidateResultsNode),
    ("createDiagnosisNode", createDiagnosisNode),
    ("diseaseNormalizeNode", diseaseNormalizeNode),
    ("diseaseSearchNode", diseaseSearchNode),
    ("reflectionNode", reflectionNode),
    ("finalDiagnosisNode", finalDiagnosisNode),
    ("diseaseNormalizeForFinalNode", diseaseNormalizeForFinalNode),
]

EDGES = [
    (START, "BeginningOfFlowNode"),
    ("BeginningOfFlowNode", "PCFnode"),
    ("PCFnode", "NormalizePCFNode"),
    ("BeginningOfFlowNode", "PhenoBrainNode"),
    ("BeginningOfFlowNode", "createHPODictNode"),
    ("BeginningOfFlowNode", "GestaltMatcherNode"),
    ("GestaltMatcherNode", "NormalizeGestaltMatcherNode"),
    ("BeginningOfFlowNode", "createAbsentHPODictNode"),
    (["createHPODictNode", "createAbsentHPODictNode"], "createZeroShotNode"),
    ("createZeroShotNode", "NormalizeZeroShotNode"),
    ("createHPODictNode", "HPOwebSearchNode"),
    ("createHPODictNode", "DiseaseSearchWithHPONode"),
    (["NormalizeZeroShotNode", "NormalizePCFNode", "NormalizeGestaltMatcherNode", "DiseaseSearchWithHPONode", "PhenoBrainNode"], "mergeCandidateResultsNode"),
    (["mergeCandidateResultsNode", "HPOwebSearchNode"], "createDiagnosisNode"),
    ("createDiagnosisNode", "diseaseNormalizeNode"),
    ("diseaseNormalizeNode", "diseaseSearchNode"),
    ("diseaseSearchNode", "reflectionNode"),
    ("finalDiagnosisNode", "diseaseNormalizeForFinalNode"),
    ("diseaseNormalizeForFinalNode", END),
]


def _has_any_correct_reflection(reflection) -> bool:
    if not reflection or not hasattr(reflection, "ans") or not reflection.ans:
        return False
    return any(getattr(ans_item, "Correctness", False) for ans_item in reflection.ans)

class RareDiseaseDiagnosisPipeline:
    DEFAULT_EVIDENCE_MODE = "v2"

    def __init__(self, model_name: str = 'gpt-4o', enable_log=False, log_filename=None, log_dir=None):
        self.enable_log = enable_log
        self.logfile_path = None
        self.log_filename = log_filename
        self.log_dir = log_dir
        self.graph = self._build_graph("legacy")
        self.v2_graph = self._build_graph("v2")
        if self.enable_log:
            self.logfile_path = self._get_logfile_path()
        
        self.llm = get_llm_instance(model_name)
            
    def _get_logfile_path(self):
        log_dir = self.log_dir or os.path.join(os.getcwd(), "log")
        os.makedirs(log_dir, exist_ok=True)
        if self.log_filename:
            return os.path.join(log_dir, self.log_filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(log_dir, f"agent_log_{timestamp}.log")
    
    def _write_graph_ascii_to_log(self, evidence_mode: str):
        # 実際に実行するモードのフロー図をログ先頭に出力する。
        try:
            graph = self.v2_graph if evidence_mode == "v2" else self.graph
            ascii_graph = graph.get_graph().draw_ascii()
        except Exception as e:
            ascii_graph = f"[Failed to draw graph: {e}]"
        with open(self.logfile_path, "w", encoding="utf-8") as f:
            f.write(f"=== Agent Flow Graph (evidence_mode={evidence_mode}) ===\n")
            f.write(ascii_graph)
            f.write("\n\n")

    def _log(self, node_name, result, state=None):
        if not self.enable_log:
            return
        log_node_result(self.logfile_path, node_name, result)

    def _build_graph(self, evidence_mode: str = DEFAULT_EVIDENCE_MODE):
        graph_builder = StateGraph(State)
        # ラップして各ノードの結果をログに記録
        def wrap_node(node_func, node_name):
            def wrapped(state):
                result = node_func(state)
                self._log(node_name, result, state)
                # Decorated nodes already persist themselves. Persist only
                # nodes without the legacy decorator so graph execution still
                # produces a complete artifact set without duplicate writes.
                if not getattr(node_func, "_saves_result", False):
                    save_node_result(node_name, result, state)
                # プロンプト付きdictの場合はresult["result"]を返す
                if isinstance(result, dict) and "result" in result:
                    return result["result"]
                return result
            return wrapped

        if evidence_mode == "v2":
            node_definitions = [
                item for item in NODE_DEFINITIONS
                if item[0] not in {"diseaseSearchNode"}
            ] + [
                ("candidateMetadataNode", candidateMetadataNode),
                ("literatureSearchNode", literatureSearchNode),
                ("evidenceExtractionNode", evidenceExtractionNode),
                ("diagnosisOutputValidationNode", diagnosisOutputValidationNode),
            ]
        else:
            node_definitions = NODE_DEFINITIONS

        for node_name, node_func in node_definitions:
            graph_builder.add_node(node_name, wrap_node(node_func, node_name))

        if evidence_mode == "v2":
            v2_edges = [
                (START, "BeginningOfFlowNode"),
                ("BeginningOfFlowNode", "PCFnode"),
                ("PCFnode", "NormalizePCFNode"),
                ("BeginningOfFlowNode", "PhenoBrainNode"),
                ("BeginningOfFlowNode", "createHPODictNode"),
                ("BeginningOfFlowNode", "GestaltMatcherNode"),
                ("GestaltMatcherNode", "NormalizeGestaltMatcherNode"),
                ("BeginningOfFlowNode", "createAbsentHPODictNode"),
                (["createHPODictNode", "createAbsentHPODictNode"], "createZeroShotNode"),
                ("createZeroShotNode", "NormalizeZeroShotNode"),
                ("createHPODictNode", "HPOwebSearchNode"),
                ("createHPODictNode", "DiseaseSearchWithHPONode"),
                (["NormalizeZeroShotNode", "NormalizePCFNode", "NormalizeGestaltMatcherNode", "DiseaseSearchWithHPONode", "PhenoBrainNode"], "mergeCandidateResultsNode"),
                (["mergeCandidateResultsNode", "HPOwebSearchNode"], "candidateMetadataNode"),
                ("candidateMetadataNode", "createDiagnosisNode"),
                ("createDiagnosisNode", "diseaseNormalizeNode"),
                ("diseaseNormalizeNode", "literatureSearchNode"),
                ("literatureSearchNode", "evidenceExtractionNode"),
                ("evidenceExtractionNode", "reflectionNode"),
                ("finalDiagnosisNode", "diseaseNormalizeForFinalNode"),
                ("diseaseNormalizeForFinalNode", "diagnosisOutputValidationNode"),
                ("diagnosisOutputValidationNode", END),
            ]
            for src, dst in v2_edges:
                graph_builder.add_edge(src, dst)

            def after_v2_reflection_edge(state: State):
                reflection = state.get("reflection")
                iteration = state.get("flow_iteration", 0)
                maximum = state.get("max_flow_iterations", 1)
                if not isinstance(reflection, ReflectionOutputV2) or not reflection.ans:
                    return "ProceedToFinalDiagnosisNode"
                # Retry only candidates whose retrieval/Evidence remains
                # unresolved. Missing literature for every candidate is not
                # required, and it is never a clinical rejection rule.
                has_unresolved_literature = any(
                    item.literature_status in {"not_retrieved", "insufficient"}
                    for item in reflection.ans
                )
                if has_unresolved_literature and iteration < maximum:
                    return "RetryLiteratureSearchNode"
                return "ProceedToFinalDiagnosisNode"

            graph_builder.add_conditional_edges(
                "reflectionNode",
                after_v2_reflection_edge,
                path_map={
                    "RetryLiteratureSearchNode": "literatureSearchNode",
                    "ProceedToFinalDiagnosisNode": "finalDiagnosisNode",
                },
            )
            return graph_builder.compile()
        
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
            should_proceed = _has_any_correct_reflection(reflection)
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

        for src, dst in EDGES:
            graph_builder.add_edge(src, dst)

        graph_builder.add_conditional_edges(
            "reflectionNode", after_reflection_edge, path_map={
                "ReturnToBeginningNode": "BeginningOfFlowNode",
                "ProceedToFinalDiagnosisNode": "finalDiagnosisNode"
            }
        )
        
        return graph_builder.compile()

    def _build_initial_state(self, hpo_list, image_path=None, absent_hpo_list=None, onset=None, sex=None, patient_id=None, use_absentHPO=False, filter_impotance=False, use_phenobrain=False, evidence_mode=DEFAULT_EVIDENCE_MODE, max_flow_iterations=1, gestalt_matcher_result_loader=None):
        if filter_impotance:
            hpo_list = filter_hpo_by_importance(hpo_list)
            absent_hpo_list = filter_hpo_by_importance(absent_hpo_list or [])

        return {
            "depth": 0,
            "evidence_mode": evidence_mode,
            "clinicalText": None,
            "hpoList": hpo_list,
            "absentHpoList": absent_hpo_list or [],
            "use_absentHPO": use_absentHPO,
            "use_phenobrain": use_phenobrain,
            "filter_impotance": filter_impotance,
            "imagePath": image_path,
            "pubCaseFinder": [],
            "phenoBrain": [],
            "GestaltMatcher": [],
            "hpoDict": {},
            "absentHpoDict": {},
            "zeroShotResult": None,
            "phenotypeSearchResult": None,
            "mergedDiseaseCandidates": [],
            "webresources": [],
            "memory": [],
            "sources": [],
            "evidence": [],
            "validation_status": "not_run",
            "validation_errors": [],
            "flow_iteration": 0,
            "max_flow_iterations": max_flow_iterations,
            "tentativeDiagnosis": None,
            "zeroShotRaw": None,
            "zeroShotValidation": None,
            "tentativeRaw": None,
            "tentativeValidation": None,
            "reflection": None,
            "finalDiagnosis": None,
            "onset": onset if onset else "Unknown",
            "sex": sex if sex else "Unknown",
            "patient_id": patient_id if patient_id else "unknown",
            "llm": self.llm,
            "gestalt_matcher_result_loader": gestalt_matcher_result_loader,
        }

    def run(self, hpo_list, image_path=None, verbose=False, absent_hpo_list=None, onset=None, sex=None, patient_id=None, use_absentHPO=False, filter_impotance=False, use_phenobrain=False, public_response=True, evidence_mode=DEFAULT_EVIDENCE_MODE, max_flow_iterations=1):
        if evidence_mode not in {"legacy", "v2"}:
            raise ValueError("evidence_mode must be 'legacy' or 'v2'")
        if max_flow_iterations < 0:
            raise ValueError("max_flow_iterations must be non-negative")
        initial_state = self._build_initial_state(
            hpo_list=hpo_list,
            image_path=image_path,
            absent_hpo_list=absent_hpo_list,
            onset=onset,
            sex=sex,
            patient_id=patient_id,
            use_absentHPO=use_absentHPO,
            filter_impotance=filter_impotance,
            use_phenobrain=use_phenobrain,
            evidence_mode=evidence_mode,
            max_flow_iterations=max_flow_iterations,
        )
        graph = self.v2_graph if evidence_mode == "v2" else self.graph
        if self.enable_log:
            self._write_graph_ascii_to_log(evidence_mode)
        result = graph.invoke(initial_state)
        if verbose:
            self.pretty_print(result)
        return serialize_public_state(result) if public_response else result

    def stream(
        self,
        hpo_list,
        image_path=None,
        verbose=False,
        absent_hpo_list=None,
        onset=None,
        sex=None,
        patient_id=None,
        use_absentHPO=False,
        filter_impotance=False,
        use_phenobrain=False,
        evidence_mode=DEFAULT_EVIDENCE_MODE,
        max_flow_iterations=1,
        gestalt_matcher_result_loader=None,
    ):
        """Yield ``(node_name, node_update)`` as each graph node completes.

        ``run`` remains the compatibility entry point for local tests and
        scripts.  This method exposes the same graph execution through
        LangGraph's update stream for operational adapters.
        """
        if evidence_mode not in {"legacy", "v2"}:
            raise ValueError("evidence_mode must be 'legacy' or 'v2'")
        if max_flow_iterations < 0:
            raise ValueError("max_flow_iterations must be non-negative")

        initial_state = self._build_initial_state(
            hpo_list=hpo_list,
            image_path=image_path,
            absent_hpo_list=absent_hpo_list,
            onset=onset,
            sex=sex,
            patient_id=patient_id,
            use_absentHPO=use_absentHPO,
            filter_impotance=filter_impotance,
            use_phenobrain=use_phenobrain,
            evidence_mode=evidence_mode,
            max_flow_iterations=max_flow_iterations,
            gestalt_matcher_result_loader=gestalt_matcher_result_loader,
        )
        graph = self.v2_graph if evidence_mode == "v2" else self.graph
        if self.enable_log:
            self._write_graph_ascii_to_log(evidence_mode)

        accumulated_state = dict(initial_state)
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            if not isinstance(chunk, dict):
                continue
            for node_name, node_update in chunk.items():
                if not isinstance(node_update, dict):
                    continue
                accumulated_state.update(node_update)
                yield node_name, node_update

        if verbose:
            self.pretty_print(accumulated_state)

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
