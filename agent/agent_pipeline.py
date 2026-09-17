import os
import datetime
from langgraph.graph import StateGraph, START, END
from agent.state.state_types import State
from agent.utils.logger import log_node_result
from agent.utils.profiler import profile_node
from agent.utils.hpo_importance_filter import filter_hpo_by_importance
from agent.llm.azure_llm_instance import get_llm_instance
from agent.tools.zebraseek_flow import (
    build_patient_input,
    create_zebraseek_context,
    finalize_zebraseek_context,
    run_zebraseek,
    stage_candidate_research,
    stage_disease_normalization,
    stage_genes,
    stage_initial_tools,
    stage_reflection,
    stage_rerank,
)

from agent.nodes import (
    PCFnode, createDiagnosisNode, createZeroShotNode, createHPODictNode,createAbsentHPODictNode, 
    diseaseNormalizeNode, diseaseSearchNode, reflectionNode,
    BeginningOfFlowNode, finalDiagnosisNode, GestaltMatcherNode,
    diseaseNormalizeForFinalNode, HPOwebSearchNode,
    NormalizePCFNode, NormalizeGestaltMatcherNode, NormalizeZeroShotNode, DiseaseSearchWithHPONode,
    PhenoBrainNode, mergeCandidateResultsNode
)


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

# The active graph follows the same declarative node/edge registry style as
# the original ``main`` workflow.  The implementation methods are kept on
# ``RareDiseaseDiagnosisPipeline`` so the existing profiler and logger wrapper
# can be applied uniformly.
ZEBRASEEK_NODE_DEFINITIONS = [
    ("ZebraSeekInputNode", "ZebraSeekInputNode"),
    ("InitialToolsNode", "InitialToolsNode"),
    ("DiseaseNormalizeNode", "DiseaseNormalizeNode"),
    ("CandidateResearchNode", "CandidateResearchNode"),
    ("ReflectionNode", "ReflectionNode"),
    ("RerankNode", "RerankNode"),
    ("GeneAnnotationNode", "GeneAnnotationNode"),
    ("ZebraSeekOutputNode", "ZebraSeekOutputNode"),
]

ZEBRASEEK_EDGES = [
    (START, "ZebraSeekInputNode"),
    ("ZebraSeekInputNode", "InitialToolsNode"),
    ("InitialToolsNode", "DiseaseNormalizeNode"),
    ("DiseaseNormalizeNode", "CandidateResearchNode"),
    ("CandidateResearchNode", "ReflectionNode"),
    ("ReflectionNode", "RerankNode"),
    ("RerankNode", "GeneAnnotationNode"),
    ("GeneAnnotationNode", "ZebraSeekOutputNode"),
    ("ZebraSeekOutputNode", END),
]


def _has_any_correct_reflection(reflection) -> bool:
    if not reflection or not hasattr(reflection, "ans") or not reflection.ans:
        return False
    return any(getattr(ans_item, "Correctness", False) for ans_item in reflection.ans)

class RareDiseaseDiagnosisPipeline:
    def __init__(
        self,
        model_name: str = 'gpt-4o',
        enable_log=False,
        log_filename=None,
        log_dir=None,
        *,
        llm=None,
        ranking_mode: str = "llm",
        use_togomcp: bool = True,
        depth: int = 1,
        use_phenobrain: bool = True,
        togo_client=None,
        workflow: str = "zebraseek",
    ):
        # Keep the original LangGraph entry point.  The nodes below describe
        # the new ZebraSeek route, while ``_build_legacy_graph`` preserves the
        # historical node graph for callers that still need it.
        self.enable_log = enable_log
        self.logfile_path = None
        self.log_filename = log_filename
        self.log_dir = log_dir
        self.ranking_mode = ranking_mode
        self.use_togomcp = use_togomcp
        self.depth = depth
        self.use_phenobrain = use_phenobrain
        self.togo_client = togo_client
        if workflow not in {"zebraseek", "legacy"}:
            raise ValueError("workflow must be 'zebraseek' or 'legacy'")
        self.workflow = workflow

        self._llm_init_error = None
        if llm is not None:
            self.llm = llm
        else:
            try:
                self.llm = get_llm_instance(model_name)
            except Exception as exc:
                # Tool-only execution remains available when Azure settings
                # are absent.  This matches the old constructor's public API
                # while making the failure visible in the node log.
                self.llm = None
                self._llm_init_error = str(exc)

        self.legacy_graph = self._build_legacy_graph()
        self.zebraseek_graph = self._build_graph()
        self.graph = self.legacy_graph if workflow == "legacy" else self.zebraseek_graph
        if self.enable_log:
            if self.logfile_path is None:
                self.logfile_path = self._get_logfile_path()
            self._write_graph_ascii_to_log()
            if self._llm_init_error:
                self._log("LLM initialization", {"status": "unavailable", "error": self._llm_init_error})
            
    def _get_logfile_path(self):
        log_dir = self.log_dir or os.path.join(os.getcwd(), "log")
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

    @profile_node
    def ZebraSeekInputNode(self, state: State):
        """Normalize the legacy State input into the canonical patient input."""
        patient = state.get("patient_input")
        if not patient:
            patient = build_patient_input(
                state.get("hpoList", []),
                absent_hpo_ids=state.get("absentHpoList", []),
                sex=state.get("sex"),
                onset=state.get("onset"),
                image_path=state.get("imagePath"),
                patient_id=state.get("patient_id"),
            )
        context = create_zebraseek_context(
            patient,
            llm=state.get("llm", self.llm),
            togo_client=state.get("togo_client", self.togo_client),
            ranking_mode=state.get("ranking_mode", self.ranking_mode),
            depth=state.get("zebraseek_depth", self.depth),
            use_togomcp=state.get("use_togomcp", self.use_togomcp),
            use_phenobrain=state.get("use_phenobrain", self.use_phenobrain),
        )
        return {"patient_input": patient, "zebraseek_context": context}

    @profile_node
    def InitialToolsNode(self, state: State):
        """Run the parallel initial rankers and Top5 candidate union."""
        return {"zebraseek_context": stage_initial_tools(state["zebraseek_context"])}

    @profile_node
    def DiseaseNormalizeNode(self, state: State):
        """Apply the legacy OMIM based disease-name normalization boundary."""
        return {"zebraseek_context": stage_disease_normalization(state["zebraseek_context"])}

    @profile_node
    def CandidateResearchNode(self, state: State):
        """Run fixed TogoMCP evidence searches and one-hop expansion."""
        return {"zebraseek_context": stage_candidate_research(state["zebraseek_context"])}

    @profile_node
    def ReflectionNode(self, state: State):
        """Assess candidates after all fixed searches complete."""
        return {"zebraseek_context": stage_reflection(state["zebraseek_context"])}

    @profile_node
    def RerankNode(self, state: State):
        """Apply LLM or tool-average ranking."""
        return {"zebraseek_context": stage_rerank(state["zebraseek_context"])}

    @profile_node
    def GeneAnnotationNode(self, state: State):
        """Look up causal candidate genes for the final five diseases."""
        return {"zebraseek_context": stage_genes(state["zebraseek_context"])}

    @profile_node
    def ZebraSeekOutputNode(self, state: State):
        """Project the canonical output back into the historical State shape."""
        result = finalize_zebraseek_context(state["zebraseek_context"])
        return {
            "patient_input": result.get("patient_input", state.get("patient_input")),
            "tool_response_records": result.get("tool_response_records", []),
            "initial_tool_responses": result.get("initial_tool_responses", []),
            "source_records": result.get("source_records", []),
            "evidence_records": result.get("evidence_records", []),
            "normalization_records": result.get("normalization_records", []),
            "disease_ranking_index": result.get("disease_ranking_index", {}),
            "candidate_pool": result.get("candidate_pool", []),
            "candidate_records": result.get("candidate_records", []),
            "search_sessions": result.get("search_sessions", []),
            "candidate_search_sessions": result.get("candidate_search_sessions", []),
            "action_bindings": result.get("action_bindings", {}),
            "tool_call_records": result.get("tool_call_records", []),
            "prompt_records": result.get("prompt_records", []),
            "procedure_trace": result.get("procedure_trace", {}),
            "reflection_assessments": result.get("reflection_assessments", []),
            "ranked_candidates": result.get("ranked_candidates", []),
            "final_ranking": result.get("final_ranking", []),
            "final_candidates": result.get("final_candidates", []),
            "gene_annotations": result.get("gene_annotations", []),
            "execution_metadata": result.get("execution_metadata", {}),
            "finalDiagnosis": result.get("finalDiagnosis"),
            "final_output": result.get("final_output", result),
            "zebraseek_output": result.get("zebraseek_output", result.get("final_output", result)),
        }

    # Backwards-compatible method aliases used by earlier test harnesses.
    _zebraseek_input_node = ZebraSeekInputNode
    _zebraseek_initial_tools_node = InitialToolsNode
    _zebraseek_disease_normalize_node = DiseaseNormalizeNode
    _zebraseek_candidate_research_node = CandidateResearchNode
    _zebraseek_reflection_node = ReflectionNode
    _zebraseek_rerank_node = RerankNode
    _zebraseek_gene_annotation_node = GeneAnnotationNode
    _zebraseek_output_node = ZebraSeekOutputNode

    def _build_graph(self):
        """Build the active LangGraph for the fixed ZebraSeek workflow."""
        graph_builder = StateGraph(State)

        def wrap(node_func, node_name):
            def wrapped(state):
                result = node_func(state)
                self._log(node_name, result)
                return result
            return wrapped

        for node_name, method_name in ZEBRASEEK_NODE_DEFINITIONS:
            graph_builder.add_node(node_name, wrap(getattr(self, method_name), node_name))
        for source, destination in ZEBRASEEK_EDGES:
            graph_builder.add_edge(source, destination)
        return graph_builder.compile()

    def _build_legacy_graph(self):
        """Build the pre-ZebraSeek graph for backwards-compatible callers."""
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

        for node_name, node_func in NODE_DEFINITIONS:
            graph_builder.add_node(node_name, wrap_node(node_func, node_name))
        
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

    def _build_initial_state(self, hpo_list, image_path=None, absent_hpo_list=None, onset=None, sex=None, patient_id=None, use_absentHPO=False, filter_impotance=False, use_phenobrain=True):
        if filter_impotance:
            hpo_list = filter_hpo_by_importance(hpo_list)
            absent_hpo_list = filter_hpo_by_importance(absent_hpo_list or [])

        patient_input = build_patient_input(
            hpo_list,
            absent_hpo_ids=absent_hpo_list,
            onset=onset,
            sex=sex,
            image_path=image_path,
            patient_id=patient_id,
        )
        return {
            "depth": 0,
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
            "tentativeDiagnosis": None,
            "reflection": None,
            "finalDiagnosis": None,
            "onset": onset if onset else "Unknown",
            "sex": sex if sex else "Unknown",
            "patient_id": patient_id if patient_id else "unknown",
            "llm": self.llm,
            # Canonical ZebraSeek state and trace fields.  Keeping the legacy
            # keys above means old callers can inspect the same initial state.
            "patient_input": patient_input,
            "ranking_mode": self.ranking_mode,
            "zebraseek_depth": self.depth,
            "use_togomcp": self.use_togomcp,
            "togo_client": self.togo_client,
            "tool_response_records": [],
            "initial_tool_responses": [],
            "normalization_records": [],
            "source_records": [],
            "evidence_records": [],
            "disease_ranking_index": {},
            "candidate_pool": [],
            "candidate_records": [],
            "search_sessions": [],
            "candidate_search_sessions": [],
            "action_bindings": {},
            "tool_call_records": [],
            "prompt_records": [],
            "procedure_trace": {},
            "reflection_assessments": [],
            "ranked_candidates": [],
            "final_ranking": [],
            "final_candidates": [],
            "gene_annotations": [],
            "execution_metadata": {},
            "zebraseek_output": {},
            "zebraseek_context": {},
            "final_output": {},
        }

    def invoke(self, state: State):
        """Delegate directly to the compiled LangGraph, like the old API."""
        return self.graph.invoke(state)

    def run(self, hpo_list=None, image_path=None, verbose=False, absent_hpo_list=None, onset=None, sex=None, patient_id=None, use_absentHPO=False, filter_impotance=False, use_phenobrain=None, use_togomcp=None, *, present_hpo_ids=None, absent_hpo_ids=None, ranking_mode=None):
        if present_hpo_ids is not None:
            hpo_list = present_hpo_ids
        if absent_hpo_ids is not None:
            absent_hpo_list = absent_hpo_ids
        if hpo_list is None:
            hpo_list = []
        initial_state = self._build_initial_state(
            hpo_list=hpo_list,
            image_path=image_path,
            absent_hpo_list=absent_hpo_list,
            onset=onset,
            sex=sex,
            patient_id=patient_id,
            use_absentHPO=use_absentHPO,
            filter_impotance=filter_impotance,
            use_phenobrain=self.use_phenobrain if use_phenobrain is None else use_phenobrain,
        )
        initial_state["use_togomcp"] = self.use_togomcp if use_togomcp is None else use_togomcp
        initial_state["ranking_mode"] = ranking_mode or self.ranking_mode
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
