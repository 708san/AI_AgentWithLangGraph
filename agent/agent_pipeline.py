import os
import datetime
import json
from langgraph.graph import StateGraph, START, END
from agent.state.state_types import State, ZeroShotOutput, DiagnosisOutput, ReflectionOutput

from agent.nodes import (
    PCFnode, createDiagnosisNode, createZeroShotNode, createHPODictNode,
    diseaseNormalizeNode, dieaseSearchNode, reflectionNode,
    BeginningOfFlowNode, finalDiagnosisNode, GestaltMatcherNode,
    diseaseNormalizeForFinalNode
)

class RareDiseaseDiagnosisPipeline:
    def __init__(self, enable_log=False, log_filename=None):
        self.graph = self._build_graph()
        self.enable_log = enable_log
        self.logfile_path = None
        self.log_filename = log_filename
        if self.enable_log:
            self.logfile_path = self._get_logfile_path()
            self._write_graph_ascii_to_log()
            
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
        if not self.enable_log or not self.logfile_path:
            return
        with open(self.logfile_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== {node_name} ===\n")
            # diseaseSearchNodeの直後にSummarize Prompt for DiseaseSearchを表示
            if node_name == "diseaseSearchNode":
                f.write("\n----- Summarize Prompt for DiseaseSearch -----\n")
                f.write("""
You are an expert clinical geneticist and a diagnostician. Your critical task is to analyze a medical text and convert it into a high-yield, structured summary designed specifically for differential diagnosis. Your output must not only list symptoms but also highlight features that distinguish the condition from its clinical mimics.

Instructions:

From the text I provide, generate a summary strictly following these rules:

1. Information to Extract (Include ONLY these):

Disease: The name of the syndrome or disorder.

Genetics: The causative gene(s) and inheritance pattern. If not specified, state "Not specified".

Key Phenotypes: A concise, bulleted list of the core clinical features and symptoms.

Differentiating Features: This is the most critical section. Extract features that are particularly useful for distinguishing this syndrome from others. This includes:

Hallmark signs: Features that are highly characteristic or pathognomonic.

Key negative findings: Symptoms typically ABSENT in this condition but present in similar ones (e.g., "Absence of hyperphagia").

Unique constellations: A specific combination of symptoms that points strongly to this diagnosis.

2. Information to Exclude (Strictly Omit):

Patient case histories, family origins, or demographic details.

Treatment, management, or therapeutic strategies.

Research methodology, study populations, or author details.

Prognosis, mortality, or prevalence statistics.

General background information that isn't a clinical feature.

3. Output Format (Use this exact structure):

Disease: [Name of the disease]
Genetics: [Gene(s), Inheritance pattern]
Key Phenotypes:

[Bulleted list of core clinical features]

[Example: Intellectual disability]

[Example: Craniofacial dysmorphism]
Differentiating Features:

Hallmark(s): [List highly specific or unique signs.]

Key Negative Finding(s): [List what is typically absent, e.g., "Absence of..."]

Unique Constellation: [Describe a diagnostically powerful combination of symptoms.]

Now, process the following text:
"""
)
                f.write("----- End Summarize Prompt for DiseaseSearch -----\n\n")
            try:
                # プロンプト付きのdictの場合はプロンプトも出力
                if isinstance(result, dict) and "prompt" in result:
                    prompt = result["prompt"]
                    if node_name == "reflectionNode" and prompt.strip():
                        prompts = prompt.split("\n---\n") if "\n---\n" in prompt else prompt.split("\n\n")
                        ans_list = result.get("result", result)
                        ans_list = getattr(ans_list, "ans", None) or []
                        for i, p in enumerate(prompts):
                            disease_name = ""
                            if i < len(ans_list):
                                disease_name = getattr(ans_list[i], "disease_name", f"#{i+1}")
                            f.write(f"\n----- Reflection Prompt for: {disease_name} -----\n")
                            f.write(p.strip() + "\n")
                            f.write("----- End Reflection Prompt -----\n")
                    else:
                        f.write("\n----- Prompt Start -----\n")
                        f.write(prompt.strip() + "\n")
                        f.write("----- Prompt End -----\n")
                    f.write("Result:\n")
                    result = result.get("result", result)
                # ZeroShotOutput, DiagnosisOutput, ReflectionOutputはpydanticモデル
                if isinstance(result, ZeroShotOutput) or isinstance(result, DiagnosisOutput) or isinstance(result, ReflectionOutput):
                    f.write(result.model_dump_json(indent=2, ensure_ascii=False))
                elif hasattr(result, "dict"):
                    f.write(json.dumps(result.dict(), ensure_ascii=False, indent=2))
                elif isinstance(result, dict):
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
                # プロンプト付きdictの場合はresult["result"]を返す
                if isinstance(result, dict) and "result" in result:
                    return result["result"]
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
        graph_builder.add_node("diseaseNormalizeForFinalNode", wrap_node(diseaseNormalizeForFinalNode, "diseaseNormalizeForFinalNode"))

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
        graph_builder.add_edge("finalDiagnosisNode", "diseaseNormalizeForFinalNode")
        graph_builder.add_edge("diseaseNormalizeForFinalNode", END)
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