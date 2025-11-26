import json
from agent.state.state_types import ZeroShotOutput, DiagnosisOutput, ReflectionOutput, PhenotypeSearchFormat

def _write_disease_search_prompt(f):
    """diseaseSearchNode用の固定プロンプトを書き込む"""
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

def _write_reflection_prompt(f, prompt, result):
    """reflectionNode用のプロンプトを整形して書き込む (promptがリストであることに対応)"""
    # promptがリストでない、または空の場合は何もしない
    if not isinstance(prompt, list) or not prompt:
        return

    prompts = prompt
    ans_list = []
    # result辞書からreflectionオブジェクトを取得し、その中のansリストを取得する
    if isinstance(result, dict) and "reflection" in result:
        reflection_obj = result.get("reflection")
        if reflection_obj and hasattr(reflection_obj, "ans"):
            ans_list = reflection_obj.ans or []

    for i, p_str in enumerate(prompts):
        disease_name = ""
        # 対応するansがあれば病名を取得、なければインデックスで補完
        if i < len(ans_list):
            disease_name = getattr(ans_list[i], "disease_name", f"#{i+1}")
        else:
            disease_name = f"#{i+1}"
        
        f.write(f"\n----- Reflection Prompt for: {disease_name} -----\n")
        f.write(p_str.strip() + "\n")
        f.write("----- End Reflection Prompt -----\n")

def _write_generic_prompt(f, prompt):
    """一般的なプロンプトを書き込む"""
    if not isinstance(prompt, str) or not prompt.strip():
        return
    f.write("\n----- Prompt Start -----\n")
    f.write(prompt.strip() + "\n")
    f.write("----- Prompt End -----\n")

def _format_and_write_result(f, result):
    """結果オブジェクトを適切な形式でJSONとして書き込む"""
    if isinstance(result, (ZeroShotOutput, DiagnosisOutput, ReflectionOutput)):
        f.write(result.model_dump_json(indent=2))
    elif isinstance(result, list) and all(isinstance(item, PhenotypeSearchFormat) for item in result):
        f.write("Top similar diseases from phenotype search:\n")
        for item in result:
            f.write(f"  - {item.disease_info.disease_name} (OMIM: {item.disease_info.OMIM_id}, Score: {item.similarity_score:.4f})\n")
    elif hasattr(result, "dict"): # pydanticモデルだが上記でキャッチされなかった場合
        f.write(json.dumps(result.dict(), indent=2))
    elif isinstance(result, dict):
        def default(o):
            if hasattr(o, "model_dump"): return o.model_dump()
            if hasattr(o, "dict"): return o.dict()
            return str(o)
        f.write(json.dumps(result, indent=2, default=default))
    elif isinstance(result, list):
        for item in result:
            if hasattr(item, "model_dump_json"):
                f.write(item.model_dump_json(indent=2) + "\n")
            elif hasattr(item, "dict"):
                f.write(json.dumps(item.dict(), indent=2) + "\n")
            else:
                f.write(str(item) + "\n")
    else:
        f.write(str(result))

def log_node_result(logfile_path: str, node_name: str, result: any):
    """
    ノードの実行結果を整形してログファイルに追記する
    """
    if not logfile_path:
        return
        
    with open(logfile_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== {node_name} ===\n")
        
        if node_name == "diseaseSearchNode":
            _write_disease_search_prompt(f)
        
        try:
            prompt = None
            original_result = result
            core_result = result

            if isinstance(result, dict) and "prompt" in result:
                prompt = result["prompt"]
                # reflectionNodeの場合、結果本体は"reflection"キーの中身
                if node_name == "reflectionNode":
                    core_result = result.get("reflection", result)
                else:
                    core_result = result.get("result", result)

            if prompt:
                if node_name == "reflectionNode":
                    _write_reflection_prompt(f, prompt, original_result)
                else:
                    _write_generic_prompt(f, prompt)
                f.write("Result:\n")
            
            _format_and_write_result(f, core_result)

        except Exception as e:
            f.write(f"ログ整形エラー: {e}\n")
        f.write("\n")