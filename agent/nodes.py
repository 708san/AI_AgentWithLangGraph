import copy
import os

from .state.state_types import State, ReflectionOutput, ReflectionOutputV2, DiagnosisOutputV2
from .tools.pcf_api import callingPCF
from .tools.diagnosis import createDiagnosis
from .tools.diagnosis import createDiagnosisV2
from .tools.ZeroShot import createZeroshot
from .tools.make_HPOdic import make_hpo_dic
from .tools.reflection import create_reflection, create_reflection_v2
from .tools.diseaseSearch import diseaseSearchForDiagnosis
from .tools.diseaseNormalize import diseaseNormalizeForDiagnosis, normalize_pcf_results, normalize_gestalt_results, normalize_zeroshot_results
from .tools.candidate_metadata import canonical_omim_id
from .tools.finalDiagnosis import createFinalDiagnosis
from .tools.gestaltMathcher import call_gestalt_matcher_api
from .tools.HPOwebReserch import search_hpo_terms
from .tools.embeddingSearchWithHPO import embedding_search_with_hpo
from .tools.rankingMerge import merge_ranked_disease_candidates
from .tools.phenobrain_api import call_phenobrain
from .tools.candidate_metadata import enrich_candidates
from .tools.literature_search import search_literature_for_candidates
from .tools.evidence_extraction import extract_evidence_for_candidates
from .tools.evidence_validation import validate_v2_output
from .utils.reasoning import attach_diagnosis_reasoning, attach_reflection_reasoning

from .utils.result_saver import save_result
from .utils.profiler import profile_node

from concurrent.futures import ThreadPoolExecutor, as_completed


REFLECTION_MAX_WORKERS = int(os.getenv("REFLECTION_MAX_WORKERS", "6"))


def _empty_reflection_output() -> ReflectionOutput:
    return ReflectionOutput(ans=[])

@profile_node
def BeginningOfFlowNode(state: State):
    print("BeginningOfFlowNode called")
    state["depth"] += 1
    print(f"Current depth: {state['depth']}")
    # reset Diagnosis and Reflection when starting a new flow
    
    return {"depth": state["depth"], "tentativeDiagnosis": None, "reflection": None}

@profile_node
@save_result("HPOwebSearchNode")
def HPOwebSearchNode(state: State):
    print("HPOwebSearchNode called")
    try:
        webresources = search_hpo_terms(state)
        
        # 既存のwebresourcesとマージ（重複排除はsearch_hpo_terms内で実施済み想定）
        state["webresources"] = state.get("webresources", []) + webresources
        return {"webresources": state["webresources"]}
    except Exception as e:
        print(f"Error in HPOwebSearchNode: {e}")
        return {"webresources": state.get("webresources", [])}
    
@profile_node
@save_result("DiseaseSearchWithHPONode")
def DiseaseSearchWithHPONode(state: State):
    """
    Searches for diseases with similar phenotypes based on the patient's HPO list using embedding vector search.
    """
    print("DiseaseSearchWithHPONode called")

    # Following the design principle of passing the state to tool functions.
    # It is expected that the search_phenotypes_by_embedding function will:
    # 1. Extract the hpo_list from the state.
    # 2. Execute the search.
    # 3. Return the results as a List[PhenotypeSearchFormat].
    search_results = embedding_search_with_hpo(state)
    
    if not search_results:
        print("Phenotype search returned no results.")
        return {}
        
    return {"phenotypeSearchResult": search_results}


@profile_node
def PCFnode(state: State):
    print("PCFnode called")
    depth = state.get("depth", 0)
    hpo_list = state["hpoList"]
    if not hpo_list:
        return {"pubCaseFinder": []}
    result = callingPCF(hpo_list, depth)
    return {"pubCaseFinder": result}

@profile_node
@save_result("PhenoBrainNode")
def PhenoBrainNode(state: State):
    print("PhenoBrainNode called")
    if not state.get("use_phenobrain", False):
        return {"phenoBrain": []}
    hpo_list = state.get("hpoList", [])
    if not hpo_list:
        return {"phenoBrain": []}
    results = call_phenobrain(hpo_list)
    for result in results or []:
        result["omim_id"] = canonical_omim_id(result.get("omim_id"))
    return {"phenoBrain": results}

@profile_node
@save_result("NormalizePCFNode")
def NormalizePCFNode(state: State):
    """PCFの結果に含まれる病名をOMIM IDに基づいて正規化する"""
    print("NormalizePCFNode called")
    normalized_results = normalize_pcf_results(state)
    if not normalized_results:
        return {}
    return {"pubCaseFinder": normalized_results}

@profile_node
def GestaltMatcherNode(state: State):
    print("GestaltMatcherNode called")
    image_path = state.get("imagePath", None)
    depth = state.get("depth", 0)
    result_loader = state.get("gestalt_matcher_result_loader")
    if not image_path and not result_loader:
        print("No image path provided.")
        return {"GestaltMatcher": []}
    try:
        gestalt_results = (
            result_loader()
            if result_loader
            else call_gestalt_matcher_api(image_path, depth)
        )
        if isinstance(gestalt_results, dict):
            gestalt_results = (
                gestalt_results.get("GestaltMatcher")
                or gestalt_results.get("results")
                or gestalt_results.get("data")
                or []
            )
        syndrome_list = []
        for rank, res in enumerate(gestalt_results, 1):
            api_rank = res.get("rank")
            raw_distance = res.get("distance")
            if raw_distance is None:
                raw_distance = res.get("gestalt_score")
            syndrome_list.append({
                "subject_id": res.get("subject_id", ""),
                "syndrome_name": res.get("syndrome_name", ""),
                "omim_id": canonical_omim_id(res.get("omim_id")) or "",
                "image_id": res.get("image_id", ""),
                "score": res.get("score"),
                "rank": api_rank if api_rank is not None else rank,
                "rank_source": "api" if api_rank is not None else "agent_enumeration",
                "raw_distance": raw_distance,
            })
        return {"GestaltMatcher": syndrome_list}
    except Exception as e:
        print(f"Error calling GestaltMatcher API: {e}")
        return {"GestaltMatcher": []}
    
@profile_node    
@save_result("NormalizeGestaltMatcherNode")
def NormalizeGestaltMatcherNode(state: State):
    """GestaltMatcherの結果に含まれる病名をOMIM IDに基づいて正規化する"""
    print("NormalizeGestaltMatcherNode called")
    normalized_results = normalize_gestalt_results(state)
    if not normalized_results:
        return {}
    return {"GestaltMatcher": normalized_results}

@profile_node
def createHPODictNode(state: State):
    print("createHPODictNode called")
    hpo_list = state.get("hpoList", [])
    hpo_dict = make_hpo_dic(hpo_list, None)
    return {"hpoDict": hpo_dict}

@profile_node
def createAbsentHPODictNode(state: State):
    print("createAbsentHPODictNode called")
    absent_hpo_list = state.get("absentHpoList", [])
    absent_hpo_dict = make_hpo_dic(absent_hpo_list, None)
    return {"absentHpoDict": absent_hpo_dict}

@profile_node
def createZeroShotNode(state: State):
    """Generate reasoned Zero-shot output while keeping reasons log-only."""
    print("createZeroShotNode called")
    hpo_dict = state.get("hpoDict", {})
    if state.get("zeroShotResult") is not None:
        return {"zeroShotResult": state["zeroShotResult"]}
    if hpo_dict:
        reasoning = {}
        result, prompt = createZeroshot(state, reasoning_sink=reasoning.update)
        if result:
            # Keep an independent pre-normalization snapshot for evaluation.
            # The reasoned LLM payload remains outside the graph State.
            return {
                "result": {
                    "zeroShotResult": result,
                    "zeroShotRaw": copy.deepcopy(result),
                },
                "prompt": prompt,
                "zeroShotReasoning": reasoning,
            }
    return {"zeroShotResult": None}

@profile_node
@save_result("NormalizeZeroShotNode")
def NormalizeZeroShotNode(state: State):
    """ZeroShotの結果に含まれる病名を正規化し、重複を排除する"""
    print("NormalizeZeroShotNode called")
    # stateから値を取り出すのではなく、stateをそのまま渡す
    normalized_result = normalize_zeroshot_results(state)
    if not normalized_result:
        return {}
    # 既存のキー 'zeroShotResult' を上書きする
    return {
        "zeroShotResult": normalized_result,
        "zeroShotValidation": state.get("zeroShotValidation", []),
    }

@profile_node
@save_result("mergeCandidateResultsNode")
def mergeCandidateResultsNode(state: State):
    """各ツールの順位付き疾患候補を疾患単位に統合する"""
    print("mergeCandidateResultsNode called")
    merged_candidates = merge_ranked_disease_candidates(state)
    return {"mergedDiseaseCandidates": merged_candidates}

@profile_node
@save_result("createDiagnosisNode")
def createDiagnosisNode(state: State):
    """
    Gathers all preliminary reports and generates a tentative diagnosis by synthesizing them.
    """
    print("createDiagnosisNode called")
    
    # The createDiagnosis function now takes the entire state as input.
    # It will handle extracting all necessary information internally.
    validation = None
    if state.get("evidence_mode") == "v2":
        result, prompt, validation = createDiagnosisV2(state)
        result = attach_diagnosis_reasoning(
            result,
            enrich_candidates(state.get("mergedDiseaseCandidates", []) or []),
            state.get("sources", []) or [],
            state.get("evidence", []) or [],
        )
    else:
        result, prompt = createDiagnosis(state)
    
    if result:
        return {
            "tentativeDiagnosis": result,
            "tentativeRaw": copy.deepcopy(result),
            "tentativeValidation": validation,
            "prompt": prompt,
        }
    
    return {}


@profile_node
@save_result("diseaseNormalizeNode")
def diseaseNormalizeNode(state: State):
    print("diseaseNormalizeNode called")
    tentativeDiagnosis = state.get("tentativeDiagnosis", None)
    if state.get("evidence_mode") == "v2":
        return {"tentativeDiagnosis": tentativeDiagnosis}
    if tentativeDiagnosis is not None:
        normalizedDiagnosis = diseaseNormalizeForDiagnosis(tentativeDiagnosis)
        return {"tentativeDiagnosis": normalizedDiagnosis}
    return {"tentativeDiagnosis": None}

@profile_node
def diseaseSearchNode(state: State):
    print("diseaseSearchNode called")
    
    return diseaseSearchForDiagnosis(state)

@profile_node
@save_result("reflectionNode")
def reflectionNode(state: State):
    print("reflectionNode called")
    if state.get("evidence_mode") == "v2":
        reflection_result, prompt = create_reflection_v2(state)
        reflection_result = attach_reflection_reasoning(
            reflection_result,
            enrich_candidates(state.get("mergedDiseaseCandidates", []) or []),
            state.get("sources", []) or [],
            state.get("evidence", []) or [],
        )
        return {"reflection": reflection_result, "prompt": prompt}

    tentativeDiagnosis = state.get("tentativeDiagnosis")
    hpo_dict = state.get("hpoDict")
    
    if tentativeDiagnosis and hpo_dict and hasattr(tentativeDiagnosis, 'ans'):
        diagnosis_to_judge_lis = tentativeDiagnosis.ans
        if not diagnosis_to_judge_lis:
            return {"reflection": _empty_reflection_output()}

        reflection_result_list = []
        prompts = []
        
        # 並列実行関数
        def process_single_reflection(diagnosis_to_judge):
            try:
                reflection_result, prompt = create_reflection(state, diagnosis_to_judge)
                return reflection_result, prompt
            except Exception as e:
                print(f"[ERROR] Reflection failed for {diagnosis_to_judge.disease_name}: {e}")
                return None, None
        
        max_workers = min(len(diagnosis_to_judge_lis), REFLECTION_MAX_WORKERS)
        print(f"[Reflection] max_workers={max_workers}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_diagnosis = {
                executor.submit(process_single_reflection, diagnosis): diagnosis 
                for diagnosis in diagnosis_to_judge_lis
            }

            for future in as_completed(future_to_diagnosis):
                diagnosis = future_to_diagnosis[future]
                try:
                    reflection_result, prompt = future.result()
                    if reflection_result:
                        reflection_result_list.append(reflection_result)
                        prompts.append(prompt)
                except Exception as e:
                    print(f"[ERROR] Future exception for {diagnosis.disease_name}: {e}")
        
        if not reflection_result_list:
            return {"reflection": _empty_reflection_output()}
        
        reflection_output = ReflectionOutput(ans=reflection_result_list)
        return {"reflection": reflection_output, "prompt": prompts}
    
    return {"reflection": _empty_reflection_output()}


@profile_node
@save_result("finalDiagnosisNode")
def finalDiagnosisNode(state: State):
    print("finalDiagnosisNode called")
    if state.get("evidence_mode") == "v2":
        from .tools.finalDiagnosis import createFinalDiagnosisV2
        final_diagnosis, prompt = createFinalDiagnosisV2(state)
        final_diagnosis = attach_diagnosis_reasoning(
            final_diagnosis,
            enrich_candidates(state.get("mergedDiseaseCandidates", []) or []),
            state.get("sources", []) or [],
            state.get("evidence", []) or [],
        )
        return {"finalDiagnosis": final_diagnosis, "prompt": prompt}
    finalDiagnosis, prompt = createFinalDiagnosis(state)
    return {"finalDiagnosis": finalDiagnosis, "prompt": prompt}

@profile_node
@save_result("diseaseNormalizeForFinalNode")
def diseaseNormalizeForFinalNode(state: State):
    print("diseaseNormalizeForFinalNode called")
    finalDiagnosis = state.get("finalDiagnosis", None)
    if state.get("evidence_mode") == "v2":
        return {"finalDiagnosis": finalDiagnosis}
    if finalDiagnosis is not None:
        normalizedDiagnosis = diseaseNormalizeForDiagnosis(finalDiagnosis)
        return {"finalDiagnosis": normalizedDiagnosis}
    return {"finalDiagnosis": None}


@profile_node
def candidateMetadataNode(state: State):
    """Convert legacy merged candidates and enrich them for literature search."""
    # Keep every merged candidate available to the evidence pipeline. The
    # public differential is capped later, after literature and reflection.
    candidates = enrich_candidates(state.get("mergedDiseaseCandidates", []) or [])
    return {"mergedDiseaseCandidates": candidates}


@profile_node
def literatureSearchNode(state: State):
    """Retrieve PubMed sources for each v2 candidate; never writes legacy memory."""
    # Search the complete merged differential; final response limits are
    # applied separately from evidence retrieval.
    candidates = enrich_candidates(state.get("mergedDiseaseCandidates", []) or [])
    existing_sources = list(state.get("sources", []) or [])
    is_retry = bool(state.get("reflection"))
    retry_candidate_ids = {
        item.candidate_id
        for item in (getattr(state.get("reflection"), "ans", []) or [])
        if item.literature_status in {"not_retrieved", "insufficient"}
    }
    sources = search_literature_for_candidates(
        candidates,
        hpo_dict=state.get("hpoDict", {}),
        existing_sources=existing_sources,
        retry=is_retry,
        retry_candidate_ids=retry_candidate_ids if is_retry else None,
    )
    result = {"sources": sources}
    if is_retry:
        result["flow_iteration"] = state.get("flow_iteration", 0) + 1
    return result


@profile_node
def evidenceExtractionNode(state: State):
    """Extract structured Evidence and reject claims without source excerpts."""
    # Extract evidence for every candidate that reached the merged result.
    candidates = enrich_candidates(state.get("mergedDiseaseCandidates", []) or [])
    if state.get("flow_iteration", 0) > 0:
        # The retry search only targets candidates Reflection marked
        # unresolved. Avoid repeating extraction for candidates already
        # supported in the first pass; it is an expensive LLM call and cannot
        # improve their unchanged source set.
        retry_candidate_ids = {
            item.candidate_id
            for item in (getattr(state.get("reflection"), "ans", []) or [])
            if item.literature_status in {"not_retrieved", "insufficient"}
        }
        candidates = [
            candidate for candidate in candidates
            if candidate.candidate_id in retry_candidate_ids
        ]
        print(
            "[Evidence] retry extraction candidates="
            f"{len(candidates)} (unresolved candidates only)"
        )
    extracted = extract_evidence_for_candidates(
        candidates=candidates,
        sources=state.get("sources", []) or [],
        hpo_dict=state.get("hpoDict", {}),
        absent_hpo_dict=state.get("absentHpoDict", {}) if state.get("use_absentHPO") else {},
        llm=state.get("llm"),
    )
    existing = list(state.get("evidence", []) or [])
    evidence_by_id = {item.evidence_id: item for item in existing}
    evidence_by_id.update({item.evidence_id: item for item in extracted})
    return {"evidence": list(evidence_by_id.values())}


@profile_node
def diagnosisOutputValidationNode(state: State):
    final_diagnosis = state.get("finalDiagnosis")
    if not isinstance(final_diagnosis, DiagnosisOutputV2):
        return {"validation_status": "partial", "validation_errors": ["finalDiagnosis is not DiagnosisOutputV2"]}
    candidates = enrich_candidates(state.get("mergedDiseaseCandidates", []) or [])
    preexisting_errors = list(getattr(final_diagnosis, "validation_errors", []) or [])
    validated = validate_v2_output(
        final_diagnosis,
        candidates=candidates,
        sources=state.get("sources", []) or [],
        evidence=state.get("evidence", []) or [],
    )
    validated.validation_errors = list(dict.fromkeys(preexisting_errors + validated.validation_errors))
    if validated.validation_errors:
        validated.validation_status = "failed"
    validated = attach_diagnosis_reasoning(
        validated,
        candidates,
        state.get("sources", []) or [],
        state.get("evidence", []) or [],
    )
    return {
        "finalDiagnosis": validated,
        "validation_status": validated.validation_status,
        "validation_errors": validated.validation_errors,
    }


"""
@save_result("reflectionNode")
def reflectionNode(state: State):
    print("reflectionNode called")
    tentativeDiagnosis = state.get("tentativeDiagnosis")
    hpo_dict = state.get("hpoDict")
    if tentativeDiagnosis and hpo_dict:
        diagnosis_to_judge_lis = tentativeDiagnosis.ans
        reflection_result_list = []
        prompts = []
        for diagnosis_to_judge in diagnosis_to_judge_lis:
            reflection_result, prompt = create_reflection(
                state, diagnosis_to_judge
            )
            reflection_result_list.append(reflection_result)
            prompts.append(prompt)
        print(type(reflection_result_list[0]))
        return {"reflection": {"ans": reflection_result_list}, "prompt": prompts}
    return {"reflection": None}
"""
