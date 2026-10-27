"""Operational streaming adapter for the diagnosis pipeline.

The adapter mirrors the worker-facing ``agent_adapter.run_stream`` contract:
callers receive one ``(node_name, node_update)`` pair as each LangGraph node
completes.  Persistence remains the caller's responsibility, so local test
runners can continue using ``RareDiseaseDiagnosisPipeline.run()`` unchanged.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator, Optional

from agent.agent_pipeline import RareDiseaseDiagnosisPipeline
from agent.utils.response_serializer import (
    serialize_external_compatibility,
    serialize_public_state,
)


DEFAULT_MODEL_NAME = "gpt-4o"

AGENT_OUTPUT_KEY_BY_STEP_TYPE = {
    "PUB_CASE_FINDER": "pubCaseFinder",
    "GESTALT_MATCHER": "GestaltMatcher",
    "PHENOTYPE_SEARCH": "phenotypeSearchResult",
    "ZERO_SHOT": "zeroShotResult",
    "MERGE_CANDIDATES": "mergedDiseaseCandidates",
    "TENTATIVE_DIAGNOSIS": "tentativeDiagnosis",
    "REFLECTION": "reflection",
    "FINAL_DIAGNOSIS": "finalDiagnosis",
}

# Only terminal nodes for the externally visible stages are mapped. Internal
# helper nodes (HPO web search, candidate metadata, PhenoBrain, and raw
# normalization steps) remain available in the internal node trace but are not
# sent as separate external stages.
STEP_TYPE_BY_NODE_NAME = {
    "NormalizePCFNode": "PUB_CASE_FINDER",
    "NormalizeGestaltMatcherNode": "GESTALT_MATCHER",
    "DiseaseSearchWithHPONode": "PHENOTYPE_SEARCH",
    "NormalizeZeroShotNode": "ZERO_SHOT",
    # Candidate metadata is enriched after the raw merge.  Persist the
    # externally visible merged-candidate stage only after that enrichment so
    # tool_rankings/tool_supports are present.
    "candidateMetadataNode": "MERGE_CANDIDATES",
    "diseaseNormalizeNode": "TENTATIVE_DIAGNOSIS",
    "reflectionNode": "REFLECTION",
    "diagnosisOutputValidationNode": "FINAL_DIAGNOSIS",
}

# The compatibility serializer uses a capitalized file-level name for the
# phenotype-search artifact while the public State key is lower camel case.
EXTERNAL_SERIALIZER_KEY_BY_OUTPUT_KEY = {
    "phenotypeSearchResult": "PhenotypeSearchResult",
}


def get_step_type_for_node(node_name: str) -> Optional[str]:
    """Return the external stage for a terminal graph node, if any."""
    return STEP_TYPE_BY_NODE_NAME.get(node_name)


def resolve_env_bool(value: Optional[bool], name: str, default: bool = False) -> bool:
    """Resolve an optional flag using an environment variable fallback."""
    if value is not None:
        return value
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def make_json_serializable(value: Any) -> Any:
    """Convert model and container values into JSON-compatible values."""
    if hasattr(value, "model_dump"):
        return make_json_serializable(value.model_dump())
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
            if key not in {"llm", "gestalt_matcher_result_loader"}
        }
    if isinstance(value, (list, tuple, set)):
        return [make_json_serializable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def build_step_output_json(step_type: str, agent_result: dict[str, Any]) -> dict[str, Any]:
    """Build one of the eight external stage payloads.

    The worker-side ``agent.py`` can call this function before
    ``save_completed_step``.  Diagnosis stages use the v2 compatibility
    serializer, so bilingual fields, citation numbers, and structured
    references are retained.  Only the requested stage is returned; auxiliary
    ``phenoBrain``, ``webresources``, and trace ledgers are not emitted as
    separate external stages.

    Unknown stage names deliberately fall back to the raw serializable result
    so a graph change does not crash the operational runner.
    """
    if not isinstance(agent_result, dict):
        agent_result = {"agentResult": agent_result}

    output_key = AGENT_OUTPUT_KEY_BY_STEP_TYPE.get(step_type)
    serializable_result = make_json_serializable(agent_result)
    if not output_key:
        return {"stepType": step_type, "agentResult": serializable_result}

    # The stream yields node deltas, not the initial graph state. Supply the
    # v2 marker and empty ledgers when they have not appeared yet so the
    # compatibility serializer still applies its canonical projection.
    state = dict(agent_result)
    state.setdefault("evidence_mode", "v2")
    state.setdefault("sources", [])
    state.setdefault("evidence", [])
    state.setdefault("validation_status", "not_run")
    state.setdefault("validation_errors", [])

    external = serialize_external_compatibility(state)
    serializer_key = EXTERNAL_SERIALIZER_KEY_BY_OUTPUT_KEY.get(
        output_key,
        output_key,
    )
    stage_payload = external.get(serializer_key)
    if isinstance(stage_payload, dict):
        if serializer_key != output_key and serializer_key in stage_payload:
            return {
                output_key: make_json_serializable(stage_payload[serializer_key])
            }
        return make_json_serializable(stage_payload)

    # Be tolerant of a serializer/schema change: use the canonical public
    # projection as a fallback while preserving the eight-key contract.
    public = serialize_public_state(state)
    return {output_key: make_json_serializable(public.get(output_key))}


def run_stream(
    *,
    hpo_ids: list[str],
    gestalt_matcher_result_loader=None,
    absent_hpo_ids: Optional[list[str]] = None,
    patient_id: Optional[str] = None,
    sex: Optional[str] = None,
    onset: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    enable_log: Optional[bool] = None,
    verbose: bool = False,
    use_absent_hpo: bool = False,
    filter_importance: Optional[bool] = None,
    image_path: Optional[str] = None,
    evidence_mode: str = "v2",
    max_flow_iterations: int = 1,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Run the pipeline and yield each completed node update.

    The parameter names intentionally match the worker-side adapter contract;
    ``filter_importance`` is mapped to the pipeline's existing internal
    ``filter_impotance`` spelling for compatibility.
    """
    resolved_enable_log = resolve_env_bool(
        enable_log,
        "ZEBRASEEK_AGENT_FILE_LOG_ENABLED",
        False,
    )
    resolved_filter_importance = resolve_env_bool(
        filter_importance,
        "ZEBRASEEK_AGENT_FILTER_IMPORTANCE",
        False,
    )
    pipeline = RareDiseaseDiagnosisPipeline(
        model_name=model_name,
        enable_log=resolved_enable_log,
    )
    yield from pipeline.stream(
        hpo_list=hpo_ids,
        image_path=image_path,
        absent_hpo_list=absent_hpo_ids or [],
        onset=onset,
        sex=sex,
        patient_id=patient_id,
        use_absentHPO=use_absent_hpo,
        filter_impotance=resolved_filter_importance,
        evidence_mode=evidence_mode,
        max_flow_iterations=max_flow_iterations,
        gestalt_matcher_result_loader=gestalt_matcher_result_loader,
        verbose=verbose,
    )
