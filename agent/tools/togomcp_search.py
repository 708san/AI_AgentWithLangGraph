"""TogoMCP-backed candidate discovery for ZebraSeek.

This first integration deliberately keeps the discovery call small and
traceable.  Adaptive follow-up search will consume the same source/evidence
records in the verification-search phase.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

from langchain.schema import HumanMessage

from ..state.state_types import (
    CandidateAdmissionOutput,
    CandidateAdmissionItem,
    DiscoveredDiseaseCandidate,
    EvidenceRecord,
    MergedDiseaseCandidate,
    SourceRecord,
    State,
    TogoMCPCallRecord,
)
from .MCP.togomcp_client import TogoMCPClient, TogoMCPError


TOGOMCP_LITERATURE_LIMIT = int(os.getenv("TOGOMCP_LITERATURE_LIMIT", "3"))
TOGOMCP_RESEARCH_MAX_CANDIDATES = int(
    os.getenv("TOGOMCP_RESEARCH_MAX_CANDIDATES", "15")
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text_from_result(result: dict[str, Any]) -> str:
    texts = []
    for block in result.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(str(block.get("text", "")))
    return "\n".join(texts).strip()


def _payload_from_result(result: dict[str, Any]) -> Any:
    structured = result.get("structured_content")
    if structured not in (None, {}):
        # FastMCP commonly wraps a text tool result as {"result": "..."}.
        if isinstance(structured, dict) and "result" in structured:
            structured = structured["result"]
        if not isinstance(structured, str):
            return structured
        try:
            return json.loads(structured)
        except json.JSONDecodeError:
            return structured

    text = _text_from_result(result)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _record_call(
    history: list[TogoMCPCallRecord],
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    history.append(
        {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result or {},
            "started_at": (result or {}).get("started_at", _now()),
            "elapsed_ms": (result or {}).get("elapsed_ms", 0.0),
            "status": "error" if error else "success",
            "error": error or "",
        }
    )


def discover_diseases_with_togomcp(state: State) -> dict[str, Any]:
    """Retain the legacy node without re-running PubCaseFinder through TogoMCP.

    Candidate generation now belongs to the direct initial-tool stage.  The
    fixed TogoMCP route verifies those candidates and derives one-hop
    candidates from its case-report/PubMed responses.  Keeping this legacy
    entry point as an explicit no-op prevents callers of the old node from
    silently invoking ``pubcasefinder_rank_by_phenotypes`` again.
    """
    if not state.get("use_togomcp", False):
        return {"togomcp_search_status": "disabled"}
    return {
        "togomcp_search_status": "legacy_pcf_route_disabled",
        "togomcp_call_history": list(state.get("togomcp_call_history", []) or []),
        "discovered_disease_candidates": [],
    }


def _candidate_key(candidate_id: str | None, disease_name: str | None) -> str:
    if candidate_id:
        return candidate_id.upper()
    return "name:" + " ".join((disease_name or "").upper().split())


def _admission_prompt(state: State, discovered: list[DiscoveredDiseaseCandidate]) -> str:
    present = ", ".join((state.get("hpoDict", {}) or {}).values())
    absent = ", ".join((state.get("absentHpoDict", {}) or {}).values()) if state.get("use_absentHPO") else ""
    rows = []
    for candidate in discovered:
        found = candidate.get("discovered_by", {})
        rows.append(
            f"- candidate_id={candidate.get('candidate_id')}; "
            f"disease={candidate.get('disease_name')}; "
            f"rank={found.get('rank')}; score={found.get('score')}; "
            f"evidence_ids={candidate.get('discovery_evidence_ids', [])}"
        )
    absent_line = f"\nAbsent HPO: {absent}" if absent else ""
    return (
        "You are a candidate-admission reviewer for a rare-disease phenotype search.\n"
        "Decide only whether each disease returned by TogoMCP should be added to the "
        "verification candidate pool. Do not invent facts or disease relationships. "
        "Use only the supplied phenotype query and TogoMCP result fields.\n"
        "A candidate can be included when the returned phenotype-ranking evidence makes "
        "it relevant enough for downstream verification; otherwise exclude it.\n"
        "Return one item for every candidate, preserving candidate_id and disease_name.\n\n"
        f"Present HPO: {present}{absent_line}\n"
        "TogoMCP-discovered candidates:\n" + "\n".join(rows)
    )


def admit_discovered_candidates(state: State) -> dict[str, Any]:
    """Use a structured LLM decision to promote TogoMCP discoveries."""
    discovered = list(state.get("discovered_disease_candidates", []) or [])
    if not discovered:
        return {"candidate_admission_decisions": []}

    llm = state.get("llm")
    if not llm:
        return {"candidate_admission_decisions": []}

    prompt = _admission_prompt(state, discovered)
    try:
        structured_llm = llm.get_structured_llm(CandidateAdmissionOutput)
        output = llm.invoke_with_content_filter_retry(
            structured_llm,
            [HumanMessage(content=prompt)],
            context="TogoMCPCandidateAdmission",
        )
    except Exception as exc:
        print(f"[TogoMCP] candidate admission failed: {exc}")
        return {"candidate_admission_decisions": []}

    decisions = list(getattr(output, "candidates", []) or [])
    existing = list(state.get("mergedDiseaseCandidates", []) or [])
    existing_keys = {
        _candidate_key(candidate.get("OMIM_id"), candidate.get("disease_name"))
        for candidate in existing
    }

    for decision in decisions:
        if not isinstance(decision, CandidateAdmissionItem):
            continue
        if not decision.include:
            continue
        key = _candidate_key(decision.candidate_id, decision.disease_name)
        if key in existing_keys:
            continue
        source = next(
            (
                item
                for item in discovered
                if _candidate_key(item.get("candidate_id"), item.get("disease_name")) == key
            ),
            None,
        )
        if source is None:
            continue
        discovered_by = source.get("discovered_by", {})
        existing.append(
            {
                "disease_name": decision.disease_name,
                "OMIM_id": decision.candidate_id if decision.candidate_id.startswith("OMIM:") else None,
                "consensus_count": 1,
                "best_rank": discovered_by.get("rank", 9999),
                "tool_rankings": [
                    {
                        "tool": "TogoMCP:FixedSearch",
                        "rank": discovered_by.get("rank"),
                        "score": discovered_by.get("score"),
                        "note": decision.reason,
                    }
                ],
            }
        )
        existing_keys.add(key)

    return {
        "mergedDiseaseCandidates": existing,
        "candidate_admission_decisions": decisions,
    }


def _candidate_field(candidate: Any, field: str, default: Any = "") -> Any:
    if isinstance(candidate, dict):
        return candidate.get(field, default)
    return getattr(candidate, field, default)


def _research_candidates(state: State) -> list[dict[str, Any]]:
    tentative = state.get("tentativeDiagnosis")
    if tentative is not None and hasattr(tentative, "ans"):
        return [
            {
                "candidate_id": _candidate_field(item, "OMIM_id", ""),
                "disease_name": _candidate_field(item, "disease_name", ""),
            }
            for item in (tentative.ans or [])
            if _candidate_field(item, "disease_name", "")
        ][:TOGOMCP_RESEARCH_MAX_CANDIDATES]

    return [
        {
            "candidate_id": item.get("OMIM_id", ""),
            "disease_name": item.get("disease_name", ""),
        }
        for item in (state.get("mergedDiseaseCandidates", []) or [])
        if item.get("disease_name")
    ][:TOGOMCP_RESEARCH_MAX_CANDIDATES]


def _candidate_has_source(state: State, candidate_id: str, disease_name: str) -> bool:
    candidate_id = str(candidate_id or "").upper()
    disease_name_key = " ".join(str(disease_name or "").upper().split())
    source_types = {
        str(source.get("source_id")): str(source.get("source_type", ""))
        for source in state.get("source_records", []) or []
    }
    for evidence in state.get("evidence_records", []) or []:
        if source_types.get(str(evidence.get("source_id"))) not in {
            "literature",
            "literature_search",
        }:
            continue
        for link in evidence.get("candidate_links", []) or []:
            link_id = str(link.get("candidate_id", "")).upper()
            link_name = " ".join(str(link.get("candidate_label", "")).upper().split())
            if (candidate_id and link_id == candidate_id) or (
                disease_name_key and link_name == disease_name_key
            ):
                return True
    return False


def _extract_pmids(result: dict[str, Any]) -> list[str]:
    text = _text_from_result(result)
    match = re.search(r"PubMed IDs \(PMIDs\):\s*([0-9,\s]+)", text, re.IGNORECASE)
    if match:
        return re.findall(r"\d+", match.group(1))
    return re.findall(r"(?<!\d)\d{7,9}(?!\d)", text)


def _append_source_and_evidence(
    source_records: list[SourceRecord],
    evidence_records: list[EvidenceRecord],
    source: SourceRecord,
    evidence: EvidenceRecord,
) -> None:
    source_records.append(source)
    evidence_records.append(evidence)


def _literature_evidence(
    evidence_id: str,
    source_id: str,
    candidate: dict[str, Any],
    *,
    summary: str,
    polarity: str = "unknown",
) -> EvidenceRecord:
    candidate_id = str(candidate.get("candidate_id") or candidate.get("disease_name") or "")
    disease_name = str(candidate.get("disease_name") or "")
    return {
        "evidence_id": evidence_id,
        "source_id": source_id,
        "candidate_links": [
            {
                "candidate_id": candidate_id,
                "candidate_label": disease_name,
                "relation": "literature_search",
                "polarity": polarity,
                "reason": "Retrieved through a disease-specific PubMed search; interpretation is deferred to verification.",
            }
        ],
        "claim": {
            "subject": candidate_id,
            "predicate": "has_literature_search_result",
            "object": "PubMed",
        },
        "content": {
            "structured": {},
            "summary": summary,
            "relevant_excerpt": "",
        },
    }


def _plan_candidate_research(state: State, candidate: dict[str, Any]) -> dict[str, Any]:
    """Small deterministic policy used by the low-resource search skill.

    It selects actions from observed information gaps instead of imposing a
    disease -> gene -> phenotype sequence.  More actions can be added without
    changing the State or provenance format.
    """
    disease_name = candidate.get("disease_name", "")
    candidate_id = candidate.get("candidate_id", "")
    actions: list[str] = []
    missing: list[str] = []
    if not _candidate_has_source(state, candidate_id, disease_name):
        actions.append("pubmed_search_and_abstract_fetch")
        missing.append("disease_specific_literature")

    return {
        "candidate_id": candidate_id,
        "candidate_name": disease_name,
        "objective": "Collect candidate-specific evidence before verification.",
        "missing_information": missing,
        "selected_actions": actions,
        "need_more_search": bool(actions),
        "reason": "Only actions addressing currently missing evidence are selected.",
    }


def research_candidates_with_togomcp(state: State) -> dict[str, Any]:
    """Collect candidate-specific literature through TogoMCP before reflection."""
    if not state.get("use_togomcp", False):
        return {"togomcp_search_status": "disabled"}

    candidates = _research_candidates(state)
    if not candidates:
        return {"togomcp_search_status": "skipped_no_candidates"}

    client = TogoMCPClient()
    history = list(state.get("togomcp_call_history", []) or [])
    source_records = list(state.get("source_records", []) or [])
    evidence_records = list(state.get("evidence_records", []) or [])
    plans = list(state.get("togomcp_search_plans", []) or [])

    try:
        guide_args: dict[str, Any] = {}
        guide_result = client.call_tool("TogoMCP_Usage_Guide", guide_args)
        _record_call(history, "TogoMCP_Usage_Guide", guide_args, guide_result)
    except TogoMCPError as exc:
        _record_call(history, "TogoMCP_Usage_Guide", {}, error=str(exc))
        return {
            "togomcp_search_status": "error_usage_guide",
            "togomcp_call_history": history,
        }

    executed = 0
    for candidate in candidates:
        plan = _plan_candidate_research(state, candidate)
        plans.append(plan)
        if "pubmed_search_and_abstract_fetch" not in plan["selected_actions"]:
            continue

        disease_name = candidate["disease_name"]
        search_args = {
            "database": "pubmed",
            "query": f'"{disease_name}"',
            "max_results": max(1, TOGOMCP_LITERATURE_LIMIT),
            "sort_by": "relevance",
        }
        try:
            search_result = client.call_tool("ncbi_esearch", search_args)
            _record_call(history, "ncbi_esearch", search_args, search_result)
            executed += 1
        except TogoMCPError as exc:
            _record_call(history, "ncbi_esearch", search_args, error=str(exc))
            continue

        search_source_id = f"S-TogoMCP-PubMedSearch-{len(source_records) + 1:04d}"
        source_records.append(
            {
                "source_id": search_source_id,
                "access_layer": "TogoMCP",
                "tool": "ncbi_esearch",
                "database": "PubMed",
                "source_type": "literature_search",
                "entry_id": "",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/?term={quote_plus(disease_name)}",
                "query": json.dumps(search_args, ensure_ascii=False, sort_keys=True),
                "retrieved_at": search_result.get("started_at", _now()),
                "raw_response": search_result,
            }
        )
        pmids = _extract_pmids(search_result)
        evidence_records.append(
            _literature_evidence(
                f"E-TogoMCP-PubMedSearch-{len(evidence_records) + 1:04d}",
                search_source_id,
                candidate,
                summary=f"PubMed search returned PMIDs: {', '.join(pmids) if pmids else 'none detected'}.",
            )
        )

        if not pmids:
            continue

        fetch_args = {
            "database": "pubmed",
            "ids": pmids,
            "rettype": "abstract",
            "retmode": "text",
        }
        try:
            fetch_result = client.call_tool("ncbi_efetch", fetch_args)
            _record_call(history, "ncbi_efetch", fetch_args, fetch_result)
            executed += 1
        except TogoMCPError as exc:
            _record_call(history, "ncbi_efetch", fetch_args, error=str(exc))
            continue

        fetch_source_id = f"S-TogoMCP-PubMedAbstract-{len(source_records) + 1:04d}"
        source_records.append(
            {
                "source_id": fetch_source_id,
                "access_layer": "TogoMCP",
                "tool": "ncbi_efetch",
                "database": "PubMed",
                "source_type": "literature",
                "entry_id": ",".join(pmids),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmids[0]}/",
                "query": json.dumps(fetch_args, ensure_ascii=False, sort_keys=True),
                "retrieved_at": fetch_result.get("started_at", _now()),
                "raw_response": fetch_result,
            }
        )
        evidence_records.append(
            _literature_evidence(
                f"E-TogoMCP-PubMedAbstract-{len(evidence_records) + 1:04d}",
                fetch_source_id,
                candidate,
                summary=f"Abstract text retrieved for PMID(s): {', '.join(pmids)}; summary is deferred to verification.",
            )
        )

    return {
        "togomcp_search_status": "research_completed",
        "togomcp_call_history": history,
        "togomcp_search_plans": plans,
        "source_records": source_records,
        "evidence_records": evidence_records,
        "research_call_count": executed,
    }
