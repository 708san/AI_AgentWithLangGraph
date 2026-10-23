"""Small, deterministic public projection for the diagnosis flow.

The internal State keeps source and Evidence IDs for validation.  The public
response deliberately exposes only the information needed by the frontend:
diagnosis text, citation numbers, and a numbered reference containing title,
evidence excerpts, and URL.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel


def omim_number(value: Any) -> Any:
    if value is None or value == "":
        return value
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else value


def omim_curie(value: Any) -> Any:
    number = omim_number(value)
    if number is None or number == "":
        return number
    return f"OMIM:{number}" if str(number).isdigit() else value


def _to_public_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _to_public_value(value.model_dump())
    if isinstance(value, dict):
        return {
            key: omim_curie(item) if key in {"OMIM_id", "omim_id"} else _to_public_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_public_value(item) for item in value]
    return value


def _source_is_public_citation(source: dict[str, Any]) -> bool:
    return (
        source.get("source_type") == "literature"
        and bool(source.get("citation_allowed"))
        and bool(source.get("url") or source.get("identifier"))
    )


def _public_source(source: dict[str, Any]) -> dict[str, Any]:
    result = dict(source)
    for key in ("raw_content", "search_query", "retrieved_at"):
        result.pop(key, None)
    if not result.get("url"):
        pmid = str(result.get("pmid") or "").removeprefix("PMID:").strip()
        pmcid = str(result.get("pmcid") or "").removeprefix("PMC").strip()
        doi = str(result.get("doi") or "").removeprefix("DOI:").strip()
        if pmid.isdigit():
            result["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        elif pmcid:
            result["url"] = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/"
        elif doi:
            result["url"] = f"https://doi.org/{doi}"
    return result


def _reference_label(reference: dict[str, Any]) -> str:
    source = reference.get("source_name") or "PubMed"
    title = reference.get("title") or "Untitled source"
    url = reference.get("url") or reference.get("identifier") or ""
    return f"{source}: {title}" + (f" {url}" if url else "")


def _reference_record(
    source: dict[str, Any],
    evidence_items: list[dict[str, Any]],
    number: int,
) -> dict[str, Any]:
    """Build one frontend reference; no internal IDs or relation objects."""
    source = _public_source(source)
    return {
        "number": number,
        "source_name": source.get("source_name"),
        "title": source.get("title"),
        "identifier": source.get("identifier"),
        "url": source.get("url"),
        "authors": source.get("authors", []),
        "journal": source.get("journal"),
        "publication_year": source.get("publication_year"),
        "evidence": [
            {
                "claim_en": item.get("claim_en"),
                "claim_ja": item.get("claim_ja"),
                "excerpt": item.get("excerpt"),
            }
            for item in evidence_items
        ],
    }


def _build_references(
    sources: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, dict[str, Any]]]:
    source_by_id = {
        str(source.get("source_id")): source
        for source in sources
        if source.get("source_id")
    }
    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in evidence
        if item.get("evidence_id")
    }
    ordered_source_ids: list[str] = []
    for item in evidence:
        source_id = str(item.get("source_id") or "")
        source = source_by_id.get(source_id)
        if source_id and source and _source_is_public_citation(source) and source_id not in ordered_source_ids:
            ordered_source_ids.append(source_id)

    number_by_source_id = {
        source_id: number for number, source_id in enumerate(ordered_source_ids, 1)
    }
    references = []
    for number, source_id in enumerate(ordered_source_ids, 1):
        source_evidence = [item for item in evidence if item.get("source_id") == source_id]
        references.append(_reference_record(source_by_id[source_id], source_evidence, number))
    for item in evidence:
        source_id = str(item.get("source_id") or "")
        if source_id in number_by_source_id:
            item["reference_number"] = number_by_source_id[source_id]
    return references, number_by_source_id, evidence_by_id


def _item_reference_numbers(
    item: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    number_by_source_id: dict[str, int],
) -> list[int]:
    numbers: list[int] = []
    for evidence_id in item.get("evidence_ids", []) or []:
        evidence_item = evidence_by_id.get(str(evidence_id))
        if not evidence_item:
            continue
        number = number_by_source_id.get(str(evidence_item.get("source_id")))
        if number is not None and number not in numbers:
            numbers.append(number)
    return numbers


def _add_citations(text: Any, evidence_items: list[dict[str, Any]], numbers_by_source: dict[str, int], language: str) -> str:
    """Put [n] after a matching claim, or once at the end if wording differs."""
    result = str(text or "").strip()
    used: set[int] = set()
    for item in evidence_items:
        number = numbers_by_source.get(str(item.get("source_id") or ""))
        if number is None:
            continue
        marker = f"[{number}]"
        claim = str(item.get("claim_en" if language == "en" else "claim_ja") or "").strip()
        if claim and claim in result:
            end = result.find(claim) + len(claim)
            if marker not in result[end:end + len(marker) + 2]:
                result = result[:end].rstrip() + f" {marker}" + result[end:]
        elif number not in used:
            result = f"{result} {marker}".strip()
        used.add(number)
    return result


def _public_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": item.get("candidate_id"),
            "claim_en": item.get("claim_en"),
            "claim_ja": item.get("claim_ja"),
            "excerpt": item.get("excerpt"),
            "reference_number": item.get("reference_number"),
        }
        for item in evidence
    ]


def _project_diagnosis(
    value: Any,
    sources: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    references: list[dict[str, Any]],
    number_by_source_id: dict[str, int],
    evidence_by_id: dict[str, dict[str, Any]],
) -> Any:
    if value is None:
        return None
    output = _to_public_value(value)
    if not isinstance(output, dict) or "ans" not in output:
        return output
    # Citation numbers are local to this diagnosis block.  A source used by
    # another candidate must not make this candidate start at [6].
    selected_ids = {
        str(evidence_id)
        for item in output.get("ans", []) or []
        if isinstance(item, dict)
        for evidence_id in item.get("evidence_ids", []) or []
        if str(evidence_id) in evidence_by_id
    }
    selected_evidence = [
        evidence_by_id[evidence_id]
        for evidence_id in evidence_by_id
        if evidence_id in selected_ids
    ]
    references, number_by_source_id, evidence_by_id = _build_references(
        sources, selected_evidence
    )
    used_numbers: set[int] = set()
    for item in output.get("ans", []) or []:
        if not isinstance(item, dict):
            continue
        item["OMIM_id"] = omim_curie(item.get("OMIM_id"))
        item.setdefault("disease_name", item.get("disease_name_en", ""))
        item.setdefault("description", item.get("description_en", ""))
        item_evidence = [
            evidence_by_id[str(evidence_id)]
            for evidence_id in item.get("evidence_ids", []) or []
            if str(evidence_id) in evidence_by_id
        ]
        numbers = _item_reference_numbers(item, evidence_by_id, number_by_source_id)
        used_numbers.update(numbers)
        item["description_en"] = _add_citations(item.get("description_en"), item_evidence, number_by_source_id, "en")
        item["description_ja"] = _add_citations(item.get("description_ja"), item_evidence, number_by_source_id, "ja")
        item["description"] = _add_citations(item.get("description"), item_evidence, number_by_source_id, "en")
        item["citation_numbers"] = numbers
        # These fields are useful inside State but are not display data.
        item.pop("evidence_ids", None)
        item.pop("reasoning", None)
        item.pop("citations", None)
    used_references = [item for item in references if item["number"] in used_numbers]
    output["references"] = used_references
    output["reference"] = "\n".join(
        f"[{item['number']}] {_reference_label(item)}" for item in used_references
    ) or None
    output.pop("reference_records", None)
    output.pop("evidence", None)
    return output


def _project_reflection(
    value: Any,
    evidence_by_id: dict[str, dict[str, Any]],
    number_by_source_id: dict[str, int],
) -> Any:
    if value is None:
        return None
    output = _to_public_value(value)
    if not isinstance(output, dict):
        return output
    for item in output.get("ans", []) or []:
        if not isinstance(item, dict):
            continue
        item_evidence = [
            evidence_by_id[str(evidence_id)]
            for evidence_id in item.get("evidence_ids", []) or []
            if str(evidence_id) in evidence_by_id
        ]
        numbers = _item_reference_numbers(item, evidence_by_id, number_by_source_id)
        item["diagnosis_analysis_en"] = _add_citations(
            item.get("diagnosis_analysis_en"), item_evidence, number_by_source_id, "en"
        )
        item["diagnosis_analysis_ja"] = _add_citations(
            item.get("diagnosis_analysis_ja"), item_evidence, number_by_source_id, "ja"
        )
        item["citation_numbers"] = numbers
        item.pop("evidence_ids", None)
        item.pop("reasoning", None)
        item.pop("citations", None)
        item.pop("evidence_assessments", None)
    return output


def _project_v2(response: dict[str, Any]) -> dict[str, Any]:
    sources = [
        _public_source(item)
        for item in (_to_public_value(response.get("sources", []) or []) or [])
        if isinstance(item, dict)
    ]
    evidence = _to_public_value(response.get("evidence", []) or [])
    if not isinstance(evidence, list):
        evidence = []
    references, number_by_source_id, evidence_by_id = _build_references(sources, evidence)
    response["sources"] = sources
    response["evidence"] = _public_evidence(evidence)
    for key in ("tentativeDiagnosis", "finalDiagnosis"):
        response[key] = _project_diagnosis(
            response.get(key), sources, evidence, references,
            number_by_source_id, evidence_by_id,
        )
    response["reflection"] = _project_reflection(
        response.get("reflection"), evidence_by_id, number_by_source_id
    )
    candidates = response.get("mergedDiseaseCandidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate["OMIM_id"] = omim_curie(candidate.get("OMIM_id"))
            candidate["tool_rankings"] = [dict(item) for item in candidate.get("tool_supports", []) or []]
    return response


def serialize_public_state(state: dict) -> dict:
    response = _to_public_value({key: value for key, value in state.items() if key != "llm"})
    if isinstance(response, dict) and response.get("evidence_mode") == "v2":
        return _project_v2(response)
    return response


def serialize_node_result(node_name: str, result: Any, state: dict | None = None) -> dict[str, Any]:
    state = state or {}
    raw_result = result.model_dump() if isinstance(result, BaseModel) else result
    if not isinstance(raw_result, dict):
        return {"node": node_name, "result": _to_public_value(raw_result)}
    payload = {key: value for key, value in raw_result.items() if key != "prompt"}
    context = dict(state)
    context.update(payload)
    if context.get("evidence_mode") == "v2":
        projected = _project_v2(_to_public_value(context))
        payload = {
            key: projected[key]
            for key in payload
            if key in projected
        }
        for key in ("sources", "evidence", "reference"):
            if key in projected and key not in payload:
                payload[key] = projected[key]
    return {"node": node_name, "result": _to_public_value(payload)}
