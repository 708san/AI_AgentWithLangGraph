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


INLINE_EVIDENCE_MARKER = re.compile(r"\[\[EVIDENCE:([^\]]+)\]\]")
PUBLIC_PRIVATE_KEYS = {"prompt"}
IMAGE_PATH_KEYS = {"imagePath", "image_path"}
EXECUTION_ONLY_KEYS = {"llm", "gestalt_matcher_result_loader"}


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
        result = {}
        for key, item in value.items():
            if key in PUBLIC_PRIVATE_KEYS:
                continue
            if key in IMAGE_PATH_KEYS:
                # Local filesystem paths are not portable and can expose
                # machine/user details. Preserve only an explicit HTTP(S) URL.
                if not isinstance(item, str) or not item.strip().lower().startswith(("https://", "http://")):
                    continue
            result[key] = (
                omim_curie(item)
                if key in {"OMIM_id", "omim_id"}
                else _to_public_value(item)
            )
        return result
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
    """Build one structured public reference.

    Internal Evidence IDs and source locations remain available to the
    validation/citation pipeline, but the public contract joins records by
    ``number`` and exposes only the concise Evidence fields documented for
    ``public_response.json``.
    """
    source = _public_source(source)
    evidence_rows = [
        {
            "claim_en": item.get("claim_en"),
            "claim_ja": item.get("claim_ja"),
            "excerpt": item.get("excerpt"),
        }
        for item in evidence_items
    ]

    return {
        "number": number,
        "source_name": source.get("source_name"),
        "title": source.get("title"),
        "identifier": source.get("identifier"),
        "url": source.get("url"),
        "authors": source.get("authors", []),
        "journal": source.get("journal"),
        "publication_year": source.get("publication_year"),
        "evidence": evidence_rows,
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


def _replace_inline_markers(
    text: Any,
    evidence_items: list[dict[str, Any]],
    numbers_by_source: dict[str, int],
) -> tuple[str, set[str]]:
    """Convert internal Evidence markers to public reference numbers."""
    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in evidence_items
        if item.get("evidence_id")
    }
    used_ids: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        evidence_id = match.group(1)
        item = evidence_by_id.get(evidence_id)
        if not item:
            return ""
        source_id = str(item.get("source_id") or "")
        number = numbers_by_source.get(source_id)
        if number is None:
            return ""
        used_ids.add(evidence_id)
        return f"[{number}]"

    return INLINE_EVIDENCE_MARKER.sub(replace, str(text or "").strip()), used_ids


def _cited_evidence_ids(text: Any, evidence_items: list[dict[str, Any]], language: str) -> set[str]:
    """Return only Evidence explicitly cited or quoted verbatim in the text."""
    value = str(text or "")
    by_id = {str(item.get("evidence_id") or ""): item for item in evidence_items}
    cited = {marker_id for marker_id in INLINE_EVIDENCE_MARKER.findall(value) if marker_id in by_id}
    claim_key = "claim_en" if language == "en" else "claim_ja"
    for evidence_id, item in by_id.items():
        claim = str(item.get(claim_key) or "").strip()
        if claim and claim in value:
            cited.add(evidence_id)
    return cited


def _add_citations(text: Any, evidence_items: list[dict[str, Any]], numbers_by_source: dict[str, int], language: str) -> str:
    """Convert explicit Evidence markers; repair omission only on an exact claim."""
    result, inline_ids = _replace_inline_markers(text, evidence_items, numbers_by_source)
    for item in evidence_items:
        number = numbers_by_source.get(str(item.get("source_id") or ""))
        if number is None:
            continue
        marker = f"[{number}]"
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id in inline_ids:
            continue
        claim = str(item.get("claim_en" if language == "en" else "claim_ja") or "").strip()
        if claim and claim in result:
            end = result.find(claim) + len(claim)
            if marker not in result[end:end + len(marker) + 2]:
                result = result[:end].rstrip() + f" {marker}" + result[end:]
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
    references: list[dict[str, Any]],
    number_by_source_id: dict[str, int],
    evidence_by_id: dict[str, dict[str, Any]],
) -> Any:
    if value is None:
        return None
    output = _to_public_value(value)
    if not isinstance(output, dict) or "ans" not in output:
        return output
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
        cited_ids = (
            _cited_evidence_ids(item.get("description_en"), item_evidence, "en")
            | _cited_evidence_ids(item.get("description_ja"), item_evidence, "ja")
        )
        item["description_en"] = _add_citations(item.get("description_en"), item_evidence, number_by_source_id, "en")
        item["description_ja"] = _add_citations(item.get("description_ja"), item_evidence, number_by_source_id, "ja")
        item["description"] = _add_citations(item.get("description"), item_evidence, number_by_source_id, "en")
        numbers = _item_reference_numbers(
            {"evidence_ids": sorted(cited_ids)}, evidence_by_id, number_by_source_id
        )
        used_numbers.update(numbers)
        item["citation_numbers"] = numbers
        # These fields are useful inside State but are not display data.
        item.pop("evidence_ids", None)
        item.pop("reasoning", None)
        item.pop("citations", None)
    # All citation numbers share the top-level Reference namespace. Diagnosis
    # blocks contain only the subset they use, retaining the original numbers.
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
        # Reflection's structured ``evidence_ids`` is the authoritative
        # selection for this candidate.  Unlike free-form prose, it remains
        # available even when the LLM returns a concise fallback sentence
        # without inline Evidence markers.  Keep only evidence that has a
        # public citation number; non-citable/internal evidence must not leak
        # into the external Reference list.
        selected_ids = {
            str(evidence_id)
            for evidence_id in item.get("evidence_ids", []) or []
            if str(evidence_id) in evidence_by_id
        }
        cited_ids = {
            evidence_id
            for evidence_id in selected_ids
            if str(evidence_by_id[evidence_id].get("source_id")) in number_by_source_id
        }
        numbers = _item_reference_numbers(
            {"evidence_ids": sorted(cited_ids)}, evidence_by_id, number_by_source_id
        )
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
    response["references"] = references
    for key in ("tentativeDiagnosis", "finalDiagnosis"):
        response[key] = _project_diagnosis(
            response.get(key), references,
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
    # These are internal audit/evaluation snapshots. They must not alter the
    # established public_response.json contract.
    internal_only = {
        *EXECUTION_ONLY_KEYS,
        "zeroShotRaw",
        "zeroShotValidation",
        "tentativeRaw",
        "tentativeValidation",
    }
    response = _to_public_value({key: value for key, value in state.items() if key not in internal_only})
    if isinstance(response, dict) and response.get("evidence_mode") == "v2":
        return _project_v2(response)
    return response


def _compatibility_diagnosis_item(item: dict[str, Any]) -> dict[str, Any]:
    """Project a v2 diagnosis item toward the established external shape.

    The four legacy-facing fields are kept first and unchanged in meaning.
    The additional fields are canonical v2 metadata, so consumers can adopt
    them incrementally without losing candidate identity or literature state.
    """
    return {
        "rank": item.get("rank"),
        "OMIM_id": item.get("OMIM_id"),
        "description": item.get("description", item.get("description_en", "")),
        "disease_name": item.get("disease_name", item.get("disease_name_en", "")),
        "candidate_id": item.get("candidate_id"),
        "disease_name_en": item.get("disease_name_en"),
        "disease_name_ja": item.get("disease_name_ja"),
        "description_en": item.get("description_en"),
        "description_ja": item.get("description_ja"),
        "literature_status": item.get("literature_status"),
        "citation_numbers": item.get("citation_numbers", []),
    }


def _compatibility_diagnosis_block(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return value
    result = {
        "ans": [
            _compatibility_diagnosis_item(item)
            for item in value.get("ans", []) or []
            if isinstance(item, dict)
        ],
        "reference": value.get("reference"),
        "references": value.get("references", []),
        "validation_status": value.get("validation_status"),
        "validation_errors": value.get("validation_errors", []),
    }
    return result


def _compatibility_reflection_block(
    value: Any,
    references_by_number: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return value
    answers = []
    for item in value.get("ans", []) or []:
        if not isinstance(item, dict):
            continue
        numbers = [
            int(number)
            for number in item.get("citation_numbers", []) or []
            if str(number).isdigit() and int(number) in references_by_number
        ]
        legacy_references = [
            f"[{number}] {_reference_label(references_by_number[number])}"
            for number in numbers
        ]
        answers.append(
            {
                # Established reflection keys.
                "references": legacy_references,
                "Correctness": item.get("correctness"),
                "disease_name": item.get("disease_name", item.get("disease_name_en", "")),
                "PatientSummary": item.get("patient_summary_en", ""),
                "DiagnosisAnalysis": item.get("diagnosis_analysis_en", ""),
                # Canonical v2 fields for accurate bilingual and machine use.
                "candidate_id": item.get("candidate_id"),
                "disease_name_en": item.get("disease_name_en"),
                "disease_name_ja": item.get("disease_name_ja"),
                "patient_summary_en": item.get("patient_summary_en"),
                "patient_summary_ja": item.get("patient_summary_ja"),
                "diagnosis_analysis_en": item.get("diagnosis_analysis_en"),
                "diagnosis_analysis_ja": item.get("diagnosis_analysis_ja"),
                "literature_status": item.get("literature_status"),
                "citation_numbers": numbers,
            }
        )
    return {"ans": answers}


def serialize_external_compatibility(state: dict) -> dict[str, Any]:
    """Create the established node-wrapped external output shape.

    ``serialize_public_state`` remains the canonical v2 response.  This
    projection is intentionally separate: it preserves the old file-level
    wrappers and diagnosis/reflection keys while sourcing all references from
    validated v2 Evidence rather than copying free-form LLM references.
    """
    raw = _to_public_value({
        key: value
        for key, value in state.items()
        if key not in EXECUTION_ONLY_KEYS | {"zeroShotRaw", "zeroShotValidation", "tentativeRaw", "tentativeValidation"}
    })
    if not isinstance(raw, dict) or raw.get("evidence_mode") != "v2":
        return raw

    raw_sources = [
        item for item in (raw.get("sources", []) or []) if isinstance(item, dict)
    ]
    raw_evidence = raw.get("evidence", []) or []
    if not isinstance(raw_evidence, list):
        raw_evidence = []
    all_references, _, _ = _build_references(raw_sources, raw_evidence)
    eligible_source_ids = {
        str(source.get("source_id"))
        for source in raw_sources
        if source.get("source_id")
        and source.get("source_type") == "literature"
        and source.get("citation_allowed")
        and (source.get("url") or source.get("identifier"))
    }
    ordered_source_ids = list(dict.fromkeys(
        str(item.get("source_id"))
        for item in raw_evidence
        if isinstance(item, dict) and str(item.get("source_id")) in eligible_source_ids
    ))
    traceable_references = []
    for reference, source_id in zip(all_references, ordered_source_ids):
        record = dict(reference)
        record["source_id"] = source_id
        record["evidence"] = [
            {
                "evidence_id": item.get("evidence_id"),
                "source_id": item.get("source_id"),
                "candidate_id": item.get("candidate_id"),
                "claim_en": item.get("claim_en"),
                "claim_ja": item.get("claim_ja"),
                "excerpt": item.get("excerpt"),
                "locator": item.get("locator"),
            }
            for item in raw_evidence
            if isinstance(item, dict) and str(item.get("source_id")) == str(source_id)
        ]
        traceable_references.append(record)
    references_by_number = {
        int(item["number"]): item
        for item in all_references
        if item.get("number") is not None
    }
    projected = _project_v2(raw)
    result: dict[str, Any] = {
        "zeroShotResult": {"zeroShotResult": projected.get("zeroShotResult")},
        "pubCaseFinder": {"pubCaseFinder": projected.get("pubCaseFinder", [])},
        "GestaltMatcher": {"GestaltMatcher": projected.get("GestaltMatcher", [])},
        "PhenotypeSearchResult": {
            "phenotypeSearchResult": projected.get("phenotypeSearchResult")
        },
        "mergedDiseaseCandidates": {
            "mergedDiseaseCandidates": projected.get("mergedDiseaseCandidates", [])
        },
        "tentativeDiagnosis": {
            "tentativeDiagnosis": _compatibility_diagnosis_block(
                projected.get("tentativeDiagnosis")
            )
        },
        "reflection": {
            "reflection": _compatibility_reflection_block(
                projected.get("reflection"), references_by_number
            )
        },
        "finalDiagnosis": {
            "finalDiagnosis": _compatibility_diagnosis_block(
                projected.get("finalDiagnosis")
            )
        },
    }
    if "phenoBrain" in projected:
        result["phenoBrain"] = {"phenoBrain": projected.get("phenoBrain")}
    result["sources"] = [_public_source(item) for item in raw_sources]
    result["evidence"] = [item for item in raw_evidence if isinstance(item, dict)]
    result["references"] = traceable_references
    result["external_response"] = {
        key: value for key, value in result.items() if key != "external_response"
    }
    # Keep the established node-wrapped keys while carrying the source ledger
    # needed for audits and downstream joins. Raw article text remains private.
    result["external_response"].update({
        "sources": result["sources"],
        "evidence": result["evidence"],
        "references": traceable_references,
        "validation_status": projected.get("validation_status"),
        "validation_errors": projected.get("validation_errors", []),
    })
    return result


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
