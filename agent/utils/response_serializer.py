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
from ..tools.candidate_metadata import (
    known_disease_name,
    normalize_disease_name,
    parse_omim_id,
    resolve_known_disease_identity,
)


INLINE_EVIDENCE_MARKER = re.compile(r"\[\[EVIDENCE:([^\]]+)\]\]")
EVIDENCE_ID_PATTERN = re.compile(r"\bEVD-[A-Za-z0-9_-]+\b")
EVIDENCE_ID_GROUP_PATTERN = re.compile(
    r"(?:[\[［【]\s*)?EVD-[A-Za-z0-9_-]+"
    r"(?:\s*[,;，；、]\s*EVD-[A-Za-z0-9_-]+)*"
    r"(?:\s*[\]］】])?"
)
REFERENCE_NUMBER_CLUSTER = r"\[\d+\](?:(?:\s*[,;，；、]\s*|\s*)\[\d+\])+"
WRAPPED_REFERENCE_NUMBER_CLUSTER_PATTERN = re.compile(
    rf"(?:［|【|\[)\s*(?P<cluster>{REFERENCE_NUMBER_CLUSTER})\s*(?:］|】|\])"
)
ADJACENT_REFERENCE_GROUP_PATTERN = re.compile(
    REFERENCE_NUMBER_CLUSTER
)
PUBLIC_PRIVATE_KEYS = {"prompt"}
IMAGE_PATH_KEYS = {"imagePath", "image_path"}
EXECUTION_ONLY_KEYS = {"llm", "zero_shot_llm", "gestalt_matcher_result_loader"}


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


def _bibliographic_reference(reference: dict[str, Any]) -> dict[str, Any]:
    """Project a Reference without embedding Evidence rows."""
    return {
        "number": reference.get("number"),
        "source_name": reference.get("source_name"),
        "title": reference.get("title"),
        "identifier": reference.get("identifier"),
        "url": reference.get("url"),
        "authors": reference.get("authors", []),
        "journal": reference.get("journal"),
        "publication_year": reference.get("publication_year"),
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


def _traceable_evidence(
    evidence_ids: list[str],
    evidence_by_id: dict[str, dict[str, Any]],
    number_by_source_id: dict[str, int],
) -> list[dict[str, Any]]:
    """Project the Evidence used by one result with its public Reference number."""
    rows = []
    for evidence_id in evidence_ids:
        item = evidence_by_id.get(str(evidence_id))
        if not item:
            continue
        source_id = str(item.get("source_id") or "")
        reference_number = number_by_source_id.get(source_id)
        if reference_number is None:
            continue
        rows.append({
            "evidence_id": item.get("evidence_id"),
            "source_id": item.get("source_id"),
            "candidate_id": item.get("candidate_id"),
            "claim_en": item.get("claim_en"),
            "claim_ja": item.get("claim_ja"),
            "excerpt": item.get("excerpt"),
            "support": item.get("support"),
            "confidence": item.get("confidence"),
            "source_section": item.get("source_section"),
            "source_start": item.get("source_start"),
            "source_end": item.get("source_end"),
            "reference_number": reference_number,
        })
    return rows


def _reference_details(
    references: list[dict[str, Any]],
    evidence_ids: list[str],
    evidence_by_id: dict[str, dict[str, Any]],
    number_by_source_id: dict[str, int],
) -> list[dict[str, Any]]:
    """Return only References and Evidence used by one projected result."""
    evidence_rows = _traceable_evidence(evidence_ids, evidence_by_id, number_by_source_id)
    rows_by_number: dict[int, list[dict[str, Any]]] = {}
    for row in evidence_rows:
        rows_by_number.setdefault(int(row["reference_number"]), []).append(row)
    details = []
    for reference in references:
        number = reference.get("number")
        if number not in rows_by_number:
            continue
        detail = dict(reference)
        detail["evidence"] = rows_by_number[number]
        details.append(detail)
    return details


def _reference_subset(
    references: list[dict[str, Any]],
    evidence_ids: list[str],
    evidence_by_id: dict[str, dict[str, Any]],
    number_by_source_id: dict[str, int],
) -> list[dict[str, Any]]:
    """Filter a public Reference list to the claims used by a result."""
    details = _reference_details(
        references, evidence_ids, evidence_by_id, number_by_source_id
    )
    subset = []
    for reference in details:
        record = dict(reference)
        record["evidence"] = [
            {
                "claim_en": item.get("claim_en"),
                "claim_ja": item.get("claim_ja"),
                "excerpt": item.get("excerpt"),
            }
            for item in reference.get("evidence", [])
        ]
        subset.append(record)
    return subset


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
    return _deduplicate_reference_clusters(result)


def _deduplicate_reference_clusters(text: Any) -> str:
    """Collapse repeated reference numbers inside one citation-only cluster."""
    result = str(text or "")

    def unique_numbers(match: re.Match[str]) -> str:
        numbers = sorted({
            int(number) for number in re.findall(r"\[(\d+)\]", match.group("cluster"))
        })
        return "".join(f"[{number}]" for number in numbers)

    result = WRAPPED_REFERENCE_NUMBER_CLUSTER_PATTERN.sub(unique_numbers, result)
    result = ADJACENT_REFERENCE_GROUP_PATTERN.sub(
        lambda match: "".join(
            f"[{number}]"
            for number in sorted({
                int(value) for value in re.findall(r"\[(\d+)\]", match.group(0))
            })
        ),
        result,
    )
    return result


def _externalize_evidence_ids(
    text: Any,
    evidence_by_id: dict[str, dict[str, Any]],
    number_by_source_id: dict[str, int],
) -> tuple[str, list[int]]:
    """Render model-emitted Evidence IDs as deduplicated public References."""
    used_numbers: list[int] = []

    def replace_evidence_group(match: re.Match[str]) -> str:
        numbers: list[int] = []
        for evidence_id in EVIDENCE_ID_PATTERN.findall(match.group(0)):
            evidence_item = evidence_by_id.get(evidence_id)
            if not evidence_item:
                continue
            number = number_by_source_id.get(str(evidence_item.get("source_id") or ""))
            if number is not None:
                if number not in numbers:
                    numbers.append(number)
                if number not in used_numbers:
                    used_numbers.append(number)
        return "".join(f"[{number}]" for number in numbers)

    result = EVIDENCE_ID_GROUP_PATTERN.sub(replace_evidence_group, str(text or ""))
    result = _deduplicate_reference_clusters(result)
    return result, used_numbers


def _format_external_citation_fields(
    block: Any,
    fields: tuple[str, ...],
    evidence_by_id: dict[str, dict[str, Any]],
    number_by_source_id: dict[str, int],
) -> None:
    """Normalize citations only in the legacy external-response projection."""
    if not isinstance(block, dict):
        return
    for item in block.get("ans", []) or []:
        if not isinstance(item, dict):
            continue
        numbers = list(dict.fromkeys(
            int(number)
            for number in item.get("citation_numbers", []) or []
            if str(number).isdigit()
        ))
        for field in fields:
            item[field], cited_numbers = _externalize_evidence_ids(
                item.get(field), evidence_by_id, number_by_source_id
            )
            numbers.extend(number for number in cited_numbers if number not in numbers)
        numbers = sorted(set(numbers))
        item["citation_numbers"] = numbers
        item["reference_numbers"] = numbers


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
    used_evidence_ids: list[str] = []
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
        # The structured evidence_ids field is authoritative for the
        # candidate-to-literature relationship. Inline markers are still
        # converted in the descriptions, but must not be the only way to
        # expose the Evidence used by this diagnosis.
        selected_ids = [
            str(evidence_id)
            for evidence_id in item.get("evidence_ids", []) or []
            if str(evidence_id) in evidence_by_id
        ]
        used_evidence_ids.extend(selected_ids)
        item["description_en"] = _add_citations(item.get("description_en"), item_evidence, number_by_source_id, "en")
        item["description_ja"] = _add_citations(item.get("description_ja"), item_evidence, number_by_source_id, "ja")
        item["description"] = _add_citations(item.get("description"), item_evidence, number_by_source_id, "en")
        numbers = _item_reference_numbers(
            {"evidence_ids": selected_ids}, evidence_by_id, number_by_source_id
        )
        used_numbers.update(numbers)
        item["citation_numbers"] = numbers
        item["reference_numbers"] = numbers
        item["evidence"] = _traceable_evidence(
            selected_ids, evidence_by_id, number_by_source_id
        )
        item["reference_details"] = _reference_details(
            references, selected_ids, evidence_by_id, number_by_source_id
        )
        # These fields are useful inside State but are not display data.
        item.pop("evidence_ids", None)
        item.pop("reasoning", None)
        item.pop("citations", None)
    # All citation numbers share the top-level Reference namespace. Diagnosis
    # blocks contain only the subset they use, retaining the original numbers.
    used_references = _reference_subset(
        references,
        list(dict.fromkeys(used_evidence_ids)),
        evidence_by_id,
        number_by_source_id,
    )
    output["references"] = used_references
    output["reference_details"] = _reference_details(
        references,
        list(dict.fromkeys(used_evidence_ids)),
        evidence_by_id,
        number_by_source_id,
    )
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
    references: list[dict[str, Any]] | None = None,
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
        item["reference_numbers"] = numbers
        item["evidence"] = _traceable_evidence(
            sorted(cited_ids), evidence_by_id, number_by_source_id
        )
        item["reference_details"] = _reference_details(
            references or [],
            sorted(cited_ids),
            evidence_by_id,
            number_by_source_id,
        )
        item.pop("evidence_ids", None)
        item.pop("reasoning", None)
        item.pop("citations", None)
        item.pop("evidence_assessments", None)
    return output


def _normalize_external_zero_shot(value: Any, validation: Any = None) -> Any:
    """Keep valid named candidates and normalize identities/ranks at the API edge."""
    if not isinstance(value, dict) or not isinstance(value.get("ans"), list):
        return value

    audit_by_display_rank = {}
    for validation_row in validation or []:
        if not isinstance(validation_row, dict) or validation_row.get("display_rank") is None:
            continue
        try:
            display_rank = int(validation_row["display_rank"])
        except (TypeError, ValueError):
            continue
        audit_by_display_rank[display_rank] = validation_row
    eligible = []
    seen_identity_keys: set[str] = set()
    for input_index, item in enumerate(value["ans"]):
        if not isinstance(item, dict):
            continue
        try:
            model_order = int(item.get("rank"))
        except (TypeError, ValueError):
            continue
        if model_order < 1:
            continue

        audit_row = audit_by_display_rank.get(model_order, {})
        mapping_method = audit_row.get("mapping_method")
        omim_id = parse_omim_id(item.get("OMIM_id") or item.get("omim_id"))
        disease_name = str(item.get("disease_name") or "")
        if not disease_name.strip():
            continue

        name_id, name_label, _ = resolve_known_disease_identity(disease_name)
        if name_id and name_label:
            canonical_id = name_id
        elif omim_id:
            canonical_id = omim_id
        else:
            canonical_id = None
        # Keep the model's disease wording; the OMIM ID carries the resolved
        # canonical identity, while the original wording remains auditable.
        display_name = disease_name or (known_disease_name(canonical_id) if canonical_id else None)
        if not display_name:
            continue

        if mapping_method == "faiss_nearest_no_threshold":
            identity_key = f"NAME:{normalize_disease_name(disease_name)}"
        else:
            identity_key = canonical_id or f"NAME:{normalize_disease_name(display_name)}"
        if identity_key == "NAME:" or identity_key in seen_identity_keys:
            continue
        seen_identity_keys.add(identity_key)

        eligible.append((model_order, input_index, item, canonical_id, display_name))

    eligible.sort(key=lambda row: (row[0], row[1]))
    output = dict(value)
    output["ans"] = []
    for display_rank, (source_display_rank, _, item, omim_id, canonical_name) in enumerate(eligible, 1):
        row = dict(item)
        row["disease_name"] = canonical_name
        row["OMIM_id"] = omim_id
        source_audit = audit_by_display_rank.get(source_display_rank, {})
        row["model_OMIM_id"] = source_audit.get(
            "OMIM_id", item.get("model_OMIM_id", item.get("OMIM_id", item.get("omim_id")))
        )
        row.pop("omim_id", None)
        row.pop("model_rank", None)
        row["rank"] = display_rank
        output["ans"].append(row)
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
        response.get("reflection"), evidence_by_id, number_by_source_id, references
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
        "literature_basis_en": item.get("literature_basis_en", ""),
        "literature_basis_ja": item.get("literature_basis_ja", ""),
        "tool_considerations_en": item.get("tool_considerations_en", ""),
        "tool_considerations_ja": item.get("tool_considerations_ja", ""),
        "model_inference_en": item.get("model_inference_en", ""),
        "model_inference_ja": item.get("model_inference_ja", ""),
        "uncertainty_en": item.get("uncertainty_en", ""),
        "uncertainty_ja": item.get("uncertainty_ja", ""),
        "literature_status": item.get("literature_status"),
        "citation_numbers": item.get("citation_numbers", []),
        "reference_numbers": item.get("reference_numbers", item.get("citation_numbers", [])),
        "evidence": item.get("evidence", []),
        "reference_details": item.get("reference_details", []),
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
        "reference_details": value.get("reference_details", []),
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
        # Reflection needs the same bibliographic projection as Final, but its
        # public References deliberately do not carry nested Evidence. The
        # reflection answer may still expose its separate ``evidence`` field;
        # it is not embedded in or joined to these Reference records.
        reflection_references = [
            _bibliographic_reference(references_by_number[number])
            for number in numbers
        ]
        answers.append(
            {
                # Established reflection keys.
                "references": reflection_references,
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
                "literature_basis_en": item.get("literature_basis_en", ""),
                "literature_basis_ja": item.get("literature_basis_ja", ""),
                "tool_considerations_en": item.get("tool_considerations_en", ""),
                "tool_considerations_ja": item.get("tool_considerations_ja", ""),
                "model_inference_en": item.get("model_inference_en", ""),
                "model_inference_ja": item.get("model_inference_ja", ""),
                "uncertainty_en": item.get("uncertainty_en", ""),
                "uncertainty_ja": item.get("uncertainty_ja", ""),
                "literature_status": item.get("literature_status"),
                "citation_numbers": numbers,
                "reference_numbers": numbers,
                "evidence": item.get("evidence", []),
                "reference_details": reflection_references,
            }
        )
    used_numbers = []
    for item in answers:
        for number in item.get("reference_numbers", []):
            if number not in used_numbers:
                used_numbers.append(number)
    return {
        "ans": answers,
        "references": [
            _bibliographic_reference(references_by_number[number])
            for number in used_numbers
            if number in references_by_number
        ],
    }


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
    (
        all_references,
        external_number_by_source_id,
        external_evidence_by_id,
    ) = _build_references(raw_sources, raw_evidence)
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
    projected["zeroShotResult"] = _normalize_external_zero_shot(
        projected.get("zeroShotResult"),
        _to_public_value(state.get("zeroShotValidation", []) or []),
    )
    _format_external_citation_fields(
        projected.get("finalDiagnosis"),
        ("description_en", "description_ja", "description"),
        external_evidence_by_id,
        external_number_by_source_id,
    )
    _format_external_citation_fields(
        projected.get("reflection"),
        ("diagnosis_analysis_en", "diagnosis_analysis_ja"),
        external_evidence_by_id,
        external_number_by_source_id,
    )
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
