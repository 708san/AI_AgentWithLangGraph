"""Validation and public Reference projection for v2 diagnosis output."""

from __future__ import annotations

import re
from typing import Iterable

from ..state.state_types import (
    DiagnosisOutputV2,
    EvidenceItem,
    MergedDiseaseCandidateV2,
    ReferenceRecord,
    SourceRecord,
)
from .candidate_metadata import canonical_omim_id, names_compatible


INLINE_EVIDENCE_MARKER = re.compile(r"\[\[EVIDENCE:([^\]]+)\]\]")


def validate_v2_identity_fields(
    output: DiagnosisOutputV2,
    candidates: Iterable[MergedDiseaseCandidateV2],
) -> list[str]:
    """Validate candidate identity without replacing model output silently."""
    if output is None:
        return ["Diagnosis generation returned no structured output."]
    candidate_list = list(candidates)
    candidate_map = {candidate.candidate_id: candidate for candidate in candidate_list}
    items = list(output.ans or [])
    errors: list[str] = []
    ids = [item.candidate_id for item in items]
    duplicates = sorted({candidate_id for candidate_id in ids if ids.count(candidate_id) > 1})
    if duplicates:
        errors.append(f"Duplicate candidate_id values: {duplicates}")
    unknown = sorted({candidate_id for candidate_id in ids if candidate_id not in candidate_map})
    for candidate_id in unknown:
        errors.append(f"Unknown candidate_id: {candidate_id}")
    ranks = [item.rank for item in items]
    if ranks != list(range(1, len(ranks) + 1)):
        errors.append("Diagnosis ranks must be contiguous and start at 1.")
    for item in items:
        candidate = candidate_map.get(item.candidate_id)
        if candidate is None:
            continue
        expected_omim = canonical_omim_id(candidate.OMIM_id)
        actual_omim = canonical_omim_id(item.OMIM_id)
        if expected_omim != actual_omim:
            errors.append(
                f"Candidate {item.candidate_id} OMIM mismatch: expected {expected_omim}, got {actual_omim}."
            )
        if not names_compatible(item.disease_name_en, candidate.disease_name):
            errors.append(f"Candidate {item.candidate_id} disease name conflicts with the input candidate.")
    return errors


def build_reference_records(
    output: DiagnosisOutputV2,
    evidence: Iterable[EvidenceItem],
    sources: Iterable[SourceRecord],
) -> list[ReferenceRecord]:
    evidence_list = list(evidence)
    source_map = {source.source_id: source for source in sources}
    requested_ids = {evidence_id for item in output.ans for evidence_id in item.evidence_ids}
    evidence_by_source: dict[str, list[str]] = {}
    for item in evidence_list:
        if item.evidence_id in requested_ids and item.source_id in source_map:
            evidence_by_source.setdefault(item.source_id, []).append(item.evidence_id)
    references = []
    for source_id in sorted(evidence_by_source):
        source = source_map[source_id]
        if not source.citation_allowed or source.source_type != "literature":
            continue
        references.append(
            ReferenceRecord(
                **source.model_dump(),
                evidence_ids=sorted(evidence_by_source[source_id]),
            )
        )
    return references


def validate_v2_output(
    output: DiagnosisOutputV2,
    candidates: Iterable[MergedDiseaseCandidateV2],
    sources: Iterable[SourceRecord],
    evidence: Iterable[EvidenceItem],
) -> DiagnosisOutputV2:
    candidate_list = list(candidates)
    source_list = list(sources)
    evidence_list = list(evidence)
    candidate_ids = {candidate.candidate_id for candidate in candidate_list}
    source_map = {source.source_id: source for source in source_list}
    evidence_map = {item.evidence_id: item for item in evidence_list}
    evidence_by_candidate: dict[str, list[EvidenceItem]] = {}
    for evidence_item in evidence_list:
        evidence_by_candidate.setdefault(evidence_item.candidate_id, []).append(evidence_item)
    errors: list[str] = validate_v2_identity_fields(output, candidate_list)

    ranks = [item.rank for item in output.ans]
    for item in output.ans:
        if item.candidate_id not in candidate_ids:
            errors.append(f"Unknown candidate_id: {item.candidate_id}")
        if item.OMIM_id and not re.fullmatch(r"OMIM:\d+", item.OMIM_id):
            errors.append(f"Non-canonical OMIM_id: {item.OMIM_id}")
        if not item.disease_name_en.strip() or not item.disease_name_ja.strip():
            errors.append(f"Bilingual disease name is incomplete: {item.candidate_id}")
        if not item.description_en.strip() or not item.description_ja.strip():
            errors.append(f"Bilingual description is incomplete: {item.candidate_id}")
        candidate_evidence = evidence_by_candidate.get(item.candidate_id, [])
        if candidate_evidence and not item.evidence_ids:
            errors.append(
                f"Candidate {item.candidate_id} has retrieved Evidence but final diagnosis uses none."
            )
        marker_ids = set(
            INLINE_EVIDENCE_MARKER.findall(item.description_en)
            + INLINE_EVIDENCE_MARKER.findall(item.description_ja)
        )
        for marker_id in marker_ids:
            marker_evidence = evidence_map.get(marker_id)
            if not marker_evidence:
                errors.append(f"Unknown inline Evidence marker: {marker_id}")
            elif marker_evidence.candidate_id != item.candidate_id:
                errors.append(f"Inline Evidence candidate mismatch: {marker_id}")
        for evidence_id in item.evidence_ids:
            evidence_item = evidence_map.get(evidence_id)
            if not evidence_item:
                errors.append(f"Unknown evidence_id: {evidence_id}")
                continue
            if evidence_item.candidate_id != item.candidate_id:
                errors.append(f"Evidence candidate mismatch: {evidence_id}")
            source = source_map.get(evidence_item.source_id)
            if not source:
                errors.append(f"Evidence source is missing: {evidence_item.source_id}")
            elif source.source_type != "literature" or not source.citation_allowed:
                errors.append(f"Non-citable source used as diagnosis evidence: {source.source_id}")
        candidate_evidence = [
            evidence_map[evidence_id]
            for evidence_id in item.evidence_ids
            if evidence_id in evidence_map
        ]
        has_literature_support = any(
            evidence_item.support == "supports"
            and source_map.get(evidence_item.source_id)
            and source_map[evidence_item.source_id].source_type == "literature"
            for evidence_item in candidate_evidence
        )
        if item.literature_status == "supported" and not has_literature_support:
            errors.append(f"Supported diagnosis has no literature Evidence: {item.candidate_id}")
        if item.literature_status in {"insufficient", "not_retrieved"} and not item.description_en.strip():
            errors.append(f"Evidence insufficiency is not explained: {item.candidate_id}")

    output.evidence = evidence_list
    output.references = build_reference_records(output, evidence_list, source_list)
    output.validation_errors = list(dict.fromkeys(errors))
    has_unresolved_literature_status = any(
        item.literature_status != "supported" for item in output.ans
    )
    output.validation_status = (
        "failed" if errors else "partial" if has_unresolved_literature_status else "passed"
    )
    return output
