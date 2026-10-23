"""Structured, excerpt-validated evidence extraction."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

from langchain.schema import HumanMessage

from ..state.state_types import (
    EvidenceExtractionOutput,
    EvidenceItem,
    MergedDiseaseCandidateV2,
    SourceRecord,
)
from .literature_search import matches_candidate_identity


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _contains_excerpt(raw_content: str, excerpt: str) -> bool:
    source = _normalized_text(raw_content)
    target = _normalized_text(excerpt)
    return bool(target) and target in source


def _exact_excerpt_span(raw_content: str, excerpt: str) -> tuple[int, int] | None:
    start = raw_content.casefold().find(excerpt.strip().casefold())
    if start < 0:
        return None
    return start, start + len(excerpt.strip())


CHUNK_MAX_CHARS = int(os.getenv("EVIDENCE_CHUNK_MAX_CHARS", "3500"))
MAX_CHUNKS_PER_SOURCE = int(os.getenv("EVIDENCE_MAX_CHUNKS_PER_SOURCE", "2"))
MAX_CHUNKS_PER_CANDIDATE = int(os.getenv("EVIDENCE_MAX_CHUNKS_PER_CANDIDATE", "10"))


@dataclass(frozen=True)
class _SourceChunk:
    chunk_id: str
    source_id: str
    title: str
    identifier: str
    section: str
    text: str
    start: int
    end: int


def _section_name(block: str, current: str) -> str:
    first_line = block.strip().splitlines()[0].strip() if block.strip() else ""
    known = (
        "abstract", "introduction", "background", "case", "clinical", "methods",
        "materials", "results", "genotype", "molecular", "phenotype", "discussion",
        "conclusion", "limitations", "references",
    )
    normalized = re.sub(r"[^a-z ]", "", first_line.casefold()).strip()
    if len(first_line) <= 100 and any(normalized.startswith(value) for value in known):
        return first_line
    return current


def _source_chunks(source: SourceRecord) -> list[_SourceChunk]:
    """Split a source without losing its original offsets or section order."""
    text = source.raw_content or ""
    if not text.strip():
        return []
    # Do not use ``\s*`` here: it can consume newlines and merge separate
    # section blocks into one paragraph.
    blocks = list(re.finditer(r"\S.*?(?=\n[ \t]*\n|\Z)", text, flags=re.DOTALL))
    if not blocks:
        blocks = [re.match(r"[\s\S]*", text)]

    units: list[tuple[int, int, str]] = []
    section = "Unknown"
    for match in blocks:
        if match is None:
            continue
        start, end = match.span()
        block = text[start:end]
        section = _section_name(block, section)
        if end - start <= CHUNK_MAX_CHARS:
            units.append((start, end, section))
            continue
        for offset in range(start, end, CHUNK_MAX_CHARS):
            units.append((offset, min(offset + CHUNK_MAX_CHARS, end), section))

    chunks: list[_SourceChunk] = []
    current_start: int | None = None
    current_end: int | None = None
    current_section = "Unknown"

    def flush() -> None:
        nonlocal current_start, current_end, current_section
        if current_start is None or current_end is None:
            return
        chunks.append(_SourceChunk(
            chunk_id=f"{source.source_id}#chunk-{len(chunks) + 1:02d}",
            source_id=source.source_id,
            title=source.title,
            identifier=source.identifier or "N/A",
            section=current_section,
            text=text[current_start:current_end].strip(),
            start=current_start,
            end=current_end,
        ))
        current_start = None
        current_end = None
        current_section = "Unknown"

    for start, end, unit_section in units:
        if current_start is None:
            current_start, current_end, current_section = start, end, unit_section
            continue
        if unit_section != current_section or end - current_start > CHUNK_MAX_CHARS:
            flush()
            current_start, current_end, current_section = start, end, unit_section
        else:
            current_end = end
    flush()
    return chunks


def _chunk_score(chunk: _SourceChunk, candidate: MergedDiseaseCandidateV2, hpo: list[str]) -> int:
    text = _normalized_text(chunk.text)
    terms = [
        str(candidate.disease_name or ""),
        str(candidate.OMIM_id or ""),
        *(candidate.gene_symbols or []),
        *(candidate.synonyms or []),
        *(candidate.phenotype_labels or []),
        *hpo,
    ]
    score = 0
    for term in terms:
        normalized = _normalized_text(term)
        if normalized and len(normalized) >= 4 and normalized in text:
            score += 4 if normalized in {_normalized_text(candidate.disease_name), _normalized_text(candidate.OMIM_id)} else 2
    category_terms = (
        "inheritance", "mutation", "variant", "gene", "phenotype", "clinical",
        "facial", "developmental", "growth", "limb", "absence", "without", "limitation",
    )
    score += sum(1 for term in category_terms if term in text)
    section = chunk.section.casefold()
    if any(term in section for term in ("abstract", "clinical", "genotype", "molecular", "results")):
        score += 3
    return score


def _select_relevant_chunks(
    candidate: MergedDiseaseCandidateV2,
    sources: list[SourceRecord],
    hpo: list[str],
) -> list[_SourceChunk]:
    """Keep source coverage while limiting the prompt size.

    The first chunk of every source is always retained so an abstract or
    identity statement is not lost.  A second chunk is selected by relevance,
    which usually reaches Results/Genotype sections in long PMC articles.
    """
    selected: list[_SourceChunk] = []
    all_scored: list[tuple[int, _SourceChunk]] = []
    for source in sources:
        chunks = _source_chunks(source)
        if not chunks:
            continue
        selected.extend(chunks[:1])
        ranked = sorted(
            ((_chunk_score(chunk, candidate, hpo), chunk) for chunk in chunks[1:]),
            key=lambda value: (-value[0], value[1].start),
        )
        selected.extend(chunk for _, chunk in ranked[: max(0, MAX_CHUNKS_PER_SOURCE - 1)])
        all_scored.extend((_chunk_score(chunk, candidate, hpo), chunk) for chunk in chunks)

    unique = {chunk.chunk_id: chunk for chunk in selected}
    for _, chunk in sorted(all_scored, key=lambda value: (-value[0], value[1].start)):
        if len(unique) >= MAX_CHUNKS_PER_CANDIDATE:
            break
        unique.setdefault(chunk.chunk_id, chunk)
    return sorted(unique.values(), key=lambda chunk: (chunk.source_id, chunk.start))


def _evidence_id(candidate_id: str, source_id: str, evidence_type: str, excerpt: str) -> str:
    digest = hashlib.sha1(
        f"{candidate_id}|{source_id}|{evidence_type}|{_normalized_text(excerpt)}".encode()
    ).hexdigest()[:10]
    return f"EVD-{digest}"


def _candidate_prompt(
    candidate: MergedDiseaseCandidateV2,
    chunks: list[_SourceChunk],
    hpo: list[str],
    absent_hpo: list[str],
) -> str:
    source_blocks = []
    for chunk in chunks:
        source_blocks.append(
            "\n".join(
                [
                    f"SOURCE_ID: {chunk.source_id}",
                    f"CHUNK_ID: {chunk.chunk_id}",
                    f"TITLE: {chunk.title}",
                    f"IDENTIFIER: {chunk.identifier}",
                    f"SECTION: {chunk.section}",
                    f"CONTENT: {chunk.text}",
                ]
            )
        )
    return f"""You are extracting auditable clinical-genetics evidence for one candidate disease.
Do not summarize the papers. Extract individual claims that are explicitly supported by an exact excerpt.
Every excerpt must be copied verbatim from the supplied CONTENT, without quotation marks added.
Return English and Japanese claims with the same meaning. Do not invent a gene, inheritance pattern, phenotype, mechanism, or negative finding.
Use only the supplied SOURCE_ID and CHUNK_ID values. If a required fact is not present in the supplied chunks, do not infer it.
Try to cover disease identity, gene/inheritance, core phenotype, distinctive phenotype, patient overlap, and important limitations when the source provides them.
Do not create a generic phenotype item when a more specific phenotype or genetic statement is present in the same source.
For every item, return the SOURCE_ID, CHUNK_ID, and SECTION containing the excerpt.
Extract only literature-grounded disease claims and their verbatim excerpts. The later diagnosis and reflection
steps will explain how the patient's findings relate to these claims; do not create a separate relation object.

CANDIDATE_ID: {candidate.candidate_id}
DISEASE: {candidate.disease_name}
OMIM: {candidate.OMIM_id or 'N/A'}
GENES: {', '.join(candidate.gene_symbols) or 'N/A'}
PATIENT_PRESENT_HPO: {', '.join(hpo) or 'N/A'}
PATIENT_ABSENT_HPO: {', '.join(absent_hpo) or 'N/A'}

Allowed evidence types: disease_identity, gene_and_inheritance, core_phenotype, distinctive_phenotype,
negative_finding, phenotype_overlap, phenotype_mismatch, disease_mechanism.

{chr(10).join(source_blocks)}"""


def extract_evidence_for_candidates(
    candidates: Iterable[MergedDiseaseCandidateV2],
    sources: Iterable[SourceRecord],
    hpo_dict: dict[str, str] | None,
    absent_hpo_dict: dict[str, str] | None,
    llm: Any,
) -> list[EvidenceItem]:
    if not llm:
        return []
    source_list = list(sources)
    present_hpo = list((hpo_dict or {}).values())
    absent_hpo = list((absent_hpo_dict or {}).values())
    existing: dict[str, EvidenceItem] = {}
    for candidate in candidates:
        candidate_sources = [
            source for source in source_list if candidate.candidate_id in source.candidate_ids
            and (
                source.source_type != "literature"
                or matches_candidate_identity(
                    {"title": source.title, "abstract": source.raw_content}, candidate
                )
            )
        ]
        if not candidate_sources:
            continue
        selected_chunks = _select_relevant_chunks(candidate, candidate_sources, present_hpo)
        if not selected_chunks:
            continue
        try:
            structured = llm.get_structured_llm(EvidenceExtractionOutput)
            result = structured.invoke(
                [HumanMessage(content=_candidate_prompt(candidate, selected_chunks, present_hpo, absent_hpo))]
            )
        except Exception as exc:
            print(f"[Evidence] extraction failed for {candidate.candidate_id}: {exc}")
            continue
        drafts = getattr(result, "evidence", []) or []
        source_map = {source.source_id: source for source in candidate_sources}
        chunks_by_source = {}
        for chunk in selected_chunks:
            chunks_by_source.setdefault(chunk.source_id, []).append(chunk)
        for draft in drafts:
            source_id = _value(draft, "source_id")
            source = source_map.get(source_id)
            excerpt = str(_value(draft, "excerpt", "") or "").strip()
            claim_en = str(_value(draft, "claim_en", "") or "").strip()
            claim_ja = str(_value(draft, "claim_ja", "") or "").strip()
            if not source or not excerpt or not claim_en or not claim_ja:
                continue
            if not _contains_excerpt(source.raw_content, excerpt):
                print(f"[Evidence] rejected non-verbatim excerpt from {source_id}")
                continue
            source_chunks = chunks_by_source.get(source_id, [])
            draft_chunk_id = str(_value(draft, "source_chunk_id", "") or "")
            chunk = next((item for item in source_chunks if item.chunk_id == draft_chunk_id), None)
            if chunk is None:
                chunk = next(
                    (item for item in source_chunks if _contains_excerpt(item.text, excerpt)),
                    None,
                )
            source_section = _value(draft, "source_section") or (chunk.section if chunk else None)
            excerpt_span = _exact_excerpt_span(source.raw_content, excerpt)
            source_start, source_end = excerpt_span if excerpt_span else (None, None)
            evidence_id = _evidence_id(
                candidate.candidate_id,
                source_id,
                str(_value(draft, "evidence_type")),
                excerpt,
            )
            item = EvidenceItem(
                evidence_id=evidence_id,
                candidate_id=candidate.candidate_id,
                source_id=source_id,
                evidence_type=_value(draft, "evidence_type"),
                claim_en=claim_en,
                claim_ja=claim_ja,
                excerpt=excerpt,
                support=_value(draft, "support", "unknown"),
                confidence=_value(draft, "confidence", "low"),
                source_chunk_id=chunk.chunk_id if chunk else None,
                source_section=source_section,
                source_start=source_start,
                source_end=source_end,
            )
            existing[item.evidence_id] = item
    return list(existing.values())
