"""Convert legacy candidate dictionaries into evidence-pipeline candidates."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable
from functools import lru_cache

from ..state.state_types import (
    CandidateSupport,
    MergedDiseaseCandidate,
    MergedDiseaseCandidateV2,
)


_OMIM_RE = re.compile(r"(?:OMIM\s*[:#]?\s*)?(\d{4,7})", re.IGNORECASE)
_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "DataForDiseaseSearchFromHPO" / "omim_database.json"
_MAPPING_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "DataForOmimMapping" / "DataForOmimMapping.json"


def canonical_omim_id(value: Any) -> str | None:
    if value is None or value == "":
        return None
    match = _OMIM_RE.search(str(value))
    return f"OMIM:{match.group(1)}" if match else None


def parse_omim_id(value: Any) -> str | None:
    """Parse only a complete OMIM identifier, never digits embedded in prose."""
    if value is None or isinstance(value, bool):
        return None
    match = re.fullmatch(
        r"\s*(?:OMIM\s*[:#]?\s*)?(\d{4,7})\s*",
        str(value),
        flags=re.IGNORECASE,
    )
    return f"OMIM:{match.group(1)}" if match else None


def candidate_id_for(disease_name: str, omim_id: Any = None) -> str:
    canonical = canonical_omim_id(omim_id)
    if canonical:
        return canonical
    normalized = re.sub(r"[^a-z0-9]+", "-", (disease_name or "unknown").lower()).strip("-")
    return f"DISEASE:{normalized or 'unknown'}"


def normalize_disease_name(value: Any) -> str:
    """Normalize only presentation details; do not infer a new disease."""
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


@lru_cache(maxsize=1)
def _load_omim_records() -> dict[str, dict[str, Any]]:
    try:
        with _DATA_PATH.open(encoding="utf-8") as handle:
            records = json.load(handle)
    except (OSError, ValueError):
        return {}
    return {
        str(canonical_omim_id(record.get("OMIM_id"))): record
        for record in records
        if canonical_omim_id(record.get("OMIM_id"))
    }


def _metadata_for(omim_id: str | None, disease_name: str) -> dict[str, Any]:
    record = _load_omim_records().get(omim_id or "", {})
    synonyms = [item.strip() for item in str(record.get("synonym") or "").split(",") if item.strip()]
    phenotype_labels = [item.strip() for item in (record.get("phenotype") or []) if str(item).strip()]
    return {
        # A known OMIM ID is the canonical identity; use its local title even
        # when an upstream tool supplied a different label.
        "disease_name": record.get("disease_name") or disease_name or "Unknown disease",
        "synonyms": list(dict.fromkeys(synonyms)),
        "phenotype_labels": list(dict.fromkeys(phenotype_labels)),
        "gene_symbols": [],
    }


def known_disease_name(omim_id: Any) -> str | None:
    """Return a locally known canonical label for an OMIM disease ID."""
    canonical = canonical_omim_id(omim_id)
    if not canonical:
        return None
    record = _load_omim_records().get(canonical)
    if record and record.get("disease_name"):
        return str(record["disease_name"])
    return None


def known_disease_aliases(omim_id: Any) -> list[str]:
    canonical = canonical_omim_id(omim_id)
    record = _load_omim_records().get(canonical or "", {})
    return [
        item.strip()
        for item in str(record.get("synonym") or "").split(",")
        if item.strip()
    ]


@lru_cache(maxsize=1)
def _known_disease_name_index() -> dict[str, tuple[str, ...]]:
    """Index local OMIM names and FAISS labels, including semicolon variants."""
    matches: dict[str, set[str]] = {}

    def add_name(name: Any, omim_id: Any) -> None:
        canonical_id = canonical_omim_id(omim_id)
        if not canonical_id:
            return
        text = str(name or "").strip()
        if not text:
            return
        # Preserve the combined label and index each semicolon-delimited part.
        # FAISS labels commonly use "OFFICIAL DISEASE NAME; ABBREVIATION".
        for variant in [text, *(part.strip() for part in text.split(";") if part.strip())]:
            normalized = normalize_disease_name(variant)
            if normalized:
                matches.setdefault(normalized, set()).add(canonical_id)

    for omim_id, record in _load_omim_records().items():
        add_name(record.get("disease_name"), omim_id)
        for comma_part in str(record.get("synonym") or "").split(","):
            if comma_part.strip():
                add_name(comma_part, omim_id)

    # Also index the exact labels used to build the FAISS mapping. In many
    # entries the official title is before ';' and the acronym is after it.
    try:
        with _MAPPING_INDEX_PATH.open(encoding="utf-8") as handle:
            mapping_index = json.load(handle)
    except (OSError, ValueError):
        mapping_index = {}
    if isinstance(mapping_index, dict):
        for label, omim_id in zip(
            mapping_index.get("labels", []),
            mapping_index.get("omim_ids", []),
        ):
            add_name(label, omim_id)

    return {name: tuple(sorted(ids)) for name, ids in matches.items()}


def resolve_known_disease_identity(value: Any) -> tuple[str | None, str | None, bool]:
    """Resolve a name only when it exactly identifies one local OMIM record.

    The final boolean indicates that the exact name is ambiguous across
    multiple OMIM identities. Similarity search is intentionally not used as
    an identity authority.
    """
    text = str(value or "")
    normalized = normalize_disease_name(text)
    if not normalized:
        return None, None, False

    name_index = _known_disease_name_index()
    matches = set(name_index.get(normalized, ()))
    if len(matches) == 1:
        omim_id = next(iter(matches))
        return omim_id, known_disease_name(omim_id), False

    # Some OMIM/model titles append abbreviations as "full name; ABBR".
    # Intersect the identities of resolvable components so an abbreviation can
    # disambiguate a shared disease title. Unmatched components are ignored.
    if ";" in text:
        component_matches = [
            set(name_index.get(normalize_disease_name(component), ()))
            for component in text.split(";")
        ]
        component_matches = [ids for ids in component_matches if ids]
        if component_matches:
            intersection = set.intersection(*component_matches)
            matches = intersection if intersection else set.union(*component_matches, matches)
    if len(matches) != 1:
        return None, None, len(matches) > 1
    omim_id = next(iter(matches))
    return omim_id, known_disease_name(omim_id), False


def disease_name_matches_omim(value: Any, omim_id: Any) -> bool:
    """Return whether a supplied name exactly matches an OMIM title/alias."""
    normalized = normalize_disease_name(value)
    canonical = canonical_omim_id(omim_id)
    if not normalized or not canonical:
        return False
    resolved_id, _, ambiguous = resolve_known_disease_identity(value)
    if resolved_id:
        return resolved_id == canonical
    if ambiguous:
        return False
    expected = [known_disease_name(canonical), *known_disease_aliases(canonical)]
    expected = [
        part.strip()
        for name in expected if name
        for part in str(name).split(";")
        if part.strip()
    ]
    observed_names = {normalized}
    observed_names.update(
        normalize_disease_name(part)
        for part in str(value or "").split(";")
        if normalize_disease_name(part)
    )
    expected_names = {
        normalize_disease_name(name) for name in expected if normalize_disease_name(name)
    }
    return bool(observed_names & expected_names)


def names_compatible(observed: Any, expected: Any, synonyms: Iterable[Any] = ()) -> bool:
    """Compare normalized disease labels against the canonical name and aliases.

    This deliberately uses one deterministic rule: exact or contained matches
    after presentation normalization. Candidate retention is handled by the
    caller; a failed identity match must not itself remove a candidate.
    """
    observed_norm = normalize_disease_name(observed)
    expected_norm = normalize_disease_name(expected)
    if not observed_norm or not expected_norm:
        return False
    alternatives = [expected_norm] + [normalize_disease_name(item) for item in synonyms]
    for alternative in alternatives:
        if not alternative:
            continue
        if observed_norm == alternative or observed_norm in alternative or alternative in observed_norm:
            return True
    return False


def _support_from_legacy(item: dict[str, Any]) -> CandidateSupport:
    tool = item.get("tool", "ZeroShot")
    allowed = {"PubCaseFinder", "GestaltMatcher", "ZeroShot", "PhenoBrain", "PhenotypeSearch"}
    if tool not in allowed:
        tool = "ZeroShot"
    return CandidateSupport(
        tool=tool,
        rank=item.get("rank"),
        score=item.get("score"),
        rank_source=item.get("rank_source") or "search",
        raw_distance=item.get("raw_distance"),
        matched_hpo_id=item.get("matched_hpo_id"),
        note=item.get("note"),
        citation_allowed=False,
    )


def to_v2_candidate(candidate: MergedDiseaseCandidate | dict[str, Any]) -> MergedDiseaseCandidateV2:
    data = _as_dict(candidate)
    omim_id = canonical_omim_id(data.get("OMIM_id") or data.get("omim_id"))
    disease_name = str(data.get("disease_name") or "Unknown disease").strip()
    metadata = _metadata_for(omim_id, disease_name)
    disease_name = metadata["disease_name"]
    supports = data.get("tool_supports")
    if supports is None:
        supports = data.get("tool_rankings", [])
    support_models = [
        item if isinstance(item, CandidateSupport) else _support_from_legacy(_as_dict(item))
        for item in supports
    ]
    return MergedDiseaseCandidateV2(
        candidate_id=data.get("candidate_id") or candidate_id_for(disease_name, omim_id),
        disease_name=disease_name,
        OMIM_id=omim_id,
        gene_symbols=list(dict.fromkeys(data.get("gene_symbols") or metadata["gene_symbols"])),
        synonyms=list(dict.fromkeys(data.get("synonyms") or metadata["synonyms"])),
        phenotype_labels=list(dict.fromkeys(data.get("phenotype_labels") or metadata["phenotype_labels"])),
        consensus_count=int(data.get("consensus_count") or len({item.tool for item in support_models})),
        best_rank=int(data.get("best_rank") or min((item.rank for item in support_models if item.rank), default=9999)),
        tool_supports=support_models,
    )


def enrich_candidates(candidates: Iterable[MergedDiseaseCandidate | dict[str, Any]]) -> list[MergedDiseaseCandidateV2]:
    return [to_v2_candidate(candidate) for candidate in candidates]
