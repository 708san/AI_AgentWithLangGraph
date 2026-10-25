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


def canonical_omim_id(value: Any) -> str | None:
    if value is None or value == "":
        return None
    match = _OMIM_RE.search(str(value))
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
        "disease_name": disease_name or record.get("disease_name") or "Unknown disease",
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


def names_compatible(observed: Any, expected: Any, synonyms: Iterable[Any] = ()) -> bool:
    """Conservative compatibility check for aliases and subtype formatting.

    This is deliberately not a fuzzy disease replacer. It accepts exact names,
    contained OMIM-style titles, and substantial token overlap, but does not
    turn an unrelated name into the expected disease merely because an ID exists.
    """
    observed_norm = normalize_disease_name(observed)
    expected_norm = normalize_disease_name(expected)
    if not observed_norm or not expected_norm:
        return False
    alternatives = [expected_norm] + [normalize_disease_name(item) for item in synonyms]
    observed_tokens = set(observed_norm.split())
    for alternative in alternatives:
        if not alternative:
            continue
        if observed_norm == alternative or observed_norm in alternative or alternative in observed_norm:
            return True
        alternative_tokens = set(alternative.split())
        overlap = len(observed_tokens & alternative_tokens) / max(len(observed_tokens | alternative_tokens), 1)
        if overlap >= 0.6:
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
