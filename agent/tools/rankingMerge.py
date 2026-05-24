import re
from typing import Optional

from ..state.state_types import State, MergedDiseaseCandidate, ToolRankingItem


def _normalize_omim_id(omim_id) -> Optional[str]:
    if not omim_id:
        return None
    match = re.search(r"\d+", str(omim_id))
    if not match:
        return None
    return f"OMIM:{match.group(0)}"


def _candidate_key(disease_name: str, omim_id: Optional[str]) -> str:
    if omim_id:
        return omim_id
    return re.sub(r"\s+", " ", disease_name.strip().upper())


def _add_candidate(
    merged: dict,
    disease_name: str,
    omim_id,
    tool_ranking: ToolRankingItem,
):
    if not disease_name:
        return

    normalized_omim_id = _normalize_omim_id(omim_id)
    key = _candidate_key(disease_name, normalized_omim_id)

    if key not in merged:
        merged[key] = {
            "disease_name": disease_name,
            "OMIM_id": normalized_omim_id,
            "tool_rankings": [],
        }

    candidate = merged[key]
    if normalized_omim_id and not candidate.get("OMIM_id"):
        candidate["OMIM_id"] = normalized_omim_id
    if len(disease_name) > len(candidate.get("disease_name", "")):
        candidate["disease_name"] = disease_name

    candidate["tool_rankings"].append(tool_ranking)


def _finalize_candidates(merged: dict) -> list[MergedDiseaseCandidate]:
    candidates = []
    for candidate in merged.values():
        rankings = candidate["tool_rankings"]
        tools = {ranking["tool"] for ranking in rankings}
        ranks = [ranking["rank"] for ranking in rankings if ranking.get("rank") is not None]
        candidate["consensus_count"] = len(tools)
        candidate["best_rank"] = min(ranks) if ranks else 9999
        candidates.append(candidate)

    return sorted(
        candidates,
        key=lambda item: (
            -item["consensus_count"],
            item["best_rank"],
            item["disease_name"],
        ),
    )


def merge_ranked_disease_candidates(state: State) -> list[MergedDiseaseCandidate]:
    merged = {}

    for index, result in enumerate(state.get("pubCaseFinder", []) or [], 1):
        _add_candidate(
            merged,
            result.get("disease_name") or result.get("omim_disease_name_en", ""),
            result.get("omim_id"),
            {
                "tool": "PubCaseFinder",
                "rank": result.get("rank") or index,
                "score": result.get("score"),
                "matched_hpo_id": result.get("matched_hpo_id", ""),
                "note": result.get("description", ""),
            },
        )

    zeroshot_output = state.get("zeroShotResult")
    if zeroshot_output and getattr(zeroshot_output, "ans", None):
        for index, result in enumerate(zeroshot_output.ans, 1):
            _add_candidate(
                merged,
                result.disease_name,
                result.OMIM_id,
                {
                    "tool": "ZeroShot",
                    "rank": result.rank or index,
                    "note": "Generated from present and optional absent HPO terms.",
                },
            )

    for index, result in enumerate(state.get("GestaltMatcher", []) or [], 1):
        _add_candidate(
            merged,
            result.get("syndrome_name", ""),
            result.get("omim_id"),
            {
                "tool": "GestaltMatcher",
                "rank": index,
                "score": result.get("score"),
                "note": f"image_id={result.get('image_id', '')}",
            },
        )

    phenotype_results = state.get("phenotypeSearchResult") or []
    for index, result in enumerate(phenotype_results, 1):
        disease_info = result.disease_info
        _add_candidate(
            merged,
            disease_info.disease_name,
            disease_info.OMIM_id,
            {
                "tool": "PhenotypeSearch",
                "rank": index,
                "score": result.similarity_score,
                "note": disease_info.definition or "",
            },
        )

    return _finalize_candidates(merged)
