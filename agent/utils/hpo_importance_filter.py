import json
import os
from functools import lru_cache
from typing import List


TOP_HPO_IMPORTANCE_LIMIT = 15
UNKNOWN_HPO_IMPORTANCE = float("inf")
HPO_IMPORTANCE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "HPO_importance",
        "HPO_importance.json",
    )
)


@lru_cache(maxsize=1)
def load_hpo_importance() -> dict[str, int]:
    with open(HPO_IMPORTANCE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        item["HPO_id"]: int(item["related_disease_num"])
        for item in data
        if "HPO_id" in item and "related_disease_num" in item
    }


def filter_hpo_by_importance(
    hpo_list: List[str],
    limit: int = TOP_HPO_IMPORTANCE_LIMIT,
) -> List[str]:
    if len(hpo_list) < limit:
        return hpo_list

    importance = load_hpo_importance()
    ranked_hpo = sorted(
        enumerate(hpo_list),
        key=lambda item: (
            importance.get(item[1], UNKNOWN_HPO_IMPORTANCE),
            item[0],
        ),
    )
    return [hpo_id for _, hpo_id in ranked_hpo[:limit]]
