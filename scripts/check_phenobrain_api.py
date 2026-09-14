import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.tools.phenobrain_api import call_phenobrain


README_HPO_LIST = [
    "HP:0001913",
    "HP:0008513",
    "HP:0001123",
    "HP:0000365",
    "HP:0002857",
    "HP:0001744",
]


def main():
    results = call_phenobrain(
        README_HPO_LIST,
        model="Ensemble",
        topk=5,
    )

    if not results:
        print("No PhenoBrain results returned.")
        return

    for result in results:
        print(f"rank: {result.get('rank')}")
        print(f"disease_name: {result.get('disease_name')}")
        print(f"RD ID: {result.get('rd_id')}")
        print(f"OMIM ID: {result.get('omim_id')}")
        print(f"ORPHA ID: {result.get('orpha_id')}")
        print(f"score: {result.get('score')}")
        print("-" * 40)


if __name__ == "__main__":
    main()
