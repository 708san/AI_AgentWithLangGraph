"""Run only the Zero-shot LLM, save outputs, and score the benchmark."""
from scripts.evaluation.runner import main

if __name__ == "__main__":
    raise SystemExit(main("zero-shot"))
