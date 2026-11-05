# Evaluation management

The existing `scripts/evaluation/runner.py` and `scripts/evaluation/evaluate.py`
remain the evaluation implementation. `project_ops analyze-run` registers their
saved `evaluation.json` output against a managed Run without rerunning APIs.

```bash
python -m project_ops analyze-run \
  --run-dir artifacts/runs/gpt5_old74_001 \
  --report artifacts/runs/gpt5_old74_001/repeat_001/evaluation.json \
  --baseline-report artifacts/runs/gpt4_old74_001/repeat_001/evaluation.json
```

Generated files:

```text
artifacts/runs/gpt5_old74_001/analysis/
├── metrics.json
└── comparison.json
```

The manifest and registry link to these files. Comparisons include stage-level
MRR and Hit@K deltas plus case-level gained/lost identifiers.

