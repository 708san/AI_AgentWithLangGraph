# Repository instructions

## Purpose

This repository develops, runs, evaluates, and improves the Zebra Seek rare-disease diagnosis agent. It contains runtime code, external-tool adapters, benchmark and image inputs, evaluation scripts, execution artifacts, and reports.

## Working rules

- Preserve the external-output boundary. `external_response.json` is the file delivered to external consumers; its shape may evolve, so record its output version and validate every change.
- Keep runtime code, experiment code, generated artifacts, and reports distinguishable. Do not silently treat generated output as source code.
- Before changing runtime behavior, inspect the applicable contract, tests, and evaluation path. Prefer a focused change with observed verification evidence.
- Use saved artifacts for analysis whenever possible. Do not rerun external APIs merely to summarize an existing run.
- Do not put API keys, raw credentials, or unnecessary personal/image data in logs or committed artifacts.
- Unless explicitly requested, do not create or switch Git branches. Do not commit, push, or create pull requests without explicit approval.
- Record the current branch, commit, working-tree state, data identity, model, configuration, and command for every managed run.

## Important entry points

- Runtime: `agent/agent_pipeline.py`, `agent/nodes.py`, `agent/state/state_types.py`
- Output projection: `agent/utils/response_serializer.py`
- Result persistence: `agent/utils/result_saver.py`
- Phenopacket execution: `scripts/run_from_phenopacket.py`
- v2 execution: `scripts/run_single_v2_case.py`
- Image-case execution: `testCode/phenopacket_image_case/run_one_case.py`
- Batch execution: `testCode/phenopacket_image_case/run_all_cases.py`
- Evaluation: `scripts/evaluation/`

## Change boundaries

The migration starts with cataloging and run-management foundations. Do not refactor the diagnosis runtime or change external JSON fields as part of repository-foundation work unless a later approved phase explicitly includes it.
