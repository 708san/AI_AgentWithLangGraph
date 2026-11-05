# Project goals

## Purpose

Make the repository understandable and reproducible as a development and evaluation project for the Zebra Seek diagnosis agent.

## Desired outcomes

- Discoverable ownership and purpose for source files, scripts, datasets, fixtures, generated artifacts, and reports.
- Every managed execution records its command, code revision, data identity, model/configuration, runtime conditions, status, and outputs.
- Existing execution methods remain usable while being reachable through a common management layer.
- `external_response.json` remains the explicit external delivery artifact and can evolve with versioned validation and impact evidence.
- Saved results can be compared and analyzed without rerunning external APIs.
- Standard reports summarize quality, output-contract validity, runtime behavior, failures, and changes between runs.
- Codex work is guided by concise repository instructions, bounded phases, verification evidence, and explicit handoff points.

## Scope

- Repository and artifact inventory.
- Dataset and run manifests.
- Adapters around existing execution scripts.
- Evaluation, comparison, analysis, and visualization of saved results.
- Gradual separation of runtime, tooling, experiments, and reports.

## Non-goals for the foundation phase

- Changing diagnostic behavior.
- Redesigning the LangGraph flow.
- Changing the external JSON shape.
- Creating or switching Git branches.
- Deploying services or calling production systems.
