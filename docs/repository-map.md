# Repository map

## Snapshot

The repository develops a LangGraph-based rare-disease diagnosis agent and its evaluation workflow. The current runtime supports legacy and evidence-grounded v2 execution. The public-facing execution artifacts include `public_response.json` and `external_response.json`.

## Runtime

| Area | Current location | Responsibility |
|---|---|---|
| Pipeline and graph | `agent/agent_pipeline.py` | Builds legacy/v2 graphs and invokes the pipeline |
| Nodes | `agent/nodes.py` | LangGraph node implementations and node adapters |
| State and models | `agent/state/state_types.py` | Shared state and Pydantic/TypedDict models |
| LLM | `agent/llm/` | Azure/OpenAI wrapper and prompts |
| Tools | `agent/tools/` | HPO, PCF, GestaltMatcher, PhenoBrain, literature, Evidence, diagnosis |
| Public projection | `agent/utils/response_serializer.py` | Public and external-compatible JSON projections |
| Incremental persistence | `agent/utils/result_saver.py` | Node, phase, and compatibility artifacts |
| Project operations | `project_ops/` | Read-only cataloging and future run/evaluation management |

## Execution entry points

| Entry point | Use |
|---|---|
| `scripts/run_from_phenopacket.py` | Phenopacket-to-pipeline execution |
| `scripts/run_single_v2_case.py` | Single v2 case and response artifacts |
| `scripts/run_old_phenopacket_patient.py` | Legacy patient adapter |
| `testCode/phenopacket_image_case/run_one_case.py` | Stable image-backed case execution |
| `testCode/phenopacket_image_case/run_all_cases.py` | Resumable parallel batch execution |
| `scripts/run_stream_case.py` | Streaming node updates |
| `scripts/evaluation/runner.py` | Repeated stage-level evaluation |
| `scripts/evaluation/evaluate.py` | Offline scoring and baseline comparison |

## Output boundaries

- `public_response.json`: pipeline-native public projection.
- `external_response.json`: external delivery artifact with compatibility-oriented node wrappers and traceable evidence.
- `phase_results.json`: internal audit and partial-run artifact.
- `node_results/`: per-node snapshots.
- `execution_summary.json`: case execution status and artifact paths.
- `summary.json`: batch-level status and case records.

`external_response.json` may change shape. Each managed run must therefore record the effective output version, validation result, and any comparison or migration note.

## Current risks to address later

- Runtime, serialization, logging, and persistence responsibilities are distributed across large modules.
- Several scripts independently orchestrate similar execution paths.
- Data, experiments, ignored artifacts, and reusable source are spread across the repository and workspace.
- There is no single run registry connecting command, code, data, conditions, and results.
- Output contracts are documented in prose and tests but are not yet represented by a unified versioned registry.

## Target direction

Introduce a project-operations layer that catalogs files, data, scripts, and runs; wraps existing execution methods; registers immutable run artifacts; computes offline metrics; and generates project and run reports. Runtime refactoring follows those foundations.
