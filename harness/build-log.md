# Build log

This file records completed or attempted migration phases. It is an evidence log, not a substitute for the phase plan.

## 2026-11-04 — Phase 1 started

- Current branch: `spec/evidence-grounded-bilingual-diagnosis`
- Working tree: clean before foundation files were added
- Scope: add repository instructions, goals, migration plan, reusable prompts, and the initial repository map
- Explicit non-goals: runtime changes, output-shape changes, live API execution, branch creation, commit, push
- Evidence to collect before completion: repository map review, artifact inventory review, `git diff --check`, and final changed-file list

## 2026-11-04 — Phase 1 foundation recorded

- Added: `AGENTS.md`, `GOALS.md`, `PLANS.md`, `PROMPTS.md`
- Added: `docs/repository-map.md`, `docs/artifact-inventory.md`
- Added: `harness/build/00_repository_foundation.md`
- Verified: no changes under `agent/`, `scripts/`, `testCode/`, or `tests/`
- Verified: no new branch, commit, push, live API execution, or runtime change
- Verification commands: `git status --short --branch`, `git diff --check`, `rg --files`, policy/path inspection
- Status: foundation artifacts added; Phase 2 requires separate approval

## 2026-11-04 — Phase 1 completed

- Reviewed the repository map and artifact inventory.
- Confirmed that runtime, scripts, tests, data, and generated outputs were not modified.
- Foundation files are now the durable repository instructions for subsequent phases.

## 2026-11-04 — Phase 2 completed

- Added `project_ops` read-only inventory CLI.
- Added file, script, data, derived-data, workflow, and artifact-root classification.
- Added Git branch, commit, dirty-state, and tracked-file metadata.
- Added credential exclusion and generated-run-root exclusion.
- Generated `local_artifacts/catalog/project_catalog.json` with 3,011 non-generated files, 20 execution scripts, 563 data/derived-data files, and 6 artifact roots.
- Added `tests/test_project_catalog.py` and `docs/catalog-schema.md`.
- Verification: `python -m unittest tests/test_project_catalog.py`, `python -m project_ops inventory`, `python -m project_ops inventory --hash`, JSON parsing, and policy/path inspection.
- Hash policy: source, configuration, tests, workflow, and registered data are hashable; experimental and generated result trees remain metadata-only.
- Status: Phase 3 is next; no runtime or external-output changes were made.

## 2026-11-04 — Phase 3 completed

- Added `project_ops/run_management.py` for Run manifests, safe initialization, status lifecycle, and registry upsert.
- Extended `python -m project_ops` with `run-init`, `run-status`, and `run-list`.
- Added `docs/run-manifest-schema.md` and `tests/test_run_management.py`.
- Added `artifacts/` to `.gitignore` for generated Run directories.
- Verified a `phase3_smoke` Run: initialized, marked completed, registered, and listed by status.
- Verified manifest records command, Git branch/commit/dirty state, dataset, model, pipeline, environment presence flags, and artifact paths.
- Verification: 5 unit tests passed, CLI smoke lifecycle passed, JSON parsing passed, `git diff --check` passed.
- Status: Phase 4 is next; no diagnosis runtime or external-output changes were made.

## 2026-11-04 — Phase 4 completed

- Added `project_ops/output_validation.py` with schema-flexible external-output validation and SHA-256 fingerprinting.
- Extended the Run manifest and registry with validated output records.
- Extended `python -m project_ops` with `validate-output`.
- Added `docs/output-validation.md` and `tests/test_output_validation.py`.
- Verified node-wrapped output, explicit schema version, unknown/future fields, malformed rank handling, and missing-version warnings.
- Verified `phase3_smoke` output attachment: validation report, manifest output record, registry output record, and SHA-256.
- Verification: 8 unit tests passed, CLI validation lifecycle passed, JSON parsing passed, `git diff --check` passed.
- Status: Phase 5 is next; diagnosis runtime and existing external response files remain unchanged.

## 2026-11-04 — Phase 5 completed

- Added `project_ops/execution.py` for managed execution of existing repository commands.
- Extended `python -m project_ops` with `run-managed`.
- Captured stdout, stderr, return code, duration, and process errors under each Run.
- Added optional automatic output validation and Run attachment.
- Added case-qualified output syntax: `KIND[CASE_ID]=PATH` for batch results.
- Added `docs/managed-execution.md` and `tests/test_execution.py`.
- Verified successful and failed subprocess lifecycle behavior.
- Verified a CLI batch-style smoke Run with case `276`, output validation, report creation, and registry attachment.
- Verification: 11 unit tests passed, managed CLI lifecycle passed, JSON parsing passed, `git diff --check` passed.
- Status: Phase 6 is next; existing execution scripts and diagnosis runtime remain unchanged.

## 2026-11-04 — Phase 6 completed

- Added `project_ops/evaluation.py` for offline registration and baseline comparison of saved evaluation reports.
- Extended `project_ops` with `analyze-run`.
- Added Run `analysis` links in the manifest and registry.
- Added `docs/evaluation-management.md` and `tests/test_evaluation_management.py`.
- Verified MRR and Hit@K deltas plus gained/lost case IDs.
- Verified CLI registration of a saved evaluation report without API execution.
- Verification: 13 unit tests passed, metrics registration passed, JSON parsing passed, `git diff --check` passed.
- Status: Phase 7 is next; inference and evaluation execution logic remain unchanged.

## 2026-11-04 — Phase 7 completed

- Added `project_ops/reporting.py` for dependency-free project overview generation.
- Extended `project_ops` with `report-project`.
- Added HTML, JSON, Run CSV, and stage-metrics CSV reports.
- Added `docs/project-reporting.md` and `tests/test_reporting.py`.
- Generated a real project report from the current catalog, registry, and saved metrics: 3 registered completed Runs and 1 evaluation stage metric.
- Verification: 14 unit tests passed, project report generation passed, JSON parsing passed, `git diff --check` passed.
- Status: Phase 8 is next; runtime and existing execution scripts remain unchanged.

## 2026-11-04 — Phase 8 completed

- Added `project_ops/legacy_import.py` for non-destructive import of existing local execution roots.
- Extended `project_ops` with `import-legacy`.
- Imported 24 existing legacy/image/compatibility Run roots without rerunning APIs or copying source artifacts.
- Linked and validated 239 existing response artifacts; validation status is preserved per output.
- Added `docs/legacy-import.md` and `tests/test_legacy_import.py`.
- Regenerated the project report with 27 registered Runs (3 managed smoke Runs and 24 imported Runs).
- Verification: 15 unit tests passed, real legacy import passed, project report regeneration passed, JSON parsing passed, `git diff --check` passed.
- Status: Phase 9 is next; original legacy directories remain untouched.

## 2026-11-04 — Phase 9 completed

- Added `configs/execution_profiles.json` with 8 canonical profiles for Phenopacket, v2, image-case, batch, evaluation, and offline-analysis entry points.
- Added `project_ops/profiles.py` and `list-profiles`.
- Added `docs/execution-profiles.md` and `tests/test_profiles.py`.
- Extended project overview to include execution profiles.
- Verification: 17 unit tests passed, profile CLI passed, project report regenerated with 27 Runs and 8 profiles, JSON parsing passed, `git diff --check` passed.
- Status: Phase 10 is next; runtime behavior and existing script entrypoints remain unchanged.

## 2026-11-04 — Phase 10 first slice

- Added `project_ops/profile_execution.py` and `run-profile`.
- Selected `phenopacket.single_v2` as the first canonical profile because it has a clear single-case contract and existing output validation coverage.
- Preserved `scripts/run_single_v2_case.py` unchanged as the compatibility entrypoint.
- Verified profile command construction and a no-API `--help` execution through the managed Run lifecycle.
- Verification: 19 unit tests passed, profile CLI lifecycle passed, stdout/return-code/entrypoint/profile metadata recorded.
- Status: Phase 10 remains in progress; a behavior comparison with a fixed fixture is required before deeper runtime refactoring.

## 2026-11-04 — Phase 10 compatibility gate

- Added `project_ops/compatibility.py` and the `compare-artifacts` CLI command.
- Added a profile-level compatibility contract for `phenopacket.single_v2`:
  `external_response` is required; `public_response` and `state` are optional.
- The checker is offline and does not call diagnosis APIs. It validates both
  external responses, compares semantic JSON, and reports missing/changed/
  invalid artifacts.
- Run-location and timing metadata (`imagePath`, artifact paths, timestamps,
  and durations) are excluded from semantic comparison; diagnosis content is
  preserved in the comparison.
- Added `tests/test_compatibility.py`; all five compatibility/profile tests pass.
- Verified the CLI against an existing response directory as a fixed self-
  comparison (`equivalent`) and against two historical runs with different
  conditions (`changed`); both outcomes matched the expected gate behavior.
- The phase remains in progress until a recorded old-versus-managed fixture
  pair is compared successfully and the runtime refactor is separately tested.
