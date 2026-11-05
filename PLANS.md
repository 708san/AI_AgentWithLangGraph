# Migration plan

## Operating model

The migration is artifact-first: first make code, data, conditions, outputs, and analysis discoverable; then consolidate execution; then refactor runtime code. Existing scripts remain the behavioral baseline until replacement paths have equivalent evidence.

## Phases

1. Repository foundation and inventory
2. File, dataset, and script catalog
3. Run manifest and artifact registry
4. External-output versioning and validation
5. Unified execution adapters
6. Evaluation and baseline comparison
7. Analysis and visualization reports
8. Legacy history import and consolidation preparation
9. Canonical execution profiles
10. Runtime and script refactoring
11. CI, review, and release workflow

## Current phase

Phases 1 through 9 are complete. Phase 10 is in progress: the canonical profile adapter and offline fixed-fixture compatibility checker are implemented; a recorded old-versus-managed comparison remains before deeper runtime refactoring.

## Approval gates

- Do not begin a later phase automatically after completing the current phase.
- Runtime behavior, external-output changes, dependency changes, live API runs, commits, pushes, and deployments require explicit scope approval.
- A phase is complete only when its acceptance criteria and observed verification evidence are recorded.
