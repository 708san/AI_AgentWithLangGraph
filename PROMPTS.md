# Reusable Codex prompts

## Repository orientation

Read `AGENTS.md`, `GOALS.md`, `PLANS.md`, `harness/build-log.md`, the applicable phase file under `harness/build/`, and the relevant runtime, script, test, data, and documentation files. Perform a read-only preflight. Report the current branch, working-tree state, files inspected, execution paths, output contracts, assumptions, and unresolved questions. Do not modify files.

## Prepare a phase

Read the applicable phase file and current context. Summarize the exact scope, non-goals, files to change, risks, verification commands, and approval needed. Do not start a later phase or create a branch unless explicitly requested.

## Execute an approved phase

Read the phase plan and applicable `AGENTS.md` files. Implement only the approved scope on the current branch. Preserve existing user changes. Keep external API calls and external writes behind the phase's explicit authorization. Record commands and observed evidence. Stop at the phase handoff condition.

## Verify and analyze

Use saved run artifacts when possible. Run the smallest relevant tests first, then the broader checks named by the phase. For result analysis, do not call diagnosis APIs again unless the phase explicitly requests a new run. Report failures, missing evidence, and limitations rather than inferring success.

## Review and hand off

Inspect the diff, contract impact, data lineage, generated artifacts, tests, and documentation. Run `git diff --check`. Summarize changed files, commands actually run, evidence, unresolved risks, and the next approval gate. Do not commit, push, or create a pull request unless explicitly requested.
