# Artifact inventory

This inventory is the initial classification for migration. It will become a machine-readable catalog in a later phase.

| Category | Current locations | Git policy | Migration treatment |
|---|---|---|---|
| Runtime source | `agent/` | Track | Preserve behavior; refactor gradually |
| Execution scripts | `scripts/`, `testCode/phenopacket_image_case/` | Track selected reusable scripts | Wrap first, consolidate later |
| Tests | `tests/` | Track | Split into unit, integration, contract, evaluation, and e2e groups |
| Specifications | `docs/` | Track | Add architecture, data, run, and output-history documents |
| Project operations | `project_ops/` | Track | Catalog first; add run registry and reporting incrementally |
| Input schema/data | `PhenoPacketSchema/`, `agent/data/`, `Data/` | Mixed | Add dataset manifests and hashes |
| Evaluation inputs | `scripts/data/`, benchmark files, local evaluation inputs | Mostly local | Register version and expected-label policy |
| Generated runs | `run_outputs/`, `res/`, `log/`, `local_artifacts/` | Ignore | Store under managed run artifacts |
| Experimental code | `tryToolsForNewData/`, `Improved_test/`, selected ignored scripts | Mostly local | Classify, archive, promote, or remove deliberately |
| Derived indexes | `*.bin`, generated HPO/OMIM indexes | Ignore | Record source data and regeneration command |
| Credentials/local config | `.env`, `.venv/`, caches | Ignore | Never copy into managed artifacts |

## Required future catalog fields

- path
- category
- purpose
- owner or responsible workflow
- source/generated status
- Git tracking status
- inputs and outputs
- regeneration command
- consumers
- deprecation status
- privacy and retention notes
