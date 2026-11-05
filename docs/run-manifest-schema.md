# Run manifest and registry

Phase 3 introduces a local Run manifest without changing existing execution scripts.

## Initialize a Run

```bash
python -m project_ops run-init \
  --run-id smoke_001 \
  --kind single_case \
  --entrypoint scripts/run_single_v2_case.py \
  --dataset-id fixture \
  --dataset-version 1 \
  --model-name gpt-test \
  --evidence-mode v2 \
  --command python scripts/run_single_v2_case.py --patient-id 1
```

This creates:

```text
artifacts/runs/smoke_001/
├── run_manifest.json
├── command.txt
├── cases/
└── report/
```

and registers the Run in:

```text
local_artifacts/catalog/run_registry.json
```

## Status lifecycle

```text
initialized → running → completed
                    ├→ failed
                    └→ interrupted
```

Use the status command to update both the manifest and registry:

```bash
python -m project_ops run-status \
  --run-dir artifacts/runs/smoke_001 \
  --status running
```

## Safety

- Run IDs cannot contain path separators.
- Existing Run directories are never overwritten by `run-init`.
- Secret environment values are never stored; only presence flags are recorded.
- The registry is a searchable summary; the full manifest remains authoritative in the Run directory.
- `artifacts/` and local registries are generated outputs and are ignored by Git.

