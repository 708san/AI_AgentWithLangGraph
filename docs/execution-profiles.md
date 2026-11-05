# Canonical execution profiles

`configs/execution_profiles.json` describes the supported existing entry
points without executing them. It is the bridge between the current script
layout and the managed Run layer.

```bash
python -m project_ops list-profiles
```

Each profile records:

- purpose
- entrypoint
- Run kind
- input types
- output types
- evidence mode
- whether external APIs are expected

The profiles are descriptions, not a replacement for the scripts yet. A later
refactor can use them to select the canonical command while preserving a
compatibility adapter for existing callers.

The adapter is available now:

```bash
python -m project_ops run-profile \
  --profile-id phenopacket.single_v2 \
  --run-id single_v2_001 \
  --script-args --input case.json --model gpt-5-2
```

The selected script is still the original entrypoint. The adapter adds Run
conditions, stdout/stderr capture, status updates, and optional output
validation around it.

## Fixed-fixture compatibility gate

Before moving orchestration into a new runtime module, compare two saved
artifact directories without calling a diagnosis API:

```bash
python -m project_ops compare-artifacts \
  --profile-id phenopacket.single_v2 \
  --baseline-dir path/to/old-artifacts \
  --candidate-dir path/to/managed-artifacts \
  --report local_artifacts/reports/compatibility/single_v2.json \
  --assert-equivalent
```

The checker requires `external_response.json` for this profile and treats
`public_response.json` and `state.json` as optional artifacts. It records
missing files, invalid JSON, response validation results, and JSON differences.
Run-location metadata such as `imagePath` and timing fields is excluded from
the semantic comparison; diagnosis content is not.

This is an offline gate. It does not silently rerun an external API or change
the external response contract. If the external response shape changes, record
the new output version and review the resulting compatibility report as a
separate contract change.

