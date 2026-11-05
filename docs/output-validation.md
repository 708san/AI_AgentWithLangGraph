# External output validation

Phase 4 adds schema-flexible validation for `external_response.json`.

The validator does not assume that one output shape will remain permanent. It records:

- output kind
- declared or detected schema version
- shape profile
- file size and SHA-256
- structural errors
- compatibility warnings
- validation status

## Example

```bash
python -m project_ops validate-output \
  --run-dir artifacts/runs/example_001 \
  --response artifacts/runs/example_001/cases/1/external_response.json \
  --case-id 1 \
  --schema-version zebra-v2
```

The report is written under the Run:

```text
artifacts/runs/example_001/validation/external_response__1.json
```

The Run manifest receives an `outputs` entry containing the response path,
schema version, hash, validation status, and validation-report path. The same
summary is copied into the local Run registry.

## Status meanings

- `passed`: no structural errors or warnings.
- `partial`: structurally readable, but version or compatibility information is incomplete.
- `failed`: invalid JSON, invalid root/block types, or invalid rank/reference structure.

Unknown or future fields are retained and do not fail validation. This allows
`external_response.json` to evolve while keeping each Run auditable.

