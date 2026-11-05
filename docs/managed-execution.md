# Managed execution

`run-managed` wraps an existing repository command without changing that
command's implementation or output path.

## Example: existing v2 script

```bash
python -m project_ops run-managed \
  --run-id old74_v2_gpt5_001 \
  --kind batch_cases \
  --entrypoint testCode/phenopacket_image_case/run_all_cases.py \
  --dataset-id old74 \
  --dataset-version 1 \
  --model-name gpt-5-2 \
  --evidence-mode v2 \
  --output 'external_response[276]=artifacts/runs/old74_v2_gpt5_001/cases/276/external_response.json' \
  --command python testCode/phenopacket_image_case/run_all_cases.py --limit 1
```

The wrapper:

1. Initializes and registers a Run.
2. Records the exact command and conditions.
3. Marks the Run as `running`.
4. Executes the existing command without a shell.
5. Saves `stdout.log`, `stderr.log`, return code, and duration.
6. Marks the Run `completed` or `failed`.
7. Optionally validates and attaches declared output files.

For batch runs, use `KIND[CASE_ID]=PATH` so each case receives a distinct
validation report. Without `[CASE_ID]`, the output is attached at Run level.

The wrapped command remains responsible for its own application outputs.
The wrapper only manages lifecycle and evidence around it.

