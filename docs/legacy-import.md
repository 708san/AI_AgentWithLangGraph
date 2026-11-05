# Legacy artifact import

Existing local runs can be registered without rerunning the diagnosis pipeline
or copying large result trees.

```bash
python -m project_ops import-legacy \
  --source-root testCode/phenopacket_image_case/runs/all_test
```

The importer creates a small managed Run under `artifacts/runs/` containing:

- source artifact path
- source run conditions when available
- current importer Git context
- links to legacy response/evaluation files
- validation reports for `external_response.json` and `public_response.json`

The original legacy directory is never modified or deleted. A second import of
the same source path is detected by its stable generated Run ID.

