# Project reporting

`report-project` combines the saved project catalog, Run registry, evaluation
metrics, output validation statuses, and analysis links into a small static
report. It does not execute diagnosis or evaluation APIs.

```bash
python -m project_ops inventory
python -m project_ops report-project
```

Generated files:

```text
local_artifacts/reports/project_overview/
├── project_overview.json
├── project_overview.html
├── runs.csv
└── stage_metrics.csv
```

The HTML report includes repository file counts, Run status counts, registered
Run conditions, and stage metrics from saved evaluation reports. CSV files are
provided for further analysis in spreadsheet or notebook tools.

