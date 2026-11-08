# Rare Disease Diagnosis Agent

## Purpose
This project implements an AI agent for assisting in the diagnosis of rare diseases.

---


# Usage

## 1. Using `agent_pipeline.py` from Another Script

You can use the pipeline as a Python module from your own script.  
See the minimal example below:

```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.agent_pipeline import RareDiseaseDiagnosisPipeline 

if __name__ == "__main__":
    input_hpo_list = [
        "HP:0000054", "HP:0000286", "HP:0000297", "HP:0000965", "HP:0001263",
        "HP:0001513", "HP:0002265", "HP:0002342", "HP:0030820"
    ]
    image_path = "/path/to/your/image.jpg"

    pipeline = RareDiseaseDiagnosisPipeline(enable_log=True)
    result = pipeline.run(input_hpo_list, image_path)

```

・input_hpo_list: List of HPO IDs (strings)
・image_path: Path to the patient image (optional, can be None)
・enable_log=True: Enables logging of all node results and prompts

The pipeline uses the evidence-grounded v2 flow by default. The legacy flow is
still available when compatibility testing requires it:

```python
result = pipeline.run(input_hpo_list, image_path, evidence_mode="legacy")
```

---
## 2. Running From a PhenoPacket
You can also run the pipeline through the helper script:

```
python scripts/run_from_phenopacket.py --help
```

Local sample datasets and historical experiments are kept under
`local_artifacts/` in this workspace and are intentionally ignored by git.

### Running legacy patient 276 with a facial image

The repository includes a legacy TSV/image pair for patient `276` (image
`359.png`). The adapter below materializes the old TSV row as a compatible
Phenopacket and saves the complete run under a timestamped directory:

```bash
python scripts/run_old_phenopacket_patient.py --patient-id 276
```

Before making API calls, validate the resolved input and image with:

```bash
python scripts/run_old_phenopacket_patient.py --patient-id 276 --dry-run
```

Each run saves `input/phenopacket.json`, `input/metadata_row.json`, a copy of
the face image, `process.log`, `logs/`, `node_results/`, the final JSON result,
and `execution_summary.json` under `run_outputs/legacy_phenopacket/`.

### Running the current v2 route on all 74 old cases

The reusable batch runner defaults to `Data/OldData/browser_input_74_cases.tsv`
and `Data/OldData/test_images/`:

```bash
python testCode/phenopacket_image_case/run_all_cases.py \
  --model gpt-5-2 \
  --max-parallel-cases 2 \
  --pubmed-max-workers 1 \
  --evidence-max-workers 2
```

`--max-parallel-cases` controls how many patient/image cases run at once. It
defaults to 2; increase it gradually because each case also makes concurrent
PubMed and Evidence requests.

PhenoBrainはデフォルトで無効です。使用する場合だけ
`--use-phenobrain` を指定してください。

Completed run folders can be summarized with the existing artifacts, without
calling the diagnosis APIs again:

```bash
.venv/bin/python scripts/analyze_old74_v2_results.py
```

The tables and graphs are written to
`testCode/phenopacket_image_case/runs/all_test/analysis/`.

Each patient is written to a stable folder under
`testCode/phenopacket_image_case/runs/all_test/`.
`results/phase_results.json` contains the tool phases, normalized and raw
Zero-shot output, `tentativeDiagnosis`, reflection, evidence, and
`finalDiagnosis`; `node_results/<patient_id>/<node>.json` contains one snapshot
per graph node, and `logs/agent.log` contains the prompts and formatted node
results. The batch is resumable: press Ctrl-C and rerun the same command; cases
whose completed output and input/configuration fingerprints match are skipped,
while the interrupted case is safely retried with its partial artifacts
preserved until that retry starts. Duplicate patient IDs with different face
images are executed separately under image-qualified case folders, while
`case_manifest.json` and `summary.json` retain `patient_id` for optional later
aggregation. Use `--force` to rebuild completed cases.

---
## Log
If you set enable_log=True when creating the pipeline, all node results and prompts will be saved in a human-readable log file under the log/ directory.
The log file name will be unique and timestamped (e.g., agent_log_20250918_123456.log).
Prompts used for LLM calls are also included in the log for traceability.

---
## Notes
You must set the following in your .env file (project root):

```
GESTALT_API_USER=your_username
GESTALT_API_PASS=your_password
```

The default diagnosis pipeline uses GPT-5.2 / medium for the main stages and
GPT-5.4 / high for Zero-shot. Configure both Azure deployments:

```dotenv
AZURE_OPENAI_5-2_ENDPOINT=your_endpoint
AZURE_OPENAI_5-2_API_KEY=your_api_key
AZURE_OPENAI_5-2_DEPLOYMENT_NAME=your_deployment
AZURE_OPENAI_5-2_API_VERSION=your_api_version
AZURE_OPENAI_5-4_ENDPOINT=your_endpoint
AZURE_OPENAI_5-4_API_KEY=your_api_key
AZURE_OPENAI_5-4_DEPLOYMENT_NAME=your_deployment
AZURE_OPENAI_5-4_API_VERSION=your_api_version
```

The single-case runners accept `--model`, `--reasoning-effort`,
`--zero-shot-model`, and `--zero-shot-reasoning-effort` overrides. For example,
to use GPT-5.2 / medium for both the main flow and Zero-shot:

```bash
python testCode/phenopacket_image_case/run_one_case.py \
  --patient-id 272 \
  --model gpt-5-2 \
  --reasoning-effort medium \
  --zero-shot-model gpt-5-2 \
  --zero-shot-reasoning-effort medium
```

GPT-5.x requests use a default maximum of 30,000 completion tokens. GPT-4o
retains its model-specific 8,192-token default. Reflection already has its own
30,000-token limit. For reasoning models, reasoning tokens count against this
completion budget and leave fewer tokens for the visible structured response.

---

## Features (Implemented)
- **PCF Integration:** Calls the PubCaseFinder (PCF) API for case-based retrieval.
- **Zero-Shot Diagnosis:** Utilizes zero-shot learning for disease suggestion.
- **Tentative Diagnosis:** Generates a preliminary diagnosis based on input data.
- **Reflection Step:** Performs a reflection process to refine diagnostic suggestions.
- **Final Diagnosis:** Outputs a final diagnosis after all reasoning steps.
- **External Knowledge Search Logic:** Mechanism for searching and integrating external knowledge sources.
- **Memory:** Persistent memory for accumulating and utilizing information across loops.

---

## Features (Not Yet Implemented)
- **MCP Server Integration:** Support for MCP server communication.

---

## Notes
- Further development is needed for memory management and external knowledge integration.
- Contributions and suggestions are welcome!
