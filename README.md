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
    result = pipeline.run(
        present_hpo_ids=input_hpo_list,
        image_path=image_path,
        ranking_mode="llm",  # or "tool_average"
        use_togomcp=True,
    )

```

・`present_hpo_ids`: List of HPO IDs (strings)
・`absent_hpo_ids`: Explicitly absent HPO IDs (optional; unknown remains distinct from contradiction)
・image_path: Path to the patient image (optional, can be None)
・enable_log=True: Enables logging of all node results and prompts
・`ranking_mode="llm"`: Evidence-based structured LLM reranking (default); `tool_average` uses rank-normalized tool averages
・`use_togomcp=True`: Enables the fixed TogoMCP verification route

---
## 2. Running From a PhenoPacket
You can also run the pipeline through the helper script:

```
python scripts/run_from_phenopacket.py --help
```

## 3. Patient-level audit run

To run selected OldData patients and save the complete audit bundle:

```bash
.venv/bin/python scripts/test_patient11721_improved.py \
  --patient-ids 11721 \
  --ranking-mode llm
```

Multiple IDs can be passed as `--patient-ids 11721 272`, `--patient-ids 11721,272`, or a JSON list. Each patient is written below `Improved_test/patient_<id>/`, including `final_state.json`, `prompts.json`, `procedure.json`, `timing.json`, `node_profile.txt`, `node_profile.json`, `run.log`, and the existing pipeline log under `pipeline_logs/`.

`Improved_test/` is intentionally ignored by git.

Local sample datasets and historical experiments are kept under
`local_artifacts/` in this workspace and are intentionally ignored by git.

---
## Log
If you set enable_log=True when creating the pipeline, all node results and prompts will be saved in a human-readable log file under the log/ directory.
The log file name will be unique and timestamped (e.g., agent_log_20250918_123456.log).
Prompts used for LLM calls are also included in the log for traceability.

---
## Notes
For image matching, set the following in your `.env` file (project root):

```
GESTALT_API_USER=your_username
GESTALT_API_PASS=your_password
```

Azure LLM credentials are required for `ranking_mode="llm"` and ZeroShot/Reflection. If they are absent, the pipeline keeps the complete tool traces, returns `uncertain` Reflection judgments, and falls back to the deterministic tool average.

---

## Features

- The five initial rankers run in parallel. Their complete responses are stored, while the union of each Top5 forms the normal candidate pool.
- Every candidate follows the fixed TogoMCP route. Search-derived candidates receive one additional verification hop and are not expanded again.
- Reflection returns `correct`, `incorrect`, or `uncertain` after searches finish. Final ranking can use the LLM or `tool_average`.
- Known causal candidate genes are fetched after ranking and retain source and Evidence IDs; they do not affect inference.
- `clinical_text` is not an input field. Missing `sex` and `onset` are represented as `unknown`.
- **Reflection Step:** Performs a reflection process to refine diagnostic suggestions.
- **Final Diagnosis:** Outputs a final diagnosis after all reasoning steps.
- **External Knowledge Search Logic:** Mechanism for searching and integrating external knowledge sources.
- **Memory:** Persistent memory for accumulating and utilizing information across loops.
- **TogoMCP Integration:** Uses the official TogoMCP MCP endpoint for phenotype-based candidate discovery and PubMed evidence collection. The full MCP/tool responses, source metadata, and candidate-specific evidence are retained in the State/result JSON.

---

## Notes
- TogoMCP is enabled by default in the new fixed flow. Pass `use_togomcp=False` for offline tool-only execution.
- The default remote endpoint is `https://togomcp.rdfportal.org/mcp`. Override it with `TOGOMCP_MCP_URL`.
- Set `TOGOMCP_TRANSPORT=stdio` and the corresponding `TOGOMCP_STDIO_*` variables when using a local TogoMCP server.
- `TOOL_FULL_RESULT_LIMIT`, `TOGOMCP_DISCOVERY_LIMIT`, `TOGOMCP_LITERATURE_LIMIT`, and `TOGOMCP_RESEARCH_MAX_CANDIDATES` are developer-side controls for retrieval size.
- Contributions and suggestions are welcome!
