# Rare Disease Diagnosis Agent

## Purpose
This project implements an AI agent for assisting in the diagnosis of rare diseases.

---


# Usage

## 1. Using `agent_pipeline.py` from Another Script

You can use the pipeline as a Python module from your own script.  
See the example below (`testCode/test.py`):

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

---
 ## 2. Running graph_main.py Directly
You can also run the pipeline directly from the command line:

```
python [graph_main.py](http://_vscodecontentref_/0)
```

Edit the input_hpo_list and image_path variables in graph_main.py as needed.

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

