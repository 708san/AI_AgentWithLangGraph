import os
import json
import numpy as np
import faiss
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import List, Optional

from ..state.state_types import State, PhenotypeSearchFormat, OMIMEntry

# --- Initialization ---
# This block runs only once when the module is first imported.

load_dotenv()

# 1. Configure file paths
# Construct relative paths to data files based on this file's location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, '..', 'data', 'DataForDiseaseSearchFromHPO', 'phenotype_index.bin')
MAPPING_PATH = os.path.join(BASE_DIR, '..', 'data', 'DataForDiseaseSearchFromHPO', 'phenotype_index.json')

# 2. Configure Azure OpenAI client
try:
    AZURE_TENANT = "dbcls"
    AZURE_REGION = "japaneast"
    AZURE_MODEL = "text-embedding-3-large"
    DEPLOYMENT_NAME = f"{AZURE_REGION}-{AZURE_MODEL}"
    ENDPOINT = f"https://{AZURE_TENANT}-{AZURE_REGION}.openai.azure.com/"
    API_KEY = os.getenv(f"AZURE_{AZURE_TENANT.upper()}_{AZURE_REGION.upper()}")

    if not API_KEY:
        raise ValueError(f"API key AZURE_{AZURE_TENANT.upper()}_{AZURE_REGION.upper()} is not set in .env file.")

    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version="2024-05-01-preview"
    )
except Exception as e:
    print(f"Error initializing Azure OpenAI client: {e}")
    client = None

# 3. Load FAISS index and mapping data
try:
    index = faiss.read_index(INDEX_PATH)
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        phenotype_mapping = json.load(f)
    print("Successfully loaded phenotype FAISS index and mapping data.")
except Exception as e:
    print(f"Fatal Error: Could not load FAISS index or mapping file. {e}")
    index = None
    phenotype_mapping = None

# --- Main Search Function ---

def embedding_search_with_hpo(state: State) -> Optional[List[PhenotypeSearchFormat]]:
    """
    Generates a search query from the patient's HPO dictionary and uses a FAISS index
    to find diseases with similar phenotypes.
    """
    if not index or not phenotype_mapping or not client:
        print("Search cannot be performed due to initialization errors.")
        return None

    hpo_dict = state.get("hpoDict")
    if not hpo_dict:
        print("No HPO dictionary found in state. Skipping phenotype search.")
        return None

    # 1. Generate search query (comma-separated HPO labels)
    query_text = ", ".join(hpo_dict.values())
    if not query_text:
        print("Generated query text is empty. Skipping phenotype search.")
        return None
    
    #print(f"Phenotype search query: {query_text}")

    try:
        # 2. Vectorize the query
        response = client.embeddings.create(
            model=DEPLOYMENT_NAME,
            input=[query_text],
        )
        query_vector = np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)
        
        # 3. Normalize the vector (required for cosine similarity search with IndexFlatIP)
        faiss.normalize_L2(query_vector)

        # 4. Execute search with FAISS (top 5*<depth> results)
        depth = state.get("depth", 1)
        k = 5 * depth
        distances, indices = index.search(query_vector, k)

        # 5. Format the results into a list of PhenotypeSearchFormat
        search_results = []
        for i in range(k):
            idx = indices[0][i]
            score = distances[0][i]

            # An index of -1 indicates no more valid results
            if idx == -1:
                continue

            # Retrieve the corresponding disease data from the mapping file
            disease_data = phenotype_mapping[idx]
            
            # Convert to Pydantic models
            omim_entry = OMIMEntry(**disease_data)
            
            result_format = PhenotypeSearchFormat(
                disease_info=omim_entry,
                similarity_score=float(score)
            )
            search_results.append(result_format)
        
        return search_results

    except Exception as e:
        print(f"An error occurred during phenotype embedding search: {e}")
        return None