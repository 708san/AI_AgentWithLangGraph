 #please create a FAISS index from omim_mapping.json and put agent/data/DataForOmimMapping/.
import os
import sys
import json
import argparse
import numpy as np
import faiss
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Create FAISS index for OMIM disease labels using Azure OpenAI embeddings")
    parser.add_argument('-j', '--json', required=True, help='Path to omim_mapping.json')
    parser.add_argument('-o', '--output', default='omim_label_index', help='Output index file path (without extension)')
    parser.add_argument('--tenant', default='dbcls', help='Azure tenant name')
    parser.add_argument('--region', default='japaneast', help='Azure region')
    parser.add_argument('--model', default='text-embedding-3-large', help='Azure OpenAI embedding model')
    args = parser.parse_args()

    # Azure OpenAIの設定
    deployment_name = f"{args.region}-{args.model}"
    endpoint = f"https://{args.tenant}-{args.region}.openai.azure.com/"
    api_key = os.getenv(f"AZURE_{args.tenant.upper()}_{args.region.upper()}")
    if not api_key:
        print(f"AZURE_{args.tenant.upper()}_{args.region.upper()} is not set in .env")
        sys.exit(1)

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-05-01-preview"
    )

    # OMIMマッピングデータの読み込み
    with open(args.json, encoding="utf-8") as f:
        omim_map = json.load(f)

    omim_ids = []
    disease_labels = []
    for omim_id, label in omim_map.items():
        omim_ids.append(omim_id)
        disease_labels.append(label)

    print(f"Loaded {len(disease_labels)} disease labels.")

    # ベクトル化
    print("Embedding disease labels with Azure OpenAI...")
    batch_size = 100
    embeddings = []
    for i in range(0, len(disease_labels), batch_size):
        batch = disease_labels[i:i+batch_size]
        response = client.embeddings.create(
            model=deployment_name,
            input=batch,
        )
        batch_vectors = [item.embedding for item in response.data]
        embeddings.extend(batch_vectors)
        print(f"Embedded {i+len(batch)}/{len(disease_labels)}")

    embeddings = np.array(embeddings, dtype='float32')
    print(f"Embedding shape: {embeddings.shape}")

    # FAISSインデックス作成（コサイン類似度）
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # 保存
    output_base = args.output
    faiss.write_index(index, f"{output_base}.bin")
    with open(f"{output_base}.json", "w", encoding="utf-8") as f:
        json.dump({"omim_ids": omim_ids, "labels": disease_labels}, f, ensure_ascii=False, indent=2)
    print(f"Index and mapping saved to {output_base}.bin and {output_base}.json")

if __name__ == "__main__":
    main()