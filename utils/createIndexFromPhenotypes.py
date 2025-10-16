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
    parser = argparse.ArgumentParser(description="Create FAISS index from OMIM phenotype data using Azure OpenAI embeddings")
    parser.add_argument(
        '-j', '--json', 
        default='agent/data/DataForDiseaseSearchFromHPO/omim_database.json', 
        help='Path to omim_database.json'
    )
    parser.add_argument(
        '-o', '--output', 
        default='agent/data/DataForDiseaseSearchFromHPO/phenotype_index', 
        help='Output index file path (without extension)'
    )
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

    # OMIMデータベースの読み込み
    try:
        with open(args.json, encoding="utf-8") as f:
            omim_db = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.json}")
        sys.exit(1)

    # phenotypeリストをカンマ区切りの文字列に変換
    phenotype_texts = []
    valid_entries = []
    for entry in omim_db:
        # 'phenotype'キーが存在し、かつリストが空でないことを確認
        if "phenotype" in entry and entry["phenotype"]:
            phenotype_str = ", ".join(entry["phenotype"])
            phenotype_texts.append(phenotype_str)
            valid_entries.append(entry)
        else:
            print(f"Skipping OMIM ID {entry.get('OMIM_id', 'N/A')} due to missing or empty phenotype list.")

    if not phenotype_texts:
        print("No valid phenotypes found to process. Exiting.")
        sys.exit(0)

    print(f"Found {len(phenotype_texts)} diseases with phenotypes to embed.")

    # ベクトル化
    print("Embedding phenotype lists with Azure OpenAI...")
    batch_size = 100  # Azure OpenAIのバッチサイズ制限を考慮
    embeddings = []
    for i in range(0, len(phenotype_texts), batch_size):
        batch = phenotype_texts[i:i+batch_size]
        response = client.embeddings.create(
            model=deployment_name,
            input=batch,
        )
        batch_vectors = [item.embedding for item in response.data]
        embeddings.extend(batch_vectors)
        print(f"Embedded {i+len(batch)}/{len(phenotype_texts)}")

    embeddings = np.array(embeddings, dtype='float32')
    print(f"Embedding shape: {embeddings.shape}")

    # FAISSインデックス作成（コサイン類似度）
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)  # ベクトルを正規化
    index.add(embeddings)

    # 保存
    output_base = args.output
    faiss.write_index(index, f"{output_base}.bin")
    
    # マッピングファイルとして、phenotypeを持つエントリのリスト全体を保存
    with open(f"{output_base}.json", "w", encoding="utf-8") as f:
        json.dump(valid_entries, f, ensure_ascii=False, indent=2)
        
    print(f"Index and mapping data saved to {output_base}.bin and {output_base}.json")

if __name__ == "__main__":
    main()