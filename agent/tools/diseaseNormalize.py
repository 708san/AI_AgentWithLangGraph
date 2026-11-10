import os
import numpy as np
import faiss
import json
import re
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import Optional
from ..state.state_types import State,ZeroShotOutput
from .candidate_metadata import (
    canonical_omim_id,
    disease_name_matches_omim,
    known_disease_aliases,
    known_disease_name,
    names_compatible,
    normalize_disease_name,
    parse_omim_id,
    resolve_known_disease_identity,
)

load_dotenv()

# 設定
tenant = "dbcls"
region = "japaneast"
model = "text-embedding-3-large"
deployment_name = f"{region}-{model}"
endpoint = f"https://{tenant}-{region}.openai.azure.com/"
api_key = os.getenv(f"AZURE_{tenant.upper()}_{region.upper()}")
if not api_key:
    raise RuntimeError(f"AZURE_{tenant.upper()}_{region.upper()} is not set in .env")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-05-01-preview"
)

# インデックスとマッピングファイルのパス
INDEX_BASE = os.path.join(os.path.dirname(__file__), "../data/DataForOmimMapping/DataForOmimMapping")
INDEX_BIN = INDEX_BASE + ".bin"
INDEX_JSON = INDEX_BASE + ".json"
OMIM_MAPPING_JSON = os.path.join(os.path.dirname(__file__), "../data/DataForOmimMapping/omim_mapping.json")

def extract_omim_number(omim_id_str: any) -> Optional[str]:
    """OMIM ID文字列から数字部分のみを抽出する"""
    if not omim_id_str or not isinstance(omim_id_str, (str, int)):
        return None
    
    match = re.search(r'\d+', str(omim_id_str))
    return match.group(0) if match else None

# インデックスとマッピングのロード
faiss_index = faiss.read_index(INDEX_BIN)
with open(INDEX_JSON, encoding="utf-8") as f:
    index_map = json.load(f)
with open(OMIM_MAPPING_JSON, encoding="utf-8") as f:
    original_omim_mapping = json.load(f)

# 数字IDをキーとする新しい検索用マッピングを作成
omim_mapping_by_number = {
    extract_omim_number(key): value 
    for key, value in original_omim_mapping.items() 
    if extract_omim_number(key)
}

def normalize_pcf_results(state: State) -> list:
    """
    Stateを受け取り、その中のPCFの結果リストを正規化する。
    """
    pcf_results = state.get("pubCaseFinder", [])
    for result in pcf_results:
        omim_id_num = extract_omim_number(result.get("omim_id"))
        if omim_id_num:
            result["omim_id"] = canonical_omim_id(omim_id_num)
        if omim_id_num and omim_id_num in omim_mapping_by_number:
            result["disease_name"] = omim_mapping_by_number[omim_id_num]
    return pcf_results


def normalize_gestalt_results(state: State) -> list:
    """
    Stateを受け取り、その中のGestaltMatcherの結果リストを正規化する。
    """
    gestalt_results = state.get("GestaltMatcher", [])
    for result in gestalt_results:
        omim_id_num = extract_omim_number(result.get("omim_id"))
        if omim_id_num:
            result["omim_id"] = canonical_omim_id(omim_id_num)
        if omim_id_num and omim_id_num in omim_mapping_by_number:
            # GestaltMatcherの出力キーに合わせて 'syndrome_name' を更新
            result["syndrome_name"] = omim_mapping_by_number[omim_id_num]
    return gestalt_results

def normalize_zeroshot_results(
    state: State,
    *,
    disease_lookup=None,
) -> Optional[ZeroShotOutput]:
    """Resolve name/OMIM pairs, then batch-map unresolved names with FAISS.

    FAISS always returns its nearest OMIM result (no similarity threshold).
    Search failures retain candidates and fall back to a locally known supplied
    OMIM ID or the generated disease name. Display ranks are always contiguous.
    """
    zeroshot_output = state.get("zeroShotResult")
    if not zeroshot_output or not zeroshot_output.ans:
        state["zeroShotValidation"] = []
        return zeroshot_output

    prepared = []
    rank_counts: dict[int, int] = {}
    for input_index, diag in enumerate(list(zeroshot_output.ans)):
        original_rank = getattr(diag, "rank", None)
        try:
            model_rank = int(original_rank)
        except (TypeError, ValueError, AttributeError):
            model_rank = None
        if model_rank is not None:
            rank_counts[model_rank] = rank_counts.get(model_rank, 0) + 1
        original_name = str(getattr(diag, "disease_name", "") or "")
        cleaned_name = re.sub(r"\s+", " ", original_name.strip().strip("*"))
        original_omim = getattr(diag, "OMIM_id", None)
        supplied_id = parse_omim_id(original_omim)
        known_id = supplied_id if supplied_id and known_disease_name(supplied_id) else None
        exact_name_id, exact_name_label, name_ambiguous = resolve_known_disease_identity(cleaned_name)
        exact_pair = bool(known_id and exact_name_id == known_id)
        # Resolution priority is name -> exact local OMIM ID -> FAISS.
        # A known supplied ID is not overwritten by a similarity search.
        needs_faiss = bool(cleaned_name and not exact_name_id and not known_id)
        prepared.append({
            "input_index": input_index,
            "diag": diag,
            "model_rank": model_rank,
            "original_rank": original_rank,
            "original_name": original_name,
            "cleaned_name": cleaned_name,
            "original_omim": original_omim,
            "supplied_id": supplied_id,
            "known_id": known_id,
            "exact_name_id": exact_name_id,
            "exact_name_label": exact_name_label,
            "name_ambiguous": name_ambiguous,
            "exact_pair": exact_pair,
            "needs_faiss": needs_faiss,
        })

    names_to_search = list(dict.fromkeys(
        row["cleaned_name"] for row in prepared if row["needs_faiss"]
    ))
    faiss_by_name = {}
    lookup_error = None
    if names_to_search:
        try:
            lookup = disease_lookup or disease_normalize_many
            lookup_results = list(lookup(names_to_search))
            if len(lookup_results) != len(names_to_search):
                raise ValueError(
                    f"Disease lookup count mismatch: expected {len(names_to_search)}, got {len(lookup_results)}"
                )
            faiss_by_name = dict(zip(names_to_search, lookup_results))
        except Exception as exc:
            lookup_error = type(exc).__name__

    # Valid model ranks determine order. Invalid ranks remain in the list after
    # them in original response order instead of removing those candidates.
    prepared.sort(key=lambda row: (
        row["model_rank"] is None or row["model_rank"] < 1,
        row["model_rank"] if row["model_rank"] is not None and row["model_rank"] > 0 else float("inf"),
        row["input_index"],
    ))

    normalized_ans = []
    validation = []
    seen_identity_keys: set[str] = set()
    for row in prepared:
        diag = row["diag"]
        cleaned_name = row["cleaned_name"]
        supplied_id = row["supplied_id"]
        known_id = row["known_id"]
        exact_name_id = row["exact_name_id"]
        resolved_id = None
        resolved_label = cleaned_name
        similarity = None
        mapping_method = "unresolved_name"
        status = "unresolved_name_retained"
        reason = "No local match found; disease name retained."

        if exact_name_id:
            resolved_id = exact_name_id
            resolved_label = row["exact_name_label"] or cleaned_name
            mapping_method = "exact_name"
            if known_id and known_id != exact_name_id:
                status = "name_authoritative_id_corrected"
                reason = "Unique local disease-name match corrected the conflicting supplied OMIM ID."
            elif row["exact_pair"]:
                status = "exact_name_id_pair"
                reason = "Disease name and supplied OMIM ID matched the same local record."
            else:
                status = "exact_name_match"
                reason = "Disease name matched a unique local title or alias."
        elif known_id:
            # The exact name lookup failed, so a locally present supplied
            # OMIM ID is the next authoritative source. Preserve the model's
            # wording in the Zero-shot display row.
            resolved_id = known_id
            resolved_label = known_disease_name(known_id) or cleaned_name
            mapping_method = "exact_omim_id"
            status = "exact_omim_id_name_variant"
            reason = "Disease name had no unique exact match; the supplied OMIM ID matched a local record."
        elif row["needs_faiss"] and cleaned_name in faiss_by_name:
            faiss_id, faiss_label, similarity = faiss_by_name[cleaned_name]
            resolved_id = canonical_omim_id(faiss_id)
            if resolved_id and faiss_label:
                resolved_label = str(faiss_label)
                mapping_method = "faiss_nearest_no_threshold"
                status = "faiss_nearest"
                reason = "No exact name/ID match; nearest FAISS result used without a similarity threshold."
            elif supplied_id:
                resolved_id = supplied_id
                mapping_method = "supplied_omim_id_fallback"
                status = "faiss_result_invalid_supplied_id_fallback"
                reason = "FAISS returned no usable identity; retained the syntactically valid supplied OMIM ID."
        elif supplied_id:
            # This covers an embedding/index failure and also ensures a valid
            # source ID is not lost when the local title table lacks it.
            resolved_id = supplied_id
            mapping_method = "supplied_omim_id_after_faiss_error" if lookup_error else "supplied_omim_id_fallback"
            status = "faiss_error_supplied_id_fallback" if lookup_error else "supplied_omim_id_retained"
            reason = (
                "FAISS lookup failed; retained the syntactically valid supplied OMIM ID."
                if lookup_error
                else "No exact local name match; retained the syntactically valid supplied OMIM ID."
            )
        elif lookup_error:
            status = "faiss_error_name_retained"
            reason = f"FAISS lookup failed ({lookup_error}); disease name retained without an OMIM ID."
        elif row["name_ambiguous"]:
            status = "ambiguous_name_retained"
            reason = "Exact name matched multiple local records; nearest FAISS result was used if available."

        if not resolved_label and resolved_id:
            resolved_label = known_disease_name(resolved_id) or ""
        excluded = not bool(resolved_label)
        exclusion_status = "empty_disease_name" if excluded else None
        # A FAISS nearest neighbor is a mapping suggestion, not proof that two
        # differently named model candidates are duplicates. Collapse those
        # only when their original normalized names are the same.
        if mapping_method == "faiss_nearest_no_threshold":
            identity_key = f"NAME:{normalize_disease_name(cleaned_name)}"
        else:
            identity_key = resolved_id or f"NAME:{normalize_disease_name(cleaned_name or resolved_label)}"
        duplicate = bool(not excluded and identity_key != "NAME:" and identity_key in seen_identity_keys)
        if duplicate:
            status = "duplicate"
            reason = "A higher-ranked row already resolved to this disease identity."
            excluded = True
            exclusion_status = "duplicate_identity"

        original_rank = row["original_rank"]
        validation_item = {
            "disease_name": row["original_name"],
            "OMIM_id": row["original_omim"],
            "rank": original_rank,
            "model_rank": original_rank,
            "input_index": row["input_index"],
            "display_rank": None,
            "normalized_disease_name": resolved_label,
            "normalized_OMIM_id": resolved_id,
            "status": status,
            "reason": reason,
            "mapping_method": mapping_method,
            "embedding_similarity": similarity,
            "lookup_error": lookup_error if row["needs_faiss"] else None,
            "name_ambiguous": row["name_ambiguous"],
            "duplicate": duplicate,
            "rank_tie": bool(row["model_rank"] is not None and rank_counts.get(row["model_rank"], 0) > 1),
            "rank_invalid": row["model_rank"] is None or row["model_rank"] < 1,
            "excluded": excluded,
            "exclusion_status": exclusion_status,
        }
        if excluded or duplicate:
            validation.append(validation_item)
            continue

        display_rank = len(normalized_ans) + 1
        validation_item["display_rank"] = display_rank
        validation.append(validation_item)
        normalized_diag = diag.model_copy(deep=True)
        # Keep the original clinical wording visible. Canonical labels remain
        # available in the audit record alongside the resolved OMIM identity.
        normalized_diag.disease_name = row["original_name"] if cleaned_name else resolved_label
        normalized_diag.OMIM_id = resolved_id
        normalized_diag.rank = display_rank
        normalized_ans.append(normalized_diag)
        seen_identity_keys.add(identity_key)

    state["zeroShotValidation"] = validation
    zeroshot_output.ans = normalized_ans
    return zeroshot_output

def disease_normalize(disease_name: str):
    """
    疾患名をembeddingし、FAISSインデックスで最も類似するOMIM IDと正規化病名を返す。
    """
    return disease_normalize_many([disease_name])[0]


def disease_normalize_many(
    disease_names: list[str],
) -> list[tuple[str | None, str | None, float | None]]:
    """Embed and map multiple disease names in one Azure and FAISS request.

    The nearest result is returned for every input; this function deliberately
    applies no similarity threshold.
    """
    if not disease_names:
        return []

    response = client.embeddings.create(
        model=deployment_name,
        input=disease_names,
    )
    embedding_items = list(response.data)
    embedding_items.sort(key=lambda item: getattr(item, "index", 0))
    if len(embedding_items) != len(disease_names):
        raise ValueError(
            f"Embedding response count mismatch: expected {len(disease_names)}, got {len(embedding_items)}"
        )
    vectors = np.asarray([item.embedding for item in embedding_items], dtype="float32")
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embedding response contained invalid vectors")
    faiss.normalize_L2(vectors)
    distances, indices = faiss_index.search(vectors, 1)

    results = []
    indexed_ids = index_map.get("omim_ids", [])
    indexed_labels = index_map.get("labels", [])
    for distance, index in zip(distances[:, 0], indices[:, 0]):
        index = int(index)
        if index < 0 or index >= len(indexed_ids) or index >= len(indexed_labels):
            results.append((None, None, float(distance)))
            continue
        raw_id = indexed_ids[index]
        label = indexed_labels[index]
        omim_number = extract_omim_number(raw_id)
        canonical_id = canonical_omim_id(raw_id)
        canonical_label = (
            known_disease_name(canonical_id)
            or omim_mapping_by_number.get(omim_number)
            or label
        )
        results.append((raw_id, canonical_label, float(distance)))
    return results

def diseaseNormalizeForDiagnosis(Diagnosis):
    """
    tentativeDiagnosis: DiagnosisOutput
    病名に互換性のあるOMIM IDと正規化病名を付与する。
    埋め込み類似度による候補除外は行わず、互換性を確認できない候補は
    OMIM IDを未確定のまま保持する。既存OMIM IDは優先する。
    """
    filtered_ans = []
    if not hasattr(Diagnosis, "ans"):
        return Diagnosis
        
    for diag in Diagnosis.ans:
        existing_omim_num = extract_omim_number(getattr(diag, "OMIM_id", None))
        if existing_omim_num:
            diag.OMIM_id = canonical_omim_id(existing_omim_num)
            diag.disease_name = (
                known_disease_name(diag.OMIM_id)
                or omim_mapping_by_number.get(
                    existing_omim_num,
                    diag.disease_name.strip().strip("*"),
                )
            )
            filtered_ans.append(diag)
            continue

        disease_name_upper = diag.disease_name.upper()
        omim_id, omim_label, sim = disease_normalize(disease_name_upper)

        candidate_id = canonical_omim_id(omim_id)
        if candidate_id and omim_label and names_compatible(
            diag.disease_name,
            omim_label,
            known_disease_aliases(candidate_id),
        ):
            diag.OMIM_id = candidate_id
            diag.disease_name = omim_label
            filtered_ans.append(diag)
        else:
            diag.OMIM_id = None
            filtered_ans.append(diag)
            print(
                f"Retained {diag.disease_name} without an OMIM ID: "
                f"no compatible local name match (embedding similarity={sim:.2f})"
            )
    Diagnosis.ans = filtered_ans
    return Diagnosis
