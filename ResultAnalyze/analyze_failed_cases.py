"""Re-query the diagnostic tools for the previously difficult PhenoPacket cases.

The historical result files are not used to determine the new ranks.  The
script reads the original benchmark TSV, sends the same HPO/image inputs to
each service, and compares returned candidates with the known OMIM diagnosis.

Run ``python ResultAnalyze/analyze_failed_cases.py --dry-run`` to validate
inputs, or add ``--model gpt-5-2`` to perform the external calls.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_TSV = PROJECT_ROOT / "local_artifacts/evaluation/sampleData/ValidationDataWithoutDupli_newest.tsv"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "local_artifacts/evaluation/sampleData/PhenoPacketStore_25072025"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ResultAnalyze"
DEFAULT_TARGET_IDS = (
    "11702", "11706", "11710", "11715", "11718", "11721", "12293", "12300",
    "12289", "12292", "12299", "12291", "11700", "11708", "11714",
)
load_dotenv(PROJECT_ROOT / ".env")
HTTP_TIMEOUT = float(os.getenv("RESULT_ANALYZE_HTTP_TIMEOUT", "30"))
HTTP_RETRIES = int(os.getenv("RESULT_ANALYZE_HTTP_RETRIES", "2"))


def norm_id(value: Any) -> str:
    value = "" if value is None else str(value).strip().upper()
    if not value:
        return ""
    return value if value.startswith("OMIM:") else f"OMIM:{value}" if value.isdigit() else value


def norm_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def split_hpo(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return jsonable(value.dict())
    return str(value)


def request_with_retries(method: str, url: str, *, retries: int = HTTP_RETRIES, timeout: float = HTTP_TIMEOUT, **kwargs: Any) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"{method} {url} failed after {retries} attempts: {last_error}") from last_error


def load_cases(path: Path, target_ids: Iterable[str], image_dir: Path) -> list[dict[str, Any]]:
    target_ids = list(map(str, target_ids))
    target = set(target_ids)
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            patient_id = str(row.get("patient_id", "")).strip()
            if patient_id not in target:
                continue
            omim_values = [item.strip() for item in str(row.get("omim_ids", "")).split(",") if item.strip()]
            cases.append({
                "patient_id": patient_id,
                "phenopacket_id": row.get("PhenoPacket ID", ""),
                "image_id": row.get("image_id", ""),
                "image_path": str(image_dir / f"{row.get('image_id', '')}.jpg"),
                "present_hpo": split_hpo(row.get("present_features")),
                "absent_hpo": split_hpo(row.get("absent_features")),
                "sex": row.get("gender") or "Unknown",
                "onset": "Unknown",
                "target_omim": norm_id(omim_values[0] if omim_values else ""),
                "target_disease": str(row.get("disorder_names", "")).strip(),
                "gene": row.get("gene_names", ""),
                "pmid": row.get("pmid", ""),
            })
    found = {case["patient_id"] for case in cases}
    missing = target - found
    if missing:
        raise ValueError(f"Target patient IDs missing from {path}: {', '.join(sorted(missing))}")
    order = {patient_id: index for index, patient_id in enumerate(target_ids)}
    return sorted(cases, key=lambda case: order[case["patient_id"]])


def match_target(results: list[dict[str, Any]], target_omim: str, target_name: str) -> dict[str, Any]:
    for index, item in enumerate(results, 1):
        candidate_id = norm_id(item.get("omim_id") or item.get("OMIM_id") or item.get("id"))
        if candidate_id and candidate_id == target_omim:
            return {"status": "exact", "rank": index, "matched_by": "omim_id"}
    target_name_norm = norm_text(target_name)
    for index, item in enumerate(results, 1):
        candidate_name = (
            item.get("disease_name") or item.get("disease_info", {}).get("disease_name")
            or item.get("omim_disease_name_en") or item.get("syndrome_name") or item.get("ENG_NAME") or ""
        )
        if target_name_norm and norm_text(candidate_name) == target_name_norm:
            return {"status": "name_only", "rank": index, "matched_by": "normalized_disease_name"}
    return {"status": "not_returned", "rank": None, "matched_by": None}


def process_pcf(hpo: list[str], rank_probe_k: int) -> dict[str, Any]:
    url = "https://pubcasefinder.dbcls.jp/api/pcf_get_ranked_list"
    params = {"target": "omim", "format": "json", "hpo_id": ",".join(hpo)}
    response = request_with_retries("GET", url, params=params)
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"PCF response is not a list: {type(payload).__name__}")
    results = [{
        "rank": index,
        "omim_disease_name_en": item.get("omim_disease_name_en", ""),
        "description": item.get("description", ""),
        "score": item.get("score"),
        "omim_id": item.get("id", ""),
    } for index, item in enumerate(payload, 1) if isinstance(item, dict)]
    return {
        "request": {"method": "GET", "url": response.url, "params": params},
        "server_returned_count": len(results), "configured_output_depth": 5,
        "rank_probe_k": rank_probe_k, "results": results[:rank_probe_k],
    }


def process_gestalt(image_path: Path, rank_probe_k: int) -> dict[str, Any]:
    username, password = os.getenv("GESTALT_API_USER"), os.getenv("GESTALT_API_PASS")
    if not username or not password:
        raise ValueError("GESTALT_API_USER/GESTALT_API_PASS are not set")
    url = "https://pubcasefinder.dbcls.jp/gm_endpoint/predict"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    response = request_with_retries(
        "POST", url, headers={"Content-Type": "application/json"},
        json={"img": image_b64}, auth=(username, password),
    )
    payload = response.json()
    raw_results = payload.get("suggested_syndromes_list", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_results, list):
        raise ValueError("GestaltMatcher response has no syndrome list")
    results = []
    for index, item in enumerate(raw_results, 1):
        if not isinstance(item, dict):
            continue
        distance = item.get("distance")
        if distance is None:
            distance = item.get("gestalt_score")
        score = (1.3 - float(distance)) / 1.3 if distance is not None else None
        results.append({
            "rank": index, "subject_id": item.get("subject_id", ""),
            "syndrome_name": item.get("syndrome_name", ""), "omim_id": item.get("omim_id", ""),
            "image_id": item.get("image_id", ""), "score": score, "distance": distance,
        })
    return {
        "request": {"method": "POST", "url": response.url, "json": {"img": "<base64 image omitted>"}},
        "server_returned_count": len(results), "configured_output_depth": 5,
        "rank_probe_k": rank_probe_k, "results": results[:rank_probe_k],
    }


class VectorSearch:
    """The same Azure embedding and FAISS query as embeddingSearchWithHPO.py."""

    def __init__(self) -> None:
        import faiss
        from openai import AzureOpenAI
        self.faiss = faiss
        self.index_path = PROJECT_ROOT / "agent/data/DataForDiseaseSearchFromHPO/phenotype_index.bin"
        self.mapping_path = PROJECT_ROOT / "agent/data/DataForDiseaseSearchFromHPO/phenotype_index.json"
        self.index = faiss.read_index(str(self.index_path))
        self.mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        api_key = os.getenv("AZURE_DBCLS_JAPANEAST")
        if not api_key:
            raise ValueError("AZURE_DBCLS_JAPANEAST is not set")
        self.deployment = "japaneast-text-embedding-3-large"
        self.client = AzureOpenAI(
            azure_endpoint="https://dbcls-japaneast.openai.azure.com/", api_key=api_key,
            api_version="2024-05-01-preview",
            timeout=HTTP_TIMEOUT,
            max_retries=HTTP_RETRIES - 1,
        )
        self.hpo_mapping = json.loads((PROJECT_ROOT / "agent/data/phenotype_mapping.json").read_text(encoding="utf-8"))

    def search(self, hpo: list[str], rank_probe_k: int) -> dict[str, Any]:
        query_text = ", ".join(self.hpo_mapping.get(hpo_id, "") for hpo_id in hpo)
        response = self.client.embeddings.create(model=self.deployment, input=[query_text])
        query_vector = np.asarray(response.data[0].embedding, dtype="float32").reshape(1, -1)
        self.faiss.normalize_L2(query_vector)
        k = min(rank_probe_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)
        results = []
        for rank, (distance, index) in enumerate(zip(distances[0], indices[0]), 1):
            if index < 0:
                continue
            disease = self.mapping[int(index)]
            results.append({
                "rank": rank, "disease_info": disease, "similarity_score": float(distance),
                "omim_id": disease.get("OMIM_id", ""), "disease_name": disease.get("disease_name", ""),
            })
        return {
            "request": {"method": "AzureOpenAI.embeddings.create", "deployment": self.deployment,
                        "api_version": "2024-05-01-preview", "input": query_text},
            "index_path": str(self.index_path), "index_size": self.index.ntotal,
            "configured_output_depth": 5, "rank_probe_k": k, "results": results,
        }


class PhenoBrainClient:
    """Client matching the historical /predict -> poll -> detail API flow."""

    def __init__(self) -> None:
        self.base_url = os.getenv("PHENOBRAIN_BASE_URL", "https://www.phenobrain.cs.tsinghua.edu.cn").rstrip("/")

    @staticmethod
    def source_codes(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item for item in re.split(r"[,;]\s*", value) if item]
        if isinstance(value, list):
            return [str(item) for item in value if item]
        return [str(value)]

    @staticmethod
    def first_code(codes: list[str], prefix: str) -> str | None:
        return next((code for code in codes if code.upper().startswith(prefix)), None)

    @staticmethod
    def detail_map(details: Any) -> dict[str, dict[str, Any]]:
        if isinstance(details, dict):
            for key in ("result", "data", "diseaseList", "diseases"):
                if isinstance(details.get(key), list):
                    details = details[key]
                    break
            else:
                if details.get("CODE"):
                    details = [details]
        if isinstance(details, dict):
            return {str(key): value for key, value in details.items() if isinstance(value, dict)}
        return {str(item.get("CODE") or item.get("RD_ID") or item.get("rd_id")): item
                for item in (details or []) if isinstance(item, dict) and
                (item.get("CODE") or item.get("RD_ID") or item.get("rd_id"))}

    def call(self, hpo: list[str], rank_probe_k: int) -> dict[str, Any]:
        params: list[tuple[str, Any]] = [("model", "Ensemble")]
        params.extend(("hpoList[]", item) for item in hpo)
        params.append(("topk", rank_probe_k))
        predict = request_with_retries("GET", f"{self.base_url}/predict", params=params).json()
        task_id = predict.get("TASK_ID") if isinstance(predict, dict) else None
        if not task_id:
            raise ValueError(f"PhenoBrain predict response has no TASK_ID: {predict}")
        deadline = time.monotonic() + float(os.getenv("PHENOBRAIN_MAX_POLL_SECONDS", "120"))
        while True:
            poll = request_with_retries(
                "GET", f"{self.base_url}/query-predict-result", params={"taskId": str(task_id)}
            ).json()
            state = poll.get("state") if isinstance(poll, dict) else None
            if state == "SUCCESS":
                predictions = poll.get("result", [])
                break
            if state not in {"MODEL_INIT", "MODEL_PREDICT"}:
                raise ValueError(f"PhenoBrain returned state {state}: {poll}")
            if time.monotonic() >= deadline:
                raise TimeoutError("PhenoBrain prediction polling timed out")
            time.sleep(float(os.getenv("PHENOBRAIN_POLL_INTERVAL", "1")))
        predictions = predictions if isinstance(predictions, list) else []
        rd_ids = [str(item.get("CODE") or item.get("RD_ID") or item.get("rd_id")) for item in predictions
                  if isinstance(item, dict) and (item.get("CODE") or item.get("RD_ID") or item.get("rd_id"))]
        details = self.detail_map(request_with_retries(
            "POST", f"{self.base_url}/disease-list-detail", json={"diseaseList": rd_ids}
        ).json())
        results = []
        for rank, item in enumerate(predictions, 1):
            if not isinstance(item, dict):
                continue
            rd_id = str(item.get("CODE") or item.get("RD_ID") or item.get("rd_id") or "")
            detail = details.get(rd_id, {})
            codes = self.source_codes(detail.get("SOURCE_CODES"))
            results.append({
                "rank": rank, "disease_name": detail.get("ENG_NAME") or item.get("ENG_NAME") or "",
                "omim_id": self.first_code(codes, "OMIM:"), "orpha_id": self.first_code(codes, "ORPHA:"),
                "source_codes": codes, "rd_id": rd_id, "score": item.get("SCORE"),
            })
        return {
            "request": {"predict": {"method": "GET", "url": f"{self.base_url}/predict", "params": params},
                        "poll": {"method": "GET", "url": f"{self.base_url}/query-predict-result"},
                        "detail": {"method": "POST", "url": f"{self.base_url}/disease-list-detail"}},
            "model": "Ensemble", "configured_output_depth": 5, "rank_probe_k": rank_probe_k,
            "results": results,
        }


def make_hpo_dict(hpo: list[str]) -> dict[str, str]:
    mapping = json.loads((PROJECT_ROOT / "agent/data/phenotype_mapping.json").read_text(encoding="utf-8"))
    return {hpo_id: mapping.get(hpo_id, "") for hpo_id in hpo}


def run_gpt(hpo: list[str], absent_hpo: list[str], sex: str, onset: str, model_name: str,
            include_absent: bool, num_diagnoses: int) -> dict[str, Any]:
    """Run the repository's existing Zero-Shot prompt without truth leakage."""
    from agent.llm.azure_llm_instance import get_llm_instance
    from agent.tools.ZeroShot import createZeroshot
    llm = get_llm_instance(model_name)
    # AzureChatOpenAI has no finite timeout in the production wrapper by
    # default.  Re-analysis must fail fast per tool, otherwise one unavailable
    # deployment can hold all remaining cases indefinitely.
    llm.llm = llm.get_temp_llm_with_max_tokens(
        llm.default_max_tokens,
        timeout_seconds=HTTP_TIMEOUT,
    )
    result, prompt = createZeroshot({
        "hpoDict": make_hpo_dict(hpo), "absentHpoDict": make_hpo_dict(absent_hpo),
        "use_absentHPO": include_absent, "onset": onset, "sex": sex,
        "num_diagnoses": num_diagnoses,
        "llm": llm,
    })
    return {"request": {"model": model_name, "include_absent_hpo": include_absent, "prompt": prompt},
            "results": jsonable(result)}


@dataclass
class ToolSpec:
    name: str
    call: Callable[[dict[str, Any]], dict[str, Any]]


def build_tools(model_name: str, rank_probe_k: int, include_absent: bool, gpt_top_k: int) -> list[ToolSpec]:
    vector, phenobrain = VectorSearch(), PhenoBrainClient()
    return [
        ToolSpec("GestaltMatcher", lambda case: process_gestalt(Path(case["image_path"]), rank_probe_k)),
        ToolSpec("PubCaseFinder", lambda case: process_pcf(case["present_hpo"], rank_probe_k)),
        ToolSpec("VectorSimilarity", lambda case: vector.search(case["present_hpo"], rank_probe_k)),
        ToolSpec("PhenoBrain", lambda case: phenobrain.call(case["present_hpo"], rank_probe_k)),
        ToolSpec("GPTZeroShot", lambda case: run_gpt(
            case["present_hpo"], case["absent_hpo"], case["sex"], case["onset"],
            model_name, include_absent, gpt_top_k,
        )),
    ]


def add_match(tool_name: str, payload: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    if tool_name == "GPTZeroShot":
        answer = results.get("ans", []) if isinstance(results, dict) else []
        results = [{"rank": item.get("rank", index), "disease_name": item.get("disease_name", ""),
                    "omim_id": item.get("OMIM_id", "")} for index, item in enumerate(answer, 1)]
    return {"target_match": match_target(results, case["target_omim"], case["target_disease"]),
            "returned_count": len(results), **payload}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_summary_rows(output_dir: Path) -> list[dict[str, Any]]:
    """Collect the latest per-tool rows from every saved case file."""
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "cases").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(record.get("summary_rows", []))
    return rows


def write_summary(output_dir: Path, rows: list[dict[str, Any]], patient_ids: list[str]) -> None:
    tool_order = {name: index for index, name in enumerate(
        ["GestaltMatcher", "PubCaseFinder", "VectorSimilarity", "PhenoBrain", "GPTZeroShot"]
    )}
    case_order = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    rows.sort(key=lambda row: (case_order.get(row["patient_id"], 999), tool_order.get(row["tool"], 999)))
    write_json(output_dir / "summary.json", {"results": rows})
    fields = ["patient_id", "target_omim", "target_disease", "tool", "status", "match_status", "rank", "matched_by", "returned_count", "error"]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", type=Path, default=DEFAULT_INPUT_TSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patient-ids", nargs="+", default=list(DEFAULT_TARGET_IDS))
    parser.add_argument("--model", choices=["gpt-4o", "gpt-5-1", "gpt-5-2"], default="gpt-5-2")
    parser.add_argument("--rank-probe-k", type=int, default=100)
    parser.add_argument(
        "--tools", nargs="+",
        choices=["GestaltMatcher", "PubCaseFinder", "VectorSimilarity", "PhenoBrain", "GPTZeroShot"],
        default=["GestaltMatcher", "PubCaseFinder", "VectorSimilarity", "PhenoBrain", "GPTZeroShot"],
        help="Tools to execute. With --force, selected tools replace only those results in existing case files.",
    )
    parser.add_argument("--include-absent-hpo", action="store_true", help="Include explicitly absent HPO terms in the GPT request.")
    parser.add_argument("--gpt-top-k", type=int, default=5, help="Number of diagnoses requested from GPT (default: 5).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-summary", action="store_true", help="Rebuild summary files from existing case JSON files without API calls.")
    parser.add_argument("--force", action="store_true", help="Re-run cases even if their output already exists.")
    args = parser.parse_args()
    if args.rank_probe_k < 5:
        parser.error("--rank-probe-k must be at least 5")
    if args.gpt_top_k < 1:
        parser.error("--gpt-top-k must be at least 1")
    if not args.input_tsv.is_file():
        parser.error(f"Input TSV not found: {args.input_tsv}")
    cases = load_cases(args.input_tsv, args.patient_ids, args.image_dir)
    print(f"Cases: {len(cases)}")
    for case in cases:
        image_exists = Path(case["image_path"]).is_file()
        print(f"- {case['patient_id']}: target={case['target_omim']} HPO+={len(case['present_hpo'])} HPO-={len(case['absent_hpo'])} image={image_exists}")
        if not image_exists:
            raise SystemExit(f"Image not found for {case['patient_id']}: {case['image_path']}")
    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.rebuild_summary:
        rows = collect_summary_rows(args.output_dir)
        write_summary(args.output_dir, rows, args.patient_ids)
        print(f"Rebuilt summary from {len(rows)} rows: {args.output_dir / 'summary.json'}")
        return 0

    write_json(args.output_dir / "run_metadata.json", {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "input_tsv": str(args.input_tsv),
        "image_dir": str(args.image_dir), "model": args.model, "rank_probe_k": args.rank_probe_k,
        "gpt_top_k": args.gpt_top_k,
        "configured_original_output_depth": 5, "include_absent_hpo_for_gpt": args.include_absent_hpo,
        "target_patient_ids": args.patient_ids,
        "tools": args.tools,
        "environment_variables_used": ["GESTALT_API_USER", "GESTALT_API_PASS", "AZURE_DBCLS_JAPANEAST",
            f"AZURE_OPENAI_{ {'gpt-4o': '4o', 'gpt-5-1': '5-1', 'gpt-5-2': '5-2'}[args.model] }_ENDPOINT",
            f"AZURE_OPENAI_{ {'gpt-4o': '4o', 'gpt-5-1': '5-1', 'gpt-5-2': '5-2'}[args.model] }_API_KEY",
            f"AZURE_OPENAI_{ {'gpt-4o': '4o', 'gpt-5-1': '5-1', 'gpt-5-2': '5-2'}[args.model] }_API_VERSION",
            f"AZURE_OPENAI_{ {'gpt-4o': '4o', 'gpt-5-1': '5-1', 'gpt-5-2': '5-2'}[args.model] }_DEPLOYMENT_NAME"],
        "note": "Secret values are not written. Each tool records historical depth 5 and the expanded rank probe separately.",
    })
    tools = [tool for tool in build_tools(args.model, args.rank_probe_k, args.include_absent_hpo, args.gpt_top_k)
             if tool.name in args.tools]
    tool_order = {name: index for index, name in enumerate(
        ["GestaltMatcher", "PubCaseFinder", "VectorSimilarity", "PhenoBrain", "GPTZeroShot"]
    )}
    summary_rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, 1):
        output_path = args.output_dir / "cases" / f"{case['patient_id']}.json"
        if output_path.exists() and not args.force:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            summary_rows.extend(existing.get("summary_rows", []))
            print(f"[{case_index}/{len(cases)}] {case['patient_id']} skipped (exists)")
            continue
        if output_path.exists():
            case_record = json.loads(output_path.read_text(encoding="utf-8"))
            case_record["case"] = case
            case_record.setdefault("tools", {})
            case_record["summary_rows"] = [
                row for row in case_record.get("summary_rows", []) if row.get("tool") not in args.tools
            ]
        else:
            case_record = {"case": case, "tools": {}, "summary_rows": []}
        for tool in tools:
            print(f"[{case_index}/{len(cases)}] {case['patient_id']} {tool.name} ...", flush=True)
            started = time.perf_counter()
            try:
                payload, status, error = add_match(tool.name, tool.call(case), case), "ok", None
            except Exception as exc:
                payload, status, error = {"target_match": {"status": "error", "rank": None, "matched_by": None}, "results": []}, "error", f"{type(exc).__name__}: {exc}"
                print(f"  ERROR: {error}")
            tool_record = {"status": status, "error": error, "elapsed_seconds": round(time.perf_counter() - started, 3), **jsonable(payload)}
            case_record["tools"][tool.name] = tool_record
            row = {"patient_id": case["patient_id"], "target_omim": case["target_omim"], "target_disease": case["target_disease"],
                   "tool": tool.name, "status": status, "match_status": tool_record["target_match"]["status"],
                   "rank": tool_record["target_match"].get("rank"), "matched_by": tool_record["target_match"].get("matched_by"),
                   "returned_count": tool_record.get("returned_count", len(tool_record.get("results", []))), "error": error or ""}
            case_record["summary_rows"].append(row)
        case_record["summary_rows"].sort(key=lambda row: tool_order.get(row.get("tool", ""), 999))
        summary_rows.extend(case_record["summary_rows"])
        write_json(output_path, case_record)

    summary_rows = collect_summary_rows(args.output_dir)
    write_summary(args.output_dir, summary_rows, args.patient_ids)
    print(f"Saved: {args.output_dir / 'summary.json'}")
    return 0 if all(row["status"] == "ok" for row in summary_rows) else 1


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
