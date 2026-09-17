"""The fixed ZebraSeek execution flow.

This module intentionally keeps the old LangGraph nodes available for existing
experiments, while providing the new deterministic route described in the
requirements.  The route is split into input, initial-tools, disease
normalization, candidate research, reflection, rerank, gene annotation, and
output stages.  All external responses are retained and every assertion is
linked to a source record.  LLM calls only classify/rerank data already fetched;
they never choose the next tool call.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from langchain.schema import HumanMessage

from agent.state.state_types import (
    CandidateRecord,
    DiagnosisFormat,
    DiagnosisOutput,
    EvidenceRecord,
    LLMRankingOutput,
    PatientInput,
    ReflectionAssessment,
    ReflectionAssessmentOutput,
    SourceRecord,
    ToolResponseRecord,
)


INITIAL_TOOLS = (
    "PubCaseFinder",
    "GestaltMatcher",
    "VectorSearch",
    "PhenoBrain",
    "ZeroShot",
)
FIXED_ACTIONS = (
    "resolve_identity",
    "check_present_hpo",
    "check_absent_hpo",
    "search_case_reports",
    "search_pubmed",
    "search_contradiction",
    "expand_candidates",
)
SEARCH_ACTIONS = FIXED_ACTIONS[:-1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _jsonable(dump(mode="json"))
        except TypeError:
            return _jsonable(dump())
    if hasattr(value, "dict") and callable(value.dict):
        return _jsonable(value.dict())
    if hasattr(value, "content"):
        return _jsonable(value.content)
    return str(value)


def _stable_id(prefix: str, *parts: Any) -> str:
    text = "|".join(json.dumps(_jsonable(p), sort_keys=True, ensure_ascii=False) for p in parts)
    return f"{prefix}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_patient_input(
    present_hpo_ids: Iterable[str],
    *,
    absent_hpo_ids: Iterable[str] | None = None,
    sex: str | None = None,
    onset: str | None = None,
    image_path: str | None = None,
    patient_id: str | None = None,
) -> PatientInput:
    """Create the canonical input without a clinical text field."""
    return {
        "patient_id": str(patient_id or "unknown"),
        "present_hpo_ids": [str(x) for x in (present_hpo_ids or []) if x],
        "absent_hpo_ids": [str(x) for x in (absent_hpo_ids or []) if x],
        "sex": str(sex or "unknown"),
        "onset": str(onset or "unknown"),
        "image_path": image_path,
    }


def _source(
    *,
    source_id: str,
    access_layer: str,
    tool: str,
    database: str,
    source_type: str,
    request: dict | None = None,
    response: Any = None,
    entry_id: str | None = None,
    url: str | None = None,
    query: str | None = None,
) -> SourceRecord:
    return {
        "source_id": source_id,
        "access_layer": access_layer,
        "tool": tool,
        "database": database,
        "source_type": source_type,
        "entry_id": entry_id or "",
        "url": url or "",
        "query": query or "",
        "uri": url or "",
        "endpoint": url or "",
        "retrieved_at": utc_now(),
        "request": _jsonable(request or {}),
        "raw_response": _jsonable(response),
        "raw_response_ref": source_id,
        "raw_response_hash": _hash(response),
        "content_hash": _hash(response),
        "status": response.get("status", "ok") if isinstance(response, dict) else "ok",
    }


def _evidence(
    *, source_id: str, candidate_id: str | None, claim: str, polarity: str,
    content: Any, assertion: str = "",
) -> EvidenceRecord:
    links = []
    if candidate_id:
        links.append({"candidate_id": candidate_id, "relation": polarity, "polarity": polarity})
    return {
        "evidence_id": _stable_id("ev", source_id, candidate_id or "", claim, content),
        "source_id": source_id,
        "source_ids": [source_id],
        "candidate_id": candidate_id or "",
        "candidate_links": links,
        "claim": {"text": claim, "polarity": polarity},
        "assertion": assertion or claim,
        "polarity": polarity,
        "content": _jsonable(content) if isinstance(content, dict) else {"text": str(content)},
        "relation": polarity,
        "excerpt": str(content)[:1000],
        "structured_value": _jsonable(content) if isinstance(content, dict) else {},
        "extraction_method": "api_mapping",
        "created_at": utc_now(),
        "retrieved_at": utc_now(),
    }


def _result_list(payload: Any) -> list[dict]:
    payload = _jsonable(payload)
    if isinstance(payload, dict):
        for key in ("top5", "all", "results", "data", "items", "ans", "diseases"):
            if isinstance(payload.get(key), list):
                rows = [x for x in payload[key] if isinstance(x, dict)]
                if rows:
                    return rows
        for key in ("structured_content", "result", "data", "content"):
            nested = payload.get(key)
            if isinstance(nested, (dict, list)):
                rows = _result_list(nested)
                if rows:
                    return rows
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def _identifier(item: dict) -> tuple[str, dict]:
    ids: dict[str, str] = {}
    for key, value in item.items():
        if value is None:
            continue
        text = str(value)
        upper = key.upper()
        if "OMIM" in upper or text.upper().startswith("OMIM:"):
            ids["omim_id"] = text if text.upper().startswith("OMIM:") else f"OMIM:{text}"
        elif "ORPHA" in upper or text.upper().startswith("ORPHA:"):
            ids["orpha_id"] = text if text.upper().startswith("ORPHA:") else f"ORPHA:{text}"
        elif "MONDO" in upper or text.upper().startswith("MONDO:"):
            ids["mondo_id"] = text if text.upper().startswith("MONDO:") else f"MONDO:{text}"
    raw_id = item.get("candidate_id") or item.get("id") or item.get("disease_id") or item.get("rd_id")
    if raw_id:
        ids.setdefault("source_id", str(raw_id))
    return str(raw_id or ids.get("omim_id") or ids.get("orpha_id") or ids.get("mondo_id") or ""), ids


def _disease_name(item: dict) -> str:
    for key in ("disease_name", "disease_name_en", "omim_disease_name_en", "ENG_NAME", "name", "label", "syndrome_name"):
        if item.get(key):
            return str(item[key])
    nested = item.get("disease_info")
    if isinstance(nested, dict):
        return _disease_name(nested)
    return ""


def _normalise_results(tool: str, payload: Any) -> list[dict]:
    rows = _result_list(payload)
    normalized = []
    for index, row in enumerate(rows, 1):
        nested = row.get("disease_info") if isinstance(row.get("disease_info"), dict) else row
        raw_id, ids = _identifier({**row, **nested})
        name = _disease_name(row) or _disease_name(nested)
        if not name and not raw_id:
            continue
        normalized.append({
            "candidate_id": raw_id or _stable_id("cand", tool, name.lower()),
            "disease_name": name or raw_id,
            "omim_id": ids.get("omim_id"),
            "orpha_id": ids.get("orpha_id"),
            "mondo_id": ids.get("mondo_id"),
            "rank": int(row.get("rank") or index),
            "score": row.get("score", row.get("SCORE", row.get("similarity_score"))),
            "tool": tool,
            "raw": row,
        })
    return normalized


def _find_result_for_candidate(record: ToolResponseRecord, candidate: CandidateRecord) -> dict | None:
    """Resolve a candidate against Top5 first, then the retained raw results.

    Full responses are deliberately not normalized during the initial merge.
    They are consulted only when a later candidate (for example a one-hop
    TogoMCP discovery) needs to be linked back to an initial tool response.
    """
    candidate_ids = {
        str(value)
        for value in (candidate.get("candidate_id"), *(candidate.get("identifiers", {}) or {}).values(), *(candidate.get("normalized_ids", []) or []))
        if value
    }
    candidate_name = str(candidate.get("disease_name", "")).strip().lower()

    def matches(row: dict) -> bool:
        nested = row.get("disease_info") if isinstance(row.get("disease_info"), dict) else {}
        merged = {**row, **nested}
        raw_id, ids = _identifier(merged)
        row_ids = {str(raw_id)} if raw_id else set()
        row_ids.update(str(value) for value in ids.values() if value)
        name = (_disease_name(row) or _disease_name(nested)).strip().lower()
        return bool(candidate_ids & row_ids) or bool(candidate_name and name and candidate_name == name)

    for row in record.get("top5", []) or []:
        if isinstance(row, dict) and matches(row):
            return row

    # ``all_results`` is intentionally raw and is scanned only for a later
    # lookup.  This keeps the initial candidate union bounded by Top5.
    for raw in record.get("all_results", []) or []:
        if isinstance(raw, dict) and matches(raw):
            normalized = _normalise_results(str(record.get("tool", "")), [raw])
            return normalized[0] if normalized else raw
    return None


def _call_llm_structured(llm: Any, schema: Any, prompt: str, context: str) -> Any:
    if llm is None:
        return None
    try:
        runnable = llm.get_structured_llm(schema) if hasattr(llm, "get_structured_llm") else llm.with_structured_output(schema)
        messages = [HumanMessage(content=prompt)]
        if hasattr(llm, "invoke_with_content_filter_retry"):
            return llm.invoke_with_content_filter_retry(runnable, messages, context=context)
        return runnable.invoke(messages)
    except Exception:
        return None


def _as_model(value: Any, schema: Any) -> Any:
    if isinstance(value, schema):
        return value
    if isinstance(value, dict):
        try:
            return schema.model_validate(value)
        except AttributeError:
            try:
                return schema.parse_obj(value)
            except Exception:
                return None
        except Exception:
            return None
    return None


def _tool_error_result(exc: Exception) -> dict:
    return {"status": "error", "error": str(exc), "results": []}


def _run_initial_tool(name: str, patient: PatientInput, *, depth: int, llm: Any = None) -> Any:
    present = patient["present_hpo_ids"]
    if name == "PubCaseFinder":
        from agent.tools.pcf_api import callingPCF
        return callingPCF(present, depth, return_full=True)
    if name == "GestaltMatcher":
        if not patient.get("image_path"):
            return {"status": "skipped", "reason": "image_path is not provided", "top5": [], "all": []}
        from agent.tools.gestaltMathcher import call_gestalt_matcher_api
        return call_gestalt_matcher_api(patient["image_path"], depth, return_full=True)
    if name == "VectorSearch":
        from agent.tools.embeddingSearchWithHPO import embedding_search_with_hpo
        legacy = {"hpoDict": {h: h for h in present}, "hpoList": present, "depth": depth}
        return embedding_search_with_hpo(legacy, return_metadata=True)
    if name == "PhenoBrain":
        from agent.tools.phenobrain_api import call_phenobrain
        return call_phenobrain(present, topk=5, return_metadata=True)
    # ZeroShot receives the same context as every other reasoning prompt.  It
    # cannot fetch evidence and therefore contributes ranking results only.
    if name == "ZeroShot":
        if not llm:
            return {"status": "skipped", "reason": "LLM is not configured", "top5": [], "all": []}
        prompt = (
            "Return the five most likely rare diseases from these HPO IDs. "
            "Use only the supplied data. Present HPO IDs: " + ", ".join(present) +
            ". Absent HPO IDs: " + ", ".join(patient.get("absent_hpo_ids", [])) +
            f". Sex: {patient.get('sex','unknown')}. Onset: {patient.get('onset','unknown')}."
        )
        from agent.state.state_types import ZeroShotOutput
        result = _as_model(_call_llm_structured(llm, ZeroShotOutput, prompt, "ZeroShot"), ZeroShotOutput)
        if result is None:
            return {"status": "error", "reason": "structured LLM call failed", "top5": [], "all": []}
        rows = []
        for item in getattr(result, "ans", []) or []:
            rows.append({"disease_name": item.disease_name, "omim_id": item.OMIM_id, "rank": item.rank})
        return {"status": "ok", "top5": rows[:5], "all": rows, "raw": _jsonable(result), "request": {"prompt": prompt}}
    return {"status": "skipped", "reason": "unknown tool", "top5": [], "all": []}


def _record_initial_results(patient: PatientInput, *, depth: int, llm: Any, enabled_tools: Iterable[str] | None = None, prompt_records: list[dict] | None = None) -> tuple[list[ToolResponseRecord], list[SourceRecord], list[EvidenceRecord], dict]:
    records: list[ToolResponseRecord] = []
    sources: list[SourceRecord] = []
    evidence: list[EvidenceRecord] = []
    index: dict[str, dict] = {}

    def one(tool: str):
        started = time.perf_counter()
        payload = None
        try:
            payload = _run_initial_tool(tool, patient, depth=depth, llm=llm)
            return tool, payload
        except Exception as exc:
            payload = _tool_error_result(exc)
            return tool, payload
        finally:
            status = payload.get("status", "ok") if isinstance(payload, dict) else ("ok" if payload else "empty")
            print(
                f"[ZebraSeek] initial tool completed: {tool} "
                f"elapsed={((time.perf_counter() - started) * 1000):.2f} ms status={status}",
                flush=True,
            )

    tool_names = tuple(enabled_tools or INITIAL_TOOLS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(tool_names))) as pool:
        pairs = list(pool.map(one, tool_names))
    print(f"[ZebraSeek] initial tool barrier completed: {len(pairs)} tools", flush=True)
    for tool, payload in pairs:
        payload = _jsonable(payload)
        # Only Top5 participates in candidate union and evidence creation.
        # The complete response is retained as-is for later ID-based lookup.
        if isinstance(payload, dict):
            top_payload = payload.get("top5")
            if not isinstance(top_payload, list):
                top_payload = payload.get("all", [])[:5] if isinstance(payload.get("all"), list) else []
            all_payload = payload.get("all", []) if isinstance(payload.get("all"), list) else []
        else:
            top_payload = payload[:5] if isinstance(payload, list) else []
            all_payload = payload if isinstance(payload, list) else []
        rows = _normalise_results(tool, top_payload)
        if isinstance(payload, dict):
            status = payload.get("status") or ("ok" if rows else "not_returned")
        else:
            status = "ok" if rows else "not_returned"
        source_id = _stable_id("src", "initial", tool, patient.get("patient_id"), patient.get("present_hpo_ids"))
        request = payload.get("request", {}) if isinstance(payload, dict) else {}
        source = _source(source_id=source_id, access_layer="llm" if tool == "ZeroShot" else "direct_api", tool=tool, database=tool,
                         source_type="ranking", request=request, response=payload,
                         url=request.get("url") if isinstance(request, dict) else "",
                         query=json.dumps(request, ensure_ascii=False, sort_keys=True) if isinstance(request, dict) else "")
        sources.append(source)
        if tool == "ZeroShot" and isinstance(request, dict) and request.get("prompt") and prompt_records is not None:
            prompt_records.append({"prompt_id": _stable_id("prompt", "ZeroShot", source_id), "stage": "ZeroShot", "tool": tool, "candidate_id": None, "prompt": request["prompt"], "source_id": source_id, "created_at": utc_now()})
        record: ToolResponseRecord = {
            "tool": tool, "request": request, "response": payload,
            "top5": rows[:5], "all_results": _jsonable(all_payload),
            "retrieved_at": utc_now(), "rank_status": status,
            "response_status": status,
            "completeness": "complete" if isinstance(payload, dict) and isinstance(payload.get("all"), list) else "unknown",
            "elapsed_ms": float(payload.get("elapsed_ms", 0.0)) if isinstance(payload, dict) else 0.0,
            "raw_response_ref": source_id, "raw_response_hash": _hash(payload),
            "source_ids": [source_id],
        }
        records.append(record)
        for row in rows:
            # Different rankers often expose different identifiers (for
            # example OMIM versus a rare-disease ID).  Merge rows with a
            # shared identifier or normalized disease label into one index
            # entry while retaining each raw row in its tool record.
            cid = row["candidate_id"]
            norm_name = re.sub(r"[^a-z0-9]+", "", row.get("disease_name", "").lower())
            for existing_id, existing in index.items():
                existing_names = re.sub(r"[^a-z0-9]+", "", existing.get("disease_name", "").lower())
                existing_ids = set(existing.get("identifiers", {}).values())
                row_ids = {v for k, v in row.items() if k.endswith("_id") and v}
                if (existing_ids & row_ids) or (norm_name and norm_name == existing_names):
                    cid = existing_id
                    row["candidate_id"] = cid
                    break
            item = index.setdefault(cid, {
                "candidate_id": cid, "disease_name": row["disease_name"],
                "identifiers": {k: v for k, v in row.items() if k.endswith("_id") and k != "candidate_id" and v},
                "disease_key": cid,
                "normalized_ids": [v for k, v in row.items() if k.endswith("_id") and v and k != "candidate_id"],
                "rankings": [],
                "by_tool": {}, "top5_tools": [], "initial_candidate": False,
                "discovered_candidate": False, "discovery_evidence_ids": [],
            })
            item["by_tool"][tool] = row
            item["identifiers"].update({k: v for k, v in row.items() if k.endswith("_id") and k != "candidate_id" and v})
            item["normalized_ids"] = [v for k, v in item["identifiers"].items() if v]
            item["rankings"].append({"tool": tool, "rank": row.get("rank"), "score": row.get("score"), "status": "found", "candidate_id": cid})
            if row.get("rank", 999) <= 5:
                item["top5_tools"].append(tool)
                item["initial_candidate"] = True
            ev = _evidence(source_id=source_id, candidate_id=cid,
                           claim=f"{tool} ranked {row['disease_name']} at position {row.get('rank')}",
                           polarity="supports", content=row)
            evidence.append(ev)
    return records, sources, evidence, index


def _tool_name_match(specs: list[dict], *terms: str) -> dict | None:
    for spec in specs:
        text = f"{spec.get('name','')} {spec.get('description','')}".lower()
        if all(term.lower() in text for term in terms):
            return spec
    return None


def _exact_tool_match(specs: list[dict], *names: str) -> dict | None:
    """Select only an explicitly supported TogoMCP tool name.

    Matching descriptions by a substring caused ``pubcasefinder_rank_by_phenotypes``
    to be selected for every HPO action and even selected the usage guide for
    PubMed.  Fixed actions must never silently fall back to an unrelated tool.
    """
    by_name = {str(spec.get("name", "")): spec for spec in specs}
    for name in names:
        if name in by_name:
            return by_name[name]
    return None


def bind_togomcp_actions(specs: list[dict]) -> dict[str, dict]:
    """Bind fixed research actions to a small, explicit TogoMCP allow-list.

    The direct PubCaseFinder API is the initial ranker.  It is deliberately
    excluded from this map.  TogoMCP is used for entity/literature/knowledge
    graph verification through the currently enabled ``ncbi_esearch`` route.
    ``run_sparql`` is reserved for a future MIE-grounded binding.
    Candidate expansion is derived from the fixed search responses and makes
    no second disease-ranking call.
    """
    bindings: dict[str, dict] = {}
    candidates = {
        "resolve_identity": ("ncbi_esearch",),
        "check_present_hpo": ("ncbi_esearch",),
        "check_absent_hpo": ("ncbi_esearch",),
        "search_case_reports": ("ncbi_esearch",),
        "search_pubmed": ("ncbi_esearch",),
        "search_contradiction": ("ncbi_esearch",),
        "expand_candidates": (),
        "search_genes": ("ncbi_esearch",),
    }
    for action, options in candidates.items():
        chosen = _exact_tool_match(specs, *options) if options else None
        reason = "fixed TogoMCP allow-list match" if chosen else (
            "derived locally from fixed search responses" if action == "expand_candidates"
            else "no supported fixed TogoMCP tool advertised"
        )
        bindings[action] = {
            "action": action,
            "tool_name": chosen.get("name") if chosen else None,
            "status": "derived" if action == "expand_candidates" else ("bound" if chosen else "not_applicable"),
            "reason": reason,
            "matched_by": chosen.get("name") if chosen else "",
            "input_schema": chosen.get("input_schema", {}) if chosen else {},
        }
    return bindings


def _extract_text(result: Any) -> str:
    result = _jsonable(result)
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        parts = []
        for key in ("text", "content", "structured_content", "result", "data"):
            value = result.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (dict, list)):
                parts.append(json.dumps(value, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(result, ensure_ascii=False)


def _extract_new_candidates(result: Any, known_ids: set[str]) -> list[dict]:
    rows = _result_list(result)
    text = _extract_text(result)
    if not rows:
        # Keep extraction conservative: only identifiers accompanied by a
        # plausible disease-name field are admitted.
        for match in re.finditer(r"(?P<id>(?:OMIM|ORPHA|MONDO):[0-9]+)[^\n,;]{0,100}", text, re.I):
            rows.append({"id": match.group("id"), "name": match.group(0).strip()})
    found = []
    for row in rows:
        cid, ids = _identifier(row)
        name = _disease_name(row)
        if not cid:
            cid = next((v for v in ids.values() if v), "")
        if not cid or cid in known_ids or not name:
            continue
        found.append({"candidate_id": cid, "disease_name": name, "identifiers": ids, "raw": row})
    return found


def _compact_evidence(evidence: EvidenceRecord, limit: int = 1800) -> dict:
    """Keep LLM context bounded while raw evidence remains in State."""
    content = evidence.get("content", {})
    text = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return {
        "evidence_id": evidence.get("evidence_id"),
        "source_id": evidence.get("source_id"),
        "polarity": evidence.get("polarity"),
        "claim": evidence.get("claim"),
        "excerpt": text[:limit],
    }


class _UnavailableTogo:
    def list_tools(self):
        return []

    def call_tool(self, tool_name, arguments=None):
        raise RuntimeError("TogoMCP client is not configured")


def _sparql_literal(value: Any) -> str:
    return json.dumps(str(value or ""), ensure_ascii=False)


def _fixed_togomcp_args(action: str, candidate: CandidateRecord, patient: PatientInput, tool_name: str) -> dict:
    """Build arguments for the selected fixed tool without PCF ranking calls."""
    disease_name = str(candidate.get("disease_name", ""))
    identifiers = candidate.get("identifiers", {}) or {}
    disease_id = str(
        identifiers.get("mondo_id")
        or identifiers.get("omim_id")
        or identifiers.get("orpha_id")
        or candidate.get("candidate_id", "")
    )
    if tool_name == "ncbi_esearch":
        if action == "resolve_identity":
            database = "medgen"
            query = f'"{disease_id}"[Source ID] OR "{disease_name}"[All Fields]'
        elif action == "search_pubmed":
            database = "pubmed"
            query = f'("{disease_id}" OR "{disease_name}")'
        elif action == "search_case_reports":
            database = "pubmed"
            query = f'("{disease_id}" OR "{disease_name}") AND ("case report" OR "case series")'
        elif action == "search_contradiction":
            database = "pubmed"
            absent = " OR ".join(f'"{hpo}"' for hpo in patient.get("absent_hpo_ids", []))
            query = f'("{disease_id}" OR "{disease_name}")' + (f" AND ({absent})" if absent else "")
        elif action == "search_genes":
            database = "gene"
            query = f'"{disease_name}"[All Fields]'
        else:  # fixed MedGen route for the HPO checks
            database = "medgen"
            hpo_ids = patient.get("present_hpo_ids", []) if action == "check_present_hpo" else patient.get("absent_hpo_ids", [])
            hpo_query = " OR ".join(f'"{hpo}"[Clinical Features]' for hpo in hpo_ids)
            query = f'("{disease_id}" OR "{disease_name}")' + (f" AND ({hpo_query})" if hpo_query else "")
        return {"database": database, "query": query, "max_results": 20, "sort_by": "relevance"}

    # ``run_sparql`` is only selected when it is explicitly advertised.  The
    # query is intentionally bounded; its exact graph/predicate details are
    # supplied by the MIE setup record and remain traceable in the call.
    hpo_ids = patient.get("present_hpo_ids", []) if action == "check_present_hpo" else patient.get("absent_hpo_ids", [])
    hpo_values = ", ".join(_sparql_literal(hpo) for hpo in hpo_ids) or '""'
    query = (
        "SELECT DISTINCT ?disease ?label ?phenotype WHERE { "
        f"?disease ?label_predicate ?label . FILTER(CONTAINS(LCASE(STR(?label)), LCASE({_sparql_literal(disease_name)}))) "
        f"OPTIONAL {{ ?disease ?phenotype_predicate ?phenotype . FILTER(STR(?phenotype) IN ({hpo_values})) }} "
        "} LIMIT 20"
    )
    database = "mondo" if action in {"resolve_identity", "check_present_hpo", "check_absent_hpo", "search_contradiction"} else "ncbigene"
    return {"database": database, "sparql_query": query}


def _togo_call(client: Any, action: str, binding: dict, candidate: CandidateRecord, patient: PatientInput, source_records: list, evidence_records: list, calls: list, *, allow_not_applicable=True) -> Any:
    started = time.perf_counter()
    tool_name = binding.get("tool_name")
    cid = candidate["candidate_id"]
    call_id = _stable_id("call", action, cid)
    if not tool_name:
        source_id = _stable_id("src", "togomcp", action, cid, "na")
        source_records.append(_source(source_id=source_id, access_layer="TogoMCP", tool="", database="", source_type="not_applicable", request={"action": action}, response={"status": "not_applicable", "reason": binding.get("reason", "")}, url=getattr(client, "url", "")))
        calls.append({"tool_call_id": call_id, "action": action, "candidate_id": cid, "tool_name": None, "arguments": {}, "result": {"status": "not_applicable", "reason": binding.get("reason", "")}, "source_id": source_id, "started_at": utc_now(), "elapsed_ms": 0.0, "status": "not_applicable"})
        print(f"[ZebraSeek] TogoMCP action completed: {action} candidate={cid} elapsed=0.00 ms status=not_applicable", flush=True)
        return {"status": "not_applicable", "reason": binding.get("reason", "")}
    args = _fixed_togomcp_args(action, candidate, patient, tool_name)
    args["action"] = action
    # Keep calls compatible with the advertised input schema where available.
    schema = binding.get("input_schema") or {}
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(properties, dict) and properties:
        aliases = {"hpo_ids": ["hpo_id", "hpoList", "hpo_ids"], "query": ["term", "query"], "database": ["db", "database"], "max_results": ["retmax", "max_results", "limit"]}
        filtered = {}
        for key, value in args.items():
            if key in properties:
                filtered[key] = value
                continue
            for alias in aliases.get(key, []):
                if alias in properties:
                    filtered[alias] = value
                    break
        args = filtered
    try:
        result = client.call_tool(tool_name, args)
        status = "error" if isinstance(result, dict) and result.get("is_error") else "ok"
        error = ""
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
        status = "error"
        error = str(exc)
    if status == "error" and isinstance(result, dict):
        result.setdefault("status", "error")
    source_id = _stable_id("src", "togomcp", action, cid)
    source_records.append(_source(source_id=source_id, access_layer="TogoMCP", tool=tool_name, database="TogoMCP", source_type=action, request=args, response=result, entry_id=str(result.get("entry_id") or result.get("pmid") or result.get("id") or "") if isinstance(result, dict) else "", url=(result.get("url") if isinstance(result, dict) else "") or getattr(client, "url", ""), query=json.dumps(args, ensure_ascii=False, sort_keys=True)))
    calls.append({"tool_call_id": call_id, "action": action, "candidate_id": cid, "tool_name": tool_name, "arguments": args, "result": _jsonable(result), "source_id": source_id, "started_at": utc_now(), "elapsed_ms": result.get("elapsed_ms", 0) if isinstance(result, dict) else 0.0, "status": status, "error": error})
    if status == "ok":
        if action == "search_contradiction":
            polarity = "contradicts"
        elif action == "check_absent_hpo":
            # An absent annotation is not a contradiction unless the returned
            # payload explicitly says so; interpretation is deferred to
            # Reflection and the raw response remains attached.
            polarity = "unknown"
        else:
            polarity = "supports"
        evidence_records.append(_evidence(source_id=source_id, candidate_id=cid, claim=f"TogoMCP {action} result for {candidate.get('disease_name','')}", polarity=polarity, content=result))
    print(
        f"[ZebraSeek] TogoMCP action completed: {action} candidate={cid} "
        f"elapsed={((time.perf_counter() - started) * 1000):.2f} ms status={status}",
        flush=True,
    )
    return result


def _search_candidate(candidate: CandidateRecord, patient: PatientInput, client: Any, bindings: dict, sources: list, evidence: list, calls: list, *, hop: int) -> tuple[CandidateRecord, list[dict]]:
    # Identity is deliberately first. The four independent fixed searches are
    # then launched concurrently, followed by contradiction. Candidate
    # expansion is derived from those responses; it never calls the initial
    # PubCaseFinder ranker again.
    identity_result = _togo_call(client, "resolve_identity", bindings.get("resolve_identity", {}), candidate, patient, sources, evidence, calls)
    identity_rows = _result_list(identity_result)
    identity_payload = identity_rows[0] if identity_rows else (_jsonable(identity_result) if isinstance(_jsonable(identity_result), dict) else {})
    preferred = (
        identity_payload.get("preferred_id")
        or identity_payload.get("mondo_id")
        or identity_payload.get("omim_id")
        if isinstance(identity_payload, dict)
        else None
    )
    if preferred:
        candidate.setdefault("identifiers", {})["preferred_id"] = str(preferred)
        if str(preferred) not in candidate.setdefault("normalized_ids", []):
            candidate["normalized_ids"].append(str(preferred))
    independent = ["check_present_hpo", "check_absent_hpo", "search_case_reports", "search_pubmed"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(independent)) as pool:
        futures = [pool.submit(_togo_call, client, action, bindings.get(action, {}), candidate, patient, sources, evidence, calls) for action in independent]
        independent_results = [f.result() for f in futures]
    _togo_call(client, "search_contradiction", bindings.get("search_contradiction", {}), candidate, patient, sources, evidence, calls)
    discovered: list[dict] = []
    if hop == 0:
        known_ids = {candidate["candidate_id"]}
        # Extract only diseases explicitly returned by the fixed case-report
        # and PubMed searches. This is a local transformation, not a second
        # phenotype-ranking query.
        for result in independent_results:
            discovered.extend(_extract_new_candidates(result, known_ids))
            for row in discovered:
                known_ids.add(row.get("candidate_id", ""))
        source_ids = [
            call.get("source_id") for call in calls
            if call.get("candidate_id") == candidate["candidate_id"]
            and call.get("action") in {"search_case_reports", "search_pubmed"}
            and call.get("source_id")
        ]
        expansion_evidence_ids = [
            e.get("evidence_id") for e in evidence
            if e.get("source_id") in source_ids and e.get("evidence_id")
        ]
        for row in discovered:
            row["discovery_evidence_ids"] = expansion_evidence_ids
            row["discovery_source_ids"] = source_ids
            row["discovered_from_candidate_id"] = candidate["candidate_id"]
        calls.append({
            "tool_call_id": _stable_id("call", "expand_candidates", candidate["candidate_id"]),
            "action": "expand_candidates",
            "candidate_id": candidate["candidate_id"],
            "tool_name": None,
            "arguments": {"derived_from": ["search_case_reports", "search_pubmed"]},
            "result": {"status": "derived_from_fixed_searches", "count": len(discovered)},
            "source_id": "",
            "started_at": utc_now(),
            "elapsed_ms": 0.0,
            "status": "derived",
        })
    candidate["searches"] = [x for x in FIXED_ACTIONS if x != "expand_candidates" or hop == 0]
    candidate["search_status"] = "complete"
    return candidate, discovered


def _reflection_for_candidate(candidate: CandidateRecord, evidence: list[EvidenceRecord], llm: Any, patient: PatientInput, prompt_records: list[dict] | None = None) -> ReflectionAssessment:
    cid = candidate["candidate_id"]
    relevant = [e for e in evidence if any(link.get("candidate_id") == cid for link in e.get("candidate_links", []))]
    packet = [_compact_evidence(e) for e in relevant]
    prompt = (
        "Assess this candidate using only the evidence packet. Return judgment exactly as "
        "correct, incorrect, or uncertain. Lack of an HPO annotation is unknown, not contradiction.\n"
        + json.dumps({"patient": patient, "candidate": candidate, "evidence": packet}, ensure_ascii=False)
    )
    prompt_id = _stable_id("prompt", "Reflection", cid)
    if prompt_records is not None:
        prompt_records.append({"prompt_id": prompt_id, "stage": "Reflection", "tool": "LLM", "candidate_id": cid, "prompt": prompt, "evidence_ids": [e.get("evidence_id") for e in relevant], "created_at": utc_now()})
    out = _as_model(_call_llm_structured(llm, ReflectionAssessmentOutput, prompt, "Reflection"), ReflectionAssessmentOutput)
    valid_ids = {e.get("evidence_id") for e in relevant}
    if out is None:
        judgment, rationale = "uncertain", "LLM unavailable or failed"
        supports = [e.get("evidence_id") for e in relevant if e.get("polarity") == "supports"]
        contradicts = [e.get("evidence_id") for e in relevant if e.get("polarity") == "contradicts"]
        unknown = [e.get("evidence_id") for e in relevant if e.get("polarity") == "unknown"]
    else:
        judgment = out.judgment
        rationale = out.rationale
        supports = [x for x in out.supporting_evidence_ids if x in valid_ids]
        contradicts = [x for x in out.contradicting_evidence_ids if x in valid_ids]
        unknown = [x for x in out.unknown_evidence_ids if x in valid_ids]
    return {"reflection_id": _stable_id("reflection", cid), "candidate_id": cid, "disease_name": candidate.get("disease_name", ""), "judgment": judgment, "rationale": rationale, "analysis": rationale, "patient_summary": f"Present HPO: {', '.join(patient.get('present_hpo_ids', []))}; absent HPO: {', '.join(patient.get('absent_hpo_ids', [])) or 'none supplied'}.", "supporting_evidence_ids": supports, "contradicting_evidence_ids": contradicts, "unknown_evidence_ids": unknown, "source_ids": [e.get("source_id") for e in relevant if e.get("source_id")], "model": getattr(getattr(llm, "llm", None), "model_name", "") if llm else "", "prompt_ref": prompt_id, "created_at": utc_now()}


def _tool_average(candidates: list[CandidateRecord], records: list[ToolResponseRecord]) -> dict[str, float]:
    active = [r for r in records if r.get("rank_status") not in {"skipped", "error", "not_applicable"}]
    scores = {c["candidate_id"]: [] for c in candidates}
    for record in active:
        rows = record.get("all_results", record.get("top5", [])) or []
        n = max(len(rows), 1)
        by_id = {str(row.get("candidate_id")): row for row in rows}
        for candidate in candidates:
            row = by_id.get(candidate["candidate_id"])
            if row is None:
                scores[candidate["candidate_id"]].append(0.0)
            else:
                rank = max(1, int(row.get("rank", n)))
                rank_score = 1.0 if n <= 1 else max(0.0, 1.0 - (rank - 1) / (n - 1))
                scores[candidate["candidate_id"]].append(rank_score)
    return {cid: (sum(vals) / len(vals) if vals else 0.0) for cid, vals in scores.items()}


def _rerank(candidates: list[CandidateRecord], reflections: dict[str, ReflectionAssessment], evidence: list[EvidenceRecord], records: list[ToolResponseRecord], llm: Any, mode: str, prompt_records: list[dict] | None = None) -> tuple[list[dict], str]:
    if not candidates:
        return [], mode
    average = _tool_average(candidates, records)
    used_mode = mode
    llm_items = None
    if mode == "llm" and llm:
        compact = []
        for c in candidates:
            cid = c["candidate_id"]
            compact.append({"candidate_id": cid, "disease_name": c.get("disease_name"), "tool_scores": average.get(cid, 0), "reflection": reflections.get(cid), "evidence": [_compact_evidence(e) for e in evidence if any(l.get("candidate_id") == cid for l in e.get("candidate_links", []))]})
        prompt = "Rank candidates. Do not invent evidence IDs.\n" + json.dumps(compact, ensure_ascii=False)
        if prompt_records is not None:
            prompt_records.append({"prompt_id": _stable_id("prompt", "Rerank"), "stage": "Rerank", "tool": "LLM", "candidate_id": None, "prompt": prompt, "evidence_ids": [e.get("evidence_id") for e in evidence], "created_at": utc_now()})
        out = _as_model(_call_llm_structured(llm, LLMRankingOutput, prompt, "Rerank"), LLMRankingOutput)
        if out is not None and out.candidates:
            llm_items = {item.candidate_id: item for item in out.candidates}
        else:
            used_mode = "tool_average_fallback"
    elif mode != "tool_average":
        used_mode = "tool_average"
    rows = []
    for c in candidates:
        cid = c["candidate_id"]
        item = llm_items.get(cid) if llm_items else None
        valid_evidence = {e.get("evidence_id") for e in evidence if any(l.get("candidate_id") == cid for l in e.get("candidate_links", []))}
        score = (len(candidates) - (item.rank if item else len(candidates))) / max(len(candidates), 1) if item else average.get(cid, 0.0)
        support_ids = item.supporting_evidence_ids if item else reflections.get(cid, {}).get("supporting_evidence_ids", [])
        contradict_ids = item.contradicting_evidence_ids if item else reflections.get(cid, {}).get("contradicting_evidence_ids", [])
        rows.append({"candidate_id": cid, "disease_name": c.get("disease_name", ""), "identifiers": c.get("identifiers", {}), "normalized_ids": c.get("normalized_ids", []), "tool_rankings": c.get("tool_rankings", []), "ranking_score": float(score), "ranking_mode": used_mode, "rationale": item.rationale if item else "rank-normalized arithmetic mean of initial tool results", "supporting_evidence_ids": [eid for eid in support_ids if eid in valid_evidence], "contradicting_evidence_ids": [eid for eid in contradict_ids if eid in valid_evidence], "reflection": reflections.get(cid)})
    rows.sort(key=lambda x: (-x["ranking_score"], x["disease_name"], x["candidate_id"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows, used_mode


def _gene_annotations(candidate: dict, client: Any, binding: dict, patient: PatientInput, sources: list, evidence: list, calls: list) -> list[dict]:
    c: CandidateRecord = candidate
    result = _togo_call(client, "search_genes", binding, c, patient, sources, evidence, calls)
    rows = _result_list(result)
    annotations = []
    for row in rows:
        gene = row.get("gene_symbol") or row.get("gene") or row.get("symbol") or row.get("label")
        if not gene:
            continue
        source_id = _stable_id("src", "gene", c["candidate_id"], gene)
        ev = _evidence(source_id=source_id, candidate_id=c["candidate_id"], claim=f"{gene} is reported as a causal candidate for {c.get('disease_name','')}", polarity="supports", content=row)
        evidence.append(ev)
        sources.append(_source(source_id=source_id, access_layer="TogoMCP", tool=binding.get("tool_name", ""), database="TogoMCP", source_type="gene_annotation", request={"candidate_id": c["candidate_id"]}, response=row, entry_id=str(row.get("gene_id", ""))))
        annotation_id = _stable_id("gene", c["candidate_id"], gene)
        annotations.append({"gene_annotation_id": annotation_id, "annotation_id": annotation_id, "candidate_id": c["candidate_id"], "disease_candidate_id": c["candidate_id"], "disease_id": c["candidate_id"], "gene_symbol": str(gene), "gene_id": row.get("gene_id") or row.get("id"), "relation": row.get("relation", "known_causal_candidate"), "evidence_ids": [ev["evidence_id"]], "source_ids": [source_id], "retrieved_at": utc_now(), "status": "ok"})
    return annotations


def create_zebraseek_context(
    patient: PatientInput,
    *,
    llm: Any = None,
    togo_client: Any = None,
    ranking_mode: str = "llm",
    depth: int = 1,
    use_togomcp: bool = True,
    use_phenobrain: bool = True,
    max_final_candidates: int = 5,
) -> dict[str, Any]:
    """Create the mutable context passed between the LangGraph stage nodes."""
    if ranking_mode not in {"llm", "tool_average"}:
        raise ValueError("ranking_mode must be 'llm' or 'tool_average'")
    canonical = build_patient_input(
        patient.get("present_hpo_ids", []),
        absent_hpo_ids=patient.get("absent_hpo_ids", []),
        sex=patient.get("sex"),
        onset=patient.get("onset"),
        image_path=patient.get("image_path"),
        patient_id=patient.get("patient_id"),
    )
    if not canonical["present_hpo_ids"]:
        raise ValueError("present_hpo_ids must contain at least one HPO ID")
    return {
        "patient": canonical,
        "patient_input": canonical,
        "llm": llm,
        "togo_client": togo_client,
        "ranking_mode": ranking_mode,
        "depth": depth,
        "use_togomcp": use_togomcp,
        "use_phenobrain": use_phenobrain,
        "max_final_candidates": max_final_candidates,
        "run_id": str(uuid.uuid4()),
        "started_at": utc_now(),
        "prompt_records": [],
        "records": [],
        "sources": [],
        "evidence": [],
        "normalization_records": [],
        "index": {},
        "initial": [],
        "bindings": {a: {"action": a, "tool_name": None, "status": "not_applicable", "reason": "TogoMCP disabled"} for a in (*FIXED_ACTIONS, "search_genes")},
        "calls": [],
        "discovered": [],
        "new_candidates": [],
        "reflections": {},
        "ranked": [],
        "used_mode": ranking_mode,
        "final_rows": [],
    }


def stage_initial_tools(ctx: dict[str, Any]) -> dict[str, Any]:
    """Run all initial rankers and merge only their Top5 results."""
    patient = ctx["patient"]
    records, sources, evidence, index = _record_initial_results(
        patient,
        depth=ctx["depth"],
        llm=ctx["llm"],
        enabled_tools=tuple(t for t in INITIAL_TOOLS if ctx["use_phenobrain"] or t != "PhenoBrain"),
        prompt_records=ctx["prompt_records"],
    )
    for record in records:
        record["run_id"] = ctx["run_id"]
        record["request_context"] = dict(patient)
    initial = []
    for item in index.values():
        if item.get("initial_candidate"):
            initial.append({
                "candidate_id": item["candidate_id"],
                "disease_name": item["disease_name"],
                "identifiers": item.get("identifiers", {}),
                "normalized_ids": item.get("normalized_ids", []),
                "discovery": {"kind": "initial_tool_top5", "tools": item.get("top5_tools", [])},
                "discovery_source_ids": [],
                "discovery_evidence_ids": [],
                "ranking": item.get("by_tool", {}),
                "tool_rankings": item.get("rankings", []),
                "evidence_ids": [],
                "search_status": "pending",
                "searches": [],
                "reflection_id": None,
                "gene_annotation_ids": [],
            })
    ctx.update({"records": records, "sources": sources, "evidence": evidence, "index": index, "initial": initial})
    print(f"[ZebraSeek] InitialToolsNode completed: {len(records)} tools, {len(initial)} Top5 candidates", flush=True)
    return ctx


_OMIM_LABELS: dict[str, str] | None = None


def _omim_label(omim_id: Any) -> str | None:
    """Return the canonical OMIM label used by the legacy normalizer.

    The old ``NormalizePCFNode`` and ``NormalizeGestaltMatcherNode`` resolve
    labels from ``omim_mapping.json`` after extracting the numeric OMIM ID.
    Keep that same rule here, but load the data lazily so a tool-only run does
    not require an Azure embedding client just to normalize an ID.
    """
    global _OMIM_LABELS
    if not omim_id:
        return None
    if _OMIM_LABELS is None:
        _OMIM_LABELS = {}
        mapping_path = __file__
        try:
            from pathlib import Path
            path = Path(mapping_path).parent.parent / "data" / "DataForOmimMapping" / "omim_mapping.json"
            with path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
            for key, value in raw.items():
                match = re.search(r"\d+", str(key))
                if match and value:
                    _OMIM_LABELS[match.group(0)] = str(value)
        except (OSError, ValueError, TypeError):
            # A missing local mapping must not make the external ranking
            # route fail.  The source supplied label remains usable.
            _OMIM_LABELS = {}
    match = re.search(r"\d+", str(omim_id))
    return _OMIM_LABELS.get(match.group(0)) if match else None


def _normalise_candidate_record(candidate: CandidateRecord) -> tuple[CandidateRecord, dict]:
    """Apply the legacy disease-name/identifier normalization rules.

    OMIM identifiers are canonicalized first and, when present in the local
    OMIM mapping, their official label replaces the tool supplied label.  A
    candidate without an OMIM identifier keeps its source label; this mirrors
    the legacy nodes, which only rewrite labels when an identifier can anchor
    the mapping.
    """
    before = {
        "candidate_id": candidate.get("candidate_id", ""),
        "disease_name": candidate.get("disease_name", ""),
        "identifiers": dict(candidate.get("identifiers", {}) or {}),
    }
    merged = dict(candidate)
    merged.update(candidate.get("identifiers", {}) or {})
    raw_id, identifiers = _identifier(merged)
    if identifiers:
        candidate["identifiers"] = {**(candidate.get("identifiers", {}) or {}), **identifiers}
    normalized_ids = [
        str(value)
        for key, value in candidate.get("identifiers", {}).items()
        if value and key != "source_id"
    ]
    candidate["normalized_ids"] = normalized_ids
    name = re.sub(r"\s+", " ", str(candidate.get("disease_name", "") or "")).strip().strip("*")
    omim_id = candidate.get("identifiers", {}).get("omim_id")
    canonical_label = _omim_label(omim_id)
    if canonical_label:
        name = canonical_label
    candidate["disease_name"] = name or str(candidate.get("candidate_id", ""))
    # Keep the stable source candidate ID unless no ID was supplied at all.
    if not candidate.get("candidate_id") and raw_id:
        candidate["candidate_id"] = raw_id
    after = {
        "candidate_id": candidate.get("candidate_id", ""),
        "disease_name": candidate.get("disease_name", ""),
        "identifiers": dict(candidate.get("identifiers", {}) or {}),
    }
    return candidate, {"before": before, "after": after, "method": "omim_id_mapping" if canonical_label else "identifier_normalization"}


def stage_disease_normalization(ctx: dict[str, Any]) -> dict[str, Any]:
    """Normalize merged disease candidates before external verification.

    This is the explicit counterpart of the legacy normalization nodes.  It
    runs after the five initial rankers have returned and before any candidate
    is sent to TogoMCP, so logs and LangGraph state show the same boundary as
    the original main workflow.
    """
    started = time.perf_counter()
    records = ctx.get("records", [])
    normalization_records = []
    for item in ctx.get("index", {}).values():
        _, trace = _normalise_candidate_record(item)
        normalization_records.append({"scope": "disease_ranking_index", **trace})
        for row in (item.get("by_tool", {}) or {}).values():
            if isinstance(row, dict):
                row_candidate = {"candidate_id": item.get("candidate_id", ""), "disease_name": row.get("disease_name", ""), "identifiers": {k: v for k, v in row.items() if k.endswith("_id") and v}}
                _, row_trace = _normalise_candidate_record(row_candidate)
                row["disease_name"] = row_candidate["disease_name"]
                row["omim_id"] = row_candidate.get("identifiers", {}).get("omim_id") or row.get("omim_id")
                normalization_records.append({"scope": f"tool:{item.get('candidate_id','')}", **row_trace})
    for candidate in ctx.get("initial", []):
        _, trace = _normalise_candidate_record(candidate)
        normalization_records.append({"scope": "candidate_pool", **trace})
    ctx["normalization_records"] = normalization_records
    print(
        f"[ZebraSeek] DiseaseNormalizeNode completed: {len(ctx.get('initial', []))} candidates, "
        f"{len(normalization_records)} records elapsed={((time.perf_counter() - started) * 1000):.2f} ms",
        flush=True,
    )
    return ctx


def stage_candidate_research(ctx: dict[str, Any]) -> dict[str, Any]:
    """Run fixed TogoMCP searches and one-hop candidate expansion."""
    patient = ctx["patient"]
    initial = ctx["initial"]
    sources = ctx["sources"]
    evidence = ctx["evidence"]
    calls = ctx["calls"]
    togo_client = ctx["togo_client"]
    bindings = ctx["bindings"]
    discovered: list[dict] = []
    if togo_client is None and ctx["use_togomcp"]:
        try:
            from agent.tools.MCP.togomcp_client import TogoMCPClient
            togo_client = TogoMCPClient()
        except Exception:
            togo_client = _UnavailableTogo()
    else:
        togo_client = togo_client or _UnavailableTogo()
    ctx["togo_client"] = togo_client

    if ctx["use_togomcp"] and initial:
        togo_started = time.perf_counter()
        try:
            specs = togo_client.list_tools()
            print(f"[ZebraSeek] TogoMCP tools/list completed: {len(specs)} tools elapsed={((time.perf_counter() - togo_started) * 1000):.2f} ms", flush=True)
            bindings = bind_togomcp_actions(specs)
            guide_spec = _exact_tool_match(specs, "TogoMCP_Usage_Guide")
            if guide_spec:
                try:
                    setup_result = togo_client.call_tool(guide_spec["name"], {})
                    setup_status = "ok"
                except Exception as exc:
                    setup_result = {"status": "error", "error": str(exc)}
                    setup_status = "error"
                setup_source_id = _stable_id("src", "togomcp", "usage_guide", patient.get("patient_id"))
                sources.append(_source(source_id=setup_source_id, access_layer="TogoMCP", tool=guide_spec["name"], database="TogoMCP", source_type="usage_guide", request={}, response=setup_result, url=getattr(togo_client, "url", "")))
                calls.append({"tool_call_id": _stable_id("call", "usage_guide"), "action": "usage_guide", "candidate_id": None, "tool_name": guide_spec["name"], "arguments": {}, "result": _jsonable(setup_result), "source_id": setup_source_id, "started_at": utc_now(), "elapsed_ms": setup_result.get("elapsed_ms", 0) if isinstance(setup_result, dict) else 0, "status": setup_status})

            mie_spec = _exact_tool_match(specs, "get_MIE_file")
            if mie_spec and any(binding.get("tool_name") == "run_sparql" for binding in bindings.values()):
                for database in ("mondo", "ncbigene"):
                    mie_args = {"database": database}
                    try:
                        setup_result = togo_client.call_tool(mie_spec["name"], mie_args)
                        setup_status = "ok"
                    except Exception as exc:
                        setup_result = {"status": "error", "error": str(exc)}
                        setup_status = "error"
                    setup_source_id = _stable_id("src", "togomcp", "mie", patient.get("patient_id"), database)
                    sources.append(_source(source_id=setup_source_id, access_layer="TogoMCP", tool=mie_spec["name"], database=database, source_type="mie", request=mie_args, response=setup_result, url=getattr(togo_client, "url", "")))
                    calls.append({"tool_call_id": _stable_id("call", "mie", database), "action": "mie", "candidate_id": None, "tool_name": mie_spec["name"], "arguments": mie_args, "result": _jsonable(setup_result), "source_id": setup_source_id, "started_at": utc_now(), "elapsed_ms": setup_result.get("elapsed_ms", 0) if isinstance(setup_result, dict) else 0, "status": setup_status})
        except Exception as exc:
            print(f"[ZebraSeek] TogoMCP tools/list completed: error elapsed={((time.perf_counter() - togo_started) * 1000):.2f} ms", flush=True)
            bindings = {a: {"action": a, "tool_name": None, "status": "not_applicable", "reason": f"tools/list failed: {exc}"} for a in (*FIXED_ACTIONS, "search_genes")}
            sources.append(_source(source_id=_stable_id("src", "togomcp", "tools_list"), access_layer="TogoMCP", tool="tools/list", database="TogoMCP", source_type="error", response={"status": "error", "error": str(exc)}))

        def search_initial(candidate: CandidateRecord):
            started = time.perf_counter()
            try:
                return _search_candidate(candidate, patient, togo_client, bindings, sources, evidence, calls, hop=0)
            finally:
                print(f"[ZebraSeek] CandidateResearchNode candidate completed: {candidate.get('candidate_id')} elapsed={((time.perf_counter() - started) * 1000):.2f} ms", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(8, len(initial)))) as pool:
            futures = [pool.submit(search_initial, c) for c in initial]
            for future in futures:
                _, new_rows = future.result()
                discovered.extend(new_rows)
        print(f"[ZebraSeek] CandidateResearchNode initial barrier completed: {len(initial)} candidates", flush=True)

    seen = {c["candidate_id"] for c in initial}
    for row in discovered:
        cid = row["candidate_id"]
        if cid in seen:
            continue
        seen.add(cid)
        discovered_candidate: CandidateRecord = {
            "candidate_id": cid,
            "disease_name": row["disease_name"],
            "identifiers": row.get("identifiers", {}),
            "normalized_ids": list(row.get("identifiers", {}).values()),
            "discovery": {"kind": "one_hop_togomcp", "raw": row.get("raw", {}), "parent_candidate_id": row.get("discovered_from_candidate_id")},
            "discovery_source_ids": row.get("discovery_source_ids", []),
            "discovery_evidence_ids": row.get("discovery_evidence_ids", []),
            "ranking": {}, "tool_rankings": [], "evidence_ids": [],
            "search_status": "pending", "searches": [], "reflection_id": None, "gene_annotation_ids": [],
        }
        _, normalization_trace = _normalise_candidate_record(discovered_candidate)
        ctx.setdefault("normalization_records", []).append({"scope": "discovered_candidate", **normalization_trace})
        initial.append(discovered_candidate)
        for evidence_item in evidence:
            if evidence_item.get("evidence_id") in row.get("discovery_evidence_ids", []):
                evidence_item.setdefault("candidate_links", []).append({"candidate_id": cid, "candidate_label": row["disease_name"], "relation": "discovery", "polarity": "supports"})

    new_candidates = [c for c in initial if c.get("discovery", {}).get("kind") == "one_hop_togomcp"]
    for candidate in initial:
        candidate["tool_rankings"] = []
        candidate["ranking"] = {}
        for record in ctx["records"]:
            row = _find_result_for_candidate(record, candidate)
            if row is None:
                record_status = record.get("response_status", record.get("rank_status", "ok"))
                ranking_status = "skipped" if record_status == "skipped" else ("error" if record_status == "error" else "not_returned")
                ranking = {"tool": record.get("tool"), "rank": None, "score": None, "status": ranking_status, "run_id": ctx["run_id"]}
            else:
                ranking = {"tool": record.get("tool"), "rank": row.get("rank"), "score": row.get("score"), "status": "found", "run_id": ctx["run_id"]}
                candidate["ranking"][record.get("tool")] = row
            candidate["tool_rankings"].append(ranking)
        if candidate["candidate_id"] not in ctx["index"]:
            ctx["index"][candidate["candidate_id"]] = {
                "candidate_id": candidate["candidate_id"], "disease_key": candidate["candidate_id"], "disease_name": candidate.get("disease_name", ""), "identifiers": candidate.get("identifiers", {}), "normalized_ids": candidate.get("normalized_ids", []), "rankings": list(candidate["tool_rankings"]), "by_tool": {}, "top5_tools": [], "initial_candidate": False, "discovered_candidate": True, "discovery_evidence_ids": candidate.get("discovery_evidence_ids", []),
            }

    if ctx["use_togomcp"] and new_candidates:
        def search_discovered(candidate: CandidateRecord):
            started = time.perf_counter()
            try:
                return _search_candidate(candidate, patient, togo_client, bindings, sources, evidence, calls, hop=1)
            finally:
                print(f"[ZebraSeek] CandidateResearchNode discovered candidate completed: {candidate.get('candidate_id')} elapsed={((time.perf_counter() - started) * 1000):.2f} ms", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(8, len(new_candidates)))) as pool:
            list(pool.map(search_discovered, new_candidates))
        print(f"[ZebraSeek] CandidateResearchNode discovered barrier completed: {len(new_candidates)} candidates", flush=True)

    ctx.update({"initial": initial, "bindings": bindings, "discovered": discovered, "new_candidates": new_candidates})
    print(f"[ZebraSeek] CandidateResearchNode completed: {len(initial)} total candidates", flush=True)
    return ctx


def stage_reflection(ctx: dict[str, Any]) -> dict[str, Any]:
    """Assess every candidate after its fixed evidence route is complete."""
    initial = ctx["initial"]
    evidence = ctx["evidence"]
    llm = ctx["llm"]
    patient = ctx["patient"]
    reflections: dict[str, ReflectionAssessment] = {}
    started_all = time.perf_counter()

    def reflect_one(candidate: CandidateRecord):
        started = time.perf_counter()
        try:
            return _reflection_for_candidate(candidate, evidence, llm, patient, ctx["prompt_records"])
        finally:
            print(f"[ZebraSeek] ReflectionNode candidate completed: {candidate.get('candidate_id')} elapsed={((time.perf_counter() - started) * 1000):.2f} ms", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(8, len(initial)))) as pool:
        future_to_candidate = {pool.submit(reflect_one, c): c for c in initial}
        for future in concurrent.futures.as_completed(future_to_candidate):
            candidate = future_to_candidate[future]
            reflections[candidate["candidate_id"]] = future.result()
            candidate["reflection_id"] = reflections[candidate["candidate_id"]]["reflection_id"]
    ctx["reflections"] = reflections
    print(f"[ZebraSeek] ReflectionNode completed: {len(reflections)} candidates elapsed={((time.perf_counter() - started_all) * 1000):.2f} ms", flush=True)
    return ctx


def stage_rerank(ctx: dict[str, Any]) -> dict[str, Any]:
    """Rank candidates using the configured LLM or tool-average mode."""
    started = time.perf_counter()
    ranked, used_mode = _rerank(ctx["initial"], ctx["reflections"], ctx["evidence"], ctx["records"], ctx["llm"], ctx["ranking_mode"], ctx["prompt_records"])
    final_rows = ranked[:ctx["max_final_candidates"]]
    evidence_by_id = {e.get("evidence_id"): e for e in ctx["evidence"]}
    for row in final_rows:
        row["supporting_evidence"] = [evidence_by_id[eid] for eid in row.get("supporting_evidence_ids", []) if eid in evidence_by_id]
        row["contradicting_evidence"] = [evidence_by_id[eid] for eid in row.get("contradicting_evidence_ids", []) if eid in evidence_by_id]
        reflection = row.get("reflection") or {}
        row["judgment"] = reflection.get("judgment", "uncertain")
        row["unknown_evidence_ids"] = reflection.get("unknown_evidence_ids", [])
        row["normalized_ids"] = list(row.get("identifiers", {}).values())
        row["known_causal_gene_annotation_ids"] = []
    ctx.update({"ranked": ranked, "used_mode": used_mode, "final_rows": final_rows})
    print(f"[ZebraSeek] RerankNode completed: {len(ranked)} candidates, final={len(final_rows)} mode={used_mode} elapsed={((time.perf_counter() - started) * 1000):.2f} ms", flush=True)
    return ctx


def stage_genes(ctx: dict[str, Any]) -> dict[str, Any]:
    """Retrieve known causal candidate genes for the final ranked diseases."""
    final_rows = ctx["final_rows"]
    if ctx["use_togomcp"] and final_rows:
        def genes_one(row: dict):
            started = time.perf_counter()
            try:
                return _gene_annotations(row, ctx["togo_client"], ctx["bindings"].get("search_genes", {}), ctx["patient"], ctx["sources"], ctx["evidence"], ctx["calls"])
            finally:
                print(f"[ZebraSeek] GeneAnnotationNode candidate completed: {row.get('candidate_id')} elapsed={((time.perf_counter() - started) * 1000):.2f} ms", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(8, len(final_rows)))) as pool:
            futures = [pool.submit(genes_one, row) for row in final_rows]
            for row, future in zip(final_rows, futures):
                row["genes"] = future.result()
                row["known_causal_gene_annotation_ids"] = [g["gene_annotation_id"] for g in row["genes"]]
                source_candidate = next((c for c in ctx["initial"] if c["candidate_id"] == row["candidate_id"]), None)
                if source_candidate is not None:
                    source_candidate["gene_annotation_ids"] = [g["gene_annotation_id"] for g in row["genes"]]
    else:
        for row in final_rows:
            row["genes"] = []
    print(f"[ZebraSeek] GeneAnnotationNode completed: {len(final_rows)} candidates", flush=True)
    return ctx


def finalize_zebraseek_context(ctx: dict[str, Any]) -> dict:
    """Build the traceable output from the staged context."""
    calls = ctx["calls"]
    sources = ctx["sources"]
    evidence = ctx["evidence"]
    initial = ctx["initial"]
    discovered = ctx["discovered"]
    final_rows = ctx["final_rows"]
    action_order = {action: index for index, action in enumerate(("usage_guide", "mie", *FIXED_ACTIONS, "search_genes"))}
    calls.sort(key=lambda call: (str(call.get("candidate_id") or ""), action_order.get(call.get("action"), 999), str(call.get("tool_call_id", ""))))
    for call in calls:
        call["call_id"] = call.get("tool_call_id", "")
        call["source_ids"] = [call.get("source_id")] if call.get("source_id") else []
        call["result_ref"] = call.get("source_id", "")
        call["database"] = "TogoMCP"
        call["query"] = json.dumps(call.get("arguments", {}), ensure_ascii=False, sort_keys=True)
    sources.sort(key=lambda source: str(source.get("source_id", "")))
    evidence.sort(key=lambda item: str(item.get("evidence_id", "")))
    sessions = []
    for candidate in initial:
        cid = candidate["candidate_id"]
        candidate_calls = [call for call in calls if call.get("candidate_id") == cid]
        candidate_evidence = [ev.get("evidence_id") for ev in evidence if any(link.get("candidate_id") == cid for link in ev.get("candidate_links", []))]
        candidate["evidence_ids"] = candidate_evidence
        hop = 1 if candidate.get("discovery", {}).get("kind") == "one_hop_togomcp" else 0
        sessions.append({
            "session_id": _stable_id("session", cid),
            "candidate_id": cid,
            "candidate_name": candidate.get("disease_name", ""),
            "hop": hop,
            "expansion_depth": hop,
            "fixed_actions": [a for a in FIXED_ACTIONS if hop == 0 or a != "expand_candidates"],
            "planned_actions": [a for a in FIXED_ACTIONS if hop == 0 or a != "expand_candidates"],
            "completed_actions": [call.get("action") for call in candidate_calls],
            "action_bindings": ctx["bindings"],
            "tool_call_ids": [call.get("tool_call_id") for call in candidate_calls],
            "action_records": [call.get("tool_call_id") for call in candidate_calls],
            "evidence_ids": candidate_evidence,
            "discovered_candidate_ids": [row.get("candidate_id") for row in discovered if row.get("discovered_from_candidate_id") == cid],
            "status": "completed" if ctx["use_togomcp"] and candidate_calls else "not_run",
            "started_at": ctx["started_at"],
            "finished_at": utc_now(),
        })
    diagnoses = [DiagnosisFormat(disease_name=row["disease_name"], OMIM_id=row.get("identifiers", {}).get("omim_id"), description=row.get("rationale", ""), rank=row["rank"]) for row in final_rows]
    final_diag = DiagnosisOutput(ans=diagnoses, reference=None)
    procedure_trace = {"stages": ["ZebraSeekInputNode", "InitialToolsNode", "DiseaseNormalizeNode", "CandidateResearchNode", "ReflectionNode", "RerankNode", "GeneAnnotationNode", "ZebraSeekOutputNode"], "fixed_actions": FIXED_ACTIONS, "initial_tools": INITIAL_TOOLS, "search_sessions": sessions, "tool_call_ids": [call.get("tool_call_id") for call in calls]}
    final_output = {"patient_id": ctx["patient"]["patient_id"], "ranking_mode": ctx["used_mode"], "ranked_candidates": final_rows, "candidates": final_rows, "source_records": sources, "evidence_records": evidence, "sources": sources, "evidence": evidence, "tool_response_records": ctx["records"], "normalization_records": ctx.get("normalization_records", []), "reflection_assessments": list(ctx["reflections"].values()), "gene_annotations": [gene for row in final_rows for gene in row.get("genes", [])], "prompt_records": ctx["prompt_records"], "procedure_trace": procedure_trace}
    return {
        "patient": ctx["patient"],
        "patient_input": ctx["patient"],
        "tool_response_records": ctx["records"],
        "initial_tool_responses": ctx["records"],
        "source_records": sources,
        "evidence_records": evidence,
        "normalization_records": ctx.get("normalization_records", []),
        "disease_ranking_index": ctx["index"],
        "candidate_records": initial,
        "candidate_pool": initial,
        "search_sessions": sessions,
        "candidate_search_sessions": sessions,
        "action_bindings": ctx["bindings"],
        "tool_call_records": calls,
        "prompt_records": ctx["prompt_records"],
        "procedure_trace": procedure_trace,
        "reflection_assessments": list(ctx["reflections"].values()),
        "ranked_candidates": ctx["ranked"],
        "final_ranking": final_rows,
        "gene_annotations": [gene for row in final_rows for gene in row.get("genes", [])],
        "final_candidates": final_rows,
        "finalDiagnosis": final_diag,
        "final_output": final_output,
        "zebraseek_output": final_output,
        "execution_metadata": {"run_id": ctx["run_id"], "started_at": ctx["started_at"], "finished_at": utc_now(), "ranking_mode": ctx["used_mode"], "config": {"depth": ctx["depth"], "use_togomcp": ctx["use_togomcp"], "use_phenobrain": ctx["use_phenobrain"]}},
    }


def run_zebraseek(
    patient: PatientInput,
    *,
    llm: Any = None,
    togo_client: Any = None,
    ranking_mode: str = "llm",
    depth: int = 1,
    use_togomcp: bool = True,
    use_phenobrain: bool = True,
    max_final_candidates: int = 5,
) -> dict:
    """Compatibility entry point executing the same stages as the LangGraph."""
    ctx = create_zebraseek_context(patient, llm=llm, togo_client=togo_client, ranking_mode=ranking_mode, depth=depth, use_togomcp=use_togomcp, use_phenobrain=use_phenobrain, max_final_candidates=max_final_candidates)
    for stage in (stage_initial_tools, stage_disease_normalization, stage_candidate_research, stage_reflection, stage_rerank, stage_genes):
        ctx = stage(ctx)
    return finalize_zebraseek_context(ctx)
