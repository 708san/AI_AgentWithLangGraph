#!/usr/bin/env python3
"""Generate GPT-5.2 zero-shot disease rankings for NewData.

Two independent variants are written under ``tryToolsForNewData``:

* ``LLM``: HPO/sex/onset only (no image and no external retrieval).
* ``LLMwithMCP``: GPT-directed, bounded TogoMCP research followed by the same
  short phenotype prompt with compact evidence notes. This script deliberately never calls
  ``pubcasefinder_rank_by_phenotypes`` (or any PubCaseFinder phenotype
  ranking tool).

The complete prompt, structured response, raw model metadata, MCP calls,
timings, and input provenance are saved per image. Existing complete JSON
bundles are skipped by default; incomplete or stale JSON is rerun so an
interrupted run can be resumed safely.

Examples:
    .venv/bin/python tryToolsForNewData/run_zero_shot_for_new_data.py --limit 1
    .venv/bin/python tryToolsForNewData/run_zero_shot_for_new_data.py
    .venv/bin/python tryToolsForNewData/run_zero_shot_for_new_data.py --overwrite
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import mimetypes
import os
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TSV = ROOT / (
    "Data/NewData/phenopacket_v1.0.27_New/"
    "phenopacket_test_metadata_v0.1.27_GM_v1.1.5_with_phenopacket_data.tsv"
)
DEFAULT_IMAGE_DIR = DEFAULT_TSV.parent / "test_images"
DEFAULT_OUTPUT_ROOT = ROOT / "tryToolsForNewData"
DEFAULT_MCP_URL = "https://togomcp.rdfportal.org/mcp"
MODEL_NAME = "gpt-5-2"
REASONING_EFFORT = "medium"
MCP_MAX_ROUNDS = 3
MCP_MAX_TOOL_CALLS = 6
PROMPT_REVISION = "zero_shot_reasoning_v3_dynamic_mcp"
SCHEMA_VERSION = "zero_shot_ranking.v2"


class ZeroShotDisease(BaseModel):
    rank: int = Field(..., ge=1, le=30)
    disease_name: str = Field(..., min_length=1)
    omim_id: Optional[str] = Field(default=None)


class ZeroShotRanking(BaseModel):
    diagnoses: list[ZeroShotDisease] = Field(..., min_length=1, max_length=30)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return jsonable(dump(mode="json"))
        except TypeError:
            return jsonable(dump())
    dump = getattr(value, "dict", None)
    if callable(dump):
        return jsonable(dump())
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def split_values(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").replace(",", ";").split(";") if item.strip()]


def paired_hpo(row: dict[str, str], id_key: str, label_key: str) -> list[dict[str, str]]:
    ids = split_values(row.get(id_key))
    labels = split_values(row.get(label_key))
    return [
        {"hpo_id": hpo_id, "label": labels[index] if index < len(labels) else ""}
        for index, hpo_id in enumerate(ids)
    ]


def find_image(image_dir: Path, image_id: str) -> Path | None:
    matches = sorted(image_dir.glob(f"{image_id}.*"))
    return matches[0] if matches else None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"No rows found: {path}")
    return rows


def image_content(image_path: Path | None) -> dict[str, Any] | None:
    if image_path is None or not image_path.is_file():
        return None
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
    }


def hpo_text(items: list[dict[str, str]]) -> str:
    if not items:
        return "(none reported)"
    return "\n".join(
        f"- {item['hpo_id']}: {item['label']}" if item["label"] else f"- {item['hpo_id']}"
        for item in items
    )


def patient_context(row: dict[str, str], image_dir: Path) -> dict[str, Any]:
    image_id = str(row.get("image_id", "")).strip()
    image_path = find_image(image_dir, image_id)
    present = paired_hpo(row, "pp_present_hpo", "pp_present_hpo_label")
    absent = paired_hpo(row, "pp_absent_hpo", "pp_absent_hpo_label")
    return {
        "patient_id": str(row.get("patient_id", "unknown")),
        "image_id": image_id,
        "image_path": str(image_path) if image_path else None,
        "present_hpo": present,
        "absent_hpo": absent,
        "sex": row.get("gender") or "unknown",
        "onset": row.get("age_note") or (
            f"{row.get('age_year')} years" if row.get("age_year") else "unknown"
        ),
        "ground_truth": {
            "omim_id": row.get("pp_omim") or row.get("disease_id") or "",
            "disease_name": row.get("pp_disease") or row.get("disease") or "",
        },
        "source_row": row,
    }


def no_mcp_prompt(patient: dict[str, Any]) -> str:
    return f"""You are a rare-disease clinical genetics specialist.
Generate a phenotype-based differential diagnosis.
Return up to 30 distinct rare diseases, ordered from most to least likely.
Use only the supplied HPO findings, sex, and onset. Do not use external tools
or web search.
Do not invent an OMIM identifier: return null when you are not confident.
Output only the requested structured fields; do not include explanations.

Before producing the structured output, reason through these steps internally:
1. Separate the explicitly present phenotypes from the explicitly absent phenotypes.
   Unreported findings are unknown, not absent.
2. Group the Present HPO findings into major clinical domains such as
   neurodevelopment, hypotonia or neuromuscular findings, craniofacial features,
   growth, imaging findings, and other organ-system findings.
3. Identify the most discriminative phenotype combinations. Give more weight to
   distinctive combinations than to common findings such as developmental delay.
4. Generate several disease or gene-level hypotheses from the complete phenotype
   pattern, including recognizable syndromic and phenocopy possibilities.
5. Compare each hypothesis against the important Present HPO findings and check
   whether any explicitly Absent HPO is a meaningful contradiction. Do not reject
   a disease only because a feature was not reported.
6. Consider sex and onset only as modifiers. Do not use them to exclude a disease
   when the phenotype pattern is otherwise strong.
7. Deduplicate synonyms and allelic descriptions, then rank the final candidates
   by overall phenotype fit and specificity.
8. Attach an OMIM ID only when the disease identity is sufficiently clear.

Do not output this intermediate reasoning. Return only the final structured list.

Present HPO:
{hpo_text(patient['present_hpo'])}

Explicitly absent HPO (unreported findings are unknown):
{hpo_text(patient['absent_hpo'])}

Sex: {patient['sex']}
Onset: {patient['onset']}
"""


def mcp_prompt(patient: dict[str, Any], compact_context: str) -> str:
    return f"""You are a rare-disease clinical genetics specialist.
Generate up to 30 distinct rare diseases, ordered from most to least likely,
using the supplied HPO findings, sex, onset, and the compact literature context
below.

The TogoMCP context is secondary literature context, not a disease ranking.
Do not call, reproduce, or rely on PubCaseFinder phenotype ranking.
Do not invent an OMIM identifier: return null when you are not confident.
Output only the requested structured fields; do not include explanations.

Before producing the structured output, reason internally in this order:
1. Build a phenotype summary from Present and Absent HPO, treating unreported
   findings as unknown.
2. Identify the most specific phenotype clusters and generate candidate disease
   or gene hypotheses independently of the retrieved literature.
3. Use the TogoMCP notes only to confirm, refine, or challenge those hypotheses.
   A generic article title is not evidence for a specific disease.
4. Compare phenotype fit, explicit contradictions, disease identity, and evidence
   quality for each hypothesis.
5. Deduplicate disease synonyms and rank the final differential.

Do not output this intermediate reasoning. Return only the final structured list.

Present HPO:
{hpo_text(patient['present_hpo'])}

Explicitly absent HPO (unreported findings are unknown):
{hpo_text(patient['absent_hpo'])}

Sex: {patient['sex']}
Onset: {patient['onset']}

Compact TogoMCP PubMed context:
{compact_context or '(no relevant context returned)'}
"""


class TogoMCPError(RuntimeError):
    pass


FORBIDDEN_MCP_MARKERS = (
    "pubcasefinder_rank_by_phenotypes",
    "pubcasefinder phenotype ranking",
)


def _run_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: dict[str, Any] = {}
    errors: list[BaseException] = []

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result.get("value")


class TogoMCPClient:
    def __init__(self, url: str, timeout: float = 30.0, sse_read_timeout: float = 180.0):
        self.url = url
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self.catalog: list[dict[str, Any]] = []

    @asynccontextmanager
    async def session(self):
        async with streamablehttp_client(
            self.url, timeout=self.timeout, sse_read_timeout=self.sse_read_timeout
        ) as (read_stream, write_stream, _session_id):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session

    async def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with self.session() as session:
            result = await session.call_tool(name, arguments=arguments)
            return {
                "is_error": bool(getattr(result, "isError", False)),
                "content": jsonable(getattr(result, "content", [])),
                "structured_content": jsonable(getattr(result, "structuredContent", None)),
            }

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        lowered = name.lower()
        if any(marker in lowered for marker in FORBIDDEN_MCP_MARKERS):
            raise TogoMCPError(f"Forbidden TogoMCP tool was requested: {name}")
        started = time.perf_counter()
        try:
            result = _run_sync(self._call(name, arguments or {}))
            result["tool_name"] = name
            result["arguments"] = arguments or {}
            result["started_at"] = utc_now()
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            if result.get("is_error"):
                raise TogoMCPError(json.dumps(result, ensure_ascii=False))
            return result
        except Exception as exc:
            if isinstance(exc, TogoMCPError):
                raise
            raise TogoMCPError(f"{name}: {exc}") from exc

    async def _list(self) -> list[dict[str, Any]]:
        async with self.session() as session:
            result = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": jsonable(tool.inputSchema),
                }
                for tool in result.tools
            ]

    def list_tools(self) -> list[dict[str, Any]]:
        self.catalog = _run_sync(self._list())
        return self.catalog


def mcp_text(result: dict[str, Any]) -> str:
    blocks = result.get("content", []) or []
    texts = [
        str(block.get("text", ""))
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(texts).strip()


def mcp_payload(result: dict[str, Any]) -> Any:
    structured = result.get("structured_content")
    if structured not in (None, {}):
        return structured
    text = mcp_text(result)
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text


def make_pubmed_query(patient: dict[str, Any]) -> str:
    labels = [str(item.get("label", "")).strip() for item in patient["present_hpo"]]
    labels = [label for label in labels if label]
    # Keep the MCP request and the resulting prompt compact. The full HPO list
    # remains in the zero-shot prompt; MCP supplies only secondary context.
    selected = labels[:3]
    if not selected:
        return "rare disease genetic syndrome"
    return " AND ".join(f'"{label}"' for label in selected)


def compact_mcp_context(search_result: dict[str, Any], summary_result: dict[str, Any] | None) -> str:
    summary_text = mcp_text(summary_result or {})
    # Keep only a compact title/date/source list. The complete MCP response is
    # still saved in the per-case trace; it is intentionally not put into the
    # reasoning context.
    try:
        parsed = json.loads(summary_text)
        result = parsed.get("result", {}) if isinstance(parsed, dict) else {}
        uids = result.get("uids", []) if isinstance(result, dict) else []
        lines: list[str] = []
        for uid in uids[:5]:
            item = result.get(uid, {}) if isinstance(result, dict) else {}
            if not isinstance(item, dict):
                continue
            title = re.sub(r"\s+", " ", str(item.get("title", "")).strip())
            source = str(item.get("source", "")).strip()
            pubdate = str(item.get("pubdate", "")).strip()
            if title:
                lines.append(f"- PMID:{uid} | {title} | {source} | {pubdate}")
        if lines:
            return "\n".join(lines)[:1800]
    except (TypeError, json.JSONDecodeError):
        pass
    text = mcp_text(search_result)
    return re.sub(r"\s+", " ", text).strip()[:1200]


# These are the information-retrieval tools that can help a zero-shot
# phenotype differential. Their schemas are supplied to GPT dynamically from
# the live MCP catalog. PubCaseFinder phenotype ranking is deliberately not in
# this allow-list: it is an initial ranking tool, not a TogoMCP verification
# route for this experiment.
MCP_ZERO_SHOT_TOOLS = {
    "get_workflow",
    "get_sparql_endpoints",
    "get_graph_list",
    "get_MIE_file",
    "run_sparql",
    "search_mesh_descriptor",
    "ncbi_esearch",
    "ncbi_esummary",
    "ncbi_efetch",
    "pubcasefinder_get_case_reports",
    "togoid_getAllRelation",
    "togoid_convertId",
}


def _tool_spec(catalog_item: dict[str, Any]) -> dict[str, Any]:
    """Convert an MCP tools/list item to the OpenAI tool schema."""
    return {
        "type": "function",
        "function": {
            "name": catalog_item["name"],
            "description": str(catalog_item.get("description", ""))[:1800],
            "parameters": catalog_item.get("input_schema") or {
                "type": "object",
                "properties": {},
            },
        },
    }


def _allowed_tool_catalog(client: TogoMCPClient) -> list[dict[str, Any]]:
    catalog = client.catalog or client.list_tools()
    allowed: list[dict[str, Any]] = []
    for item in catalog:
        name = str(item.get("name", ""))
        lowered = name.lower()
        if any(marker in lowered for marker in FORBIDDEN_MCP_MARKERS):
            continue
        if name in MCP_ZERO_SHOT_TOOLS:
            allowed.append(item)
    return allowed


def _compact_tool_result(tool_name: str, result: dict[str, Any]) -> str:
    """Keep the next LLM context bounded while preserving the raw result on disk."""
    text = mcp_text(result)
    if tool_name == "ncbi_esummary":
        try:
            parsed = json.loads(text)
            body = parsed.get("result", {}) if isinstance(parsed, dict) else {}
            lines = []
            for uid in (body.get("uids", []) if isinstance(body, dict) else [])[:5]:
                row = body.get(uid, {}) if isinstance(body, dict) else {}
                if isinstance(row, dict):
                    title = re.sub(r"\s+", " ", str(row.get("title", "")).strip())
                    if title:
                        lines.append(f"PMID:{uid} | {title} | {row.get('source','')} | {row.get('pubdate','')}")
            if lines:
                return "\n".join(lines)[:2400]
        except (TypeError, json.JSONDecodeError):
            pass
    if tool_name == "ncbi_efetch":
        # Abstract XML is useful, but the full record is not. Strip tags and
        # retain a bounded excerpt for the next reasoning turn.
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()[:4200]
    return re.sub(r"\s+", " ", text).strip()[:3000]


def _planner_system_prompt(tool_names: list[str]) -> str:
    return """You are the evidence-planning component of a rare-disease zero-shot system.
The final output will be a ranked list of up to 30 disease names and OMIM IDs.
The patient input contains only Present HPO, explicitly Absent HPO, sex, and onset.
There is no image and no clinical narrative.

Your task in this turn is to decide whether a small amount of external evidence
would improve the differential, and if so call only the TogoMCP tools needed for
that purpose. Use the TogoMCP Usage Guide workflow: identify the information need,
choose the appropriate database/tool, and keep the search bounded. Prefer PubMed
search followed by summaries or abstracts when literature is needed. Use MONDO or
SPARQL only when the available information justifies it; read get_MIE_file before
run_sparql. Do not repeat a failed query without changing its information need.

Before selecting a tool, reason internally through these steps:
1. Summarize the phenotype pattern and identify the highest-value unresolved
   diagnostic questions.
2. Decide whether the phenotype alone is sufficient for a useful differential.
3. If external evidence is useful, specify the exact information gap: literature,
   disease identity, ontology cross-reference, phenotype definition, or another
   supported question.
4. Select the smallest tool call or tool sequence that can resolve that gap.
5. After each result, reassess whether it changed the differential or exposed a
   new, concrete information gap. Stop when additional searching is unlikely to
   change the ranking.

Hard constraints:
- NEVER call pubcasefinder_rank_by_phenotypes. It is excluded from this experiment.
- Do not use any tool as an initial phenotype-ranking substitute.
- Do not invent a disease or OMIM ID from a tool result.
- At most a few focused calls are allowed; stop when the returned evidence is enough.
- Tool results are evidence/context, not ground truth.

Available research tools:
""" + "\n".join(f"- {name}" for name in tool_names)


def _planner_user_prompt(patient: dict[str, Any]) -> str:
    return f"""Plan a bounded evidence search for this patient. You may call tools if useful.

Present HPO:
{hpo_text(patient['present_hpo'])}

Explicitly absent HPO (unreported findings are unknown):
{hpo_text(patient['absent_hpo'])}

Sex: {patient['sex']}
Onset: {patient['onset']}

Do not produce the final differential yet. If no external evidence is needed,
respond without a tool call and say that the phenotype input is sufficient."""


def _final_mcp_prompt(patient: dict[str, Any], research_notes: str) -> str:
    return f"""You are a rare-disease clinical genetics specialist.
Using only the patient phenotype and the bounded TogoMCP evidence notes below,
produce a differential diagnosis of up to 30 distinct diseases ordered by
clinical plausibility. This is phenotype reasoning, not a confirmed diagnosis.
Return only the structured fields rank, disease_name, and omim_id.
Do not invent an OMIM ID; use null when it is not supported.
Treat unreported findings as unknown and explicit Absent HPO as negative evidence.
Do not add a disease merely because it appeared in a generic literature title.

Reason internally through these steps before returning the structured list:
1. Reconstruct the core phenotype pattern and its most discriminative combinations.
2. Generate and compare candidate disease or gene hypotheses from the phenotype.
3. For each hypothesis, distinguish supporting phenotype evidence, explicit
   contradictions, missing-but-unknown findings, and retrieved evidence quality.
4. Use TogoMCP evidence to update the ranking only when it is relevant to the
   candidate identity or phenotype. Do not treat search rank or publication count
   as diagnostic proof.
5. Resolve synonyms and OMIM identity carefully, remove duplicates, and rank up to
   30 candidates by the integrated phenotype and evidence fit.

Do not output this intermediate reasoning. Return only the final structured list.

Present HPO:
{hpo_text(patient['present_hpo'])}

Explicitly absent HPO:
{hpo_text(patient['absent_hpo'])}

Sex: {patient['sex']}
Onset: {patient['onset']}

TogoMCP evidence notes:
{research_notes or '(no external evidence was selected)'}"""


def _call_dynamic_mcp(
    llm: AzureChatOpenAI,
    client: TogoMCPClient,
    patient: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Let GPT select bounded TogoMCP calls, while enforcing the MCP tutorial gates."""
    calls: list[dict[str, Any]] = []
    trace: dict[str, Any] = {
        "planner_system_prompt": "",
        "planner_user_prompt": "",
        "planner_turns": [],
        "tool_allowlist": [],
        "max_rounds": MCP_MAX_ROUNDS,
        "max_tool_calls": MCP_MAX_TOOL_CALLS,
    }

    # The official guide must be read first. It is retained in calls but not
    # copied wholesale into the GPT context.
    guide_started = time.perf_counter()
    try:
        guide = client.call("TogoMCP_Usage_Guide", {})
        calls.append({"tool_name": "TogoMCP_Usage_Guide", "arguments": {}, "result": guide})
    except Exception as exc:
        calls.append({"tool_name": "TogoMCP_Usage_Guide", "arguments": {}, "error": str(exc), "elapsed_ms": round((time.perf_counter() - guide_started) * 1000, 2)})

    allowed_catalog = _allowed_tool_catalog(client)
    tool_names = [str(item["name"]) for item in allowed_catalog]
    tool_specs = [_tool_spec(item) for item in allowed_catalog]
    system = _planner_system_prompt(tool_names)
    user = _planner_user_prompt(patient)
    trace["planner_system_prompt"] = system
    trace["planner_user_prompt"] = user
    trace["tool_allowlist"] = tool_names
    messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=user)]
    notes: list[str] = []
    mie_databases: set[str] = set()
    tool_call_count = 0

    for round_index in range(MCP_MAX_ROUNDS):
        if tool_call_count >= MCP_MAX_TOOL_CALLS:
            break
        planner = llm.bind_tools(tool_specs, tool_choice="auto")
        started = time.perf_counter()
        try:
            response = planner.invoke(messages)
        except Exception as exc:
            trace["planner_turns"].append({"round": round_index + 1, "error": str(exc)})
            break
        turn: dict[str, Any] = {
            "round": round_index + 1,
            "response": jsonable(response),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        tool_calls = list(getattr(response, "tool_calls", []) or [])
        turn["tool_call_names"] = [str(item.get("name", "")) for item in tool_calls]
        trace["planner_turns"].append(turn)
        messages.append(response)
        if not tool_calls:
            break

        for tool_call in tool_calls:
            if tool_call_count >= MCP_MAX_TOOL_CALLS:
                break
            name = str(tool_call.get("name", ""))
            arguments = tool_call.get("args") or {}
            if name.lower() in {marker.lower() for marker in FORBIDDEN_MCP_MARKERS} or name not in tool_names:
                error = f"Tool is not allowed in this workflow: {name}"
                calls.append({"tool_name": name, "arguments": arguments, "status": "rejected", "error": error})
                messages.append(ToolMessage(content=error, tool_call_id=tool_call.get("id", f"rejected-{tool_call_count}")))
                tool_call_count += 1
                continue

            # The guide requires an MIE call before run_sparql. Insert it at
            # the gateway if the planner has not already done so.
            if name == "run_sparql":
                database = str(arguments.get("database") or "").strip()
                if database and database not in mie_databases:
                    try:
                        mie_args = {"database": database}
                        mie_result = client.call("get_MIE_file", mie_args)
                        calls.append({"tool_name": "get_MIE_file", "arguments": mie_args, "result": mie_result, "inserted_by_gateway": True})
                        mie_databases.add(database)
                        mie_note = _compact_tool_result("get_MIE_file", mie_result)
                        notes.append(f"get_MIE_file({database}) gateway note:\n{mie_note}")
                        messages.append(ToolMessage(content=mie_note, tool_call_id=f"gateway-mie-{tool_call_count}"))
                    except Exception as exc:
                        notes.append(f"get_MIE_file({database}) failed: {exc}")

            started_tool = time.perf_counter()
            try:
                result = client.call(name, arguments)
                calls.append({"tool_name": name, "arguments": arguments, "result": result})
                compact = _compact_tool_result(name, result)
                notes.append(f"{name}({json.dumps(arguments, ensure_ascii=False, sort_keys=True)}):\n{compact}")
                messages.append(ToolMessage(content=compact or "(empty result)", tool_call_id=tool_call.get("id", f"tool-{tool_call_count}")))
            except Exception as exc:
                error = str(exc)
                calls.append({"tool_name": name, "arguments": arguments, "status": "error", "error": error, "elapsed_ms": round((time.perf_counter() - started_tool) * 1000, 2)})
                messages.append(ToolMessage(content=f"Tool error: {error}", tool_call_id=tool_call.get("id", f"tool-{tool_call_count}")))
            tool_call_count += 1

        # Keep the planner conversation bounded. Tool results are already
        # compacted and only the latest few turns are necessary.
        if len(messages) > 10:
            messages = [messages[0], messages[1], *messages[-8:]]

    research_notes = "\n\n".join(notes)
    trace["tool_call_count"] = tool_call_count
    trace["research_notes"] = research_notes
    return research_notes[:9000], calls, trace


def run_mcp_context(
    client: TogoMCPClient,
    patient: dict[str, Any],
    llm: AzureChatOpenAI,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Run the bounded, LLM-directed TogoMCP research route."""
    return _call_dynamic_mcp(llm, client, patient)


def build_llm(request_timeout: float | None = None) -> AzureChatOpenAI:
    prefix = "AZURE_OPENAI_5-2"
    values = {
        "azure_endpoint": os.environ.get(f"{prefix}_ENDPOINT"),
        "api_key": os.environ.get(f"{prefix}_API_KEY"),
        "api_version": os.environ.get(f"{prefix}_API_VERSION"),
        "azure_deployment": os.environ.get(f"{prefix}_DEPLOYMENT_NAME"),
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"Missing Azure GPT-5.2 environment variables: {', '.join(missing)}")
    options: dict[str, Any] = {
        **values,
        "extra_body": {
            "reasoning_effort": REASONING_EFFORT,
            "verbosity": "medium",
            "max_completion_tokens": 15000,
        },
    }
    if request_timeout is not None:
        options["request_timeout"] = request_timeout
    return AzureChatOpenAI(**options)


def invoke_structured(llm: AzureChatOpenAI, prompt: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    messages = [HumanMessage(content=content)]
    runnable = llm.with_structured_output(ZeroShotRanking, method="json_schema", include_raw=True)
    started = time.perf_counter()
    result = runnable.invoke(messages)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    parsed = result.get("parsed") if isinstance(result, dict) else result
    raw = result.get("raw") if isinstance(result, dict) else None
    parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
    if parsed is None:
        raise RuntimeError(f"Structured output parsing failed: {parsing_error or raw}")
    return {
        "parsed": jsonable(parsed),
        "raw_response": jsonable(raw),
        "parsing_error": jsonable(parsing_error),
        "elapsed_ms": elapsed_ms,
    }


def normalize_ranking(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    rows = parsed.get("diagnoses", []) if isinstance(parsed, dict) else []
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        name = str(row.get("disease_name", "")).strip()
        if not name:
            continue
        omim = row.get("omim_id")
        if omim is not None:
            omim = str(omim).strip() or None
            if omim and omim.upper() in seen_ids:
                continue
            if omim:
                seen_ids.add(omim.upper())
        normalized.append({"rank": len(normalized) + 1, "disease_name": name, "omim_id": omim})
        if len(normalized) >= 30:
            break
    return normalized


def config_hash(config: dict[str, Any]) -> str:
    return sha256(config)[:16]


def is_complete_output(path: Path, variant: str) -> bool:
    """Return True only for a resumable, successful case bundle.

    A partially written JSON, an error JSON, or a result produced by an older
    prompt revision must be rerun. This prevents interruption recovery from
    silently preserving incomplete or stale rankings.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    expected_variant = "LLMwithMCP" if variant == "mcp" else "LLM"
    if payload.get("schema_version") != SCHEMA_VERSION:
        return False
    if payload.get("status") != "ok" or payload.get("variant") != expected_variant:
        return False
    if payload.get("model") != MODEL_NAME or payload.get("reasoning_effort") != REASONING_EFFORT:
        return False
    if payload.get("llm_input_modalities") != ["text"]:
        return False
    if payload.get("run_config", {}).get("prompt_revision") != PROMPT_REVISION:
        return False
    trace = payload.get("trace")
    if not isinstance(trace, dict) or not isinstance(trace.get("prompt"), str) or not trace["prompt"].strip():
        return False
    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or not 1 <= len(ranking) <= 30:
        return False
    for index, row in enumerate(ranking, start=1):
        if not isinstance(row, dict) or row.get("rank") != index or not str(row.get("disease_name", "")).strip():
            return False
        if row.get("omim_id") is not None and not str(row.get("omim_id")).strip():
            return False
    if not isinstance(trace.get("llm"), dict) or not isinstance(trace["llm"].get("raw_response"), (dict, str, type(None))):
        return False
    if variant == "mcp":
        if not isinstance(trace.get("mcp_calls"), list) or not isinstance(trace.get("mcp_planner"), dict):
            return False
        if any("pubcasefinder_rank_by_phenotypes" in str(call.get("tool_name", "")).lower() for call in trace["mcp_calls"]):
            return False
    else:
        if trace.get("mcp_calls") not in ([], None) or trace.get("mcp_planner") not in ({}, None):
            return False
    return True


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[tuple[int, dict[str, str]]]:
    selected: list[tuple[int, dict[str, str]]] = []
    ids = set(args.image_ids or [])
    patients = set(args.patient_ids or [])
    for index, row in enumerate(rows):
        image_id = str(row.get("image_id", "")).strip()
        patient_id = str(row.get("patient_id", "")).strip()
        if ids and image_id not in ids:
            continue
        if patients and patient_id not in patients:
            continue
        selected.append((index, row))
    if args.start:
        selected = selected[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def run_case(
    index: int,
    row: dict[str, str],
    *,
    args: argparse.Namespace,
    llm: AzureChatOpenAI,
    mcp_client: TogoMCPClient | None,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    patient = patient_context(row, args.image_dir)
    image_id = patient["image_id"] or f"row_{index + 1}"
    started = time.perf_counter()
    common = {
        "schema_version": SCHEMA_VERSION,
        "case_index": index,
        "variant": "LLMwithMCP" if mcp_client else "LLM",
        "model": MODEL_NAME,
        "reasoning_effort": REASONING_EFFORT,
        "llm_input_modalities": ["text"],
        "run_config": run_config,
        "input": patient,
        "started_at": utc_now(),
    }
    mcp_context = ""
    mcp_calls: list[dict[str, Any]] = []
    mcp_trace: dict[str, Any] = {}
    if mcp_client is not None:
        mcp_context, mcp_calls, mcp_trace = run_mcp_context(mcp_client, patient, llm)
        prompt = _final_mcp_prompt(patient, mcp_context)
    else:
        prompt = no_mcp_prompt(patient)
    try:
        llm_result = invoke_structured(llm, prompt)
        parsed = llm_result["parsed"]
        trace = {
            "prompt": prompt,
            "prompt_sha256": sha256(prompt),
            "mcp_context": mcp_context if mcp_client is not None else None,
            "mcp_calls": mcp_calls,
            "mcp_planner": mcp_trace,
            "llm": {
                "raw_response": llm_result.get("raw_response"),
                "parsing_error": llm_result.get("parsing_error"),
                "elapsed_ms": llm_result.get("elapsed_ms"),
            },
        }
        output = {
            **common,
            "status": "ok",
            "ranking": normalize_ranking(parsed),
            "trace": trace,
            "finished_at": utc_now(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except Exception as exc:
        trace = {
            "prompt": prompt,
            "prompt_sha256": sha256(prompt),
            "mcp_context": mcp_context if mcp_client is not None else None,
            "mcp_calls": mcp_calls,
            "mcp_planner": mcp_trace,
        }
        output = {
            **common,
            "status": "error",
            "trace": trace,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "finished_at": utc_now(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-ids", nargs="*", default=[])
    parser.add_argument("--patient-ids", nargs="*", default=[])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--variant",
        choices=("both", "llm", "mcp"),
        default="llm",
        help="Run without MCP by default; choose both or mcp explicitly",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mcp-url", default=os.getenv("TOGOMCP_MCP_URL", DEFAULT_MCP_URL))
    parser.add_argument("--mcp-timeout", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=5, help="Maximum parallel case/variant jobs (default: 5)")
    parser.add_argument(
        "--job-timeout",
        type=float,
        default=600.0,
        help="Per-job watchdog/request timeout in seconds (default: 600)",
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    args.tsv = args.tsv.expanduser().resolve()
    args.image_dir = args.image_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.tsv.is_file():
        parser.error(f"TSV not found: {args.tsv}")
    if not args.image_dir.is_dir():
        parser.error(f"Image directory not found: {args.image_dir}")

    # Create the output tree before any metadata or case result is written.
    # Case processing below is deliberately sequential so logs and partial
    # results follow the input TSV order and interruption is easy to resume.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "LLM").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "LLMwithMCP").mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.tsv)
    selected = select_rows(rows, args)
    variants = ["llm", "mcp"] if args.variant == "both" else [args.variant]
    run_config = {
        "tsv": str(args.tsv),
        "image_dir": str(args.image_dir),
        "model": MODEL_NAME,
        "reasoning_effort": REASONING_EFFORT,
        "variants": variants,
        "mcp_url": args.mcp_url,
        "mcp_mode": "llm_directed_bounded_tool_calls",
        "prompt_revision": PROMPT_REVISION,
        "top_k": 30,
        "structured_output": "ZeroShotRanking",
        "mcp_forbidden_tools": list(FORBIDDEN_MCP_MARKERS),
        "workers": max(1, args.workers),
        "job_timeout_sec": args.job_timeout,
    }
    run_hash = config_hash(run_config)
    write_json(args.output_dir / "zero_shot_run_config.json", {**run_config, "config_hash": run_hash, "created_at": utc_now()})
    print(f"selected_rows={len(selected)} variants={variants} output={args.output_dir}", flush=True)

    if args.job_timeout <= 0:
        parser.error("--job-timeout must be positive")
    llm = build_llm(request_timeout=args.job_timeout)
    mcp_client = TogoMCPClient(args.mcp_url, timeout=args.mcp_timeout) if "mcp" in variants else None
    if mcp_client is not None:
        # A catalog is saved once for audit. The actual workflow lets GPT
        # select from the bounded allow-list; the forbidden ranking tool is
        # never selected even if the server advertises it.
        try:
            catalog = mcp_client.list_tools()
            write_json(args.output_dir / "togomcp_tools_catalog.json", {"retrieved_at": utc_now(), "tools": catalog})
        except Exception as exc:
            write_json(args.output_dir / "togomcp_tools_catalog.json", {"retrieved_at": utc_now(), "error": str(exc), "tools": []})

    summary = {variant: {"ok": 0, "error": 0, "skipped": 0, "elapsed_ms": 0.0} for variant in variants}
    jobs: list[tuple[int, dict[str, str], str, Path]] = []
    for index, row in selected:
        image_id = str(row.get("image_id", "")).strip() or f"row_{index + 1}"
        for variant in variants:
            out_dir = args.output_dir / ("LLMwithMCP" if variant == "mcp" else "LLM")
            path = out_dir / f"{image_id}.json"
            if path.exists() and not args.overwrite and is_complete_output(path, variant):
                summary[variant]["skipped"] += 1
                print(f"image_id={image_id} {variant}: skipped (complete JSON exists)", flush=True)
                continue
            if path.exists() and not args.overwrite:
                print(f"image_id={image_id} {variant}: rerun (incomplete or stale JSON)", flush=True)
            jobs.append((index, row, variant, path))

    def process_job(job: tuple[int, dict[str, str], str, Path]) -> tuple[str, str, dict[str, Any]]:
        index, row, variant, path = job
        image_id = str(row.get("image_id", "")).strip() or f"row_{index + 1}"
        print(f"image_id={image_id} {variant}: started", flush=True)
        started = time.perf_counter()
        try:
            result = run_case(
                index,
                row,
                args=args,
                llm=llm,
                mcp_client=mcp_client if variant == "mcp" else None,
                run_config={**run_config, "variant": variant, "config_hash": run_hash},
            )
        except Exception as exc:
            result = {
                "schema_version": SCHEMA_VERSION,
                "case_index": index,
                "variant": "LLMwithMCP" if variant == "mcp" else "LLM",
                "status": "error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "finished_at": utc_now(),
            }
        return image_id, variant, {"path": str(path), "result": result}

    print(f"jobs_to_run={len(jobs)} workers={max(1, args.workers)} job_timeout_sec={args.job_timeout}", flush=True)
    executor = ThreadPoolExecutor(max_workers=max(1, args.workers), thread_name_prefix="zero-shot")
    future_map = {executor.submit(process_job, job): job for job in jobs}
    pending = set(future_map)
    deadlines = {future: time.monotonic() + args.job_timeout for future in pending}
    try:
        while pending:
            done, _ = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            now = time.monotonic()
            for future in done:
                pending.discard(future)
                index, row, variant, path = future_map[future]
                image_id = str(row.get("image_id", "")).strip() or f"row_{index + 1}"
                try:
                    result_image_id, result_variant, wrapped = future.result()
                    result = wrapped["result"]
                    write_json(path, result)
                    status = result.get("status", "error")
                except Exception as exc:
                    status = "error"
                    result = {"status": status, "error": {"type": type(exc).__name__, "message": str(exc)}}
                    write_json(path, result)
                summary[variant][status] = summary[variant].get(status, 0) + 1
                summary[variant]["elapsed_ms"] += float(result.get("elapsed_ms", 0.0))
                print(f"image_id={image_id} {variant}: {status} elapsed_ms={result.get('elapsed_ms')}", flush=True)
            for future in list(pending):
                if now < deadlines[future]:
                    continue
                pending.discard(future)
                index, row, variant, path = future_map[future]
                image_id = str(row.get("image_id", "")).strip() or f"row_{index + 1}"
                future.cancel()
                timeout_result = {
                    "schema_version": SCHEMA_VERSION,
                    "case_index": index,
                    "variant": "LLMwithMCP" if variant == "mcp" else "LLM",
                    "status": "error",
                    "error": {"type": "JobTimeout", "message": f"job exceeded {args.job_timeout} seconds"},
                    "finished_at": utc_now(),
                    "elapsed_ms": round(args.job_timeout * 1000, 2),
                }
                write_json(path, timeout_result)
                summary[variant]["error"] += 1
                summary[variant]["elapsed_ms"] += timeout_result["elapsed_ms"]
                print(f"image_id={image_id} {variant}: timeout after {args.job_timeout}s", flush=True)
    finally:
        # Running requests have their own HTTP/request timeouts. Do not wait
        # indefinitely here if a third-party client ignores cancellation.
        executor.shutdown(wait=False, cancel_futures=True)
    write_json(args.output_dir / "zero_shot_summary.json", {"summary": summary, "finished_at": utc_now()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
