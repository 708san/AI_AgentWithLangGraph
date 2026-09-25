#!/usr/bin/env python3
"""Use the OpenAI API to choose per-case PCF/GM rank cutoffs (0..30)."""
from __future__ import annotations
import argparse, base64, csv, json, os, time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.llm.azure_llm_instance import get_llm_instance
load_dotenv(ROOT / ".env")
TSV = ROOT / "Data/NewData/phenopacket_v1.0.27_New/phenopacket_test_metadata_v0.1.27_GM_v1.1.5_with_phenopacket_data.tsv"
IMAGES = TSV.parent / "test_images"
INPUT_ONLY_OUT = ROOT / "tryToolsForNewData/llm_routing"
TOOL_RESULTS_OUT = ROOT / "tryToolsForNewData/llmWithToolResults_routing"
MODEL = os.getenv("OPENAI_ROUTING_MODEL", "gpt-5-2")
MAX_TOTAL_K = 20

class RoutingDecision(BaseModel):
    pcf_k: int = Field(..., ge=0, le=30, description="Number of PCF results to retain; together with gm_k must be <= 20")
    gm_k: int = Field(..., ge=0, le=30, description="Number of GestaltMatcher results to retain; together with pcf_k must be <= 20")
    reason: str

def prompt(row: dict[str,str], pcf: list[dict[str,Any]] | None = None, gm: list[dict[str,Any]] | None = None) -> str:
    present = list(zip((row.get("pp_present_hpo") or "").split(";"), (row.get("pp_present_hpo_label") or "").split(";")))
    absent = list(zip((row.get("pp_absent_hpo") or "").split(";"), (row.get("pp_absent_hpo_label") or "").split(";")))
    data = {"sex":row.get("gender"),"onset":row.get("age_note"),"present_hpo":present,"absent_hpo":absent}
    if pcf is not None and gm is not None:
        instruction = "You triage two disease ranking tools using the supplied PCF and GestaltMatcher rankings. Choose how many top results to retain from each tool (0 to 30) for downstream evidence review. Use rank, disease name, and score/distance as evidence. This is a constrained budget problem: maximize the probability that the true disease remains in the retained union while minimizing review cost. You MUST satisfy pcf_k + gm_k <= 20. Allocate more ranks to a tool only when its evidence supports meaningful additional coverage; do not use large symmetric cutoffs as a safety default."
        data.update({"pcf_top30": pcf, "gm_top30": gm})
    else:
        instruction = "You must allocate review depth for two disease ranking tools without seeing their results. Choose how many top results to retain from PCF and GestaltMatcher (0 to 30) using only the patient phenotype and facial image. Do not assume any particular disease or tool outcome. This is a constrained budget problem: maximize expected probability that the true disease remains in the retained union while minimizing review cost. You MUST satisfy pcf_k + gm_k <= 20. Do not choose large symmetric cutoffs as a safety default."
    return (instruction + " Do not diagnose and do not use the known answer. Prefer the smallest cutoffs that preserve plausible candidates. Return only the requested JSON.\nPATIENT:\n" + json.dumps(data,ensure_ascii=False))

def enforce_budget(pcf_k: int, gm_k: int) -> tuple[int, int]:
    """Apply the declared rank budget even if a model violates the schema description."""
    pcf_k, gm_k = max(0, min(30, int(pcf_k))), max(0, min(30, int(gm_k)))
    while pcf_k + gm_k > MAX_TOTAL_K:
        if pcf_k >= gm_k and pcf_k > 0:
            pcf_k -= 1
        elif gm_k > 0:
            gm_k -= 1
    return pcf_k, gm_k

def load_tool_results(path: Path, tool: str) -> list[dict[str, Any]]:
    """Read only useful fields from the saved top-30 tool ranking."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("result", {}).get("ranking", [])
    except (OSError, ValueError, TypeError):
        return []
    output = []
    for item in rows[:30]:
        if tool == "PCF":
            output.append({"rank": item.get("rank"), "omim_id": item.get("omim_id"), "disease_name": item.get("omim_disease_name_en"), "score": item.get("score")})
        else:
            output.append({"rank": item.get("rank"), "omim_id": str(item.get("omim_id")) if item.get("omim_id") is not None else None, "disease_name": item.get("syndrome_name"), "distance": item.get("distance"), "gestalt_score": item.get("gestalt_score"), "score": item.get("score")})
    return output

def call(llm: Any, text: str, image: Path|None) -> dict[str,Any]:
    content: list[dict[str,Any]] = [{"type":"text","text":text}]
    if image and image.is_file():
        mime = "image/jpeg" if image.suffix.lower() in (".jpg",".jpeg") else "image/png"
        content.append({"type":"image_url","image_url":{"url":f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode()}"}})
    runnable = llm.with_structured_output(RoutingDecision, method="json_schema", include_raw=True)
    result = runnable.invoke([SystemMessage(content="You are a careful biomedical triage assistant."), HumanMessage(content=content)])
    parsed = result.get("parsed") if isinstance(result, dict) else result
    if parsed is None:
        raise RuntimeError(f"Structured output parsing failed: {result}")
    values = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
    raw = result.get("raw") if isinstance(result, dict) else None
    values["raw_response"] = raw.model_dump() if hasattr(raw, "model_dump") else raw
    return values

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int); ap.add_argument("--image-ids",nargs="*",default=[]); ap.add_argument("--mode",choices=("input_only","tool_results","both"),default="both"); ap.add_argument("--output-dir",type=Path,default=None); ap.add_argument("--workers",type=int,default=5); ap.add_argument("--timeout",type=float,default=180); ap.add_argument("--overwrite",action="store_true"); ap.add_argument("--model",default=MODEL); args=ap.parse_args()
    with TSV.open(newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f, delimiter="\t"))
    if args.image_ids:
        wanted = set(args.image_ids)
        rows = [row for row in rows if str(row.get("image_id", "")).strip() in wanted]
    if args.limit is not None: rows=rows[:args.limit]
    llm = get_llm_instance(args.model).get_temp_llm_with_max_tokens(4096, timeout_seconds=args.timeout)
    variants = ("input_only", "tool_results") if args.mode == "both" else (args.mode,)
    output_dirs = {
        "input_only": (args.output_dir.expanduser().resolve() if args.output_dir else INPUT_ONLY_OUT),
        "tool_results": (args.output_dir.expanduser().resolve() if args.output_dir else TOOL_RESULTS_OUT),
    }
    for directory in (output_dirs[v] for v in variants): directory.mkdir(parents=True, exist_ok=True)
    def one(row, mode):
        image_id=str(row.get("image_id","")).strip(); image=next(iter(sorted(IMAGES.glob(image_id+".*"))),None)
        pcf=load_tool_results(ROOT/f"tryToolsForNewData/PCF/{image_id}.json", "PCF") if mode == "tool_results" else []
        gm=load_tool_results(ROOT/f"tryToolsForNewData/GM/{image_id}.json", "GM") if mode == "tool_results" else []
        path=output_dirs[mode]/f"{image_id}.json"
        if path.exists() and not args.overwrite:
            try:
                x=json.loads(path.read_text());
                if x.get("status")=="ok" and x.get("mode")==mode and 0<=int(x["pcf_k"])<=30 and 0<=int(x["gm_k"])<=30:return image_id,mode,"skipped",x
            except Exception: pass
        started=time.perf_counter()
        request_prompt = prompt(row, pcf, gm) if mode == "tool_results" else prompt(row)
        try:
            result=call(llm,request_prompt,image); pcf_k, gm_k = enforce_budget(result["pcf_k"], result["gm_k"]); out={"status":"ok","mode":mode,"image_id":image_id,"model":args.model,"pcf_k":pcf_k,"gm_k":gm_k,"requested_pcf_k":result["pcf_k"],"requested_gm_k":result["gm_k"],"reason":result["reason"],"budget":MAX_TOTAL_K,"elapsed_ms":round((time.perf_counter()-started)*1000,2),"prompt":request_prompt,"pcf_top30":pcf if mode == "tool_results" else None,"gm_top30":gm if mode == "tool_results" else None,"raw_response":result.get("raw_response")}
        except Exception as e: out={"status":"error","mode":mode,"image_id":image_id,"prompt":request_prompt,"pcf_top30":pcf if mode == "tool_results" else None,"gm_top30":gm if mode == "tool_results" else None,"error":{"type":type(e).__name__,"message":str(e)},"elapsed_ms":round((time.perf_counter()-started)*1000,2)}
        path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); return image_id,mode,out["status"],out
    jobs = [(row, mode) for row in rows for mode in variants]
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as ex:
        futures=[ex.submit(one,row,mode) for row,mode in jobs]
        for f in as_completed(futures):
            image_id,mode,status,_=f.result(); print(f"image_id={image_id} mode={mode} status={status}",flush=True)
    print("saved=" + ",".join(str(output_dirs[v]) for v in variants))
if __name__=="__main__": main()
