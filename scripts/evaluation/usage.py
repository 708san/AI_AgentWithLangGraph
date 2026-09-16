"""Evaluation-only HTTP usage observation and offline token-cost estimates.

Hooks are installed only on the two synchronous clients used by Zero-shot.
Reading a non-streaming HTTP response caches its body; SDK parsing, retries,
arguments and exceptions remain unchanged. No request bodies/headers are saved.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path


DEFAULT_PRICING = Path(__file__).with_name("pricing.azure.json")


def load_pricing(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("currency"), str) or not data["currency"]:
        raise ValueError("Pricing requires a currency")
    for stage in ("zero_shot", "normalization"):
        rates = data.get(stage, {})
        if not isinstance(rates, dict) or not isinstance(rates.get("model"), str) or not rates["model"]:
            raise ValueError("Pricing requires a model for each stage")
        for key in ("input_per_million", "cached_input_per_million", "output_per_million"):
            value = rates.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid price: {stage}.{key}")
    return data


class UsageCapture:
    def __init__(self, clients, record, checkpoint):
        self.clients = clients  # [(stage, SDK client)]
        self.record = record
        self.checkpoint = checkpoint
        self.hooks = []
        self.pending = {}

    def persist(self):
        try:
            self.checkpoint()
        except Exception as exc:
            self.record["usage_recording_errors"] += 1
            warnings.warn(f"Usage checkpoint failed: {type(exc).__name__}")

    def __enter__(self):
        self.record["api_usage"] = []
        self.record["usage_recording_errors"] = 0
        try:
            for stage, client in self.clients:
                # OpenAI SDK has no public per-response usage callback. Scope
                # these HTTPX hooks to evaluation and restore them in __exit__.
                hooks = client._client.event_hooks

                def request_hook(request, stage=stage):
                    row = {"stage": stage, "status": "pending", "usage": None}
                    self.record["api_usage"].append(row)
                    self.pending[id(request)] = (row, time.perf_counter())
                    self.persist()

                def response_hook(response):
                    entry = self.pending.pop(id(response.request), None)
                    if entry is None:
                        return
                    row, started = entry
                    row.update(status="response", http_status=response.status_code)
                    try:
                        if "text/event-stream" not in response.headers.get("content-type", ""):
                            response.read()
                            body = response.json()
                            # Whitelist metadata; never store content or embeddings.
                            row.update(model=body.get("model"), response_id=body.get("id"),
                                       usage=body.get("usage"), service_tier=body.get("service_tier"))
                            row["finish_reasons"] = [c.get("finish_reason") for c in body.get("choices", [])]
                    except Exception as exc:
                        row["observation_error"] = type(exc).__name__
                    row["elapsed_seconds"] = time.perf_counter() - started
                    self.persist()

                for event, fn in (("request", request_hook), ("response", response_hook)):
                    hooks.setdefault(event, []).append(fn)
                    self.hooks.append((hooks[event], fn))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for hooks, fn in reversed(self.hooks):
            hooks.remove(fn)
        self.hooks.clear()
        for row, started in self.pending.values():
            row.update(status="no_response", elapsed_seconds=time.perf_counter() - started)
        self.persist()


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def summarize(records, pricing):
    """Sum observed attempts, never interpret missing usage as a free request."""
    records = list(records)
    result = {"currency": pricing["currency"], "pricing": pricing,
              "basis": "API-reported usage times supplied retail rates; not an invoice",
              "records": len(records), "records_without_usage": 0,
              "recording_errors": 0, "stages": {}}
    for stage in ("zero_shot", "normalization"):
        stats = dict(attempts=0, measured_attempts=0, unpriced_attempts=0,
                     input_tokens=0, cached_input_tokens=0, output_tokens=0,
                     reasoning_tokens=0, unknown_reasoning_attempts=0,
                     known_cost=0.0, known_cost_without_cache_discount=0.0)
        rates = pricing[stage]
        for record in records:
            for call in record.get("api_usage", []):
                if call["stage"] != stage:
                    continue
                stats["attempts"] += 1
                usage = call.get("usage") or {}
                inp = _count(usage.get("prompt_tokens"))
                out = 0 if stage == "normalization" else _count(usage.get("completion_tokens"))
                cached = 0 if stage == "normalization" else _count((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
                reasoning = 0 if stage == "normalization" else _count((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
                for key, count in (("input_tokens", inp), ("output_tokens", out),
                                   ("cached_input_tokens", cached), ("reasoning_tokens", reasoning)):
                    if count is not None:
                        stats[key] += count
                stats["unknown_reasoning_attempts"] += reasoning is None
                if inp is not None and out is not None:
                    stats["measured_attempts"] += 1
                model = call.get("model") or ""
                matches = model == rates["model"] or model.startswith(rates["model"] + "-20")
                # Snapshot suffixes are allowed; e.g. gpt-5.2-pro is not gpt-5.2.
                if inp is None or out is None or cached is None or cached > inp or not matches:
                    stats["unpriced_attempts"] += 1
                    continue
                # completion_tokens already includes reasoning_tokens.
                stats["known_cost"] += ((inp - cached) * rates["input_per_million"] +
                                        cached * rates["cached_input_per_million"] +
                                        out * rates["output_per_million"]) / 1_000_000
                stats["known_cost_without_cache_discount"] += (
                    inp * rates["input_per_million"] + out * rates["output_per_million"]) / 1_000_000
        result["stages"][stage] = stats
    result["records_without_usage"] = sum(not r.get("api_usage") for r in records)
    result["recording_errors"] = sum(r.get("usage_recording_errors", 0) for r in records)
    result["known_cost"] = sum(s["known_cost"] for s in result["stages"].values())
    result["known_cost_without_cache_discount"] = sum(s["known_cost_without_cache_discount"] for s in result["stages"].values())
    result["usage_complete"] = bool(records) and not (result["records_without_usage"] or
        result["recording_errors"] or any(s["unpriced_attempts"] for s in result["stages"].values()))
    result["estimated_cost"] = result["known_cost"] if result["usage_complete"] else None
    return result


def format_cost(report):
    amount = report["estimated_cost"]
    total = f"{amount:.8f}" if amount is not None else "unknown (incomplete usage/pricing)"
    parts = [f"Estimated total: {total} {report['currency']}"]
    for stage, stats in report["stages"].items():
        parts.append(f"{stage}: known_cost={stats['known_cost']:.8f}, "
                     f"input={stats['input_tokens']}, cached={stats['cached_input_tokens']}, "
                     f"output={stats['output_tokens']}, reasoning={stats['reasoning_tokens']}, "
                     f"unpriced_attempts={stats['unpriced_attempts']}")
    return "\n".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recalculate saved Zero-shot usage costs without API calls")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pricing-file", type=Path, help="Defaults to the pricing snapshot saved in metadata.json")
    args = parser.parse_args(argv)
    try:
        pricing = load_pricing(args.pricing_file) if args.pricing_file else json.loads((args.run_dir / "metadata.json").read_text())["pricing"]
        records = [json.loads(p.read_text()) for p in sorted(args.run_dir.glob("repeat_*/predictions/*.json"))]
        if not records:
            raise ValueError("No prediction records found")
        print(json.dumps(summarize(records, pricing), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
