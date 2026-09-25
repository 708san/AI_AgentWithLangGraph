"""Deterministic formatting for values exposed to API/UI consumers."""

import re
from typing import Any

from pydantic import BaseModel


def omim_number(value: Any) -> Any:
    """Return only the numeric part of an OMIM identifier."""
    if value is None or value == "":
        return value
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else value


def omim_curie(value: Any) -> Any:
    """Preserve the canonical OMIM namespace alongside display-only numbers."""
    number = omim_number(value)
    if number is None or number == "":
        return number
    return f"OMIM:{number}" if str(number).isdigit() else value


def _to_public_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _to_public_value(value.model_dump())
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"OMIM_id", "omim_id"}:
                result[key] = omim_number(item)
                result[f"{key}_curie"] = omim_curie(item)
            else:
                result[key] = _to_public_value(item)
        return result
    if isinstance(value, list):
        return [_to_public_value(item) for item in value]
    if isinstance(value, tuple):
        return [_to_public_value(item) for item in value]
    return value


def serialize_public_state(state: dict) -> dict:
    """Create a JSON-ready response without exposing the LLM client object."""
    return {
        key: _to_public_value(value)
        for key, value in state.items()
        if key != "llm"
    }
