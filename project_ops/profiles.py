"""Canonical descriptions of existing execution and analysis entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_profiles(root: Path) -> dict[str, Any]:
    path = root.expanduser().resolve() / "configs/execution_profiles.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("profiles"), list):
        raise ValueError(f"Invalid execution profile registry: {path}")
    return value


def list_profiles(root: Path) -> list[dict[str, Any]]:
    return [item for item in load_profiles(root)["profiles"] if isinstance(item, dict)]


def get_profile(root: Path, profile_id: str) -> dict[str, Any]:
    for profile in list_profiles(root):
        if profile.get("profile_id") == profile_id:
            return profile
    raise KeyError(f"Unknown execution profile: {profile_id}")

