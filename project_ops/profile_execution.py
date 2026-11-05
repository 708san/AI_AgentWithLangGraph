"""Execute an existing entry point selected by its canonical profile."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

from project_ops.execution import run_managed_command
from project_ops.profiles import get_profile


def build_profile_command(
    root: Path,
    profile_id: str,
    script_args: Iterable[str] = (),
) -> list[str]:
    root = root.expanduser().resolve()
    profile = get_profile(root, profile_id)
    entrypoint = root / str(profile["entrypoint"])
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Profile entrypoint does not exist: {entrypoint}")
    return [sys.executable, str(entrypoint), *(str(item) for item in script_args)]


def run_profile(
    *,
    root: Path,
    profile_id: str,
    run_id: str,
    script_args: Iterable[str] = (),
    dataset: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    pipeline: dict[str, Any] | None = None,
    output_specs: Iterable[tuple[str, str] | tuple[str, str, str | None]] = (),
    registry_path: Path | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    profile = get_profile(root, profile_id)
    command = build_profile_command(root, profile_id, script_args)
    metadata = {
        "profile_id": profile_id,
        "profile": profile,
    }
    return run_managed_command(
        root=root,
        run_id=run_id,
        kind=str(profile["kind"]),
        command=command,
        purpose=str(profile.get("purpose") or ""),
        entrypoint=str(profile["entrypoint"]),
        dataset=dataset,
        model=model,
        pipeline=pipeline,
        metadata=metadata,
        output_specs=output_specs,
        registry_path=registry_path,
    )

