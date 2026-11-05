"""Build a safe, read-only catalog of repository sources and local artifacts.

The catalog is intentionally metadata-first. It records paths, categories,
Git tracking state, sizes, and script/data hints without importing application
modules or reading credentials. Generated output is written outside Git by the
CLI's default path.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path("local_artifacts/catalog/project_catalog.json")
IGNORED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "local_artifacts",
    "run_outputs",
    "res",
    "format_res",
    "log",
    "artifacts",
}
SENSITIVE_NAMES = {".env", ".env.local", ".env.production", ".env.development"}
ARTIFACT_ROOTS = (
    "local_artifacts",
    "run_outputs",
    "res",
    "log",
    "format_res",
    "testCode/phenopacket_image_case/runs",
)
IGNORED_RELATIVE_ROOTS = tuple(
    Path(item).parts
    for item in (
        "testCode/phenopacket_image_case/runs",
    )
)
DATA_ROOTS = {
    "Data",
    "agent/data",
    "PhenoPacketSchema",
    "scripts/data",
}
HASH_CATEGORIES = {
    "runtime",
    "execution",
    "tests",
    "documentation",
    "configuration",
    "data",
    "workflow",
}


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_context(root: Path) -> dict[str, Any]:
    tracked_output = _git(root, "ls-files") or ""
    tracked = sorted(item for item in tracked_output.splitlines() if item)
    status = _git(root, "status", "--porcelain", "--untracked-files=all") or ""
    return {
        "branch": _git(root, "branch", "--show-current"),
        "commit": _git(root, "rev-parse", "HEAD"),
        "working_tree_dirty": bool(status),
        "status_entries": len(status.splitlines()) if status else 0,
        "tracked_file_count": len(tracked),
        "tracked_files": tracked,
    }


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_hidden_or_ignored(relative_path: str) -> bool:
    parts = Path(relative_path).parts
    if any(part in IGNORED_DIRS for part in parts):
        return True
    if Path(relative_path).name in SENSITIVE_NAMES:
        return True
    return any(parts[: len(root)] == root for root in IGNORED_RELATIVE_ROOTS)


def _category(relative_path: str) -> str:
    path = Path(relative_path)
    first = path.parts[0] if path.parts else ""
    normalized = relative_path.replace("\\", "/")
    if path.name == ".DS_Store":
        return "workspace_noise"
    if first == "HPO_analysis":
        if "output" in path.parts or path.suffix.lower() in {".csv", ".png", ".svg"}:
            return "derived_data"
        if "scripts" in path.parts and path.suffix.lower() == ".py":
            return "execution"
        if path.suffix.lower() == ".md":
            return "documentation"
        return "experiment"
    if first == "scripts" and len(path.parts) > 1 and path.parts[1] == "data":
        return "data"
    if path.suffix.lower() == ".md":
        return "documentation"
    if first == "agent":
        return "runtime" if len(path.parts) < 2 or path.parts[1] not in {"data"} else "data"
    if first in {"scripts", "testCode"}:
        return "execution"
    if first == "tests":
        return "tests"
    if first == "docs" or path.name in {"README.md", "AGENTS.md", "GOALS.md", "PLANS.md", "PROMPTS.md"}:
        return "documentation"
    if first in {"Data", "PhenoPacketSchema"} or any(normalized == root or normalized.startswith(root + "/") for root in DATA_ROOTS):
        return "data"
    if first in {"DEV", "tryToolsForNewData", "Improved_test"}:
        return "experiment"
    if first in {"HPO_analysis", "HPO_importance"}:
        return "derived_data"
    if first in {"harness", ".agents", "project_ops"}:
        return "workflow"
    if path.name in {"requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", ".gitignore", ".python-version"}:
        return "configuration"
    return "other"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _script_metadata(path: Path) -> dict[str, Any]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        return {"parse_error": f"{type(exc).__name__}: {exc}"}

    docstring = ast.get_docstring(tree)
    return {
        "module_doc": docstring.splitlines()[0] if docstring else None,
        "has_main_guard": "if __name__" in source and "__main__" in source,
        "uses_argparse": "argparse" in source,
        "top_level_functions": [
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ],
    }


def _data_hint(relative_path: str, size_bytes: int) -> dict[str, Any]:
    suffix = Path(relative_path).suffix.lower()
    return {
        "format": suffix.removeprefix(".") or "none",
        "is_image": suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"},
        "is_json_like": suffix in {".json", ".jsonl", ".yaml", ".yml", ".tsv", ".csv"},
        "size_bytes": size_bytes,
    }


def _artifact_root_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    file_count = 0
    directory_count = 0
    total_bytes = 0
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_dir():
                directory_count += 1
            elif child.is_file():
                file_count += 1
                try:
                    total_bytes += child.stat().st_size
                except OSError:
                    pass
    return {
        "path": relative_path,
        "exists": path.exists(),
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
    }


def build_catalog(
    root: Path,
    *,
    include_hash: bool = False,
    max_hash_bytes: int = 50 * 1024 * 1024,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    git = _git_context(root)
    tracked = set(git["tracked_files"])
    files: list[dict[str, Any]] = []
    scripts: list[dict[str, Any]] = []
    data_files: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = _relative(path, root)
        if _is_hidden_or_ignored(relative_path):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        category = _category(relative_path)
        record: dict[str, Any] = {
            "path": relative_path,
            "category": category,
            "tracked": relative_path in tracked,
            "size_bytes": stat.st_size,
            "suffix": path.suffix.lower(),
        }
        if (
            include_hash
            and category in HASH_CATEGORIES
            and stat.st_size <= max_hash_bytes
        ):
            record["sha256"] = _sha256(path)
        files.append(record)

        if path.suffix.lower() == ".py" and category == "execution":
            scripts.append({
                **record,
                **_script_metadata(path),
            })
        if category in {"data", "derived_data"}:
            data_files.append({
                **record,
                **_data_hint(relative_path, stat.st_size),
            })

    category_counts = Counter(item["category"] for item in files)
    extension_counts = Counter(item["suffix"] or "<none>" for item in files)
    artifact_roots = [_artifact_root_record(root, item) for item in ARTIFACT_ROOTS]
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_type": "project_inventory",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(root),
        "git": git,
        "summary": {
            "file_count": len(files),
            "script_count": len(scripts),
            "data_file_count": len(data_files),
            "artifact_root_count": sum(item["exists"] for item in artifact_roots),
            "category_counts": dict(sorted(category_counts.items())),
            "extension_counts": dict(sorted(extension_counts.items())),
        },
        "artifact_roots": artifact_roots,
        "scripts": scripts,
        "data_files": data_files,
        "files": files,
    }


def write_catalog(path: Path, catalog: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

