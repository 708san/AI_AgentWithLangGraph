#!/usr/bin/env python3
"""Download raw ontology/annotation files for HPO_analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RAW_DIR = Path(__file__).resolve().parents[1] / "raw__data"
MANIFEST_PATH = RAW_DIR / "raw_data_manifest.json"

DATA_SOURCES = {
    "hp.json": "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.json",
    "phenotype.hpoa": "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa",
    "mondo.json": "http://purl.obolibrary.org/obo/mondo.json",
    "mondo_exactmatch_omim.sssom.tsv": "http://purl.obolibrary.org/obo/mondo/mappings/mondo_exactmatch_omim.sssom.tsv",
    "mondo_exactmatch_orphanet.sssom.tsv": "http://purl.obolibrary.org/obo/mondo/mappings/mondo_exactmatch_orphanet.sssom.tsv",
}


def log(message: str) -> None:
    print(f"[download-raw-data] {message}", file=sys.stderr)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, path: Path, force: bool) -> bool:
    if path.exists() and path.stat().st_size > 0 and not force:
        log(f"skip existing {path.name}")
        return False
    log(f"download {path.name}")
    urllib.request.urlretrieve(url, path)
    if path.stat().st_size == 0:
        raise RuntimeError(f"downloaded file is empty: {path}")
    return True


def write_manifest(rows: list[dict[str, Any]]) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(RAW_DIR),
        "note": "HPO_analysis/raw__data is intentionally gitignored; regenerate it with this script.",
        "files": rows,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="download files again even when non-empty local files already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for filename, url in DATA_SOURCES.items():
        path = RAW_DIR / filename
        downloaded = download_file(url, path, args.force)
        rows.append(
            {
                "filename": filename,
                "source_url": url,
                "local_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "downloaded_this_run": downloaded,
            }
        )

    write_manifest(rows)
    log(f"wrote manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
