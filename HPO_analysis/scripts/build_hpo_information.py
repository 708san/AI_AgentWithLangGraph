#!/usr/bin/env python3
"""Build HPO information-content data with MONDO disease normalization."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHENOTYPIC_ABNORMALITY = "HP:0000118"
RAW_DIR = Path(__file__).resolve().parents[1] / "raw__data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

DATA_SOURCES = {
    "hp.json": "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.json",
    "phenotype.hpoa": "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa",
    "mondo.json": "http://purl.obolibrary.org/obo/mondo.json",
    "mondo_exactmatch_omim.sssom.tsv": "http://purl.obolibrary.org/obo/mondo/mappings/mondo_exactmatch_omim.sssom.tsv",
    "mondo_exactmatch_orphanet.sssom.tsv": "http://purl.obolibrary.org/obo/mondo/mappings/mondo_exactmatch_orphanet.sssom.tsv",
}


@dataclass(frozen=True)
class Annotation:
    raw_disease_id: str
    disease_name: str
    hpo_id: str


@dataclass
class Normalization:
    raw_id: str
    lookup_id: str
    canonical_id: str
    status: str
    mondo_ids: list[str]


def log(message: str) -> None:
    print(f"[hpo-analysis] {message}", file=sys.stderr)


def curie(value: str) -> str:
    if value.startswith("http://purl.obolibrary.org/obo/"):
        value = value.rsplit("/", 1)[-1]
    if "_" in value and ":" not in value:
        prefix, local_id = value.split("_", 1)
        return f"{prefix}:{local_id}"
    return value


def ensure_data() -> dict[str, dict[str, Any]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict[str, Any]] = {}
    checked_at = datetime.now(timezone.utc).isoformat()
    for filename, url in DATA_SOURCES.items():
        path = RAW_DIR / filename
        downloaded = False
        if not path.exists() or path.stat().st_size == 0:
            log(f"downloading {filename}")
            urllib.request.urlretrieve(url, path)
            downloaded = True
        if path.stat().st_size == 0:
            raise RuntimeError(f"downloaded file is empty: {path}")
        metadata[filename] = {
            "source_url": url,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "checked_at": checked_at,
            "file_modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "downloaded_this_run": downloaded,
        }
    return metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_version(data: dict[str, Any]) -> str:
    graph = data.get("graphs", [{}])[0]
    meta = graph.get("meta", {})
    for key in ("version", "date", "saved-by"):
        if key in meta:
            return str(meta[key])
    basic_values = meta.get("basicPropertyValues", [])
    for item in basic_values:
        pred = item.get("pred", "")
        if pred.endswith("owl#versionInfo") or pred.endswith("terms/date"):
            return str(item.get("val", ""))
    return ""


def parse_obo_graph(data: dict[str, Any]) -> tuple[dict[str, str], dict[str, set[str]], dict[str, bool], dict[str, set[str]]]:
    names: dict[str, str] = {}
    parents: dict[str, set[str]] = defaultdict(set)
    children: dict[str, set[str]] = defaultdict(set)
    obsolete: dict[str, bool] = {}

    graph = data.get("graphs", [{}])[0]
    for node in graph.get("nodes", []):
        node_id = curie(node.get("id", ""))
        if not node_id:
            continue
        names[node_id] = node.get("lbl", node_id)
        obsolete[node_id] = bool(node.get("meta", {}).get("deprecated", False))

    for edge in graph.get("edges", []):
        pred = edge.get("pred", "")
        if pred not in {"is_a", "subClassOf", "http://www.w3.org/2000/01/rdf-schema#subClassOf"}:
            continue
        sub = curie(edge.get("sub", ""))
        obj = curie(edge.get("obj", ""))
        if sub and obj:
            parents[sub].add(obj)
            children[obj].add(sub)

    return names, parents, obsolete, children


def descendants(root: str, children: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    queue: deque[str] = deque([root])
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        queue.extend(children.get(current, set()) - seen)
    return seen


def ancestor_cache(parents: dict[str, set[str]]) -> dict[str, set[str]]:
    cache: dict[str, set[str]] = {}

    def visit(term: str) -> set[str]:
        if term in cache:
            return cache[term]
        terms = {term}
        for parent in parents.get(term, set()):
            terms.update(visit(parent))
        cache[term] = terms
        return terms

    for term in set(parents) | {p for values in parents.values() for p in values}:
        visit(term)
    return cache


def top_level_categories(term: str, parents: dict[str, set[str]], names: dict[str, str]) -> list[dict[str, str]]:
    if term == PHENOTYPIC_ABNORMALITY:
        return []
    found: set[str] = set()

    def walk(current: str, child_on_path: str | None) -> None:
        for parent in parents.get(current, set()):
            if parent == PHENOTYPIC_ABNORMALITY:
                found.add(current)
            else:
                walk(parent, current)

    walk(term, None)
    return [{"hpo_id": hpo_id, "name": names.get(hpo_id, hpo_id)} for hpo_id in sorted(found)]


def parse_hpoa(path: Path, phenotypic_terms: set[str]) -> tuple[list[Annotation], dict[str, Any]]:
    header: list[str] | None = None
    rows = 0
    skipped_not = 0
    skipped_non_p = 0
    annotations: list[Annotation] = []
    fallback_header = [
        "database_id",
        "disease_name",
        "qualifier",
        "hpo_id",
        "reference",
        "evidence",
        "onset",
        "frequency",
        "sex",
        "modifier",
        "aspect",
        "biocuration",
    ]

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                maybe_header = line.lstrip("#").split("\t")
                normalized = [col.lower().replace(" ", "_") for col in maybe_header]
                if "databaseid" in normalized or "database_id" in normalized:
                    header = normalized
                continue
            rows += 1
            parts = line.split("\t")
            columns = header or fallback_header
            record = {columns[i]: parts[i] if i < len(parts) else "" for i in range(min(len(columns), len(parts)))}
            database_id = record.get("databaseid") or record.get("database_id") or parts[0]
            disease_name = record.get("diseasename") or record.get("disease_name") or (parts[1] if len(parts) > 1 else "")
            qualifier = record.get("qualifier", "")
            hpo_id = record.get("hpo_id") or record.get("hpoid") or (parts[3] if len(parts) > 3 else "")
            aspect = record.get("aspect") or (parts[10] if len(parts) > 10 else "")
            qualifiers = {q.strip().upper() for q in qualifier.replace("|", ";").split(";") if q.strip()}
            if "NOT" in qualifiers:
                skipped_not += 1
                continue
            if aspect and aspect != "P":
                skipped_non_p += 1
                continue
            if hpo_id not in phenotypic_terms:
                skipped_non_p += 1
                continue
            annotations.append(Annotation(database_id, disease_name, hpo_id))

    return annotations, {
        "header": header,
        "raw_annotation_rows": rows,
        "positive_phenotypic_annotation_rows": len(annotations),
        "skipped_not_annotations": skipped_not,
        "skipped_non_phenotypic_annotations": skipped_non_p,
    }


def normalize_source_for_lookup(raw_id: str) -> str:
    if raw_id.startswith("MIM:"):
        return "OMIM:" + raw_id.split(":", 1)[1]
    if raw_id.startswith("ORPHA:"):
        return raw_id
    if raw_id.startswith("Orphanet:"):
        return "ORPHA:" + raw_id.split(":", 1)[1]
    return raw_id


def source_prefix(raw_id: str) -> str:
    return raw_id.split(":", 1)[0] if ":" in raw_id else "UNKNOWN"


def parse_sssom(path: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    exact: dict[str, set[str]] = defaultdict(set)
    non_exact: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as f:
        filtered = (line for line in f if line.strip() and not line.startswith("#"))
        reader = csv.DictReader(filtered, delimiter="\t")
        if not reader.fieldnames:
            raise RuntimeError(f"missing SSSOM header: {path}")
        for row in reader:
            subject = row.get("subject_id", "")
            obj = row.get("object_id", "")
            predicate = row.get("predicate_id", "")
            if not subject or not obj:
                continue
            subject = normalize_source_for_lookup(subject)
            obj = normalize_source_for_lookup(obj)
            ids = [subject, obj]
            mondo_ids = [x for x in ids if x.startswith("MONDO:")]
            source_ids = [x for x in ids if not x.startswith("MONDO:")]
            if not mondo_ids or not source_ids:
                continue
            target = exact if predicate.endswith("exactMatch") or "exact" in predicate else non_exact
            for source_id in source_ids:
                for mondo_id in mondo_ids:
                    target[source_id].add(mondo_id)
    return exact, non_exact


def merge_mappings(paths: list[Path]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    exact: dict[str, set[str]] = defaultdict(set)
    non_exact: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        exact_part, non_exact_part = parse_sssom(path)
        for key, values in exact_part.items():
            exact[key].update(values)
        for key, values in non_exact_part.items():
            non_exact[key].update(values)
    return exact, non_exact


def normalize_diseases(
    raw_disease_ids: set[str],
    exact_map: dict[str, set[str]],
    mondo_obsolete: dict[str, bool],
) -> tuple[dict[str, Normalization], dict[str, Any]]:
    normalized: dict[str, Normalization] = {}
    status_counts: Counter[str] = Counter()
    prefix_counts = Counter(source_prefix(raw_id) for raw_id in raw_disease_ids)
    canonical_to_raws: dict[str, set[str]] = defaultdict(set)
    conflicts: list[dict[str, Any]] = []
    unmapped: list[str] = []

    for raw_id in sorted(raw_disease_ids):
        lookup_id = normalize_source_for_lookup(raw_id)
        if raw_id.startswith("MONDO:"):
            canonical_id = raw_id
            status = "already_mondo"
            mondo_ids = [raw_id]
        else:
            candidates = sorted(exact_map.get(lookup_id, set()))
            active = [m for m in candidates if not mondo_obsolete.get(m, False)]
            if len(active) == 1:
                canonical_id = active[0]
                status = "mapped"
                mondo_ids = active
            elif len(active) > 1:
                canonical_id = f"CONFLICT:{raw_id}"
                status = "conflicted"
                mondo_ids = active
                conflicts.append({"raw_id": raw_id, "lookup_id": lookup_id, "mondo_ids": active})
            elif candidates:
                canonical_id = f"UNMAPPED:{raw_id}"
                status = "unmapped_obsolete_only"
                mondo_ids = candidates
                unmapped.append(raw_id)
            else:
                canonical_id = f"UNMAPPED:{raw_id}"
                status = "unmapped"
                mondo_ids = []
                unmapped.append(raw_id)
        normalized[raw_id] = Normalization(raw_id, lookup_id, canonical_id, status, mondo_ids)
        canonical_to_raws[canonical_id].add(raw_id)
        status_counts[status] += 1

    merged = [
        {"canonical_id": canonical, "raw_ids": sorted(raws)}
        for canonical, raws in canonical_to_raws.items()
        if canonical.startswith("MONDO:") and len(raws) > 1
    ]
    merged.sort(key=lambda item: (-len(item["raw_ids"]), item["canonical_id"]))
    report = {
        "raw_unique_disease_count": len(raw_disease_ids),
        "canonical_unique_disease_count": len(canonical_to_raws),
        "disease_count_reduction_after_mondo_normalization": len(raw_disease_ids) - len(canonical_to_raws),
        "already_mondo_count": status_counts["already_mondo"],
        "mapped_to_mondo_count": status_counts["mapped"],
        "unmapped_count": status_counts["unmapped"] + status_counts["unmapped_obsolete_only"],
        "conflict_count": status_counts["conflicted"],
        "source_prefix_counts": dict(sorted(prefix_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "examples": {
            "merged": merged[:20],
            "unmapped": unmapped[:20],
            "conflicted": conflicts[:20],
        },
    }
    return normalized, report


def build_associations(
    annotations: list[Annotation],
    ancestors: dict[str, set[str]],
    normalizations: dict[str, Normalization] | None,
) -> tuple[dict[str, set[str]], set[str]]:
    hpo_to_diseases: dict[str, set[str]] = defaultdict(set)
    all_diseases: set[str] = set()
    for annotation in annotations:
        disease_id = (
            normalizations[annotation.raw_disease_id].canonical_id
            if normalizations is not None
            else annotation.raw_disease_id
        )
        all_diseases.add(disease_id)
        for hpo_id in ancestors.get(annotation.hpo_id, {annotation.hpo_id}):
            hpo_to_diseases[hpo_id].add(disease_id)
    return hpo_to_diseases, all_diseases


def ic_entry(count: int, total: int) -> dict[str, Any]:
    fraction = count / total if total else 0.0
    if count == 0 or total == 0:
        ic = None
    else:
        ic = -math.log2(fraction)
    return {
        "information_content": ic,
        "disease_count": count,
        "total_disease_count": total,
        "disease_fraction": fraction,
    }


def make_terms(
    phenotypic_terms: set[str],
    names: dict[str, str],
    parents: dict[str, set[str]],
    canonical_assoc: dict[str, set[str]],
    canonical_total: int,
    raw_assoc: dict[str, set[str]],
    raw_total: int,
) -> dict[str, dict[str, Any]]:
    terms: dict[str, dict[str, Any]] = {}
    for hpo_id in sorted(phenotypic_terms):
        primary = ic_entry(len(canonical_assoc.get(hpo_id, set())), canonical_total)
        raw = ic_entry(len(raw_assoc.get(hpo_id, set())), raw_total)
        terms[hpo_id] = {
            "hpo_id": hpo_id,
            "name": names.get(hpo_id, hpo_id),
            **primary,
            "phen2disease_raw": raw,
            "categories": top_level_categories(hpo_id, parents, names),
        }
    return terms


def find_duplicate_propagation_example(
    annotations: list[Annotation],
    ancestors: dict[str, set[str]],
    raw_assoc: dict[str, set[str]],
) -> dict[str, Any] | None:
    by_disease: dict[str, set[str]] = defaultdict(set)
    for annotation in annotations:
        by_disease[annotation.raw_disease_id].add(annotation.hpo_id)
    for disease_id, hpo_ids in by_disease.items():
        for child in hpo_ids:
            for parent in ancestors.get(child, set()) - {child}:
                if parent in hpo_ids and disease_id in raw_assoc.get(parent, set()):
                    return {
                        "raw_disease_id": disease_id,
                        "direct_parent_hpo": parent,
                        "direct_child_hpo": child,
                        "parent_count_contains_disease_once": True,
                    }
    return None


def sanity_checks(
    terms: dict[str, dict[str, Any]],
    parents: dict[str, set[str]],
    annotations: list[Annotation],
    ancestors: dict[str, set[str]],
    raw_assoc: dict[str, set[str]],
    normalization_report: dict[str, Any],
    non_exact_map: dict[str, set[str]],
) -> dict[str, Any]:
    monotonic_examples: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for child, parent_ids in parents.items():
        if child not in terms:
            continue
        child_count = terms[child]["disease_count"]
        child_ic = terms[child]["information_content"]
        for parent in parent_ids:
            if parent not in terms:
                continue
            parent_count = terms[parent]["disease_count"]
            parent_ic = terms[parent]["information_content"]
            ok_count = parent_count >= child_count
            ok_ic = parent_ic is None or child_ic is None or parent_ic <= child_ic
            example = {
                "parent": parent,
                "parent_name": terms[parent]["name"],
                "parent_disease_count": parent_count,
                "parent_ic": parent_ic,
                "child": child,
                "child_name": terms[child]["name"],
                "child_disease_count": child_count,
                "child_ic": child_ic,
                "count_monotonic": ok_count,
                "ic_monotonic": ok_ic,
            }
            if not (ok_count and ok_ic):
                violations.append(example)
            elif len(monotonic_examples) < 10 and child_count > 0:
                monotonic_examples.append(example)

    annotated = [t for t in terms.values() if t["disease_count"] > 0]
    lowest = sorted(annotated, key=lambda x: (x["information_content"], -x["disease_count"]))[:10]
    highest = sorted(annotated, key=lambda x: (-x["information_content"], x["disease_count"]))[:10]
    check_ids = [PHENOTYPIC_ABNORMALITY, "HP:0000707", "HP:0001250"]
    comparison = {
        hpo_id: {
            "name": terms[hpo_id]["name"],
            "mondo_normalized": {
                "ic": terms[hpo_id]["information_content"],
                "disease_count": terms[hpo_id]["disease_count"],
                "fraction": terms[hpo_id]["disease_fraction"],
            },
            "phen2disease_raw": terms[hpo_id]["phen2disease_raw"],
        }
        for hpo_id in check_ids
        if hpo_id in terms
    }
    non_exact_examples = [
        {"source_id": source_id, "mondo_ids": sorted(values)}
        for source_id, values in sorted(non_exact_map.items())[:10]
    ]
    return {
        "monotonic_examples": monotonic_examples,
        "monotonic_violation_count": len(violations),
        "monotonic_violations_sample": violations[:20],
        "duplicate_propagation_example": find_duplicate_propagation_example(annotations, ancestors, raw_assoc),
        "merged_examples": normalization_report["examples"]["merged"][:10],
        "non_exact_mapping_examples_not_used": non_exact_examples,
        "ic_comparison": comparison,
        "lowest_ic_terms": summarize_terms(lowest),
        "highest_ic_terms": summarize_terms(highest),
    }


def summarize_terms(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "hpo_id": row["hpo_id"],
            "name": row["name"],
            "information_content": row["information_content"],
            "disease_count": row["disease_count"],
        }
        for row in rows
    ]


def write_readme(
    data_metadata: dict[str, dict[str, Any]],
    hpo_release: str,
    mondo_release: str,
    hpoa_stats: dict[str, Any],
    normalization_report: dict[str, Any],
    terms: dict[str, dict[str, Any]],
    sanity: dict[str, Any],
) -> None:
    annotated_count = sum(1 for term in terms.values() if term["disease_count"] > 0)
    generated = datetime.now(timezone.utc).isoformat()
    lines = [
        "# HPO Analysis",
        "",
        "## Method",
        "",
        "Primary analysis uses Phen2Disease-style information content plus MONDO exact disease normalization.",
        "Phen2Disease counts OMIM, ORPHA, MONDO, and other database IDs as separate diseases; this analysis maps exact OMIM/ORPHA disease matches to MONDO to avoid double-counting the same disease concept.",
        "",
        "IC(t) = -log2(N(t) / N_all). HPO annotations are propagated to all ancestors in the HPO DAG, and each disease is counted at most once per HPO term.",
        "",
        "Only positive phenotypic abnormality annotations are used: `aspect = P`, descendants of `HP:0000118`, and annotations with `qualifier = NOT` are excluded.",
        "",
        "Disease normalization uses exact MONDO mappings only. Unmapped diseases are retained as source-specific fallback IDs, and conflicted mappings are not silently merged.",
        "",
        "## Sources",
        "",
        f"Generated at: {generated}",
        f"HPO release/version: {hpo_release or 'not found in source metadata'}",
        f"MONDO release/version: {mondo_release or 'not found in source metadata'}",
        "",
    ]
    for filename, meta in data_metadata.items():
        lines.extend(
            [
                f"### {filename}",
                f"- source URL: {meta['source_url']}",
                f"- checked at: {meta['checked_at']}",
                f"- local file modified/downloaded at: {meta['file_modified_at']}",
                f"- downloaded this run: {meta['downloaded_this_run']}",
                f"- bytes: {meta['bytes']}",
                f"- SHA-256: `{meta['sha256']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Summary",
            "",
            f"- total raw disease IDs: {normalization_report['raw_unique_disease_count']}",
            f"- total canonical disease concepts: {normalization_report['canonical_unique_disease_count']}",
            f"- disease count reduction after MONDO normalization: {normalization_report['disease_count_reduction_after_mondo_normalization']}",
            f"- already MONDO IDs: {normalization_report['already_mondo_count']}",
            f"- mapped to MONDO: {normalization_report['mapped_to_mondo_count']}",
            f"- unmapped diseases: {normalization_report['unmapped_count']}",
            f"- mapping conflicts: {normalization_report['conflict_count']}",
            f"- total HPO terms in output: {len(terms)}",
            f"- annotated HPO terms: {annotated_count}",
            f"- unannotated HPO terms: {len(terms) - annotated_count}",
            f"- positive phenotypic annotation rows: {hpoa_stats['positive_phenotypic_annotation_rows']}",
            f"- skipped NOT annotations: {hpoa_stats['skipped_not_annotations']}",
            "- raw data validation: all required files exist, have non-zero size, JSON files were parsed successfully, and phenotype.hpoa header was detected.",
            "",
            "## Required Examples",
            "",
        ]
    )
    for hpo_id in ("HP:0000707", "HP:0001250"):
        if hpo_id in terms:
            row = terms[hpo_id]
            lines.extend(
                [
                    f"### {hpo_id} {row['name']}",
                    f"- MONDO-normalized disease_count: {row['disease_count']}",
                    f"- MONDO-normalized IC: {row['information_content']}",
                    f"- Phen2Disease raw disease_count: {row['phen2disease_raw']['disease_count']}",
                    f"- Phen2Disease raw IC: {row['phen2disease_raw']['information_content']}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Sanity Checks",
            "",
            f"- monotonic violation count: {sanity['monotonic_violation_count']}",
            f"- duplicate propagation example: `{json.dumps(sanity['duplicate_propagation_example'], ensure_ascii=False)}`",
            f"- merged OMIM/ORPHA examples: `{json.dumps(sanity['merged_examples'][:3], ensure_ascii=False)}`",
            "",
            "### IC comparison",
            "",
            "```json",
            json.dumps(sanity["ic_comparison"], ensure_ascii=False, indent=2),
            "```",
            "",
            "### Lowest IC terms",
            "",
            "```json",
            json.dumps(sanity["lowest_ic_terms"], ensure_ascii=False, indent=2),
            "```",
            "",
            "### Highest IC terms",
            "",
            "```json",
            json.dumps(sanity["highest_ic_terms"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Re-run",
            "",
            "```bash",
            "cd HPO_analysis",
            "python scripts/build_hpo_information.py",
            "```",
            "",
        ]
    )
    (Path(__file__).resolve().parents[1] / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_metadata = ensure_data()
    log("loading ontology files")
    hpo_json = load_json(RAW_DIR / "hp.json")
    mondo_json = load_json(RAW_DIR / "mondo.json")
    hpo_names, hpo_parents, _, hpo_children = parse_obo_graph(hpo_json)
    mondo_names, _, mondo_obsolete, _ = parse_obo_graph(mondo_json)
    hpo_release = extract_version(hpo_json)
    mondo_release = extract_version(mondo_json)

    log("building HPO ancestor cache")
    phenotypic_terms = descendants(PHENOTYPIC_ABNORMALITY, hpo_children)
    ancestors = ancestor_cache(hpo_parents)
    phenotypic_ancestors = {term: ancestors.get(term, {term}) & phenotypic_terms for term in phenotypic_terms}

    log("parsing HPO disease annotations")
    annotations, hpoa_stats = parse_hpoa(RAW_DIR / "phenotype.hpoa", phenotypic_terms)
    raw_disease_ids = {annotation.raw_disease_id for annotation in annotations}

    log("loading exact MONDO mappings")
    exact_map, non_exact_map = merge_mappings(
        [
            RAW_DIR / "mondo_exactmatch_omim.sssom.tsv",
            RAW_DIR / "mondo_exactmatch_orphanet.sssom.tsv",
        ]
    )
    normalizations, normalization_report = normalize_diseases(raw_disease_ids, exact_map, mondo_obsolete)

    log("propagating disease-HPO associations")
    canonical_assoc, canonical_diseases = build_associations(annotations, phenotypic_ancestors, normalizations)
    raw_assoc, raw_diseases = build_associations(annotations, phenotypic_ancestors, None)

    log("computing IC and categories")
    terms = make_terms(
        phenotypic_terms,
        hpo_names,
        hpo_parents,
        canonical_assoc,
        len(canonical_diseases),
        raw_assoc,
        len(raw_diseases),
    )
    sanity = sanity_checks(terms, hpo_parents, annotations, phenotypic_ancestors, raw_assoc, normalization_report, non_exact_map)

    metadata = {
        "method": "Phen2Disease-style IC with MONDO disease normalization",
        "primary_ic_formula": "-log2(canonical_disease_count / total_canonical_disease_count)",
        "comparison_ic_formula": "-log2(raw_disease_count / total_raw_disease_count)",
        "log_base": 2,
        "ancestor_propagation": True,
        "disease_deduplication_per_hpo": True,
        "disease_normalization": {
            "ontology": "MONDO",
            "mapping_policy": "exact match only",
            "unmapped_policy": "retain as source-specific fallback disease",
        },
        "excluded_negated_annotations": True,
        "phenotypic_abnormality_only": True,
        "total_canonical_disease_count": len(canonical_diseases),
        "total_raw_disease_count": len(raw_diseases),
        "hpo_release": hpo_release,
        "mondo_release": mondo_release,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": data_metadata,
        "hpoa_stats": hpoa_stats,
    }
    hpo_information = {"metadata": metadata, "terms": terms}

    normalization_report["generated_at"] = metadata["generated_at"]
    normalization_report["mondo_name_count"] = len(mondo_names)
    normalization_report["normalizations_sample"] = [
        {
            "raw_id": item.raw_id,
            "lookup_id": item.lookup_id,
            "canonical_id": item.canonical_id,
            "status": item.status,
            "mondo_ids": item.mondo_ids,
        }
        for item in list(normalizations.values())[:50]
    ]

    (OUTPUT_DIR / "hpo_information.json").write_text(
        json.dumps(hpo_information, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "disease_normalization_report.json").write_text(
        json.dumps(normalization_report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "sanity_checks.json").write_text(
        json.dumps(sanity, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_readme(data_metadata, hpo_release, mondo_release, hpoa_stats, normalization_report, terms, sanity)
    log(f"wrote {OUTPUT_DIR / 'hpo_information.json'}")
    log(f"terms={len(terms)} canonical_diseases={len(canonical_diseases)} raw_diseases={len(raw_diseases)}")


if __name__ == "__main__":
    main()
