#!/usr/bin/env python3
"""Analyze why TRAF7/CAFDADD phenopackets split by tool success."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_OMIM = "618164"
TARGET_OMIM_ID = "OMIM:618164"
TARGET_DISEASE = "Cardiac, facial, and digital anomalies with developmental delay"

RANK_OVERVIEW = ROOT / "local_artifacts/evaluation/validationCode/phenopacket_rank_overview_with_context/correct_disease_rank_overview_with_context.csv"
PATIENT_IC = ROOT / "local_artifacts/IC_eval/Patient_IC_eval/output/patient_ic_eval_table.csv"
HPO_INFO = ROOT / "HPO_analysis/output/hpo_information.json"
RUN_RESULTS_DIR = ROOT / "local_artifacts/run_outputs/res_5-2"
OUT_DIR = ROOT / "docs/cafdadd_zebraseek_group_analysis"

TOOLS = ["PubCaseFinder", "PhenotypeSearch", "ZeroShot", "GestaltMatcher", "TentativeDiagnosis", "FinalDiagnosis"]
CATEGORY_COLS = [
    "cat_cardiovascular",
    "cat_head_neck",
    "cat_limbs",
    "cat_musculoskeletal",
    "cat_nervous",
    "cat_respiratory",
    "cat_eye",
    "cat_ear",
    "cat_growth",
    "cat_genitourinary",
    "cat_prenatal_birth",
]

FACIAL_RE = re.compile(
    r"facies|facial|blepharo|palpebral|ptosis|hypertelorism|telecanthus|epicanthus|"
    r"philtrum|forehead|brow|eyebrow|nose|nasal|nares|chin|mandib|gnathia|palate|ear|neck",
    re.I,
)
DIGITAL_RE = re.compile(
    r"finger|toe|digit|phalan|phalangeal|thumb|nail|clinodactyly|syndactyly|brachydactyly|"
    r"camptodactyly|polydactyly|talipes|pes planus|foot|feet|hand",
    re.I,
)
DEVELOPMENT_RE = re.compile(
    r"development|delayed|delay|speech|language|intellectual|autism|hypotonia|seizure|"
    r"motor|walk|cns|brain|cerebral|cerebellar|myelination|leukomalacia",
    re.I,
)
ACUTE_NEONATAL_RE = re.compile(
    r"feeding|tube feeding|dysphagia|poor suck|hypotonia|respiratory|sepsis|lethargy|"
    r"hyperbilirubinemia|failure to thrive|oral aversion|neonatal",
    re.I,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def yes(value: str | None) -> bool:
    return value not in (None, "", "-", "999")


def as_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except ValueError:
        return default


def as_int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value or default))
    except ValueError:
        return default


def hpo_terms() -> dict[str, dict[str, Any]]:
    with HPO_INFO.open() as f:
        return json.load(f)["terms"]


def hpo_name(hpo: str, terms: dict[str, dict[str, Any]]) -> str:
    return terms.get(hpo, {}).get("name", hpo)


def hpo_categories(hpo: str, terms: dict[str, dict[str, Any]]) -> str:
    cats = terms.get(hpo, {}).get("categories", [])
    return "; ".join(cat.get("name", "") for cat in cats if cat.get("name"))


def split_hpo_ids(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def load_result(pid: str) -> dict[str, Any]:
    path = RUN_RESULTS_DIR / f"{pid}.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def normalize_omim(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text if text.startswith("OMIM:") else f"OMIM:{text}"


def candidate_names(result: dict[str, Any], key: str, limit: int = 5) -> str:
    block = result.get(key) or {}
    ans = block.get("ans") if isinstance(block, dict) else block
    if not isinstance(ans, list):
        return ""
    names: list[str] = []
    for cand in ans[:limit]:
        if isinstance(cand, dict):
            names.append(str(cand.get("disease_name") or cand.get("syndrome_name") or ""))
    return " | ".join(name for name in names if name)


def md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def truth_rank_in_answer(result: dict[str, Any], key: str) -> str:
    block = result.get(key) or {}
    ans = block.get("ans") if isinstance(block, dict) else block
    if not isinstance(ans, list):
        return ""
    for idx, cand in enumerate(ans, 1):
        if isinstance(cand, dict) and normalize_omim(cand.get("OMIM_id") or cand.get("omim_id")) == TARGET_OMIM_ID:
            return str(cand.get("rank") or idx)
    return ""


def best_initial_hits(result: dict[str, Any]) -> tuple[str, str]:
    rows: list[tuple[str, int, str, str]] = []
    for idx, cand in enumerate(result.get("pubCaseFinder") or [], 1):
        if normalize_omim(cand.get("omim_id")) == TARGET_OMIM_ID:
            rows.append(("PCF", idx, cand.get("disease_name", ""), str(cand.get("score", ""))))
    for idx, cand in enumerate(result.get("GestaltMatcher") or [], 1):
        if normalize_omim(cand.get("omim_id")) == TARGET_OMIM_ID:
            rows.append(("GM", idx, cand.get("syndrome_name", ""), str(cand.get("score", ""))))
    if not rows:
        return "", ""
    brief = "; ".join(f"{tool} rank {rank} score {score}" for tool, rank, _, score in rows)
    names = "; ".join(name for _, _, name, _ in rows if name)
    return brief, names


def group_for(row: dict[str, str]) -> str:
    final_hit = yes(row.get("FinalDiagnosis_label"))
    pcf_hit = yes(row.get("PubCaseFinder_label"))
    gm_hit = yes(row.get("GestaltMatcher_label"))
    if final_hit:
        return "A_zebraseek_final_correct"
    if pcf_hit or gm_hit:
        return "B_initial_pcf_or_gm_correct_final_wrong"
    return "C_no_major_tool_correct"


def parse_raw_phenopacket(row: dict[str, str]) -> dict[str, Any]:
    raw = ROOT / "local_artifacts/evaluation" / row["raw_path"]
    if not raw.exists():
        return {}
    with raw.open() as f:
        return json.load(f)


def labels_from_raw(raw: dict[str, Any]) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = []
    for feature in raw.get("phenotypicFeatures", []) or []:
        term = feature.get("type") or {}
        if feature.get("excluded"):
            continue
        hid = term.get("id")
        label = term.get("label")
        if hid and label:
            labels.append((hid, label))
    return labels


def count_matching(labels: list[tuple[str, str]], pattern: re.Pattern[str]) -> int:
    return sum(1 for _, label in labels if pattern.search(label))


def top_terms_by_group(cases: list[dict[str, Any]], terms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    group_case_count = Counter(case["group"] for case in cases)
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        for hpo_id in set(case["present_hpo_ids"]):
            by_group[case["group"]][hpo_id] += 1

    all_hpos = set()
    for counter in by_group.values():
        all_hpos.update(counter)

    rows: list[dict[str, Any]] = []
    for hpo_id in sorted(all_hpos):
        row: dict[str, Any] = {
            "hpo_id": hpo_id,
            "hpo_name": hpo_name(hpo_id, terms),
            "categories": hpo_categories(hpo_id, terms),
            "ic": terms.get(hpo_id, {}).get("information_content", ""),
        }
        rates: dict[str, float] = {}
        for group in sorted(group_case_count):
            n = by_group[group][hpo_id]
            denom = group_case_count[group]
            rate = n / denom if denom else 0
            row[f"{group}_count"] = n
            row[f"{group}_rate"] = round(rate, 3)
            rates[group] = rate
        row["max_rate_delta"] = round(max(rates.values()) - min(rates.values()), 3) if rates else 0
        rows.append(row)

    return sorted(rows, key=lambda r: (-float(r["max_rate_delta"]), -as_float(str(r["ic"])), r["hpo_id"]))


def group_summary(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in sorted({case["group"] for case in cases}):
        members = [case for case in cases if case["group"] == group]
        rows.append(
            {
                "group": group,
                "case_count": len(members),
                "patient_ids": ";".join(case["patient_id"] for case in members),
                "mean_present_hpo_count": round(mean(case["present_hpo_count"] for case in members), 2),
                "mean_absent_hpo_count": round(mean(case["absent_hpo_count"] for case in members), 2),
                "mean_total_ic": round(mean(case["total_ic"] for case in members), 2),
                "mean_mean_ic": round(mean(case["mean_ic"] for case in members), 3),
                "mean_anchor_count": round(mean(case["anchor_count"] for case in members), 2),
                "mean_acute_neonatal_count": round(mean(case["acute_neonatal_count"] for case in members), 2),
                "cardiac_case_rate": round(mean(1 if case["cardiac_count"] else 0 for case in members), 3),
                "facial_case_rate": round(mean(1 if case["facial_count"] else 0 for case in members), 3),
                "digital_or_limb_case_rate": round(mean(1 if case["digital_or_limb_count"] else 0 for case in members), 3),
                "developmental_neuro_case_rate": round(mean(1 if case["developmental_neuro_count"] else 0 for case in members), 3),
                "pcf_hit_rate": round(mean(1 if case["pcf_hit"] else 0 for case in members), 3),
                "gm_hit_rate": round(mean(1 if case["gm_hit"] else 0 for case in members), 3),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(cases: list[dict[str, Any]], summaries: list[dict[str, Any]], top_hpos: list[dict[str, Any]]) -> None:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_group[case["group"]].append(case)

    lines = [
        "# CAFDADD / TRAF7 Phenopacket Tool-Group Analysis",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Question",
        "",
        "同じ Phenopacket 正解疾患 `Cardiac, facial, and digital anomalies with developmental delay` (CAFDADD; TRAF7; OMIM:618164) の中で、ZebraSeek が答えられる症例、ZebraSeek は外すが PCF/GM が拾う症例、主要ツールがどれも拾えない症例の違いを調べる。",
        "",
        "## Inputs",
        "",
        f"- Ranking overview: `{RANK_OVERVIEW.relative_to(ROOT)}`",
        f"- Patient IC/category table: `{PATIENT_IC.relative_to(ROOT)}`",
        f"- Raw phenopackets: `local_artifacts/evaluation/sampleData/0.1.25_filtered/TRAF7/*.json`",
        f"- ZebraSeek run JSON: `{RUN_RESULTS_DIR.relative_to(ROOT)}/*.json`",
        f"- HPO labels/categories: `{HPO_INFO.relative_to(ROOT)}`",
        "",
        "## Investigation Plan",
        "",
        "1. OMIM:618164 の Phenopacket 症例だけを評価CSVから抽出する。",
        "2. 評価CSVの `FinalDiagnosis`, `PubCaseFinder`, `GestaltMatcher` 判定を一次ソースとして A/B/C の3群へ分ける。",
        "3. 患者IC表と raw phenopacket から、HPO数、IC、カテゴリ、cardiac/facial/digital/developmental anchor、急性新生児サインを症例別に付与する。",
        "4. 実行結果JSONから tentative/final の候補名を取り、初期候補から最終候補への脱落を確認する。",
        "5. 群別平均と HPO 出現率差をCSV化し、読み物用のMarkdownに要約する。",
        "",
        "## Classification Rule",
        "",
        "- A: ZebraSeek final diagnosis contains OMIM:618164 within the evaluated ranked answer.",
        "- B: ZebraSeek final diagnosis misses OMIM:618164, but PubCaseFinder or GestaltMatcher contains OMIM:618164.",
        "- C: ZebraSeek final diagnosis, PubCaseFinder, and GestaltMatcher all miss OMIM:618164.",
        "",
        "## Summary",
        "",
        "| Group | n | Patient IDs | Mean present HPO | Mean anchor count | Mean acute/neonatal count | PCF hit | GM hit |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['group']} | {s['case_count']} | {s['patient_ids']} | "
            f"{s['mean_present_hpo_count']} | {s['mean_anchor_count']} | {s['mean_acute_neonatal_count']} | "
            f"{s['pcf_hit_rate']} | {s['gm_hit_rate']} |"
        )

    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            "1. ZebraSeek が最終的に正解した群は 10/19 例。GM が 9/10 例で先に拾っており、顔貌・眼周囲・発達/神経系の組み合わせが最終順位に残りやすい。",
            "2. ZebraSeek は外したが PCF/GM が拾った群は 3/19 例。正解候補は初期候補または tentative に入っているが、最終 reranking で急性新生児像、骨格/筋緊張、広い多発奇形候補に押し出されている。",
            "3. どれも拾えない群は 6/19 例。HPO数はむしろ多めで、単なる情報量不足ではない。症例ごとの表現が分散し、PCF/GM が使う典型的な CAFDADD アンカーへ接続できていない。",
            "4. 疾患名の cardiac/facial/digital/developmental 4ドメインが揃うほど良い、という単純な構造ではない。失敗群にも cardiac/facial/digital/neuro の記載は存在するが、`feeding/hypotonia/respiratory/sepsis` などの非特異的な新生児重症サインが候補選択を強く引っ張る症例がある。",
            "",
            "## Case Table",
            "",
            "| Patient | Group | Eval final rank | JSON truth rank | PCF rank | GM rank | Anchors | Acute/neonatal | Top final candidates |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for case in cases:
        lines.append(
            f"| {case['patient_id']} | {case['group']} | {case['final_rank']} | {case['final_json_truth_rank'] or '-'} | {case['pcf_rank']} | "
            f"{case['gm_rank']} | {case['anchor_count']} | {case['acute_neonatal_count']} | "
            f"{md_cell(case['final_top_candidates'][:180])} |"
        )

    mismatches = [case for case in cases if bool(case["final_json_truth_rank"]) != bool(case["final_hit"])]
    if mismatches:
        lines.extend(
            [
                "",
                "## Data Quality Note",
                "",
                "Primary grouping uses the evaluated ranking overview CSV. The current run JSON can differ from that snapshot:",
            ]
        )
        for case in mismatches:
            lines.append(
                f"- `{case['patient_id']}`: evaluation CSV says final miss, but current run JSON contains OMIM:618164 at rank {case['final_json_truth_rank']}."
            )

    lines.extend(["", "## Group Notes", ""])
    for group in sorted(by_group):
        lines.append(f"### {group}")
        for case in by_group[group]:
            notes = case["interpretation_note"]
            lines.append(
                f"- `{case['patient_id']}`: {notes} Present preview: {case['present_hpo_preview']}"
            )
        lines.append("")

    lines.extend(
        [
            "## HPO Signals With Large Group Differences",
            "",
            "| HPO | Name | Delta | A rate | B rate | C rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in top_hpos[:25]:
        lines.append(
            f"| {row['hpo_id']} | {md_cell(row['hpo_name'])} | {row['max_rate_delta']} | "
            f"{row.get('A_zebraseek_final_correct_rate', 0)} | "
            f"{row.get('B_initial_pcf_or_gm_correct_final_wrong_rate', 0)} | "
            f"{row.get('C_no_major_tool_correct_rate', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Next Checks",
            "",
            "1. Final reranking に「PCF/GM top-5 に exact OMIM がある場合は少なくとも最終候補に保持する」ルールを入れ、11700/11708/11714 が救済されるか確認する。",
            "2. CAFDADD の anchor HPO セットを disease-level recurrent HPO から作り、急性新生児サインだけで上位候補が置き換わるケースを監査する。",
            "3. GM ヒットがあるのに final から落ちた症例は、Reflection/Final の negative evidence が「未記載」を「否定」に近く扱っていないか確認する。",
            "",
            "## Output Files",
            "",
            "- `case_group_table.csv`: 症例単位のツール順位、HPOドメイン、最終候補。",
            "- `group_summary.csv`: 3群の平均値とヒット率。",
            "- `hpo_group_signal.csv`: HPOごとの群別出現率。",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    terms = hpo_terms()
    rank_rows = [row for row in read_csv(RANK_OVERVIEW) if row.get("omim_id") == TARGET_OMIM]
    ic_rows = {row["patient_id"]: row for row in read_csv(PATIENT_IC) if row.get("omim_id") == TARGET_OMIM}

    cases: list[dict[str, Any]] = []
    for row in sorted(rank_rows, key=lambda r: as_int(r["patient_id"])):
        pid = row["patient_id"]
        raw = parse_raw_phenopacket(row)
        labels = labels_from_raw(raw)
        result = load_result(pid)
        ic = ic_rows.get(pid, {})

        pcf_hit = yes(row.get("PubCaseFinder_label"))
        gm_hit = yes(row.get("GestaltMatcher_label"))
        final_hit = yes(row.get("FinalDiagnosis_label"))
        group = group_for(row)
        initial_brief, initial_names = best_initial_hits(result)

        cardiac_count = as_int(ic.get("cat_cardiovascular"))
        facial_count = count_matching(labels, FACIAL_RE)
        digital_or_limb_count = count_matching(labels, DIGITAL_RE)
        developmental_neuro_count = count_matching(labels, DEVELOPMENT_RE)
        acute_neonatal_count = count_matching(labels, ACUTE_NEONATAL_RE)
        anchor_count = sum(
            1
            for count in [cardiac_count, facial_count, digital_or_limb_count, developmental_neuro_count]
            if count > 0
        )

        if final_hit:
            note = "FinalDiagnosis に正解が残った。"
        elif pcf_hit or gm_hit:
            note = f"初期候補では拾えた ({initial_brief}) が、final から落ちた。"
        else:
            note = "FinalDiagnosis/PCF/GM のいずれにも正解が出ていない。"

        case: dict[str, Any] = {
            "patient_id": pid,
            "phenopacket_id": row["phenopacket_id"],
            "group": group,
            "disease_name": TARGET_DISEASE,
            "omim_id": TARGET_OMIM,
            "gene_name": row["gene_name"],
            "raw_path": row["raw_path"],
            "present_hpo_count": as_int(row.get("present_hpo_count")),
            "absent_hpo_count": as_int(row.get("absent_hpo_count")),
            "present_hpo_ids": split_hpo_ids(row.get("present_hpo_ids", "")),
            "present_hpo_preview": row.get("present_hpo_preview", ""),
            "tool_rank_summary": row.get("tool_rank_summary", ""),
            "pcf_hit": pcf_hit,
            "pcf_rank": row.get("PubCaseFinder_sort_rank", "999"),
            "gm_hit": gm_hit,
            "gm_rank": row.get("GestaltMatcher_sort_rank", "999"),
            "final_hit": final_hit,
            "final_rank": row.get("FinalDiagnosis_sort_rank", "999"),
            "final_json_truth_rank": truth_rank_in_answer(result, "finalDiagnosis"),
            "tentative_rank": row.get("TentativeDiagnosis_sort_rank", "999"),
            "phenotype_search_rank": row.get("PhenotypeSearch_sort_rank", "999"),
            "zero_shot_rank": row.get("ZeroShot_sort_rank", "999"),
            "total_ic": round(as_float(ic.get("total_ic")), 4),
            "mean_ic": round(as_float(ic.get("mean_ic")), 4),
            "max_ic": round(as_float(ic.get("max_ic")), 4),
            "cardiac_count": cardiac_count,
            "facial_count": facial_count,
            "digital_or_limb_count": digital_or_limb_count,
            "developmental_neuro_count": developmental_neuro_count,
            "acute_neonatal_count": acute_neonatal_count,
            "anchor_count": anchor_count,
            "initial_hit_brief": initial_brief,
            "initial_hit_names": initial_names,
            "tentative_top_candidates": candidate_names(result, "tentativeDiagnosis", limit=5),
            "final_top_candidates": candidate_names(result, "finalDiagnosis", limit=5),
            "interpretation_note": note,
        }
        for col in CATEGORY_COLS:
            case[col] = as_int(ic.get(col))
        cases.append(case)

    case_rows = []
    for case in cases:
        row = dict(case)
        row["present_hpo_ids"] = ";".join(case["present_hpo_ids"])
        case_rows.append(row)

    summaries = group_summary(cases)
    top_hpos = top_terms_by_group(cases, terms)

    write_csv(OUT_DIR / "case_group_table.csv", case_rows)
    write_csv(OUT_DIR / "group_summary.csv", summaries)
    write_csv(OUT_DIR / "hpo_group_signal.csv", top_hpos)
    write_markdown(cases, summaries, top_hpos)

    print(f"Wrote {OUT_DIR.relative_to(ROOT)}")
    print(f"Cases: {len(cases)}")
    for s in summaries:
        print(f"{s['group']}: {s['case_count']} cases")


if __name__ == "__main__":
    main()
