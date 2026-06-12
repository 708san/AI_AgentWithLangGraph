#!/usr/bin/env python3
"""Audit cases where an upstream tool found the exact disease but Final missed it.

The script is intentionally dependency-free so it can run in the minimal project
environment used for validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from textwrap import shorten


INITIAL_TOOLS = ["PubCaseFinder", "PhenotypeSearch", "ZeroShot", "GestaltMatcher"]
RANK_TOOLS = INITIAL_TOOLS + ["TentativeDiagnosis", "FinalDiagnosis"]

GENERIC_TOKENS = {
    "AND",
    "AUTOSOMAL",
    "CANDIDATE",
    "DEFICIENCY",
    "DEVELOPMENTAL",
    "DISEASE",
    "DISORDER",
    "DOMINANT",
    "INTELLECTUAL",
    "LINKED",
    "MENTAL",
    "NEURODEVELOPMENTAL",
    "PRIMARY",
    "RECESSIVE",
    "SYNDROME",
    "TYPE",
    "WITH",
}


DISEASE_NOTES = {
    "618164": (
        "CAFDADD / TRAF7関連疾患。心奇形、顔貌、指趾異常、発達遅滞を軸にした"
        "多発奇形・神経発達症候群で、症例側の所見が広いNDD/先天異常として扱われると"
        "他の症候群に押されやすい。"
    ),
    "618505": (
        "Stolerman neurodevelopmental syndrome (NEDSST) / UNC13A関連。"
        "粗な顔貌、軽度遠位骨格異常、発達遅滞などを含むが、入力所見が"
        "発達遅滞・顔貌中心だと特異的特徴が不足すると判定されやすい。"
    ),
    "609942": (
        "Noonan syndrome 3 / KRAS関連RASopathy。Noonan/CFC/Costelloなど近縁疾患と"
        "表現型が重なり、FinalでNoonan syndrome 1など近縁OMIMへ寄ることで"
        "Exactでは落ちやすい。"
    ),
    "305400": (
        "Aarskog-Scott syndrome / FGD1関連X連鎖疾患。FinalではAarskog関連名を"
        "出していても別OMIMへ寄ると、臨床的には近いがExact判定では失敗になる。"
    ),
    "616364": (
        "White-Sutton syndrome / POGZ関連NDD。発達遅滞、言語遅滞、行動特徴、視覚異常などが"
        "合う一方、KBG、Au-Kline、CdLS/CSS系など広いNDD鑑別にFinalで押されることがある。"
    ),
    "605282": (
        "Temtamy preaxial brachydactyly syndrome。前軸性短指・顔貌・発達遅滞などを軸にするが、"
        "手足の特徴が十分に強調されないとreflectionで表面的類似として落ちやすい。"
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_omim(value: object) -> str:
    match = re.search(r"(\d{3,})", str(value or ""))
    return match.group(1) if match else ""


def parse_rank(value: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        rank = float(value)
    except ValueError:
        return None
    if rank != rank:
        return None
    return int(rank)


def norm_text(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", " ", str(value or "").upper())
    return re.sub(r"\s+", " ", text).strip()


def tokens(value: object) -> set[str]:
    return {
        token
        for token in norm_text(value).split()
        if len(token) > 1 and token not in GENERIC_TOKENS
    }


def text_similarity(left: object, right: object) -> float:
    left_norm = norm_text(left)
    right_norm = norm_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm in right_norm or right_norm in left_norm:
        return 1.0

    left_tokens = tokens(left_norm)
    right_tokens = tokens(right_norm)
    jaccard = 0.0
    if left_tokens and right_tokens:
        jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        short_tokens = left_tokens if len(left_tokens) <= len(right_tokens) else right_tokens
        if short_tokens and short_tokens <= (left_tokens | right_tokens):
            if short_tokens <= left_tokens and short_tokens <= right_tokens:
                jaccard = max(jaccard, 0.95)

    ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    return max(jaccard, ratio)


def aliases_for(*names: object) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for name in names:
        for part in re.split(r"[;/()]", str(name or "")):
            cleaned = norm_text(part)
            if cleaned and cleaned not in seen:
                aliases.append(cleaned)
                seen.add(cleaned)
        cleaned = norm_text(name)
        if cleaned and cleaned not in seen:
            aliases.append(cleaned)
            seen.add(cleaned)
    return aliases


def best_reflection_match(reflection_items: list[dict], aliases: list[str]) -> tuple[dict | None, float]:
    best_item = None
    best_score = 0.0
    for item in reflection_items:
        disease_name = item.get("disease_name", "")
        score = max((text_similarity(alias, disease_name) for alias in aliases), default=0.0)
        if score > best_score:
            best_item = item
            best_score = score
    if best_score < 0.58:
        return None, best_score
    return best_item, best_score


def load_csv(path: Path, delimiter: str = ",") -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def load_truth(path: Path) -> dict[str, dict]:
    rows = load_csv(path, delimiter="\t")
    return {str(row["patient_id"]): row for row in rows}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def ans_items(section: object) -> list[dict]:
    if isinstance(section, dict):
        items = section.get("ans", [])
        return items if isinstance(items, list) else []
    return section if isinstance(section, list) else []


def candidate_rank_by_omim(items: list[dict], true_omim: str) -> tuple[int | None, dict | None]:
    matches = []
    for index, item in enumerate(items, start=1):
        if normalize_omim(item.get("OMIM_id") or item.get("omim_id")) == true_omim:
            rank = parse_rank(str(item.get("rank", ""))) or index
            matches.append((rank, item))
    if not matches:
        return None, None
    return sorted(matches, key=lambda pair: pair[0])[0]


def tool_summary(row: dict, suffix: str = "Match_rank", threshold: int = 5) -> list[str]:
    found = []
    for tool in INITIAL_TOOLS:
        rank = parse_rank(row.get(f"{tool}_{suffix}", ""))
        if rank is not None and rank <= threshold:
            found.append(f"{tool}:R{rank}")
    return found


def final_candidates(ans_file: Path) -> str:
    if not ans_file.exists():
        return ""
    data = load_json(ans_file)
    items = ans_items(data.get("finalDiagnosis") or data.get("ans"))
    if not items:
        items = ans_items(data.get("ans"))
    labels = []
    for index, item in enumerate(items, start=1):
        rank = parse_rank(str(item.get("rank", ""))) or index
        omim = normalize_omim(item.get("OMIM_id") or item.get("omim_id"))
        labels.append(f"{rank}. {item.get('disease_name','')} (OMIM:{omim})")
    return " | ".join(labels)


def likely_issue(true_omim: str, stage: str, final_close_rank: int | None) -> str:
    if stage == "tentative_absent":
        return "初期ツールでは正解が出たが、merged/tentative候補生成の段階で正解OMIMが採用されていない。"
    if stage == "reflection_false":
        return "reflectionで疾患特異的な所見が不足、または矛盾所見があると判断され、Final前に除外されている。"
    if true_omim in {"609942", "305400"} or final_close_rank is not None:
        return "reflection後には残ったが、Finalで近縁疾患・別サブタイプへ寄り、Exact OMIMが落ちている。"
    return "reflection後には残ったが、Finalの再ランキング/候補数制限で別の鑑別が優先されている。"


def analyze(args: argparse.Namespace) -> tuple[list[dict], dict]:
    base = args.repo_root
    mondo_rows = load_csv(args.mondo_csv)
    truth = load_truth(args.truth_tsv)

    audit_rows: list[dict] = []
    rank_summary = {tool: {"match": 0, "close_only": 0, "total": 0} for tool in RANK_TOOLS}

    for row in mondo_rows:
        patient_id = str(row["patient_id"])
        for tool in RANK_TOOLS:
            match_rank = parse_rank(row.get(f"{tool}_Match_rank", ""))
            close_rank = parse_rank(row.get(f"{tool}_Close_rank", ""))
            if match_rank is not None and match_rank <= args.rank_threshold:
                rank_summary[tool]["match"] += 1
            elif close_rank is not None and close_rank <= args.rank_threshold:
                rank_summary[tool]["close_only"] += 1
            if (match_rank is not None and match_rank <= args.rank_threshold) or (
                close_rank is not None and close_rank <= args.rank_threshold
            ):
                rank_summary[tool]["total"] += 1

        initial_exact = tool_summary(row, "Match_rank", args.rank_threshold)
        final_match_rank = parse_rank(row.get("FinalDiagnosis_Match_rank", ""))
        if not initial_exact or (final_match_rank is not None and final_match_rank <= args.rank_threshold):
            continue

        truth_row = truth.get(patient_id, {})
        true_omim = normalize_omim(truth_row.get("omim_ids", ""))
        true_name = truth_row.get("disorder_names", "")

        res_path = args.res_dir / f"{patient_id}.json"
        ans_path = args.ans_dir / f"{patient_id}.json"
        if not res_path.exists():
            continue

        res_data = load_json(res_path)
        tentative_items = ans_items(res_data.get("tentativeDiagnosis"))
        final_items = ans_items(res_data.get("finalDiagnosis"))
        if ans_path.exists():
            ans_data = load_json(ans_path)
            final_items_from_ans = ans_items(ans_data.get("ans"))
            if final_items_from_ans:
                final_items = final_items_from_ans

        tentative_rank, tentative_item = candidate_rank_by_omim(tentative_items, true_omim)
        final_rank, final_item = candidate_rank_by_omim(final_items, true_omim)
        final_close_rank = parse_rank(row.get("FinalDiagnosis_Close_rank", ""))

        reflection_items = ans_items(res_data.get("reflection"))
        reflection_item = None
        reflection_score = 0.0
        if tentative_item:
            aliases = aliases_for(
                true_name,
                tentative_item.get("disease_name", ""),
                tentative_item.get("omim_disease_name_en", ""),
            )
            reflection_item, reflection_score = best_reflection_match(reflection_items, aliases)

        if tentative_item is None:
            stage = "tentative_absent"
        elif reflection_item is not None and reflection_item.get("Correctness") is False:
            stage = "reflection_false"
        else:
            stage = "final_omitted"

        reflection_correctness = ""
        reflection_name = ""
        reflection_analysis = ""
        if reflection_item:
            reflection_correctness = str(reflection_item.get("Correctness", ""))
            reflection_name = reflection_item.get("disease_name", "")
            reflection_analysis = reflection_item.get("DiagnosisAnalysis", "")

        audit_rows.append(
            {
                "patient_id": patient_id,
                "true_omim": f"OMIM:{true_omim}" if true_omim else "",
                "true_disease": true_name,
                "disease_note": DISEASE_NOTES.get(true_omim, ""),
                "initial_exact_tools": "; ".join(initial_exact),
                "initial_close_tools": "; ".join(tool_summary(row, "Close_rank", args.rank_threshold)),
                "tentative_rank": tentative_rank or "",
                "tentative_disease": tentative_item.get("disease_name", "") if tentative_item else "",
                "reflection_match_score": f"{reflection_score:.2f}" if reflection_item else f"{reflection_score:.2f}",
                "reflection_disease": reflection_name,
                "reflection_correctness": reflection_correctness,
                "reflection_analysis": reflection_analysis,
                "final_exact_rank": final_rank or "",
                "final_close_rank": final_close_rank or "",
                "final_candidates": final_candidates(ans_path),
                "drop_stage": stage,
                "likely_issue": likely_issue(true_omim, stage, final_close_rank),
            }
        )

    summary = {
        "total_cases": len(mondo_rows),
        "rank_threshold": args.rank_threshold,
        "drop_count": len(audit_rows),
        "stage_counts": Counter(row["drop_stage"] for row in audit_rows),
        "disease_counts": Counter(row["true_disease"] for row in audit_rows),
        "rank_summary": rank_summary,
    }
    return sorted(audit_rows, key=lambda row: (row["drop_stage"], row["true_omim"], int(row["patient_id"]))), summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def grouped_rows(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(grouped)


def write_markdown(path: Path, rows: list[dict], summary: dict) -> None:
    lines: list[str] = []
    lines.append("# Final Drop Case Audit")
    lines.append("")
    lines.append(
        f"- 対象: 初期4ツールのいずれかがExact rank<= {summary['rank_threshold']} で正解したが、FinalDiagnosisのExact rank<= {summary['rank_threshold']} に入らなかったケース"
    )
    lines.append(f"- 総症例数: {summary['total_cases']}")
    lines.append(f"- 該当ケース: {summary['drop_count']}")
    lines.append("")
    lines.append("## correct_disease_rank_overview の集計確認")
    lines.append("")
    total = summary["total_cases"]
    for tool in RANK_TOOLS:
        values = summary["rank_summary"][tool]
        percent = values["total"] / total * 100 if total else 0
        lines.append(
            f"- {tool}: Exact {values['match']}/{total}, Close-only {values['close_only']}/{total}, "
            f"Exact+Close {values['total']}/{total} ({percent:.1f}%)"
        )
    lines.append("")
    lines.append("## 落ちた段階")
    lines.append("")
    for stage, count in summary["stage_counts"].most_common():
        lines.append(f"- {stage}: {count}")
    for stage in ["tentative_absent", "reflection_false", "final_omitted"]:
        if stage not in summary["stage_counts"]:
            lines.append(f"- {stage}: 0")
    lines.append("")
    lines.append("## 疾患別の傾向")
    lines.append("")
    for disease, disease_rows in grouped_rows(rows, "true_disease").items():
        true_omim = disease_rows[0]["true_omim"]
        note = disease_rows[0]["disease_note"]
        stage_counts = Counter(row["drop_stage"] for row in disease_rows)
        stage_text = ", ".join(f"{stage}={count}" for stage, count in stage_counts.items())
        lines.append(f"### {disease} ({true_omim})")
        lines.append("")
        lines.append(f"- 件数: {len(disease_rows)} ({stage_text})")
        if note:
            lines.append(f"- 疾患の性質: {note}")
        lines.append("")
        for row in disease_rows:
            refl = row["reflection_correctness"] or "not matched"
            analysis = shorten(row["reflection_analysis"].replace("\n", " "), width=360, placeholder="...")
            lines.append(
                f"- patient_id {row['patient_id']}: stage={row['drop_stage']}; "
                f"initial={row['initial_exact_tools']}; tentative_rank={row['tentative_rank'] or '-'}; "
                f"reflection={refl}; final_close_rank={row['final_close_rank'] or '-'}"
            )
            if row["reflection_disease"]:
                lines.append(f"  - reflection matched: {row['reflection_disease']}")
            lines.append(f"  - 推定要因: {row['likely_issue']}")
            if analysis:
                lines.append(f"  - reflection根拠要約: {analysis}")
            lines.append(f"  - final candidates: {row['final_candidates']}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    base = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=base)
    parser.add_argument("--mondo-csv", type=Path, default=base / "MONDO_BASE_match_5-2.csv")
    parser.add_argument("--truth-tsv", type=Path, default=base / "sampleData/ValidationDataWithoutDupli_newest.tsv")
    parser.add_argument("--res-dir", type=Path, default=base / "res_5-2")
    parser.add_argument("--ans-dir", type=Path, default=base / "ans_5-2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "validationCode/tool_difference_visualization_5-2",
    )
    parser.add_argument("--rank-threshold", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = analyze(args)
    write_csv(args.output_dir / "final_drop_case_audit.csv", rows)
    write_markdown(args.output_dir / "final_drop_case_audit.md", rows, summary)

    print(f"Audited {summary['drop_count']} final-drop cases from {summary['total_cases']} total cases.")
    print("Stage counts:")
    for stage in ["tentative_absent", "reflection_false", "final_omitted"]:
        print(f"  {stage}: {summary['stage_counts'].get(stage, 0)}")
    print(f"Wrote {args.output_dir / 'final_drop_case_audit.csv'}")
    print(f"Wrote {args.output_dir / 'final_drop_case_audit.md'}")


if __name__ == "__main__":
    main()
