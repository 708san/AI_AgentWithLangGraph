import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch


DEFAULT_EXCLUDED_TOOLS = {"TentativeDiagnosis", "FinalDiagnosis"}
DEFAULT_TOOL_ORDER = ["PubCaseFinder", "PhenotypeSearch", "ZeroShot", "GestaltMatcher"]
DEFAULT_RANK_TOOL_ORDER = DEFAULT_TOOL_ORDER + ["TentativeDiagnosis", "FinalDiagnosis"]
STATUS_LABELS = {
    0: "No hit",
    1: "Close only",
    2: "Exact",
}
STATUS_COLORS = ["#d9d9d9", "#f6b26b", "#2f7ed8"]
TOOL_COLORS = {
    "PubCaseFinder": "#ff7f7f",
    "PhenotypeSearch": "#7fbf7f",
    "ZeroShot": "#7f7fff",
    "GestaltMatcher": "#d67fd6",
    "TentativeDiagnosis": "#9467bd",
    "FinalDiagnosis": "#4c78a8",
}
TOOL_ABBREVIATIONS = {
    "PubCaseFinder": "PCF",
    "PhenotypeSearch": "PS",
    "ZeroShot": "ZS",
    "GestaltMatcher": "GM",
    "TentativeDiagnosis": "Tent",
    "FinalDiagnosis": "Final",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def format_patient_id(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def infer_tools(df: pd.DataFrame, excluded_tools: set[str]) -> list[str]:
    tools = []
    for column in df.columns:
        if column.endswith("_Match_rank"):
            tool = column[: -len("_Match_rank")]
            if tool not in excluded_tools and f"{tool}_Close_rank" in df.columns:
                tools.append(tool)

    ordered = [tool for tool in DEFAULT_TOOL_ORDER if tool in tools]
    ordered.extend(sorted(tool for tool in tools if tool not in ordered))
    return ordered


def infer_rank_tools(df: pd.DataFrame) -> list[str]:
    tools = []
    for column in df.columns:
        if column.endswith("_Match_rank"):
            tool = column[: -len("_Match_rank")]
            if f"{tool}_Close_rank" in df.columns:
                tools.append(tool)

    ordered = [tool for tool in DEFAULT_RANK_TOOL_ORDER if tool in tools]
    ordered.extend(sorted(tool for tool in tools if tool not in ordered))
    return ordered


def status_for_tool(df: pd.DataFrame, tool: str, rank_threshold: int) -> pd.Series:
    match_rank = pd.to_numeric(df[f"{tool}_Match_rank"], errors="coerce")
    close_rank = pd.to_numeric(df[f"{tool}_Close_rank"], errors="coerce")

    exact = match_rank.le(rank_threshold)
    close_only = close_rank.le(rank_threshold) & ~exact

    status = pd.Series(0, index=df.index, dtype=int)
    status.loc[close_only.fillna(False)] = 1
    status.loc[exact.fillna(False)] = 2
    return status


def rank_summary(row: pd.Series, tool: str) -> str:
    match_rank = row.get(f"{tool}_Match_rank")
    close_rank = row.get(f"{tool}_Close_rank")
    parts = []
    if pd.notna(match_rank):
        parts.append(f"M{int(match_rank)}")
    if pd.notna(close_rank):
        parts.append(f"C{int(close_rank)}")
    return "/".join(parts) if parts else "-"


def build_patient_status(df: pd.DataFrame, tools: list[str], rank_threshold: int) -> pd.DataFrame:
    status_df = pd.DataFrame({"patient_id": df["patient_id"].map(format_patient_id)})
    for tool in tools:
        status_df[f"{tool}_status_code"] = status_for_tool(df, tool, rank_threshold)
        status_df[f"{tool}_status"] = status_df[f"{tool}_status_code"].map(STATUS_LABELS)
        status_df[f"{tool}_rank"] = df.apply(lambda row: rank_summary(row, tool), axis=1)

    exact_tools = []
    close_only_tools = []
    hit_tools = []
    missed_tools = []
    status_patterns = []
    for _, row in status_df.iterrows():
        exact = [tool for tool in tools if row[f"{tool}_status_code"] == 2]
        close = [tool for tool in tools if row[f"{tool}_status_code"] == 1]
        hit = exact + close
        missed = [tool for tool in tools if row[f"{tool}_status_code"] == 0]
        pattern = " | ".join(f"{tool}:{row[f'{tool}_status']}" for tool in tools)

        exact_tools.append(";".join(exact) if exact else "-")
        close_only_tools.append(";".join(close) if close else "-")
        hit_tools.append(";".join(hit) if hit else "-")
        missed_tools.append(";".join(missed) if missed else "-")
        status_patterns.append(pattern)

    status_df["exact_tools"] = exact_tools
    status_df["close_only_tools"] = close_only_tools
    status_df["hit_tools"] = hit_tools
    status_df["missed_tools"] = missed_tools
    status_df["status_pattern"] = status_patterns
    status_df["hit_tool_count"] = [
        sum(row[f"{tool}_status_code"] > 0 for tool in tools)
        for _, row in status_df.iterrows()
    ]
    status_df["exact_tool_count"] = [
        sum(row[f"{tool}_status_code"] == 2 for tool in tools)
        for _, row in status_df.iterrows()
    ]
    return status_df


def build_long_status(status_df: pd.DataFrame, tools: list[str]) -> pd.DataFrame:
    rows = []
    for _, row in status_df.iterrows():
        for tool in tools:
            rows.append(
                {
                    "patient_id": row["patient_id"],
                    "tool": tool,
                    "status_code": row[f"{tool}_status_code"],
                    "status": row[f"{tool}_status"],
                    "rank": row[f"{tool}_rank"],
                }
            )
    return pd.DataFrame(rows)


def build_pairwise_differences(status_df: pd.DataFrame, tools: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    summary_rows = []

    for left, right in itertools.combinations(tools, 2):
        left_code = status_df[f"{left}_status_code"]
        right_code = status_df[f"{right}_status_code"]
        left_hit = left_code.gt(0)
        right_hit = right_code.gt(0)

        both_hit = left_hit & right_hit
        only_left = left_hit & ~right_hit
        only_right = ~left_hit & right_hit
        both_miss = ~left_hit & ~right_hit
        different_strength = both_hit & left_code.ne(right_code)

        summary_rows.append(
            {
                "tool_pair": f"{left} vs {right}",
                "both_hit": int(both_hit.sum()),
                "only_left_hit": int(only_left.sum()),
                "only_right_hit": int(only_right.sum()),
                "both_miss": int(both_miss.sum()),
                "different_strength_among_both_hit": int(different_strength.sum()),
            }
        )

        diff_mask = only_left | only_right | different_strength
        for _, row in status_df.loc[diff_mask].iterrows():
            if row[f"{left}_status_code"] > 0 and row[f"{right}_status_code"] == 0:
                difference_type = f"{left} only"
            elif row[f"{left}_status_code"] == 0 and row[f"{right}_status_code"] > 0:
                difference_type = f"{right} only"
            else:
                difference_type = "both hit but exact/close differs"

            detail_rows.append(
                {
                    "patient_id": row["patient_id"],
                    "tool_pair": f"{left} vs {right}",
                    "left_tool": left,
                    "right_tool": right,
                    "difference_type": difference_type,
                    "left_status": row[f"{left}_status"],
                    "left_rank": row[f"{left}_rank"],
                    "right_status": row[f"{right}_status"],
                    "right_rank": row[f"{right}_rank"],
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def build_combination_counts(status_df: pd.DataFrame, tools: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in status_df.iterrows():
        hit_tools = [tool for tool in tools if row[f"{tool}_status_code"] > 0]
        exact_tools = [tool for tool in tools if row[f"{tool}_status_code"] == 2]
        rows.append(
            {
                "hit_combination": " + ".join(hit_tools) if hit_tools else "No tool hit",
                "exact_combination": " + ".join(exact_tools) if exact_tools else "No exact hit",
            }
        )

    hit_counts = rows_to_counts(rows, "hit_combination")
    exact_counts = rows_to_counts(rows, "exact_combination")
    return hit_counts, exact_counts


def rows_to_counts(rows: list[dict], key: str) -> pd.DataFrame:
    series = pd.Series([row[key] for row in rows], name=key)
    counts = series.value_counts().rename(f"{key}_count").to_frame()
    counts.index.name = "combination"
    return counts


def save_status_matrix(status_df: pd.DataFrame, tools: list[str], output_path: Path) -> None:
    sort_cols = [f"{tool}_status_code" for tool in tools]
    sorted_df = status_df.sort_values(sort_cols + ["patient_id"], ascending=[False] * len(sort_cols) + [True])
    matrix = sorted_df[sort_cols].to_numpy()

    fig_height = max(8, len(sorted_df) * 0.16)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap=ListedColormap(STATUS_COLORS), vmin=0, vmax=2)

    ax.set_xticks(np.arange(len(tools)))
    ax.set_xticklabels(tools, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(sorted_df)))
    ax.set_yticklabels(sorted_df["patient_id"].astype(str), fontsize=6)
    ax.set_xlabel("Tool")
    ax.set_ylabel("patient_id")
    ax.set_title("Per-patient tool status excluding Tentative/Final")

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.04, pad=0.02)
    cbar.ax.set_yticklabels([STATUS_LABELS[i] for i in [0, 1, 2]])

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_case_result_overview(status_df: pd.DataFrame, tools: list[str], output_path: Path) -> None:
    sort_cols = ["hit_tool_count", "exact_tool_count"] + [f"{tool}_status_code" for tool in tools]
    ascending = [False, False] + [False] * len(tools)
    sorted_df = status_df.sort_values(sort_cols + ["patient_id"], ascending=ascending + [True])
    matrix = sorted_df[[f"{tool}_status_code" for tool in tools]].to_numpy()

    fig_height = max(11, len(sorted_df) * 0.28)
    fig_width = max(10, len(tools) * 1.8 + 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(matrix, aspect="auto", cmap=ListedColormap(STATUS_COLORS), vmin=0, vmax=2)

    for y, (_, row) in enumerate(sorted_df.iterrows()):
        for x, tool in enumerate(tools):
            code = row[f"{tool}_status_code"]
            rank = row[f"{tool}_rank"]
            label = rank if code > 0 else "-"
            color = "white" if code == 2 else "black"
            ax.text(x, y, label, ha="center", va="center", fontsize=7, color=color, fontweight="bold")

        summary = f"{row['hit_tool_count']} hit / {row['exact_tool_count']} exact"
        ax.text(len(tools) + 0.15, y, summary, ha="left", va="center", fontsize=7)

    ax.set_xticks(np.arange(len(tools)))
    ax.set_xticklabels(tools, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(sorted_df)))
    ax.set_yticklabels(sorted_df["patient_id"].astype(str), fontsize=6)
    ax.set_xlim(-0.5, len(tools) + 1.7)
    ax.set_xlabel("Tool")
    ax.set_ylabel("patient_id")
    ax.set_title("Case-by-case tool results (cell text: M=Exact rank, C=Close rank)")

    ax.set_xticks(np.arange(-0.5, len(tools), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sorted_df), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    legend_handles = [
        Patch(facecolor=STATUS_COLORS[2], label="Exact"),
        Patch(facecolor=STATUS_COLORS[1], label="Close only"),
        Patch(facecolor=STATUS_COLORS[0], label="No hit"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1))

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def save_correct_rank_overview(df: pd.DataFrame, tools: list[str], output_path: Path) -> None:
    rows = []
    for _, row in df.iterrows():
        item = {"patient_id": format_patient_id(row["patient_id"])}
        exact_count = 0
        for tool in tools:
            match_rank = pd.to_numeric(pd.Series([row.get(f"{tool}_Match_rank")]), errors="coerce").iloc[0]
            close_rank = pd.to_numeric(pd.Series([row.get(f"{tool}_Close_rank")]), errors="coerce").iloc[0]
            if pd.notna(match_rank):
                code = 2
                label = f"M{int(match_rank)}"
                sort_rank = int(match_rank)
                exact_count += 1
            elif pd.notna(close_rank):
                code = 1
                label = f"C{int(close_rank)}"
                sort_rank = 500 + int(close_rank)
            else:
                code = 0
                label = "-"
                sort_rank = 999
            item[f"{tool}_status_code"] = code
            item[f"{tool}_label"] = label
            item[f"{tool}_sort_rank"] = sort_rank
        item["exact_tool_count"] = exact_count
        rows.append(item)

    rank_df = pd.DataFrame(rows)
    sort_by = []
    ascending = []
    for preferred_tool in ["FinalDiagnosis", "TentativeDiagnosis", "PubCaseFinder"]:
        if preferred_tool in tools:
            sort_by.append(f"{preferred_tool}_sort_rank")
            ascending.append(True)
    sort_by.extend(["exact_tool_count", "patient_id"])
    ascending.extend([False, True])
    rank_df = rank_df.sort_values(sort_by, ascending=ascending)

    matrix = rank_df[[f"{tool}_status_code" for tool in tools]].to_numpy()
    fig_height = max(12, len(rank_df) * 0.28)
    fig_width = max(12, len(tools) * 1.65)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(matrix, aspect="auto", cmap=ListedColormap(STATUS_COLORS), vmin=0, vmax=2)

    for y, (_, row) in enumerate(rank_df.iterrows()):
        for x, tool in enumerate(tools):
            code = row[f"{tool}_status_code"]
            color = "white" if code == 2 else "black"
            ax.text(
                x,
                y,
                row[f"{tool}_label"],
                ha="center",
                va="center",
                fontsize=7,
                color=color,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(len(tools)))
    ax.set_xticklabels(tools, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(rank_df)))
    ax.set_yticklabels(rank_df["patient_id"].astype(str), fontsize=6)
    ax.set_xlabel("Tool")
    ax.set_ylabel("patient_id")
    ax.set_title("Correct disease rank by tool (M=exact rank, C=close-only rank)")

    summary_lines = []
    total_cases = len(df)
    for tool in tools:
        exact_count = int((pd.to_numeric(df[f"{tool}_Match_rank"], errors="coerce") <= 5).sum())
        close_count = int(
            (
                (pd.to_numeric(df[f"{tool}_Close_rank"], errors="coerce") <= 5)
                & ~(pd.to_numeric(df[f"{tool}_Match_rank"], errors="coerce") <= 5)
            ).sum()
        )
        total_count = exact_count + close_count
        summary_lines.append(
            f"{TOOL_ABBREVIATIONS.get(tool, tool)}: "
            f"M {exact_count}/{total_cases}, C {close_count}/{total_cases}, "
            f"total {total_count / total_cases * 100:.1f}%"
        )
    fig.text(
        0.5,
        0.995,
        "Rank <= 5 summary: " + " | ".join(summary_lines),
        ha="center",
        va="top",
        fontsize=8,
        color="#333333",
    )

    ax.set_xticks(np.arange(-0.5, len(tools), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rank_df), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    legend_handles = [
        Patch(facecolor=STATUS_COLORS[2], label="Exact disease found"),
        Patch(facecolor=STATUS_COLORS[1], label="Close disease only"),
        Patch(facecolor=STATUS_COLORS[0], label="No exact/close rank"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1))

    plt.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def exclusive_combination_counts(
    status_df: pd.DataFrame,
    tools: list[str],
    mode: str,
) -> dict[tuple[str, ...], int]:
    counts = {combo: 0 for r in range(1, len(tools) + 1) for combo in itertools.combinations(tools, r)}
    counts[()] = 0

    for _, row in status_df.iterrows():
        if mode == "exact":
            active = tuple(tool for tool in tools if row[f"{tool}_status_code"] == 2)
        elif mode == "hit":
            active = tuple(tool for tool in tools if row[f"{tool}_status_code"] > 0)
        else:
            raise ValueError(f"Unknown Venn mode: {mode}")
        counts[active] = counts.get(active, 0) + 1
    return counts


def save_four_tool_venn(status_df: pd.DataFrame, tools: list[str], output_path: Path) -> None:
    if len(tools) != 4:
        print("Four-tool Venn output requires exactly 4 tools. Skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 13), gridspec_kw={"width_ratios": [1.05, 1.15]})
    panels = [
        ("Exact only", exclusive_combination_counts(status_df, tools, mode="exact")),
        ("Exact + Close", exclusive_combination_counts(status_df, tools, mode="hit")),
    ]

    for row_index, (title, counts) in enumerate(panels):
        draw_fixed_four_set_venn(axes[row_index, 0], tools, counts, title, len(status_df))
        draw_exclusive_combination_bars(axes[row_index, 1], tools, counts, len(status_df), title)

    fig.suptitle(
        "4-tool overlap overview",
        fontsize=16,
        fontweight="bold",
    )
    abbreviation_text = " / ".join(f"{TOOL_ABBREVIATIONS.get(tool, tool)}={tool}" for tool in tools)
    fig.text(
        0.5,
        0.025,
        abbreviation_text,
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.text(
        0.5,
        0.005,
        "The circles are fixed visual guides. The bars show exclusive case counts for each exact tool combination.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def draw_fixed_four_set_venn(
    ax,
    tools: list[str],
    counts: dict[tuple[str, ...], int],
    title: str,
    total_cases: int,
) -> None:
    circle_specs = {
        tools[0]: {"xy": (-0.72, 0.18), "color": TOOL_COLORS.get(tools[0], "#ff7f7f")},
        tools[1]: {"xy": (0.72, 0.18), "color": TOOL_COLORS.get(tools[1], "#7fbf7f")},
        tools[2]: {"xy": (-0.30, -0.58), "color": TOOL_COLORS.get(tools[2], "#7f7fff")},
        tools[3]: {"xy": (0.30, -0.58), "color": TOOL_COLORS.get(tools[3], "#d67fd6")},
    }

    for tool, spec in circle_specs.items():
        ax.add_patch(
            Circle(
                spec["xy"],
                radius=0.95,
                facecolor=spec["color"],
                edgecolor=spec["color"],
                alpha=0.45,
                linewidth=2,
            )
        )

    tool_label_positions = {
        tools[0]: (-1.45, 1.28),
        tools[1]: (1.45, 1.28),
        tools[2]: (-1.10, -1.72),
        tools[3]: (1.10, -1.72),
    }
    for tool, (x, y) in tool_label_positions.items():
        label = f"{TOOL_ABBREVIATIONS.get(tool, tool)}\n{tool}"
        ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold")

    label_positions = {
        (tools[0],): (-1.48, 0.48),
        (tools[1],): (1.48, 0.48),
        (tools[2],): (-0.82, -1.36),
        (tools[3],): (0.82, -1.36),
        (tools[0], tools[1]): (0.0, 0.84),
        (tools[0], tools[2]): (-0.98, -0.36),
        (tools[0], tools[3]): (-0.34, -0.02),
        (tools[1], tools[2]): (0.34, -0.02),
        (tools[1], tools[3]): (0.98, -0.36),
        (tools[2], tools[3]): (0.0, -1.08),
        (tools[0], tools[1], tools[2]): (-0.38, 0.34),
        (tools[0], tools[1], tools[3]): (0.38, 0.34),
        (tools[0], tools[2], tools[3]): (-0.36, -0.58),
        (tools[1], tools[2], tools[3]): (0.36, -0.58),
        (tools[0], tools[1], tools[2], tools[3]): (0.0, -0.22),
    }

    for combo, (x, y) in label_positions.items():
        count = counts.get(combo, 0)
        label = "All 4\n" + str(count) if len(combo) == 4 else str(count)
        is_zero = count == 0
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=12 if len(combo) == 4 else 10,
            fontweight="bold" if not is_zero or len(combo) == 4 else "normal",
            color="#777777" if is_zero else "#111111",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#777777" if len(combo) == 4 else "none",
                "alpha": 0.88,
            },
        )

    no_tool_count = counts.get((), 0)
    ax.text(
        0.0,
        -2.08,
        f"No tool: {no_tool_count} / {total_cases}",
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#eeeeee", "edgecolor": "#999999"},
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-2.25, 1.55)
    ax.axis("off")


def combo_label(combo: tuple[str, ...]) -> str:
    if not combo:
        return "No tool"
    return " + ".join(TOOL_ABBREVIATIONS.get(tool, tool) for tool in combo)


def draw_exclusive_combination_bars(
    ax,
    tools: list[str],
    counts: dict[tuple[str, ...], int],
    total_cases: int,
    title: str,
) -> None:
    nonzero = [(combo, count) for combo, count in counts.items() if count > 0]
    nonzero.sort(key=lambda item: (item[1], len(item[0]), combo_label(item[0])), reverse=True)

    labels = [combo_label(combo) for combo, _ in nonzero]
    values = [count for _, count in nonzero]
    y_pos = np.arange(len(nonzero))

    colors = []
    for combo, _ in nonzero:
        if not combo:
            colors.append("#bdbdbd")
        elif len(combo) == 1:
            colors.append(TOOL_COLORS.get(combo[0], "#4c78a8"))
        else:
            colors.append("#4c78a8")

    ax.barh(y_pos, values, color=colors, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Cases")
    ax.set_title(f"{title}: exclusive combinations", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(values) + 4 if values else 5)
    ax.grid(axis="x", linestyle=":", alpha=0.45)

    for y, value in zip(y_pos, values):
        ax.text(
            value + 0.25,
            y,
            f"{value} ({value / total_cases * 100:.1f}%)",
            va="center",
            fontsize=9,
        )


def save_pairwise_heatmap(pairwise_summary: pd.DataFrame, tools: list[str], output_path: Path) -> None:
    jaccard = pd.DataFrame(np.eye(len(tools)), index=tools, columns=tools)
    only_difference = pd.DataFrame(np.zeros((len(tools), len(tools))), index=tools, columns=tools)

    for _, row in pairwise_summary.iterrows():
        left, right = row["tool_pair"].split(" vs ")
        both_hit = row["both_hit"]
        only_left = row["only_left_hit"]
        only_right = row["only_right_hit"]
        union = both_hit + only_left + only_right
        score = both_hit / union if union else 0.0
        diff_count = only_left + only_right + row["different_strength_among_both_hit"]
        jaccard.loc[left, right] = score
        jaccard.loc[right, left] = score
        only_difference.loc[left, right] = diff_count
        only_difference.loc[right, left] = diff_count

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    draw_annotated_heatmap(axes[0], jaccard, "Hit overlap Jaccard", fmt="{:.2f}", cmap="Blues")
    draw_annotated_heatmap(axes[1], only_difference, "Different patients count", fmt="{:.0f}", cmap="Reds")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def draw_annotated_heatmap(ax, data: pd.DataFrame, title: str, fmt: str, cmap: str) -> None:
    im = ax.imshow(data.to_numpy(), cmap=cmap)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(data.columns, rotation=30, ha="right")
    ax.set_yticklabels(data.index)
    ax.set_title(title)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            ax.text(x, y, fmt.format(data.iloc[y, x]), ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_combination_bar(combination_counts: pd.DataFrame, output_path: Path, total_cases: int) -> None:
    data = combination_counts.sort_values("hit_combination_count", ascending=True)

    fig_height = max(6, len(data) * 0.32)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    y_pos = np.arange(len(data))
    counts = data["hit_combination_count"].to_numpy()

    ax.barh(y_pos, counts, color="#4c78a8")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data.index)
    ax.set_xlabel("Number of patients")
    ax.set_title("Hit tool combinations (Exact or Close within threshold)")

    for y, count in zip(y_pos, counts):
        ax.text(count + 0.2, y, f"{count} ({count / total_cases * 100:.1f}%)", va="center", fontsize=9)

    ax.set_xlim(0, max(counts.max() + 3, 5))
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_pairwise_difference_bar(pairwise_summary: pd.DataFrame, output_path: Path) -> None:
    data = pairwise_summary.set_index("tool_pair")[
        ["both_hit", "only_left_hit", "only_right_hit", "different_strength_among_both_hit", "both_miss"]
    ]
    colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#bab0ac"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(data))
    x_pos = np.arange(len(data))
    for column, color in zip(data.columns, colors):
        values = data[column].to_numpy()
        ax.bar(x_pos, values, bottom=bottom, label=column, color=color)
        bottom += values

    ax.set_xticks(x_pos)
    ax.set_xticklabels(data.index, rotation=30, ha="right")
    ax.set_ylabel("Number of patients")
    ax.set_title("Pairwise overlap/difference counts")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def write_outputs(
    df: pd.DataFrame,
    tools: list[str],
    rank_threshold: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    status_df = build_patient_status(df, tools, rank_threshold)
    rank_tools = infer_rank_tools(df)
    long_status_df = build_long_status(status_df, tools)
    pairwise_summary, pairwise_detail = build_pairwise_differences(status_df, tools)
    hit_combination_counts, exact_combination_counts = build_combination_counts(status_df, tools)

    for obsolete_name in ("tool_venn_triplets.png", "tool_venn_exact_vs_close.png"):
        obsolete_path = output_dir / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()

    status_df.to_csv(output_dir / "patient_tool_status_wide.csv", index=False)
    long_status_df.to_csv(output_dir / "patient_tool_status_long.csv", index=False)
    pairwise_summary.to_csv(output_dir / "pairwise_difference_summary.csv", index=False)
    pairwise_detail.to_csv(output_dir / "pairwise_difference_by_patient.csv", index=False)
    hit_combination_counts.to_csv(output_dir / "tool_hit_combination_counts.csv")
    exact_combination_counts.to_csv(output_dir / "tool_exact_combination_counts.csv")

    save_status_matrix(status_df, tools, output_dir / "patient_tool_status_matrix.png")
    save_case_result_overview(status_df, tools, output_dir / "case_result_overview.png")
    save_correct_rank_overview(df, rank_tools, output_dir / "correct_disease_rank_overview.png")
    save_four_tool_venn(status_df, tools, output_dir / "tool_venn_4tools_exact_and_hit.png")
    save_pairwise_heatmap(pairwise_summary, tools, output_dir / "pairwise_overlap_heatmap.png")
    save_combination_bar(hit_combination_counts, output_dir / "tool_hit_combination_counts.png", len(df))
    save_pairwise_difference_bar(pairwise_summary, output_dir / "pairwise_difference_counts.png")


def parse_args() -> argparse.Namespace:
    default_input = repo_root() / "MONDO_BASE_match_5-2.csv"
    default_output_dir = repo_root() / "validationCode" / "tool_difference_visualization_5-2"

    parser = argparse.ArgumentParser(
        description=(
            "Visualize overlap and differences among individual tools, excluding "
            "TentativeDiagnosis and FinalDiagnosis by default."
        )
    )
    parser.add_argument("--input", default=str(default_input), help="Input MONDO match CSV path.")
    parser.add_argument("--output-dir", default=str(default_output_dir), help="Directory to save PNG and CSV outputs.")
    parser.add_argument("--rank-threshold", type=int, default=5, help="Rank threshold used as top-k hit.")
    parser.add_argument(
        "--tools",
        nargs="+",
        default=None,
        help="Tools to compare. Defaults to all non-Tentative/Final tools found in the CSV.",
    )
    parser.add_argument(
        "--exclude-tools",
        nargs="+",
        default=sorted(DEFAULT_EXCLUDED_TOOLS),
        help="Tools to exclude when tools are inferred.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    df = pd.read_csv(input_path)
    excluded_tools = set(args.exclude_tools)
    tools = args.tools if args.tools else infer_tools(df, excluded_tools)

    missing_columns = [
        column
        for tool in tools
        for column in (f"{tool}_Match_rank", f"{tool}_Close_rank")
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if "patient_id" not in df.columns:
        raise ValueError("Input CSV must contain a patient_id column.")
    if not tools:
        raise ValueError("No tools found to compare.")

    write_outputs(df, tools, args.rank_threshold, output_dir)
    print(f"Input: {input_path}")
    print(f"Tools: {', '.join(tools)}")
    print(f"Rank threshold: {args.rank_threshold}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
