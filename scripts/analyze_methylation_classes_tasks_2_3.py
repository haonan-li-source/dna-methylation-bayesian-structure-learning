from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
OUT_DIR = ROOT / "outputs" / "methylation_class_markov_tasks_2_3"
CHART_DIR = OUT_DIR / "png_charts"
SUMMARY_PATH = ROOT / "outputs" / "all_dataset_analysis" / "dataset_compatibility_summary.csv"

STATES = ["000", "001", "010", "011", "100", "101", "110", "111"]
COUNT_COLS = [f"n_{state}" for state in STATES]
MC_CLASSES = {
    "MC0": ["000"],
    "MC1": ["001", "010", "100"],
    "MC2": ["011", "101", "110"],
    "MC3": ["111"],
}
SCOPE_ALL = "Input_All_Triplets"
SCOPE_VALID = "Valid_Triplets_Only"
SCOPES = [SCOPE_ALL, SCOPE_VALID]


COLORS = {
    "MC0": (37, 99, 235),
    "MC1": (22, 163, 74),
    "MC2": (245, 158, 11),
    "MC3": (220, 38, 38),
    SCOPE_ALL: (37, 99, 235),
    SCOPE_VALID: (22, 163, 74),
    "dark": (17, 24, 39),
    "muted": (107, 114, 128),
    "grid": (229, 231, 235),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font(28, True)
F_HEAD = font(20, True)
F_LABEL = font(18)
F_SMALL = font(14)
F_TINY = font(12)


def round_float(value: Any, digits: int = 8) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)


def clean_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [clean_json_value(inner) for inner in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def valid_mask(df: pd.DataFrame) -> pd.Series:
    total = df[COUNT_COLS].sum(axis=1)
    x1_one = df[["n_100", "n_101", "n_110", "n_111"]].sum(axis=1)
    x2_one = df[["n_010", "n_011", "n_110", "n_111"]].sum(axis=1)
    x3_one = df[["n_001", "n_011", "n_101", "n_111"]].sum(axis=1)
    return (
        (total > 0)
        & (x1_one > 0) & (x1_one < total)
        & (x2_one > 0) & (x2_one < total)
        & (x3_one > 0) & (x3_one < total)
    )


def state_counts_from_df(df: pd.DataFrame) -> dict[str, float]:
    return {state: float(df[f"n_{state}"].sum()) for state in STATES}


def mc_counts_from_state_counts(state_counts: dict[str, float]) -> dict[str, float]:
    return {
        mc: sum(state_counts[state] for state in states)
        for mc, states in MC_CLASSES.items()
    }


def update_transition_counts(transitions: dict[str, dict[tuple[int, int], float]], scope: str, state_counts: dict[str, float]) -> None:
    for state, count in state_counts.items():
        if count == 0:
            continue
        bits = [int(ch) for ch in state]
        transitions[scope][(bits[0], bits[1])] += count
        transitions[scope][(bits[1], bits[2])] += count


def update_position_counts(position_counts: dict[str, dict[tuple[int, int], float]], scope: str, state_counts: dict[str, float]) -> None:
    for state, count in state_counts.items():
        if count == 0:
            continue
        bits = [int(ch) for ch in state]
        for pos, bit in enumerate(bits, start=1):
            position_counts[scope][(pos, bit)] += count


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_metadata() -> dict[str, dict[str, Any]]:
    if not SUMMARY_PATH.exists():
        return {}
    df = pd.read_csv(SUMMARY_PATH)
    return {
        str(row["File_ID"]): {
            "Experiment_ID": row.get("Experiment_ID", ""),
            "Tissue": row.get("Tissue", ""),
            "Input_Triplets_Previous": row.get("Input_Triplets", None),
            "Valid_Triplets_Previous": row.get("Valid_Triplets", None),
            "Skipped_Triplets_Previous": row.get("Skipped_Triplets", None),
        }
        for _, row in df.iterrows()
    }


def aggregate_data() -> dict[str, Any]:
    metadata = load_metadata()
    global_state_counts = {scope: {state: 0.0 for state in STATES} for scope in SCOPES}
    global_triplet_counts = {scope: 0 for scope in SCOPES}
    transitions = {scope: defaultdict(float) for scope in SCOPES}
    position_counts = {scope: defaultdict(float) for scope in SCOPES}
    per_dataset_rows: list[dict[str, Any]] = []
    equality_dataset_rows: list[dict[str, Any]] = []

    tsv_files = sorted(ROOT.glob("ENCFF*.tsv"))
    for path in tsv_files:
        file_id = path.stem
        meta = metadata.get(file_id, {})
        df = pd.read_csv(path, sep="\t", usecols=COUNT_COLS)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        mask = valid_mask(df)

        scope_frames = {
            SCOPE_ALL: df,
            SCOPE_VALID: df.loc[mask],
        }
        for scope, scope_df in scope_frames.items():
            state_counts = state_counts_from_df(scope_df)
            mc_counts = mc_counts_from_state_counts(state_counts)
            total_reads = sum(state_counts.values())
            for state, count in state_counts.items():
                global_state_counts[scope][state] += count
            global_triplet_counts[scope] += int(len(scope_df))
            update_transition_counts(transitions, scope, state_counts)
            update_position_counts(position_counts, scope, state_counts)

            row = {
                "Scope": scope,
                "File_ID": file_id,
                "Experiment_ID": meta.get("Experiment_ID", ""),
                "Tissue": meta.get("Tissue", ""),
                "Triplets_In_Scope": int(len(scope_df)),
                "Skipped_By_Validity_Filter": int(len(df) - int(mask.sum())) if scope == SCOPE_ALL else 0,
                "Total_Reads": int(round(total_reads)),
            }
            for state in STATES:
                row[f"Count_{state}"] = int(round(state_counts[state]))
                row[f"Prop_{state}"] = round_float(state_counts[state] / total_reads if total_reads else None)
            for mc in MC_CLASSES:
                row[f"Count_{mc}"] = int(round(mc_counts[mc]))
                row[f"Prop_{mc}"] = round_float(mc_counts[mc] / total_reads if total_reads else None)
            per_dataset_rows.append(row)

            for mc in ["MC1", "MC2"]:
                states = MC_CLASSES[mc]
                class_total = sum(state_counts[state] for state in states)
                proportions = {state: state_counts[state] / class_total if class_total else None for state in states}
                observed_values = [proportions[state] for state in states if proportions[state] is not None]
                max_dev = max((abs(value - 1 / len(states)) for value in observed_values), default=None)
                equality_dataset_rows.append({
                    "Scope": scope,
                    "File_ID": file_id,
                    "Experiment_ID": meta.get("Experiment_ID", ""),
                    "Tissue": meta.get("Tissue", ""),
                    "MC_Class": mc,
                    "Class_Reads": int(round(class_total)),
                    **{f"Count_{state}": int(round(state_counts[state])) for state in states},
                    **{f"Within_{state}": round_float(proportions[state]) for state in states},
                    "Equal_Reference": round_float(1 / len(states)),
                    "Max_Abs_Deviation_From_Equal_pp": round_float(max_dev * 100 if max_dev is not None else None),
                })

    return {
        "global_state_counts": global_state_counts,
        "global_triplet_counts": global_triplet_counts,
        "transitions": transitions,
        "position_counts": position_counts,
        "per_dataset_rows": per_dataset_rows,
        "equality_dataset_rows": equality_dataset_rows,
        "dataset_count": len(tsv_files),
    }


def build_summary_tables(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    global_state_counts = data["global_state_counts"]
    global_triplet_counts = data["global_triplet_counts"]
    transitions = data["transitions"]
    position_counts = data["position_counts"]

    mc_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    equality_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    transition_matrix_rows: list[dict[str, Any]] = []
    stationary_rows: list[dict[str, Any]] = []

    for scope in SCOPES:
        state_counts = global_state_counts[scope]
        mc_counts = mc_counts_from_state_counts(state_counts)
        total_reads = sum(state_counts.values())
        total_positions = total_reads * 3

        for mc, states in MC_CLASSES.items():
            count = mc_counts[mc]
            mc_rows.append({
                "Scope": scope,
                "MC_Class": mc,
                "States": ", ".join(states),
                "Methylated_CpG_Count": int(mc[-1]),
                "Read_Count": int(round(count)),
                "Read_Proportion": round_float(count / total_reads if total_reads else None),
                "Read_Percent": round_float(count / total_reads * 100 if total_reads else None),
                "Triplets_In_Scope": global_triplet_counts[scope],
            })

        for state in STATES:
            mc = next(name for name, states in MC_CLASSES.items() if state in states)
            within_total = mc_counts[mc]
            count = state_counts[state]
            state_rows.append({
                "Scope": scope,
                "State": state,
                "MC_Class": mc,
                "Methylated_CpG_Count": state.count("1"),
                "Read_Count": int(round(count)),
                "Overall_Read_Proportion": round_float(count / total_reads if total_reads else None),
                "Overall_Read_Percent": round_float(count / total_reads * 100 if total_reads else None),
                "Within_MC_Proportion": round_float(count / within_total if within_total else None),
                "Within_MC_Percent": round_float(count / within_total * 100 if within_total else None),
            })

        for mc in ["MC1", "MC2"]:
            states = MC_CLASSES[mc]
            class_total = sum(state_counts[state] for state in states)
            expected = class_total / len(states) if states else None
            chi_square = None
            p_value_df2 = None
            proportions = {}
            if expected and expected > 0:
                chi_square = sum((state_counts[state] - expected) ** 2 / expected for state in states)
                # For chi-square with 2 degrees of freedom, the survival function is exp(-x/2).
                p_value_df2 = math.exp(-chi_square / 2) if chi_square < 1400 else 0.0
                proportions = {state: state_counts[state] / class_total for state in states}
            max_dev = max((abs(value - 1 / len(states)) for value in proportions.values()), default=None)
            ratio = None
            positive_counts = [state_counts[state] for state in states if state_counts[state] > 0]
            if positive_counts:
                ratio = max(positive_counts) / min(positive_counts)
            equality_rows.append({
                "Scope": scope,
                "MC_Class": mc,
                "States_Compared": ", ".join(states),
                "Class_Reads": int(round(class_total)),
                **{f"Count_{state}": int(round(state_counts[state])) for state in states},
                **{f"Within_{state}": round_float(proportions.get(state)) for state in states},
                "Equal_Reference_Proportion": round_float(1 / len(states)),
                "Max_Abs_Deviation_From_Equal_pp": round_float(max_dev * 100 if max_dev is not None else None),
                "Max_to_Min_Count_Ratio": round_float(ratio),
                "Chi_Square_vs_Equal": round_float(chi_square),
                "Approx_P_Value_df2": round_float(p_value_df2, 12),
                "Interpretation": "Empirically not equally likely" if max_dev is not None and max_dev > 0.01 else "Close to equal at aggregate level",
            })

        trans_counts = transitions[scope]
        row_totals = {
            frm: trans_counts[(frm, 0)] + trans_counts[(frm, 1)]
            for frm in [0, 1]
        }
        probs = {}
        for frm in [0, 1]:
            for to in [0, 1]:
                probs[(frm, to)] = trans_counts[(frm, to)] / row_totals[frm] if row_totals[frm] else None
                transition_rows.append({
                    "Scope": scope,
                    "From_State": frm,
                    "To_State": to,
                    "Transition_Count": int(round(trans_counts[(frm, to)])),
                    "From_State_Total": int(round(row_totals[frm])),
                    "Transition_Probability": round_float(probs[(frm, to)]),
                })
            transition_matrix_rows.append({
                "Scope": scope,
                "From_State": frm,
                "P_to_0": round_float(probs[(frm, 0)]),
                "P_to_1": round_float(probs[(frm, 1)]),
                "Count_to_0": int(round(trans_counts[(frm, 0)])),
                "Count_to_1": int(round(trans_counts[(frm, 1)])),
            })

        p01 = probs[(0, 1)]
        p10 = probs[(1, 0)]
        p00 = probs[(0, 0)]
        p11 = probs[(1, 1)]
        denom = (p01 or 0) + (p10 or 0)
        pi0 = p10 / denom if denom else None
        pi1 = p01 / denom if denom else None
        bit0 = sum(position_counts[scope][(pos, 0)] for pos in [1, 2, 3])
        bit1 = sum(position_counts[scope][(pos, 1)] for pos in [1, 2, 3])
        stationary_rows.append({
            "Scope": scope,
            "P00": round_float(p00),
            "P01": round_float(p01),
            "P10": round_float(p10),
            "P11": round_float(p11),
            "Stationary_pi_0": round_float(pi0),
            "Stationary_pi_1": round_float(pi1),
            "Empirical_Position_Prop_0": round_float(bit0 / total_positions if total_positions else None),
            "Empirical_Position_Prop_1": round_float(bit1 / total_positions if total_positions else None),
            "Irreducible_Unique_Stationary": bool(p01 and p01 > 0 and p10 and p10 > 0),
            "Aperiodic_By_Self_Loops": bool(p00 and p00 > 0 and p11 and p11 > 0),
            "Interpretation": "The empirical 2-state transition matrix has a unique stationary distribution, but this is a local descriptive estimate rather than proof of a genome-wide stationary Markov chain.",
        })

    return {
        "mc_class_summary": mc_rows,
        "state_distribution_summary": state_rows,
        "within_mc_equal_likelihood_check": equality_rows,
        "markov_transition_counts": transition_rows,
        "markov_transition_matrix": transition_matrix_rows,
        "stationary_distribution_summary": stationary_rows,
        "per_dataset_mc_summary": data["per_dataset_rows"],
        "per_dataset_equal_likelihood_check": data["equality_dataset_rows"],
    }


def canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), "white")
    return img, ImageDraw.Draw(img)


def draw_axes(draw: ImageDraw.ImageDraw, left: int, top: int, right: int, bottom: int) -> None:
    draw.line((left, bottom, right, bottom), fill=COLORS["dark"], width=2)
    draw.line((left, top, left, bottom), fill=COLORS["dark"], width=2)


def draw_mc_distribution(mc_rows: list[dict[str, Any]]) -> None:
    img, draw = canvas(1180, 620)
    draw.text((35, 25), "MC0-MC3 methylation class distribution", font=F_TITLE, fill=COLORS["dark"])
    draw.text((35, 62), "MC class is defined by the number of methylated CpGs in a 3-CpG triplet.", font=F_SMALL, fill=COLORS["muted"])

    left, top, right, bottom = 105, 145, 1110, 500
    y_max = max(row["Read_Percent"] for row in mc_rows) * 1.15

    def y(value: float) -> int:
        return int(top + (y_max - value) / y_max * (bottom - top))

    for tick in range(0, int(math.ceil(y_max / 10) * 10) + 1, 10):
        yy = y(tick)
        draw.line((left, yy, right, yy), fill=COLORS["grid"], width=1)
        draw.text((55, yy - 8), f"{tick}%", font=F_SMALL, fill=COLORS["muted"])
    draw_axes(draw, left, top, right, bottom)

    group_width = 210
    bar_width = 62
    scope_offsets = {SCOPE_ALL: -38, SCOPE_VALID: 38}
    rows_by = {(row["Scope"], row["MC_Class"]): row for row in mc_rows}
    for i, mc in enumerate(MC_CLASSES):
        x_center = left + 120 + i * group_width
        for scope in SCOPES:
            row = rows_by[(scope, mc)]
            value = float(row["Read_Percent"])
            x0 = x_center + scope_offsets[scope] - bar_width // 2
            x1 = x0 + bar_width
            draw.rectangle((x0, y(value), x1, bottom), fill=COLORS[scope])
            draw.text((x0 - 8, y(value) - 24), f"{value:.1f}%", font=F_TINY, fill=COLORS[scope])
        draw.text((x_center - 20, bottom + 18), mc, font=F_LABEL, fill=COLORS["dark"])

    legend_x, legend_y = 810, 92
    for idx, scope in enumerate(SCOPES):
        yy = legend_y + idx * 34
        draw.rectangle((legend_x, yy, legend_x + 22, yy + 18), fill=COLORS[scope])
        label = "All input triplets" if scope == SCOPE_ALL else "Valid triplets only"
        draw.text((legend_x + 34, yy - 1), label, font=F_SMALL, fill=COLORS["dark"])
    draw.text((35, 560), "Read-count weighted: counts n_000...n_111 are treated as observed read counts.", font=F_SMALL, fill=COLORS["muted"])
    img.save(CHART_DIR / "mc_class_distribution.png")


def draw_within_mc_equality(state_rows: list[dict[str, Any]]) -> None:
    img, draw = canvas(1260, 660)
    draw.text((35, 25), "Within-class state proportions: are states equally likely?", font=F_TITLE, fill=COLORS["dark"])
    draw.text((35, 62), "Reference line is 1/3 for MC1 and MC2. Large deviations mean position information matters.", font=F_SMALL, fill=COLORS["muted"])

    left, top, right, bottom = 105, 120, 1190, 540
    y_max = 70

    def y(value_pct: float) -> int:
        return int(top + (y_max - value_pct) / y_max * (bottom - top))

    for tick in range(0, y_max + 1, 10):
        yy = y(tick)
        draw.line((left, yy, right, yy), fill=COLORS["grid"], width=1)
        draw.text((55, yy - 8), f"{tick}%", font=F_SMALL, fill=COLORS["muted"])
    draw_axes(draw, left, top, right, bottom)
    draw.line((left, y(100 / 3), right, y(100 / 3)), fill=(148, 163, 184), width=2)
    draw.text((right - 100, y(100 / 3) - 24), "1/3", font=F_SMALL, fill=COLORS["muted"])

    rows_by = {(row["Scope"], row["State"]): row for row in state_rows if row["MC_Class"] in ["MC1", "MC2"]}
    states = ["001", "010", "100", "011", "101", "110"]
    x_gap = 170
    bar_width = 48
    offsets = {SCOPE_ALL: -30, SCOPE_VALID: 30}
    for i, state in enumerate(states):
        x_center = left + 80 + i * x_gap
        for scope in SCOPES:
            pct = float(rows_by[(scope, state)]["Within_MC_Percent"])
            x0 = x_center + offsets[scope] - bar_width // 2
            draw.rectangle((x0, y(pct), x0 + bar_width, bottom), fill=COLORS[scope])
            draw.text((x0 - 4, y(pct) - 20), f"{pct:.1f}", font=F_TINY, fill=COLORS[scope])
        draw.text((x_center - 18, bottom + 18), state, font=F_SMALL, fill=COLORS["dark"])
        if state == "100":
            draw.line((x_center + 65, top, x_center + 65, bottom + 35), fill=COLORS["grid"], width=2)

    draw.text((left + 155, bottom + 52), "MC1", font=F_HEAD, fill=COLORS["dark"])
    draw.text((left + 665, bottom + 52), "MC2", font=F_HEAD, fill=COLORS["dark"])

    legend_x, legend_y = 925, 125
    for idx, scope in enumerate(SCOPES):
        yy = legend_y + idx * 34
        draw.rectangle((legend_x, yy, legend_x + 22, yy + 18), fill=COLORS[scope])
        label = "All input triplets" if scope == SCOPE_ALL else "Valid triplets only"
        draw.text((legend_x + 34, yy - 1), label, font=F_SMALL, fill=COLORS["dark"])
    img.save(CHART_DIR / "within_mc_equal_likelihood.png")


def draw_markov_matrix(stationary_rows: list[dict[str, Any]], matrix_rows: list[dict[str, Any]]) -> None:
    img, draw = canvas(1160, 620)
    draw.text((35, 25), "Empirical local Markov transition matrix", font=F_TITLE, fill=COLORS["dark"])
    draw.text((35, 62), "Estimated from adjacent positions X1->X2 and X2->X3 in triplet read counts.", font=F_SMALL, fill=COLORS["muted"])

    matrix_by = {(row["Scope"], int(row["From_State"])): row for row in matrix_rows}
    stat_by = {row["Scope"]: row for row in stationary_rows}

    for panel_idx, scope in enumerate(SCOPES):
        x0 = 95 + panel_idx * 530
        y0 = 135
        title = "All input triplets" if scope == SCOPE_ALL else "Valid triplets only"
        draw.text((x0, y0 - 50), title, font=F_HEAD, fill=COLORS["dark"])
        cell = 125
        draw.text((x0 + 130, y0 - 24), "To 0", font=F_LABEL, fill=COLORS["dark"])
        draw.text((x0 + 265, y0 - 24), "To 1", font=F_LABEL, fill=COLORS["dark"])
        draw.text((x0 - 70, y0 + 45), "From 0", font=F_LABEL, fill=COLORS["dark"])
        draw.text((x0 - 70, y0 + 175), "From 1", font=F_LABEL, fill=COLORS["dark"])
        values = [
            [float(matrix_by[(scope, 0)]["P_to_0"]), float(matrix_by[(scope, 0)]["P_to_1"])],
            [float(matrix_by[(scope, 1)]["P_to_0"]), float(matrix_by[(scope, 1)]["P_to_1"])],
        ]
        max_val = max(max(row) for row in values)
        min_val = min(min(row) for row in values)
        for r in range(2):
            for c in range(2):
                value = values[r][c]
                intensity = 0.15 + 0.75 * ((value - min_val) / (max_val - min_val) if max_val > min_val else 0.5)
                base = COLORS[scope]
                color = tuple(int(255 * (1 - intensity) + channel * intensity) for channel in base)
                xx = x0 + 110 + c * cell
                yy = y0 + r * cell
                draw.rectangle((xx, yy, xx + cell, yy + cell), fill=color, outline="white", width=3)
                draw.text((xx + 33, yy + 42), f"{value:.3f}", font=F_HEAD, fill="white")
        stat = stat_by[scope]
        draw.text((x0, y0 + 295), f"Stationary pi: ({stat['Stationary_pi_0']:.3f}, {stat['Stationary_pi_1']:.3f})", font=F_LABEL, fill=COLORS["dark"])
        draw.text((x0, y0 + 325), f"Empirical bit prop: ({stat['Empirical_Position_Prop_0']:.3f}, {stat['Empirical_Position_Prop_1']:.3f})", font=F_SMALL, fill=COLORS["muted"])
        draw.text((x0, y0 + 350), "Unique if both 0->1 and 1->0 are positive.", font=F_TINY, fill=COLORS["muted"])

    draw.text((35, 565), "This is a local descriptive Markov estimate, not proof that the whole genomic CpG chain is stationary first-order Markov.", font=F_SMALL, fill=COLORS["muted"])
    img.save(CHART_DIR / "empirical_markov_transition_matrix.png")


def write_latex(tables: dict[str, list[dict[str, Any]]]) -> None:
    tex = r"""\subsection{Methylation Classes and Local Markov Analysis}
\label{subsec:methylation-classes-markov}

\paragraph{Methylation classes.}
For a triplet of three CpG sites, let
\((X_1,X_2,X_3) \in \{0,1\}^3\), where 1 denotes a methylated CpG and
0 denotes an unmethylated CpG. We define the methylation class of a triplet
state by the number of methylated CpGs:
\[
  C = X_1 + X_2 + X_3.
\]
This gives four methylation classes:
\[
\begin{aligned}
  \mathrm{MC0} &= \{000\},\\
  \mathrm{MC1} &= \{001,010,100\},\\
  \mathrm{MC2} &= \{011,101,110\},\\
  \mathrm{MC3} &= \{111\}.
\end{aligned}
\]
For a triplet \(\tau\), the read-count estimate of each class is obtained by
summing the corresponding state counts:
\[
\begin{aligned}
  m_0(\tau) &= n_{000},\\
  m_1(\tau) &= n_{001}+n_{010}+n_{100},\\
  m_2(\tau) &= n_{011}+n_{101}+n_{110},\\
  m_3(\tau) &= n_{111}.
\end{aligned}
\]
The empirical class probabilities are then
\[
  \widehat{P}_{\tau}(\mathrm{MC}k)
  =
  \frac{m_k(\tau)}
       {n_{000}+n_{001}+n_{010}+n_{011}+n_{100}+n_{101}+n_{110}+n_{111}}.
\]

\paragraph{Equal-likelihood question within a methylation class.}
Although the states in \(\mathrm{MC1}\) and \(\mathrm{MC2}\) have the same
number of methylated CpGs, they are not assumed to be equally likely. For
example, the equality assumption within \(\mathrm{MC2}\) would imply
\[
  P(011 \mid \mathrm{MC2})
  =
  P(101 \mid \mathrm{MC2})
  =
  P(110 \mid \mathrm{MC2})
  =
  \frac{1}{3}.
\]
This assumption would be valid only under an exchangeability condition in
which the three positions in the triplet are symmetric. In the analysis, we
therefore report the within-class empirical proportions separately, rather
than replacing them by \(1/3\). This preserves positional information such as
the difference between \(110\), \(101\), and \(011\).

\paragraph{Empirical local Markov transition matrix.}
To summarize local dependence along the CpG chain, we estimate a two-state
transition matrix from adjacent positions inside each triplet. For
\(a,b\in\{0,1\}\), define
\[
  C_{ab}
  =
  \sum_{\tau}\sum_{c\in\{0,1\}} n_{abc}^{(\tau)}
  +
  \sum_{\tau}\sum_{c\in\{0,1\}} n_{cab}^{(\tau)}.
\]
The first term counts transitions \(X_1=a\) to \(X_2=b\), and the second term
counts transitions \(X_2=a\) to \(X_3=b\). The empirical transition matrix is
\[
  \widehat{P}_{ab}
  =
  \frac{C_{ab}}{C_{a0}+C_{a1}}.
\]
If this finite two-state chain has positive probabilities for both
\(0\to 1\) and \(1\to 0\), then it is irreducible and has a unique stationary
distribution. If it also has positive self-transition probabilities, it is
aperiodic. However, the estimated matrix should be interpreted as a local
empirical summary of adjacent CpG dependence. It does not by itself prove that
the full genomic CpG sequence follows a stationary first-order Markov chain.
"""
    (OUT_DIR / "tasks_2_3_methylation_class_markov_en.tex").write_text(tex, encoding="utf-8")


def write_chinese_notes(tables: dict[str, list[dict[str, Any]]]) -> None:
    mc_rows = [row for row in tables["mc_class_summary"] if row["Scope"] == SCOPE_ALL]
    eq_rows = [row for row in tables["within_mc_equal_likelihood_check"] if row["Scope"] == SCOPE_ALL]
    stat = next(row for row in tables["stationary_distribution_summary"] if row["Scope"] == SCOPE_ALL)
    mc_lines = "\n".join(
        f"- {row['MC_Class']}: {row['Read_Percent']:.2f}%"
        for row in mc_rows
    )
    eq_lines = "\n".join(
        f"- {row['MC_Class']}: max deviation from equal = {row['Max_Abs_Deviation_From_Equal_pp']:.2f} percentage points; {row['Interpretation']}."
        for row in eq_rows
    )
    text = f"""# 任务 2/3 中文说明

## 任务 2：Methylation class

对三个 CpG 的状态 \\((X_1,X_2,X_3)\\)，按照 methylated CpG 的数量分成四类：

- MC0 = {{000}}
- MC1 = {{001, 010, 100}}
- MC2 = {{011, 101, 110}}
- MC3 = {{111}}

全体 input triplets 的 read-count weighted 分布为：

{mc_lines}

## 任务 3：同一 MC 内是否等概率，以及 Markov chain 问题

MC1 和 MC2 里面虽然状态的 methylated CpG 数量一样，但位置不同，所以不能默认等概率。例如 110、101、011 都属于 MC2，但是它们代表 methylated CpG 出现在不同位置。

本次数据的聚合结果：

{eq_lines}

局部 Markov transition matrix 是从 triplet 内的 X1->X2 和 X2->X3 两个相邻位置估计的。对于 all input triplets，经验 stationary distribution 为：

- pi_0 = {stat['Stationary_pi_0']:.4f}
- pi_1 = {stat['Stationary_pi_1']:.4f}

需要注意：这个 transition matrix 可以作为局部相邻 CpG 依赖的经验总结，但不能单独证明整条 genome CpG chain 一定是 stationary first-order Markov chain。
"""
    (OUT_DIR / "tasks_2_3_chinese_notes.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    data = aggregate_data()
    tables = build_summary_tables(data)

    for name, rows in tables.items():
        write_csv(OUT_DIR / f"{name}.csv", rows)

    draw_mc_distribution(tables["mc_class_summary"])
    draw_within_mc_equality(tables["state_distribution_summary"])
    draw_markov_matrix(tables["stationary_distribution_summary"], tables["markov_transition_matrix"])
    write_latex(tables)
    write_chinese_notes(tables)

    workbook_data = {
        "dataset_count": data["dataset_count"],
        **tables,
        "charts": [
            {
                "Chart": "mc_class_distribution.png",
                "Meaning_CN": "MC0-MC3 的 read-count weighted 分布，包含 all input 和 valid-only 两个口径。",
                "Path": str(CHART_DIR / "mc_class_distribution.png"),
            },
            {
                "Chart": "within_mc_equal_likelihood.png",
                "Meaning_CN": "MC1/MC2 内部状态比例与 1/3 等概率参考线的比较。",
                "Path": str(CHART_DIR / "within_mc_equal_likelihood.png"),
            },
            {
                "Chart": "empirical_markov_transition_matrix.png",
                "Meaning_CN": "由 triplet 内相邻位置估计的 2x2 局部 Markov transition matrix。",
                "Path": str(CHART_DIR / "empirical_markov_transition_matrix.png"),
            },
        ],
        "latex_file": str(OUT_DIR / "tasks_2_3_methylation_class_markov_en.tex"),
        "chinese_notes": str(OUT_DIR / "tasks_2_3_chinese_notes.md"),
    }
    (OUT_DIR / "workbook_data.json").write_text(
        json.dumps(clean_json_value(workbook_data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "output_dir": str(OUT_DIR),
        "xlsx_planned": str(OUT_DIR / "tasks_2_3_methylation_class_markov_summary.xlsx"),
        "latex": str(OUT_DIR / "tasks_2_3_methylation_class_markov_en.tex"),
        "chinese_notes": str(OUT_DIR / "tasks_2_3_chinese_notes.md"),
        "charts": workbook_data["charts"],
        "csv_files": [str(OUT_DIR / f"{name}.csv") for name in tables],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
