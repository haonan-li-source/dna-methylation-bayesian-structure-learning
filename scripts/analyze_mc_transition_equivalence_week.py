from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
OUT_DIR = ROOT / "outputs" / "mc_transition_equivalence_week_2026_06_15"
CHART_DIR = OUT_DIR / "png_charts"

STATE_ORDER = ["000", "001", "010", "100", "011", "101", "110", "111"]
DATA_STATE_ORDER = ["000", "001", "010", "011", "100", "101", "110", "111"]
COUNT_COLS = [f"n_{state}" for state in DATA_STATE_ORDER]
META_COLS = ["Chromosome number", "Triplet index (first CpG)"]
MC_ORDER = ["MC0", "MC1", "MC2", "MC3"]
SCOPES = ["Input_All_Triplets", "Valid_Triplets_Only"]

STATE_TO_MC = {
    "000": "MC0",
    "001": "MC1",
    "010": "MC1",
    "100": "MC1",
    "011": "MC2",
    "101": "MC2",
    "110": "MC2",
    "111": "MC3",
}
MC_TO_STATES = {
    "MC0": ["000"],
    "MC1": ["001", "010", "100"],
    "MC2": ["011", "101", "110"],
    "MC3": ["111"],
}


COLORS = {
    "dark": (17, 24, 39),
    "muted": (107, 114, 128),
    "grid": (229, 231, 235),
    "blue": (37, 99, 235),
    "green": (22, 163, 74),
    "orange": (245, 158, 11),
    "red": (220, 38, 38),
    "purple": (126, 34, 206),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font(28, True)
F_HEAD = font(20, True)
F_LABEL = font(16)
F_SMALL = font(13)
F_TINY = font(11)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [clean_json(inner) for inner in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def round_float(value: Any, digits: int = 8) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)


def valid_mask_from_counts(counts: np.ndarray) -> np.ndarray:
    # counts columns are DATA_STATE_ORDER: 000,001,010,011,100,101,110,111
    total = counts.sum(axis=1)
    x1 = counts[:, [4, 5, 6, 7]].sum(axis=1)
    x2 = counts[:, [2, 3, 6, 7]].sum(axis=1)
    x3 = counts[:, [1, 3, 5, 7]].sum(axis=1)
    return (
        (total > 0)
        & (x1 > 0) & (x1 < total)
        & (x2 > 0) & (x2 < total)
        & (x3 > 0) & (x3 < total)
    )


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    result = np.zeros_like(matrix, dtype=float)
    row_sums = matrix.sum(axis=1)
    for i, total in enumerate(row_sums):
        if total > 0:
            result[i, :] = matrix[i, :] / total
    return result


def stationary_distribution(row_stochastic: np.ndarray) -> np.ndarray:
    p = np.asarray(row_stochastic, dtype=float)
    n = p.shape[0]
    # Solve (P^T - I) pi = 0 plus sum(pi)=1.
    a = p.T - np.eye(n)
    a[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eig(p.T)
        idx = int(np.argmin(np.abs(values - 1)))
        pi = np.real(vectors[:, idx])
    pi = np.where(pi < 0, 0, pi)
    if pi.sum() == 0:
        pi = np.ones(n) / n
    else:
        pi = pi / pi.sum()
    return pi


def next_states(state: str) -> tuple[str, str]:
    return state[1:] + "0", state[1:] + "1"


STATE_INDEX = {state: i for i, state in enumerate(STATE_ORDER)}
MC_INDEX = {mc: i for i, mc in enumerate(MC_ORDER)}
DATA_INDEX = {state: i for i, state in enumerate(DATA_STATE_ORDER)}


def reorder_counts(row_counts: np.ndarray) -> dict[str, float]:
    return {state: float(row_counts[DATA_INDEX[state]]) for state in STATE_ORDER}


def update_transition_counts(
    trans8: np.ndarray,
    diagnostics: dict[str, float],
    left_counts_data_order: np.ndarray,
    right_counts_data_order: np.ndarray,
) -> None:
    left_counts = reorder_counts(left_counts_data_order)
    right_counts = reorder_counts(right_counts_data_order)
    for state in STATE_ORDER:
        left_count = left_counts[state]
        if left_count <= 0:
            continue
        state0, state1 = next_states(state)
        denom = right_counts[state0] + right_counts[state1]
        if denom <= 0:
            diagnostics["Lost_Left_Read_Mass_No_Right_Prefix"] += left_count
            continue
        trans8[STATE_INDEX[state], STATE_INDEX[state0]] += left_count * right_counts[state0] / denom
        trans8[STATE_INDEX[state], STATE_INDEX[state1]] += left_count * right_counts[state1] / denom
        diagnostics["Allocated_Left_Read_Mass"] += left_count


def aggregate_micro_counts_to_macro(trans8_counts: np.ndarray) -> np.ndarray:
    macro = np.zeros((4, 4), dtype=float)
    for state in STATE_ORDER:
        i = MC_INDEX[STATE_TO_MC[state]]
        s_idx = STATE_INDEX[state]
        for next_state in STATE_ORDER:
            j = MC_INDEX[STATE_TO_MC[next_state]]
            macro[i, j] += trans8_counts[s_idx, STATE_INDEX[next_state]]
    return macro


def aggregate_stationary_to_mc(pi8: np.ndarray) -> np.ndarray:
    pi4 = np.zeros(4, dtype=float)
    for state, value in zip(STATE_ORDER, pi8):
        pi4[MC_INDEX[STATE_TO_MC[state]]] += value
    return pi4


def macro_from_micro_using_stationary(t8: np.ndarray, pi8: np.ndarray) -> np.ndarray:
    pi4 = aggregate_stationary_to_mc(pi8)
    p4 = np.zeros((4, 4), dtype=float)
    for s in STATE_ORDER:
        s_idx = STATE_INDEX[s]
        i = MC_INDEX[STATE_TO_MC[s]]
        if pi4[i] == 0:
            continue
        for t in STATE_ORDER:
            j = MC_INDEX[STATE_TO_MC[t]]
            p4[i, j] += pi8[s_idx] * t8[s_idx, STATE_INDEX[t]] / pi4[i]
    return p4


def analyze_data() -> dict[str, Any]:
    trans8_counts = {scope: np.zeros((8, 8), dtype=float) for scope in SCOPES}
    diagnostics = {
        scope: defaultdict(float, {
            "Comparable_Window_Pairs": 0.0,
            "Allocated_Left_Read_Mass": 0.0,
            "Lost_Left_Read_Mass_No_Right_Prefix": 0.0,
        })
        for scope in SCOPES
    }

    for path in sorted(ROOT.glob("ENCFF*.tsv")):
        df = pd.read_csv(path, sep="\t", usecols=META_COLS + COUNT_COLS)
        df = df.sort_values(["Chromosome number", "Triplet index (first CpG)"]).reset_index(drop=True)
        counts = df[COUNT_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        chrom = pd.to_numeric(df["Chromosome number"], errors="coerce").to_numpy()
        index = pd.to_numeric(df["Triplet index (first CpG)"], errors="coerce").to_numpy()
        valid = valid_mask_from_counts(counts)

        for row in range(len(df) - 1):
            if not (chrom[row] == chrom[row + 1] and index[row + 1] == index[row] + 1):
                continue
            for scope in SCOPES:
                if scope == "Valid_Triplets_Only" and not (valid[row] and valid[row + 1]):
                    continue
                diagnostics[scope]["Comparable_Window_Pairs"] += 1
                update_transition_counts(
                    trans8_counts[scope],
                    diagnostics[scope],
                    counts[row],
                    counts[row + 1],
                )

    return {"trans8_counts": trans8_counts, "diagnostics": diagnostics}


def matrix_to_rows(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    scope: str,
    value_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for i, row_label in enumerate(row_labels):
        row: dict[str, Any] = {"Scope": scope, "From": row_label, "Value_Type": value_name}
        for j, col_label in enumerate(col_labels):
            row[f"To_{col_label}"] = round_float(matrix[i, j])
        rows.append(row)
    return rows


def long_matrix_rows(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    scope: str,
    value_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for i, row_label in enumerate(row_labels):
        for j, col_label in enumerate(col_labels):
            rows.append({
                "Scope": scope,
                "From": row_label,
                "To": col_label,
                "Value_Type": value_name,
                "Value": round_float(matrix[i, j]),
            })
    return rows


def build_tables(analysis: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    structural_rows = []
    for state in STATE_ORDER:
        state0, state1 = next_states(state)
        structural_rows.append({
            "State": state,
            "MC_Class": STATE_TO_MC[state],
            "Allowed_Next_State_0": state0,
            "Allowed_Next_State_0_MC": STATE_TO_MC[state0],
            "Allowed_Next_State_1": state1,
            "Allowed_Next_State_1_MC": STATE_TO_MC[state1],
            "Explanation": "A sliding triplet window abc can only move to bc0 or bc1.",
        })

    macro_structural_rows = []
    for mc in MC_ORDER:
        allowed = set()
        for state in MC_TO_STATES[mc]:
            for nxt in next_states(state):
                allowed.add(STATE_TO_MC[nxt])
        macro_structural_rows.append({
            "From_MC": mc,
            **{f"To_{target}": 1 if target in allowed else 0 for target in MC_ORDER},
            "Allowed_Targets": ", ".join(target for target in MC_ORDER if target in allowed),
        })

    state8_matrix_rows: list[dict[str, Any]] = []
    state8_long_rows: list[dict[str, Any]] = []
    mc4_matrix_rows: list[dict[str, Any]] = []
    mc4_equiv_rows: list[dict[str, Any]] = []
    stationary_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    micro_to_macro_rows: list[dict[str, Any]] = []
    lumpability_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []

    for scope in SCOPES:
        counts8 = analysis["trans8_counts"][scope]
        t8 = normalize_rows(counts8)
        counts4 = aggregate_micro_counts_to_macro(counts8)
        p4_raw = normalize_rows(counts4)
        pi8 = stationary_distribution(t8)
        p4_equiv = macro_from_micro_using_stationary(t8, pi8)
        pi4_equiv = aggregate_stationary_to_mc(pi8)
        pi4_raw = stationary_distribution(p4_raw)

        state8_matrix_rows.extend(matrix_to_rows(t8, STATE_ORDER, STATE_ORDER, scope, "Probability"))
        state8_long_rows.extend(long_matrix_rows(t8, STATE_ORDER, STATE_ORDER, scope, "Probability"))
        mc4_matrix_rows.extend(matrix_to_rows(p4_raw, MC_ORDER, MC_ORDER, scope, "Raw_Count_Normalized_Probability"))
        mc4_equiv_rows.extend(matrix_to_rows(p4_equiv, MC_ORDER, MC_ORDER, scope, "Stationary_Aggregated_From_8_State"))

        for state, value in zip(STATE_ORDER, pi8):
            stationary_rows.append({
                "Scope": scope,
                "Level": "8_state",
                "State": state,
                "MC_Class": STATE_TO_MC[state],
                "Stationary_Probability": round_float(value),
            })
        for mc, value in zip(MC_ORDER, pi4_equiv):
            stationary_rows.append({
                "Scope": scope,
                "Level": "MC4_from_8_state",
                "State": mc,
                "MC_Class": mc,
                "Stationary_Probability": round_float(value),
            })
        for mc, value in zip(MC_ORDER, pi4_raw):
            stationary_rows.append({
                "Scope": scope,
                "Level": "MC4_raw_count_matrix",
                "State": mc,
                "MC_Class": mc,
                "Stationary_Probability": round_float(value),
            })

        diag = analysis["diagnostics"][scope]
        diagnostics_rows.append({
            "Scope": scope,
            "Comparable_Window_Pairs": int(diag["Comparable_Window_Pairs"]),
            "Allocated_Left_Read_Mass": round_float(diag["Allocated_Left_Read_Mass"], 2),
            "Lost_Left_Read_Mass_No_Right_Prefix": round_float(diag["Lost_Left_Read_Mass_No_Right_Prefix"], 2),
            "Lost_Mass_Percent": round_float(
                diag["Lost_Left_Read_Mass_No_Right_Prefix"]
                / (diag["Allocated_Left_Read_Mass"] + diag["Lost_Left_Read_Mass_No_Right_Prefix"])
                * 100
                if (diag["Allocated_Left_Read_Mass"] + diag["Lost_Left_Read_Mass_No_Right_Prefix"]) > 0
                else 0
            ),
        })

        for state in STATE_ORDER:
            s_idx = STATE_INDEX[state]
            row = {
                "Scope": scope,
                "State": state,
                "MC_Class": STATE_TO_MC[state],
            }
            for target_mc in MC_ORDER:
                target_indices = [STATE_INDEX[target] for target in MC_TO_STATES[target_mc]]
                row[f"P_to_{target_mc}"] = round_float(float(t8[s_idx, target_indices].sum()))
            micro_to_macro_rows.append(row)

        for mc in MC_ORDER:
            states = MC_TO_STATES[mc]
            for target_mc in MC_ORDER:
                values = []
                for state in states:
                    s_idx = STATE_INDEX[state]
                    target_indices = [STATE_INDEX[target] for target in MC_TO_STATES[target_mc]]
                    values.append(float(t8[s_idx, target_indices].sum()))
                lumpability_rows.append({
                    "Scope": scope,
                    "MC_Class": mc,
                    "Target_MC": target_mc,
                    "Min_Micro_Row_Probability": round_float(min(values)),
                    "Max_Micro_Row_Probability": round_float(max(values)),
                    "Range": round_float(max(values) - min(values)),
                    "Strong_Lumpability_Would_Require_Range_0": "YES" if max(values) - min(values) < 1e-10 else "NO",
                })

        max_abs = float(np.max(np.abs(p4_raw - p4_equiv)))
        stationarity_residual = float(np.max(np.abs(pi4_equiv @ p4_equiv - pi4_equiv)))
        agreement_rows.append({
            "Scope": scope,
            "Max_Abs_Difference_Raw_MC4_vs_Stationary_Aggregated_MC4": round_float(max_abs),
            "Stationarity_Residual_For_Aggregated_MC4": round_float(stationarity_residual, 12),
            "Interpretation": "The 8-state matrix agrees with the stationary-aggregated MC4 matrix by construction; raw count-normalized MC4 can differ because MC classes are not strongly lumpable.",
        })

    mc2_rows = [
        {
            "State": "110",
            "MC_Class": "MC2",
            "Methylated_Positions": "X_n and X_{n+1}",
            "Spatial_Pattern": "left adjacent pair methylated",
            "Allowed_Next_States": "100, 101",
            "Allowed_Next_MC": "MC1 or MC2",
            "BN_Implication": "MC2 alone does not say whether dependence is mainly in the left adjacent pair.",
        },
        {
            "State": "101",
            "MC_Class": "MC2",
            "Methylated_Positions": "X_n and X_{n+2}",
            "Spatial_Pattern": "endpoints methylated, middle unmethylated",
            "Allowed_Next_States": "010, 011",
            "Allowed_Next_MC": "MC1 or MC2",
            "BN_Implication": "This pattern separates the two methylated CpGs and may suggest a different local relation than adjacent methylation.",
        },
        {
            "State": "011",
            "MC_Class": "MC2",
            "Methylated_Positions": "X_{n+1} and X_{n+2}",
            "Spatial_Pattern": "right adjacent pair methylated",
            "Allowed_Next_States": "110, 111",
            "Allowed_Next_MC": "MC2 or MC3",
            "BN_Implication": "Only this MC2 microstate can move directly to MC3 under the sliding-window overlap rule.",
        },
    ]

    return {
        "structural_8_state_mapping": structural_rows,
        "structural_mc4_allowed_matrix": macro_structural_rows,
        "state8_transition_matrix": state8_matrix_rows,
        "state8_transition_long": state8_long_rows,
        "mc4_transition_matrix_raw": mc4_matrix_rows,
        "mc4_transition_matrix_equiv_from_8": mc4_equiv_rows,
        "stationary_distributions": stationary_rows,
        "transition_diagnostics": diagnostics_rows,
        "micro_to_macro_rows": micro_to_macro_rows,
        "lumpability_diagnostic": lumpability_rows,
        "agreement_check": agreement_rows,
        "mc2_bn_implication": mc2_rows,
    }


def matrix_from_rows(rows: list[dict[str, Any]], scope: str, labels: list[str]) -> np.ndarray:
    filtered = [row for row in rows if row["Scope"] == scope]
    m = np.zeros((len(filtered), len(labels)), dtype=float)
    for i, row in enumerate(filtered):
        for j, label in enumerate(labels):
            m[i, j] = float(row.get(f"To_{label}", 0) or 0)
    return m


def draw_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, subtitle: str, path: Path) -> None:
    cell = 82 if len(row_labels) <= 4 else 58
    left = 170
    top = 120
    width = left + cell * len(col_labels) + 120
    height = top + cell * len(row_labels) + 130
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((35, 25), title, font=F_TITLE, fill=COLORS["dark"])
    draw.text((35, 62), subtitle, font=F_SMALL, fill=COLORS["muted"])
    max_value = float(np.max(matrix)) if matrix.size else 1
    for j, label in enumerate(col_labels):
        draw.text((left + j * cell + 12, top - 28), label, font=F_SMALL, fill=COLORS["dark"])
    for i, label in enumerate(row_labels):
        draw.text((35, top + i * cell + cell // 2 - 8), label, font=F_SMALL, fill=COLORS["dark"])
        for j in range(len(col_labels)):
            value = float(matrix[i, j])
            intensity = value / max_value if max_value else 0
            color = tuple(int(255 * (1 - intensity) + COLORS["blue"][k] * intensity) for k in range(3))
            x = left + j * cell
            y = top + i * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=color, outline="white", width=2)
            text_color = "white" if intensity > 0.45 else COLORS["dark"]
            draw.text((x + 8, y + cell // 2 - 10), f"{value:.3f}", font=F_TINY, fill=text_color)
    img.save(path)


def draw_stationary_chart(stationary_rows: list[dict[str, Any]]) -> None:
    rows = [
        row for row in stationary_rows
        if row["Scope"] == "Input_All_Triplets" and row["Level"] == "MC4_from_8_state"
    ]
    width, height = 940, 540
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((35, 25), "Stationary distribution on MC0-MC3", font=F_TITLE, fill=COLORS["dark"])
    draw.text((35, 62), "Aggregated from the empirical 8-state transition matrix.", font=F_SMALL, fill=COLORS["muted"])
    left, top, right, bottom = 95, 115, 850, 430
    max_value = max(float(row["Stationary_Probability"]) for row in rows) * 1.15
    for tick in np.linspace(0, max_value, 6):
        y = int(top + (max_value - tick) / max_value * (bottom - top))
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        draw.text((40, y - 8), f"{tick:.2f}", font=F_TINY, fill=COLORS["muted"])
    draw.line((left, bottom, right, bottom), fill=COLORS["dark"], width=2)
    draw.line((left, top, left, bottom), fill=COLORS["dark"], width=2)
    bar_w = 88
    gap = 90
    for i, row in enumerate(rows):
        value = float(row["Stationary_Probability"])
        x = left + 95 + i * (bar_w + gap)
        y = int(top + (max_value - value) / max_value * (bottom - top))
        draw.rectangle((x, y, x + bar_w, bottom), fill=[COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]][i])
        draw.text((x + 15, bottom + 18), row["State"], font=F_SMALL, fill=COLORS["dark"])
        draw.text((x + 8, y - 22), f"{value:.3f}", font=F_TINY, fill=COLORS["dark"])
    img.save(CHART_DIR / "mc_stationary_distribution.png")


def draw_mc2_chart(rows: list[dict[str, Any]]) -> None:
    width, height = 1040, 520
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((35, 25), "Why MC2 alone cannot identify BN structure", font=F_TITLE, fill=COLORS["dark"])
    draw.text((35, 62), "MC2 groups three spatially different states: 110, 101, and 011.", font=F_SMALL, fill=COLORS["muted"])
    x_positions = [95, 385, 675]
    for x, row in zip(x_positions, rows):
        draw.rounded_rectangle((x, 120, x + 245, 390), radius=8, outline=COLORS["grid"], width=2, fill=(248, 250, 252))
        draw.text((x + 20, 140), row["State"], font=F_TITLE, fill=COLORS["blue"])
        draw.text((x + 20, 190), row["Spatial_Pattern"], font=F_SMALL, fill=COLORS["dark"])
        draw.text((x + 20, 230), f"Next: {row['Allowed_Next_States']}", font=F_SMALL, fill=COLORS["muted"])
        draw.text((x + 20, 255), f"MC: {row['Allowed_Next_MC']}", font=F_SMALL, fill=COLORS["muted"])
        text = row["BN_Implication"]
        words = text.split()
        lines = []
        current = []
        for word in words:
            if sum(len(w) for w in current) + len(word) + len(current) > 30:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        for i, line in enumerate(lines[:5]):
            draw.text((x + 20, 305 + i * 18), line, font=F_TINY, fill=COLORS["dark"])
    draw.text((35, 455), "Conclusion: MC-level transitions count methyl groups, but BN structure needs the concrete positional pattern.", font=F_SMALL, fill=COLORS["muted"])
    img.save(CHART_DIR / "mc2_bn_structure_implication.png")


def write_latex(tables: dict[str, list[dict[str, Any]]]) -> None:
    tex = r"""\subsection{MC-level and State-level Transition Matrices}
\label{subsec:mc-state-transition}

Following the methylation-class idea used by Affinito et al., we define
\(\mathrm{MC}k\) as the class of triplet states with exactly \(k\) methylated
CpG sites. For three CpG sites,
\[
\begin{aligned}
\mathrm{MC0} &= \{000\},\\
\mathrm{MC1} &= \{001,010,100\},\\
\mathrm{MC2} &= \{011,101,110\},\\
\mathrm{MC3} &= \{111\}.
\end{aligned}
\]
The induced four-state transition matrix has the structural form
\[
P_{\mathrm{MC}}=
\begin{pmatrix}
p_{00} & p_{01} & 0      & 0\\
p_{10} & p_{11} & p_{12} & 0\\
0      & p_{21} & p_{22} & p_{23}\\
0      & 0      & p_{32} & p_{33}
\end{pmatrix},
\]
where each row sums to one. The zeros are not arbitrary: under a sliding
triplet window, the methylation count can change by at most one because
\[
C_{n+1}=C_n-X_n+X_{n+3}.
\]

\paragraph{Eight-state representation.}
The corresponding state-level chain is defined on
\[
\mathcal{S}=\{000,001,010,100,011,101,110,111\}.
\]
If the current triplet state is \(abc\), then the next sliding window must be
either \(bc0\) or \(bc1\). Thus the eight-state transition matrix \(T\) must
satisfy
\[
T_{abc,t}=0
\quad\text{unless}\quad
t\in\{bc0,bc1\}.
\]
This overlap constraint is the state-level version of the structural zeros in
\(P_{\mathrm{MC}}\).

\paragraph{Agreement between the two matrices.}
Let \(g(s)\) map an eight-state triplet state \(s\) to its methylation class.
If \(\pi_s\) is the stationary distribution of the eight-state chain, then the
aggregated methylation-class stationary distribution is
\[
\pi_i^{\mathrm{MC}}=\sum_{s:g(s)=i}\pi_s.
\]
The MC-level transition matrix induced by the eight-state chain is
\[
P^{\mathrm{MC}}_{ij}
=
\frac{
\sum_{s:g(s)=i}\pi_s\sum_{t:g(t)=j}T_{st}
}{
\sum_{s:g(s)=i}\pi_s
}.
\]
With this definition, \(\pi^{\mathrm{MC}}P^{\mathrm{MC}}=\pi^{\mathrm{MC}}\)
by construction.

\paragraph{Non-uniqueness and loss of positional information.}
An MC-level matrix does not uniquely determine an eight-state matrix. The
reason is that each MC class combines states with different positions of
methylated CpGs. For example,
\[
\mathrm{MC2}=\{110,101,011\}.
\]
These three states all contain two methylated CpGs, but they have different
spatial patterns. Under the sliding-window rule, \(011\) can move to
\(\mathrm{MC3}\), whereas \(110\) and \(101\) cannot. Therefore, knowing only
that a triplet belongs to MC2 is insufficient to identify the local Bayesian
network structure of \((X_n,X_{n+1},X_{n+2})\). The MC chain summarizes changes
in methylation count, while the BN analysis requires the full state-level
positional information.
"""
    (OUT_DIR / "mc_transition_equivalence_bn_en.tex").write_text(tex, encoding="utf-8")


def write_notes(tables: dict[str, list[dict[str, Any]]]) -> None:
    stationary = [
        row for row in tables["stationary_distributions"]
        if row["Scope"] == "Input_All_Triplets" and row["Level"] == "MC4_from_8_state"
    ]
    station_text = "\n".join(f"- {row['State']}: {row['Stationary_Probability']}" for row in stationary)
    text = f"""# MC transition equivalence and BN implication

## 本周任务结论

1. MC0-MC3 的 transition matrix 应该是一个 4x4 matrix，但因为滑动窗口每次只移走一个 CpG、加入一个新 CpG，所以 MC count 一次最多变化 1。

2. 8-state matrix 的状态是 `000,001,010,100,011,101,110,111`。如果当前 state 是 `abc`，下一步只能是 `bc0` 或 `bc1`。

3. 4-state MC matrix 不能唯一决定 8-state matrix，因为 MC class 丢掉了位置。例如 MC2 包含 `110,101,011`，但这三个状态的空间含义不同。

4. 因此 MC-level Markov chain 可以描述 methylation count 的变化，但不能单独推出 `(X_n,X_{{n+1}},X_{{n+2}})` 的 BN structure。

## Empirical stationary distribution, all input triplets

{station_text}

## 给老师解释的一句话

MC2 only tells us that two CpGs are methylated, but it does not tell us whether the methylated sites are adjacent on the left, adjacent on the right, or separated by an unmethylated middle CpG. Therefore, MC2 transitions are useful for methylation-count dynamics, but the BN structure needs the full 8-state positional information.
"""
    (OUT_DIR / "mc_transition_equivalence_chinese_notes.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    analysis = analyze_data()
    tables = build_tables(analysis)

    for name, rows in tables.items():
        write_csv(OUT_DIR / f"{name}.csv", rows)

    input_mc4 = matrix_from_rows(tables["mc4_transition_matrix_raw"], "Input_All_Triplets", MC_ORDER)
    input_t8 = matrix_from_rows(tables["state8_transition_matrix"], "Input_All_Triplets", STATE_ORDER)
    draw_heatmap(
        input_mc4,
        MC_ORDER,
        MC_ORDER,
        "Empirical MC0-MC3 transition matrix",
        "Raw count-normalized macro transitions from overlap-consistent state transitions.",
        CHART_DIR / "mc4_transition_heatmap.png",
    )
    draw_heatmap(
        input_t8,
        STATE_ORDER,
        STATE_ORDER,
        "Empirical 8-state transition matrix",
        "Only overlap-compatible moves abc -> bc0 or bc1 can be nonzero.",
        CHART_DIR / "state8_transition_heatmap.png",
    )
    draw_stationary_chart(tables["stationary_distributions"])
    draw_mc2_chart(tables["mc2_bn_implication"])
    write_latex(tables)
    write_notes(tables)

    workbook_data = {
        **tables,
        "charts": [
            {
                "Chart": "mc4_transition_heatmap.png",
                "Meaning_CN": "MC0-MC3 的 empirical transition matrix。",
                "Path": str(CHART_DIR / "mc4_transition_heatmap.png"),
            },
            {
                "Chart": "state8_transition_heatmap.png",
                "Meaning_CN": "8 个具体 methylation states 的 empirical transition matrix。",
                "Path": str(CHART_DIR / "state8_transition_heatmap.png"),
            },
            {
                "Chart": "mc_stationary_distribution.png",
                "Meaning_CN": "从 8-state matrix 聚合得到的 MC stationary distribution。",
                "Path": str(CHART_DIR / "mc_stationary_distribution.png"),
            },
            {
                "Chart": "mc2_bn_structure_implication.png",
                "Meaning_CN": "为什么 MC2 本身不能确定 triplet BN structure。",
                "Path": str(CHART_DIR / "mc2_bn_structure_implication.png"),
            },
        ],
        "latex": str(OUT_DIR / "mc_transition_equivalence_bn_en.tex"),
        "chinese_notes": str(OUT_DIR / "mc_transition_equivalence_chinese_notes.md"),
        "affinito_note": "Affinito et al. define MCs as the number of methylated cytosines per molecule, irrespective of position, and model MC dynamics with a discrete-time Markov chain.",
    }
    (OUT_DIR / "workbook_data.json").write_text(json.dumps(clean_json(workbook_data), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "output_dir": str(OUT_DIR),
        "xlsx_planned": str(OUT_DIR / "mc_transition_equivalence_bn_summary.xlsx"),
        "latex": str(OUT_DIR / "mc_transition_equivalence_bn_en.tex"),
        "chinese_notes": str(OUT_DIR / "mc_transition_equivalence_chinese_notes.md"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(clean_json({
        "output_dir": str(OUT_DIR),
        "diagnostics": tables["transition_diagnostics"],
        "agreement": tables["agreement_check"],
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
