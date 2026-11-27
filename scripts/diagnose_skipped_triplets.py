from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
OUT_DIR = ROOT / "outputs" / "skipped_triplet_diagnosis"
CHART_DIR = OUT_DIR / "png_charts"

STATES = ["000", "001", "010", "011", "100", "101", "110", "111"]
COUNT_COLS = [f"n_{state}" for state in STATES]
META_COLS = ["Chromosome number", "Triplet index (first CpG)", "Triplet location (centroid)"]
VARS = ["X1", "X2", "X3"]


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
F_LABEL = font(17)
F_SMALL = font(13)
F_TINY = font(11)

COLORS = {
    "dark": (17, 24, 39),
    "muted": (107, 114, 128),
    "grid": (229, 231, 235),
    "blue": (37, 99, 235),
    "green": (22, 163, 74),
    "orange": (245, 158, 11),
    "red": (220, 38, 38),
    "slate": (100, 116, 139),
}


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


def bit_sums(counts: dict[str, int]) -> dict[str, int]:
    return {
        "X1_ones": sum(counts[state] for state in STATES if state[0] == "1"),
        "X2_ones": sum(counts[state] for state in STATES if state[1] == "1"),
        "X3_ones": sum(counts[state] for state in STATES if state[2] == "1"),
    }


def deterministic_status(total: int, ones: int, var: str) -> tuple[str, str | None]:
    if total == 0:
        return f"{var}_undefined_total0", None
    if ones == 0:
        return f"{var}=0_constant", "0"
    if ones == total:
        return f"{var}=1_constant", "1"
    return f"{var}_varies", None


def valid_mask(df: pd.DataFrame) -> pd.Series:
    total = df[COUNT_COLS].sum(axis=1)
    x1_ones = df[["n_100", "n_101", "n_110", "n_111"]].sum(axis=1)
    x2_ones = df[["n_010", "n_011", "n_110", "n_111"]].sum(axis=1)
    x3_ones = df[["n_001", "n_011", "n_101", "n_111"]].sum(axis=1)
    return (
        (total > 0)
        & (x1_ones > 0) & (x1_ones < total)
        & (x2_ones > 0) & (x2_ones < total)
        & (x3_ones > 0) & (x3_ones < total)
    )


def classify_row(counts: dict[str, int]) -> dict[str, Any]:
    total = sum(counts.values())
    sums = bit_sums(counts)
    statuses: dict[str, str] = {}
    constants: dict[str, str] = {}
    for var, key in zip(VARS, ["X1_ones", "X2_ones", "X3_ones"]):
        status, constant = deterministic_status(total, sums[key], var)
        statuses[f"{var}_Status"] = status
        if constant is not None:
            constants[var] = constant

    nonzero_states = [state for state in STATES if counts[state] > 0]
    dominant_state = max(STATES, key=lambda state: counts[state]) if total > 0 else ""
    deterministic_pattern = ""
    if len(constants) == 3:
        deterministic_pattern = constants["X1"] + constants["X2"] + constants["X3"]

    if total == 0:
        reason = "zero_total_count"
    elif len(constants) == 3 and len(nonzero_states) == 1:
        reason = "all_three_variables_constant_single_state"
    elif len(constants) == 3:
        reason = "all_three_variables_constant"
    elif len(constants) == 2:
        reason = "two_variables_constant"
    elif len(constants) == 1:
        reason = "one_variable_constant"
    else:
        reason = "other"

    return {
        "Total_Count": total,
        "Nonzero_State_Count": len(nonzero_states),
        "Nonzero_States": ", ".join(nonzero_states),
        "Dominant_State": dominant_state,
        "Dominant_State_Count": counts[dominant_state] if total > 0 else 0,
        "Dominant_State_Proportion": round(counts[dominant_state] / total, 8) if total > 0 else None,
        "Deterministic_Pattern": deterministic_pattern,
        "Constant_Variable_Count": len(constants),
        "Skipped_Reason": reason,
        **sums,
        **statuses,
    }


def load_summary_metadata() -> dict[str, dict[str, Any]]:
    path = ROOT / "outputs" / "all_dataset_analysis" / "dataset_compatibility_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {
        row["File_ID"]: {
            "Experiment_ID": row.get("Experiment_ID", ""),
            "Tissue": row.get("Tissue", ""),
            "Previous_Input_Triplets": int(row.get("Input_Triplets", 0)),
            "Previous_Valid_Triplets": int(row.get("Valid_Triplets", 0)),
            "Previous_Skipped_Triplets": int(row.get("Skipped_Triplets", 0)),
        }
        for _, row in df.iterrows()
    }


def analyze() -> dict[str, Any]:
    metadata = load_summary_metadata()
    skipped_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    pattern_counter: Counter[str] = Counter()
    constant_count_counter: Counter[int] = Counter()
    examples_by_reason: dict[str, dict[str, Any]] = {}
    examples_by_pattern: dict[str, dict[str, Any]] = {}

    for path in sorted(ROOT.glob("ENCFF*.tsv")):
        file_id = path.stem
        meta = metadata.get(file_id, {})
        df = pd.read_csv(path, sep="\t", usecols=META_COLS + COUNT_COLS)
        for col in COUNT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype(int)
        mask = valid_mask(df)
        skipped = df.loc[~mask].copy()
        dataset_reason_counter: Counter[str] = Counter()

        for row_index, (_, row) in enumerate(skipped.iterrows(), start=1):
            counts = {state: int(row[f"n_{state}"]) for state in STATES}
            diagnosis = classify_row(counts)
            record = {
                "File_ID": file_id,
                "Experiment_ID": meta.get("Experiment_ID", ""),
                "Tissue": meta.get("Tissue", ""),
                "Original_Row_Number_1Based_With_Header": int(row.name) + 2,
                "Chromosome": int(float(row["Chromosome number"])) if pd.notna(row["Chromosome number"]) else None,
                "Triplet_Index_First_CpG": int(float(row["Triplet index (first CpG)"])) if pd.notna(row["Triplet index (first CpG)"]) else None,
                "Triplet_Location_Centroid": int(float(row["Triplet location (centroid)"])) if pd.notna(row["Triplet location (centroid)"]) else None,
                **diagnosis,
                **{f"n_{state}": counts[state] for state in STATES},
            }
            skipped_rows.append(record)
            reason_counter[diagnosis["Skipped_Reason"]] += 1
            dataset_reason_counter[diagnosis["Skipped_Reason"]] += 1
            constant_count_counter[diagnosis["Constant_Variable_Count"]] += 1
            if diagnosis["Deterministic_Pattern"]:
                pattern_counter[diagnosis["Deterministic_Pattern"]] += 1
                examples_by_pattern.setdefault(diagnosis["Deterministic_Pattern"], record)
            examples_by_reason.setdefault(diagnosis["Skipped_Reason"], record)

        dataset_rows.append({
            "File_ID": file_id,
            "Experiment_ID": meta.get("Experiment_ID", ""),
            "Tissue": meta.get("Tissue", ""),
            "Input_Triplets": int(len(df)),
            "Valid_Triplets": int(mask.sum()),
            "Skipped_Triplets": int((~mask).sum()),
            "Skipped_Percent": round((~mask).sum() / len(df) * 100, 6) if len(df) else 0,
            **{f"Reason_{reason}": count for reason, count in dataset_reason_counter.items()},
        })

    reason_rows = [
        {
            "Skipped_Reason": reason,
            "Triplet_Count": count,
            "Percent_of_Skipped": round(count / len(skipped_rows) * 100, 6) if skipped_rows else 0,
            "Explanation_CN": {
                "all_three_variables_constant_single_state": "三个 CpG 都没有变化，所有 reads 都落在同一个 000-111 状态。",
                "two_variables_constant": "有两个 CpG 没有变化，另一个 CpG 有变化。",
                "one_variable_constant": "有一个 CpG 没有变化，另外两个 CpG 有变化。",
                "zero_total_count": "该 triplet 的总 count 为 0。",
                "all_three_variables_constant": "三个 CpG 都没有变化。",
                "other": "其他原因。",
            }.get(reason, ""),
        }
        for reason, count in reason_counter.most_common()
    ]
    pattern_rows = [
        {
            "Deterministic_Pattern": pattern,
            "Triplet_Count": count,
            "Percent_of_Skipped": round(count / len(skipped_rows) * 100, 6) if skipped_rows else 0,
            "Meaning_CN": f"所有 reads 都是状态 {pattern}。",
        }
        for pattern, count in pattern_counter.most_common()
    ]
    constant_count_rows = [
        {
            "Constant_Variable_Count": count_key,
            "Triplet_Count": count,
            "Percent_of_Skipped": round(count / len(skipped_rows) * 100, 6) if skipped_rows else 0,
        }
        for count_key, count in sorted(constant_count_counter.items())
    ]

    one_good_example = (
        examples_by_pattern.get("111")
        or examples_by_pattern.get("000")
        or (skipped_rows[0] if skipped_rows else {})
    )

    return {
        "skipped_rows": skipped_rows,
        "dataset_rows": dataset_rows,
        "reason_rows": reason_rows,
        "pattern_rows": pattern_rows,
        "constant_count_rows": constant_count_rows,
        "examples_by_reason": list(examples_by_reason.values()),
        "examples_by_pattern": list(examples_by_pattern.values()),
        "one_good_example": one_good_example,
        "totals": {
            "Datasets": len(dataset_rows),
            "Input_Triplets": sum(row["Input_Triplets"] for row in dataset_rows),
            "Valid_Triplets": sum(row["Valid_Triplets"] for row in dataset_rows),
            "Skipped_Triplets": len(skipped_rows),
            "Skipped_Percent": round(len(skipped_rows) / sum(row["Input_Triplets"] for row in dataset_rows) * 100, 6),
        },
    }


def draw_reason_chart(reason_rows: list[dict[str, Any]]) -> None:
    width, height = 1240, 620
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((35, 25), "Why were triplets skipped?", font=F_TITLE, fill=COLORS["dark"])
    draw.text((35, 62), "Skipped = at least one CpG variable has no empirical variation; zeros are observed zero counts, not missing data.", font=F_SMALL, fill=COLORS["muted"])

    left, top, right, bottom = 360, 125, 1060, 500
    max_count = max(row["Triplet_Count"] for row in reason_rows) if reason_rows else 1
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"], COLORS["slate"]]
    bar_h = 44
    gap = 24
    for i, row in enumerate(reason_rows):
        y = top + i * (bar_h + gap)
        label = row["Skipped_Reason"].replace("_", " ")
        count = row["Triplet_Count"]
        pct = row["Percent_of_Skipped"]
        draw.text((35, y + 10), label[:42], font=F_SMALL, fill=COLORS["dark"])
        bar_w = int(count / max_count * (right - left))
        draw.rectangle((left, y, left + bar_w, y + bar_h), fill=colors[i % len(colors)])
        count_label = f"{count:,} ({pct:.1f}%)"
        if bar_w > 190:
            draw.text((left + bar_w - 150, y + 10), count_label, font=F_SMALL, fill="white")
        else:
            draw.text((left + bar_w + 10, y + 10), count_label, font=F_SMALL, fill=COLORS["dark"])
    draw.text((35, 560), "Interpretation: these rows are complete deterministic or partially deterministic distributions, but were filtered out by the old variation rule.", font=F_SMALL, fill=COLORS["muted"])
    img.save(CHART_DIR / "skipped_reason_distribution.png")


def draw_pattern_chart(pattern_rows: list[dict[str, Any]]) -> None:
    rows = pattern_rows[:8]
    width, height = 1000, 560
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((35, 25), "Single-state deterministic skipped triplets", font=F_TITLE, fill=COLORS["dark"])
    draw.text((35, 62), "When all three variables are constant, all reads fall into one of the eight states.", font=F_SMALL, fill=COLORS["muted"])

    left, top, right, bottom = 95, 120, 920, 440
    max_count = max(row["Triplet_Count"] for row in rows) if rows else 1
    bar_width = 72
    gap = 34
    for i, row in enumerate(rows):
        x = left + i * (bar_width + gap)
        count = row["Triplet_Count"]
        h = int(count / max_count * (bottom - top))
        draw.rectangle((x, bottom - h, x + bar_width, bottom), fill=COLORS["blue"])
        draw.text((x + 8, bottom + 16), row["Deterministic_Pattern"], font=F_LABEL, fill=COLORS["dark"])
        draw.text((x - 2, bottom - h - 22), f"{count:,}", font=F_TINY, fill=COLORS["blue"])
    draw.line((left - 20, bottom, right, bottom), fill=COLORS["dark"], width=2)
    draw.text((35, 505), "Example: if only n_111 is positive, then P(X1=1)=P(X2=1)=P(X3=1)=1 and the empirical distribution is complete.", font=F_SMALL, fill=COLORS["muted"])
    img.save(CHART_DIR / "deterministic_pattern_distribution.png")


def write_explanation_files(results: dict[str, Any]) -> None:
    example = results["one_good_example"]
    counts_line = ", ".join(f"n_{state}={example.get(f'n_{state}', '')}" for state in STATES)
    cn = f"""# Skipped triplets diagnosis

## 结论

这些 skipped triplets 不是因为数据不完整，也不是因为 0 表示 missing。原始数据是完整的；0 的意思是这个状态在 reads 中没有被观察到。

旧 pipeline 把它们 skip 掉，是因为使用了 variation filter：如果三个 CpG 变量中至少一个变量在该 triplet 内永远是 0 或永远是 1，那么这个 triplet 不进入 DAG structure learning。

## 总数

- Datasets: {results['totals']['Datasets']}
- Input triplets: {results['totals']['Input_Triplets']:,}
- Valid triplets: {results['totals']['Valid_Triplets']:,}
- Skipped triplets: {results['totals']['Skipped_Triplets']:,}
- Skipped percent: {results['totals']['Skipped_Percent']:.4f}%

## 可以跟老师解释的例子

File: {example.get('File_ID')}

Original row number: {example.get('Original_Row_Number_1Based_With_Header')}

Chromosome: {example.get('Chromosome')}

Triplet index: {example.get('Triplet_Index_First_CpG')}

Counts:

`{counts_line}`

这个 triplet 被 skip 的原因是：{example.get('Skipped_Reason')}。

如果它是只有一个状态有 count，比如只有 `n_111` 或 `n_000` 非零，那么它是一个退化但完整的经验分布。比如只有 `n_111` 非零时：

`P(X1=1)=1, P(X2=1)=1, P(X3=1)=1`

所以其他状态为 0 并不表示缺失；只是这些状态没有在 reads 中出现。

## 建议修改说法

不要说 skipped triplets 是 incomplete data。建议说：

`Skipped triplets are complete but non-varying or partially deterministic triplets. They were excluded by the previous variation filter because at least one CpG variable has no empirical variation, which makes DAG structure learning non-informative or non-identifiable for that variable.`
"""
    (OUT_DIR / "skipped_triplets_chinese_explanation.md").write_text(cn, encoding="utf-8")

    tex = r"""\subsection{Skipped Triplets and Non-varying CpG Variables}
\label{subsec:skipped-triplets}

The skipped triplets in our previous pipeline should not be interpreted as
incomplete observations. The input count vector
\[
  (n_{000}, n_{001}, n_{010}, n_{011}, n_{100}, n_{101}, n_{110}, n_{111})
\]
is complete for each triplet. A zero count means that the corresponding
methylation state was not observed among the reads; it does not mean that the
state is missing.

The previous pipeline skipped a triplet when at least one of the three CpG
variables had no empirical variation. For a triplet with total count
\[
  N = \sum_{a,b,c\in\{0,1\}} n_{abc},
\]
the variable \(X_k\) has no variation if
\[
  \sum_{\mathbf{x}:x_k=1} n_{\mathbf{x}} = 0
  \quad\text{or}\quad
  \sum_{\mathbf{x}:x_k=1} n_{\mathbf{x}} = N.
\]
Thus, a skipped triplet is a complete but non-varying, or partially
deterministic, empirical distribution.

For example, if \(n_{111}=N\) and all other state counts are zero, then
\[
  \widehat{P}(X_1=1)=
  \widehat{P}(X_2=1)=
  \widehat{P}(X_3=1)=1.
\]
This triplet is complete and the zero entries are meaningful observed zero
counts. It is skipped only because all three variables are deterministic, so
the local DAG structure is not informative from the data. In other words, the
skip rule is a structure-learning filter, not a missing-data filter.
"""
    (OUT_DIR / "skipped_triplets_latex_explanation_en.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    results = analyze()

    write_csv(OUT_DIR / "skipped_triplets_detail.csv", results["skipped_rows"])
    write_csv(OUT_DIR / "skipped_by_dataset.csv", results["dataset_rows"])
    write_csv(OUT_DIR / "skipped_reason_summary.csv", results["reason_rows"])
    write_csv(OUT_DIR / "deterministic_pattern_summary.csv", results["pattern_rows"])
    write_csv(OUT_DIR / "constant_variable_count_summary.csv", results["constant_count_rows"])
    write_csv(OUT_DIR / "examples_by_reason.csv", results["examples_by_reason"])
    write_csv(OUT_DIR / "examples_by_pattern.csv", results["examples_by_pattern"])

    draw_reason_chart(results["reason_rows"])
    draw_pattern_chart(results["pattern_rows"])
    write_explanation_files(results)

    workbook_data = {
        "totals": results["totals"],
        "reason_rows": results["reason_rows"],
        "pattern_rows": results["pattern_rows"],
        "constant_count_rows": results["constant_count_rows"],
        "dataset_rows": results["dataset_rows"],
        "examples_by_reason": results["examples_by_reason"],
        "examples_by_pattern": results["examples_by_pattern"],
        "skipped_rows_preview": results["skipped_rows"][:500],
        "one_good_example": results["one_good_example"],
        "charts": [
            {
                "Chart": "skipped_reason_distribution.png",
                "Meaning_CN": "skipped triplets 按原因分类的数量。",
                "Path": str(CHART_DIR / "skipped_reason_distribution.png"),
            },
            {
                "Chart": "deterministic_pattern_distribution.png",
                "Meaning_CN": "三变量都 constant 且只有单一状态非零的 skipped triplets 分布。",
                "Path": str(CHART_DIR / "deterministic_pattern_distribution.png"),
            },
        ],
        "latex": str(OUT_DIR / "skipped_triplets_latex_explanation_en.tex"),
        "chinese_explanation": str(OUT_DIR / "skipped_triplets_chinese_explanation.md"),
    }
    (OUT_DIR / "workbook_data.json").write_text(json.dumps(workbook_data, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "output_dir": str(OUT_DIR),
        "detail_csv": str(OUT_DIR / "skipped_triplets_detail.csv"),
        "xlsx_planned": str(OUT_DIR / "skipped_triplets_diagnosis.xlsx"),
        "latex": workbook_data["latex"],
        "chinese_explanation": workbook_data["chinese_explanation"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(workbook_data["totals"], ensure_ascii=False, indent=2))
    print(f"Example: {results['one_good_example'].get('File_ID')} row {results['one_good_example'].get('Original_Row_Number_1Based_With_Header')}")


if __name__ == "__main__":
    main()
