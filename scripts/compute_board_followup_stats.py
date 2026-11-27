from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
ANALYSIS_DIR = ROOT / "outputs" / "all_dataset_analysis"
SUMMARY_PATH = ANALYSIS_DIR / "dataset_compatibility_summary.csv"
CLASS_DIR = ANALYSIS_DIR / "s_class_results"
OUT_DIR = ANALYSIS_DIR / "board_followup_stats"
CHART_DIR = OUT_DIR / "png_charts"

METHODS = ["BIC", "AIC", "BDeu"]
S_CLASSES = [f"S{i}" for i in range(1, 12)]

COLORS = {
    "BIC": (37, 99, 235),
    "AIC": (220, 38, 38),
    "BDeu": (22, 163, 74),
    "dark": (17, 24, 39),
    "muted": (107, 114, 128),
    "grid": (229, 231, 235),
    "gap": (245, 158, 11),
    "boundary": (148, 163, 184),
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


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)


def round_float(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def weighted_compatibility(summary: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    per_dataset_rows: list[dict] = []
    for method in METHODS:
        comparable_col = f"{method}_Comparable_Pairs"
        compatible_col = f"{method}_Compatible_Pairs"
        percent_col = f"{method}_Compatibility_Percent"

        comparable = summary[comparable_col].astype(float)
        compatible = summary[compatible_col].astype(float)
        total_comparable = comparable.sum()
        total_compatible = compatible.sum()
        weighted_pct = total_compatible / total_comparable * 100 if total_comparable else float("nan")
        simple_mean = summary[percent_col].mean()
        median = summary[percent_col].median()

        rows.append({
            "Method": method,
            "Datasets": int((comparable > 0).sum()),
            "Total_Comparable_Pairs": int(total_comparable),
            "Total_Compatible_Pairs": int(total_compatible),
            "Total_Incompatible_Pairs": int(total_comparable - total_compatible),
            "Weighted_Compatibility_Percent": round_float(weighted_pct),
            "Simple_Mean_Dataset_Percent": round_float(simple_mean),
            "Dataset_Median_Percent": round_float(median),
            "Weighted_minus_Simple_Mean": round_float(weighted_pct - simple_mean),
            "Formula": "sum_j compatible_pairs_j / sum_j comparable_pairs_j * 100",
        })

        for _, row in summary.iterrows():
            n_j = float(row[comparable_col])
            pct_j = float(row[percent_col])
            per_dataset_rows.append({
                "Method": method,
                "File_ID": row["File_ID"],
                "Experiment_ID": row["Experiment_ID"],
                "Tissue": row["Tissue"],
                "n_j_Comparable_Pairs": int(n_j),
                "Compatible_Pairs": int(row[compatible_col]),
                "Compatibility_Percent": round_float(pct_j),
                "Weighted_Numerator_Contribution": round_float(n_j * pct_j),
                "Weight_in_Method_Total": round_float(n_j / total_comparable if total_comparable else None, 10),
            })
    return pd.DataFrame(rows), per_dataset_rows


def sample_size_by_s_class(class_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    accum: dict[tuple[str, str], list[float]] = {(m, s): [] for m in METHODS for s in S_CLASSES}
    total_triplets_by_method = defaultdict(int)

    for path in class_files:
        df = pd.read_csv(path, usecols=["Total_Count", "Class_BIC", "Class_AIC", "Class_BDeu"])
        for method in METHODS:
            class_col = f"Class_{method}"
            total_triplets_by_method[method] += len(df)
            grouped = df.groupby(class_col)["Total_Count"]
            for s_class, values in grouped:
                if s_class in S_CLASSES:
                    accum[(method, s_class)].extend(values.astype(float).tolist())

    detail_rows: list[dict] = []
    wide_mean_rows: list[dict] = []
    wide_count_rows: list[dict] = []
    for method in METHODS:
        mean_row = {"Method": method, "Measure": "Mean_Total_Count"}
        count_row = {"Method": method, "Measure": "Triplet_Count"}
        for s_class in S_CLASSES:
            values = pd.Series(accum[(method, s_class)], dtype=float)
            count = int(values.count())
            mean_value = float(values.mean()) if count else float("nan")
            median_value = float(values.median()) if count else float("nan")
            std_value = float(values.std(ddof=1)) if count > 1 else None
            min_value = float(values.min()) if count else None
            max_value = float(values.max()) if count else None
            total_reads = float(values.sum()) if count else 0.0
            detail_rows.append({
                "Method": method,
                "S_Class": s_class,
                "Triplet_Count": count,
                "Share_of_Method_Triplets_Percent": round_float(count / total_triplets_by_method[method] * 100 if total_triplets_by_method[method] else None),
                "Mean_Total_Count": round_float(mean_value),
                "Median_Total_Count": round_float(median_value),
                "Std_Total_Count": round_float(std_value),
                "Min_Total_Count": round_float(min_value),
                "Max_Total_Count": round_float(max_value),
                "Total_Count_Sum": int(total_reads),
            })
            mean_row[s_class] = round_float(mean_value)
            count_row[s_class] = count
        wide_mean_rows.append(mean_row)
        wide_count_rows.append(count_row)

    wide = pd.DataFrame(wide_mean_rows + wide_count_rows)
    detail = pd.DataFrame(detail_rows)
    return wide, detail


def normalize_chromosome(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def comparable_pair_relationship(summary: pd.DataFrame, class_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_by_file = {str(row["File_ID"]): row for _, row in summary.iterrows()}
    rows: list[dict] = []

    for path in class_files:
        file_id = path.name.replace("_classes.csv", "")
        df = pd.read_csv(path, usecols=["Chromosome", "Triplet_Index_First_CpG"])
        df["Chromosome"] = df["Chromosome"].map(normalize_chromosome)
        valid_triplets = int(len(df))

        computed_comparable = 0
        gap_broken_pairs = 0
        nonempty_chromosomes = 0
        for _, group in df.groupby("Chromosome", sort=False):
            indices = sorted(pd.Series(group["Triplet_Index_First_CpG"]).dropna().astype(int).unique().tolist())
            if not indices:
                continue
            nonempty_chromosomes += 1
            for left, right in zip(indices, indices[1:]):
                if right == left + 1:
                    computed_comparable += 1
                else:
                    gap_broken_pairs += 1

        max_pairs_if_one_run = max(valid_triplets - 1, 0)
        max_pairs_within_chrom = max(valid_triplets - nonempty_chromosomes, 0)
        chromosome_boundary_breaks = max(nonempty_chromosomes - 1, 0)
        summary_row = summary_by_file[file_id]
        summary_comparable = int(summary_row["BIC_Comparable_Pairs"])

        rows.append({
            "File_ID": file_id,
            "Experiment_ID": summary_row["Experiment_ID"],
            "Tissue": summary_row["Tissue"],
            "Valid_Triplets": valid_triplets,
            "Nonempty_Chromosomes": nonempty_chromosomes,
            "Comparable_Pairs": computed_comparable,
            "Comparable_Pairs_from_Summary": summary_comparable,
            "Comparable_Count_Check": "OK" if computed_comparable == summary_comparable else "CHECK",
            "Gap_Broken_Adjacent_Valid_Pairs": gap_broken_pairs,
            "Chromosome_Boundary_Breaks": chromosome_boundary_breaks,
            "Max_Pairs_If_All_Valid_Triplets_One_Run": max_pairs_if_one_run,
            "Max_Pairs_Within_Chromosomes": max_pairs_within_chrom,
            "Comparable_Per_ValidMinusOne_Percent": round_float(computed_comparable / max_pairs_if_one_run * 100 if max_pairs_if_one_run else None),
            "Comparable_Per_WithinChromosomePairs_Percent": round_float(computed_comparable / max_pairs_within_chrom * 100 if max_pairs_within_chrom else None),
            "Gap_Broken_Per_WithinChromosomePairs_Percent": round_float(gap_broken_pairs / max_pairs_within_chrom * 100 if max_pairs_within_chrom else None),
        })

    rel = pd.DataFrame(rows).sort_values(["Comparable_Per_WithinChromosomePairs_Percent", "File_ID"], ascending=[True, True])

    total_valid = int(rel["Valid_Triplets"].sum())
    total_comparable = int(rel["Comparable_Pairs"].sum())
    total_gap = int(rel["Gap_Broken_Adjacent_Valid_Pairs"].sum())
    total_boundary = int(rel["Chromosome_Boundary_Breaks"].sum())
    total_one_run = int(rel["Max_Pairs_If_All_Valid_Triplets_One_Run"].sum())
    total_within = int(rel["Max_Pairs_Within_Chromosomes"].sum())

    aggregate_rows = [
        {
            "Metric": "Datasets",
            "Value": int(len(rel)),
            "Percent_of_Max_One_Run": None,
            "Percent_of_Within_Chromosome_Max": None,
            "Explanation_CN": "参与统计的数据集数量。",
            "Explanation_EN": "Number of datasets included.",
        },
        {
            "Metric": "Valid_Triplets",
            "Value": total_valid,
            "Percent_of_Max_One_Run": None,
            "Percent_of_Within_Chromosome_Max": None,
            "Explanation_CN": "通过过滤、可以学习 BN 的 triplets 总数。",
            "Explanation_EN": "Total valid triplets after filtering.",
        },
        {
            "Metric": "Comparable_Pairs",
            "Value": total_comparable,
            "Percent_of_Max_One_Run": round_float(total_comparable / total_one_run * 100 if total_one_run else None),
            "Percent_of_Within_Chromosome_Max": round_float(total_comparable / total_within * 100 if total_within else None),
            "Explanation_CN": "真正用于 compatibility 的相邻 triplet pairs。",
            "Explanation_EN": "Adjacent triplet pairs actually used for compatibility.",
        },
        {
            "Metric": "Gap_Broken_Adjacent_Valid_Pairs",
            "Value": total_gap,
            "Percent_of_Max_One_Run": round_float(total_gap / total_one_run * 100 if total_one_run else None),
            "Percent_of_Within_Chromosome_Max": round_float(total_gap / total_within * 100 if total_within else None),
            "Explanation_CN": "同一 chromosome 内相邻 valid triplets 的 index 不连续，所以不能比较。",
            "Explanation_EN": "Adjacent valid triplets within a chromosome whose indices are not consecutive.",
        },
        {
            "Metric": "Chromosome_Boundary_Breaks",
            "Value": total_boundary,
            "Percent_of_Max_One_Run": round_float(total_boundary / total_one_run * 100 if total_one_run else None),
            "Percent_of_Within_Chromosome_Max": None,
            "Explanation_CN": "不同 chromosome 之间不能组成连续 pair。",
            "Explanation_EN": "Breaks between chromosomes; these cannot form comparable pairs.",
        },
        {
            "Metric": "Max_Pairs_If_All_Valid_Triplets_One_Run",
            "Value": total_one_run,
            "Percent_of_Max_One_Run": 100.0,
            "Percent_of_Within_Chromosome_Max": None,
            "Explanation_CN": "粗略上限：如果所有 valid triplets 排成一条连续序列，就是 N-1。",
            "Explanation_EN": "Crude upper bound: N-1 if all valid triplets were one continuous run.",
        },
        {
            "Metric": "Max_Pairs_Within_Chromosomes",
            "Value": total_within,
            "Percent_of_Max_One_Run": round_float(total_within / total_one_run * 100 if total_one_run else None),
            "Percent_of_Within_Chromosome_Max": 100.0,
            "Explanation_CN": "去掉 chromosome boundary 后的理论上限。",
            "Explanation_EN": "Upper bound after removing chromosome-boundary breaks.",
        },
    ]
    aggregate = pd.DataFrame(aggregate_rows)
    return rel, aggregate


def save_dataframe(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")


def clean_json_value(value):
    if isinstance(value, dict):
        return {key: clean_json_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [clean_json_value(inner) for inner in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def draw_axes(draw: ImageDraw.ImageDraw, left: int, top: int, right: int, bottom: int) -> None:
    draw.line((left, bottom, right, bottom), fill=COLORS["dark"], width=2)
    draw.line((left, top, left, bottom), fill=COLORS["dark"], width=2)


def draw_weighted_chart(weighted: pd.DataFrame) -> None:
    width, height = 980, 560
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((35, 25), "Pair-weighted structural compatibility", font=F_TITLE, fill=COLORS["dark"])
    draw.text((35, 62), "Weighted percent = total compatible pairs / total comparable pairs", font=F_SMALL, fill=COLORS["muted"])

    left, top, right, bottom = 110, 110, 900, 445
    y_min, y_max = 75.0, 90.0

    def y(value: float) -> int:
        return int(top + (y_max - value) / (y_max - y_min) * (bottom - top))

    for tick in [75, 78, 81, 84, 87, 90]:
        yy = y(tick)
        draw.line((left, yy, right, yy), fill=COLORS["grid"], width=1)
        draw.text((58, yy - 8), f"{tick}%", font=F_SMALL, fill=COLORS["muted"])
    draw_axes(draw, left, top, right, bottom)

    bar_width = 130
    gap = 90
    x = left + 100
    for _, row in weighted.iterrows():
        method = row["Method"]
        value = float(row["Weighted_Compatibility_Percent"])
        color = COLORS[method]
        y0 = y(value)
        draw.rectangle((x, y0, x + bar_width, bottom), fill=color)
        draw.text((x + 15, bottom + 18), method, font=F_LABEL, fill=COLORS["dark"])
        draw.text((x + 8, y0 - 28), f"{value:.2f}%", font=F_HEAD, fill=color)
        x += bar_width + gap

    draw.text((35, 500), "Each method is weighted by its number of comparable adjacent-window pairs.", font=F_SMALL, fill=COLORS["muted"])
    img.save(CHART_DIR / "weighted_compatibility.png")


def draw_sample_size_chart(wide: pd.DataFrame) -> None:
    means = wide[wide["Measure"] == "Mean_Total_Count"].copy()
    width, height = 1320, 620
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((35, 25), "Mean triplet sample size by S class", font=F_TITLE, fill=COLORS["dark"])
    draw.text((35, 62), "Sample size = n_000 + ... + n_111; means are grouped by method and S class.", font=F_SMALL, fill=COLORS["muted"])

    left, top, right, bottom = 95, 110, 1245, 505
    all_values = [float(row[s]) for _, row in means.iterrows() for s in S_CLASSES if not pd.isna(row[s])]
    y_min = 0.0
    y_max = math.ceil(max(all_values) / 50) * 50

    def x(index: int) -> int:
        return int(left + index / (len(S_CLASSES) - 1) * (right - left))

    def y(value: float) -> int:
        return int(top + (y_max - value) / (y_max - y_min) * (bottom - top))

    for tick in range(0, int(y_max) + 1, max(50, int(y_max // 5))):
        yy = y(tick)
        draw.line((left, yy, right, yy), fill=COLORS["grid"], width=1)
        draw.text((35, yy - 8), str(tick), font=F_SMALL, fill=COLORS["muted"])
    draw_axes(draw, left, top, right, bottom)

    for idx, s_class in enumerate(S_CLASSES):
        xx = x(idx)
        draw.text((xx - 13, bottom + 18), s_class, font=F_SMALL, fill=COLORS["dark"])

    for _, row in means.iterrows():
        method = row["Method"]
        points = [(x(i), y(float(row[s]))) for i, s in enumerate(S_CLASSES)]
        draw.line(points, fill=COLORS[method], width=4)
        for px, py in points:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=COLORS[method])

    legend_x, legend_y = 1040, 115
    for idx, method in enumerate(METHODS):
        yy = legend_y + idx * 34
        draw.line((legend_x, yy, legend_x + 45, yy), fill=COLORS[method], width=5)
        draw.text((legend_x + 55, yy - 10), method, font=F_LABEL, fill=COLORS["dark"])

    draw.text((35, 565), "Higher values indicate larger average read counts among triplets assigned to that S class.", font=F_SMALL, fill=COLORS["muted"])
    img.save(CHART_DIR / "mean_sample_size_by_s_class.png")


def draw_relationship_chart(rel: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    width, height = 1280, 680
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((35, 25), "Valid triplets and comparable adjacent-window pairs", font=F_TITLE, fill=COLORS["dark"])
    draw.text((35, 62), "A comparable pair must share a chromosome and have consecutive triplet indices.", font=F_SMALL, fill=COLORS["muted"])

    left, top, right, bottom = 115, 120, 760, 590
    x_max = math.ceil(rel["Valid_Triplets"].max() / 1000) * 1000
    y_max = math.ceil(rel["Comparable_Pairs"].max() / 1000) * 1000

    def x(value: float) -> int:
        return int(left + value / x_max * (right - left))

    def y(value: float) -> int:
        return int(top + (y_max - value) / y_max * (bottom - top))

    for tick in range(0, int(x_max) + 1, max(2000, int(x_max // 5))):
        xx = x(tick)
        draw.line((xx, top, xx, bottom), fill=COLORS["grid"], width=1)
        draw.text((xx - 18, bottom + 15), str(tick), font=F_TINY, fill=COLORS["muted"])
    for tick in range(0, int(y_max) + 1, max(2000, int(y_max // 5))):
        yy = y(tick)
        draw.line((left, yy, right, yy), fill=COLORS["grid"], width=1)
        draw.text((50, yy - 8), str(tick), font=F_TINY, fill=COLORS["muted"])
    draw_axes(draw, left, top, right, bottom)
    draw.line((x(0), y(0), x(min(x_max, y_max)), y(min(x_max, y_max))), fill=(156, 163, 175), width=2)

    for _, row in rel.iterrows():
        px = x(float(row["Valid_Triplets"]))
        py = y(float(row["Comparable_Pairs"]))
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=COLORS["BIC"], outline="white")
    draw.text((280, 635), "Valid triplets per dataset", font=F_LABEL, fill=COLORS["dark"])
    draw.text((18, 330), "Comparable", font=F_LABEL, fill=COLORS["dark"])

    panel_x = 835
    draw.text((panel_x, 130), "Aggregate decomposition", font=F_HEAD, fill=COLORS["dark"])
    metrics = {row["Metric"]: row for _, row in aggregate.iterrows()}
    total_within = float(metrics["Max_Pairs_Within_Chromosomes"]["Value"])
    comparable = float(metrics["Comparable_Pairs"]["Value"])
    gap = float(metrics["Gap_Broken_Adjacent_Valid_Pairs"]["Value"])
    boundary = float(metrics["Chromosome_Boundary_Breaks"]["Value"])
    total_one_run = float(metrics["Max_Pairs_If_All_Valid_Triplets_One_Run"]["Value"])

    bar_left, bar_top, bar_right, bar_h = panel_x, 205, 1195, 38
    segments = [
        ("Comparable", comparable, COLORS["BDeu"]),
        ("Index gap", gap, COLORS["gap"]),
        ("Chr boundary", boundary, COLORS["boundary"]),
    ]
    cur = bar_left
    for label, value, color in segments:
        segment_w = int(value / total_one_run * (bar_right - bar_left)) if total_one_run else 0
        draw.rectangle((cur, bar_top, cur + segment_w, bar_top + bar_h), fill=color)
        cur += segment_w
    draw.rectangle((bar_left, bar_top, bar_right, bar_top + bar_h), outline=COLORS["dark"], width=1)

    rows = [
        ("Comparable pairs", comparable, comparable / total_one_run * 100 if total_one_run else 0),
        ("Index-gap broken", gap, gap / total_one_run * 100 if total_one_run else 0),
        ("Chromosome boundaries", boundary, boundary / total_one_run * 100 if total_one_run else 0),
        ("Within-chromosome max", total_within, total_within / total_one_run * 100 if total_one_run else 0),
    ]
    y0 = 285
    for idx, (label, value, pct) in enumerate(rows):
        yy = y0 + idx * 48
        draw.text((panel_x, yy), label, font=F_LABEL, fill=COLORS["dark"])
        draw.text((1080, yy), f"{int(value):,}", font=F_LABEL, fill=COLORS["dark"])
        draw.text((1080, yy + 22), f"{pct:.2f}%", font=F_TINY, fill=COLORS["muted"])

    draw.text((panel_x, 525), "Comparable pairs are consecutive adjacent", font=F_SMALL, fill=COLORS["muted"])
    draw.text((panel_x, 548), "valid-triplet pairs. Chromosome boundaries", font=F_SMALL, fill=COLORS["muted"])
    draw.text((panel_x, 571), "and triplet-index gaps reduce availability.", font=F_SMALL, fill=COLORS["muted"])
    img.save(CHART_DIR / "valid_triplets_vs_comparable_pairs.png")


def write_markdown(
    weighted: pd.DataFrame,
    sample_wide: pd.DataFrame,
    relationship_aggregate: pd.DataFrame,
) -> None:
    weighted_lines = []
    for _, row in weighted.iterrows():
        weighted_lines.append(
            f"- {row['Method']}: {row['Weighted_Compatibility_Percent']:.4f}% "
            f"({int(row['Total_Compatible_Pairs']):,}/{int(row['Total_Comparable_Pairs']):,})"
        )

    mean_rows = sample_wide[sample_wide["Measure"] == "Mean_Total_Count"]
    sample_lines = []
    for _, row in mean_rows.iterrows():
        highest = max(S_CLASSES, key=lambda s: float(row[s]))
        lowest = min(S_CLASSES, key=lambda s: float(row[s]))
        sample_lines.append(
            f"- {row['Method']}: highest average sample size {highest}={row[highest]:.2f}; "
            f"lowest {lowest}={row[lowest]:.2f}."
        )

    metrics = {row["Metric"]: row for _, row in relationship_aggregate.iterrows()}
    text = f"""# Board follow-up tasks 1-3

## 1. Weighted average compatibility

Formula used:

`sum_j compatible_pairs_j / sum_j comparable_pairs_j * 100`

This is equivalent to the whiteboard formula `sum_j n_j * (% compatible in dataset j) / sum_j n_j`, where `n_j` is the number of comparable pairs in dataset `j`.

{chr(10).join(weighted_lines)}

## 2. Average triplet sample size by S class

For one triplet, sample size is `Total_Count = n_000 + ... + n_111`. For each scoring method, triplets are grouped by the S class assigned by that method, and the mean `Total_Count` is reported.

{chr(10).join(sample_lines)}

## 3. Relationship between valid triplets and comparable pairs

Valid triplets are rows that passed the variation/count filter and therefore can receive an S class. Comparable pairs are stricter: two valid triplets must be on the same chromosome and have consecutive triplet indices.

Across all datasets:

- Valid triplets: {int(metrics['Valid_Triplets']['Value']):,}
- Comparable pairs: {int(metrics['Comparable_Pairs']['Value']):,}
- Index-gap broken adjacent valid pairs: {int(metrics['Gap_Broken_Adjacent_Valid_Pairs']['Value']):,}
- Chromosome-boundary breaks: {int(metrics['Chromosome_Boundary_Breaks']['Value']):,}
- Within-chromosome theoretical max: {int(metrics['Max_Pairs_Within_Chromosomes']['Value']):,}

So the relationship is:

`valid triplets -> adjacent valid triplets within chromosomes -> consecutive comparable pairs`

The main loss happens when valid triplets are not consecutive in their triplet index.
"""
    (OUT_DIR / "followup_analysis_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    summary = pd.read_csv(SUMMARY_PATH)
    class_files = sorted(CLASS_DIR.glob("*_classes.csv"))
    if not class_files:
        raise FileNotFoundError(f"No class files found in {CLASS_DIR}")

    weighted, weighted_dataset_rows = weighted_compatibility(summary)
    sample_wide, sample_detail = sample_size_by_s_class(class_files)
    relationship, relationship_aggregate = comparable_pair_relationship(summary, class_files)

    save_dataframe(weighted, "weighted_average_compatibility")
    write_csv(OUT_DIR / "weighted_average_compatibility_by_dataset.csv", weighted_dataset_rows)
    save_dataframe(sample_wide, "mean_sample_size_by_s_class_wide")
    save_dataframe(sample_detail, "sample_size_by_s_class_detail")
    save_dataframe(relationship, "valid_triplets_vs_comparable_pairs")
    save_dataframe(relationship_aggregate, "valid_comparable_relationship_aggregate")

    draw_weighted_chart(weighted)
    draw_sample_size_chart(sample_wide)
    draw_relationship_chart(relationship, relationship_aggregate)
    write_markdown(weighted, sample_wide, relationship_aggregate)

    workbook_data = {
        "weighted_average_compatibility": weighted.to_dict(orient="records"),
        "mean_sample_size_by_s_class_wide": sample_wide.to_dict(orient="records"),
        "sample_size_by_s_class_detail": sample_detail.to_dict(orient="records"),
        "valid_triplets_vs_comparable_pairs": relationship.to_dict(orient="records"),
        "valid_comparable_relationship_aggregate": relationship_aggregate.to_dict(orient="records"),
        "weighted_average_compatibility_by_dataset": weighted_dataset_rows,
        "charts": [
            {
                "Chart": "weighted_compatibility.png",
                "Meaning_CN": "三种 scoring method 按 comparable pair 数加权后的总体 compatibility。",
                "Path": str(CHART_DIR / "weighted_compatibility.png"),
            },
            {
                "Chart": "mean_sample_size_by_s_class.png",
                "Meaning_CN": "BIC/AIC/BDeu 各自分到 S1-S11 的 triplets 的平均 Total_Count。",
                "Path": str(CHART_DIR / "mean_sample_size_by_s_class.png"),
            },
            {
                "Chart": "valid_triplets_vs_comparable_pairs.png",
                "Meaning_CN": "valid triplets 到 comparable pairs 的关系，以及 index gap / chromosome boundary 造成的减少。",
                "Path": str(CHART_DIR / "valid_triplets_vs_comparable_pairs.png"),
            },
        ],
    }
    (OUT_DIR / "workbook_data.json").write_text(
        json.dumps(clean_json_value(workbook_data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "output_dir": str(OUT_DIR),
        "charts": [
            str(CHART_DIR / "weighted_compatibility.png"),
            str(CHART_DIR / "mean_sample_size_by_s_class.png"),
            str(CHART_DIR / "valid_triplets_vs_comparable_pairs.png"),
        ],
        "csvs": [
            str(OUT_DIR / "weighted_average_compatibility.csv"),
            str(OUT_DIR / "mean_sample_size_by_s_class_wide.csv"),
            str(OUT_DIR / "sample_size_by_s_class_detail.csv"),
            str(OUT_DIR / "valid_triplets_vs_comparable_pairs.csv"),
            str(OUT_DIR / "valid_comparable_relationship_aggregate.csv"),
            str(OUT_DIR / "weighted_average_compatibility_by_dataset.csv"),
        ],
        "markdown": str(OUT_DIR / "followup_analysis_summary.md"),
        "workbook_data": str(OUT_DIR / "workbook_data.json"),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
