from __future__ import annotations

import html
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()
INPUT = ROOT / "outputs" / "all_dataset_analysis" / "dataset_compatibility_summary.csv"
OUT_DIR = ROOT / "outputs" / "all_dataset_analysis" / "visual_analysis"
METHODS = ["BIC", "AIC", "BDeu"]
COLORS = {"BIC": "#2563eb", "AIC": "#dc2626", "BDeu": "#16a34a"}
S_CLASSES = [f"S{i}" for i in range(1, 12)]


def pct_col(method: str) -> str:
    return f"{method}_Compatibility_Percent"


def escape(value: object) -> str:
    return html.escape("" if pd.isna(value) else str(value))


def tissue_base(tissue: str) -> str:
    text = str(tissue)
    text = re.sub(r"\s+(male|female)\s+(adult|embryo)\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"\s+(male|female)\s+adult\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"\s+(male|female)\s+embryo\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"\s+(male|female)\s+adult\s*$", "", text)
    return text.strip()


def quantile_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        values = df[pct_col(method)].dropna().to_numpy()
        rows.append(
            {
                "Method": method,
                "Mean": values.mean(),
                "Median": np.median(values),
                "Std": values.std(ddof=1),
                "Min": values.min(),
                "Q1": np.quantile(values, 0.25),
                "Q3": np.quantile(values, 0.75),
                "Max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        total = sum(df[f"{method}_{s}_Count"].sum() for s in S_CLASSES)
        for s in S_CLASSES:
            count = int(df[f"{method}_{s}_Count"].sum())
            rows.append(
                {
                    "Method": method,
                    "S_Class": s,
                    "Count": count,
                    "Percent": count / total * 100 if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def top_bottom(df: pd.DataFrame, method: str = "BIC", n: int = 10) -> pd.DataFrame:
    col = pct_col(method)
    keep = ["File_ID", "Experiment_ID", "Tissue", col, f"{method}_Comparable_Pairs"]
    top = df.nlargest(n, col)[keep].assign(Group=f"Top {n}")
    bottom = df.nsmallest(n, col)[keep].assign(Group=f"Bottom {n}")
    return pd.concat([top, bottom], ignore_index=True)


def svg_boxplot(summary: pd.DataFrame) -> str:
    width, height = 760, 310
    margin = {"l": 70, "r": 30, "t": 35, "b": 55}
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]
    x_min, x_max = 70, 100

    def x(v: float) -> float:
        return margin["l"] + (v - x_min) / (x_max - x_min) * plot_w

    ys = {"BIC": 90, "AIC": 155, "BDeu": 220}
    items = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Compatibility percentage boxplot">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="24" font-size="16" font-weight="700">Compatibility percentage distribution by scoring method</text>',
    ]
    for tick in range(70, 101, 5):
        xx = x(tick)
        items.append(f'<line x1="{xx:.1f}" y1="{margin["t"]}" x2="{xx:.1f}" y2="{height-margin["b"]}" stroke="#e5e7eb"/>')
        items.append(f'<text x="{xx:.1f}" y="{height-25}" text-anchor="middle" font-size="11" fill="#374151">{tick}%</text>')
    items.append(f'<line x1="{margin["l"]}" y1="{height-margin["b"]}" x2="{width-margin["r"]}" y2="{height-margin["b"]}" stroke="#111827"/>')

    for _, row in summary.iterrows():
        method = row["Method"]
        y = ys[method]
        color = COLORS[method]
        items.append(f'<text x="20" y="{y+5}" font-size="13" font-weight="700" fill="{color}">{method}</text>')
        items.append(f'<line x1="{x(row["Min"]):.1f}" y1="{y}" x2="{x(row["Max"]):.1f}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        items.append(f'<rect x="{x(row["Q1"]):.1f}" y="{y-15}" width="{x(row["Q3"])-x(row["Q1"]):.1f}" height="30" fill="{color}" opacity="0.18" stroke="{color}" stroke-width="2"/>')
        items.append(f'<line x1="{x(row["Median"]):.1f}" y1="{y-18}" x2="{x(row["Median"]):.1f}" y2="{y+18}" stroke="{color}" stroke-width="3"/>')
        items.append(f'<circle cx="{x(row["Mean"]):.1f}" cy="{y}" r="4" fill="{color}"/>')
        items.append(f'<text x="{x(row["Max"])+6:.1f}" y="{y+4}" font-size="11" fill="#374151">mean {row["Mean"]:.1f}%</text>')
    items.append('<text x="20" y="292" font-size="11" fill="#6b7280">Box = Q1-Q3, vertical line = median, dot = mean.</text>')
    items.append("</svg>")
    return "\n".join(items)


def svg_ranked_lines(df: pd.DataFrame) -> str:
    ordered = df.sort_values(pct_col("BIC")).reset_index(drop=True)
    width, height = 900, 360
    margin = {"l": 60, "r": 35, "t": 38, "b": 55}
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]
    y_min, y_max = 70, 100

    def x(i: int) -> float:
        return margin["l"] + i / (len(ordered) - 1) * plot_w

    def y(v: float) -> float:
        return margin["t"] + (y_max - v) / (y_max - y_min) * plot_h

    items = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Ranked compatibility percentages">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="24" font-size="16" font-weight="700">Datasets ranked by BIC compatibility percentage</text>',
    ]
    for tick in range(70, 101, 5):
        yy = y(tick)
        items.append(f'<line x1="{margin["l"]}" y1="{yy:.1f}" x2="{width-margin["r"]}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        items.append(f'<text x="52" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#374151">{tick}%</text>')
    for method in METHODS:
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(ordered[pct_col(method)]))
        items.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[method]}" stroke-width="2.2"/>')
    legend_x = width - 190
    for idx, method in enumerate(METHODS):
        yy = 46 + idx * 20
        items.append(f'<line x1="{legend_x}" y1="{yy}" x2="{legend_x+28}" y2="{yy}" stroke="{COLORS[method]}" stroke-width="3"/>')
        items.append(f'<text x="{legend_x+36}" y="{yy+4}" font-size="12">{method}</text>')
    items.append(f'<text x="{margin["l"]}" y="{height-25}" font-size="11" fill="#374151">Low BIC compatibility</text>')
    items.append(f'<text x="{width-margin["r"]}" y="{height-25}" text-anchor="end" font-size="11" fill="#374151">High BIC compatibility</text>')
    items.append("</svg>")
    return "\n".join(items)


def svg_scatter(df: pd.DataFrame) -> str:
    width, height = 460, 420
    margin = {"l": 58, "r": 25, "t": 40, "b": 55}
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]
    x_min, x_max, y_min, y_max = 75, 98, 75, 98

    def x(v: float) -> float:
        return margin["l"] + (v - x_min) / (x_max - x_min) * plot_w

    def y(v: float) -> float:
        return margin["t"] + (y_max - v) / (y_max - y_min) * plot_h

    items = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="BIC versus BDeu scatter plot">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="24" font-size="16" font-weight="700">BIC vs BDeu compatibility</text>',
    ]
    for tick in range(75, 99, 5):
        xx = x(tick)
        yy = y(tick)
        items.append(f'<line x1="{xx:.1f}" y1="{margin["t"]}" x2="{xx:.1f}" y2="{height-margin["b"]}" stroke="#f3f4f6"/>')
        items.append(f'<line x1="{margin["l"]}" y1="{yy:.1f}" x2="{width-margin["r"]}" y2="{yy:.1f}" stroke="#f3f4f6"/>')
        items.append(f'<text x="{xx:.1f}" y="{height-30}" text-anchor="middle" font-size="10">{tick}</text>')
        items.append(f'<text x="50" y="{yy+4:.1f}" text-anchor="end" font-size="10">{tick}</text>')
    items.append(f'<line x1="{x(75):.1f}" y1="{y(75):.1f}" x2="{x(98):.1f}" y2="{y(98):.1f}" stroke="#9ca3af" stroke-dasharray="4 4"/>')
    for _, row in df.iterrows():
        items.append(
            f'<circle cx="{x(row[pct_col("BIC")]):.1f}" cy="{y(row[pct_col("BDeu")]):.1f}" r="3.2" fill="#16a34a" opacity="0.62">'
            f'<title>{escape(row["File_ID"])}: BIC {row[pct_col("BIC")]:.1f}%, BDeu {row[pct_col("BDeu")]:.1f}%</title></circle>'
        )
    items.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-size="12">BIC compatibility %</text>')
    items.append(f'<text x="16" y="{height/2}" transform="rotate(-90 16 {height/2})" text-anchor="middle" font-size="12">BDeu compatibility %</text>')
    items.append("</svg>")
    return "\n".join(items)


def svg_class_distribution(class_df: pd.DataFrame) -> str:
    width, height = 900, 330
    margin = {"l": 58, "r": 140, "t": 35, "b": 50}
    plot_w = width - margin["l"] - margin["r"]
    bar_h = 42
    gap = 28
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#111827"
    ]
    color_map = dict(zip(S_CLASSES, colors))
    items = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="S class distribution">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="24" font-size="16" font-weight="700">Overall S-class distribution by scoring method</text>',
    ]
    for m_idx, method in enumerate(METHODS):
        y0 = 75 + m_idx * (bar_h + gap)
        x0 = margin["l"]
        items.append(f'<text x="20" y="{y0+27}" font-size="13" font-weight="700">{method}</text>')
        subset = class_df[class_df["Method"] == method]
        for _, row in subset.iterrows():
            w = row["Percent"] / 100 * plot_w
            if w <= 0:
                continue
            items.append(f'<rect x="{x0:.1f}" y="{y0}" width="{w:.1f}" height="{bar_h}" fill="{color_map[row["S_Class"]]}"/>')
            if w > 28:
                items.append(f'<text x="{x0 + w/2:.1f}" y="{y0+27}" text-anchor="middle" font-size="10" fill="#ffffff">{row["S_Class"]}</text>')
            x0 += w
    legend_x, legend_y = width - 120, 56
    for idx, s in enumerate(S_CLASSES):
        yy = legend_y + idx * 21
        items.append(f'<rect x="{legend_x}" y="{yy-10}" width="12" height="12" fill="{color_map[s]}"/>')
        items.append(f'<text x="{legend_x+18}" y="{yy}" font-size="11">{s}</text>')
    items.append("</svg>")
    return "\n".join(items)


def svg_bar_top_bottom(tb: pd.DataFrame, method: str = "BIC") -> str:
    col = pct_col(method)
    bottom = tb[tb["Group"].str.startswith("Bottom")].sort_values(col)
    top = tb[tb["Group"].str.startswith("Top")].sort_values(col)
    data = pd.concat([bottom, top], ignore_index=True)
    width, height = 900, 520
    margin = {"l": 125, "r": 45, "t": 45, "b": 40}
    plot_w = width - margin["l"] - margin["r"]
    bar_h = 16
    row_gap = 8
    x_min, x_max = 70, 100

    def x(v: float) -> float:
        return margin["l"] + (v - x_min) / (x_max - x_min) * plot_w

    items = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Top and bottom datasets by BIC compatibility">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="20" y="24" font-size="16" font-weight="700">Top and bottom datasets by {method} compatibility</text>',
    ]
    for tick in range(70, 101, 5):
        xx = x(tick)
        items.append(f'<line x1="{xx:.1f}" y1="{margin["t"]}" x2="{xx:.1f}" y2="{height-margin["b"]}" stroke="#f3f4f6"/>')
        items.append(f'<text x="{xx:.1f}" y="{height-15}" text-anchor="middle" font-size="10">{tick}%</text>')
    for idx, row in data.iterrows():
        y = margin["t"] + idx * (bar_h + row_gap)
        value = row[col]
        color = "#ef4444" if row["Group"].startswith("Bottom") else "#22c55e"
        items.append(f'<text x="{margin["l"]-8}" y="{y+12}" text-anchor="end" font-size="10">{escape(row["File_ID"])}</text>')
        items.append(f'<rect x="{x(x_min):.1f}" y="{y}" width="{x(value)-x(x_min):.1f}" height="{bar_h}" fill="{color}" opacity="0.82"/>')
        items.append(f'<text x="{x(value)+5:.1f}" y="{y+12}" font-size="10">{value:.1f}%</text>')
    items.append("</svg>")
    return "\n".join(items)


def html_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    show = df if max_rows is None else df.head(max_rows)
    rows = ["<table>", "<thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in columns) + "</tr></thead>", "<tbody>"]
    for _, row in show.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"<td>{value:.2f}</td>")
            else:
                cells.append(f"<td>{escape(value)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    df["Tissue_Base"] = df["Tissue"].map(tissue_base)

    method_summary = quantile_summary(df)
    class_df = class_distribution(df)
    tb_bic = top_bottom(df, "BIC", 10)

    tissue_summary = (
        df.groupby("Tissue_Base")
        .agg(
            Dataset_Count=("File_ID", "count"),
            BIC_Mean=("BIC_Compatibility_Percent", "mean"),
            AIC_Mean=("AIC_Compatibility_Percent", "mean"),
            BDeu_Mean=("BDeu_Compatibility_Percent", "mean"),
            Mean_Comparable_Pairs=("BIC_Comparable_Pairs", "mean"),
        )
        .reset_index()
        .sort_values(["Dataset_Count", "BIC_Mean"], ascending=[False, False])
    )

    method_summary.to_csv(OUT_DIR / "method_summary_statistics.csv", index=False)
    class_df.to_csv(OUT_DIR / "s_class_distribution_overall.csv", index=False)
    tb_bic.to_csv(OUT_DIR / "top_bottom_10_by_BIC.csv", index=False)
    tissue_summary.to_csv(OUT_DIR / "tissue_group_summary.csv", index=False)

    for name, svg in {
        "compatibility_boxplot.svg": svg_boxplot(method_summary),
        "ranked_dataset_lines.svg": svg_ranked_lines(df),
        "bic_bdeu_scatter.svg": svg_scatter(df),
        "s_class_distribution.svg": svg_class_distribution(class_df),
        "top_bottom_bic.svg": svg_bar_top_bottom(tb_bic, "BIC"),
    }.items():
        (OUT_DIR / name).write_text(svg, encoding="utf-8")

    bdeu_minus_bic = (df[pct_col("BDeu")] - df[pct_col("BIC")]).mean()
    aic_minus_bic = (df[pct_col("AIC")] - df[pct_col("BIC")]).mean()
    highest = df.nlargest(3, pct_col("BIC"))[["File_ID", "Tissue", pct_col("BIC")]]
    lowest = df.nsmallest(3, pct_col("BIC"))[["File_ID", "Tissue", pct_col("BIC")]]

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>All Dataset Compatibility Analysis</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; }}
h1, h2 {{ margin-bottom: 0.35rem; }}
.note {{ color: #4b5563; max-width: 980px; line-height: 1.45; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 22px; max-width: 1020px; }}
.card {{ border: 1px solid #e5e7eb; padding: 18px; border-radius: 8px; background: #fff; }}
table {{ border-collapse: collapse; margin: 12px 0 20px; font-size: 13px; }}
th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
th {{ background: #f9fafb; }}
.kpi {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.kpi div {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 16px; min-width: 140px; }}
.kpi strong {{ display: block; font-size: 22px; }}
</style>
</head>
<body>
<h1>All Dataset Compatibility Analysis</h1>
<p class="note">This report summarizes 121 ENCODE methylation datasets. Compatibility is computed for consecutive triplet pairs on the same chromosome under BIC, AIC, and BDeu scoring.</p>

<div class="kpi">
  <div><span>Datasets</span><strong>{len(df)}</strong></div>
  <div><span>BIC mean</span><strong>{method_summary.loc[method_summary.Method == "BIC", "Mean"].iloc[0]:.2f}%</strong></div>
  <div><span>AIC mean</span><strong>{method_summary.loc[method_summary.Method == "AIC", "Mean"].iloc[0]:.2f}%</strong></div>
  <div><span>BDeu mean</span><strong>{method_summary.loc[method_summary.Method == "BDeu", "Mean"].iloc[0]:.2f}%</strong></div>
</div>

<h2>Key Interpretation</h2>
<ul class="note">
  <li>BDeu gives the highest average compatibility ({method_summary.loc[method_summary.Method == "BDeu", "Mean"].iloc[0]:.2f}%), followed by BIC ({method_summary.loc[method_summary.Method == "BIC", "Mean"].iloc[0]:.2f}%) and AIC ({method_summary.loc[method_summary.Method == "AIC", "Mean"].iloc[0]:.2f}%).</li>
  <li>On average, BDeu is {bdeu_minus_bic:+.2f} percentage points above BIC, while AIC is {aic_minus_bic:+.2f} percentage points relative to BIC.</li>
  <li>The compatibility percentages are generally high, suggesting adjacent triplet structures are usually consistent on the shared CpG pair.</li>
</ul>

<div class="grid">
  <div class="card">{svg_boxplot(method_summary)}</div>
  <div class="card">{svg_ranked_lines(df)}</div>
  <div class="card">{svg_scatter(df)}</div>
  <div class="card">{svg_class_distribution(class_df)}</div>
  <div class="card">{svg_bar_top_bottom(tb_bic, "BIC")}</div>
</div>

<h2>Method Summary Statistics</h2>
{html_table(method_summary, ["Method", "Mean", "Median", "Std", "Min", "Q1", "Q3", "Max"])}

<h2>Highest BIC Compatibility</h2>
{html_table(highest, ["File_ID", "Tissue", pct_col("BIC")])}

<h2>Lowest BIC Compatibility</h2>
{html_table(lowest, ["File_ID", "Tissue", pct_col("BIC")])}

<h2>Tissue Group Summary</h2>
<p class="note">Tissue grouping is a light cleanup of the provided tissue labels. Use this as exploratory, not as a final biological grouping.</p>
{html_table(tissue_summary, ["Tissue_Base", "Dataset_Count", "BIC_Mean", "AIC_Mean", "BDeu_Mean", "Mean_Comparable_Pairs"], 25)}

</body>
</html>
"""
    (OUT_DIR / "compatibility_visual_report.html").write_text(html_doc, encoding="utf-8")

    md = f"""# Compatibility Result Analysis

Analyzed datasets: {len(df)}

## Main result

- BIC mean compatibility: {method_summary.loc[method_summary.Method == "BIC", "Mean"].iloc[0]:.2f}%
- AIC mean compatibility: {method_summary.loc[method_summary.Method == "AIC", "Mean"].iloc[0]:.2f}%
- BDeu mean compatibility: {method_summary.loc[method_summary.Method == "BDeu", "Mean"].iloc[0]:.2f}%

Interpretation: BDeu produces the highest average adjacent-triplet compatibility, while AIC is the most variable. Overall compatibility is high for all three methods, supporting the idea that adjacent triplet models are usually coherent on their shared CpG pair.

## Generated visualizations

- compatibility_boxplot.svg
- ranked_dataset_lines.svg
- bic_bdeu_scatter.svg
- s_class_distribution.svg
- top_bottom_bic.svg
- compatibility_visual_report.html
"""
    (OUT_DIR / "analysis_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
