from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
VIS_DIR = ROOT / "outputs" / "all_dataset_analysis" / "visual_analysis"
PNG_DIR = VIS_DIR / "png_charts"
PNG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = ROOT / "outputs" / "all_dataset_analysis" / "dataset_compatibility_summary.csv"
METHODS = ["BIC", "AIC", "BDeu"]
COLORS = {"BIC": (37, 99, 235), "AIC": (220, 38, 38), "BDeu": (22, 163, 74)}
S_CLASSES = [f"S{i}" for i in range(1, 12)]
S_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (17, 24, 39),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = font(26, True)
F_LABEL = font(18)
F_SMALL = font(14)
F_TINY = font(12)


def pct_col(method: str) -> str:
    return f"{method}_Compatibility_Percent"


def canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), "white")
    return img, ImageDraw.Draw(img)


def draw_axes_box(draw: ImageDraw.ImageDraw, left: int, top: int, right: int, bottom: int) -> None:
    draw.line((left, bottom, right, bottom), fill=(17, 24, 39), width=2)
    draw.line((left, top, left, bottom), fill=(17, 24, 39), width=2)


def method_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        values = df[pct_col(method)].to_numpy()
        rows.append({
            "Method": method,
            "Mean": values.mean(),
            "Median": np.median(values),
            "Std": values.std(ddof=1),
            "Min": values.min(),
            "Q1": np.quantile(values, 0.25),
            "Q3": np.quantile(values, 0.75),
            "Max": values.max(),
        })
    return pd.DataFrame(rows)


def draw_boxplot(df: pd.DataFrame) -> None:
    stats = method_stats(df)
    w, h = 1300, 520
    img, draw = canvas(w, h)
    draw.text((35, 25), "Compatibility percentage distribution by scoring method", font=F_TITLE, fill=(17, 24, 39))
    left, top, right, bottom = 120, 90, 1220, 430
    x_min, x_max = 70, 100

    def x(v: float) -> int:
        return int(left + (v - x_min) / (x_max - x_min) * (right - left))

    for tick in range(70, 101, 5):
        xx = x(tick)
        draw.line((xx, top, xx, bottom), fill=(229, 231, 235), width=1)
        draw.text((xx - 18, bottom + 15), f"{tick}%", font=F_SMALL, fill=(55, 65, 81))
    draw_axes_box(draw, left, top, right, bottom)
    ys = [165, 260, 355]
    for y, (_, row) in zip(ys, stats.iterrows()):
        method = row["Method"]
        color = COLORS[method]
        draw.text((35, y - 12), method, font=F_LABEL, fill=color)
        draw.line((x(row["Min"]), y, x(row["Max"]), y), fill=color, width=4)
        draw.rectangle((x(row["Q1"]), y - 24, x(row["Q3"]), y + 24), outline=color, width=4, fill=(*color,))
        # semi-transparent fill is awkward in RGB, overlay a light rectangle.
        draw.rectangle((x(row["Q1"]) + 2, y - 22, x(row["Q3"]) - 2, y + 22), fill=tuple(int(c * 0.18 + 255 * 0.82) for c in color))
        draw.line((x(row["Median"]), y - 30, x(row["Median"]), y + 30), fill=color, width=5)
        draw.ellipse((x(row["Mean"]) - 7, y - 7, x(row["Mean"]) + 7, y + 7), fill=color)
        draw.text((x(row["Max"]) + 8, y - 10), f"mean {row['Mean']:.1f}%", font=F_SMALL, fill=(55, 65, 81))
    draw.text((35, 475), "Box = Q1-Q3, vertical line = median, dot = mean.", font=F_SMALL, fill=(107, 114, 128))
    img.save(PNG_DIR / "compatibility_boxplot.png")


def draw_ranked_lines(df: pd.DataFrame) -> None:
    ordered = df.sort_values(pct_col("BIC")).reset_index(drop=True)
    w, h = 1450, 620
    img, draw = canvas(w, h)
    draw.text((35, 25), "Datasets ranked by BIC compatibility percentage", font=F_TITLE, fill=(17, 24, 39))
    left, top, right, bottom = 100, 85, 1370, 535
    y_min, y_max = 70, 100

    def x(i: int) -> int:
        return int(left + i / (len(ordered) - 1) * (right - left))

    def y(v: float) -> int:
        return int(top + (y_max - v) / (y_max - y_min) * (bottom - top))

    for tick in range(70, 101, 5):
        yy = y(tick)
        draw.line((left, yy, right, yy), fill=(229, 231, 235), width=1)
        draw.text((35, yy - 9), f"{tick}%", font=F_SMALL, fill=(55, 65, 81))
    draw_axes_box(draw, left, top, right, bottom)
    for method in METHODS:
        pts = [(x(i), y(v)) for i, v in enumerate(ordered[pct_col(method)])]
        draw.line(pts, fill=COLORS[method], width=4, joint="curve")
    lx, ly = 1170, 80
    for i, method in enumerate(METHODS):
        yy = ly + i * 32
        draw.line((lx, yy, lx + 45, yy), fill=COLORS[method], width=5)
        draw.text((lx + 55, yy - 10), method, font=F_LABEL, fill=(17, 24, 39))
    draw.text((left, 570), "Low BIC compatibility", font=F_SMALL, fill=(55, 65, 81))
    draw.text((right - 170, 570), "High BIC compatibility", font=F_SMALL, fill=(55, 65, 81))
    img.save(PNG_DIR / "ranked_dataset_lines.png")


def draw_scatter(df: pd.DataFrame) -> None:
    w, h = 760, 680
    img, draw = canvas(w, h)
    draw.text((35, 25), "BIC vs BDeu compatibility", font=F_TITLE, fill=(17, 24, 39))
    left, top, right, bottom = 105, 85, 700, 585
    x_min, x_max = 75, 98
    y_min, y_max = 75, 98

    def x(v: float) -> int:
        return int(left + (v - x_min) / (x_max - x_min) * (right - left))

    def y(v: float) -> int:
        return int(top + (y_max - v) / (y_max - y_min) * (bottom - top))

    for tick in range(75, 99, 5):
        xx, yy = x(tick), y(tick)
        draw.line((xx, top, xx, bottom), fill=(243, 244, 246), width=1)
        draw.line((left, yy, right, yy), fill=(243, 244, 246), width=1)
        draw.text((xx - 12, bottom + 15), str(tick), font=F_SMALL, fill=(55, 65, 81))
        draw.text((55, yy - 8), str(tick), font=F_SMALL, fill=(55, 65, 81))
    draw_axes_box(draw, left, top, right, bottom)
    draw.line((x(75), y(75), x(98), y(98)), fill=(156, 163, 175), width=2)
    for _, row in df.iterrows():
        xx, yy = x(row[pct_col("BIC")]), y(row[pct_col("BDeu")])
        draw.ellipse((xx - 5, yy - 5, xx + 5, yy + 5), fill=(22, 163, 74), outline=(255, 255, 255))
    draw.text((300, 630), "BIC compatibility %", font=F_LABEL, fill=(17, 24, 39))
    draw.text((18, 300), "BDeu", font=F_LABEL, fill=(17, 24, 39))
    img.save(PNG_DIR / "bic_bdeu_scatter.png")


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        total = sum(df[f"{method}_{s}_Count"].sum() for s in S_CLASSES)
        for s in S_CLASSES:
            count = int(df[f"{method}_{s}_Count"].sum())
            rows.append({"Method": method, "S_Class": s, "Count": count, "Percent": count / total * 100})
    return pd.DataFrame(rows)


def draw_class_distribution(df: pd.DataFrame) -> None:
    dist = class_distribution(df)
    w, h = 1450, 500
    img, draw = canvas(w, h)
    draw.text((35, 25), "Overall S-class distribution by scoring method", font=F_TITLE, fill=(17, 24, 39))
    left, right = 120, 1210
    bar_h = 62
    ys = {"BIC": 110, "AIC": 220, "BDeu": 330}
    for method in METHODS:
        x0, y0 = left, ys[method]
        draw.text((35, y0 + 18), method, font=F_LABEL, fill=(17, 24, 39))
        subset = dist[dist["Method"] == method]
        for _, row in subset.iterrows():
            width = int(row["Percent"] / 100 * (right - left))
            color = S_COLORS[S_CLASSES.index(row["S_Class"])]
            draw.rectangle((x0, y0, x0 + width, y0 + bar_h), fill=color)
            if width > 40:
                draw.text((x0 + width // 2 - 10, y0 + 21), row["S_Class"], font=F_TINY, fill="white")
            x0 += width
    lx, ly = 1260, 82
    for i, s in enumerate(S_CLASSES):
        yy = ly + i * 31
        draw.rectangle((lx, yy, lx + 18, yy + 18), fill=S_COLORS[i])
        draw.text((lx + 28, yy - 1), s, font=F_SMALL, fill=(17, 24, 39))
    img.save(PNG_DIR / "s_class_distribution.png")


def draw_top_bottom(df: pd.DataFrame) -> None:
    col = pct_col("BIC")
    data = pd.concat([
        df.nsmallest(10, col).assign(Group="Bottom"),
        df.nlargest(10, col).assign(Group="Top"),
    ]).reset_index(drop=True)
    w, h = 1400, 760
    img, draw = canvas(w, h)
    draw.text((35, 25), "Top and bottom datasets by BIC compatibility", font=F_TITLE, fill=(17, 24, 39))
    left, top, right, bottom = 210, 80, 1270, 700
    x_min, x_max = 70, 100

    def x(v: float) -> int:
        return int(left + (v - x_min) / (x_max - x_min) * (right - left))

    for tick in range(70, 101, 5):
        xx = x(tick)
        draw.line((xx, top, xx, bottom), fill=(243, 244, 246), width=1)
        draw.text((xx - 18, bottom + 18), f"{tick}%", font=F_SMALL, fill=(55, 65, 81))
    draw_axes_box(draw, left, top, right, bottom)
    bar_h, gap = 22, 8
    for i, (_, row) in enumerate(data.iterrows()):
        y0 = top + 14 + i * (bar_h + gap)
        color = (239, 68, 68) if row["Group"] == "Bottom" else (34, 197, 94)
        draw.text((35, y0 + 4), row["File_ID"], font=F_SMALL, fill=(17, 24, 39))
        draw.rectangle((x(x_min), y0, x(row[col]), y0 + bar_h), fill=color)
        draw.text((x(row[col]) + 8, y0 + 3), f"{row[col]:.1f}%", font=F_SMALL, fill=(55, 65, 81))
    img.save(PNG_DIR / "top_bottom_bic.png")


def main() -> None:
    df = pd.read_csv(SUMMARY)
    draw_boxplot(df)
    draw_ranked_lines(df)
    draw_scatter(df)
    draw_class_distribution(df)
    draw_top_bottom(df)
    print(PNG_DIR)


if __name__ == "__main__":
    main()
