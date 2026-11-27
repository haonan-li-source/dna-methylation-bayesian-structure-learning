from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STATE_BITS = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=np.int8,
)

VARIABLES = ("X", "Y", "Z")
VAR_INDEX = {"X": 0, "Y": 1, "Z": 2}
STATE_COLUMNS = [
    "n_000",
    "n_001",
    "n_010",
    "n_011",
    "n_100",
    "n_101",
    "n_110",
    "n_111",
]


@dataclass(frozen=True)
class DagSpec:
    dag_id: int
    edges: tuple[tuple[str, str], ...]
    s_class: str


DAGS: tuple[DagSpec, ...] = (
    DagSpec(1, tuple(), "S1"),
    DagSpec(2, (("X", "Y"),), "S2"),
    DagSpec(3, (("Y", "X"),), "S2"),
    DagSpec(4, (("X", "Z"),), "S3"),
    DagSpec(5, (("Z", "X"),), "S3"),
    DagSpec(6, (("Y", "Z"),), "S4"),
    DagSpec(7, (("Z", "Y"),), "S4"),
    DagSpec(8, (("X", "Y"), ("Y", "Z")), "S5"),
    DagSpec(9, (("Z", "Y"), ("Y", "X")), "S5"),
    DagSpec(10, (("Y", "X"), ("Y", "Z")), "S5"),
    DagSpec(11, (("X", "Y"), ("Z", "Y")), "S8"),
    DagSpec(12, (("Y", "X"), ("X", "Z")), "S6"),
    DagSpec(13, (("Z", "X"), ("X", "Y")), "S6"),
    DagSpec(14, (("X", "Y"), ("X", "Z")), "S6"),
    DagSpec(15, (("Y", "X"), ("Z", "X")), "S9"),
    DagSpec(16, (("X", "Z"), ("Z", "Y")), "S7"),
    DagSpec(17, (("Y", "Z"), ("Z", "X")), "S7"),
    DagSpec(18, (("Z", "X"), ("Z", "Y")), "S7"),
    DagSpec(19, (("X", "Z"), ("Y", "Z")), "S10"),
    DagSpec(20, (("X", "Y"), ("Y", "Z"), ("X", "Z")), "S11"),
    DagSpec(21, (("X", "Z"), ("Z", "Y"), ("X", "Y")), "S11"),
    DagSpec(22, (("Y", "X"), ("X", "Z"), ("Y", "Z")), "S11"),
    DagSpec(23, (("Y", "Z"), ("Z", "X"), ("Y", "X")), "S11"),
    DagSpec(24, (("Z", "X"), ("X", "Y"), ("Z", "Y")), "S11"),
    DagSpec(25, (("Z", "Y"), ("Y", "X"), ("Z", "X")), "S11"),
)

PAIR_RELATIONS = {
    "S1": {"left": "independent", "right": "independent"},
    "S2": {"left": "dependent", "right": "independent"},
    "S3": {"left": "independent", "right": "independent"},
    "S4": {"left": "independent", "right": "dependent"},
    "S5": {"left": "dependent", "right": "dependent"},
    "S6": {"left": "dependent", "right": "dependent"},
    "S7": {"left": "dependent", "right": "dependent"},
    "S8": {"left": "dependent", "right": "dependent"},
    "S9": {"left": "dependent", "right": "independent"},
    "S10": {"left": "independent", "right": "dependent"},
    "S11": {"left": "dependent", "right": "dependent"},
}

SCORE_COLUMNS = ("Class_BIC", "Class_AIC", "Class_BDeu")


def parent_map(edges: Iterable[tuple[str, str]]) -> dict[str, tuple[str, ...]]:
    parents: dict[str, list[str]] = {v: [] for v in VARIABLES}
    for source, target in edges:
        parents[target].append(source)
    return {node: tuple(items) for node, items in parents.items()}


PARENT_MAPS = [parent_map(dag.edges) for dag in DAGS]


def masks_for_family(child: str, parents: tuple[str, ...]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    child_idx = VAR_INDEX[child]
    parent_indices = [VAR_INDEX[p] for p in parents]
    q = 2 ** len(parent_indices)
    masks = []
    for parent_state_id in range(q):
        parent_values = [
            (parent_state_id >> (len(parent_indices) - 1 - pos)) & 1
            for pos in range(len(parent_indices))
        ]
        parent_mask = np.ones(8, dtype=bool)
        for idx, value in zip(parent_indices, parent_values):
            parent_mask &= STATE_BITS[:, idx] == value
        child0 = parent_mask & (STATE_BITS[:, child_idx] == 0)
        child1 = parent_mask & (STATE_BITS[:, child_idx] == 1)
        masks.append((parent_mask, child0, child1))
    return masks


FAMILY_MASKS = {
    (child, parents): masks_for_family(child, parents)
    for parents_map in PARENT_MAPS
    for child, parents in parents_map.items()
}


def local_log_likelihood(counts: np.ndarray, child: str, parents: tuple[str, ...]) -> np.ndarray:
    score = np.zeros(counts.shape[0], dtype=np.float64)
    for _, child0_mask, child1_mask in FAMILY_MASKS[(child, parents)]:
        n0 = counts[:, child0_mask].sum(axis=1)
        n1 = counts[:, child1_mask].sum(axis=1)
        nij = n0 + n1
        for nk in (n0, n1):
            valid = (nk > 0) & (nij > 0)
            score[valid] += nk[valid] * np.log(nk[valid] / nij[valid])
    return score


def build_lgamma_delta(max_n: int, alpha_values: Iterable[float]) -> dict[float, np.ndarray]:
    lookup: dict[float, np.ndarray] = {}
    ns = np.arange(max_n + 1)
    for alpha in sorted(set(alpha_values)):
        base = math.lgamma(alpha)
        lookup[alpha] = np.array([math.lgamma(alpha + int(n)) - base for n in ns], dtype=np.float64)
    return lookup


def local_bdeu(
    counts: np.ndarray,
    child: str,
    parents: tuple[str, ...],
    lookup: dict[float, np.ndarray],
    equivalent_sample_size: float,
) -> np.ndarray:
    q = 2 ** len(parents)
    alpha_ij = equivalent_sample_size / q
    alpha_ijk = equivalent_sample_size / (q * 2)
    score = np.zeros(counts.shape[0], dtype=np.float64)
    delta_ij = lookup[alpha_ij]
    delta_ijk = lookup[alpha_ijk]
    for _, child0_mask, child1_mask in FAMILY_MASKS[(child, parents)]:
        n0 = counts[:, child0_mask].sum(axis=1).astype(np.int64)
        n1 = counts[:, child1_mask].sum(axis=1).astype(np.int64)
        nij = n0 + n1
        score += -delta_ij[nij] + delta_ijk[n0] + delta_ijk[n1]
    return score


def dag_parameter_count(parents_map: dict[str, tuple[str, ...]]) -> int:
    return sum(2 ** len(parents_map[node]) for node in VARIABLES)


PARAM_COUNTS = np.array([dag_parameter_count(parents_map) for parents_map in PARENT_MAPS], dtype=np.float64)


def score_all_dags(
    counts: np.ndarray,
    equivalent_sample_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = counts.shape[0]
    total_n = counts.sum(axis=1).astype(np.float64)
    max_n = int(total_n.max(initial=0))
    lookup = build_lgamma_delta(max_n, [10.0, 5.0, 2.5, 1.25])

    log_likelihood = np.zeros((n_rows, len(DAGS)), dtype=np.float64)
    bdeu = np.zeros((n_rows, len(DAGS)), dtype=np.float64)

    local_ll_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
    local_bdeu_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}

    for dag_idx, parents_map in enumerate(PARENT_MAPS):
        for child in VARIABLES:
            key = (child, parents_map[child])
            if key not in local_ll_cache:
                local_ll_cache[key] = local_log_likelihood(counts, child, parents_map[child])
                local_bdeu_cache[key] = local_bdeu(
                    counts,
                    child,
                    parents_map[child],
                    lookup,
                    equivalent_sample_size,
                )
            log_likelihood[:, dag_idx] += local_ll_cache[key]
            bdeu[:, dag_idx] += local_bdeu_cache[key]

    # Maximization forms of AIC/BIC, matching the earlier notebook's intent:
    # AIC = log-likelihood - k, BIC = log-likelihood - 0.5*k*log(N).
    with np.errstate(divide="ignore", invalid="ignore"):
        bic = log_likelihood - 0.5 * np.log(total_n)[:, None] * PARAM_COUNTS[None, :]
    aic = log_likelihood - PARAM_COUNTS[None, :]
    return bic, aic, bdeu


def best_classes(score_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best_idx = np.argmax(score_matrix, axis=1)
    classes = np.array([DAGS[i].s_class for i in best_idx], dtype=object)
    dag_ids = np.array([DAGS[i].dag_id for i in best_idx], dtype=np.int16)
    return classes, dag_ids


def valid_triplet_mask(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=1) > 0
    x0 = counts[:, :4].sum(axis=1)
    x1 = counts[:, 4:].sum(axis=1)
    y0 = counts[:, [0, 1, 4, 5]].sum(axis=1)
    y1 = counts[:, [2, 3, 6, 7]].sum(axis=1)
    z0 = counts[:, [0, 2, 4, 6]].sum(axis=1)
    z1 = counts[:, [1, 3, 5, 7]].sum(axis=1)
    varying = (x0 > 0) & (x1 > 0) & (y0 > 0) & (y1 > 0) & (z0 > 0) & (z1 > 0)
    return total & varying


def read_metadata_map(root: Path) -> dict[str, dict[str, str]]:
    path = root / "ENCODE_IDs.xlsx"
    if not path.exists():
        return {}
    try:
        ids = pd.read_excel(path, dtype=str)
    except Exception:
        return {}
    mapping: dict[str, dict[str, str]] = {}
    for _, row in ids.iterrows():
        file_id = str(row.get("File ID (prefixed by “ENCFF”)", "")).strip()
        if not file_id:
            continue
        full_file_id = file_id if file_id.startswith("ENCFF") else f"ENCFF{file_id}"
        mapping[full_file_id] = {
            "Experiment_ID": str(row.get("Experiment ID (prefixed by “ENCSR”)", "")).strip(),
            "Tissue": str(row.get("Tissue (patient age)", "")).strip(),
        }
    return mapping


def analyze_dataset(
    input_path: Path,
    class_dir: Path,
    pair_dir: Path,
    equivalent_sample_size: float,
    max_rows: int | None = None,
) -> dict[str, object]:
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    if max_rows is not None:
        df = df.head(max_rows)

    counts = (
        df[STATE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .round()
        .astype(np.int64)
        .to_numpy()
    )

    valid = valid_triplet_mask(counts)
    valid_df = df.loc[valid].copy()
    valid_counts = counts[valid]

    class_output = class_dir / f"{input_path.stem}_classes.csv"
    pair_output = pair_dir / f"{input_path.stem}_compatibility_pairs.csv"

    if len(valid_df) == 0:
        pd.DataFrame().to_csv(class_output, index=False)
        pd.DataFrame().to_csv(pair_output, index=False)
        return {
            "File_ID": input_path.stem,
            "Input_Triplets": len(df),
            "Valid_Triplets": 0,
            "Skipped_Triplets": len(df),
            "Class_Output": str(class_output.relative_to(class_dir.parent)).replace("\\", "/"),
            "Pair_Output": str(pair_output.relative_to(pair_dir.parent)).replace("\\", "/"),
        }

    bic_scores, aic_scores, bdeu_scores = score_all_dags(valid_counts, equivalent_sample_size)
    classes_bic, dags_bic = best_classes(bic_scores)
    classes_aic, dags_aic = best_classes(aic_scores)
    classes_bdeu, dags_bdeu = best_classes(bdeu_scores)

    result = pd.DataFrame(
        {
            "Source_File": input_path.name,
            "Original_Row_Number": valid_df.index.to_numpy() + 2,
            "Chromosome": pd.to_numeric(valid_df["Chromosome number"], errors="coerce"),
            "Triplet_Index_First_CpG": pd.to_numeric(valid_df["Triplet index (first CpG)"], errors="coerce"),
            "Triplet_Location_Centroid": pd.to_numeric(valid_df["Triplet location (centroid)"], errors="coerce"),
            "Total_Count": valid_counts.sum(axis=1),
            "Class_BIC": classes_bic,
            "DAG_ID_BIC": dags_bic,
            "Class_AIC": classes_aic,
            "DAG_ID_AIC": dags_aic,
            "Class_BDeu": classes_bdeu,
            "DAG_ID_BDeu": dags_bdeu,
        }
    )
    result.to_csv(class_output, index=False)

    pairs = compatibility_pairs(result)
    pairs.to_csv(pair_output, index=False)

    summary: dict[str, object] = {
        "File_ID": input_path.stem,
        "Input_Triplets": len(df),
        "Valid_Triplets": len(result),
        "Skipped_Triplets": len(df) - len(result),
        "Class_Output": str(class_output.relative_to(class_dir.parent)).replace("\\", "/"),
        "Pair_Output": str(pair_output.relative_to(pair_dir.parent)).replace("\\", "/"),
    }
    for score_name, col in (
        ("BIC", "Class_BIC"),
        ("AIC", "Class_AIC"),
        ("BDeu", "Class_BDeu"),
    ):
        compatible_col = f"{score_name}_Compatible"
        comparable = int(pairs[compatible_col].notna().sum()) if compatible_col in pairs else 0
        compatible_count = int(pairs[compatible_col].sum()) if comparable else 0
        incompatible = comparable - compatible_count
        pct = (compatible_count / comparable * 100.0) if comparable else np.nan
        summary[f"{score_name}_Comparable_Pairs"] = comparable
        summary[f"{score_name}_Compatible_Pairs"] = compatible_count
        summary[f"{score_name}_Incompatible_Pairs"] = incompatible
        summary[f"{score_name}_Compatibility_Percent"] = pct

        counts_by_class = result[col].value_counts().to_dict()
        for s_class in [f"S{i}" for i in range(1, 12)]:
            summary[f"{score_name}_{s_class}_Count"] = int(counts_by_class.get(s_class, 0))

    return summary


def is_compatible(left_class: str, right_class: str) -> bool:
    return PAIR_RELATIONS[left_class]["right"] == PAIR_RELATIONS[right_class]["left"]


def compatibility_pairs(result: pd.DataFrame) -> pd.DataFrame:
    ordered = result.sort_values(["Chromosome", "Triplet_Index_First_CpG"]).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    if len(ordered) < 2:
        return pd.DataFrame(rows)

    chrom = ordered["Chromosome"].to_numpy()
    triplet_index = ordered["Triplet_Index_First_CpG"].to_numpy()

    for i in range(len(ordered) - 1):
        if not np.isfinite(chrom[i]) or not np.isfinite(chrom[i + 1]):
            continue
        if chrom[i] != chrom[i + 1]:
            continue
        if not np.isfinite(triplet_index[i]) or not np.isfinite(triplet_index[i + 1]):
            continue
        if int(round(triplet_index[i + 1])) != int(round(triplet_index[i])) + 1:
            continue

        left = ordered.iloc[i]
        right = ordered.iloc[i + 1]
        record: dict[str, object] = {
            "Chromosome": left["Chromosome"],
            "Left_Triplet_Index": int(round(left["Triplet_Index_First_CpG"])),
            "Right_Triplet_Index": int(round(right["Triplet_Index_First_CpG"])),
            "Shared_CpG_Indices": f"{int(round(left['Triplet_Index_First_CpG'])) + 1},{int(round(left['Triplet_Index_First_CpG'])) + 2}",
        }
        for score_name, col in (
            ("BIC", "Class_BIC"),
            ("AIC", "Class_AIC"),
            ("BDeu", "Class_BDeu"),
        ):
            left_class = str(left[col])
            right_class = str(right[col])
            record[f"{score_name}_Left_Class"] = left_class
            record[f"{score_name}_Right_Class"] = right_class
            record[f"{score_name}_Compatible"] = int(is_compatible(left_class, right_class))
        rows.append(record)

    return pd.DataFrame(rows)


def write_readme(output_dir: Path, max_rows: int | None, equivalent_sample_size: float) -> None:
    text = f"""# All dataset DAG and compatibility analysis

Generated files:

- `s_class_results/*_classes.csv`: per-triplet S-class results for BIC, AIC, and BDeu.
- `compatibility_pairs/*_compatibility_pairs.csv`: adjacent-triplet compatibility details.
- `dataset_compatibility_summary.csv`: per-dataset compatibility percentages and S-class counts.

Important analysis choices:

- Uses all rows in each TSV unless `--max-rows` is supplied. Current max_rows: {max_rows}.
- Skips triplets where any of X, Y, or Z has no variation across observed molecule states, matching the earlier notebook's `local_df.nunique().min() < 2` skip rule.
- Adjacent pair comparison only includes consecutive triplets on the same chromosome where `next Triplet index = current Triplet index + 1`.
- BDeu equivalent sample size: {equivalent_sample_size}.
- Scores are computed directly from the eight count columns (`n_000` through `n_111`) over all 25 three-node DAGs.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze all ENCODE triplet TSV datasets.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing ENCFF*.tsv files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/all_dataset_analysis"), help="Output directory.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap per TSV for testing.")
    parser.add_argument("--equivalent-sample-size", type=float, default=10.0, help="BDeu equivalent sample size.")
    parser.add_argument("--files", nargs="*", default=None, help="Optional TSV file names to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    class_dir = output_dir / "s_class_results"
    pair_dir = output_dir / "compatibility_pairs"
    output_dir.mkdir(parents=True, exist_ok=True)
    class_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        tsv_files = [root / name for name in args.files]
    else:
        tsv_files = sorted(root.glob("ENCFF*.tsv"))

    metadata = read_metadata_map(root)
    summaries = []
    for index, tsv_path in enumerate(tsv_files, start=1):
        print(f"[{index}/{len(tsv_files)}] {tsv_path.name}", flush=True)
        summary = analyze_dataset(
            tsv_path,
            class_dir,
            pair_dir,
            equivalent_sample_size=args.equivalent_sample_size,
            max_rows=args.max_rows,
        )
        summary.update(metadata.get(summary["File_ID"], {}))
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    preferred_cols = [
        "File_ID",
        "Experiment_ID",
        "Tissue",
        "Input_Triplets",
        "Valid_Triplets",
        "Skipped_Triplets",
        "BIC_Comparable_Pairs",
        "BIC_Compatible_Pairs",
        "BIC_Incompatible_Pairs",
        "BIC_Compatibility_Percent",
        "AIC_Comparable_Pairs",
        "AIC_Compatible_Pairs",
        "AIC_Incompatible_Pairs",
        "AIC_Compatibility_Percent",
        "BDeu_Comparable_Pairs",
        "BDeu_Compatible_Pairs",
        "BDeu_Incompatible_Pairs",
        "BDeu_Compatibility_Percent",
    ]
    other_cols = [col for col in summary_df.columns if col not in preferred_cols]
    summary_df = summary_df[[col for col in preferred_cols if col in summary_df.columns] + other_cols]
    summary_path = output_dir / "dataset_compatibility_summary.csv"
    summary_df.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    write_readme(output_dir, args.max_rows, args.equivalent_sample_size)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
