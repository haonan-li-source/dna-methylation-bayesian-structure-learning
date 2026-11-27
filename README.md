[English](README.md) | [简体中文](README.zh-CN.md)

# DNA Methylation Bayesian Structure Learning

This ongoing research asks whether local Bayesian-network structures inferred from adjacent CpG triplets remain stable as the genomic window moves by one site. I built the analysis pipeline under the supervision of Prof. Pearson Antony, combining exhaustive three-node DAG scoring, adjacent-window compatibility analysis, and methylation-state transition modeling.

## Results at a glance

| Analysis quantity | Result |
|---|---:|
| ENCODE datasets | 121 |
| Input CpG triplets | 650,152 |
| Triplets retained after variability checks | 634,824 (97.64%) |
| Three-node DAGs scored per triplet | 25 |
| Triplet-structure score combinations | approximately 15.87 million |
| Comparable adjacent-window pairs | 461,271 |
| Pair-weighted agreement, BDeu | 89.35% |
| Pair-weighted agreement, BIC / AIC | 87.65% / 85.26% |
| Equal-weight dataset mean, BDeu | 87.60% |

![Pair-weighted structural agreement across scoring methods](figures/weighted_compatibility.png)

Pair weighting answers how often two comparable windows agree across all observed pairs; the equal-weight dataset mean gives every dataset the same influence. Both are reported because dataset sizes vary substantially.

## What I implemented

- Vectorized ingestion and quality control for ENCODE methylation triplet files.
- Exhaustive scoring of all 25 three-node DAGs with BIC, AIC, and BDeu.
- Alignment of overlapping triplets and comparison of the shared two-node structure.
- Dataset-level and pair-weighted summaries with diagnostics for skipped observations.
- Empirical transition matrices for eight methylation states and a category-level Markov representation.
- Reproducible result tables and publication-ready figures.

The fitted local state process produced P(1 to 1) = 0.8369, P(0 to 0) = 0.3858, and stationary methylated-state probability pi1 = 0.7902. These values describe persistence within this fitted local model; they are not genome-wide causal estimates.

## Repository guide

```text
scripts/       analysis, diagnostics, transition modeling, and visualization
figures/       selected aggregate figures
results/       compact tables used in this README
docs/          method definitions and analysis notes
data/          input schema and data-access instructions
```

## Reproduce the analysis

```bash
python -m pip install -r requirements.txt
python scripts/analyze_all_datasets.py --help
```

The full run expects locally downloaded ENCODE files in the schema described in [`data/README.md`](data/README.md). Bulk source files are not redistributed here.

## My role

Main researcher. I designed and implemented the processing, DAG-scoring, adjacent-structure comparison, transition analysis, diagnostics, and visualization workflow under faculty supervision.

## Project status

The repository contains the reviewed analysis code and compact aggregate results for the current research stage. Large ENCODE inputs and unpublished collaborator materials remain outside the public repository.

