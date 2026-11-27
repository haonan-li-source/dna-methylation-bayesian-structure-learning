# Input data

Bulk ENCODE files are not redistributed in this repository.

The analysis expects one tab-separated file per dataset. Required columns identify the chromosome and triplet position and provide counts for the eight binary methylation states:

```text
n_000 n_001 n_010 n_011 n_100 n_101 n_110 n_111
```

Dataset accession IDs and metadata should be maintained in a local `ENCODE_IDs.xlsx` file. File paths and command-line options are documented by `scripts/analyze_all_datasets.py --help`.

