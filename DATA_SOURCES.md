# Dataset source, attribution, and license

## UCI Iranian Churn

- Provider: UCI Machine Learning Repository, dataset ID 563.
- Dataset page: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
- DOI: https://doi.org/10.24432/C5JW3Z
- Download URL: https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip
- License stated by UCI: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Dataset citation: *Iranian Churn* [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5JW3Z.
- UCI describes 3,150 rows from an Iranian telecommunications company collected over 12 months. It says fields other than churn are aggregates for months 1–9 and the churn label is the state at month 12, leaving a three-month planning gap.
- Dataset contents are downloaded locally by `churn-lab download`; raw files under `data/raw/` are excluded from Git by default. This project does not rehost the source CSV.

## Reproducibility and provenance

For the local copy used for the current benchmark, the UCI archive SHA-256 is `696c3a1812267980751f30d19193a1b430ac4ad76172bb089e664c96813ead66` and the extracted CSV SHA-256 is `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1`. The benchmark writes the raw CSV checksum, sample counts, split description, and random seed to `reports/evaluation_summary.json`.

The project's MIT license covers original code only; the dataset retains its separate CC BY 4.0 license. Credit UCI and review the current UCI terms before using or adapting the data.

## Feature availability and leakage audit

The model excludes the source field `Status` even though it appears in the CSV. UCI's field definition calls it active/non-active, but its timing relative to the month-12 label is not clearly specified. In this copy, churn prevalence is 5.3% for status 1 and 47.3% for status 2. Because the field could encode an outcome proxy, it is withheld rather than treated as a safe predictor. The exact crosstab and rationale are recorded in `reports/leakage_audit.json`.

The source contains exact repeated rows and some identical feature profiles with conflicting labels. The benchmark retains the rows, builds groups from all 12 eligible model-input fields (not the label), and uses `StratifiedGroupKFold` so identical input profiles do not cross train, validation, and test partitions.
