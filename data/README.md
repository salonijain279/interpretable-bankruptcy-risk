# Data access

The analysis requires a corporate-bankruptcy dataset with 132 firms, a binary outcome column named `D`, and 24 accounting-ratio predictors named `R1` through `R24`.

The source dataset is not redistributed because documented reuse terms were unavailable. To reproduce the analysis, obtain an authorized copy and save it here as:

```text
data/bankruptcy.csv
```

Expected schema:

- `D`: `0` for failed/bankrupt firms and `1` for healthy firms
- `R1` through `R24`: accounting-ratio predictors
