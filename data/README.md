# Data access

The analysis uses a classroom-provided corporate-bankruptcy case dataset with 132 firms, a binary outcome column named `D`, and 24 accounting-ratio predictors named `R1` through `R24`.

The original dataset is not redistributed in this repository because its reuse terms were not included with the course files. To reproduce the analysis, obtain an authorized copy and save it here as:

```text
data/bankruptcy.csv
```

Expected schema:

- `D`: `0` for failed/bankrupt firms and `1` for healthy firms
- `R1` through `R24`: accounting-ratio predictors
