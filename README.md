# Interpretable Bankruptcy Risk Modeling

I built this responsible-AI case study to compare not only which bankruptcy-risk model performed better, but also whether different interpretation methods told a consistent story. I evaluated Decision Tree and Random Forest models alongside their global feature rankings and SHAP explanations.

## The question I asked

Can accounting ratios help identify firms that may need closer financial review, while keeping the model's reasoning understandable to analysts and executives?

I treat the model as a screening aid, not as an automated lending, investment, or advisory decision.

## What I did

- Modeled 132 manufacturing and retail firms using 24 accounting ratios.
- Preserved a held-out test set of 27 firms and selected model settings with five-fold cross-validation on the training set.
- Compared Decision Tree and Random Forest classification performance.
- Compared four global-interpretation methods using feature rankings and Spearman rank correlation.
- Translated the model findings into executive-level recommendations and documented limitations.

## Results

| Model | Test accuracy | Healthy-class F1 |
|---|---:|---:|
| Decision Tree | 0.741 | 0.588 |
| Random Forest | **0.852** | **0.800** |

Random Forest impurity importance, SHAP TreeExplainer, and SHAP KernelExplainer produced highly consistent feature rankings, with correlations of approximately **0.97**. Ratios related to asset, earnings, and working-capital coverage of debt appeared consistently important.

These results are descriptive of this fitted model and dataset. They do not establish that changing a financial ratio will cause bankruptcy risk to change.

## Repository structure

```text
.
├── data/
│   └── README.md
├── notebooks/
│   └── interpretable_bankruptcy_risk.ipynb
├── src/
│   └── model_audit.py
└── requirements.txt
```

## Reproduce the analysis

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Obtain the classroom case dataset and save it as `data/bankruptcy.csv`. Then run the notebook from the `notebooks/` directory, or run the reusable script:

```bash
python src/model_audit.py --data data/bankruptcy.csv --output artifacts
```

Use `--skip-kernel-shap` for a faster run.

## Responsible-use notes

- The dataset contains only 132 firms and the test set contains 27 observations.
- The analysis uses one train/test split; reported metrics are estimates, not stable benchmarks.
- Related financial ratios can share importance.
- SHAP values explain model behavior, not causal effects.
- Human review and additional financial context are required before acting on a risk signal.
