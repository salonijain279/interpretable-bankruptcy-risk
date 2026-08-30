"""Train and compare interpretable bankruptcy-risk classifiers.

The script reproduces the core analysis from the accompanying notebook while
keeping model selection separate from final holdout evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit model performance and global feature importance."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--skip-kernel-shap",
        action="store_true",
        help="Skip the slower model-agnostic SHAP calculation.",
    )
    return parser.parse_args()


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    expected = {"D", *(f"R{i}" for i in range(1, 25))}
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    if data[list(expected)].isna().any().any():
        raise ValueError("The modeling columns contain missing values.")
    return data.drop(columns="D"), data["D"]


def class_one_shap_values(raw_values: object) -> np.ndarray:
    if isinstance(raw_values, list):
        return np.asarray(raw_values[1])
    values = np.asarray(raw_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def class_one_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "healthy_precision": float(precision[0]),
        "healthy_recall": float(recall[0]),
        "healthy_f1": float(f1[0]),
    }


def save_importance_plot(importance: pd.Series, title: str, path: Path) -> None:
    top = importance.head(10).sort_values()
    fig, axis = plt.subplots(figsize=(8, 5))
    top.plot(kind="barh", ax=axis, color="#3E6C91")
    axis.set_title(title)
    axis.set_xlabel("Global importance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    features, target = load_data(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    tree_search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"max_depth": [2, 3, 4, 5, 6, None]},
        cv=5,
        scoring="f1",
    )
    tree_search.fit(x_train, y_train)

    forest_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [100, 300],
            "max_depth": [3, 5, None],
            "min_samples_leaf": [1, 3],
        },
        cv=5,
        scoring="f1",
    )
    forest_search.fit(x_train, y_train)

    tree = tree_search.best_estimator_
    forest = forest_search.best_estimator_

    metrics = {
        "decision_tree": {
            **class_one_metrics(y_test, tree.predict(x_test)),
            "best_parameters": tree_search.best_params_,
            "cross_validated_f1": float(tree_search.best_score_),
        },
        "random_forest": {
            **class_one_metrics(y_test, forest.predict(x_test)),
            "best_parameters": forest_search.best_params_,
            "cross_validated_f1": float(forest_search.best_score_),
        },
        "data": {
            "rows": int(len(features)),
            "features": int(features.shape[1]),
            "test_rows": int(len(x_test)),
        },
    }

    tree_importance = pd.Series(
        tree.feature_importances_, index=features.columns, name="Decision Tree MDI"
    )
    forest_importance = pd.Series(
        forest.feature_importances_, index=features.columns, name="Random Forest MDI"
    )

    tree_explainer = shap.TreeExplainer(forest)
    tree_shap = class_one_shap_values(tree_explainer.shap_values(x_test))
    shap_tree_importance = pd.Series(
        np.abs(tree_shap).mean(axis=0),
        index=features.columns,
        name="SHAP TreeExplainer",
    )

    importance_series = [tree_importance, forest_importance, shap_tree_importance]

    if not args.skip_kernel_shap:
        np.random.seed(RANDOM_STATE)
        background = shap.sample(x_train, 50, random_state=RANDOM_STATE)

        def predict_healthy(values: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(values, columns=features.columns)
            return forest.predict_proba(frame)[:, 1]

        kernel_explainer = shap.KernelExplainer(predict_healthy, background)
        kernel_values = np.asarray(
            kernel_explainer.shap_values(x_test, nsamples=200, silent=True)
        )
        importance_series.append(
            pd.Series(
                np.abs(kernel_values).mean(axis=0),
                index=features.columns,
                name="SHAP KernelExplainer",
            )
        )

    comparison = pd.concat(importance_series, axis=1).fillna(0)
    rankings = comparison.rank(axis=0, ascending=False, method="min")

    correlations = pd.DataFrame(
        index=comparison.columns,
        columns=comparison.columns,
        dtype=float,
    )
    for first in comparison.columns:
        for second in comparison.columns:
            correlations.loc[first, second] = spearmanr(
                comparison[first], comparison[second]
            ).correlation

    with (args.output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    comparison.to_csv(args.output / "feature_importance.csv")
    rankings.to_csv(args.output / "feature_rankings.csv")
    correlations.to_csv(args.output / "rank_correlations.csv")

    save_importance_plot(
        forest_importance.sort_values(ascending=False),
        "Random Forest Global Feature Importance",
        args.output / "random_forest_importance.png",
    )
    save_importance_plot(
        shap_tree_importance.sort_values(ascending=False),
        "SHAP TreeExplainer Global Importance",
        args.output / "shap_tree_importance.png",
    )

    print(json.dumps(metrics, indent=2))
    print("\nRank correlations:\n", correlations.round(3))


if __name__ == "__main__":
    main()
