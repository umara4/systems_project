"""Logistic regression on Hospital A: raw vs trend features.

- Fit on train_A, tune on validation_A (window PR-AUC), choose the threshold on validation_A (window F2).
- test_B is never used here.
Run:  python src/train_logistic_regression.py
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing_split_data import BIOMARKERS, PROJECT_ROOT, build_datasets
from window_eval import evaluate, tune_threshold, window_pr_auc, window_target

OUTPUT_DIR = PROJECT_ROOT / "challenge" / "outputs" / "logistic_regression2"

PARAM_GRID = {
    "C": [0.001, .005, 0.01, 0.05, 0.1],
    "class_weight": [None, "balanced"],
}


def build_pipeline(params):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="liblinear", max_iter=1000, random_state=42, **params),
    )


def tune_and_fit(name, feature_cols, train, val, keep_val, y_val):
    X_train, y_train = train[feature_cols], train["SepsisLabel"]
    X_val = val[feature_cols]

    grid_rows = []
    best = {"score": -1.0}
    for params in ParameterGrid(PARAM_GRID):
        model = build_pipeline(params).fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        score = window_pr_auc(keep_val, y_val, proba)
        grid_rows.append({**params, "val_window_pr_auc": score})
        if score > best["score"]:
            best = {"score": score, "params": params, "model": model, "proba": proba}

    threshold, _ = tune_threshold(keep_val, y_val, best["proba"])
    metrics = evaluate(keep_val, y_val, best["proba"], threshold)

    pd.DataFrame(grid_rows).sort_values("val_window_pr_auc", ascending=False).to_csv(
        OUTPUT_DIR / f"grid_{name}.csv", index=False
    )
    joblib.dump(
        {
            "model": best["model"],
            "threshold": threshold,
            "feature_cols": feature_cols,
            "best_params": best["params"],
            "validation_metrics": metrics,
        },
        OUTPUT_DIR / f"logreg_{name}.joblib",
    )
    return metrics, best["params"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, _, final, feature_cols, _ = build_datasets()
    train, val = final["train_A"], final["validation_A"]          # test_B is not used here
    keep_val, y_val = window_target(val)                          # once; reused for every configuration

    feature_sets = {
        "raw": BIOMARKERS + [f"{c}_measured" for c in BIOMARKERS],  # forward-filled values + measured flags
        "trend": feature_cols,                                      # raw + lab changes + vital differences/rolling stats
    }

    results = {}
    for name, cols in feature_sets.items():
        print(f"\n--- {name}: {len(cols)} features ---")
        metrics, params = tune_and_fit(name, cols, train, val, keep_val, y_val)
        results[name] = {**metrics, **{f"best_{k}": v for k, v in params.items()}}

    summary = pd.DataFrame(results).T
    print("\nValidation results (Hospital A, 20% of patients):")
    print(summary.to_string())
    summary.to_csv(OUTPUT_DIR / "comparison_validation.csv")
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()