"""Random forest and XGBoost on Hospital A, same features as the logistic regression 'trend' model.

- Fit on train_A, tune on validation_A (window PR-AUC), choose the threshold on validation_A (window F2).
- test_B is never used here.
Run:  python src/train_tree_models.py
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier

from preprocessing_split_data import PROJECT_ROOT, build_datasets
from window_eval import evaluate, tune_threshold, window_pr_auc, window_target

OUTPUT_DIR = PROJECT_ROOT / "challenge" / "outputs"


def make_rf(params):
    return RandomForestClassifier(
        n_estimators=200, max_samples=0.3, n_jobs=-1, random_state=42, **params
    )


def make_xgb(params):
    return XGBClassifier(
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, random_state=42, **params
    )


def tune_and_fit(name, make_model, grid, feature_cols, train, val, keep_val, y_val):
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train = train[feature_cols], train["SepsisLabel"]
    X_val = val[feature_cols]

    grid_rows, best = [], {"score": -1.0}
    for params in ParameterGrid(grid):
        model = make_model(params).fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        score = window_pr_auc(keep_val, y_val, proba)
        print(f"  {params} -> val window PR-AUC {score:.4f}")
        grid_rows.append({**params, "val_window_pr_auc": score})
        if score > best["score"]:
            best = {"score": score, "params": params, "model": model, "proba": proba}

    threshold, _ = tune_threshold(keep_val, y_val, best["proba"])
    metrics = evaluate(keep_val, y_val, best["proba"], threshold)

    pd.DataFrame(grid_rows).sort_values("val_window_pr_auc", ascending=False).to_csv(
        out_dir / "grid.csv", index=False
    )
    joblib.dump(
        {
            "model": best["model"],
            "threshold": threshold,
            "feature_cols": feature_cols,
            "best_params": best["params"],
            "validation_metrics": metrics,
        },
        out_dir / "model.joblib",
    )

    importances = pd.Series(best["model"].feature_importances_, index=feature_cols)
    print("  top 10 feature importances:")
    print(importances.sort_values(ascending=False).head(10).round(4).to_string())
    return metrics, best["params"]


def main():
    _, _, final, feature_cols, _ = build_datasets()
    train, val = final["train_A"], final["validation_A"]          # test_B is not used here
    keep_val, y_val = window_target(val)

    # XGBoost's equivalent of class_weight="balanced"
    y = train["SepsisLabel"]
    spw = round(float((y == 0).sum() / (y == 1).sum()), 1)

    models = {
       "random_forest": (make_rf, {
            "max_depth": [16, 24],
            "min_samples_leaf": [100, 200, 400],
            "class_weight": [None, "balanced_subsample"],
        }),
        "xgboost": (make_xgb, {
            "max_depth": [2, 3, 4],
            "n_estimators": [100, 200, 300],
            "scale_pos_weight": [1.0, spw],
        }),
    }

    # Same features as the logistic regression 'trend' model.
    # If that ablation showed trend features don't help, swap in the raw feature list here instead.
    features = feature_cols

    results = {}
    for name, (make_model, grid) in models.items():
        print(f"\n--- {name}: {len(features)} features, {len(ParameterGrid(grid))} configurations ---")
        metrics, params = tune_and_fit(name, make_model, grid, features, train, val, keep_val, y_val)
        results[name] = {**metrics, **{f"best_{k}": v for k, v in params.items()}}

    summary = pd.DataFrame(results).T
    print("\nValidation results (Hospital A, 20% of patients):")
    print(summary.to_string())
    summary.to_csv(OUTPUT_DIR / "tree_models_validation.csv")


if __name__ == "__main__":
    main()