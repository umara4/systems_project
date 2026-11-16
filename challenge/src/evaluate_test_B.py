"""Final evaluation on Hospital B. Run on test_B ONCE, after all models are frozen.

Loads every saved model, applies its frozen validation threshold, and reports window PR-AUC
(patient-level bootstrap intervals), F2, precision, recall and alerts per patient-day.
Nothing is tuned or selected here.

Smoke test first: set SPLIT = "validation_A". Every model's test_pr_auc must then equal its
stored val_pr_auc (change_vs_val = 0). Then set SPLIT = "test_B" and run once.
Run:  python src/evaluate_test_B.py
"""
import joblib
import pandas as pd

from preprocessing_split_data import DEMOGRAPHICS, EXT_LABS, EXT_VITALS, OUTPUT_DIR, build_datasets
from window_eval import evaluate, paired_bootstrap, window_target

SPLIT = "test_B"          # change to "test_B" for the real run
REFERENCE = "xgb_baseline"

MODELS = {
    "logreg_raw": OUTPUT_DIR / "logistic_regression2" / "logreg_raw.joblib",
    "logreg_trend": OUTPUT_DIR / "logistic_regression2" / "logreg_trend.joblib",
    "random_forest": OUTPUT_DIR / "random_forest" / "model.joblib",
    "xgb_baseline": OUTPUT_DIR / "xgb_baseline" / "model.joblib",
    "xgb_no_flags": OUTPUT_DIR / "xgb_no_flags" / "model.joblib",
    "xgb_extended": OUTPUT_DIR / "xgb_extended" / "model.joblib",
    "xgb_extended_demo": OUTPUT_DIR / "xgb_extended_demo" / "model.joblib",
    "xgb_extended_no_flags": OUTPUT_DIR / "xgb_extended_no_flags" / "model.joblib",
}


def load_frames():
    """Feature tables for the chosen split: default biomarkers, and the extended set."""
    _, _, final, _, _ = build_datasets()
    default = final[SPLIT]
    del final
    _, _, final_x, _, _ = build_datasets(EXT_VITALS, EXT_LABS, static=DEMOGRAPHICS)
    extended = final_x[SPLIT]
    del final_x
    return default, extended


def main():
    missing = [n for n, p in MODELS.items() if not p.exists()]
    assert not missing, f"Missing model files: {missing}"

    default, extended = load_frames()
    assert extended[["patient_id", "ICULOS"]].equals(default[["patient_id", "ICULOS"]]), "row order differs"
    keep, y = window_target(default)

    rows, probas = {}, {}
    for name, path in MODELS.items():
        art = joblib.load(path)
        cols = art["feature_cols"]
        frame = default if set(cols) <= set(default.columns) else extended   # pick the table that has the columns
        proba = art["model"].predict_proba(frame[cols])[:, 1]
        probas[name] = proba
        m = evaluate(keep, y, proba, art["threshold"])
        rows[name] = {
            "n_features": len(cols),
            "val_pr_auc": art["validation_metrics"]["pr_auc"],
            "test_pr_auc": m["pr_auc"],
            "f2": m["f2"],
            "precision": m["precision"],
            "recall": m["recall"],
            "threshold": m["threshold"],
            "alert_hours_per_patient_day": float((proba >= art["threshold"]).mean() * 24),   # hourly rows
        }

    table = pd.DataFrame(rows).T
    table["change_vs_val"] = table["test_pr_auc"] - table["val_pr_auc"]
    boot = paired_bootstrap(keep, y, default["patient_id"].to_numpy(), probas, reference=REFERENCE)
    table = table.join(boot[["ci_low", "ci_high", "diff_vs_ref", "diff_ci_low", "diff_ci_high"]])

    print(f"Split: {SPLIT} | no-skill window PR-AUC (positive rate): {y[keep].mean():.4f}")
    print(table.round(4).to_string())
    table.to_csv(OUTPUT_DIR / f"{SPLIT}_final_results.csv")


if __name__ == "__main__":
    main()