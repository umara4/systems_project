"""XGBoost extensions on Hospital A (validation only; test_B is not scored here).

Four variants, each tuned with the same grid and procedure:
  xgb_baseline : 10 biomarkers with *_measured flags (should reproduce the earlier XGBoost run)
  xgb_no_flags : same features without the *_measured flags
  xgb_extended : 33 biomarkers (uses all vitals and labs in the dataset) with flags -- EtCO2 was removed becasue it contained all NaN in train_A
  xgb_extended_demo : 33 biomarkers with flags + demographic info (age and gender)
  xgb_extended_no_flags: 33 biomarkers without flags

  Lists of VITALS, LABS, and DEMOGRAPHICS are found in preprocessing_split_data.py

Run:  python src/xgb_extensions.py
"""
import joblib
import pandas as pd
import pyarrow.parquet as pq
from train_tree_models import OUTPUT_DIR, make_xgb, tune_and_fit
from window_eval import paired_bootstrap, window_target
from preprocessing_split_data import (
    DATA_DIR, DEMOGRAPHICS, EXT_LABS, EXT_VITALS, EXTRA_LABS, EXTRA_VITALS, build_datasets,
)


def main():
    available = set(pq.read_schema(DATA_DIR / "train_A.parquet").names)
    missing = [c for c in EXTRA_VITALS + EXTRA_LABS + DEMOGRAPHICS if c not in available]
    assert not missing, f"Not in the parquet files (check spelling): {missing}"

    # Baseline and no-flags share one preprocessing run
    _, _, final, base_cols, _ = build_datasets()
    train, val = final["train_A"], final["validation_A"]
    keep_val, y_val = window_target(val)

    y = train["SepsisLabel"]
    spw = round(float((y == 0).sum() / (y == 1).sum()), 1)
    grid = {"max_depth": [3, 4], "n_estimators": [300, 500], "scale_pos_weight": [1.0, spw]}

    variants = {
        "xgb_baseline": (train, val, base_cols),
        "xgb_no_flags": (train, val, [c for c in base_cols if not c.endswith("_measured")]),
    }

    # Extended variable set needs its own preprocessing run (same code, longer lists)
    _, _, final_x, ext_cols, _ = build_datasets(EXT_VITALS, EXT_LABS, static=DEMOGRAPHICS)
    val_x = final_x["validation_A"]
    assert val_x[["patient_id", "ICULOS"]].equals(val[["patient_id", "ICULOS"]]), "row order differs"
    # Static columns must be constant within a patient
    assert (final_x["train_A"].groupby("patient_id")[DEMOGRAPHICS].nunique() <= 1).all().all()

    labs_only = [c for c in ext_cols if c not in DEMOGRAPHICS]
    variants["xgb_extended"] = (final_x["train_A"], val_x, labs_only)
    variants["xgb_extended_demo"] = (final_x["train_A"], val_x, ext_cols)
    variants["xgb_extended_no_flags"] = (
    final_x["train_A"], val_x, [c for c in labs_only if not c.endswith("_measured")]
)

    results, probas = {}, {}
    for name, (tr, va, cols) in variants.items():
        print(f"\n=== {name}: {len(cols)} features ===")
        metrics, params = tune_and_fit(name, make_xgb, grid, cols, tr, va, keep_val, y_val)
        results[name] = {"n_features": len(cols), **metrics, **{f"best_{k}": v for k, v in params.items()}}
        art = joblib.load(OUTPUT_DIR / name / "model.joblib")
        probas[name] = art["model"].predict_proba(va[cols])[:, 1]

    boot = paired_bootstrap(keep_val, y_val, val["patient_id"].to_numpy(), probas, reference="xgb_baseline")

    demo_boot = paired_bootstrap(
        keep_val, y_val, val["patient_id"].to_numpy(),
        {k: probas[k] for k in ["xgb_extended", "xgb_extended_demo", "xgb_extended_no_flags"]},
        reference="xgb_extended",
    )
    print("\nComparisons vs xgb_extended:")
    print(demo_boot.round(4).to_string())
    demo_boot.to_csv(OUTPUT_DIR / "xgb_demographics_bootstrap.csv")

    summary = pd.DataFrame(results).T
    print("\nValidation results (Hospital A, 20% of patients):")
    print(summary.to_string())
    print("\nWindow PR-AUC with patient-level bootstrap 95% intervals (differences vs xgb_baseline):")
    print(boot.round(4).to_string())

    summary.to_csv(OUTPUT_DIR / "xgb_extensions_validation.csv")
    boot.to_csv(OUTPUT_DIR / "xgb_extensions_bootstrap.csv")


if __name__ == "__main__":
    main()