
"""Find one false-negative case on Hospital B.
Run: uv run python src/failure_case.py
"""

import joblib
import numpy as np
import pandas as pd

from preprocessing_split_data import OUTPUT_DIR, build_datasets
from window_eval import window_target

MODEL_PATH = OUTPUT_DIR / "xgb_baseline" / "model.joblib"
FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "Lactate", "WBC", "Creatinine", "Platelets",
]
CONTEXT_HOURS = 6


def main():
    # Load test data and frozen model
    _, _, datasets, _, _ = build_datasets()
    test = datasets["test_B"]
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    threshold = artifact["threshold"]
    features = artifact["feature_cols"]

    # Predict and identify false negatives
    proba = model.predict_proba(test[features])[:, 1]
    keep, y = window_target(test)

    keep = np.asarray(keep, dtype=bool)
    y = np.asarray(y).astype(int)
    pred = (proba >= threshold).astype(int)

    fn = np.flatnonzero(keep & (y == 1) & (pred == 0))

    if len(fn) == 0:
        print("No false-negative cases found.")
        return

    # Select FN closest to the threshold
    idx = fn[np.argmax(proba[fn])]
    patient = test.iloc[idx]
    pid, hour = patient["patient_id"], int(patient["ICULOS"])

    # Extract surrounding clinical data
    context = test[
        (test["patient_id"] == pid)
        & test["ICULOS"].between(hour - CONTEXT_HOURS, hour + CONTEXT_HOURS)
    ].copy()

    columns = [
        c for c in [
            "patient_id", "ICULOS", "SepsisLabel",
            "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
            "Lactate", "WBC", "Creatinine", "Platelets",
        ] if c in context.columns
    ]

    context[columns].to_csv(
        OUTPUT_DIR / "failure_case_context.csv", index=False
    )

    # Print case summary
    print("\nFALSE NEGATIVE CASE")
    print(f"Patient: {pid}")
    print(f"ICULOS: {hour}")
    print(f"Actual window target: {y[idx]}")
    print(f"Predicted: {pred[idx]}")
    print(f"Probability: {proba[idx]:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Total false negatives: {len(fn)}")
    print("\nSurrounding clinical data:")
    print(context[columns].to_string(index=False))
    print("\nSaved: outputs/failure_case_context.csv")


if __name__ == "__main__":
    main()
