

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# 1. Configuration


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "challenge" / "data" / "parquet_split"
OUTPUT_DIR = PROJECT_ROOT / "challenge" / "outputs" / "logistic_regression"

TRAIN_FILE = DATA_DIR / "train_A.parquet"
VALIDATION_FILE = DATA_DIR / "validation_A.parquet"
TEST_FILE = DATA_DIR / "test_B.parquet"

FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP",
    "DBP", "Resp", "WBC", "Lactate", "Creatinine",
]
TARGET = "SepsisLabel"
PATIENT_ID = "patient_id"

RANDOM_STATE = 42
MLFLOW_EXPERIMENT = "Sepsis_Logistic_Regression"


# ---------------------------------------------------------------------
# 2. Load and validate data
# ---------------------------------------------------------------------

def load_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} file not found: {path}\n"
            "Run challenge/src/split_data.py first."
        )

    df = pd.read_parquet(path)

    required = FEATURES + [TARGET, PATIENT_ID]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing columns: {missing}")

    if df[TARGET].isna().any():
        raise ValueError(f"{dataset_name} contains missing target labels.")

    labels = set(df[TARGET].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{dataset_name} contains unexpected target values: {labels}"
        )

    # Ensure features are numeric. Invalid values become missing and are
    # handled by the imputer fitted on training data.
    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    print(
        f"{dataset_name}: rows={len(df):,}, "
        f"patients={df[PATIENT_ID].nunique():,}, "
        f"positive_rate={df[TARGET].mean():.4%}"
    )
    return df


def prepare_xy(df: pd.DataFrame):
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET].astype(int)
    return X, y


# ---------------------------------------------------------------------
# 3. Model pipeline
# ---------------------------------------------------------------------

def build_pipeline(
    C: float,
    class_weight: str | None,
    penalty: str,
) -> Pipeline:
    # liblinear supports both L1 and L2 for binary classification.
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    penalty=penalty,
                    solver="liblinear",
                    class_weight=class_weight,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return model


# ---------------------------------------------------------------------
# 4. Metrics and threshold selection
# ---------------------------------------------------------------------

def calculate_metrics(y_true, probabilities, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "f2": float(fbeta_score(
            y_true, predictions, beta=2, zero_division=0
        )),
        "threshold": float(threshold),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def choose_threshold(y_true, probabilities) -> tuple[float, dict]:
    """
    Select the threshold with the highest F2 on validation data.
    Threshold selection is performed on validation only, never on Hospital B.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_f2 = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = fbeta_score(
            y_true, predictions, beta=2, zero_division=0
        )
        if score > best_f2:
            best_f2 = score
            best_threshold = float(threshold)

    return best_threshold, calculate_metrics(
        y_true, probabilities, best_threshold
    )


# ---------------------------------------------------------------------
# 5. Optuna hyperparameter tuning
# ---------------------------------------------------------------------

def tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials: int):
    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 100.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        class_weight_option = trial.suggest_categorical(
            "class_weight", ["none", "balanced"]
        )
        class_weight = (
            None if class_weight_option == "none" else "balanced"
        )

        pipeline = build_pipeline(C, class_weight, penalty)
        pipeline.fit(X_train, y_train)

        val_probabilities = pipeline.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, val_probabilities)

        # Keep each trial's configuration and validation score in Optuna.
        trial.set_user_attr("validation_pr_auc", float(score))
        return score

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="logistic_regression_pr_auc",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print("\nBest Optuna parameters:")
    print(study.best_params)
    print(f"Best validation PR-AUC: {study.best_value:.6f}")

    return study


# ---------------------------------------------------------------------
# 6. Failure analysis
# ---------------------------------------------------------------------

def create_failure_analysis(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    output_dir: Path,
    dataset_name: str,
):
    """
    Save row-level predictions and separate false-negative / false-positive
    records. Includes patient_id for investigation, but patient_id is never
    used as a model feature.
    """
    result = df[[PATIENT_ID, TARGET]].copy()
    result["probability"] = probabilities
    result["predicted_label"] = (probabilities >= threshold).astype(int)

    result["error_type"] = np.select(
        [
            (result[TARGET] == 1) & (result["predicted_label"] == 0),
            (result[TARGET] == 0) & (result["predicted_label"] == 1),
        ],
        ["False Negative", "False Positive"],
        default="Correct",
    )

    result_path = output_dir / f"{dataset_name}_predictions.csv"
    result.to_csv(result_path, index=False)

    false_negatives = result[result["error_type"] == "False Negative"]
    false_positives = result[result["error_type"] == "False Positive"]

    false_negatives.to_csv(
        output_dir / f"{dataset_name}_false_negatives.csv", index=False
    )
    false_positives.to_csv(
        output_dir / f"{dataset_name}_false_positives.csv", index=False
    )

    # Patient-level error summary: how many patients had at least one FN/FP?
    patient_errors = (
        result.assign(
            is_fn=(result["error_type"] == "False Negative").astype(int),
            is_fp=(result["error_type"] == "False Positive").astype(int),
        )
        .groupby(PATIENT_ID)
        .agg(
            hourly_records=(TARGET, "size"),
            positive_hours=(TARGET, "sum"),
            false_negative_hours=("is_fn", "sum"),
            false_positive_hours=("is_fp", "sum"),
        )
        .reset_index()
    )
    patient_errors.to_csv(
        output_dir / f"{dataset_name}_patient_error_summary.csv",
        index=False,
    )

    # Compare original clinical feature values and missingness across
    # false negatives, false positives, and correct predictions.
    # This is descriptive error analysis, not causal evidence.
    feature_rows = []
    for error_group in ["False Negative", "False Positive", "Correct"]:
        row_mask = result["error_type"] == error_group
        group_features = df.loc[row_mask, FEATURES]
        profile = {
            "error_type": error_group,
            "hourly_records": int(row_mask.sum()),
        }
        for feature in FEATURES:
            profile[f"{feature}_median"] = float(
                group_features[feature].median()
            ) if group_features[feature].notna().any() else np.nan
            profile[f"{feature}_missing_pct"] = float(
                group_features[feature].isna().mean() * 100
            )
        feature_rows.append(profile)

    pd.DataFrame(feature_rows).to_csv(
        output_dir / f"{dataset_name}_feature_error_profile.csv",
        index=False,
    )

    # Confusion matrix figure.
    matrix = confusion_matrix(
        result[TARGET], result["predicted_label"], labels=[0, 1]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix)
    ax.set_title(f"{dataset_name}: Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])

    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{value:,}", ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{dataset_name}_confusion_matrix.png",
        dpi=150,
    )
    plt.close(fig)

    return {
        "false_negatives": int(len(false_negatives)),
        "false_positives": int(len(false_positives)),
        "correct": int((result["error_type"] == "Correct").sum()),
        "patients_with_fn": int(
            (patient_errors["false_negative_hours"] > 0).sum()
        ),
        "patients_with_fp": int(
            (patient_errors["false_positive_hours"] > 0).sum()
        ),
    }


# ---------------------------------------------------------------------
# 7. Main workflow
# ---------------------------------------------------------------------

def main(n_trials: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_FILE, "Hospital A - Training")
    val_df = load_dataset(VALIDATION_FILE, "Hospital A - Validation")
    test_df = load_dataset(TEST_FILE, "Hospital B - Final Test")

    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)
    X_test, y_test = prepare_xy(test_df)

    print("\nTraining Logistic Regression with Optuna...")
    study = tune_hyperparameters(
        X_train, y_train, X_val, y_val, n_trials=n_trials
    )

    best_params = study.best_params
    best_class_weight = (
        None if best_params["class_weight"] == "none" else "balanced"
    )

    # Fit the selected model on training data only.
    best_model = build_pipeline(
        C=best_params["C"],
        class_weight=best_class_weight,
        penalty=best_params["penalty"],
    )
    best_model.fit(X_train, y_train)

    # Select decision threshold using validation only.
    val_probabilities = best_model.predict_proba(X_val)[:, 1]
    best_threshold, val_metrics = choose_threshold(
        y_val, val_probabilities
    )

    # Final evaluation on Hospital B. Do not tune anything using this result.
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(
        y_test, test_probabilities, best_threshold
    )

    # Failure analysis for both validation and final test.
    val_errors = create_failure_analysis(
        val_df, val_probabilities, best_threshold,
        OUTPUT_DIR, "validation_A"
    )
    test_errors = create_failure_analysis(
        test_df, test_probabilities, best_threshold,
        OUTPUT_DIR, "test_B"
    )

    # Save model and experiment results.
    model_path = OUTPUT_DIR / "logistic_regression_model.joblib"
    joblib.dump(best_model, model_path)

    results = {
        "best_params": best_params,
        "selected_threshold": best_threshold,
        "validation_metrics": val_metrics,
        "hospital_b_test_metrics": test_metrics,
        "validation_error_counts": val_errors,
        "hospital_b_error_counts": test_errors,
        "features": FEATURES,
        "target": TARGET,
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # Log the best model run to MLflow.
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="Optuna_Best_Logistic_Regression"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_param("target", TARGET)
        mlflow.log_param("optuna_trials", n_trials)
        mlflow.log_param("C", best_params["C"])
        mlflow.log_param("penalty", best_params["penalty"])
        mlflow.log_param("class_weight", best_params["class_weight"])
        mlflow.log_param("decision_threshold", best_threshold)

        for name, value in val_metrics.items():
            mlflow.log_metric(f"validation_{name}", value)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"hospital_b_test_{name}", value)

        for name, value in val_errors.items():
            mlflow.log_metric(f"validation_{name}", value)
        for name, value in test_errors.items():
            mlflow.log_metric(f"hospital_b_{name}", value)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="logistic_regression_model",
        )
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(model_path))

        for artifact in OUTPUT_DIR.glob("*.csv"):
            mlflow.log_artifact(str(artifact))
        for artifact in OUTPUT_DIR.glob("*.png"):
            mlflow.log_artifact(str(artifact))

    # Print concise report.
    print("\n========== FINAL RESULTS ==========")
    print(f"Selected threshold: {best_threshold:.3f}")

    print("\nHospital A - Validation:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nHospital B - Final Test:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nValidation errors:", val_errors)
    print("Hospital B errors:", test_errors)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("MLflow experiment:", MLFLOW_EXPERIMENT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna hyperparameter trials (default: 30).",
    )
    args = parser.parse_args()
    main(n_trials=args.n_trials)
