from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Define file paths
# --------------------------------------------------

# Project root: the folder containing pyproject.toml
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "challenge" / "data"

INPUT_DIR = DATA_DIR / "parquet_raw"
OUTPUT_DIR = DATA_DIR / "parquet_split"

HOSPITAL_A_FILE = INPUT_DIR / "training_setA.parquet"
HOSPITAL_B_FILE = INPUT_DIR / "training_setB.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the two hospital datasets
# --------------------------------------------------

print("Loading Hospital A and Hospital B...")

df_a = pd.read_parquet(HOSPITAL_A_FILE)
df_b = pd.read_parquet(HOSPITAL_B_FILE)

print(f"Hospital A: {df_a.shape}")
print(f"Hospital B: {df_b.shape}")


# --------------------------------------------------
# 3. Check required columns
# --------------------------------------------------

required_columns = ["patient_id", "SepsisLabel"]

for name, df in [("Hospital A", df_a), ("Hospital B", df_b)]:
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} is missing required columns: {missing_columns}"
        )

    if df["patient_id"].isna().any():
        raise ValueError(f"{name} contains missing patient IDs.")

    if df["SepsisLabel"].isna().any():
        raise ValueError(f"{name} contains missing SepsisLabel values.")

    if not set(df["SepsisLabel"].unique()).issubset({0, 1}):
        raise ValueError(f"{name} contains unexpected SepsisLabel values.")


# --------------------------------------------------
# 4. Create one row per Hospital A patient
# --------------------------------------------------

# A patient is considered to have developed sepsis
# if at least one of their hourly records has SepsisLabel = 1.

patient_summary = (
    df_a.groupby("patient_id")["SepsisLabel"]
    .max()
    .reset_index()
)

patient_summary = patient_summary.rename(
    columns={"SepsisLabel": "ever_sepsis"}
)

print("\nHospital A patient summary:")
print(patient_summary["ever_sepsis"].value_counts())


# --------------------------------------------------
# 5. Split Hospital A patients: 80% train, 20% validation
# --------------------------------------------------

train_ids, validation_ids = train_test_split(
    patient_summary["patient_id"],
    test_size=0.20,
    random_state=42,
    stratify=patient_summary["ever_sepsis"]
)

train_ids = set(train_ids)
validation_ids = set(validation_ids)

# Make sure no patient appears in both groups.
assert train_ids.isdisjoint(validation_ids)


# --------------------------------------------------
# 6. Assign all hourly records to their patient's split
# --------------------------------------------------

train_a = df_a[df_a["patient_id"].isin(train_ids)].copy()

validation_a = df_a[
    df_a["patient_id"].isin(validation_ids)
].copy()

# Check that no hourly records were lost.
assert len(train_a) + len(validation_a) == len(df_a)

# Check that the patient groups do not overlap.
assert set(train_a["patient_id"]).isdisjoint(
    set(validation_a["patient_id"])
)


# --------------------------------------------------
# 7. Reserve all of Hospital B for final testing
# --------------------------------------------------

test_b = df_b.copy()


# --------------------------------------------------
# 8. Save the split datasets
# --------------------------------------------------

train_a.to_parquet(
    OUTPUT_DIR / "train_A.parquet",
    index=False
)

validation_a.to_parquet(
    OUTPUT_DIR / "validation_A.parquet",
    index=False
)

test_b.to_parquet(
    OUTPUT_DIR / "test_B.parquet",
    index=False
)


# Also save the patient assignments for reproducibility.
patient_assignments = pd.DataFrame({
    "patient_id": list(train_ids) + list(validation_ids),
    "split": (
        ["train"] * len(train_ids)
        + ["validation"] * len(validation_ids)
    )
})

patient_assignments.to_csv(
    OUTPUT_DIR / "patient_split_assignments.csv",
    index=False
)


# --------------------------------------------------
# 9. Print a summary of the results
# --------------------------------------------------

def print_summary(name, df):
    patient_count = df["patient_id"].nunique()

    patients_with_sepsis = (
        df.groupby("patient_id")["SepsisLabel"]
        .max()
        .sum()
    )

    patient_sepsis_rate = (
        patients_with_sepsis / patient_count * 100
    )

    hourly_sepsis_rate = df["SepsisLabel"].mean() * 100

    print(f"\n{name}")
    print("-" * 40)
    print(f"Hourly records: {len(df):,}")
    print(f"Unique patients: {patient_count:,}")
    print(f"Patients ever with sepsis: {int(patients_with_sepsis):,}")
    print(f"Patient sepsis rate: {patient_sepsis_rate:.2f}%")
    print(f"Positive hourly labels: {hourly_sepsis_rate:.2f}%")


print("\n========== SPLIT RESULTS ==========")

print_summary("Hospital A - Training", train_a)
print_summary("Hospital A - Validation", validation_a)
print_summary("Hospital B - Final Test", test_b)

print("\nAll files saved successfully!")
print(f"Output folder: {OUTPUT_DIR}")