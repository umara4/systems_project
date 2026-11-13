from pathlib import Path
import pandas as pd

# Locate the two datasets
data_dir = Path("data/parquet_raw")

files = {
    "Hospital A": data_dir / "training_setA.parquet",
    "Hospital B": data_dir / "training_setB.parquet",
}

for hospital, file_path in files.items():

    print("\n" + "=" * 50)
    print(hospital)
    print("=" * 50)

    # 1. Load the dataset
    df = pd.read_parquet(file_path)

    # 2. Dataset size
    print("\n[1] Dataset shape:")
    print(df.shape)
    print("Number of rows:", len(df))
    print("Number of columns:", len(df.columns))

    # 3. Column names and data types
    print("\n[2] Columns and data types:")
    print(df.dtypes)

    # 4. Check patient identifiers
    print("\n[3] Patient identifiers:")
    if "patient_id" in df.columns:
        print("Unique patients:", df["patient_id"].nunique())
    else:
        print("WARNING: patient_id column not found")

    # 5. Missing values
    print("\n[4] Missing values:")
    missing = df.isna().sum()
    missing_pct = df.isna().mean() * 100

    missing_report = pd.DataFrame({
        "missing_count": missing,
        "missing_percentage": missing_pct
    })

    print(missing_report.sort_values(
        "missing_percentage",
        ascending=False
    ))

    # 6. Target label distribution
    print("\n[5] SepsisLabel distribution:")
    if "SepsisLabel" in df.columns:
        print(df["SepsisLabel"].value_counts(dropna=False))
        print("\nLabel percentages:")
        print(df["SepsisLabel"].value_counts(
            normalize=True,
            dropna=False
        ) * 100)

    # 7. Count patients who ever enter the sepsis window
    print("\n[6] Patients with positive sepsis labels:")
    if "patient_id" in df.columns and "SepsisLabel" in df.columns:
        positive_patients = (
            df.groupby("patient_id")["SepsisLabel"]
            .max()
            .eq(1)
            .sum()
        )
        print("Patients with SepsisLabel = 1:", positive_patients)

    # 8. Check duplicate patient-hour records
    print("\n[7] Duplicate patient-hour records:")
    if "patient_id" in df.columns and "ICULOS" in df.columns:
        print(
            df.duplicated(["patient_id", "ICULOS"]).sum()
        )

    # 9. Numerical feature summary
    print("\n[8] Numerical summary:")
    print(df.describe().T)

    # 10. Release memory before loading the next dataset
    del df