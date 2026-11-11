"""Convert the PhysioNet 2019 Sepsis PSV files to raw Parquet, one file per training set.

No values are changed: NaNs stay null, nothing is dropped, filled, or rescaled.
Two provenance columns are added, since the patient ID only exists in the filename:
    patient_id    e.g. "p000001"
    training_set  "A" or "B"

Layout (relative to the challenge/ project root):
    data/raw/training_setA/*.psv
    data/raw/training_setB/*.psv
    data/parquet_raw/training_setA.parquet
    data/parquet_raw/training_setB.parquet

Run from challenge/:  uv run python src/psv_to_parquet.py
"""

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "parquet_raw"
SETS = {"A": "training_setA", "B": "training_setB"}

N_WORKERS = min(16, (os.cpu_count() or 4) * 2)  # file reads are I/O bound, so threads help
PROGRESS_EVERY = 500  # print a progress line every this many files
MAX_RETRIES = 3  # retries for transient open/read errors (e.g. OneDrive or antivirus locks)


def load_one(path: Path, set_label: str):
    """Read one PSV file with a single open.

    Returns (path, dataframe or None, raw_data_rows, error message or None).
    raw_data_rows is counted from the text itself (non-empty lines minus the header),
    independent of pandas, and is used later for validation.
    """
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            with open(path, "r") as fh:
                text = fh.read()
            raw_rows = sum(1 for line in text.splitlines() if line.strip()) - 1
            # All measurement columns are numeric, so float64 keeps dtypes identical across
            # files (integer columns such as Gender and SepsisLabel become 0.0/1.0).
            df = pd.read_csv(io.StringIO(text), sep="|", dtype="float64")
            df.insert(0, "patient_id", path.stem)
            df.insert(1, "training_set", set_label)
            return path, df, raw_rows, None
        except Exception as e:  # noqa: BLE001 - report every failure with the file name
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1 + attempt)
    return path, None, 0, last_err


def convert_set(set_label: str, folder_name: str) -> dict:
    files = sorted((RAW_DIR / folder_name).glob("*.psv"))
    if not files:
        raise FileNotFoundError(f"No .psv files found in {RAW_DIR / folder_name}")

    total = len(files)
    print(f"\n=== Training set {set_label}: {total} files, {N_WORKERS} threads ===", flush=True)

    start = time.time()
    frames, failures, raw_rows_total = [], [], 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        # map() yields results in input order, so the sorted file order is preserved.
        for done, (path, df, raw_rows, err) in enumerate(
            pool.map(lambda p: load_one(p, set_label), files), start=1
        ):
            if err is not None:
                failures.append((path.name, err))
            else:
                frames.append(df)
                raw_rows_total += raw_rows

            if done % PROGRESS_EVERY == 0 or done == total:
                elapsed = time.time() - start
                eta = elapsed / done * (total - done)
                print(
                    f"[{set_label}] {done}/{total} files ({done / total:.1%}) | "
                    f"{elapsed:.0f}s elapsed | ~{eta:.0f}s left",
                    flush=True,
                )

    if failures:
        print(f"\n{len(failures)} file(s) failed in set {set_label}:", flush=True)
        for name, err in failures[:20]:
            print(f"  {name}: {err}", flush=True)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more", flush=True)
        raise RuntimeError(
            f"Set {set_label} not written: {len(failures)} of {total} files could not be read."
        )

    print(f"[{set_label}] concatenating...", flush=True)
    df = pd.concat(frames, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{folder_name}.parquet"
    print(f"[{set_label}] writing {out_path.name}...", flush=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)

    # Validate against the row count written to disk, read from the Parquet metadata.
    parquet_rows = pq.ParquetFile(out_path).metadata.num_rows
    assert raw_rows_total == parquet_rows, (
        f"Row mismatch in set {set_label}: raw={raw_rows_total}, parquet={parquet_rows}"
    )

    return {
        "set": set_label,
        "psv_files": total,
        "unique_patients": df["patient_id"].nunique(),
        "raw_data_rows": raw_rows_total,
        "parquet_rows": parquet_rows,
        "columns": df.shape[1],
        "output": str(out_path.relative_to(ROOT)),
        "null_counts": df.isna().sum(),
        "seconds": time.time() - start,
    }


def main() -> None:
    for set_label, folder_name in SETS.items():
        r = convert_set(set_label, folder_name)
        print(f"\n--- Summary: training set {r['set']} ({r['seconds']:.0f}s) ---")
        print(f"PSV files:        {r['psv_files']}")
        print(f"Unique patients:  {r['unique_patients']}")
        print(f"Raw data rows:    {r['raw_data_rows']}")
        print(f"Parquet rows:     {r['parquet_rows']}  (match: {r['raw_data_rows'] == r['parquet_rows']})")
        print(f"Columns:          {r['columns']}")
        print(f"Written to:       {r['output']}")
        print("Null counts per column:")
        print(r["null_counts"].to_string())


if __name__ == "__main__":
    main()