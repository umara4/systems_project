"""Build the clean layer from the raw Parquet files, with every loss counted.

Reads   data/parquet_raw/training_setA.parquet, training_setB.parquet
Writes  data/clean/measured_setA.parquet, measured_setB.parquet
            validated MEASURED values only. NaN means "not measured" (or invalidated).
        data/clean/filled_ffill_setA.parquet, filled_ffill_setB.parquet
            the same table with causal forward fill within each patient. Kept apart, so a
            reader can always tell measured from filled. No median fill here: medians must
            come from the training split only, so that step belongs in the modeling code.
        data/clean/patient_splits.csv
            one row per patient with a stratified 5-fold assignment (patients never split)
        data/raw/SHA256SUMS (only if data/raw holds the original .psv files)
        reports/*.csv and reports/clean_report.txt (counts a reader can reproduce)

Raw values are never edited in data/parquet_raw. All changes happen here and are counted.

Run from challenge/:  uv run python src/clean_data.py
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RAW_PARQUET = ROOT / "data" / "parquet_raw"
RAW_PSV = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
REPORTS = ROOT / "data" / "reports"

DYNAMIC = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
VITALS = DYNAMIC[:7]
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]

# ---- decisions (every one is reported in the output) ---------------------------------
DROP_COLUMNS = ["EtCO2"]  # 100% missing in set A, about 92% in set B
# Inclusive limits. Values outside are set to NaN and counted. These are starting points:
# review them against a clinical reference and edit before reporting.
RANGES = {
    "HR": (20, 300), "O2Sat": (50, 100), "Temp": (25, 45), "SBP": (30, 300),
    "MAP": (20, 250), "DBP": (10, 200), "Resp": (3, 80),
    "FiO2": (0.2, 1.0),          # a fraction; values such as 4000 or -50 are errors
    "pH": (6.5, 8.0), "BaseExcess": (-40, 40), "HCO3": (2, 60),
    "Potassium": (1.0, 10.0),
}
MIN_STAY_ROWS = 0            # 0 = keep every stay. Short stays are over-represented among sepsis patients.
DROP_LEADING_EMPTY = False   # rows before a patient's first measurement; False = flag only
SHARED_MIN_COVERAGE = 0.5    # a variable is "shared" if >= this share of patients in BOTH sets have it
N_FOLDS, SEED = 5, 0

lines: list[str] = []


def out(text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


def snap(log: list, s: str, step: int, rule: str, df: pd.DataFrame) -> None:
    pos = df["SepsisLabel"] == 1
    log.append({
        "set": s, "step": step, "rule": rule,
        "patients": df["patient_id"].nunique(), "rows": len(df),
        "positive_rows": int(pos.sum()), "sepsis_patients": df.loc[pos, "patient_id"].nunique(),
    })


def add_flags(df: pd.DataFrame, dyn: list[str]) -> pd.DataFrame:
    df["n_measured"] = df[dyn].notna().sum(axis=1).astype("int16")
    df["is_empty_row"] = df["n_measured"] == 0
    seen = (df["n_measured"] > 0).astype(int).groupby(df["patient_id"], sort=False).cummax().astype(bool)
    df["is_leading_empty"] = ~seen
    return df


def variant_table(df: pd.DataFrame, s: str, dyn: list[str]) -> pd.DataFrame:
    """What each optional filter WOULD remove from the clean set (nothing here is applied)."""
    y = df["SepsisLabel"] == 1
    pid = df["patient_id"]
    grouped = df.groupby("patient_id", sort=False)
    pat = grouped.agg(n_rows=("ICULOS", "size"), sepsis=("SepsisLabel", "max"))
    stay = grouped["ICULOS"].transform("size")
    pct = grouped["n_measured"].transform("sum") / (stay * len(dyn))
    masks = {
        "rows before a patient's first measurement (leading empty)": df["is_leading_empty"],
        "rows with no measurement that hour (any empty row)": df["is_empty_row"],
        "rows missing any of the 7 vital signs": ~df[VITALS].notna().all(axis=1),
        "rows with any NaN in a retained time-varying column ('drop all NaN rows')": ~df[dyn].notna().all(axis=1),
        "patients with a stay <= 12 rows": stay <= 12,
        "patients with a stay <= 24 rows": stay <= 24,
        "patients with < 5% of cells measured": pct < 0.05,
    }
    mask_table = pd.DataFrame(masks)
    full_by_patient = mask_table.groupby(pid, sort=False).all()
    rows = []
    sep = pat["sepsis"].astype(bool)
    for name, m in mask_table.items():
        full = full_by_patient[name]
        rows.append({
            "set": s, "filter": name,
            "rows_removed": int(m.sum()), "pct_rows": m.mean(),
            "positive_rows_removed": int((m & y).sum()), "pct_positive_rows": (m & y).sum() / max(int(y.sum()), 1),
            "patients_removed": int(full.sum()), "sepsis_patients_removed": int((full & sep).sum()),
            "pct_sepsis_patients": (full & sep).sum() / max(int(sep.sum()), 1),
        })
    return pd.DataFrame(rows)


def clean_set(s: str):
    path = RAW_PARQUET / f"training_set{s}.parquet"
    out(f"\n--- set {s}: reading {path.relative_to(ROOT)} ---")
    df = pd.read_parquet(path)
    log: list = []
    removed: list = []
    snap(log, s, 0, "arrived (raw Parquet)", df)

    # 1. structure
    step = (df.groupby("patient_id", sort=False)["ICULOS"].diff().dropna())
    assert (step > 0).all(), "rows are not in strictly increasing ICULOS order within patients"
    df = df.drop_duplicates(["patient_id", "ICULOS"]).reset_index(drop=True)
    snap(log, s, 1, "duplicate (patient_id, ICULOS) rows removed", df)

    # 2. columns
    col_info = {c: float(df[c].isna().mean()) for c in DROP_COLUMNS}
    df = df.drop(columns=DROP_COLUMNS)
    dyn = [c for c in DYNAMIC if c not in DROP_COLUMNS]
    snap(log, s, 2, f"column(s) dropped: {', '.join(DROP_COLUMNS)} (cells, rows kept)", df)

    # 3. implausible values -> NaN
    inv = []
    for c, (lo, hi) in RANGES.items():
        x = df[c]
        below, above = x < lo, x > hi
        inv.append({"set": s, "column": c, "low": lo, "high": hi, "measured_before": int(x.notna().sum()),
                    "set_to_nan_below": int(below.sum()), "set_to_nan_above": int(above.sum())})
        df.loc[below | above, c] = np.nan
    inv = pd.DataFrame(inv)
    inv["pct_of_measured"] = (inv["set_to_nan_below"] + inv["set_to_nan_above"]) / inv["measured_before"].clip(lower=1)
    snap(log, s, 3, "implausible values set to NaN (cells, rows kept)", df)

    df = add_flags(df, dyn)

    # 4. patients with nothing measured in the whole stay
    per_pat = df.groupby("patient_id", sort=False).agg(rows=("ICULOS", "size"), meas=("n_measured", "sum"),
                                                        pos=("SepsisLabel", "sum"))
    zero = per_pat.index[per_pat["meas"] == 0]
    for pid_, r in per_pat.loc[zero].iterrows():
        removed.append({"patient_id": pid_, "training_set": s, "reason": "no measured value in the whole stay",
                        "rows": int(r["rows"]), "positive_rows": int(r["pos"])})
    df = df[~df["patient_id"].isin(zero)]
    snap(log, s, 4, "patients with no measured value in the whole stay removed", df)

    # 5. optional: short stays
    if MIN_STAY_ROWS > 0:
        stay = df.groupby("patient_id", sort=False)["ICULOS"].transform("size")
        short = df.loc[stay < MIN_STAY_ROWS]
        for pid_, g in short.groupby("patient_id", sort=False):
            removed.append({"patient_id": pid_, "training_set": s, "reason": f"stay < {MIN_STAY_ROWS} rows",
                            "rows": len(g), "positive_rows": int((g["SepsisLabel"] == 1).sum())})
        df = df.loc[stay >= MIN_STAY_ROWS]
        snap(log, s, 5, f"stays shorter than {MIN_STAY_ROWS} rows removed", df)
    else:
        snap(log, s, 5, "short-stay filter (not applied)", df)

    # 6. optional: leading empty rows
    if DROP_LEADING_EMPTY:
        df = df.loc[~df["is_leading_empty"]]
        snap(log, s, 6, "leading empty rows removed", df)
    else:
        snap(log, s, 6, "leading-empty-row filter (not applied, rows flagged)", df)
    df = df.reset_index(drop=True)

    variants = variant_table(df, s, dyn)

    # coverage (share of patients with >= 1 validated measurement)
    coverage = df[dyn].notna().groupby(df["patient_id"], sort=False).any().mean()

    # forward fill (causal: only earlier hours of the same patient)
    filled = df.copy()
    filled[dyn] = df.groupby("patient_id", sort=False)[dyn].ffill()
    meas_n = df[dyn].notna().sum()
    ff_n = (filled[dyn].notna() & df[dyn].isna()).sum()
    miss_n = filled[dyn].isna().sum()
    fill_acc = pd.DataFrame({"set": s, "measured": meas_n, "forward_filled": ff_n, "still_missing": miss_n})
    fill_acc.index.name = "column"

    # write
    order = ["patient_id", "training_set"] + dyn + STATIC + ["ICULOS", "SepsisLabel", "n_measured", "is_empty_row", "is_leading_empty"]
    for frame in (df, filled):
        for c in ("Gender", "SepsisLabel"):
            frame[c] = frame[c].astype("int8")
        frame["ICULOS"] = frame["ICULOS"].astype("int16")
    CLEAN.mkdir(parents=True, exist_ok=True)
    p_meas, p_fill = CLEAN / f"measured_set{s}.parquet", CLEAN / f"filled_ffill_set{s}.parquet"
    df[order].to_parquet(p_meas, engine="pyarrow", index=False, compression="zstd")
    filled[order].to_parquet(p_fill, engine="pyarrow", index=False, compression="zstd")
    n_written = len(pd.read_parquet(p_meas, columns=["patient_id"]))
    assert n_written == len(df), "row count written does not match row count in memory"
    out(f"wrote {p_meas.relative_to(ROOT)} and {p_fill.relative_to(ROOT)} ({n_written:,} rows each)")

    pat = df.groupby("patient_id", sort=False).agg(sepsis=("SepsisLabel", "max")).reset_index()
    pat.insert(1, "training_set", s)
    return log, removed, pd.DataFrame(col_info, index=["pct_null_raw"]).T, inv, variants, coverage, fill_acc, pat


def write_sha256sums() -> None:
    files = sorted(RAW_PSV.rglob("*.psv")) if RAW_PSV.exists() else []
    if not files:
        out(f"\nSHA256SUMS: no .psv files under {RAW_PSV.relative_to(ROOT)}, skipped")
        return
    with open(RAW_PSV / "SHA256SUMS", "w", newline="\n") as fh:
        for f in files:
            fh.write(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(RAW_PSV).as_posix()}\n")
    out(f"\nSHA256SUMS: hashed {len(files):,} raw files -> {(RAW_PSV / 'SHA256SUMS').relative_to(ROOT)}")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    logs, removed, cols, invs, variants, covs, fills, pats = [], [], [], [], [], {}, [], []
    out("CLEAN LAYER REPORT")
    for s in ("A", "B"):
        log, rem, col, inv, var, cov, fa, pat = clean_set(s)
        logs += log
        removed += rem
        cols.append(col.rename(columns={"pct_null_raw": f"pct_null_raw_{s}"}))
        invs.append(inv)
        variants.append(var)
        covs[s] = cov
        fills.append(fa.reset_index())
        pats.append(pat)

    # ---- attrition ---------------------------------------------------------
    att = pd.DataFrame(logs)
    num = ["patients", "rows", "positive_rows", "sepsis_patients"]
    tot = att.groupby(["step", "rule"], sort=False)[num].sum().reset_index().assign(set="A+B")
    att = pd.concat([att, tot], ignore_index=True)
    att["set"] = pd.Categorical(att["set"], ["A", "B", "A+B"])
    att = att.sort_values(["set", "step"]).reset_index(drop=True)
    att["rows_lost_vs_arrived"] = att["rows"] - att.groupby("set")["rows"].transform("first")
    att.to_csv(REPORTS / "attrition.csv", index=False)
    out("\n" + "=" * 78 + "\nATTRITION (patients / rows / positive rows / sepsis patients after each step)\n" + "=" * 78)
    out(att.to_string(index=False))

    out("\nColumns dropped (share of raw cells that were null):")
    out(pd.concat(cols, axis=1).round(4).to_string())

    # ---- invalidated cells -------------------------------------------------
    inv = pd.concat(invs, ignore_index=True)
    inv.to_csv(REPORTS / "invalidated_cells.csv", index=False)
    out("\n" + "=" * 78 + "\nIMPLAUSIBLE VALUES SET TO NaN (cells; rows are kept)\n" + "=" * 78)
    out(inv.assign(pct_of_measured=(inv["pct_of_measured"] * 100).round(3)).to_string(index=False))

    # ---- removed patients ---------------------------------------------------
    rem = pd.DataFrame(removed, columns=["patient_id", "training_set", "reason", "rows", "positive_rows"])
    rem.to_csv(REPORTS / "removed_patients.csv", index=False)
    out(f"\nPatients removed: {len(rem)} (full list in reports/removed_patients.csv)")
    if len(rem):
        out(rem.head(20).to_string(index=False))

    # ---- what each optional filter would cost -------------------------------
    var = pd.concat(variants, ignore_index=True)
    var.to_csv(REPORTS / "variant_sizes.csv", index=False)
    out("\n" + "=" * 78 + "\nWHAT OPTIONAL FILTERS WOULD REMOVE (not applied; for choosing and justifying)\n" + "=" * 78)
    show = var.copy()
    for c in ("pct_rows", "pct_positive_rows", "pct_sepsis_patients"):
        show[c] = (show[c] * 100).round(2)
    out(show.to_string(index=False))

    # ---- measured vs filled --------------------------------------------------
    fa = pd.concat(fills, ignore_index=True)
    fa.to_csv(REPORTS / "fill_accounting.csv", index=False)
    out("\n" + "=" * 78 + "\nMEASURED vs FORWARD-FILLED vs STILL MISSING (time-varying cells)\n" + "=" * 78)
    for s in ("A", "B"):
        g = fa[fa["set"] == s]
        m, f_, x = g["measured"].sum(), g["forward_filled"].sum(), g["still_missing"].sum()
        t = m + f_ + x
        out(f"set {s}: measured {m:,} ({m / t:.1%}) | forward-filled {f_:,} ({f_ / t:.1%}) | still missing {x:,} ({x / t:.1%}) | total cells {t:,}")

    # ---- feature coverage ----------------------------------------------------
    cov = pd.DataFrame(covs)
    cov["shared"] = (cov["A"] >= SHARED_MIN_COVERAGE) & (cov["B"] >= SHARED_MIN_COVERAGE)
    cov.index.name = "column"
    cov.to_csv(REPORTS / "feature_coverage.csv")
    out("\n" + "=" * 78 + f"\nFEATURE COVERAGE: share of patients with >= 1 measurement (shared = >= {SHARED_MIN_COVERAGE:.0%} in both)\n" + "=" * 78)
    out((cov.assign(A=(cov["A"] * 100).round(1), B=(cov["B"] * 100).round(1))).to_string())
    out("\nshared time-varying features: " + ", ".join(cov.index[cov["shared"]]))

    # ---- patient-level folds -------------------------------------------------
    allp = pd.concat(pats, ignore_index=True)
    strat = (allp["training_set"] == "B").astype(int) * 2 + allp["sepsis"].astype(int)
    fold = np.empty(len(allp), dtype=int)
    for k, (_, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(allp, strat)):
        fold[te] = k
    allp["fold"] = fold
    allp.to_csv(CLEAN / "patient_splits.csv", index=False)
    out("\n" + "=" * 78 + f"\nPATIENT-LEVEL FOLDS (seed {SEED}, stratified by set and sepsis)\n" + "=" * 78)
    out(allp.groupby(["training_set", "fold"]).agg(patients=("sepsis", "size"), sepsis_patients=("sepsis", "sum")).to_string())

    write_sha256sums()
    out("\nwrote reports/clean_report.txt and the CSVs in reports/")
    (REPORTS / "clean_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
