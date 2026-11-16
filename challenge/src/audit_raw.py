"""Audit the raw Parquet layer and write a compact report to paste back.

Reads   data/parquet_raw/training_setA.parquet and training_setB.parquet
Writes  reports/audit_report.txt     (the text report, also printed)
        reports/patient_summary.csv  (one row per patient)

Nothing here modifies data. Every number is a count a reader can reproduce.

Run from challenge/:  uv run python src/audit_raw.py
"""

import gc
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = ROOT / "data" / "parquet_raw"
REPORT_DIR = ROOT / "data" / "reports"

DYNAMIC = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
VITALS = DYNAMIC[:7]  # HR, O2Sat, Temp, SBP, MAP, DBP, Resp
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]

# Deliberately wide plausibility limits (starting points, adjust after reading the
# min/max table). Values outside these are counted, never changed.
RANGES = {
    "HR": (20, 300),
    "O2Sat": (50, 100),
    "Temp": (25, 45),
    "SBP": (30, 300),
    "MAP": (20, 250),
    "DBP": (10, 200),
    "Resp": (3, 80),
    "pH": (6.5, 8.0),
    "Age": (0, 120),
}

lines: list[str] = []


def out(text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


def audit_set(df: pd.DataFrame, s: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Print the audit for one training set. Returns (column table, patient table)."""
    out("\n" + "=" * 72)
    out(f"TRAINING SET {s}")
    out("=" * 72)

    pid = df["patient_id"]
    n_rows, n_pat = len(df), pid.nunique()
    out(f"rows={n_rows:,}  patients={n_pat:,}  columns={df.shape[1]}")

    # ---- stay length -------------------------------------------------------
    stay = df.groupby("patient_id", sort=False).size()
    qs = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
    q = stay.quantile(qs)
    out("\n[stay length, rows = hours]")
    out("  " + "  ".join(f"p{round(k * 100)}={v:.0f}" for k, v in q.items()))
    out("  patients with stay <= 6 / 12 / 24 rows: "
        + " / ".join(f"{int((stay <= k).sum()):,}" for k in (6, 12, 24)))

    # ---- ICULOS structure (checked in stored order) ------------------------
    g = df.groupby("patient_id", sort=False)["ICULOS"]
    step = g.diff()
    first = g.first()
    out("\n[ICULOS structure, stored order]")
    out(f"  patients whose first ICULOS != 1: {int((first != 1).sum()):,}"
        f"  (min first={first.min():.0f}, max first={first.max():.0f})")
    out(f"  rows with step <= 0 (duplicate or out of order): {int((step <= 0).sum()):,}")
    gap_mask = step > 1
    out(f"  rows with step > 1 (gap in hours): {int(gap_mask.sum()):,}"
        f" in {df.loc[gap_mask, 'patient_id'].nunique():,} patients")
    out(f"  duplicate (patient_id, ICULOS) rows: {int(df.duplicated(['patient_id', 'ICULOS']).sum()):,}")

    # ---- labels ------------------------------------------------------------
    y = df["SepsisLabel"]
    pat_any = y.groupby(pid, sort=False).max()
    pos_rows = int((y == 1).sum())
    sepsis_pat = int((pat_any == 1).sum())
    out("\n[labels]")
    out(f"  SepsisLabel values present: {sorted(y.dropna().unique().tolist())}, nulls={int(y.isna().sum())}")
    out(f"  positive rows:    {pos_rows:,} / {n_rows:,} ({pos_rows / n_rows:.2%})")
    out(f"  sepsis patients:  {sepsis_pat:,} / {n_pat:,} ({sepsis_pat / n_pat:.2%})")
    prev = y.groupby(pid, sort=False).shift()
    drops = (prev == 1) & (y == 0)
    out(f"  label goes 1 -> 0 within a stay: {int(drops.sum()):,} rows in {df.loc[drops, 'patient_id'].nunique():,} patients")
    pos_hours = (y == 1).groupby(pid, sort=False).sum()
    pos_hours = pos_hours[pos_hours > 0]
    if len(pos_hours):
        out(f"  positive hours per sepsis patient: median={pos_hours.median():.0f}, "
            f"p25={pos_hours.quantile(.25):.0f}, p75={pos_hours.quantile(.75):.0f}")
    first_pos = df.loc[y == 1].groupby("patient_id")["ICULOS"].min()
    if len(first_pos):
        out(f"  ICULOS of first positive label: median={first_pos.median():.0f}; "
            f"patients with first positive at ICULOS <= 6 / 12 / 24: "
            + " / ".join(f"{int((first_pos <= k).sum()):,}" for k in (6, 12, 24)))
    stay_sep = stay[pat_any == 1]
    stay_non = stay[pat_any == 0]
    out(f"  median stay: sepsis={stay_sep.median():.0f} rows, non-sepsis={stay_non.median():.0f} rows")

    # ---- measured vs missing ----------------------------------------------
    meas = df[DYNAMIC].notna()
    n_meas_row = meas.sum(axis=1)
    cells = meas.size
    meas_cells = int(meas.to_numpy().sum())
    out("\n[measured vs missing, 34 time-varying variables]")
    out(f"  measured cells: {meas_cells:,} / {cells:,} ({meas_cells / cells:.2%})")
    bins = pd.cut(n_meas_row, [-1, 0, 2, 5, 10, 34], labels=["0", "1-2", "3-5", "6-10", "11+"])
    dist = bins.value_counts().sort_index()
    out("  rows by number of variables measured that hour: "
        + ", ".join(f"{k}: {v:,} ({v / n_rows:.1%})" for k, v in dist.items()))
    no_vital = ~df[VITALS].notna().any(axis=1)
    out(f"  rows with no vital sign (HR/O2Sat/Temp/SBP/MAP/DBP/Resp) measured: {int(no_vital.sum()):,} ({no_vital.mean():.1%})")
    has_obs = (n_meas_row > 0).astype(int)
    seen = has_obs.groupby(pid, sort=False).cummax().astype(bool)
    out(f"  leading empty rows (before a patient's first measurement of anything): {int((~seen).sum()):,}")
    pat_meas = n_meas_row.groupby(pid, sort=False).sum()
    out(f"  patients with zero measured values in the whole stay: {int((pat_meas == 0).sum()):,}")

    # ---- static variables --------------------------------------------------
    out("\n[static variables, repeated on every row]")
    for c in STATIC:
        nun = df.groupby("patient_id", sort=False)[c].nunique(dropna=False)
        out(f"  {c}: patients with >1 distinct value within a stay (NaN counts as a value): {int((nun > 1).sum()):,}")
    first_rows = df.drop_duplicates("patient_id")
    combo = first_rows.groupby(["Unit1", "Unit2"], dropna=False).size()
    out("  patients by (Unit1, Unit2) on their first row:")
    for (u1, u2), n in combo.items():
        out(f"    Unit1={u1}, Unit2={u2}: {n:,}")
    hnull = df.loc[df["HospAdmTime"].isna(), "patient_id"].unique()
    out(f"  HospAdmTime null: {int(df['HospAdmTime'].isna().sum()):,} rows in {len(hnull)} patients"
        + (f" (first few: {', '.join(hnull[:5])})" if len(hnull) else ""))

    # ---- plausibility ------------------------------------------------------
    out("\n[values outside the wide plausibility limits (counted, not changed)]")
    any_bad = False
    for c, (lo, hi) in RANGES.items():
        bad = int(((df[c] < lo) | (df[c] > hi)).sum())
        if bad:
            any_bad = True
            out(f"  {c}: {bad:,} outside [{lo}, {hi}]")
    for c in ("Gender", "Unit1", "Unit2", "SepsisLabel"):
        bad = int((~df[c].isin([0, 1]) & df[c].notna()).sum())
        if bad:
            any_bad = True
            out(f"  {c}: {bad:,} values not in {{0, 1}}")
    if not any_bad:
        out("  none")

    # ---- per-column table --------------------------------------------------
    rows = []
    for c in DYNAMIC + STATIC + ["ICULOS"]:
        col = df[c]
        rows.append({
            "column": c,
            "pct_null": col.isna().mean(),
            "patients_with_any": int(col.notna().groupby(pid, sort=False).any().sum()),
            "min": col.min(),
            "p01": col.quantile(0.01),
            "median": col.median(),
            "p99": col.quantile(0.99),
            "max": col.max(),
        })
    col_tbl = pd.DataFrame(rows).set_index("column")
    out("\n[per-column: %null, patients with any measurement, observed min / p1 / median / p99 / max]")
    out(f"  {'column':<17}{'%null':>7}{'pts_any':>9}{'min':>10}{'p1':>10}{'median':>10}{'p99':>10}{'max':>10}")
    for c, r in col_tbl.iterrows():
        out(f"  {c:<17}{r.pct_null * 100:>6.1f}%{int(r.patients_with_any):>9,}"
            f"{r['min']:>10.4g}{r.p01:>10.4g}{r['median']:>10.4g}{r.p99:>10.4g}{r['max']:>10.4g}")

    # ---- patient-level table ----------------------------------------------
    pat = pd.DataFrame({
        "training_set": s,
        "n_rows": stay,
        "sepsis": pat_any.astype(int),
        "n_pos_rows": (y == 1).groupby(pid, sort=False).sum(),
        "first_pos_iculos": first_pos,
        "n_measured_cells": n_meas_row.groupby(pid, sort=False).sum(),
        "leading_empty_rows": (~seen).groupby(pid, sort=False).sum(),
    })
    pat["pct_cells_measured"] = pat["n_measured_cells"] / (pat["n_rows"] * len(DYNAMIC))
    pat.index.name = "patient_id"
    return col_tbl, pat.reset_index()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    col_tables, pat_tables, id_sets = {}, [], {}

    out("RAW PARQUET AUDIT")
    for s in ("A", "B"):
        path = PARQUET_DIR / f"training_set{s}.parquet"
        out(f"\nreading {path.relative_to(ROOT)}")
        df = pd.read_parquet(path)
        id_sets[s] = set(df["patient_id"].unique())
        col_tables[s], pat = audit_set(df, s)
        pat_tables.append(pat)
        del df
        gc.collect()

    # ---- cross-set comparison ---------------------------------------------
    out("\n" + "=" * 72)
    out("A vs B")
    out("=" * 72)
    overlap = id_sets["A"] & id_sets["B"]
    out(f"patient_ids appearing in both sets: {len(overlap):,}")
    cmp = pd.DataFrame({s: col_tables[s]["pct_null"] * 100 for s in ("A", "B")})
    cmp["B_minus_A_pp"] = cmp["B"] - cmp["A"]
    flagged = cmp[cmp["B_minus_A_pp"].abs() >= 10].sort_values("B_minus_A_pp")
    out("columns whose %null differs by >= 10 percentage points between A and B:")
    if flagged.empty:
        out("  none")
    for c, r in flagged.iterrows():
        out(f"  {c:<17} A={r['A']:5.1f}%  B={r['B']:5.1f}%  diff={r['B_minus_A_pp']:+.1f} pp")

    pd.concat(pat_tables, ignore_index=True).to_csv(REPORT_DIR / "patient_summary.csv", index=False)
    (REPORT_DIR / "audit_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out(f"\nwrote {(REPORT_DIR / 'audit_report.txt').relative_to(ROOT)} and patient_summary.csv")


if __name__ == "__main__":
    main()
