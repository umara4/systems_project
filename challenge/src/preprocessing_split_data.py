import pandas as pd
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "challenge" / "data" / "parquet_split"
OUTPUT_DIR = PROJECT_ROOT / "challenge" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LABS = ["WBC", "Lactate", "Creatinine"]
BIOMARKERS = VITALS + LABS
KEEP = ["patient_id", "ICULOS", "SepsisLabel"]
LAG = 6
WINDOW = 6

SPLITS = ["train_A", "validation_A", "test_B"]

# Extended variable set (XGBoost extension variants)
EXTRA_VITALS = []       # EtCO2 was removed, no observed values in train_A
EXTRA_LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
    "Calcium", "Chloride", "Bilirubin_direct", "Glucose", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "Fibrinogen", "Platelets",
]
DEMOGRAPHICS = ["Age", "Gender"]
EXT_VITALS = VITALS + EXTRA_VITALS
EXT_LABS = LABS + EXTRA_LABS


def load_split(name, biomarkers=BIOMARKERS, static=()):
    df = pd.read_parquet(f"{DATA_DIR}/{name}.parquet")
    df = df[KEEP + biomarkers + list(static)].sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    df["SepsisLabel"] = df["SepsisLabel"].astype(int)
    assert not df.duplicated(["patient_id", "ICULOS"]).any(), f"{name}: duplicate hours"
    return df


def make_features(df, vitals=VITALS, labs=LABS):
    biomarkers = vitals + labs
    pid = df["patient_id"]
    new = {}                                   # new columns are collected here and added once at the end

    # 1) Flags and lab trend use RAW values, so they come before the forward fill
    for c in biomarkers:
        new[f"{c}_measured"] = df[c].notna().astype(int)
    for c in labs:
        seen = df[c].notna()
        change = df.loc[seen].groupby("patient_id")[c].diff()       # change between consecutive measurements
        new[f"{c}_change_prev"] = change.reindex(df.index).groupby(pid).ffill()

    # 2) Forward fill within each patient
    filled = df.groupby("patient_id")[biomarkers].ffill()

    # 3) Vital sign trends on the filled series (backward-looking only)
    g = filled.groupby(pid)
    for c in vitals:
        new[f"{c}_diff{LAG}"] = filled[c] - g[c].shift(LAG)
        new[f"{c}_rollmean{WINDOW}"] = g[c].rolling(WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
        new[f"{c}_rollstd{WINDOW}"] = g[c].rolling(WINDOW, min_periods=2).std().reset_index(level=0, drop=True)

    out = df.copy()
    out[biomarkers] = filled                   # overwriting existing columns doesn't fragment the frame
    return pd.concat([out, pd.DataFrame(new, index=df.index)], axis=1)


def build_datasets(vitals=VITALS, labs=LABS, static=()):
    biomarkers = vitals + labs
    raw = {name: load_split(name, biomarkers, static) for name in SPLITS}

    # No patient may appear in two splits
    ids = {name: set(df["patient_id"]) for name, df in raw.items()}
    for a, b in [("train_A", "validation_A"), ("train_A", "test_B"), ("validation_A", "test_B")]:
        assert not (ids[a] & ids[b]), f"patients shared between {a} and {b}"

    # build features (NaN remains where a patient does not have history yet)
    feat = {name: make_features(df, vitals, labs) for name, df in raw.items()}
    feature_cols = [c for c in feat["train_A"].columns if c not in KEEP]

    # check added for empty feature columns
    empty = [c for c in feature_cols if feat["train_A"][c].isna().all()]
    assert not empty, f"All-NaN in train_A (remove their source variables): {empty}"

    # values + flags + lab changes + 3 trends per vital + static columns
    expected = 2 * len(biomarkers) + len(labs) + 3 * len(vitals) + len(static)
    assert len(feature_cols) == expected, f"{len(feature_cols)} feature columns, expected {expected}"

    # only use the Training medians -- these will be applied to all 3 splits
    train_medians = feat["train_A"][feature_cols].median()
    final = {}
    for name, df in feat.items():
        out = df.copy()
        out[feature_cols] = out[feature_cols].fillna(train_medians)
        assert out[feature_cols].isna().sum().sum() == 0, f"{name}: NaN left after median fill"
        final[name] = out
    return raw, feat, final, feature_cols, train_medians

if __name__ == "__main__":
    raw, feat, final, FEATURE_COLS, train_medians = build_datasets()
    
    # save training medians (to use for scoring) and column list
    joblib.dump(
        {"feature_cols": FEATURE_COLS, "train_medians": train_medians},
        OUTPUT_DIR / "preprocessing.joblib",
    )

    # Causality check: a patient's first 20 hours must not change if later rows are removed
    counts = raw["train_A"]["patient_id"].value_counts()
    check_id = counts[counts > 30].index[0]
    one = raw["train_A"][raw["train_A"]["patient_id"] == check_id].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        make_features(one).iloc[:20].reset_index(drop=True),
        make_features(one.iloc[:20]).reset_index(drop=True),
    )
    print("causality check passed")

    # check if patient data contains gaps greater than one hour
    for name, df in raw.items():
        gaps = df.groupby("patient_id")["ICULOS"].diff().dropna()
        print(f"{name}: max gap between consecutive rows (hours) = {gaps.max()}")
        assert gaps.max() == 1, f"{name}: rows are not consecutive hours"
    
    # statistics per split
    measured_cols = [f"{c}_measured" for c in BIOMARKERS]
    rows = {}
    for name in raw:
        f, d = feat[name], final[name]
        n_values = len(f) * len(BIOMARKERS)
        raw_missing = int((f[measured_cols] == 0).sum().sum())
        left = int(f[BIOMARKERS].isna().sum().sum())          # NaN left after forward fill
        per_patient = d.groupby("patient_id")["SepsisLabel"].max()
        rows[name] = {
            "rows": len(d),
            "patients": d["patient_id"].nunique(),
            "sepsis patients (%)": round(per_patient.mean() * 100, 2),
            "sepsis rows (%)": round(d["SepsisLabel"].mean() * 100, 2),
            "biomarker values missing in raw data (%)": round(raw_missing / n_values * 100, 1),
            "filled by forward fill": raw_missing - left,
            "left for median fill (biomarkers)": left,
            "NaN in features after median fill": int(d[FEATURE_COLS].isna().sum().sum()),
        }
    print(pd.DataFrame(rows))
