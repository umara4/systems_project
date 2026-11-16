"""Window-based evaluation for sepsis models.

A "useful alert" is an alert from 12 hours before to 3 hours after the clinical onset time.
SepsisLabel turns on 6 hours before clinical onset, so onset = first label-1 row + 6.
Rows after onset + 3h are dropped. All other rows (non-sepsis patients, and sepsis patients
more than 12h before onset) are negatives.
Rows must be sorted by patient_id, ICULOS, and probabilities must follow the same row order.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score

LABEL_SHIFT = 6
EARLY, LATE = 12, 3


def window_target(df, early=EARLY, late=LATE):
    """Returns (keep, y_window) as numpy arrays. Compute once per split and reuse."""
    pos = df.groupby("patient_id").cumcount()
    first_one = pos.where(df["SepsisLabel"] == 1).groupby(df["patient_id"]).transform("min")
    rel = pos - (first_one + LABEL_SHIFT)            # hours from onset; NaN for non-sepsis patients
    keep = ~(rel > late)                             # drop rows after onset + 3h
    y_window = (rel >= -early) & (rel <= late)
    return keep.to_numpy(), y_window.astype(int).to_numpy()


def window_pr_auc(keep, y, proba):
    return average_precision_score(y[keep], proba[keep])


def tune_threshold(keep, y, proba, thresholds=None, beta=2):
    """Pick the threshold with the best F-beta on the window target. Use validation data only."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    scores = [fbeta_score(y[keep], proba[keep] >= t, beta=beta, zero_division=0) for t in thresholds]
    i = int(np.argmax(scores))
    return float(thresholds[i]), float(scores[i])


def evaluate(keep, y, proba, threshold, beta=2):
    yk, pk = y[keep], proba[keep]
    pred = pk >= threshold
    return {
        "pr_auc": float(average_precision_score(yk, pk)),
        "f2": float(fbeta_score(yk, pred, beta=beta, zero_division=0)),
        "precision": float(precision_score(yk, pred, zero_division=0)),
        "recall": float(recall_score(yk, pred, zero_division=0)),
        "threshold": float(threshold),
        "rows_kept": int(keep.sum()),
        "window_positive_pct": round(float(yk.mean()) * 100, 2),
    }


def paired_bootstrap(keep, y, patient_ids, probas, reference, n_boot=200, seed=42):
    """Patient-level bootstrap of window PR-AUC for several models scored on the same rows.
    probas: dict of model name -> probability array (full length, same row order as keep)."""
    yk = y[keep]
    pk = {name: np.asarray(p)[keep] for name, p in probas.items()}
    codes, uniques = pd.factorize(np.asarray(patient_ids)[keep])
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    ends = np.cumsum(sizes)
    starts = ends - sizes
    rng = np.random.default_rng(seed)

    boot = {name: [] for name in pk}
    for _ in range(n_boot):
        picked = rng.integers(0, len(uniques), len(uniques))
        idx = np.concatenate([order[starts[i]:ends[i]] for i in picked])
        for name, p in pk.items():
            boot[name].append(average_precision_score(yk[idx], p[idx]))

    ref = np.array(boot[reference])
    rows = {}
    for name, p in pk.items():
        b = np.array(boot[name])
        row = {"pr_auc": average_precision_score(yk, p),
               "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5)}
        if name != reference:
            d = b - ref
            row.update({"diff_vs_ref": d.mean(),
                        "diff_ci_low": np.percentile(d, 2.5), "diff_ci_high": np.percentile(d, 97.5)})
        rows[name] = row
    return pd.DataFrame(rows).T


# check window_target with artifical patient data
def run_tests():
    sick = pd.DataFrame({"patient_id": "p1", "ICULOS": np.arange(1, 61),
                         "SepsisLabel": (np.arange(60) >= 42).astype(int)})   # first label at row 42 -> onset row 48
    healthy = pd.DataFrame({"patient_id": "p2", "ICULOS": np.arange(1, 11), "SepsisLabel": 0})
    df = pd.concat([sick, healthy], ignore_index=True)
    keep, y = window_target(df)
    assert y[:36].sum() == 0 and y[36:52].all()      # window = onset-12 .. onset+3 = rows 36..51
    assert keep[:52].all() and not keep[52:60].any() # rows after onset+3 dropped
    assert keep[60:].all() and y[60:].sum() == 0     # non-sepsis rows kept, all negative
    print("window_eval tests passed")


if __name__ == "__main__":
    run_tests()