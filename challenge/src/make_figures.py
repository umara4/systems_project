"""Report figures for the Hospital B results.

Needs test_B_final_results.csv (from evaluate_test_B.py run with SPLIT = "test_B") and the saved models.
Writes five PNGs to challenge/outputs/figures/.
Run:  python src/make_figures.py
"""
import joblib
import matplotlib
matplotlib.use("Agg")                      # write files only, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from preprocessing_split_data import OUTPUT_DIR, build_datasets
from window_eval import EARLY, LABEL_SHIFT, LATE, window_target

FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ORDER = ["logreg_raw", "logreg_trend", "random_forest", "xgb_baseline",      # main comparison
         "xgb_no_flags", "xgb_extended", "xgb_extended_no_flags", "xgb_extended_demo"]   # extensions
LABELS = {
    "logreg_raw": "LR (raw)", "logreg_trend": "LR (trend)", "random_forest": "Random forest",
    "xgb_baseline": "XGBoost (baseline)", "xgb_no_flags": "XGBoost (no flags)",
    "xgb_extended": "XGBoost (extended)", "xgb_extended_no_flags": "XGBoost (extended, no flags)",
    "xgb_extended_demo": "XGBoost (extended + age/sex)",
}
COLORS = {n: f"C{i}" for i, n in enumerate(ORDER)}

# Models used for the curve and lead-time figures (they all use the default 44-feature table)
MAIN = {
    "logreg_trend": OUTPUT_DIR / "logistic_regression2" / "logreg_trend.joblib",
    "random_forest": OUTPUT_DIR / "random_forest" / "model.joblib",
    "xgb_baseline": OUTPUT_DIR / "xgb_baseline" / "model.joblib",
}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


def fig_pr_auc(res, rate_t):
    names = [n for n in ORDER if n in res.index]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, n in enumerate(names):
        r = res.loc[n]
        ax.errorbar(r["test_pr_auc"], i,
                    xerr=[[r["test_pr_auc"] - r["ci_low"]], [r["ci_high"] - r["test_pr_auc"]]],
                    fmt="o", color=COLORS[n], capsize=3)
    ax.axvline(rate_t, ls="--", color="gray", label=f"no skill ({rate_t:.3f})")
    ax.axhline(3.5, ls=":", color="lightgray")            # main comparison above, extensions below
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Window PR-AUC on Hospital B (95% patient-level bootstrap interval)")
    ax.legend(loc="lower right")
    save(fig, "fig1_pr_auc_test.png")


def fig_lift(res, rate_v, rate_t):
    names = [n for n in ORDER if n in res.index]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, n in enumerate(names):
        lv = res.loc[n, "val_pr_auc"] / rate_v
        lt = res.loc[n, "test_pr_auc"] / rate_t
        ax.plot([lv, lt], [i, i], color="lightgray", zorder=1)
        ax.scatter(lv, i, facecolors="white", edgecolors="gray", zorder=2,
                   label="Hospital A validation" if i == 0 else None)
        ax.scatter(lt, i, color="C0", zorder=2, label="Hospital B test" if i == 0 else None)
    ax.axvline(1, ls="--", color="gray")
    ax.axhline(3.5, ls=":", color="lightgray")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Lift over no skill (PR-AUC / base rate; 1 = no skill)")
    ax.legend(loc="lower right")
    save(fig, "fig2_lift_validation_vs_test.png")


def fig_pr_curves(y, keep, preds, res, rate_t):
    fig, ax = plt.subplots(figsize=(6, 5))
    for n, (proba, _) in preds.items():
        p, r, _ = precision_recall_curve(y[keep], proba[keep])
        m = r >= 0.02                                       # hide the noisy tip at tiny recall
        ax.plot(r[m], p[m], color=COLORS[n], label=f"{LABELS[n]} (PR-AUC {res.loc[n, 'test_pr_auc']:.3f})")
        ax.scatter(res.loc[n, "recall"], res.loc[n, "precision"], color=COLORS[n], edgecolors="black", zorder=3)
    ax.axhline(rate_t, ls="--", color="gray", label=f"no skill ({rate_t:.3f})")
    ax.set_xlabel("Recall (window rows)")
    ax.set_ylabel("Precision")
    ax.set_ylim(bottom=0)
    ax.set_title("Hospital B (dots = frozen operating points)")
    ax.legend()
    save(fig, "fig3_pr_curves_test.png")


def lead_time_counts(df, proba, threshold):
    """For sepsis patients: hours before onset of the FIRST alert inside [onset-12h, onset+3h]."""
    pid = df["patient_id"]
    pos = df.groupby("patient_id").cumcount()
    first_one = pos.where(df["SepsisLabel"] == 1).groupby(pid).transform("min")
    rel = pos - (first_one + LABEL_SHIFT)                   # hours from onset; NaN for non-sepsis patients
    alert = pd.Series(proba >= threshold, index=df.index)
    hit = alert & (rel >= -EARLY) & (rel <= LATE)
    first_rel = rel[hit].groupby(pid[hit]).min()            # earliest alert inside the window, per patient
    lead = (-first_rel).astype(int)                         # hours before onset (negative = after onset)
    hours = list(range(EARLY, -LATE - 1, -1))               # 12, 11, ..., 0, -1, -2, -3
    counts = lead.value_counts().reindex(hours, fill_value=0)
    n_sepsis = int(df.loc[df["SepsisLabel"] == 1, "patient_id"].nunique())
    fa_rate = float(alert[first_one.isna()].mean())         # alert rate on non-sepsis patients' rows
    return counts, n_sepsis, fa_rate


def fig_lead_time(test, preds):
    fig, axes = plt.subplots(1, len(preds), figsize=(4.8 * len(preds), 4), sharey=True)
    for k, (ax, (n, (proba, thr))) in enumerate(zip(np.atleast_1d(axes), preds.items())):
        counts, n_sepsis, p = lead_time_counts(test, proba, thr)
        hours = counts.index.to_numpy()
        w = len(hours)
        caught = int(counts.sum())
        x = np.arange(w)
        ax.bar(x, counts.values, color=COLORS[n])
        ax.bar(w + 0.5, n_sepsis - caught, color="lightgray")
        # Chance reference: independent random hourly alerts at the model's false-alarm rate.
        # Real alerts cluster, so this is only a rough guide.
        chance = n_sepsis * p * (1 - p) ** (EARLY - hours)
        ax.plot(x, chance, "k.:", label="chance (random alerts)")
        ax.plot(w + 0.5, n_sepsis * (1 - p) ** w, "k.")
        ax.axvline(EARLY, ls="--", color="gray")             # hour 0 = onset
        ax.set_xticks(list(x) + [w + 0.5])
        ax.set_xticklabels([str(h) for h in hours] + ["none"], fontsize=8)
        ax.set_xlabel("Hours before onset of first alert in window (negative = after onset)", fontsize=8)
        ax.set_title(f"{LABELS[n]}\n{caught}/{n_sepsis} sepsis patients alerted in window", fontsize=10)
        if k == 0:
            ax.set_ylabel("Sepsis patients")
            ax.legend(fontsize=8)
        leads = np.repeat(hours, counts.values)
        print(f"{LABELS[n]:<18} alerted in window: {caught / n_sepsis:.1%} | chance: {1 - (1 - p) ** w:.1%} "
              f"| median lead among alerted: {np.median(leads):.0f} h")
    save(fig, "fig4_lead_time_test.png")


def fig_confusion(y, keep, preds):
    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 3.8))
    for k, (ax, (n, (proba, thr))) in enumerate(zip(np.atleast_1d(axes), preds.items())):
        yk, pred = y[keep], proba[keep] >= thr
        tn = int(((yk == 0) & ~pred).sum()); fp = int(((yk == 0) & pred).sum())
        fn = int(((yk == 1) & ~pred).sum()); tp = int(((yk == 1) & pred).sum())
        cm = np.array([[tn, fp], [fn, tp]])
        norm = cm / cm.sum(axis=1, keepdims=True)            # each row sums to 100% of that true class
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.0%})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["no alert", "alert"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["not in window", "in window"])
        ax.set_xlabel("Model output")
        ax.set_title(f"{LABELS[n]}\nprecision {tp / (tp + fp):.1%}, recall {tp / (tp + fn):.1%}", fontsize=10)
        if k == 0:
            ax.set_ylabel("Actual (rows kept in window metric)")
    save(fig, "fig5_confusion_test.png")



def main():
    res = pd.read_csv(OUTPUT_DIR / "test_B_final_results.csv", index_col=0)

    _, _, final, _, _ = build_datasets()                    # default 44-feature table
    val, test = final["validation_A"], final["test_B"]
    del final
    keep_v, y_v = window_target(val)
    keep_t, y_t = window_target(test)
    rate_v, rate_t = y_v[keep_v].mean(), y_t[keep_t].mean()
    del val

    preds = {}
    for name, path in MAIN.items():
        art = joblib.load(path)
        preds[name] = (art["model"].predict_proba(test[art["feature_cols"]])[:, 1], art["threshold"])

    fig_pr_auc(res, rate_t)
    fig_lift(res, rate_v, rate_t)
    fig_pr_curves(y_t, keep_t, preds, res, rate_t)
    fig_lead_time(test, preds)
    fig_confusion(y_t, keep_t, preds)
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()