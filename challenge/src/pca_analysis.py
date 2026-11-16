"""PCA for the presentation: fit on training set A, project training set B.

PCA needs a complete matrix, so a filled copy is built *for this analysis only*.
The Parquet files are never modified. Filling is causal and fully accounted for:
    1. measured       value present in the raw file
    2. forward-filled carried from the same patient's earlier hours (no future data used)
    3. median-filled  still missing after step 2 (no earlier measurement in that stay),
                      set to the median of set A's measured values
The share of cells in each category is printed and saved, so the PCA is never
presented as if it were built on measured data alone.

Outputs (in reports/pca/):
    pca_summary.txt                  text summary, paste this back
    pca_explained_variance.csv, pca_loadings.csv, pca_fill_accounting.csv
    pca_scree.png, pca_scatter_label.png, pca_scatter_set.png, pca_loadings.png

Run from challenge/:
    uv run python src/pca_analysis.py
    uv run python src/pca_analysis.py --max-missing 0.99   # keep sparser variables too
    uv run python src/pca_analysis.py --shared-min-coverage 0.5   # only variables both hospitals measure
Requires: uv add scikit-learn matplotlib
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = ROOT / "data" / "parquet_raw"
OUT_DIR = ROOT /"data" / "reports" / "pca"

DYNAMIC = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
EXTRA = ["Age", "Gender", "ICULOS"]  # always complete in the raw data
META = ["patient_id", "training_set", "SepsisLabel"]

lines: list[str] = []


def out(text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


def load(s: str) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_DIR / f"training_set{s}.parquet", columns=META + DYNAMIC + EXTRA)
    # Time order within each patient is required for forward fill.
    return df.sort_values(["patient_id", "ICULOS"], kind="stable").reset_index(drop=True)


def fill(df: pd.DataFrame, feats: list[str], medians: pd.Series):
    """Forward fill within patient, then median fill. Returns (filled, accounting)."""
    X = df[feats]
    measured = X.notna()
    ff = X.groupby(df["patient_id"], sort=False).ffill()
    after = ff.notna()
    filled = ff.fillna(medians)
    acct = pd.DataFrame({
        "measured": measured.mean(),
        "forward_filled": (after & ~measured).mean(),
        "median_filled": (~after).mean(),
    })
    overall = {
        "measured": float(measured.to_numpy().mean()),
        "forward_filled": float((after & ~measured).to_numpy().mean()),
        "median_filled": float((~after).to_numpy().mean()),
    }
    return filled, acct, overall


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sd = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-missing", type=float, default=0.95,
                    help="drop time-varying variables with a higher missing fraction in set A (default 0.95)")
    ap.add_argument("--shared-min-coverage", type=float, default=None,
                    help="keep only variables measured at least once in this share of patients in BOTH sets "
                         "(e.g. 0.5); overrides --max-missing")
    ap.add_argument("--n-plot", type=int, default=60000, help="rows sampled for scatter plots")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    out("PCA SUMMARY")
    A, B = load("A"), load("B")
    out(f"rows used: A={len(A):,}  B={len(B):,}")

    # ---- feature choice (decided on set A only) ---------------------------
    miss_A = A[DYNAMIC].isna().mean()
    if args.shared_min_coverage is None:
        keep = [c for c in DYNAMIC if miss_A[c] <= args.max_missing]
        drop = [c for c in DYNAMIC if miss_A[c] > args.max_missing]
        feats = keep + EXTRA
        out(f"\nfeatures kept ({len(feats)}): {', '.join(feats)}")
        out(f"time-varying variables dropped for being > {args.max_missing:.0%} missing in set A ({len(drop)}):")
        for c in drop:
            out(f"  {c:<17} {miss_A[c]:.1%} missing in A")
    else:
        thr = args.shared_min_coverage
        cov = {
            n: d[DYNAMIC].notna().groupby(d["patient_id"], sort=False).any().mean()
            for n, d in (("A", A), ("B", B))
        }
        keep = [c for c in DYNAMIC if cov["A"][c] >= thr and cov["B"][c] >= thr]
        drop = [c for c in DYNAMIC if c not in keep]
        feats = keep + EXTRA
        out(f"\nfeatures kept ({len(feats)}): {', '.join(feats)}")
        out(f"time-varying variables dropped for being measured in < {thr:.0%} of patients in set A or set B ({len(drop)}):")
        for c in drop:
            out(f"  {c:<17} patients with a measurement: A {cov['A'][c]:.1%}, B {cov['B'][c]:.1%}")

    # ---- fill, with accounting --------------------------------------------
    medians = A[feats].median()
    XA, acct_A, over_A = fill(A, feats, medians)
    XB, acct_B, over_B = fill(B, feats, medians)
    out("\ncells in the PCA input matrix (share of all cells):")
    for name, o in (("A", over_A), ("B", over_B)):
        out(f"  set {name}: measured {o['measured']:.1%} | forward-filled {o['forward_filled']:.1%} | median-filled {o['median_filled']:.1%}")
    acct = pd.concat({"A": acct_A, "B": acct_B}, axis=1)
    acct.to_csv(OUT_DIR / "pca_fill_accounting.csv")

    # ---- scale and fit on A, project B ------------------------------------
    scaler = StandardScaler().fit(XA)
    ZA, ZB = scaler.transform(XA), scaler.transform(XB)
    pca = PCA(random_state=args.seed).fit(ZA)
    PA, PB = pca.transform(ZA), pca.transform(ZB)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    pcs = [f"PC{i + 1}" for i in range(len(evr))]

    pd.DataFrame({"component": pcs, "explained_variance_ratio": evr, "cumulative": cum}).to_csv(
        OUT_DIR / "pca_explained_variance.csv", index=False)
    loadings = pd.DataFrame(pca.components_.T, index=feats, columns=pcs)
    loadings.to_csv(OUT_DIR / "pca_loadings.csv")

    out("\nexplained variance ratio (fit on set A):")
    out("  " + "  ".join(f"{p}={v:.1%}" for p, v in list(zip(pcs, evr))[:8]))
    for thr in (0.8, 0.9, 0.95):
        out(f"  components needed for {thr:.0%} of variance: {int(np.searchsorted(cum, thr) + 1)} of {len(evr)}")

    out("\ntop loadings (by absolute value):")
    for p in pcs[:3]:
        top = loadings[p].reindex(loadings[p].abs().sort_values(ascending=False).index)[:5]
        out(f"  {p}: " + ", ".join(f"{k} {v:+.2f}" for k, v in top.items()))

    # ---- how much does sepsis separate along the PCs? ---------------------
    yA = A["SepsisLabel"].to_numpy() == 1
    yB = B["SepsisLabel"].to_numpy() == 1
    out(f"\npositive rows: A={int(yA.sum()):,} ({yA.mean():.2%})  B={int(yB.sum()):,} ({yB.mean():.2%})")
    out("effect size (Cohen's d, positive minus negative rows) along each PC; |d| < 0.2 is a small separation:")
    for k in range(min(5, PA.shape[1])):
        out(f"  PC{k + 1}: A d={cohens_d(PA[yA, k], PA[~yA, k]):+.2f}   B (projected with A's PCA) d={cohens_d(PB[yB, k], PB[~yB, k]):+.2f}")
    out("\nA vs B mean position along PC1..PC3 (shift between hospitals' data):")
    out("  " + "  ".join(f"PC{k + 1}: A={PA[:, k].mean():+.2f}, B={PB[:, k].mean():+.2f}" for k in range(3)))

    # ---- figures -----------------------------------------------------------
    n_show = min(15, len(evr))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, n_show + 1), evr[:n_show] * 100, color="#4C78A8", label="per component")
    ax.plot(range(1, n_show + 1), cum[:n_show] * 100, "o-", color="#E45756", label="cumulative")
    ax.axhline(90, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance (%)")
    ax.set_title("Scree plot (PCA fit on training set A)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_scree.png", dpi=200)
    plt.close(fig)

    def lims(*arrays):
        both = np.concatenate([a[:, :2].ravel() for a in arrays])
        lo, hi = np.percentile(both, [0.5, 99.5])
        pad = 0.1 * (hi - lo)
        return lo - pad, hi + pad

    neg_idx, pos_idx = np.flatnonzero(~yA), np.flatnonzero(yA)
    neg_s = rng.choice(neg_idx, min(args.n_plot, len(neg_idx)), replace=False)
    pos_s = rng.choice(pos_idx, min(max(args.n_plot // 5, 1), len(pos_idx)), replace=False) if len(pos_idx) else pos_idx
    lo, hi = lims(PA[neg_s], PA[pos_s]) if len(pos_s) else lims(PA[neg_s])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(PA[neg_s, 0], PA[neg_s, 1], s=2, alpha=0.15, color="#9aa0a6", label=f"SepsisLabel 0 (n={len(neg_s):,} sampled)")
    if len(pos_s):
        ax.scatter(PA[pos_s, 0], PA[pos_s, 1], s=3, alpha=0.35, color="#D62728", label=f"SepsisLabel 1 (n={len(pos_s):,} sampled)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title("Training set A in PC space\n(positives over-sampled for visibility)")
    ax.legend(markerscale=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_scatter_label.png", dpi=200)
    plt.close(fig)

    ia = rng.choice(len(PA), min(args.n_plot // 2, len(PA)), replace=False)
    ib = rng.choice(len(PB), min(args.n_plot // 2, len(PB)), replace=False)
    lo, hi = lims(PA[ia], PB[ib])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(PA[ia, 0], PA[ia, 1], s=2, alpha=0.2, color="#4C78A8", label="set A (fit)")
    ax.scatter(PB[ib, 0], PB[ib, 1], s=2, alpha=0.2, color="#F58518", label="set B (projected)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title("Set A vs set B in the same PC space")
    ax.legend(markerscale=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_scatter_set.png", dpi=200)
    plt.close(fig)

    n_load = min(4, len(pcs))
    L = loadings.iloc[:, :n_load].to_numpy().T
    m = float(np.abs(L).max())
    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(feats)), 3.2))
    im = ax.imshow(L, cmap="coolwarm", vmin=-m, vmax=m, aspect="auto")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90)
    ax.set_yticks(range(n_load))
    ax.set_yticklabels([f"{pcs[i]} ({evr[i]:.1%})" for i in range(n_load)])
    ax.set_title("PCA loadings (standardized features)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_loadings.png", dpi=200)
    plt.close(fig)

    out(f"\nwrote figures and CSVs to {OUT_DIR.relative_to(ROOT)}")
    (OUT_DIR / "pca_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()