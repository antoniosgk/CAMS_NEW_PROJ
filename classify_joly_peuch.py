"""
classify_joly_peuch_v2.py
==========================
Refined Joly & Peuch classification — improvements over v1:

  1. Drops the 3 weekend-effect indicators (WE_MEAN, WE_MAX, WE_STD)
     which carry no discriminating information in GEOS-5 NR (fixed
     emissions → no weekday/weekend cycle → all values ≈ 1.0).

  2. Uses 5 informative indicators:
       DC_MAX, DC_MAX_WS, DC_AMP, DC_AMP_WS, HF_STD

  3. Explores k=3..15 with silhouette, runs k-means and hierarchical
     at best-k, k=5 (to match 5 SR classes), and k=10 (J&P's 10-class
     scheme).

  4. Cluster profiling: mean indicator values, geographic centroid,
     latitude/longitude spread, and physical interpretation per cluster.

  5. LDA with 5-fold cross-validation for honest accuracy.

  6. Publication-quality comparison plots:
       - Side-by-side maps (temporal vs spatial classification)
       - Agreement map (where do they agree / disagree?)
       - Heatmap of cluster-vs-SR proportions (normalized)
       - Cluster profiles (radar / parallel-coordinate)
       - Per-indicator correlation with SR derivative

Input:
  joly_peuch_indicators_{SPECIES}.csv   (from compute_joly_peuch_indicators.py)
  slope_stats_box04.csv                 (from aggregate_slope_stats.py)

Output:
  station_classification_v2_{SPECIES}.csv
  Plots in joly_peuch_plots_v2/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec
import time
import datetime as dt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (adjusted_rand_score, silhouette_score,
                             silhouette_samples, confusion_matrix)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# USER SETTINGS
# ============================================================
SPECIES = "SO2"
SUMMARY_DIR = Path(f"/mnt/store01/agkiokas/CAMS/fit_outputs/summary/{SPECIES}")
OUT_DIR     = Path(f"/mnt/store01/agkiokas/CAMS/fit_outputs/summary/{SPECIES}")
PLOT_DIR    = OUT_DIR / "joly_peuch_plots_v2"
PLOT_DIR.mkdir(parents=True, exist_ok=True)



# Input files
JP_FILE    = SUMMARY_DIR / f"joly_peuch_indicators_{SPECIES}.csv"
SLOPE_FILE = SUMMARY_DIR / "slope_stats_box04.csv"

# The 5 INFORMATIVE J&P indicator columns (weekend effect dropped)
JP_COLS = ["DC_MAX", "DC_MAX_WS", "DC_AMP", "DC_AMP_WS", "HF_STD"]

# Dropped indicators (for documentation)
DROPPED_COLS = ["WE_MEAN", "WE_MAX", "WE_STD"]

# K-means scan range
K_RANGE = range(3, 16)

# SR derivative column in slope_stats_box04.csv
SR_VARIABLE = "cv_w"
SR_MODEL    = "logarithmic"
SR_GROUP    = "all"
SR_STAT     = "mean"

# Number of SR classes for quantile binning
N_SR_CLASSES = 5

# Plot settings
DPI = 150

# Readable short labels for the 5 indicators
INDICATOR_LABELS = {
    "DC_MAX":    "Diurnal\nMax",
    "DC_MAX_WS": "Diurnal Max\nW/S ratio",
    "DC_AMP":    "Diurnal\nAmplitude",
    "DC_AMP_WS": "Diurnal Amp\nW/S ratio",
    "HF_STD":    "High-freq\nStd",
}

# SR class names for plots
SR_CLASS_LABELS = {
    0: "Very high SR",
    1: "High SR",
    2: "Moderate SR",
    3: "Low SR",
    4: "Very low SR",
}


# ============================================================
# STEP 1: LOAD AND MERGE
# ============================================================

def load_data():
    """Load J&P indicators and slope stats, merge on station_id."""
    print("Step 1: Loading data")
    print("-" * 50)

    df_jp = pd.read_csv(JP_FILE)
    print(f"  J&P indicators: {len(df_jp)} stations")

    df_slope = pd.read_csv(SLOPE_FILE)
    print(f"  Slope stats:    {len(df_slope)} stations")

    # Find the SR derivative column
    sr_col = f"{SR_VARIABLE}_{SR_MODEL}_{SR_GROUP}_{SR_STAT}"
    if sr_col not in df_slope.columns:
        candidates = [c for c in df_slope.columns
                      if SR_VARIABLE in c and SR_GROUP in c and SR_STAT in c]
        print(f"  [WARN] '{sr_col}' not found. Candidates: {candidates}")
        if candidates:
            sr_col = candidates[0]
            print(f"  Using: {sr_col}")
        else:
            print("  [ERROR] No SR column found!")
            return None, None

    # Merge
    df = df_jp.merge(df_slope[["station_id", sr_col]],
                     on="station_id", how="inner")
    df = df.rename(columns={sr_col: "sr_derivative"})
    print(f"  Merged:         {len(df)} stations")

    # Drop NaN in the 5 indicators + sr_derivative
    n_before = len(df)
    df = df.dropna(subset=JP_COLS + ["sr_derivative"])
    print(f"  After dropna:   {len(df)} stations "
          f"(dropped {n_before - len(df)})")

    # Report on dropped weekend indicators
    for col in DROPPED_COLS:
        if col in df.columns:
            vals = df[col].dropna()
            print(f"  [DROPPED] {col}: mean={vals.mean():.6f}  "
                  f"std={vals.std():.6f}  (all ≈ 1.0, no info)")

    print()
    return df, sr_col


# ============================================================
# STEP 2: NORMALIZE
# ============================================================

def normalize_indicators(df):
    """Robust normalization: center by median, scale by IQR, clip 1st/99th."""
    print("Step 2: Normalizing 5 indicators")
    print("-" * 50)

    X_raw = df[JP_COLS].values.copy()

    medians = np.nanmedian(X_raw, axis=0)
    q25 = np.nanpercentile(X_raw, 25, axis=0)
    q75 = np.nanpercentile(X_raw, 75, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0

    X_norm = (X_raw - medians) / iqr

    for i, col in enumerate(JP_COLS):
        print(f"  {col:12s}  median={medians[i]:10.4f}  "
              f"IQR={iqr[i]:8.4f}  "
              f"range=[{np.nanmin(X_raw[:, i]):8.4f}, "
              f"{np.nanmax(X_raw[:, i]):8.4f}]")

    # Clip at 1st/99th percentile
    p01 = np.nanpercentile(X_norm, 1, axis=0)
    p99 = np.nanpercentile(X_norm, 99, axis=0)
    n_clipped = 0
    for j in range(X_norm.shape[1]):
        lo = X_norm[:, j] < p01[j]
        hi = X_norm[:, j] > p99[j]
        n_clipped += lo.sum() + hi.sum()
        X_norm[:, j] = np.clip(X_norm[:, j], p01[j], p99[j])
    print(f"\n  Clipped {n_clipped} values at 1st/99th percentile")

    for i, col in enumerate(JP_COLS):
        df[f"{col}_norm"] = X_norm[:, i]

    print()
    return X_norm, df


# ============================================================
# STEP 3: UNSUPERVISED CLUSTERING
# ============================================================

def unsupervised_clustering(X_norm, df):
    """K-means + hierarchical; silhouette scan; fixed k=5,10 too."""
    print("Step 3: Unsupervised clustering")
    print("-" * 50)

    # --- Silhouette scan ---
    sil_scores = []
    inertias = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_norm)
        sil = silhouette_score(X_norm, labels)
        sil_scores.append(sil)
        inertias.append(km.inertia_)

    best_k = list(K_RANGE)[np.argmax(sil_scores)]
    print(f"  Silhouette scores:")
    for k, s in zip(K_RANGE, sil_scores):
        marker = " ◄ best" if k == best_k else ""
        print(f"    k={k:2d}  sil={s:.4f}{marker}")

    # --- Plot: silhouette + elbow ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(list(K_RANGE), sil_scores, "o-", color="steelblue",
             label="Silhouette", lw=2)
    ax1.set_xlabel("Number of clusters (k)", fontsize=12)
    ax1.set_ylabel("Silhouette score", color="steelblue", fontsize=12)
    ax1.axvline(best_k, color="grey", ls="--", alpha=0.5,
                label=f"Best k={best_k}")
    ax2 = ax1.twinx()
    ax2.plot(list(K_RANGE), inertias, "s--", color="coral", alpha=0.7,
             label="Inertia", lw=1.5)
    ax2.set_ylabel("Inertia", color="coral", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.set_title(f"K-means cluster selection — {SPECIES} (5 indicators)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"silhouette_scan_{SPECIES}.png", dpi=DPI)
    plt.close(fig)

    # --- Run at selected k values ---
    k_values = sorted(set([best_k, 5, 10]))
    for k in k_values:
        # K-means
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        df[f"kmeans_{k}"] = km.fit_predict(X_norm)

        # Hierarchical (Ward)
        Z = linkage(X_norm, method="ward")
        df[f"hier_{k}"] = fcluster(Z, t=k, criterion="maxclust") - 1

        # Report
        km_sizes = df[f"kmeans_{k}"].value_counts().sort_index()
        hc_sizes = df[f"hier_{k}"].value_counts().sort_index()
        print(f"\n  k={k}:")
        print(f"    K-means:       {dict(km_sizes)}")
        print(f"    Hierarchical:  {dict(hc_sizes)}")

    # --- Dendrogram (truncated) ---
    Z = linkage(X_norm, method="ward")
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, truncate_mode="lastp", p=40, ax=ax,
               leaf_rotation=90, leaf_font_size=7,
               color_threshold=Z[-best_k + 1, 2] if best_k > 1 else 0)
    ax.set_title(f"Hierarchical clustering (Ward) — {SPECIES}", fontsize=13)
    ax.set_xlabel("Cluster (truncated)")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"dendrogram_{SPECIES}.png", dpi=DPI)
    plt.close(fig)

    print()
    return df, best_k


# ============================================================
# STEP 4: SPATIAL REPRESENTATIVENESS CLASSES
# ============================================================

def make_sr_classes(df, n_classes=N_SR_CLASSES):
    """Quantile-bin |SR derivative| into n_classes."""
    print("Step 4: Spatial representativeness classes")
    print("-" * 50)

    df["sr_abs"] = df["sr_derivative"].abs()

    df["sr_class"], bins = pd.qcut(df["sr_abs"], q=n_classes,
                                   labels=False, retbins=True,
                                   duplicates="drop")

    actual_n = df["sr_class"].nunique()
    print(f"  {actual_n} classes from |derivative| quantiles:")
    for c in sorted(df["sr_class"].unique()):
        n = (df["sr_class"] == c).sum()
        label = SR_CLASS_LABELS.get(c, "")
        print(f"    Class {c} ({label:14s}): n={n:4d}  "
              f"|deriv| ∈ [{bins[c]:.6f}, {bins[c+1]:.6f}]")

    print()
    return df


# ============================================================
# STEP 5: LDA WITH CROSS-VALIDATION
# ============================================================

def lda_classification(X_norm, df):
    """LDA with SR classes as labels; 5-fold CV for honest accuracy."""
    print("Step 5: LDA classification (with 5-fold CV)")
    print("-" * 50)

    y = df["sr_class"].values
    n_classes = len(np.unique(y))

    # Fit LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_norm, y)
    X_lda = lda.transform(X_norm)

    n_comp = X_lda.shape[1]
    print(f"  {n_classes} classes → {n_comp} discriminant axes")
    print(f"  Explained variance ratio: "
          f"{np.round(lda.explained_variance_ratio_, 4)}")

    # Training accuracy
    y_pred = lda.predict(X_norm)
    train_acc = (y_pred == y).mean()

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lda, X_norm, y, cv=cv, scoring="accuracy")
    print(f"  Training accuracy:  {train_acc:.3f}")
    print(f"  5-fold CV accuracy: {cv_scores.mean():.3f} "
          f"± {cv_scores.std():.3f}  "
          f"({np.round(cv_scores, 3)})")

    # LDA coefficients
    print(f"\n  LDA coefficients (importance for separating SR classes):")
    for axis in range(min(n_comp, 2)):
        coefs = lda.scalings_[:, axis]
        print(f"    LD{axis+1}: ", end="")
        for i, col in enumerate(JP_COLS):
            print(f" {col}={coefs[i]:+.3f}", end="")
        print()

    # Store projections
    for j in range(min(n_comp, 4)):
        df[f"LDA_{j+1}"] = X_lda[:, j]

    # Percentile-based classes on LD1 (10 bins like J&P)
    ld1 = X_lda[:, 0]
    cuts = np.percentile(ld1, np.linspace(0, 100, 11))
    df["lda_class"] = np.digitize(ld1, cuts[1:-1])

    # Also create a 5-class LDA for direct comparison with SR classes
    cuts5 = np.percentile(ld1, np.linspace(0, 100, 6))
    df["lda_class5"] = np.digitize(ld1, cuts5[1:-1])

    # LDA predicted class (from the model itself)
    df["lda_pred"] = y_pred

    print(f"\n  LDA 10-class sizes: "
          f"{dict(sorted(df['lda_class'].value_counts().items()))}")
    print(f"  LDA 5-class sizes:  "
          f"{dict(sorted(df['lda_class5'].value_counts().items()))}")
    print()
    return df, lda


# ============================================================
# STEP 6: CLUSTER PROFILING
# ============================================================

def profile_clusters(df, best_k):
    """Physical interpretation of each cluster."""
    print("Step 6: Cluster profiling")
    print("-" * 50)

    km_col = f"kmeans_{best_k}"

    # Mean raw (un-normalized) indicator values per cluster
    print(f"\n  Mean indicator values per K-means cluster (k={best_k}):")
    print(f"  {'Cluster':>8s}  {'n':>5s}  ", end="")
    for col in JP_COLS:
        print(f"{col:>12s}", end="")
    print(f"  {'lat':>7s}  {'lon':>7s}  {'|SR_deriv|':>10s}")

    for c in sorted(df[km_col].unique()):
        mask = df[km_col] == c
        n = mask.sum()
        print(f"  {c:>8d}  {n:>5d}  ", end="")
        for col in JP_COLS:
            print(f"{df.loc[mask, col].mean():>12.4f}", end="")
        print(f"  {df.loc[mask, 'station_lat'].mean():>7.2f}"
              f"  {df.loc[mask, 'station_lon'].mean():>7.2f}"
              f"  {df.loc[mask, 'sr_abs'].mean():>10.6f}")

    # Mean normalized indicators per cluster (for the heatmap)
    norm_cols = [f"{c}_norm" for c in JP_COLS]
    cluster_means_norm = df.groupby(km_col)[norm_cols].mean()
    cluster_means_norm.columns = JP_COLS

    # Also compute mean SR class per cluster
    sr_by_cluster = df.groupby(km_col)["sr_class"].mean()
    print(f"\n  Mean SR class per cluster (0=most repr, 4=least):")
    for c in sorted(df[km_col].unique()):
        print(f"    Cluster {c}: mean SR class = {sr_by_cluster[c]:.2f}")

    # --- Profile by SR class ---
    print(f"\n  Mean indicator values per SR class:")
    print(f"  {'SR_class':>8s}  {'n':>5s}  ", end="")
    for col in JP_COLS:
        print(f"{col:>12s}", end="")
    print()
    for c in sorted(df["sr_class"].unique()):
        mask = df["sr_class"] == c
        n = mask.sum()
        print(f"  {c:>8d}  {n:>5d}  ", end="")
        for col in JP_COLS:
            print(f"{df.loc[mask, col].mean():>12.4f}", end="")
        print()

    print()
    return cluster_means_norm


# ============================================================
# STEP 7: COMPARISON METRICS
# ============================================================

def compare_classifications(df, best_k):
    """ARI, Spearman ρ, contingency tables."""
    print("Step 7: Comparing classifications")
    print("-" * 50)

    km_col = f"kmeans_{best_k}"
    hc_col = f"hier_{best_k}"

    comparisons = [
        (km_col, "sr_class", f"K-means(k={best_k}) vs SR"),
        (hc_col, "sr_class", f"Hierarchical(k={best_k}) vs SR"),
        ("kmeans_5", "sr_class", "K-means(k=5) vs SR"),
        ("hier_5", "sr_class", "Hierarchical(k=5) vs SR"),
        ("lda_class5", "sr_class", "LDA-5-class vs SR"),
        ("lda_pred", "sr_class", "LDA-predicted vs SR"),
        (km_col, hc_col, f"K-means vs Hierarchical (k={best_k})"),
        ("kmeans_5", "lda_class5", "K-means(5) vs LDA-5-class"),
    ]

    for col_a, col_b, label in comparisons:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        ari = adjusted_rand_score(df[col_a], df[col_b])
        rho, pval = spearmanr(df[col_a], df[col_b])
        print(f"  {label}:")
        print(f"    ARI = {ari:.4f}    Spearman ρ = {rho:+.4f} "
              f"(p={pval:.2e})")

    # Contingency: k=5 vs SR (both 5 classes → direct comparison)
    print(f"\n  Contingency table: kmeans_5 vs sr_class")
    ct = pd.crosstab(df["kmeans_5"], df["sr_class"],
                     rownames=["Temporal (k=5)"],
                     colnames=["Spatial (SR)"])
    print(ct.to_string())
    ct.to_csv(OUT_DIR / f"contingency_km5_vs_sr_{SPECIES}.csv")

    # Contingency: best_k vs SR
    if best_k != 5:
        print(f"\n  Contingency table: kmeans_{best_k} vs sr_class")
        ct2 = pd.crosstab(df[km_col], df["sr_class"],
                          rownames=[f"Temporal (k={best_k})"],
                          colnames=["Spatial (SR)"])
        print(ct2.to_string())
        ct2.to_csv(OUT_DIR / f"contingency_km{best_k}_vs_sr_{SPECIES}.csv")

    # Per-indicator Spearman with |SR derivative|
    print(f"\n  Spearman ρ: each indicator vs |SR derivative|")
    for col in JP_COLS:
        rho, pval = spearmanr(df[col], df["sr_abs"])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else \
              "*" if pval < 0.05 else "n.s."
        print(f"    {col:12s}  ρ = {rho:+.4f}  (p={pval:.2e})  {sig}")

    print()


# ============================================================
# STEP 8: PUBLICATION PLOTS
# ============================================================

def make_plots(df, best_k, cluster_means_norm):
    """All diagnostic and publication-quality plots."""
    print("Step 8: Plots")
    print("-" * 50)

    km_col = f"kmeans_{best_k}"

    # === 8a. SIDE-BY-SIDE MAPS: temporal clusters vs SR classes ===
    for k in sorted(set([best_k, 5])):
        km_c = f"kmeans_{k}"
        if km_c not in df.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Left: temporal clusters
        n_clusters = df[km_c].nunique()
        cmap_km = plt.cm.get_cmap("tab10", n_clusters)
        sc1 = axes[0].scatter(df["station_lon"], df["station_lat"],
                              c=df[km_c], cmap=cmap_km, s=18, alpha=0.7,
                              edgecolors="k", linewidths=0.2,
                              vmin=-0.5, vmax=n_clusters - 0.5)
        axes[0].set_title(f"Temporal clusters (K-means, k={k})", fontsize=13)
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        axes[0].set_aspect("equal")
        plt.colorbar(sc1, ax=axes[0], label="Cluster", pad=0.08,
                     ticks=range(n_clusters))

        # Right: SR classes
        cmap_sr = plt.cm.get_cmap("RdYlGn", N_SR_CLASSES)
        sc2 = axes[1].scatter(df["station_lon"], df["station_lat"],
                              c=df["sr_class"], cmap=cmap_sr, s=18,
                              alpha=0.7, edgecolors="k", linewidths=0.2,
                              vmin=-0.5, vmax=N_SR_CLASSES - 0.5)
        axes[1].set_title("Spatial representativeness classes", fontsize=13)
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        axes[1].set_aspect("equal")
        cbar = plt.colorbar(sc2, ax=axes[1], label="SR class", pad=0.08,
                            ticks=range(N_SR_CLASSES))
        cbar.ax.set_yticklabels(
            [SR_CLASS_LABELS.get(i, str(i)) for i in range(N_SR_CLASSES)],
            fontsize=8)

        fig.suptitle(f"Temporal vs Spatial classification — {SPECIES}",
                     fontsize=15, y=1.01)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"map_sidebyside_k{k}_{SPECIES}.png",
                    dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # === 8b. AGREEMENT MAP ===
    # For k=5: does temporal cluster agree with SR class?
    if "kmeans_5" in df.columns:
        # Reorder clusters to best match SR classes
        # (use the contingency matrix to find best alignment)
        ct = pd.crosstab(df["kmeans_5"], df["sr_class"])
        # Greedy mapping: for each temporal cluster, assign to the SR class
        # with the most overlap
        mapping = {}
        used = set()
        for tc in ct.index:
            row = ct.loc[tc].copy()
            for sc in sorted(used):
                row[sc] = -1  # don't reuse
            mapping[tc] = row.idxmax()
            used.add(mapping[tc])

        df["kmeans_5_mapped"] = df["kmeans_5"].map(mapping)
        df["agree_5"] = (df["kmeans_5_mapped"] == df["sr_class"]).astype(int)
        agree_pct = df["agree_5"].mean() * 100

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ["#d73027", "#1a9850"]  # red=disagree, green=agree
        cmap_ag = ListedColormap(colors)
        sc = ax.scatter(df["station_lon"], df["station_lat"],
                        c=df["agree_5"], cmap=cmap_ag, s=18, alpha=0.7,
                        edgecolors="k", linewidths=0.2, vmin=0, vmax=1)
        ax.set_title(f"Agreement: temporal (k=5) vs spatial classification\n"
                     f"({agree_pct:.1f}% agree) — {SPECIES}", fontsize=13)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        cbar = plt.colorbar(sc, ax=ax, ticks=[0, 1], pad=0.08)
        cbar.ax.set_yticklabels(["Disagree", "Agree"])
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"map_agreement_k5_{SPECIES}.png", dpi=DPI)
        plt.close(fig)
        print(f"  Agreement (k=5 mapped): {agree_pct:.1f}%")

    # === 8c. HEATMAP: normalized contingency (proportions) ===
    for k in sorted(set([best_k, 5])):
        km_c = f"kmeans_{k}"
        if km_c not in df.columns:
            continue

        ct = pd.crosstab(df[km_c], df["sr_class"], normalize="index")

        fig, ax = plt.subplots(figsize=(8, max(4, k * 0.6 + 1)))
        im = ax.imshow(ct.values, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=0.5)
        ax.set_xticks(range(N_SR_CLASSES))
        ax.set_xticklabels([SR_CLASS_LABELS.get(i, str(i))
                            for i in range(N_SR_CLASSES)],
                           rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(k))
        ax.set_yticklabels([f"Cluster {i}" for i in range(k)])
        ax.set_xlabel("Spatial representativeness class")
        ax.set_ylabel("Temporal cluster")
        ax.set_title(f"Proportion of SR class within each temporal cluster\n"
                     f"K-means k={k} — {SPECIES}", fontsize=12)
        # Annotate cells
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                val = ct.values[i, j]
                color = "white" if val > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color)
        plt.colorbar(im, ax=ax, label="Proportion", pad=0.08)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"heatmap_proportion_k{k}_{SPECIES}.png",
                    dpi=DPI)
        plt.close(fig)

    # === 8d. CLUSTER PROFILE: parallel coordinates ===
    for k in sorted(set([best_k, 5])):
        km_c = f"kmeans_{k}"
        if km_c not in df.columns:
            continue
        norm_cols = [f"{c}_norm" for c in JP_COLS]
        means = df.groupby(km_c)[norm_cols].mean()
        means.columns = [INDICATOR_LABELS.get(c, c) for c in JP_COLS]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(JP_COLS))
        cmap_k = plt.cm.get_cmap("tab10", k)
        for i, (idx, row) in enumerate(means.iterrows()):
            n_in = (df[km_c] == idx).sum()
            ax.plot(x, row.values, "o-", color=cmap_k(i), lw=2.5,
                    markersize=8, label=f"Cluster {idx} (n={n_in})")
        ax.axhline(0, color="grey", ls="--", alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(means.columns, fontsize=10)
        ax.set_ylabel("Normalized indicator value", fontsize=11)
        ax.set_title(f"Cluster profiles (K-means k={k}) — {SPECIES}",
                     fontsize=13)
        ax.legend(fontsize=9, loc="best")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"parallel_coords_k{k}_{SPECIES}.png", dpi=DPI)
        plt.close(fig)

    # === 8e. BOXPLOTS: each indicator by SR class ===
    fig, axes = plt.subplots(1, len(JP_COLS), figsize=(3.5 * len(JP_COLS), 5))
    for i, col in enumerate(JP_COLS):
        ax = axes[i]
        data = [df.loc[df["sr_class"] == c, col].dropna().values
                for c in range(N_SR_CLASSES)]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                        widths=0.55)
        colors_sr = plt.cm.RdYlGn(np.linspace(0.8, 0.15, N_SR_CLASSES))
        for patch, clr in zip(bp["boxes"], colors_sr):
            patch.set_facecolor(clr)
            patch.set_alpha(0.7)
        ax.set_xticklabels(range(N_SR_CLASSES), fontsize=9)
        ax.set_xlabel("SR class")
        ax.set_title(INDICATOR_LABELS.get(col, col), fontsize=10)
        if i == 0:
            ax.set_ylabel("Value")
    fig.suptitle(f"Indicator distributions by SR class — {SPECIES}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"boxplots_by_sr_{SPECIES}.png", dpi=DPI)
    plt.close(fig)

    # === 8f. LDA SCATTER: LD1 vs LD2, coloured by SR class and by cluster ===
    if "LDA_1" in df.columns and "LDA_2" in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # SR class
        sc = axes[0].scatter(df["LDA_1"], df["LDA_2"],
                             c=df["sr_class"], cmap="RdYlGn",
                             s=10, alpha=0.5)
        axes[0].set_xlabel("LD1")
        axes[0].set_ylabel("LD2")
        axes[0].set_title("Coloured by SR class")
        plt.colorbar(sc, ax=axes[0], label="SR class")

        # K-means best_k
        sc2 = axes[1].scatter(df["LDA_1"], df["LDA_2"],
                              c=df[f"kmeans_{best_k}"],
                              cmap="tab10", s=10, alpha=0.5)
        axes[1].set_xlabel("LD1")
        axes[1].set_ylabel("LD2")
        axes[1].set_title(f"Coloured by K-means (k={best_k})")
        plt.colorbar(sc2, ax=axes[1], label="Cluster")

        # K-means 5
        if "kmeans_5" in df.columns and best_k != 5:
            sc3 = axes[2].scatter(df["LDA_1"], df["LDA_2"],
                                  c=df["kmeans_5"],
                                  cmap="tab10", s=10, alpha=0.5)
            axes[2].set_xlabel("LD1")
            axes[2].set_ylabel("LD2")
            axes[2].set_title("Coloured by K-means (k=5)")
            plt.colorbar(sc3, ax=axes[2], label="Cluster")
        else:
            axes[2].set_visible(False)

        fig.suptitle(f"LDA projection — {SPECIES}", fontsize=14)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"lda_scatter_{SPECIES}.png", dpi=DPI)
        plt.close(fig)

    # === 8g. SCATTER: LD1 vs |SR derivative| ===
    if "LDA_1" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(df["LDA_1"], df["sr_abs"],
                        c=df["sr_class"], cmap="RdYlGn", s=12, alpha=0.5)
        rho, pval = spearmanr(df["LDA_1"], df["sr_abs"])
        ax.set_xlabel("LD1 (temporal fingerprint axis)", fontsize=12)
        ax.set_ylabel("|SR derivative| (spatial representativeness)",
                      fontsize=12)
        ax.set_title(f"Temporal vs spatial signal — ρ = {rho:+.3f} "
                     f"(p = {pval:.2e}) — {SPECIES}", fontsize=13)
        plt.colorbar(sc, ax=ax, label="SR class")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"ld1_vs_sr_deriv_{SPECIES}.png", dpi=DPI)
        plt.close(fig)

    # === 8h. HEATMAP: cluster mean normalized indicators ===
    fig, ax = plt.subplots(figsize=(10, max(4, best_k * 0.6 + 1)))
    labels = [INDICATOR_LABELS.get(c, c) for c in JP_COLS]
    im = ax.imshow(cluster_means_norm.values, aspect="auto", cmap="RdBu_r",
                   vmin=-2, vmax=2)
    ax.set_xticks(range(len(JP_COLS)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(cluster_means_norm)))
    n_each = [f"(n={(df[km_col]==c).sum()})" for c in cluster_means_norm.index]
    ax.set_yticklabels([f"Cluster {c} {ne}"
                        for c, ne in zip(cluster_means_norm.index, n_each)])
    for i in range(cluster_means_norm.shape[0]):
        for j in range(cluster_means_norm.shape[1]):
            val = cluster_means_norm.values[i, j]
            color = "white" if abs(val) > 1.2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_title(f"Cluster fingerprints (normalized mean) — "
                 f"K-means k={best_k} — {SPECIES}", fontsize=13)
    plt.colorbar(im, ax=ax, label="Normalized value")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"heatmap_fingerprints_k{best_k}_{SPECIES}.png",
                dpi=DPI)
    plt.close(fig)

    print(f"  Saved all plots to {PLOT_DIR}")
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("classify_joly_peuch_v2.py")
    print("=" * 60)
    print(f"Start:   {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"SPECIES: {SPECIES}")
    print(f"Using {len(JP_COLS)} indicators: {JP_COLS}")
    print(f"Dropped: {DROPPED_COLS} (no weekend effect in GEOS-5 NR)")
    print()

    # Steps 1-8
    df, sr_col = load_data()
    if df is None:
        return

    X_norm, df = normalize_indicators(df)
    df, best_k = unsupervised_clustering(X_norm, df)

    km_col = f"kmeans_{best_k}"   # used in plots

    df = make_sr_classes(df)
    df, lda = lda_classification(X_norm, df)
    cluster_means_norm = profile_clusters(df, best_k)
    compare_classifications(df, best_k)
    make_plots(df, best_k, cluster_means_norm)

    # Save
    out_path = OUT_DIR / f"station_classification_v2_{SPECIES}.csv"
    df.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"Saved:   {out_path}")
    print(f"Done in  {elapsed:.1f} s")


if __name__ == "__main__":
    main()