#%%
"""
plot_global_model_share.py
==========================
Horizontal stacked bar charts showing the percentage share of each model
across all criteria, with one panel per variable.

Two plot types:
  Type 1 — Global (station × timestep) level
            Data source: global_counts.csv
  Type 2 — Station-level modal winner
            Data source: station_summary.csv

Outputs (written to OUTPUT_DIR):
  When plotting both variables:
    global_model_share_type1.png
    global_model_share_type2.png
  When plotting a single variable:
    global_model_share_type1_<var>.png
    global_model_share_type2_<var>.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# ============================================================
# CONFIGURATION
# ============================================================

SUMMARY_DIR = Path("/mnt/store01/agkiokas/CAMS/fit_outputs/summary")
COUNTS_CSV  = SUMMARY_DIR / "global_counts.csv"
STATION_CSV = SUMMARY_DIR / "station_summary.csv"
OUTPUT_DIR  = SUMMARY_DIR / "plots"

# --- Which variables to plot ---
# Use both:    PLOT_VARIABLES = ["ratio", "cv_w"]
# Ratio only:  PLOT_VARIABLES = ["ratio"]
# CV only:     PLOT_VARIABLES = ["cv_w"]
#PLOT_VARIABLES = ["ratio", "cv_w"]
PLOT_VARIABLES=["cv_w"]
MODELS    = ["linear", "quadratic", "cubic", "logarithmic",
             "exponential", "power", "saturating"]
CRITERIA  = ["r2", "adj_r2", "aic", "aicc", "bic"]

VAR_LABELS  = {"ratio": "Concentration Ratio Cx/Co", "cv_w": " CV (%)"}
CRIT_LABELS = {
    "r2":     r"$R^2$",
    "adj_r2": r"Adj-$R^2$",
    "aic":    "AIC",
    "aicc":   "AICc",
    "bic":    "BIC",
}
MODEL_LABELS = {
    "linear":      "Linear",
    "quadratic":   "Quadratic",
    "cubic":       "Cubic",
    "logarithmic": "Logarithmic",
    "exponential": "Exponential",
    "power":       "Power-law",
    "saturating":  "Saturating",
}
MODEL_COLORS = {
    "linear":      "#4C72B0",
    "quadratic":   "#DD8452",
    "cubic":       "#55A868",
    "logarithmic": "#C44E52",
    "exponential": "#8172B2",
    "power":       "#937860",
    "saturating":  "#DA8BC3",
}

# --- Figure aesthetics ---
FIG_W      = 14
FIG_H      = 7
DPI        = 150
BAR_HEIGHT = 0.55
FONT_SIZE  = 11
TITLE_SIZE = 14
LABEL_MIN_PCT = 10.0   # hide percentage label if segment < this %


# ============================================================
# TYPE 1 — from global_counts.csv
# ============================================================

def _build_type1_data(df_counts: pd.DataFrame) -> dict:
    """
    Returns {var: DataFrame} where each DataFrame has
    rows = criteria, columns = models, values = percentages.
    Also returns the total valid pairs per criterion (should be constant).
    """
    data = {}
    totals = {}
    for var in PLOT_VARIABLES:
        rows = []
        for crit in CRITERIA:
            sub = df_counts[(df_counts["var"] == var) &
                            (df_counts["criterion"] == crit)]
            counts = {row["model"]: int(row["count"])
                      for _, row in sub.iterrows()}
            rows.append(counts)
        df = pd.DataFrame(rows, index=CRITERIA, columns=MODELS).fillna(0)
        row_totals = df.sum(axis=1)
        totals[var] = int(row_totals.iloc[0])  # same for all criteria
        # convert to percentages
        pct = df.div(row_totals, axis=0) * 100.0
        data[var] = pct
    return data, totals


def plot_type1(df_counts: pd.DataFrame):
    """Global model share — station × timestep level."""
    data, totals = _build_type1_data(df_counts)
    n_vars = len(PLOT_VARIABLES)

    n_stations = "?"
    if STATION_CSV.exists():
        n_stations = f"{len(pd.read_csv(STATION_CSV, usecols=['station_id'])):,}"

    total_pairs = totals[PLOT_VARIABLES[0]]

    fig_w = FIG_W if n_vars == 2 else FIG_W / 2
    fig, axes = plt.subplots(1, n_vars, figsize=(fig_w, FIG_H),
                             sharey=True, squeeze=False)
    axes = axes.ravel()

    fig.suptitle(
        f"Global model share (% of station × timestep wins) across all criteria\n"
        f"{n_stations} stations  |  {total_pairs:,} valid pairs per criterion",
        fontsize=TITLE_SIZE, fontweight="bold", y=0.97,
    )

    for ax, var in zip(axes, PLOT_VARIABLES):
        pct = data[var]
        _draw_horizontal_stacked(ax, pct, var)

    # Shared legend at bottom
    handles = [mpatches.Patch(facecolor=MODEL_COLORS[m],
                              label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(MODELS), fontsize=FONT_SIZE,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    suffix = f"_{PLOT_VARIABLES[0]}" if n_vars == 1 else ""
    out = OUTPUT_DIR / f"global_model_share_type1{suffix}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


# ============================================================
# TYPE 2 — from station_summary.csv
# ============================================================

def _build_type2_data(df_stations: pd.DataFrame) -> dict:
    """
    Returns {var: DataFrame} where each DataFrame has
    rows = criteria, columns = models, values = percentages.
    Counts the modal winner per station.
    """
    data = {}
    for var in PLOT_VARIABLES:
        rows = []
        for crit in CRITERIA:
            col = f"{var}_{crit}_winner"
            if col not in df_stations.columns:
                rows.append({m: 0 for m in MODELS})
                continue
            vc = df_stations[col].dropna().value_counts()
            counts = {m: int(vc.get(m, 0)) for m in MODELS}
            rows.append(counts)
        df = pd.DataFrame(rows, index=CRITERIA, columns=MODELS).fillna(0)
        row_totals = df.sum(axis=1)
        pct = df.div(row_totals, axis=0) * 100.0
        data[var] = pct
    return data


def plot_type2(df_stations: pd.DataFrame):
    """Global model share — station-level modal winner."""
    n_stations = len(df_stations)
    data = _build_type2_data(df_stations)
    n_vars = len(PLOT_VARIABLES)

    fig_w = FIG_W if n_vars == 2 else FIG_W / 2
    fig, axes = plt.subplots(1, n_vars, figsize=(fig_w, FIG_H),
                             sharey=True, squeeze=False)
    axes = axes.ravel()

    fig.suptitle(
        f"Global model share (% of station-level modal winners) across all criteria\n"
        f"{n_stations:,} stations",
        fontsize=TITLE_SIZE, fontweight="bold", y=0.97,
    )

    for ax, var in zip(axes, PLOT_VARIABLES):
        pct = data[var]
        _draw_horizontal_stacked(ax, pct, var)

    # Shared legend at bottom
    handles = [mpatches.Patch(facecolor=MODEL_COLORS[m],
                              label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(MODELS), fontsize=FONT_SIZE,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    suffix = f"_{PLOT_VARIABLES[0]}" if n_vars == 1 else ""
    out = OUTPUT_DIR / f"global_model_share_type2{suffix}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


# ============================================================
# SHARED DRAWING HELPER
# ============================================================

def _draw_horizontal_stacked(ax, pct_df: pd.DataFrame, var: str):
    """
    Draw one horizontal stacked bar subplot.
    pct_df: rows = criteria, columns = models, values = percentages.
    """
    y_pos = np.arange(len(CRITERIA))
    crit_labels = [CRIT_LABELS[c] for c in CRITERIA]

    lefts = np.zeros(len(CRITERIA))

    for model in MODELS:
        widths = pct_df[model].values
        color  = MODEL_COLORS[model]
        ax.barh(y_pos, widths, left=lefts, height=BAR_HEIGHT,
                color=color, edgecolor="white", linewidth=0.5)

        # Percentage labels inside bars (skip if too narrow)
        for i, (w, l) in enumerate(zip(widths, lefts)):
            if w >= LABEL_MIN_PCT:
                ax.text(l + w / 2, y_pos[i], f"{w:.1f}%",
                        ha="center", va="center",
                        fontsize=FONT_SIZE - 2, fontweight="bold",
                        color="white")
        lefts += widths

    ax.set_yticks(y_pos)
    ax.set_yticklabels(crit_labels, fontsize=FONT_SIZE)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share (%)", fontsize=FONT_SIZE)
    ax.set_title(VAR_LABELS[var], fontsize=TITLE_SIZE - 1, fontweight="bold")
    ax.invert_yaxis()   # R² at top
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Type 1 — global counts
    if COUNTS_CSV.exists():
        df_counts = pd.read_csv(COUNTS_CSV)
        print(f"Loaded global_counts: {len(df_counts)} rows")
        plot_type1(df_counts)
    else:
        print(f"WARNING: {COUNTS_CSV} not found — skipping Type 1")

    # Type 2 — station modal winners
    if STATION_CSV.exists():
        df_stations = pd.read_csv(STATION_CSV)
        print(f"Loaded station_summary: {len(df_stations)} stations")
        plot_type2(df_stations)
    else:
        print(f"WARNING: {STATION_CSV} not found — skipping Type 2")


if __name__ == "__main__":
    main()
# %%
# %%