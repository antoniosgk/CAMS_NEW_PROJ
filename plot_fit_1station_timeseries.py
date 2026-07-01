#%%
"""
Plot the area-based multi-model fit results produced by area_fit_backfill.py.

Tier-1 figures, for ONE station:

1. Slope time-series of the 7 models, with a model-specific companion
   parameter on the right y-axis.
     PLOT_LAYOUT = "stacked"   -> 7 figures, each with 3 stacked subplots
                                  (one per variable: ratio, cv_w, mean_w)
     PLOT_LAYOUT = "separate"  -> 21 figures, one per (model, variable)
2. Best-model bar charts: 5 selection criteria x 3 variables = 15 plots.
3. AICc time-series: all 7 models on one axes, one figure per variable = 3 plots.
4. Lambda (saturating length scale, km^2) time-series, one figure per variable = 3 plots.

Total: 28 (stacked) or 42 (separate) figures.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import time
import datetime as dt

# ============================================================
# USER SETTINGS
# ============================================================

IN_DIR  = Path("/mnt/store01/agkiokas/CAMS/fit_outputs")   # where area_fit_backfill writes
OUT_DIR = Path("/mnt/store01/agkiokas/CAMS/area_fit_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES    = "O3"
STATION_ID = "1002A"
CSV_FILENAME = f"{STATION_ID}_{SPECIES}_area_fits.csv"

# ------------------------------------------------------------
# Plot 1 — slope time-series layout
#   "stacked"  -> 7 figures, each with 3 vertically stacked subplots
#                 (one per variable). Shared x-axis within each figure.
#   "separate" -> 21 figures, one per (model, variable) pair.
# ------------------------------------------------------------
PLOT_LAYOUT = "stacked"
# PLOT_LAYOUT = "separate"

# ------------------------------------------------------------
# Time range
# ------------------------------------------------------------
PLOT_FULL_PERIOD = True
START_DATE = "2005-05-20"
END_DATE   = "2005-06-20"

# ------------------------------------------------------------
# Lambda (saturating length scale) display
# ------------------------------------------------------------
# Saturating fits can produce physically meaningless λ values when the curve
# does not actually saturate within the area range (the fit degenerates toward
# linear, pushing λ → ∞). Choose how to handle this on the λ time-series:
#
#   "clip"   -> y-axis capped at LAMBDA_MAX_KM2; values above are off-screen
#               but remain in the CSV. Clearest for presentations.
#   "log"    -> y-axis on a log scale; extreme values compressed but visible.
#               Recommended for publications.
#   "filter" -> drop timesteps where λ > LAMBDA_MAX_KM2 entirely; the figure
#               title reports the percentage dropped.
# ------------------------------------------------------------
LAMBDA_DISPLAY  = "clip"        # "clip" | "log" | "filter"
LAMBDA_MAX_KM2  = 50_000.0      # threshold for "clip" and "filter"

# ------------------------------------------------------------
# Variables and selection criteria
# ------------------------------------------------------------
VARIABLES = ["ratio", "cv_w"]
CRITERIA  = ["r2", "adj_r2", "aic", "aicc", "bic"]

# Minimum sectors used in a fit for the timestep to count in the plots.
# Same filter as the previous plotting script.
MIN_N_SECTORS = 8

# ------------------------------------------------------------
# Figure settings
# ------------------------------------------------------------
DPI         = 300
TITLE_SIZE  = 15
LABEL_SIZE  = 12
TICK_SIZE   = 10
LEGEND_SIZE = 9


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "figure.dpi":         120,
    "savefig.dpi":        DPI,
    "font.size":          11,
    "axes.titlesize":     TITLE_SIZE,
    "axes.labelsize":     LABEL_SIZE,
    "xtick.labelsize":    TICK_SIZE,
    "ytick.labelsize":    TICK_SIZE,
    "legend.fontsize":    LEGEND_SIZE,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "-",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


# ============================================================
# MODEL DEFINITIONS
# ============================================================
# Each model has a "companion" parameter that, together with the slope,
# best describes the shape of the fit. See discussion for rationale.

MODELS = ["linear", "quadratic", "cubic", "logarithmic",
          "exponential", "power", "saturating"]

MODEL_LABELS = {
    "linear":      "Linear",
    "quadratic":   "Quadratic",
    "cubic":       "Cubic",
    "logarithmic": "Logarithmic",
    "exponential": "Exponential",
    "power":       "Power-law",
    "saturating":  "Saturating",
}

# 7-colour palette tuned to be distinguishable on screen and in print.
MODEL_COLORS = {
    "linear":      "#1f77b4",   # blue
    "quadratic":   "#ff7f0e",   # orange
    "cubic":       "#2ca02c",   # green
    "logarithmic": "#9467bd",   # purple
    "exponential": "#d62728",   # red
    "power":       "#8c564b",   # brown
    "saturating":  "#17becf",   # cyan
}

# (companion_param, human-readable name, units suffix)
# Maps the stored column suffix (_a/_b/_c/_d) to the parameter name we plot
# on the right y-axis, and a short label for the axis.
COMPANION = {
    "linear":      ("a", "intercept a",      "y = a + b·x"),
    "quadratic":   ("c", "curvature c",      "y = a + b·x + c·x²"),
    "cubic":       ("d", "cubic coeff. d",   "y = a + b·x + c·x² + d·x³"),
    "logarithmic": ("a", "intercept a",      "y = a + b·ln(x)"),
    "exponential": ("c", "rate c (1/km²)",   "y = a + b·exp(c·x)"),
    "power":       ("b", "exponent b",       "y = a·x^b"),
    "saturating":  ("c", "length scale λ (km²)", "y = a + b·(1 − exp(−x/λ))"),
}

# Companion-axis line color: a desaturated grey to keep emphasis on the slope.
COMPANION_COLOR = "#555555"
SLOPE_COLOR_FOR_MODEL = lambda m: MODEL_COLORS[m]

# Variable display names
VAR_LABELS = {"ratio": "ratio", "cv_w": "CV", "mean_w": "mean (ppb)"}


# ============================================================
# HELPERS
# ============================================================

def savefig(name):
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"Saved: {path.name}")


def detect_datetime_column(df):
    """Find the first usable timestamp column and produce a plot_datetime."""
    if "datetime" in df.columns:
        df["plot_datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "timestamp" in df.columns:
        df["plot_datetime"] = pd.to_datetime(df["timestamp"].astype(str),
                                             format="%Y%m%d %H%M", errors="coerce")
        if df["plot_datetime"].isna().all():
            df["plot_datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "time_key" in df.columns:
        df["plot_datetime"] = pd.to_datetime(df["time_key"].astype(str),
                                             format="%Y%m%d_%H%M", errors="coerce")
    else:
        raise ValueError("No datetime/timestamp/time_key column found.")
    return df


def filter_by_n(df, variable, model):
    """Boolean mask: rows where the fit used at least MIN_N_SECTORS sectors."""
    n_col = f"{variable}_{model}_n"
    if n_col in df.columns:
        return df[n_col] >= MIN_N_SECTORS
    return pd.Series(True, index=df.index)


def nice_time_axis(ax):
    """Auto-locate major ticks based on the visible span."""
    locator   = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=0)


def clean(s):
    """Coerce to numeric, drop inf/nan."""
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_data():
    path = IN_DIR / CSV_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_csv(path)
    df = detect_datetime_column(df)
    df = df.dropna(subset=["plot_datetime"]).copy()
    df = df.sort_values("plot_datetime").copy()
    if not PLOT_FULL_PERIOD:
        df = df[(df["plot_datetime"] >= pd.to_datetime(START_DATE)) &
                (df["plot_datetime"] <= pd.to_datetime(END_DATE))].copy()
    print(f"Loaded {len(df):,} rows for {STATION_ID}, "
          f"range: {df['plot_datetime'].min()} -> {df['plot_datetime'].max()}")
    return df


# ============================================================
# PLOT 1: SLOPE + COMPANION TIME-SERIES
# ============================================================

def _plot_slope_panel(ax, df, variable, model):
    """
    Plot one (variable, model) panel:
      left y-axis  = slope of the fitted curve (avg slope across area range)
      right y-axis = the model-specific companion parameter
    The two series are drawn with distinct colours and explicitly labelled axes.
    """
    mask = filter_by_n(df, variable, model)
    sub  = df.loc[mask]

    slope_col = f"{variable}_{model}_slope"
    comp_key, comp_label,_ = COMPANION[model]
    comp_col  = f"{variable}_{model}_{comp_key}"

    if slope_col not in df.columns:
        ax.text(0.5, 0.5, f"missing: {slope_col}", ha="center", va="center",
                transform=ax.transAxes)
        return

    slope_color = SLOPE_COLOR_FOR_MODEL(model)
    slope_vals  = clean(sub[slope_col])
    ax.plot(sub["plot_datetime"], slope_vals,
            color=slope_color, linewidth=0.9, alpha=0.9,
            label=f"slope ({variable})")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)

    ax.set_ylabel(f"slope of {VAR_LABELS[variable]}  (per km²)",
                  color=slope_color)
    ax.tick_params(axis="y", colors=slope_color)
    ax.spines["left"].set_color(slope_color)

    # right y-axis: companion parameter
    ax2 = ax.twinx()
    if comp_col in df.columns:
        comp_vals = clean(sub[comp_col])
        ax2.plot(sub["plot_datetime"], comp_vals,
                 color=COMPANION_COLOR, linewidth=0.9, alpha=0.7,
                 label=comp_label)
    ax2.set_ylabel(comp_label, color=COMPANION_COLOR)
    ax2.tick_params(axis="y", colors=COMPANION_COLOR)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(COMPANION_COLOR)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)


def plot_slope_timeseries_stacked(df):
    """7 figures, one per model. Each has 3 subplots: ratio / cv_w / mean_w."""
    for model in MODELS:
        fig, axes = plt.subplots(
            len(VARIABLES), 1,
            figsize=(13, 9),
            sharex=True,
        )
        if len(VARIABLES) == 1:
            axes = [axes]

        for ax, var in zip(axes, VARIABLES):
            _plot_slope_panel(ax, df, var, model)
            ax.set_title(f"{VAR_LABELS[var]}",
                         loc="left", fontsize=LABEL_SIZE)
        nice_time_axis(axes[-1])

        fig.suptitle(
               f"{SPECIES} | {STATION_ID} | {MODEL_LABELS[model]} fit:  "
            f"{COMPANION[model][2]}\n"
              f"slope (left) + {COMPANION[model][1]} (right)",
               fontsize=TITLE_SIZE, y=1.00,
)
        savefig(f"slope_stacked_{STATION_ID}_{model}.png")


def plot_slope_timeseries_separate(df):
    """21 figures, one per (model, variable). Same internal layout."""
    for model in MODELS:
        for var in VARIABLES:
            fig, ax = plt.subplots(figsize=(13, 5))
            _plot_slope_panel(ax, df, var, model)
            nice_time_axis(ax)
            ax.set_title(
            f"{SPECIES} | {STATION_ID} | {MODEL_LABELS[model]} on "
            f"{VAR_LABELS[var]}:  {COMPANION[model][2]}\n"
            f"slope (left) + {COMPANION[model][1]} (right)",
            fontsize=TITLE_SIZE,
)
            savefig(f"slope_separate_{STATION_ID}_{model}_{var}.png")


def run_slope_timeseries(df):
    print("\n[1/4] Slope time-series plots")
    if PLOT_LAYOUT == "stacked":
        plot_slope_timeseries_stacked(df)
    elif PLOT_LAYOUT == "separate":
        plot_slope_timeseries_separate(df)
    else:
        raise ValueError(f"Invalid PLOT_LAYOUT: {PLOT_LAYOUT}")


# ============================================================
# PLOT 2: BEST-MODEL BAR CHARTS (one per criterion x variable)
# ============================================================

def _best_model_per_row(df, variable, criterion):
    """
    For each row, return the model with the BEST value of `criterion`.

    - R² and adjusted R² are MAXIMISED (higher is better).
    - AIC, AICc, BIC are MINIMISED (lower is better).
    Rows that fail the n filter for a given model are excluded for that model.
    """
    cols = {m: f"{variable}_{m}_{criterion}" for m in MODELS}
    have = all(c in df.columns for c in cols.values())
    if not have:
        return None

    frame = pd.DataFrame(index=df.index)
    for m in MODELS:
        vals = clean(df[cols[m]])
        mask = filter_by_n(df, variable, m)
        vals = vals.where(mask, np.nan)
        frame[m] = vals

    valid_any = frame.notna().any(axis=1)
    best = pd.Series(index=df.index, dtype=object)

    if criterion in ("r2", "adj_r2"):
        # higher is better
        best[valid_any] = frame.loc[valid_any].idxmax(axis=1)
    else:
        # lower is better (aic, aicc, bic)
        best[valid_any] = frame.loc[valid_any].idxmin(axis=1)

    return best


def plot_best_model_bar(df, variable, criterion):
    best = _best_model_per_row(df, variable, criterion)
    if best is None:
        print(f"[skip] missing {criterion} columns for {variable}")
        return

    counts = best.value_counts()
    total  = int(counts.sum())
    if total == 0:
        print(f"[skip] no valid best-model selections for {variable}/{criterion}")
        return

    heights = [counts.get(m, 0) for m in MODELS]
    pct     = [100.0 * h / total for h in heights]
    colors  = [MODEL_COLORS[m] for m in MODELS]
    labels  = [MODEL_LABELS[m] for m in MODELS]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, pct, color=colors, alpha=0.85, width=0.65)

    for bar, p in zip(bars, pct):
        if p > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, p + 0.6,
                    f"{p:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    crit_pretty = {"r2": "R²", "adj_r2": "adjusted R²",
                   "aic": "AIC", "aicc": "AICc", "bic": "BIC"}[criterion]
    direction   = ("highest" if criterion in ("r2", "adj_r2") else "lowest")

    ax.set_ylim(0, max(pct) * 1.15 if max(pct) > 0 else 1)
    ax.set_ylabel(f"% of timesteps where this model has the {direction} {crit_pretty}")
    ax.set_xlabel("Fitted model")
    ax.set_title(
        f"{SPECIES} | {STATION_ID} | {VAR_LABELS[variable]}  —  "
        f"best model by {crit_pretty}  ({total:,} timesteps)",
        fontsize=TITLE_SIZE,
    )
    plt.xticks(rotation=15, ha="right")
    savefig(f"barplot_best_{criterion}_{STATION_ID}_{variable}.png")


def run_best_model_bars(df):
    print("\n[2/4] Best-model bar charts")
    for var in VARIABLES:
        for crit in CRITERIA:
            plot_best_model_bar(df, var, crit)
# ============================================================
# PLOT 2b: DELTA DISTRIBUTION PLOTS (one per criterion x variable)
# ============================================================

def plot_delta_distribution(df, variable, criterion):
    """
    Boxplot of delta_{criterion} per model across all timesteps.

    Delta = {model}_{criterion} - min({criterion} across all 7 models).
    The winning model's delta is always 0; others show how far behind they are.

    Reference lines at delta = 2 and delta = 10 (Burnham & Anderson thresholds
    for information criteria) — not meaningful for r2 / adj_r2 which use a
    different scale, so they are suppressed for those two.
    """
    delta_cols = {m: f"{variable}_{m}_delta_{criterion}" for m in MODELS}
    have = [c for c in delta_cols.values() if c in df.columns]
    if not have:
        print(f"[skip] no delta_{criterion} columns for {variable}")
        return

    crit_pretty = {
        "r2": "R²", "adj_r2": "adj. R²",
        "aic": "AIC", "aicc": "AICc", "bic": "BIC",
    }.get(criterion, criterion.upper())

    data, labels, colors, means = [], [], [], []
    for m in MODELS:
        col = delta_cols[m]
        if col not in df.columns:
            continue
        mask = filter_by_n(df, variable, m)
        vals = clean(df.loc[mask, col])
        data.append(vals.to_numpy())
        labels.append(MODEL_LABELS[m])
        colors.append(MODEL_COLORS[m])
        means.append(float(vals.mean()) if len(vals) else np.nan)

    fig, ax = plt.subplots(figsize=(12, 6))

    bp = ax.boxplot(
        data, tick_labels=labels,
        showfliers=False, patch_artist=True, widths=0.55,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for element in ("medians", "whiskers", "caps"):
        for line in bp[element]:
            line.set_color("#1B2A3A")

    # mean diamonds
    x_pos = np.arange(1, len(means) + 1)
    ax.scatter(x_pos, means, marker="D", s=65,
               color="black", zorder=5, label="mean Δ")
    for i, (xp, mv) in enumerate(zip(x_pos, means)):
        if np.isfinite(mv):
            ax.text(xp, mv + max(means) * 0.025 + 0.3,
                    f"{mv:.1f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold")

    # reference lines (only for information criteria, not R²)
    if criterion in ("aic", "aicc", "bic"):
        for ref, label_txt in ((2, "Δ = 2  (indistinguishable)"),
                               (10, "Δ = 10  (decisive)")):
            ax.axhline(ref, color="grey", linewidth=1.0,
                       linestyle="--", alpha=0.65, label=label_txt)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fitted model")
    ax.set_ylabel(f"Δ{crit_pretty}  (0 = best model)")
    ax.set_title(
        f"{SPECIES} | {STATION_ID} | {VAR_LABELS[variable]}  —  "
        f"Δ{crit_pretty} distribution across timesteps\n"
        f"(lower = closer to winning model; 0 = winner)",
        fontsize=TITLE_SIZE,
    )
    plt.xticks(rotation=15, ha="right")
    ax.legend(frameon=True, loc="upper left")
    savefig(f"delta_{criterion}_{STATION_ID}_{variable}.png")


def run_delta_distributions(df):
    print("\n[2b/4] Delta distribution plots")
    for var in VARIABLES:
        for crit in CRITERIA:
            # skip r2 / adj_r2 if no delta columns present (old CSVs)
            plot_delta_distribution(df, var, crit)

# ============================================================
# PLOT 3: AICc TIME-SERIES (all 7 models on one axes)
# ============================================================

def plot_aicc_timeseries(df, variable):
    """One figure per variable: AICc for all 7 models over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    any_plotted = False

    for model in MODELS:
        col = f"{variable}_{model}_aicc"
        if col not in df.columns:
            continue
        mask = filter_by_n(df, variable, model)
        vals = clean(df.loc[mask, col])
        ax.plot(df.loc[mask, "plot_datetime"], vals,
                color=MODEL_COLORS[model], linewidth=0.8, alpha=0.8,
                label=MODEL_LABELS[model])
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print(f"[skip] no AICc columns for {variable}")
        return

    nice_time_axis(ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("AICc  (lower = better)")
    ax.set_title(
        f"{SPECIES} | {STATION_ID} | {VAR_LABELS[variable]}  —  "
        f"AICc time-series across the 7 candidate models",
        fontsize=TITLE_SIZE,
    )
    ax.legend(ncol=4, frameon=True, loc="best")
    savefig(f"aicc_timeseries_{STATION_ID}_{variable}.png")


def run_aicc_timeseries(df):
    print("\n[3/4] AICc time-series plots")
    for var in VARIABLES:
        plot_aicc_timeseries(df, var)


# ============================================================
# PLOT 4: LAMBDA (SATURATING LENGTH SCALE) TIME-SERIES
# ============================================================

def plot_lambda_timeseries(df, variable):
    """
    The saturating model is y = a + b (1 - exp(-x/lam)), with lam stored as
    'c'. Lambda has direct physical meaning: the area scale (km^2) over which
    the metric saturates. This is the headline publishable parameter for
    representativeness.

    Display modes (set by LAMBDA_DISPLAY):
        "clip"   - cap the visible y-range at LAMBDA_MAX_KM2 (data unchanged)
        "log"    - log y-axis to compress extreme values
        "filter" - drop timesteps where lambda exceeds LAMBDA_MAX_KM2
    """
    col = f"{variable}_saturating_c"
    if col not in df.columns:
        print(f"[skip] no saturating model for {variable}")
        return

    mask = filter_by_n(df, variable, "saturating")
    vals = clean(df.loc[mask, col])
    times = df.loc[mask, "plot_datetime"]

    # apply mode-specific data handling
    extra_title = ""
    if LAMBDA_DISPLAY == "filter":
        keep = vals <= LAMBDA_MAX_KM2
        n_total   = int(vals.notna().sum())
        n_dropped = int((vals > LAMBDA_MAX_KM2).sum())
        pct_drop  = 100.0 * n_dropped / n_total if n_total > 0 else 0.0
        vals  = vals.where(keep)
        extra_title = (f"  |  filtered: λ ≤ {LAMBDA_MAX_KM2:,.0f} km² "
                       f"({pct_drop:.1f}% dropped)")
    elif LAMBDA_DISPLAY not in ("clip", "log"):
        raise ValueError(f"Invalid LAMBDA_DISPLAY: {LAMBDA_DISPLAY!r}. "
                         f"Use 'clip', 'log', or 'filter'.")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times, vals,
            color=MODEL_COLORS["saturating"], linewidth=0.9, alpha=0.9)

    # apply mode-specific axis handling
    if LAMBDA_DISPLAY == "clip":
        ax.set_ylim(0, LAMBDA_MAX_KM2)
        n_total      = int(vals.notna().sum())
        n_clipped    = int((vals > LAMBDA_MAX_KM2).sum())
        pct_clipped  = 100.0 * n_clipped / n_total if n_total > 0 else 0.0
        if pct_clipped > 0:
            extra_title = (f"  |  clipped at {LAMBDA_MAX_KM2:,.0f} km² "
                           f"({pct_clipped:.1f}% off-scale)")
    elif LAMBDA_DISPLAY == "log":
        ax.set_yscale("log")

    # median reference line (computed AFTER any filtering)
    median_val = vals.median()
    if np.isfinite(median_val):
        ax.axhline(median_val, color="black", linewidth=0.8, linestyle="--",
                   alpha=0.6, label=f"median λ = {median_val:,.0f} km²")
        ax.legend(frameon=True, loc="best")

    nice_time_axis(ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("λ  —  saturating length scale  (km²)")
    ax.set_title(
        f"{SPECIES} | {STATION_ID} | {VAR_LABELS[variable]}  —  "
        f"representativeness length scale λ over time{extra_title}",
        fontsize=TITLE_SIZE,
    )
    savefig(f"lambda_timeseries_{STATION_ID}_{variable}_{LAMBDA_DISPLAY}.png")


def run_lambda_timeseries(df):
    print("\n[4/4] Lambda (saturating length scale) time-series")
    for var in VARIABLES:
        plot_lambda_timeseries(df, var)


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("Start:", dt.datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S"))

    df = load_data()

    run_slope_timeseries(df)
    run_best_model_bars(df)
    run_delta_distributions(df)
    run_aicc_timeseries(df)
    run_lambda_timeseries(df)

    t1 = time.time()
    print("\nEnd:", dt.datetime.fromtimestamp(t1).strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Output directory: {OUT_DIR}")
    print(f"Execution time: {(t1 - t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
# %%