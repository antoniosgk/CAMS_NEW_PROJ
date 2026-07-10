
#%%
"""
Plot the area-based multi-model fit results produced by metrics_fit_backfill_area.py.

Plots produced:
1. Slope time-series (stacked or separate) — with configurable SLOPE_MODE:
      "chord"  = average slope C1->C10 (read from CSV)
      "box04"  = analytic dy/dx at the 0.4 deg box area
      "custom" = analytic dy/dx at any user-specified area in km^2
2. Best-model bar charts (5 criteria x VARIABLES)
2b. Delta distribution boxplots (5 criteria x VARIABLES)
3. AICc time-series (all 7 models per variable)
4. Lambda time-series (saturating length scale)
5. Single-timestep fit inspection (observed points + 7 fitted curves)
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

IN_DIR  = Path("/mnt/store01/agkiokas/CAMS/fit_outputs")
OUT_DIR = Path("/mnt/store01/agkiokas/CAMS/area_fit_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES    = "O3"
STATION_ID = "1002A"
CSV_FILENAME = f"{STATION_ID}_{SPECIES}_area_fits.csv"

# ------------------------------------------------------------
# Plot 1 — slope time-series layout
# ------------------------------------------------------------
PLOT_LAYOUT = "stacked"      # "stacked" | "separate"

# ------------------------------------------------------------
# Slope mode for the slope time-series plots
#   "chord"  -> average slope C1->C10, read from {var}_{model}_slope
#   "box04"  -> analytic derivative dy/dx at the 0.4 deg box area
#               (reads area_km2_box04 from CSV; recomputes if absent)
#   "custom" -> analytic derivative dy/dx at SLOPE_AT_KM2
# ------------------------------------------------------------
SLOPE_MODE   = "box04"       # "chord" | "box04" | "custom"
SLOPE_AT_KM2 = 5000.0        # used only when SLOPE_MODE == "custom"
BOX_DEG      = 0.4           # label / recompute fallback for "box04"

# ------------------------------------------------------------
# Lambda display
# ------------------------------------------------------------
LAMBDA_DISPLAY      = "filter"       # "clip" | "log" | "filter"
LAMBDA_MAX_KM2      = 50_000.0
LAMBDA_STRICT_FRAC  = 1 / 3
LAMBDA_RELAXED_FRAC = 1 / 2

# ------------------------------------------------------------
# Single-timestep fit inspection
# FIT_TIMESTEP: int row index (0-based) or datetime string
# ------------------------------------------------------------
PLOT_SINGLE_FIT = True
FIT_TIMESTEP    = 0

# ------------------------------------------------------------
# Time range
# ------------------------------------------------------------
PLOT_FULL_PERIOD = True
START_DATE = "2005-05-20"
END_DATE   = "2005-06-20"

# ------------------------------------------------------------
# Variables and selection criteria
# ------------------------------------------------------------
VARIABLES = ["ratio", "cv_w"]
CRITERIA  = ["r2", "adj_r2", "aic", "aicc", "bic"]

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

MODELS = ["linear", "quadratic", "cubic", "logarithmic",
          "exponential", "power", "saturating"]

MODEL_LABELS = {
    "linear": "Linear", "quadratic": "Quadratic", "cubic": "Cubic",
    "logarithmic": "Logarithmic", "exponential": "Exponential",
    "power": "Power-law", "saturating": "Saturating",
}

MODEL_COLORS = {
    "linear": "#1f77b4", "quadratic": "#ff7f0e", "cubic": "#2ca02c",
    "logarithmic": "#9467bd", "exponential": "#d62728",
    "power": "#8c564b", "saturating": "#17becf",
}

COMPANION = {
    "linear":      ("a", "intercept a",            "y = a + b\u00b7x"),
    "quadratic":   ("c", "curvature c",            "y = a + b\u00b7x + c\u00b7x\u00b2"),
    "cubic":       ("d", "cubic coeff. d",         "y = a + b\u00b7x + c\u00b7x\u00b2 + d\u00b7x\u00b3"),
    "logarithmic": ("a", "intercept a",            "y = a + b\u00b7ln(x)"),
    "exponential": ("c", "rate c (1/km\u00b2)",    "y = a + b\u00b7exp(c\u00b7x)"),
    "power":       ("b", "exponent b",             "y = a\u00b7x^b"),
    "saturating":  ("c", "length scale \u03bb (km\u00b2)",
                    "y = a + b\u00b7(1 \u2212 exp(\u2212x/\u03bb))"),
}

COMPANION_COLOR = "#555555"
SLOPE_COLOR_FOR_MODEL = lambda m: MODEL_COLORS[m]
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
    n_col = f"{variable}_{model}_n"
    if n_col in df.columns:
        return df[n_col] >= MIN_N_SECTORS
    return pd.Series(True, index=df.index)


def nice_time_axis(ax):
    locator   = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=0)


def clean(s):
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
# ANALYTIC DERIVATIVE + SLOPE MODE RESOLVER
# ============================================================

def analytic_derivative(df, variable, model, x_star):
    """
    dy/dx of the fitted model at x = x_star (km^2), from stored parameters.
    Vectorised over all rows; returns a Series.
    """
    a = clean(df.get(f"{variable}_{model}_a"))
    b = clean(df.get(f"{variable}_{model}_b"))
    c = clean(df.get(f"{variable}_{model}_c"))
    d = clean(df.get(f"{variable}_{model}_d"))

    if model == "linear":
        return b
    if model == "quadratic":
        return b + 2.0 * c * x_star
    if model == "cubic":
        return b + 2.0 * c * x_star + 3.0 * d * x_star ** 2
    if model == "logarithmic":
        return b / x_star
    if model == "exponential":
        return b * c * np.exp(np.clip(c * x_star, -50, 50))
    if model == "power":
        return a * b * np.power(x_star, b - 1.0)
    if model == "saturating":
        lam = c
        return (b / lam) * np.exp(-np.clip(x_star / lam, -50, 50))
    raise ValueError(f"Unknown model: {model}")


def resolve_slope_x(df):
    """
    Return (x_star_km2, label, out_of_range) for the current SLOPE_MODE.
    For 'chord' returns (None, label, False).
    """
    if SLOPE_MODE == "chord":
        return None, "average slope C1\u2192C10", False

    if SLOPE_MODE == "box04":
        if "area_km2_box04" in df.columns:
            x_star = float(df["area_km2_box04"].iloc[0])
        else:
            lat = float(df["model_lat"].iloc[0])
            phi_s = np.radians(lat - BOX_DEG / 2.0)
            phi_n = np.radians(lat + BOX_DEG / 2.0)
            R = 6371.0088
            x_star = R**2 * (np.sin(phi_n) - np.sin(phi_s)) * np.radians(BOX_DEG)
        label = f"dy/dx at {x_star:,.0f} km\u00b2 ({BOX_DEG}\u00b0 box)"
    elif SLOPE_MODE == "custom":
        x_star = float(SLOPE_AT_KM2)
        label = f"dy/dx at {x_star:,.0f} km\u00b2"
    else:
        raise ValueError(f"Invalid SLOPE_MODE: {SLOPE_MODE!r}")

    out_of_range = False
    if "area_km2_C1" in df.columns and "area_km2_C10" in df.columns:
        x_lo = float(df["area_km2_C1"].iloc[0])
        x_hi = float(df["area_km2_C10"].iloc[0])
        if not (x_lo <= x_star <= x_hi):
            out_of_range = True
            print(f"[WARNING] x* = {x_star:,.0f} km\u00b2 is outside "
                  f"[{x_lo:,.0f}, {x_hi:,.0f}] — extrapolation.")
            label += "  (out of C1\u2013C10 range)"
    return x_star, label, out_of_range


# ============================================================
# MODEL PREDICTION (for single-timestep inspection)
# ============================================================

def model_prediction(row, variable, model, x):
    """Evaluate the fitted model on an x array (km^2) for one CSV row."""
    a = row.get(f"{variable}_{model}_a", np.nan)
    b = row.get(f"{variable}_{model}_b", np.nan)
    c = row.get(f"{variable}_{model}_c", np.nan)
    d = row.get(f"{variable}_{model}_d", np.nan)
    x = np.asarray(x, dtype=float)

    if model == "linear":
        return a + b * x
    if model == "quadratic":
        return a + b * x + c * x ** 2
    if model == "cubic":
        return a + b * x + c * x ** 2 + d * x ** 3
    if model == "logarithmic":
        return a + b * np.log(x)
    if model == "exponential":
        return a + b * np.exp(np.clip(c * x, -50, 50))
    if model == "power":
        return a * np.power(x, b)
    if model == "saturating":
        lam = c
        return a + b * (1.0 - np.exp(-np.clip(x / lam, -50, 50)))
    raise ValueError(f"Unknown model: {model}")


# ============================================================
# PLOT 1: SLOPE + COMPANION TIME-SERIES
# ============================================================

def _plot_slope_panel(ax, df, variable, model):
    mask = filter_by_n(df, variable, model)
    sub  = df.loc[mask]

    comp_key, comp_label, _ = COMPANION[model]
    comp_col = f"{variable}_{model}_{comp_key}"

    x_star, slope_label, _ = resolve_slope_x(df)

    if SLOPE_MODE == "chord":
        slope_col = f"{variable}_{model}_slope"
        if slope_col not in df.columns:
            ax.text(0.5, 0.5, f"missing: {slope_col}", ha="center",
                    va="center", transform=ax.transAxes)
            return
        slope_vals = clean(sub[slope_col])
    else:
        slope_vals = analytic_derivative(sub, variable, model, x_star)

    slope_color = SLOPE_COLOR_FOR_MODEL(model)
    ax.plot(sub["plot_datetime"], slope_vals,
            color=slope_color, linewidth=0.9, alpha=0.9,
            label=f"{slope_label}")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_ylabel(f"{VAR_LABELS[variable]}  (per km\u00b2)",
                  color=slope_color)
    ax.tick_params(axis="y", colors=slope_color)
    ax.spines["left"].set_color(slope_color)

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
    _, slope_label, _ = resolve_slope_x(df)
    for model in MODELS:
        fig, axes = plt.subplots(len(VARIABLES), 1, figsize=(13, 9), sharex=True)
        if len(VARIABLES) == 1:
            axes = [axes]
        for ax, var in zip(axes, VARIABLES):
            _plot_slope_panel(ax, df, var, model)
            ax.set_title(f"{VAR_LABELS[var]}", loc="left", fontsize=LABEL_SIZE)
        nice_time_axis(axes[-1])
        fig.suptitle(
            f"{SPECIES} | {STATION_ID} | {MODEL_LABELS[model]} fit: "
            f"{COMPANION[model][2]}\n"
            f"{slope_label} (left) + {COMPANION[model][1]} (right)",
            fontsize=TITLE_SIZE, y=1.00,
        )
        savefig(f"slope_stacked_{STATION_ID}_{model}_{SLOPE_MODE}.png")


def plot_slope_timeseries_separate(df):
    _, slope_label, _ = resolve_slope_x(df)
    for model in MODELS:
        for var in VARIABLES:
            fig, ax = plt.subplots(figsize=(13, 5))
            _plot_slope_panel(ax, df, var, model)
            nice_time_axis(ax)
            ax.set_title(
                f"{SPECIES} | {STATION_ID} | {MODEL_LABELS[model]} on "
                f"{VAR_LABELS[var]}: {COMPANION[model][2]}\n"
                f"{slope_label} (left) + {COMPANION[model][1]} (right)",
                fontsize=TITLE_SIZE,
            )
            savefig(f"slope_separate_{STATION_ID}_{model}_{var}_{SLOPE_MODE}.png")


def run_slope_timeseries(df):
    print(f"\n[1/5] Slope time-series plots (mode: {SLOPE_MODE})")
    if PLOT_LAYOUT == "stacked":
        plot_slope_timeseries_stacked(df)
    elif PLOT_LAYOUT == "separate":
        plot_slope_timeseries_separate(df)
    else:
        raise ValueError(f"Invalid PLOT_LAYOUT: {PLOT_LAYOUT}")


# ============================================================
# PLOT 2: BEST-MODEL BAR CHARTS
# ============================================================

def _best_model_per_row(df, variable, criterion):
    cols = {m: f"{variable}_{m}_{criterion}" for m in MODELS}
    if not all(c in df.columns for c in cols.values()):
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
        best[valid_any] = frame.loc[valid_any].idxmax(axis=1)
    else:
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
        return

    heights = [counts.get(m, 0) for m in MODELS]
    pct     = [100.0 * h / total for h in heights]
    colors  = [MODEL_COLORS[m] for m in MODELS]
    labels  = [MODEL_LABELS[m] for m in MODELS]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, pct, color=colors, alpha=0.85, width=0.65)
    for bar, p in zip(bars, pct):
        if p > 0:
            ax.text(bar.get_x() + bar.get_width()/2, p + 0.6,
                    f"{p:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    crit_pretty = {"r2": "R\u00b2", "adj_r2": "adjusted R\u00b2",
                   "aic": "AIC", "aicc": "AICc", "bic": "BIC"}[criterion]
    direction = "highest" if criterion in ("r2", "adj_r2") else "lowest"
    ax.set_ylim(0, max(pct) * 1.15 if max(pct) > 0 else 1)
    ax.set_ylabel(f"% timesteps with {direction} {crit_pretty}")
    ax.set_xlabel("Fitted model")
    ax.set_title(f"{SPECIES} | {STATION_ID} | {VAR_LABELS[variable]}  \u2014  "
                 f"best model by {crit_pretty}  ({total:,} timesteps)",
                 fontsize=TITLE_SIZE)
    plt.xticks(rotation=15, ha="right")
    savefig(f"barplot_best_{criterion}_{STATION_ID}_{variable}.png")


def run_best_model_bars(df):
    print("\n[2/5] Best-model bar charts")
    for var in VARIABLES:
        for crit in CRITERIA:
            plot_best_model_bar(df, var, crit)


# ============================================================
# PLOT 2b: DELTA DISTRIBUTION
# ============================================================

def plot_delta_distribution(df, variable, criterion):
    delta_cols = {m: f"{variable}_{m}_delta_{criterion}" for m in MODELS}
    have = [c for c in delta_cols.values() if c in df.columns]
    if not have:
        print(f"[skip] no delta_{criterion} columns for {variable}")
        return

    crit_pretty = {"r2": "R\u00b2", "adj_r2": "adj. R\u00b2",
                   "aic": "AIC", "aicc": "AICc", "bic": "BIC"
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
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False,
                    patch_artist=True, widths=0.55)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for element in ("medians", "whiskers", "caps"):
        for line in bp[element]:
            line.set_color("#1B2A3A")

    x_pos = np.arange(1, len(means) + 1)
    ax.scatter(x_pos, means, marker="D", s=65, color="black",
               zorder=5, label="mean \u0394")
    for xp, mv in zip(x_pos, means):
        if np.isfinite(mv):
            ax.text(xp, mv + max(m for m in means if np.isfinite(m)) * 0.025 + 0.3,
                    f"{mv:.1f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold")

    if criterion in ("aic", "aicc", "bic"):
        for ref, txt in ((2, "\u0394 = 2  (indistinguishable)"),
                         (10, "\u0394 = 10  (decisive)")):
            ax.axhline(ref, color="grey", linewidth=1.0,
                       linestyle="--", alpha=0.65, label=txt)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fitted model")
    ax.set_ylabel(f"\u0394{crit_pretty}  (0 = best model)")
    ax.set_title(f"{SPECIES} | {STATION_ID} | {VAR_LABELS[variable]}  \u2014  "
                 f"\u0394{crit_pretty} distribution across timesteps",
                 fontsize=TITLE_SIZE)
    plt.xticks(rotation=15, ha="right")
    ax.legend(frameon=True, loc="upper left")
    savefig(f"delta_{criterion}_{STATION_ID}_{variable}.png")


def run_delta_distributions(df):
    print("\n[2b/5] Delta distribution plots")
    for var in VARIABLES:
        for crit in CRITERIA:
            plot_delta_distribution(df, var, crit)


# ============================================================
# PLOT 3: AICc TIME-SERIES
# ============================================================

def plot_aicc_timeseries(df, variable):
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
        return
    nice_time_axis(ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("AICc  (lower = better)")
    ax.set_title(f"{SPECIES} | {STATION_ID} | {VAR_LABELS[variable]}  \u2014  "
                 f"AICc time-series", fontsize=TITLE_SIZE)
    ax.legend(ncol=4, frameon=True, loc="best")
    savefig(f"aicc_timeseries_{STATION_ID}_{variable}.png")


def run_aicc_timeseries(df):
    print("\n[3/5] AICc time-series plots")
    for var in VARIABLES:
        plot_aicc_timeseries(df, var)


# ============================================================
# PLOT 4: LAMBDA TIME-SERIES
# ============================================================

def plot_lambda_timeseries(df, variable):
    col = f"{variable}_saturating_c"
    if col not in df.columns:
        print(f"[skip] no saturating model for {variable}")
        return

    mask  = filter_by_n(df, variable, "saturating")
    vals  = clean(df.loc[mask, col])
    times = df.loc[mask, "plot_datetime"]

    if "area_km2_C10" in df.columns:
        c10 = df.loc[mask, "area_km2_C10"].pipe(clean)
        lambda_strict  = c10 * LAMBDA_STRICT_FRAC
        lambda_relaxed = c10 * LAMBDA_RELAXED_FRAC
    else:
        raise KeyError("Column 'area_km2_C10' not found.")

    strict_median  = float(lambda_strict.median())
    relaxed_median = float(lambda_relaxed.median())

    extra_title = ""
    if LAMBDA_DISPLAY == "filter":
        keep      = vals <= lambda_relaxed
        n_total   = int(vals.notna().sum())
        n_dropped = int((~keep).sum())
        pct_drop  = 100.0 * n_dropped / n_total if n_total > 0 else 0.0
        vals      = vals.where(keep)
        extra_title = f"  |  filtered (\u03bb \u2264 C10/2, {pct_drop:.1f}% dropped)"
    elif LAMBDA_DISPLAY == "clip":
        ax_max = relaxed_median * 4
    elif LAMBDA_DISPLAY != "log":
        raise ValueError(f"Invalid LAMBDA_DISPLAY: {LAMBDA_DISPLAY!r}")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times, vals, color=MODEL_COLORS["saturating"],
            linewidth=0.9, alpha=0.9)

    if LAMBDA_DISPLAY == "clip":
        ax.set_ylim(0, ax_max)
    elif LAMBDA_DISPLAY == "log":
        ax.set_yscale("log")

    vals_capped  = vals.where(vals <= lambda_relaxed)
    vals_indexed = vals_capped.set_axis(times)
    rolling_med  = vals_indexed.rolling("30D", center=True, min_periods=200).median()
    ax.plot(times, rolling_med, color="black", linewidth=1.8, alpha=0.9,
            label="30-day rolling median (capped at C10/2)")

    median_val = vals.median()
    if np.isfinite(median_val):
        ax.axhline(median_val, color="black", linewidth=0.8, linestyle=":",
                   alpha=0.6, label=f"overall median \u03bb = {median_val:,.0f} km\u00b2")
    ax.axhline(strict_median,  color="#d62728", linewidth=1.2, linestyle="--",
               alpha=0.8, label=f"strict \u03bb = C10/3 ({strict_median:,.0f} km\u00b2)")
    ax.axhline(relaxed_median, color="#ff7f0e", linewidth=1.2, linestyle="--",
               alpha=0.8, label=f"relaxed \u03bb = C10/2 ({relaxed_median:,.0f} km\u00b2)")
    ax.legend(frameon=True, loc="upper right")

    nice_time_axis(ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("\u03bb  \u2014  saturating length scale  (km\u00b2)")
    ax.set_title(f"{SPECIES} | {STATION_ID} | {VAR_LABELS[variable]}  \u2014  "
                 f"\u03bb over time{extra_title}", fontsize=TITLE_SIZE)
    savefig(f"lambda_timeseries_{STATION_ID}_{variable}_{LAMBDA_DISPLAY}.png")


def run_lambda_timeseries(df):
    print("\n[4/5] Lambda time-series")
    for var in VARIABLES:
        plot_lambda_timeseries(df, var)


# ============================================================
# PLOT 5: SINGLE-TIMESTEP FIT INSPECTION
# ============================================================

def _resolve_timestep_row(df):
    if isinstance(FIT_TIMESTEP, int):
        if not (0 <= FIT_TIMESTEP < len(df)):
            raise IndexError(f"FIT_TIMESTEP={FIT_TIMESTEP} outside 0..{len(df)-1}")
        return df.iloc[FIT_TIMESTEP], FIT_TIMESTEP
    target = pd.to_datetime(FIT_TIMESTEP)
    matches = df.index[df["plot_datetime"] == target]
    if len(matches) == 0:
        i = (df["plot_datetime"] - target).abs().idxmin()
        print(f"[INFO] exact timestep not found; using nearest: "
              f"{df.loc[i, 'plot_datetime']}")
        return df.loc[i], i
    return df.loc[matches[0]], matches[0]


def plot_single_timestep_fits(df, variable):
    row, idx = _resolve_timestep_row(df)

    x_obs_cols = [f"area_km2_C{s}" for s in range(1, 11)]
    y_obs_cols = [f"{variable}_C{s}" for s in range(1, 11)]

    if not all(c in df.columns for c in x_obs_cols):
        print(f"[skip] area_km2_C columns not in CSV")
        return
    if not all(c in df.columns for c in y_obs_cols):
        print(f"[skip] {variable}_C1..C10 not in CSV; "
              f"re-run metrics_fit_backfill_area.py to include observed values")
        return

    x_obs = np.array([row[c] for c in x_obs_cols], dtype=float)
    y_obs = np.array([row[c] for c in y_obs_cols], dtype=float)
    x_grid = np.linspace(x_obs.min(), x_obs.max(), 300)

    best = row.get(f"{variable}_best_model_aicc", None)

    fig, ax = plt.subplots(figsize=(12, 7))

    for model in MODELS:
        y_fit = model_prediction(row, variable, model, x_grid)
        if not np.any(np.isfinite(y_fit)):
            continue
        is_best = (model == best)
        aicc = row.get(f"{variable}_{model}_aicc", np.nan)
        label = f"{MODEL_LABELS[model]} (AICc={aicc:.1f})"
        if is_best:
            label += "  \u2190 best"
        ax.plot(x_grid, y_fit,
                color=MODEL_COLORS[model],
                linewidth=3.0 if is_best else 1.4,
                alpha=1.0 if is_best else 0.75,
                zorder=4 if is_best else 2,
                label=label)

    ax.scatter(x_obs, y_obs, s=70, color="black", zorder=5,
               label="observed (C1\u2013C10)")

    ax.set_xlabel("cumulative sector area (km\u00b2)")
    ax.set_ylabel(VAR_LABELS[variable])
    ts = row["plot_datetime"]
    ax.set_title(
        f"{SPECIES} | {STATION_ID} | {VAR_LABELS[variable]}  \u2014  "
        f"all 7 fits at {ts}  (row {idx})",
        fontsize=TITLE_SIZE)
    ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best")
    ts_safe = pd.to_datetime(ts).strftime("%Y%m%d_%H%M")
    savefig(f"single_fit_{STATION_ID}_{variable}_{ts_safe}.png")


def run_single_timestep_fits(df):
    if not PLOT_SINGLE_FIT:
        return
    print("\n[5/5] Single-timestep fit inspection")
    for var in VARIABLES:
        plot_single_timestep_fits(df, var)


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
    run_single_timestep_fits(df)

    t1 = time.time()
    print("\nEnd:", dt.datetime.fromtimestamp(t1).strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Output directory: {OUT_DIR}")
    print(f"Execution time: {(t1 - t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
# %%