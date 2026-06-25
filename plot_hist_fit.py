#%%
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

CSV_DIR = Path("/mnt/store01/agkiokas/CAMS/derived_station_metrics/")

OUT_DIR = Path("/mnt/store01/agkiokas/CAMS/derived_station_metrics/parameter_timeseries_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = "O3"

# ------------------------------------------------------------
# CSV source
# ------------------------------------------------------------
# The backfill script (backfill_fit_metrics.py) writes files with the
# extra adj_r2 / aic / bic / n / best_model columns. If you used the
# default OUT_SUFFIX="_with_aic", point the pattern at those files.
# If you overwrote in place (OUT_SUFFIX=None), use the original pattern.
CSV_PATTERN = "*_sector_ratio_cv_fits_with_aic.csv"
# CSV_PATTERN = "*_sector_ratio_cv_fits.csv"

# ------------------------------------------------------------
# 1) Time-series plots of fitted parameters
# ------------------------------------------------------------
TIMESERIES_STATIONS = ["1006A"]
# TIMESERIES_STATIONS = ["1002A", "1003A", "1004A"]

PLOT_STATIONS_TOGETHER = True

PLOT_FULL_PERIOD = True
START_DATE = "2005-05-20"
END_DATE = "2005-06-20"

# ------------------------------------------------------------
# 2) R² / adjusted-R² distribution and best-model plots
# ------------------------------------------------------------
# Options:
#   "one"       -> use R2_STATIONS
#   "selected"  -> use R2_STATIONS
#   "all"       -> use all CSV files matching CSV_PATTERN in CSV_DIR
R2_MODE = "all"

R2_STATIONS = ["1002A", "1003A", "1004A"]

# Minimum number of sectors used in a fit for it to count in the
# distribution / best-model plots. With ~10 sectors per timestep this
# removes only the rare incomplete (and degenerate) timesteps.
MIN_N_SECTORS = 8

# Figure settings
DPI = 400

TITLE_SIZE = 16
LABEL_SIZE = 13
TICK_SIZE = 11
LEGEND_SIZE = 10


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "figure.figsize": (13, 6),
    "figure.dpi": 120,
    "savefig.dpi": DPI,
    "font.size": 12,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def savefig(name):
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"Saved: {path}")


def _strip_suffix(name):
    """Strip both possible CSV suffixes to recover the station id prefix."""
    return (name
            .replace("_sector_ratio_cv_fits_with_aic.csv", "")
            .replace("_sector_ratio_cv_fits.csv", ""))


def station_id_from_csv(csv_file):
    """
    Example:
        1006A_O3_sector_ratio_cv_fits_with_aic.csv -> 1006A
    """
    stem = _strip_suffix(csv_file.name)
    # stem is like "1006A_O3"; station id is the first token.
    return stem.split("_")[0]


def get_csv_file_for_station(station_id):
    """
    Resolve the CSV for a station respecting CSV_PATTERN. Tries the
    with_aic name first, then the plain name.
    """
    candidates = [
        CSV_DIR / f"{station_id}_{SPECIES}_sector_ratio_cv_fits_with_aic.csv",
        CSV_DIR / f"{station_id}_{SPECIES}_sector_ratio_cv_fits.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # non-existent; caller warns


def get_available_csv_files():
    files = sorted(CSV_DIR.glob(CSV_PATTERN))
    files = [f for f in files if f.is_file()]
    return files


def get_selected_csv_files(stations):
    files = []
    for station_id in stations:
        f = get_csv_file_for_station(station_id)
        if f.exists():
            files.append(f)
        else:
            print(f"[WARNING] CSV not found for station {station_id}: {f}")
    return files


def get_datetime_column(df):
    """
    Create a plot_datetime column from datetime, timestamp, or time_key.
    """
    if "datetime" in df.columns:
        dt_col = "datetime"
    elif "timestamp" in df.columns:
        dt_col = "timestamp"
    elif "time_key" in df.columns:
        dt_col = "time_key"
    else:
        raise ValueError("No datetime-like column found. Expected datetime, timestamp, or time_key.")

    df[dt_col] = df[dt_col].astype(str)

    if df[dt_col].str.contains("_").any():
        df["plot_datetime"] = pd.to_datetime(df[dt_col], format="%Y%m%d_%H%M", errors="coerce")
    else:
        df["plot_datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
        if df["plot_datetime"].isna().all():
            df["plot_datetime"] = pd.to_datetime(df[dt_col], format="%Y%m%d%H%M", errors="coerce")

    return df


def _cols_for_distribution(csv_file):
    """
    Return the minimal set of columns needed for distribution/AIC plots.

    Keeps:
      - the datetime column
      - all *_r2, *_adj_r2, *_aic, *_bic, *_n, *_rss columns
      - all *_best_model_aic and *_delta_aic columns
      - only for variables ratio and cv_w (mean_w, std_w, median_w not needed)

    Drops (not needed for histograms/CDFs/bar charts):
      - wide sector columns  {var}_C1 ... {var}_C10
      - coefficient columns  {var}_{model}_a / _b / _c
      - metadata columns (coordinates, k_star, etc.)

    Typically reduces per-file memory by 60-75%.
    """
    header = pd.read_csv(csv_file, nrows=0)
    all_cols = list(header.columns)

    keep_suffixes = (
        "_r2", "_adj_r2", "_aic", "_bic", "_n", "_rss",
        "_delta_aic", "_best_model_aic", "_r2_check",
    )

    selected = []
    for c in all_cols:
        if c in ("datetime", "timestamp", "time_key"):
            selected.append(c)
            continue
        if any(c.endswith(s) for s in keep_suffixes):
            selected.append(c)

    # Distribution plots only use ratio and cv_w.
    # Drop metric columns for mean_w, std_w, median_w to save RAM.
    DIST_VARS = ("ratio", "cv_w")
    selected = [c for c in selected
                if c in ("datetime", "timestamp", "time_key")
                or any(c.startswith(v + "_") for v in DIST_VARS)]

    return selected if selected else None  # None -> read all (fallback)


# BUG FIX 1: restored usecols parameter to load_station_csv
def load_station_csv(csv_file, usecols=None):
    station_id = station_id_from_csv(csv_file)

    # BUG FIX 1: pass usecols to pd.read_csv so excluded columns
    # are never read from disk
    df = pd.read_csv(csv_file, usecols=usecols)
    df = get_datetime_column(df)

    df = df.dropna(subset=["plot_datetime"]).copy()
    df = df.sort_values("plot_datetime").copy()

    if not PLOT_FULL_PERIOD:
        df = df[
            (df["plot_datetime"] >= pd.to_datetime(START_DATE)) &
            (df["plot_datetime"] <= pd.to_datetime(END_DATE))
        ].copy()

    df["station_id"] = station_id
    return df


# BUG FIX 4: restored total_rows counter and RAM report
def load_multiple_station_csvs(csv_files, usecols=None):
    """
    Load and concatenate multiple station CSVs.

    Parameters
    ----------
    usecols : "distribution", list of str, or None
        "distribution" -> load only the fit-metric columns needed for
            R2/AIC distribution and best-model plots (~60-75% less RAM).
        list of str    -> load exactly those columns.
        None           -> load all columns (required for time-series plots).
    """
    dfs = []
    total_rows = 0

    for f in csv_files:
        try:
            cols = _cols_for_distribution(f) if usecols == "distribution" else usecols
            df = load_station_csv(f, usecols=cols)
            total_rows += len(df)
            dfs.append(df)
            print(f"  Loaded {f.name}: {len(df):,} rows, "
                  f"{len(df.columns)} cols")
        except Exception as e:
            print(f"[WARNING] Could not load {f.name}: {e}")

    if len(dfs) == 0:
        raise ValueError("No valid CSV files loaded.")

    out = pd.concat(dfs, ignore_index=True)
    print(f"  --> Combined: {total_rows:,} rows, "
          f"{len(out.columns)} cols, "
          f"~{out.memory_usage(deep=True).sum() / 1e6:.1f} MB in RAM")
    return out


def nice_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=35)


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_series(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def filter_by_n(df, variable_key, model):
    """
    Return a boolean mask selecting rows where the fit for (variable, model)
    used at least MIN_N_SECTORS sectors. If the _n column is absent (older
    CSVs), keep all rows.
    """
    n_col = f"{variable_key}_{model}_n"
    if n_col in df.columns:
        return df[n_col] >= MIN_N_SECTORS
    return pd.Series(True, index=df.index)


# ============================================================
# PARAMETER DEFINITIONS
# ============================================================
# Linear:      y = a + b*x          -> shape parameter b
# Quadratic:   y = a + b*x + c*x²    -> shape parameter c (curvature)
# Exponential: y = a + b*exp(c*x)    -> shape parameter c (rate)
# ============================================================

PARAMETER_SPECS = {
    "ratio": {
        "title": "Ratio",
        "ylabel": "Shape parameter value",
        "columns": {
            "Linear b": "ratio_linear_b",
            "Quadratic c": "ratio_quadratic_c",
            "Exponential c": "ratio_exponential_c",
            "Exponential a·b": ("ratio_exponential_a", "ratio_exponential_b"),
            "Exponential b·c": ("ratio_exponential_b", "ratio_exponential_c"),
            "Exponential b":"ratio_exponential_b",
        },
    },
    "cv_w": {
        "title": "CV",
        "ylabel": "Shape parameter value",
        "columns": {
            "Linear b": "cv_w_linear_b",
            "Quadratic c": "cv_w_quadratic_c",
            "Exponential c": "cv_w_exponential_c",
            "Exponential a·b": ("cv_w_exponential_a", "cv_w_exponential_b"),
            "Exponential b·c": ("cv_w_exponential_b", "cv_w_exponential_c"),
            "Exponential b":"cv_w_exponential_b",
        },
        "scale": 100,
        "scaled_ylabel": "Shape parameter value, CV(%)",
    },
    "mean_w": {
        "title": "Weighted sector mean concentration",
        "ylabel": "Shape parameter value",
        "columns": {
            "Linear b": "mean_w_linear_b",
            "Quadratic c": "mean_w_quadratic_c",
            "Exponential c": "mean_w_exponential_c",
            "Exponential a·b": ("mean_w_exponential_a", "mean_w_exponential_b"),
            "Exponential b·c": ("mean_w_exponential_b", "mean_w_exponential_c"),
            "Exponential b": "mean_w_exponential_b",
        },
    },
}

# Models in a fixed order, with consistent colors across all plots.
MODELS = ["linear", "quadratic", "exponential"]
MODEL_LABELS = {"linear": "Linear", "quadratic": "Quadratic", "exponential": "Exponential"}
MODEL_COLORS = {"linear": "#1f77b4", "quadratic": "#ff7f0e", "exponential": "#2ca02c"}

def _resolve_series(df, col_spec, scale=1):
    """
    Resolve a column spec to a pandas Series, applying scale.

    col_spec can be:
      - a string  -> direct column lookup
      - a tuple of two strings -> element-wise product of the two columns
    """
    if isinstance(col_spec, tuple):
        col_a, col_b = col_spec
        return df[col_a] * df[col_b] * scale
    return df[col_spec] * scale
# ============================================================
# 1) TIME SERIES OF SHAPE PARAMETERS
# ============================================================

def plot_parameter_timeseries_for_variable_one_station(df, variable_key):
    spec = PARAMETER_SPECS[variable_key]
    station_id = df["station_id"].iloc[0]

    cols = []
    for v in spec["columns"].values():
        if isinstance(v, tuple):
            cols.extend(v)
        else:
            cols.append(v)
    require_columns(df, cols)

    scale = spec.get("scale", 1)

    fig, ax = plt.subplots(figsize=(14, 6))

    for label, col_spec in spec["columns"].items():
        ax.plot(df["plot_datetime"], _resolve_series(df, col_spec, scale),
            linewidth=1.1, label=label)

    ax.axhline(0, linestyle="--", linewidth=1.4)
    ax.set_xlabel("Time")

    if "scaled_ylabel" in spec and scale != 1:
        ax.set_ylabel(spec["scaled_ylabel"])
    else:
        ax.set_ylabel(spec["ylabel"])

    ax.set_title(f"{SPECIES} {spec['title']} fitted parameter time series | {station_id}")
    nice_time_axis(ax)
    ax.legend(frameon=True)
    savefig(f"{station_id}_{variable_key}_parameter_timeseries.png")


def plot_parameter_timeseries_for_variable_multiple_stations(df_all, variable_key):
    spec = PARAMETER_SPECS[variable_key]

    # handle both plain column names and tuple (product) specs
    cols = []
    for v in spec["columns"].values():
        if isinstance(v, tuple):
            cols.extend(v)
        else:
            cols.append(v)
    require_columns(df_all, cols)

    scale = spec.get("scale", 1)

    for parameter_label, col_spec in spec["columns"].items():
        fig, ax = plt.subplots(figsize=(14, 6))

        for station_id, g in df_all.groupby("station_id"):
            ax.plot(g["plot_datetime"],
                    _resolve_series(g, col_spec, scale),
                    linewidth=1.0, alpha=0.85, label=station_id)

        ax.axhline(0, linestyle="--", linewidth=1.4)
        ax.set_xlabel("Time")

        if "scaled_ylabel" in spec and scale != 1:
            ax.set_ylabel(spec["scaled_ylabel"])
        else:
            ax.set_ylabel(spec["ylabel"])

        ax.set_title(f"{SPECIES} {spec['title']} | {parameter_label} time series")
        nice_time_axis(ax)
        ax.legend(frameon=True, ncol=2)

        safe_param = parameter_label.lower().replace(" ", "_").replace("·", "x")
        savefig(f"multi_station_{variable_key}_{safe_param}_timeseries.png")


def run_parameter_timeseries_plots():
    csv_files = get_selected_csv_files(TIMESERIES_STATIONS)
    if len(csv_files) == 0:
        print("[WARNING] No CSV files found for TIMESERIES_STATIONS.")
        return

    df_all = load_multiple_station_csvs(csv_files, usecols=None)  # full load: coefficients needed

    if PLOT_STATIONS_TOGETHER:
        for variable_key in ["ratio", "cv_w", "mean_w"]:
            plot_parameter_timeseries_for_variable_multiple_stations(df_all, variable_key)
    else:
        for station_id, g in df_all.groupby("station_id"):
            for variable_key in ["ratio", "cv_w", "mean_w"]:
                plot_parameter_timeseries_for_variable_one_station(g, variable_key)


# ============================================================
# 2) R² / ADJUSTED-R² DISTRIBUTION PLOTS
# ============================================================

def get_r2_csv_files():
    if R2_MODE == "all":
        files = get_available_csv_files()
    elif R2_MODE in ["one", "selected"]:
        files = get_selected_csv_files(R2_STATIONS)
    else:
        raise ValueError("Invalid R2_MODE. Use 'one', 'selected', or 'all'.")

    if len(files) == 0:
        raise ValueError("No CSV files selected for R² distribution plots.")
    return files


def _r2_title_name(variable_key):
    return {"ratio": "ratio", "cv_w": "CV"}.get(variable_key, variable_key)


def _metric_columns(variable_key, metric_suffix):
    """
    Build {model: column} for a metric suffix, e.g. 'r2' or 'adj_r2',
    for ratio or cv_w.
    """
    if variable_key not in ("ratio", "cv_w"):
        raise ValueError("variable_key must be 'ratio' or 'cv_w'.")
    return {m: f"{variable_key}_{m}_{metric_suffix}" for m in MODELS}


def plot_metric_distribution_for_variable(df_all, variable_key, metric_suffix):
    """
    Boxplot + jitter + mean diamonds for a fit-quality metric across the
    three models. metric_suffix is 'r2' or 'adj_r2'.
    """
    cols = _metric_columns(variable_key, metric_suffix)
    require_columns(df_all, list(cols.values()))

    title_name = _r2_title_name(variable_key)
    metric_label = r"$R^2$" if metric_suffix == "r2" else r"adjusted $R^2$"

    data, labels, colors = [], [], []
    for m in MODELS:
        col = cols[m]
        mask = filter_by_n(df_all, variable_key, m)
        values = clean_series(df_all.loc[mask, col])
        data.append(values)
        labels.append(MODEL_LABELS[m])
        colors.append(MODEL_COLORS[m])

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(data, tick_labels=labels, showfliers=False,
                    patch_artist=True, widths=0.55)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    rng = np.random.default_rng(42)
    for i, (values, color) in enumerate(zip(data, colors), start=1):
        x_jitter = rng.normal(loc=i, scale=0.045, size=len(values))
        ax.scatter(x_jitter, values, s=8, alpha=0.15, edgecolors="none", color=color)

    means = [np.nanmean(v) if len(v) else np.nan for v in data]
    ax.scatter(np.arange(1, len(means) + 1), means, marker="D", s=75,
               color="black", label=f"Mean {metric_label}", zorder=5)

    for i, mean_val in enumerate(means, start=1):
        if np.isfinite(mean_val):
            ax.text(i, mean_val + 0.025, f"{mean_val:.3f}", ha="center",
                    va="bottom", fontsize=11, fontweight="bold")

    # adjusted R² can go negative for poor fits; allow some headroom below 0.
    ax.set_ylim(-0.05 if metric_suffix == "adj_r2" else 0.0, 1.05)
    ax.set_xlabel("Fitted model")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Distribution of {title_name} fit quality "
                 f"({metric_label}) | {R2_MODE} station set")
    ax.legend(frameon=True)

    savefig(f"{metric_suffix}_distribution_{variable_key}_{R2_MODE}.png")


def plot_metric_histogram_for_variable(df_all, variable_key, metric_suffix):
    """
    Step histogram (percentage y-axis) for a fit-quality metric across the
    three models. metric_suffix is 'r2' or 'adj_r2'.
    """
    cols = _metric_columns(variable_key, metric_suffix)
    require_columns(df_all, list(cols.values()))

    title_name = _r2_title_name(variable_key)
    metric_label = r"$R^2$" if metric_suffix == "r2" else r"adjusted $R^2$"

    fig, ax = plt.subplots(figsize=(10, 6))

    # adjusted R² can be negative; widen the binning range for it.
    lo = -0.5 if metric_suffix == "adj_r2" else 0.0
    bins = np.linspace(lo, 1, 51)

    for m in MODELS:
        col = cols[m]
        mask = filter_by_n(df_all, variable_key, m)
        values = clean_series(df_all.loc[mask, col])

        if len(values) == 0:
            print(f"[WARNING] No valid values for {col}")
            continue

        weights = np.ones(len(values)) * 100.0 / len(values)
        ax.hist(values, bins=bins, weights=weights, histtype="step",
                linewidth=2.2, color=MODEL_COLORS[m],
                label=f"{MODEL_LABELS[m]}, mean={values.mean():.3f}")

    ax.set_xlim(lo, 1)
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Percentage of timestep-specific fits (%)")
    ax.set_title(f"Histogram of {title_name} {metric_label} "
                 f"| {R2_MODE} station set")
    ax.legend(frameon=True)

    savefig(f"{metric_suffix}_percentage_histogram_{variable_key}_{R2_MODE}.png")


# ============================================================
# 3) BEST-MODEL BAR CHART ACROSS ALL STATIONS
# ============================================================

def _best_model_per_row(df, variable_key):
    """
    Determine the best (lowest-AIC) model per timestep.
    Returns a pandas Series of model names (or NaN), already n-filtered.
    """
    aic_cols = {m: f"{variable_key}_{m}_aic" for m in MODELS}
    have_aic = all(c in df.columns for c in aic_cols.values())

    if not have_aic:
        return None

    aic_frame = pd.DataFrame(index=df.index)
    for m in MODELS:
        col = aic_cols[m]
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = filter_by_n(df, variable_key, m)
        vals = vals.where(mask, np.nan)
        aic_frame[m] = vals

    valid_any = aic_frame.notna().any(axis=1)
    best = pd.Series(index=df.index, dtype=object)
    best[valid_any] = aic_frame.loc[valid_any].idxmin(axis=1)
    return best


def plot_best_model_bar_for_variable(df_all, variable_key):
    """
    Bar chart of how often each model is selected (lowest AIC) across all
    stations and timesteps, for one variable.
    """
    best = _best_model_per_row(df_all, variable_key)
    if best is None:
        print(f"[WARNING] No AIC columns for {variable_key}; skipping best-model bar.")
        return

    counts = best.value_counts()
    total = int(counts.sum())
    if total == 0:
        print(f"[WARNING] No valid best-model selections for {variable_key}.")
        return

    title_name = _r2_title_name(variable_key)

    fig, ax = plt.subplots(figsize=(8, 6))

    heights = [counts.get(m, 0) for m in MODELS]
    pct = [100.0 * h / total for h in heights]
    colors = [MODEL_COLORS[m] for m in MODELS]
    xlabels = [MODEL_LABELS[m] for m in MODELS]

    bars = ax.bar(xlabels, pct, color=colors, alpha=0.8, width=0.6)

    for bar, h, p in zip(bars, heights, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.8,
                f"{p:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_ylim(0, max(pct) * 1.18 if max(pct) > 0 else 1)
    ax.set_ylabel("Share of timesteps selected as best (%)")
    ax.set_xlabel("Fitted model (lowest AIC)")
    ax.set_title(f"Best-fit model for {title_name} by AIC\n"
                 f"all stations, all timesteps | {R2_MODE}")

    savefig(f"best_model_aic_barchart_{variable_key}_{R2_MODE}.png")


def plot_reversed_cdf_for_variable(df_all, variable_key, metric_suffix):
    """
    Approach B — Reversed (survival) CDF.

    y-axis: percentage of fits where metric >= threshold.
    Reads as: "X% of all timestep fits achieved at least R²=0.9."
    """
    cols = _metric_columns(variable_key, metric_suffix)
    have = [c for c in cols.values() if c in df_all.columns]
    if not have:
        print(f"[INFO] No {metric_suffix} columns for {variable_key}; skipping reversed CDF.")
        return

    title_name   = _r2_title_name(variable_key)
    metric_label = r"$R^2$" if metric_suffix == "r2" else r"adjusted $R^2$"

    x_lo = -0.5 if metric_suffix == "adj_r2" else 0.0
    thresh = np.linspace(x_lo, 1.0, 1000)

    LSTYLES = {"linear": "-", "quadratic": "--", "exponential": ":"}

    fig, ax = plt.subplots(figsize=(11, 6))

    for m in MODELS:
        col = cols[m]
        if col not in df_all.columns:
            continue
        mask = filter_by_n(df_all, variable_key, m)
        vals = clean_series(df_all.loc[mask, col])
        if len(vals) == 0:
            continue

        frac = np.array([(vals >= t).mean() * 100.0 for t in thresh])

        # BUG FIX 3: restored missing closing parenthesis in label
        ax.plot(thresh, frac,
                color=MODEL_COLORS[m],
                linewidth=2.2,
                linestyle=LSTYLES[m],
                label=f"{MODEL_LABELS[m]}  (mean={vals.mean():.3f})")

    ref_lines = [0.80, 0.90, 0.95]
    for xref in ref_lines:
        ax.axvline(xref, color="grey", linewidth=0.9, linestyle="--", alpha=0.6)
        ax.text(xref + 0.004, 97, f"{int(xref*100)}%",
                fontsize=9, color="grey", va="top", ha="left")

    ax.set_xlim(x_lo, 1.0)
    ax.set_ylim(0, 101)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.set_xlabel(f"{metric_label} threshold", fontsize=LABEL_SIZE)
    ax.set_ylabel(f"% of fits with {metric_label} ≥ threshold", fontsize=LABEL_SIZE)
    ax.set_title(
        f"{SPECIES} {title_name} {metric_label} — survival curve\n"
        f"{R2_MODE} station set",
        fontsize=TITLE_SIZE,
    )
    ax.legend(fontsize=LEGEND_SIZE, frameon=True)
    ax.grid(alpha=0.25)

    savefig(f"survival_cdf_{metric_suffix}_{variable_key}_{R2_MODE}.png")


def plot_dual_panel_histogram_for_variable(df_all, variable_key, metric_suffix):
    """
    Approach C — Dual-panel histogram: full range (left) + tail zoom (right).

    Left panel: shows the full picture including any low-R² tail.
    Right panel: resolves fine structure inside the high-quality region.
    """
    cols = _metric_columns(variable_key, metric_suffix)
    have = [c for c in cols.values() if c in df_all.columns]
    if not have:
        print(f"[INFO] No {metric_suffix} columns for {variable_key}; skipping dual-panel.")
        return

    title_name   = _r2_title_name(variable_key)
    metric_label = r"$R^2$" if metric_suffix == "r2" else r"adjusted $R^2$"

    x_lo      = -0.5 if metric_suffix == "adj_r2" else 0.0
    zoom_lo   = 0.80
    bins_full = np.linspace(x_lo, 1.0, 26)
    bins_zoom = np.linspace(zoom_lo, 1.0, 41)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6),
        gridspec_kw={"width_ratios": [1, 1.6]},
    )

    for m in MODELS:
        col = cols[m]
        if col not in df_all.columns:
            continue
        mask = filter_by_n(df_all, variable_key, m)
        vals = clean_series(df_all.loc[mask, col])
        if len(vals) == 0:
            continue

        color  = MODEL_COLORS[m]
        w_full = np.ones(len(vals)) * 100.0 / len(vals)

        ax1.hist(vals, bins=bins_full, weights=w_full,
                 histtype="stepfilled", color=color, alpha=0.28, linewidth=0)
        ax1.hist(vals, bins=bins_full, weights=w_full,
                 histtype="step", color=color, linewidth=2.0,
                 label=MODEL_LABELS[m])

        vals_z = vals[vals >= zoom_lo]
        if len(vals_z) == 0:
            continue
        pct_in_zoom = 100.0 * len(vals_z) / len(vals)
        w_zoom = np.ones(len(vals_z)) * 100.0 / len(vals_z)

        ax2.hist(vals_z, bins=bins_zoom, weights=w_zoom,
                 histtype="stepfilled", color=color, alpha=0.28, linewidth=0)
        ax2.hist(vals_z, bins=bins_zoom, weights=w_zoom,
                 histtype="step", color=color, linewidth=2.0,
                 label=(f"{MODEL_LABELS[m]}\n"
                        f"mean={vals.mean():.3f}\n"
                        f"≥{int(zoom_lo*100)}%: {pct_in_zoom:.1f}% of fits"))

    for ax, title_sfx in [
        (ax1, "full range (−0.5 to 1.0)" if metric_suffix == "adj_r2" else "full range (0 to 1.0)"),
        (ax2, f"zoom ({zoom_lo:.2f}–1.0) | bin width = 0.005"),
    ]:
        ax.set_xlabel(metric_label, fontsize=LABEL_SIZE)
        ax.set_ylabel("% of timestep fits", fontsize=LABEL_SIZE)
        ax.set_title(title_sfx, fontsize=LABEL_SIZE)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    ax2.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax2.xaxis.set_minor_locator(mticker.MultipleLocator(0.01))

    ax1.legend(fontsize=LEGEND_SIZE, frameon=True)
    ax2.legend(fontsize=LEGEND_SIZE - 1, frameon=True, loc="upper left")

    fig.suptitle(
        f"{SPECIES} {title_name} {metric_label} distribution\n"
        f"{R2_MODE} station set",
        fontsize=TITLE_SIZE,
        y=1.01,
    )

    savefig(f"dual_panel_{metric_suffix}_{variable_key}_{R2_MODE}.png")


def run_r2_distribution_plots():
    csv_files = get_r2_csv_files()

    print(f"R² distribution mode: {R2_MODE}")
    print(f"Number of CSV files used: {len(csv_files)}")

    # BUG FIX 2: use "distribution" to load only metric columns and save RAM
    df_all = load_multiple_station_csvs(csv_files, usecols="distribution")

    has_adj = lambda var: f"{var}_linear_adj_r2" in df_all.columns
    has_aic = lambda var: f"{var}_linear_aic" in df_all.columns

    for variable_key in ["ratio", "cv_w"]:

        # Original step histograms
        plot_metric_histogram_for_variable(df_all, variable_key, "r2")

        if has_adj(variable_key):
            plot_metric_histogram_for_variable(df_all, variable_key, "adj_r2")
        else:
            print(f"[INFO] No adj_r2 columns for {variable_key}; "
                  f"run backfill_fit_metrics.py first.")

        # B — Survival / reversed-CDF
        plot_reversed_cdf_for_variable(df_all, variable_key, "r2")

        if has_adj(variable_key):
            plot_reversed_cdf_for_variable(df_all, variable_key, "adj_r2")

        # C — Dual-panel histogram (full + zoom)
        plot_dual_panel_histogram_for_variable(df_all, variable_key, "r2")

        if has_adj(variable_key):
            plot_dual_panel_histogram_for_variable(df_all, variable_key, "adj_r2")

        # Best-model bar chart
        if has_aic(variable_key):
            plot_best_model_bar_for_variable(df_all, variable_key)
        else:
            print(f"[INFO] No AIC columns for {variable_key}; "
                  f"skipping best-model bar chart.")

        # Boxplot versions (uncomment if wanted)
        # plot_metric_distribution_for_variable(df_all, variable_key, "r2")
        # plot_metric_distribution_for_variable(df_all, variable_key, "adj_r2")


# ============================================================
# MAIN
# ============================================================

def main():
    start = time.time()
    print("Start:", dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    print("Running fitted-parameter time-series plots")
    print("=" * 80)
    run_parameter_timeseries_plots()

    print("\n" + "=" * 80)
    print("Running R² / adjusted-R² distribution and best-model plots")
    print("=" * 80)
    #run_r2_distribution_plots()

    print("\nAll plots completed.")
    print(f"Output directory: {OUT_DIR}")
    end = time.time()
    print("End:", dt.datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Execution time: {(end - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
# %%