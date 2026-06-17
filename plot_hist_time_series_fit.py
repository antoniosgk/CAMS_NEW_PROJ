#%%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# USER SETTINGS
# ============================================================

CSV_DIR = Path("/mnt/store01/agkiokas/CAMS/derived_station_metrics/")

OUT_DIR = Path("/mnt/store01/agkiokas/CAMS/derived_station_metrics/parameter_timeseries_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = "O3"

# ------------------------------------------------------------
# 1) Time-series plots of fitted parameters
# ------------------------------------------------------------
# Use one station:
TIMESERIES_STATIONS = ["1006A"]
#
# Or multiple stations:
#TIMESERIES_STATIONS = ["1002A", "1003A", "1004A"]

# If True, all selected stations are plotted on the same figure.
# If False, one figure is created per station.
PLOT_STATIONS_TOGETHER = True

# Optional time subset
PLOT_FULL_PERIOD = True
START_DATE = "2005-05-20"
END_DATE = "2005-06-20"

# ------------------------------------------------------------
# 2) R² distribution plots
# ------------------------------------------------------------
# Options:
#   "one"       -> use R2_STATIONS
#   "selected"  -> use R2_STATIONS
#   "all"       -> use all CSV files in CSV_DIR
R2_MODE = "all"

R2_STATIONS = ["1002A","1003A","1004A"]

# CSV filename pattern
CSV_PATTERN = "*_sector_ratio_cv_fits.csv"

# Figure settings
DPI = 300

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


def station_id_from_csv(csv_file):
    """
    Example:
        1006A_sector_ratio_cv_fits.csv -> 1006A
    """
    return csv_file.name.replace("_sector_ratio_cv_fits.csv", "")


def get_csv_file_for_station(station_id):
    return CSV_DIR / f"{station_id}_{SPECIES}_sector_ratio_cv_fits.csv"


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

    # Case: time_key like 20050516_0000
    if df[dt_col].str.contains("_").any():
        df["plot_datetime"] = pd.to_datetime(df[dt_col], format="%Y%m%d_%H%M", errors="coerce")
    else:
        df["plot_datetime"] = pd.to_datetime(df[dt_col], errors="coerce")

        # Case: timestamp like 200505160000
        if df["plot_datetime"].isna().all():
            df["plot_datetime"] = pd.to_datetime(df[dt_col], format="%Y%m%d%H%M", errors="coerce")

    return df


def load_station_csv(csv_file):
    station_id = station_id_from_csv(csv_file)

    df = pd.read_csv(csv_file)
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


def load_multiple_station_csvs(csv_files):
    dfs = []

    for f in csv_files:
        try:
            df = load_station_csv(f)
            dfs.append(df)
            print(f"Loaded {f.name}: {len(df):,} rows")
        except Exception as e:
            print(f"[WARNING] Could not load {f.name}: {e}")

    if len(dfs) == 0:
        raise ValueError("No valid CSV files loaded.")

    return pd.concat(dfs, ignore_index=True)


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


# ============================================================
# PARAMETER DEFINITIONS
# ============================================================
# Important:
#
# Linear model:
#     y = a + b*x
# Shape parameter plotted: b
#
# Quadratic model:
#     y = a + b*x + c*x²
# Shape parameter plotted: c
#
# Exponential model:
#     y = a + b*exp(c*x)
# Shape parameter plotted: c
#
# For quadratic, c is curvature.
# For exponential, c is the exponential rate parameter.
# These are not exactly the same type of "slope" as the linear b,
# but they describe the shape/change across sectors.
# ============================================================

PARAMETER_SPECS = {
    "ratio": {
        "title": "Ratio",
        "ylabel": "Shape parameter value",
        "columns": {
            "Linear b": "ratio_linear_b",
            "Quadratic c": "ratio_quadratic_c",
            "Exponential c": "ratio_exponential_c",
        },
    },
    "cv_w": {
        "title": "Weighted CV",
        "ylabel": "Shape parameter value",
        "columns": {
            "Linear b": "cv_w_linear_b",
            "Quadratic c": "cv_w_quadratic_c",
            "Exponential c": "cv_w_exponential_c",
        },
        "scale": 100,
        "scaled_ylabel": "Shape parameter value, CV percentage units",
    },
    "mean_w": {
        "title": "Weighted sector mean concentration",
        "ylabel": "Shape parameter value",
        "columns": {
            "Linear b": "mean_w_linear_b",
            "Quadratic c": "mean_w_quadratic_c",
            "Exponential c": "mean_w_exponential_c",
        },
    },
}


# ============================================================
# 1) TIME SERIES OF SHAPE PARAMETERS
# ============================================================

def plot_parameter_timeseries_for_variable_one_station(df, variable_key):
    """
    Plot shape parameters for one variable and one station.
    One figure contains:
        linear b
        quadratic c
        exponential c
    """

    spec = PARAMETER_SPECS[variable_key]
    station_id = df["station_id"].iloc[0]

    cols = list(spec["columns"].values())
    require_columns(df, cols)

    scale = spec.get("scale", 1)

    fig, ax = plt.subplots(figsize=(14, 6))

    for label, col in spec["columns"].items():
        ax.plot(
            df["plot_datetime"],
            df[col] * scale,
            linewidth=1.1,
            label=label,
        )

    ax.axhline(0, linestyle="--", linewidth=1.4)

    ax.set_xlabel("Time")

    if "scaled_ylabel" in spec and scale != 1:
        ax.set_ylabel(spec["scaled_ylabel"])
    else:
        ax.set_ylabel(spec["ylabel"])

    ax.set_title(
        f"{SPECIES} {spec['title']} fitted parameter time series | {station_id}"
    )

    nice_time_axis(ax)
    ax.legend(frameon=True)

    savefig(f"{station_id}_{variable_key}_parameter_timeseries.png")


def plot_parameter_timeseries_for_variable_multiple_stations(df_all, variable_key):
    """
    Plot one selected shape parameter per model type,
    with multiple stations shown on the same figure.

    This creates three figures for the variable:
        1. linear b for all stations
        2. quadratic c for all stations
        3. exponential c for all stations
    """

    spec = PARAMETER_SPECS[variable_key]
    cols = list(spec["columns"].values())
    require_columns(df_all, cols)

    scale = spec.get("scale", 1)

    for parameter_label, col in spec["columns"].items():
        fig, ax = plt.subplots(figsize=(14, 6))

        for station_id, g in df_all.groupby("station_id"):
            ax.plot(
                g["plot_datetime"],
                g[col] * scale,
                linewidth=1.0,
                alpha=0.85,
                label=station_id,
            )

        ax.axhline(0, linestyle="--", linewidth=1.4)

        ax.set_xlabel("Time")

        if "scaled_ylabel" in spec and scale != 1:
            ax.set_ylabel(spec["scaled_ylabel"])
        else:
            ax.set_ylabel(spec["ylabel"])

        ax.set_title(
            f"{SPECIES} {spec['title']} | {parameter_label} time series"
        )

        nice_time_axis(ax)
        ax.legend(frameon=True, ncol=2)

        safe_param = parameter_label.lower().replace(" ", "_")
        savefig(f"multi_station_{variable_key}_{safe_param}_timeseries.png")


def run_parameter_timeseries_plots():
    """
    Run part 1:
    Time series of fitted parameters for selected station(s).
    """

    csv_files = get_selected_csv_files(TIMESERIES_STATIONS)

    if len(csv_files) == 0:
        print("[WARNING] No CSV files found for TIMESERIES_STATIONS.")
        return

    df_all = load_multiple_station_csvs(csv_files)

    if PLOT_STATIONS_TOGETHER:
        for variable_key in ["ratio", "cv_w", "mean_w"]:
            plot_parameter_timeseries_for_variable_multiple_stations(df_all, variable_key)
    else:
        for station_id, g in df_all.groupby("station_id"):
            for variable_key in ["ratio", "cv_w", "mean_w"]:
                plot_parameter_timeseries_for_variable_one_station(g, variable_key)


# ============================================================
# 2) R² DISTRIBUTION PLOTS
# ============================================================

def get_r2_csv_files():
    """
    Select CSV files for R² distribution plots.
    """

    if R2_MODE == "all":
        files = get_available_csv_files()

    elif R2_MODE in ["one", "selected"]:
        files = get_selected_csv_files(R2_STATIONS)

    else:
        raise ValueError("Invalid R2_MODE. Use 'one', 'selected', or 'all'.")

    if len(files) == 0:
        raise ValueError("No CSV files selected for R² distribution plots.")

    return files


def plot_r2_distribution_for_variable(df_all, variable_key):
    """
    Plot distribution of R² values for one variable:
        ratio or cv_w

    One plot contains the three model fits:
        linear
        quadratic
        exponential
    """

    if variable_key == "ratio":
        cols = {
            "Linear": "ratio_linear_r2",
            "Quadratic": "ratio_quadratic_r2",
            "Exponential": "ratio_exponential_r2",
        }
        title_name = "ratio"

    elif variable_key == "cv_w":
        cols = {
            "Linear": "cv_w_linear_r2",
            "Quadratic": "cv_w_quadratic_r2",
            "Exponential": "cv_w_exponential_r2",
        }
        title_name = "CV"

    else:
        raise ValueError("variable_key must be 'ratio' or 'cv_w'.")

    require_columns(df_all, list(cols.values()))

    data = []
    labels = []

    for label, col in cols.items():
        values = clean_series(df_all[col])
        data.append(values)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(
        data,
        labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.55,
    )

    rng = np.random.default_rng(42)

    for i, values in enumerate(data, start=1):
        x_jitter = rng.normal(loc=i, scale=0.045, size=len(values))

        ax.scatter(
            x_jitter,
            values,
            s=8,
            alpha=0.15,
            edgecolors="none",
        )

    means = [np.nanmean(values) for values in data]

    ax.scatter(
        np.arange(1, len(means) + 1),
        means,
        marker="D",
        s=75,
        label="Mean R²",
        zorder=5,
    )

    for i, mean_val in enumerate(means, start=1):
        ax.text(
            i,
            mean_val + 0.025,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Fitted model")
    ax.set_ylabel(r"$R^2$")
    ax.set_title(
        f"Distribution of {title_name} fit quality | {R2_MODE} station set"
    )

    ax.legend(frameon=True)

    savefig(f"R2_distribution_{variable_key}_{R2_MODE}.png")



def plot_r2_histogram_for_variable(df_all, variable_key):
    """
    Histogram of R² distributions.

    Same style as before:
        - step histograms
        - one plot with linear, quadratic, exponential

    Difference:
        - y-axis shows percentage of timestep-specific fits, not counts.
    """

    if variable_key == "ratio":
        cols = {
            "Linear": "ratio_linear_r2",
            "Quadratic": "ratio_quadratic_r2",
            "Exponential": "ratio_exponential_r2",
        }
        title_name = "ratio"

    elif variable_key == "cv_w":
        cols = {
            "Linear": "cv_w_linear_r2",
            "Quadratic": "cv_w_quadratic_r2",
            "Exponential": "cv_w_exponential_r2",
        }
        title_name = "CV"

    else:
        raise ValueError("variable_key must be 'ratio' or 'cv_w'.")

    require_columns(df_all, list(cols.values()))

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 51)

    for label, col in cols.items():
        values = clean_series(df_all[col])

        if len(values) == 0:
            print(f"[WARNING] No valid values for {col}")
            continue

        # Each value contributes equally so that the histogram sums to 100%
        weights = np.ones(len(values)) * 100.0 / len(values)

        ax.hist(
            values,
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=2.2,
            label=f"{label}, mean={values.mean():.3f}",
        )

    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$R^2$")
    ax.set_ylabel("Percentage of timestep-specific fits (%)")
    ax.set_title(
        f"Histogram of {title_name} $R^2$ values | {R2_MODE} station set"
    )

    ax.legend(frameon=True)

    savefig(f"R2_percentage_histogram_{variable_key}_{R2_MODE}.png")


def run_r2_distribution_plots():
    """
    Run part 2:
    R² distribution plots for ratio and CV.
    """

    csv_files = get_r2_csv_files()

    print(f"R² distribution mode: {R2_MODE}")
    print(f"Number of CSV files used: {len(csv_files)}")

    df_all = load_multiple_station_csvs(csv_files)

    #plot_r2_distribution_for_variable(df_all, "ratio")
    #plot_r2_distribution_for_variable(df_all, "cv_w")

    # Optional histogram versions
    plot_r2_histogram_for_variable(df_all, "ratio")
    plot_r2_histogram_for_variable(df_all, "cv_w")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("Running fitted-parameter time-series plots")
    print("=" * 80)

    run_parameter_timeseries_plots()

    print("\n" + "=" * 80)
    print("Running R² distribution plots")
    print("=" * 80)

    run_r2_distribution_plots()

    print("\nAll plots completed.")
    print(f"Output directory: {OUT_DIR}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
# %%
