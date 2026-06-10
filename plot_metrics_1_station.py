#%%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# USER SETTINGS
# ============================================================

CSV_FILE = Path("/home/agkiokas/CAMS/derived_station_metrics/1006A_sector_ratio_cv_fits.csv")

OUT_DIR = Path("/home/agkiokas/CAMS/derived_station_metrics/plots_1006A")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATION_ID = "1006A"
SPECIES = "O3"

SECTORS = list(range(1, 11))

# Optional time subset
PLOT_FULL_PERIOD = True
START_DATE = "2005-05-20"
END_DATE = "2005-06-20"

# Example timestep for fitted curves
EXAMPLE_TIMESTEP_MODE = "manual"   # "auto" or "manual"
MANUAL_TIMESTEP = "2005-05-20 03:00:00"     # example: "2005-05-20 03:00:00"

# Plot settings
DPI = 300

TITLE_SIZE = 17
LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 11
color_ratio='black'
color_cv='red'
alpha=0.8
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


def get_ratio_cols():
    return [f"ratio_C{s}" for s in SECTORS]


def get_cv_cols():
    return [f"cv_w_C{s}" for s in SECTORS]


def get_mean_cols():
    return [f"mean_w_C{s}" for s in SECTORS]


def get_std_cols():
    return [f"std_w_C{s}" for s in SECTORS]


def get_sector_x():
    return np.array(SECTORS, dtype=float)


def fmt_param(value, ndigits=4):
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{ndigits}g}"


def mean_valid(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).mean()


def nice_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=35)


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def get_datetime_column(df):
    """
    Try to identify and parse the datetime column from the derived CSV.
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


def load_and_prepare_data():
    df = pd.read_csv(CSV_FILE)
    df = get_datetime_column(df)

    df = df.dropna(subset=["plot_datetime"]).copy()
    df = df.sort_values("plot_datetime").copy()

    if not PLOT_FULL_PERIOD:
        df = df[
            (df["plot_datetime"] >= pd.to_datetime(START_DATE)) &
            (df["plot_datetime"] <= pd.to_datetime(END_DATE))
        ].copy()

    ratio_cols = get_ratio_cols()
    cv_cols = get_cv_cols()

    required_cols = (
        ratio_cols
        + cv_cols
        + [
            "R_C10_minus_1",
            "R_C10_minus_R_C1",
            "CV_C10_minus_CV_C1",

            "ratio_linear_a",
            "ratio_linear_b",
            "ratio_linear_r2",
            "ratio_quadratic_a",
            "ratio_quadratic_b",
            "ratio_quadratic_c",
            "ratio_quadratic_r2",
            "ratio_exponential_a",
            "ratio_exponential_b",
            "ratio_exponential_c",
            "ratio_exponential_r2",

            "cv_w_linear_a",
            "cv_w_linear_b",
            "cv_w_linear_r2",
            "cv_w_quadratic_a",
            "cv_w_quadratic_b",
            "cv_w_quadratic_c",
            "cv_w_quadratic_r2",
            "cv_w_exponential_a",
            "cv_w_exponential_b",
            "cv_w_exponential_c",
            "cv_w_exponential_r2",
        ]
    )

    require_columns(df, required_cols)

    print(f"Loaded rows: {len(df):,}")
    print(f"Time range: {df['plot_datetime'].min()} to {df['plot_datetime'].max()}")

    return df


# ============================================================
# 1. BASIC SECTOR PROFILE PLOTS
# ============================================================

def plot_mean_ratio_profile(df):
    ratio_cols = get_ratio_cols()

    ratio_mean = df[ratio_cols].mean()
    ratio_p25 = df[ratio_cols].quantile(0.25)
    ratio_p75 = df[ratio_cols].quantile(0.75)

    x = get_sector_x()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, ratio_mean.values, marker="o", linewidth=2.5, label="Mean ratio")
    ax.fill_between(
        x,
        ratio_p25.values,
        ratio_p75.values,
        alpha=0.25,
        label="Interquartile range",
    )

    ax.axhline(1.0, linestyle="--", linewidth=1.8, label="Central grid-cell reference")

    ax.set_xlabel("Sector")
    ax.set_ylabel("Sector mean / central grid-cell value")
    ax.set_title(f"{SPECIES} mean ratio profile across sectors | {STATION_ID}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{s}" for s in SECTORS])
    ax.legend(frameon=True)

    savefig(f"{STATION_ID}_01_mean_ratio_profile.png")


def plot_mean_cv_profile(df):
    cv_cols = get_cv_cols()

    cv_mean = df[cv_cols].mean()
    cv_p25 = df[cv_cols].quantile(0.25)
    cv_p75 = df[cv_cols].quantile(0.75)

    x = get_sector_x()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, cv_mean.values * 100, marker="o", linewidth=2.5, label="Mean CV")
    ax.fill_between(
        x,
        cv_p25.values * 100,
        cv_p75.values * 100,
        alpha=0.25,
        label="Interquartile range",
    )

    ax.set_xlabel("Sector")
    ax.set_ylabel("Weighted CV (%)")
    ax.set_title(f"{SPECIES} spatial variability across sectors | {STATION_ID}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{s}" for s in SECTORS])
    ax.legend(frameon=True)

    savefig(f"{STATION_ID}_02_mean_cv_profile.png")


# ============================================================
# 2. SCALE-SENSITIVITY TIME SERIES AND DISTRIBUTIONS
# ============================================================

def plot_ratio_scale_sensitivity_timeseries(df):
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df["plot_datetime"],
        df["R_C10_minus_1"],
        linewidth=1.2,
        label=r"$R_{C10} - 1$",
    )

    ax.plot(
        df["plot_datetime"],
        df["R_C10_minus_R_C1"],
        linewidth=1.2,
        label=r"$R_{C10} - R_{C1}$",
    )

    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Ratio difference")
    ax.set_title(f"{SPECIES} ratio-based scale sensitivity | {STATION_ID}")

    nice_time_axis(ax)
    ax.legend(frameon=True)

    savefig(f"{STATION_ID}_03_ratio_scale_sensitivity_timeseries.png")


def plot_cv_scale_sensitivity_timeseries(df):
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df["plot_datetime"],
        df["CV_C10_minus_CV_C1"] * 100,
        linewidth=1.2,
        label=r"$CV_{C10} - CV_{C1}$",
    )

    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("CV difference, percentage points")
    ax.set_title(f"{SPECIES} CV-based scale sensitivity | {STATION_ID}")

    nice_time_axis(ax)
    ax.legend(frameon=True)

    savefig(f"{STATION_ID}_04_cv_scale_sensitivity_timeseries.png")


def plot_ratio_metric_distributions(df):
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(
        df["R_C10_minus_1"].dropna(),
        bins=80,
        alpha=0.65,
        label=r"$R_{C10} - 1$",
    )

    ax.hist(
        df["R_C10_minus_R_C1"].dropna(),
        bins=80,
        alpha=0.65,
        label=r"$R_{C10} - R_{C1}$",
    )

    ax.axvline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Ratio difference")
    ax.set_ylabel("Number of timesteps")
    ax.set_title(f"Distribution of ratio scale-sensitivity metrics | {STATION_ID}")
    ax.legend(frameon=True)

    savefig(f"{STATION_ID}_05_ratio_metric_distributions.png")


def plot_cv_metric_distribution(df):
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(
        df["CV_C10_minus_CV_C1"].dropna() * 100,
        bins=80,
        alpha=0.75,
    )

    ax.axvline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("CV difference, percentage points")
    ax.set_ylabel("Number of timesteps")
    ax.set_title(f"Distribution of CV scale sensitivity | {STATION_ID}")

    savefig(f"{STATION_ID}_06_cv_metric_distribution.png")


# ============================================================
# 3. HEATMAPS
# ============================================================

def plot_ratio_heatmap(df, max_heatmap_points=1000):
    ratio_cols = get_ratio_cols()
    heat_df = df.copy()

    if len(heat_df) > max_heatmap_points:
        step = int(np.ceil(len(heat_df) / max_heatmap_points))
        heat_df = heat_df.iloc[::step].copy()

    ratio_matrix = heat_df[ratio_cols].to_numpy().T

    fig, ax = plt.subplots(figsize=(15, 6))

    im = ax.imshow(
        ratio_matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=np.nanpercentile(ratio_matrix, 2),
        vmax=np.nanpercentile(ratio_matrix, 98),
    )

    ax.set_yticks(np.arange(len(SECTORS)))
    ax.set_yticklabels([f"C{s}" for s in SECTORS])

    n_ticks = 8
    tick_positions = np.linspace(0, len(heat_df) - 1, n_ticks).astype(int)
    tick_labels = heat_df["plot_datetime"].iloc[tick_positions].dt.strftime("%b %Y")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right")

    ax.set_xlabel("Time")
    ax.set_ylabel("Sector")
    ax.set_title(f"{SPECIES} ratio heatmap by sector and time | {STATION_ID}")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sector mean / central grid-cell value")

    savefig(f"{STATION_ID}_07_ratio_heatmap_sector_time.png")


def plot_cv_heatmap(df, max_heatmap_points=1000):
    cv_cols = get_cv_cols()
    heat_df = df.copy()

    if len(heat_df) > max_heatmap_points:
        step = int(np.ceil(len(heat_df) / max_heatmap_points))
        heat_df = heat_df.iloc[::step].copy()

    cv_matrix = heat_df[cv_cols].to_numpy().T * 100

    fig, ax = plt.subplots(figsize=(15, 6))

    im = ax.imshow(
        cv_matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=np.nanpercentile(cv_matrix, 2),
        vmax=np.nanpercentile(cv_matrix, 98),
    )

    ax.set_yticks(np.arange(len(SECTORS)))
    ax.set_yticklabels([f"C{s}" for s in SECTORS])

    n_ticks = 8
    tick_positions = np.linspace(0, len(heat_df) - 1, n_ticks).astype(int)
    tick_labels = heat_df["plot_datetime"].iloc[tick_positions].dt.strftime("%b %Y")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right")

    ax.set_xlabel("Time")
    ax.set_ylabel("Sector")
    ax.set_title(f"{SPECIES} weighted CV heatmap by sector and time | {STATION_ID}")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Weighted CV (%)")

    savefig(f"{STATION_ID}_08_cv_heatmap_sector_time.png")


# ============================================================
# 4. FIT QUALITY BAR PLOTS WITH MEAN PARAMETERS
# ============================================================

def plot_time_average_ratio_fit_parameters(df):
    summary = pd.DataFrame({
        "model": ["Linear", "Quadratic", "Exponential"],

        "a_mean": [
            mean_valid(df["ratio_linear_a"]),
            mean_valid(df["ratio_quadratic_a"]),
            mean_valid(df["ratio_exponential_a"]),
        ],

        "b_mean": [
            mean_valid(df["ratio_linear_b"]),
            mean_valid(df["ratio_quadratic_b"]),
            mean_valid(df["ratio_exponential_b"]),
        ],

        "c_mean": [
            np.nan,
            mean_valid(df["ratio_quadratic_c"]),
            mean_valid(df["ratio_exponential_c"]),
        ],

        "r2_mean": [
            mean_valid(df["ratio_linear_r2"]),
            mean_valid(df["ratio_quadratic_r2"]),
            mean_valid(df["ratio_exponential_r2"]),
        ],
    })

    out_csv = OUT_DIR / f"{STATION_ID}_ratio_fit_parameter_time_averages.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(summary["model"], summary["r2_mean"], alpha=0.85)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean R²")
    ax.set_title(f"Time-averaged fit summary for ratio-sector relationships | {STATION_ID}")

    for bar, (_, r) in zip(bars, summary.iterrows()):
        label = (
            f"R²={r['r2_mean']:.3f}\n"
            f"ā={fmt_param(r['a_mean'])}\n"
            f"b̄={fmt_param(r['b_mean'])}"
        )

        if np.isfinite(r["c_mean"]):
            label += f"\nc̄={fmt_param(r['c_mean'])}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    savefig(f"{STATION_ID}_09_ratio_fit_quality_with_mean_parameters.png")


def plot_time_average_cv_fit_parameters(df):
    summary = pd.DataFrame({
        "model": ["Linear", "Quadratic", "Exponential"],

        "a_mean": [
            mean_valid(df["cv_w_linear_a"]),
            mean_valid(df["cv_w_quadratic_a"]),
            mean_valid(df["cv_w_exponential_a"]),
        ],

        "b_mean": [
            mean_valid(df["cv_w_linear_b"]),
            mean_valid(df["cv_w_quadratic_b"]),
            mean_valid(df["cv_w_exponential_b"]),
        ],

        "c_mean": [
            np.nan,
            mean_valid(df["cv_w_quadratic_c"]),
            mean_valid(df["cv_w_exponential_c"]),
        ],

        "r2_mean": [
            mean_valid(df["cv_w_linear_r2"]),
            mean_valid(df["cv_w_quadratic_r2"]),
            mean_valid(df["cv_w_exponential_r2"]),
        ],
    })

    out_csv = OUT_DIR / f"{STATION_ID}_cv_fit_parameter_time_averages.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(summary["model"], summary["r2_mean"], alpha=0.85)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean R²")
    ax.set_title(f"Time-averaged fit summary for CV-sector relationships | {STATION_ID}")

    for bar, (_, r) in zip(bars, summary.iterrows()):
        label = (
            f"R²={r['r2_mean']:.3f}\n"
            f"ā={fmt_param(r['a_mean'])}\n"
            f"b̄={fmt_param(r['b_mean'])}"
        )

        if np.isfinite(r["c_mean"]):
            label += f"\nc̄={fmt_param(r['c_mean'])}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    savefig(f"{STATION_ID}_10_cv_fit_quality_with_mean_parameters.png")


# ============================================================
# 5. EXAMPLE FITTED CURVES WITH PARAMETERS
# ============================================================

def select_example_row(df, metric_col):
    if EXAMPLE_TIMESTEP_MODE == "manual" and MANUAL_TIMESTEP is not None:
        target_time = pd.to_datetime(MANUAL_TIMESTEP)
        idx = (df["plot_datetime"] - target_time).abs().idxmin()
    else:
        idx = df[metric_col].abs().idxmax()

    return df.loc[idx]


def plot_example_ratio_fit_with_parameters(df):
    row = select_example_row(df, "R_C10_minus_R_C1")
    example_time = row["plot_datetime"]

    x = get_sector_x()
    y = np.array([row[f"ratio_C{s}"] for s in SECTORS], dtype=float)

    x_smooth = np.linspace(1, 10, 300)

    lin_y = row["ratio_linear_a"] + row["ratio_linear_b"] * x_smooth

    quad_y = (
        row["ratio_quadratic_a"]
        + row["ratio_quadratic_b"] * x_smooth
        + row["ratio_quadratic_c"] * x_smooth**2
    )

    exp_y = (
        row["ratio_exponential_a"]
        + row["ratio_exponential_b"] * np.exp(row["ratio_exponential_c"] * x_smooth)
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.scatter(x, y, s=90, zorder=5, label="Observed sector ratios")

    ax.plot(
        x_smooth,
        lin_y,
        linewidth=2.2,
        label=(
            "Linear: "
            f"a={fmt_param(row['ratio_linear_a'])}, "
            f"b={fmt_param(row['ratio_linear_b'])}, "
            f"R²={row['ratio_linear_r2']:.3f}"
        ),
    )

    ax.plot(
        x_smooth,
        quad_y,
        linewidth=2.2,
        label=(
            "Quadratic: "
            f"a={fmt_param(row['ratio_quadratic_a'])}, "
            f"b={fmt_param(row['ratio_quadratic_b'])}, "
            f"c={fmt_param(row['ratio_quadratic_c'])}, "
            f"R²={row['ratio_quadratic_r2']:.3f}"
        ),
    )

    if np.isfinite(exp_y).all():
        ax.plot(
            x_smooth,
            exp_y,
            linewidth=2.2,
            label=(
                "Exponential: "
                f"a={fmt_param(row['ratio_exponential_a'])}, "
                f"b={fmt_param(row['ratio_exponential_b'])}, "
                f"c={fmt_param(row['ratio_exponential_c'])}, "
                f"R²={row['ratio_exponential_r2']:.3f}"
            ),
        )

    ax.axhline(1.0, linestyle="--", linewidth=1.5, label="Central grid-cell reference")

    ax.set_xlabel("Sector")
    ax.set_ylabel("Sector mean / central grid-cell value")
    ax.set_title(
        f"{SPECIES} fitted ratio-sector relationship | {STATION_ID}\n"
        f"{example_time:%Y-%m-%d %H:%M}"
    )

    ax.set_xticks(SECTORS)
    ax.set_xticklabels([f"C{s}" for s in SECTORS])
    ax.legend(frameon=True, fontsize=10, loc="best")

    savefig(f"{STATION_ID}_13_example_ratio_fitted_curves_with_parameters.png")


def plot_example_cv_fit_with_parameters(df):
    row = select_example_row(df, "CV_C10_minus_CV_C1")
    example_time = row["plot_datetime"]

    x = get_sector_x()
    y = np.array([row[f"cv_w_C{s}"] * 100 for s in SECTORS], dtype=float)

    x_smooth = np.linspace(1, 10, 300)

    lin_y = (
        row["cv_w_linear_a"]
        + row["cv_w_linear_b"] * x_smooth
    ) * 100

    quad_y = (
        row["cv_w_quadratic_a"]
        + row["cv_w_quadratic_b"] * x_smooth
        + row["cv_w_quadratic_c"] * x_smooth**2
    ) * 100

    exp_y = (
        row["cv_w_exponential_a"]
        + row["cv_w_exponential_b"] * np.exp(row["cv_w_exponential_c"] * x_smooth)
    ) * 100

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.scatter(x, y, s=90, zorder=5, label="Observed weighted CV")

    ax.plot(
        x_smooth,
        lin_y,
        linewidth=2.2,
        label=(
            "Linear: "
            f"a={fmt_param(row['cv_w_linear_a'])}, "
            f"b={fmt_param(row['cv_w_linear_b'])}, "
            f"R²={row['cv_w_linear_r2']:.3f}"
        ),
    )

    ax.plot(
        x_smooth,
        quad_y,
        linewidth=2.2,
        label=(
            "Quadratic: "
            f"a={fmt_param(row['cv_w_quadratic_a'])}, "
            f"b={fmt_param(row['cv_w_quadratic_b'])}, "
            f"c={fmt_param(row['cv_w_quadratic_c'])}, "
            f"R²={row['cv_w_quadratic_r2']:.3f}"
        ),
    )

    if np.isfinite(exp_y).all():
        ax.plot(
            x_smooth,
            exp_y,
            linewidth=2.2,
            label=(
                "Exponential: "
                f"a={fmt_param(row['cv_w_exponential_a'])}, "
                f"b={fmt_param(row['cv_w_exponential_b'])}, "
                f"c={fmt_param(row['cv_w_exponential_c'])}, "
                f"R²={row['cv_w_exponential_r2']:.3f}"
            ),
        )

    ax.set_xlabel("Sector")
    ax.set_ylabel("Weighted CV (%)")
    ax.set_title(
        f"{SPECIES} fitted CV-sector relationship | {STATION_ID}\n"
        f"{example_time:%Y-%m-%d %H:%M}"
    )

    ax.set_xticks(SECTORS)
    ax.set_xticklabels([f"C{s}" for s in SECTORS])
    ax.legend(frameon=True, fontsize=10, loc="best")

    savefig(f"{STATION_ID}_14_example_cv_fitted_curves_with_parameters.png")


# ============================================================
# 6. R² DISTRIBUTION PLOTS
# ============================================================

def plot_r2_distributions_for_ratio_and_cv(df):
    """
    Distribution of timestep-specific R² values for linear, quadratic,
    and exponential fits.
    """

    # -----------------------------
    # Ratio R² distribution
    # -----------------------------
    ratio_r2_cols = [
        "ratio_linear_r2",
        "ratio_quadratic_r2",
        "ratio_exponential_r2",
    ]

    labels = ["Linear", "Quadratic", "Exponential"]

    ratio_data = [
        df[col].replace([np.inf, -np.inf], np.nan).dropna()
        for col in ratio_r2_cols
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(
        ratio_data,
        labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.55,
    )

    rng = np.random.default_rng(42)

    for i, values in enumerate(ratio_data, start=1):
        x_jitter = rng.normal(loc=i, scale=0.045, size=len(values))
        ax.scatter(
            x_jitter,
            values,
            s=8,
            alpha=0.18,
            edgecolors="none",
        )

    means = [np.nanmean(values) for values in ratio_data]

    ax.scatter(
        np.arange(1, len(means) + 1),
        means,
        marker="D",
        s=70,
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
    ax.set_title(f"Temporal distribution of ratio fit quality | {STATION_ID}")
    ax.legend(frameon=True)

    savefig(f"{STATION_ID}_35_ratio_r2_distribution_linear_quadratic_exponential.png")

    # -----------------------------
    # CV R² distribution
    # -----------------------------
    cv_r2_cols = [
        "cv_w_linear_r2",
        "cv_w_quadratic_r2",
        "cv_w_exponential_r2",
    ]

    cv_data = [
        df[col].replace([np.inf, -np.inf], np.nan).dropna()
        for col in cv_r2_cols
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(
        cv_data,
        labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.55,
    )

    rng = np.random.default_rng(42)

    for i, values in enumerate(cv_data, start=1):
        x_jitter = rng.normal(loc=i, scale=0.045, size=len(values))
        ax.scatter(
            x_jitter,
            values,
            s=8,
            alpha=0.18,
            edgecolors="none",
        )

    means = [np.nanmean(values) for values in cv_data]

    ax.scatter(
        np.arange(1, len(means) + 1),
        means,
        marker="D",
        s=70,
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
    ax.set_title(f"Temporal distribution of CV fit quality | {STATION_ID}")
    ax.legend(frameon=True)

    savefig(f"{STATION_ID}_36_cv_r2_distribution_linear_quadratic_exponential.png")


# ============================================================
# 7. EXTRA RATIO METRICS
# ============================================================

def plot_mean_absolute_ratio_deviation(df):
    values = []

    for s in SECTORS:
        col = f"ratio_C{s}"
        values.append((df[col] - 1.0).abs().mean())

    values = np.array(values)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(SECTORS, values * 100, marker="o", linewidth=2.5)

    ax.set_xticks(SECTORS)
    ax.set_xticklabels([f"C{s}" for s in SECTORS])

    ax.set_xlabel("Sector")
    ax.set_ylabel("Mean absolute deviation from center (%)")
    ax.set_title(f"{SPECIES} mean absolute ratio deviation by sector | {STATION_ID}")

    savefig(f"{STATION_ID}_19_mean_absolute_ratio_deviation_by_sector.png")


def plot_frequency_large_ratio_deviations(df, thresholds=(0.05, 0.10)):
    fig, ax = plt.subplots(figsize=(10, 6))

    for tau in thresholds:
        freq = []

        for s in SECTORS:
            col = f"ratio_C{s}"
            freq.append(((df[col] - 1.0).abs() > tau).mean() * 100)

        ax.plot(
            SECTORS,
            freq,
            marker="o",
            linewidth=2.5,
            label=f"|R - 1| > {tau*100:.0f}%",
        )

    ax.set_xticks(SECTORS)
    ax.set_xticklabels([f"C{s}" for s in SECTORS])

    ax.set_xlabel("Sector")
    ax.set_ylabel("Timesteps exceeding threshold (%)")
    ax.set_title(f"Frequency of large sector-to-center deviations | {STATION_ID}")

    ax.legend(frameon=True)

    savefig(f"{STATION_ID}_20_frequency_large_ratio_deviations.png")


def plot_sector_bias_relative_to_center(df):
    bias = []

    for s in SECTORS:
        col = f"ratio_C{s}"
        bias.append((df[col] - 1.0).mean())

    bias = np.array(bias)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar([f"C{s}" for s in SECTORS], bias * 100, alpha=0.85)

    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Sector")
    ax.set_ylabel("Mean signed bias (%)")
    ax.set_title(f"{SPECIES} sector bias relative to central grid cell | {STATION_ID}")

    for bar, value in zip(bars, bias * 100):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
        )

    savefig(f"{STATION_ID}_21_sector_bias_relative_to_center.png")


def plot_positive_negative_ratio_balance(df):
    above = []
    below = []
    equal = []

    for s in SECTORS:
        col = f"ratio_C{s}"

        above.append((df[col] > 1.0).mean() * 100)
        below.append((df[col] < 1.0).mean() * 100)
        equal.append((df[col] == 1.0).mean() * 100)

    above = np.array(above)
    below = np.array(below)
    equal = np.array(equal)

    x = np.arange(len(SECTORS))

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(x, below, label="Below center")
    ax.bar(x, above, bottom=below, label="Above center")

    if equal.sum() > 0:
        ax.bar(x, equal, bottom=below + above, label="Equal to center")

    ax.axhline(50, linestyle="--", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{s}" for s in SECTORS])

    ax.set_xlabel("Sector")
    ax.set_ylabel("Percentage of timesteps (%)")
    ax.set_title(f"Positive/negative sector deviation balance | {STATION_ID}")

    ax.legend(frameon=True)

    savefig(f"{STATION_ID}_22_positive_negative_ratio_balance.png")


# ============================================================
# 8. EXTRA CV METRICS
# ============================================================

def plot_cv_growth_factor(df, min_cv=1e-8):
    df = df.copy()

    valid = df["cv_w_C1"] > min_cv

    df["CV_growth_factor_C10_C1"] = np.nan
    df.loc[valid, "CV_growth_factor_C10_C1"] = (
        df.loc[valid, "cv_w_C10"] / df.loc[valid, "cv_w_C1"]
    )

    out_metric = OUT_DIR / f"{STATION_ID}_cv_growth_factor_metric.csv"
    df[["plot_datetime", "CV_growth_factor_C10_C1"]].to_csv(out_metric, index=False)
    print(f"Saved: {out_metric}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        df["CV_growth_factor_C10_C1"].replace([np.inf, -np.inf], np.nan).dropna(),
        bins=80,
        alpha=0.75,
    )

    ax.axvline(1, linestyle="--", linewidth=1.5)

    ax.set_xlabel(r"$CV_{C10} / CV_{C1}$")
    ax.set_ylabel("Number of timesteps")
    ax.set_title(f"Distribution of CV growth factor | {STATION_ID}")

    savefig(f"{STATION_ID}_23_cv_growth_factor_distribution.png")

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df["plot_datetime"],
        df["CV_growth_factor_C10_C1"],
        linewidth=1.0,
    )

    ax.axhline(1, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$CV_{C10} / CV_{C1}$")
    ax.set_title(f"Time series of CV growth factor | {STATION_ID}")

    nice_time_axis(ax)

    savefig(f"{STATION_ID}_24_cv_growth_factor_timeseries.png")


def plot_normalized_cv_increase(df, min_cv=1e-8):
    df = df.copy()

    valid = df["cv_w_C1"] > min_cv

    df["normalized_CV_increase_C10_C1"] = np.nan
    df.loc[valid, "normalized_CV_increase_C10_C1"] = (
        (df.loc[valid, "cv_w_C10"] - df.loc[valid, "cv_w_C1"])
        / df.loc[valid, "cv_w_C1"]
    )

    out_metric = OUT_DIR / f"{STATION_ID}_normalized_cv_increase_metric.csv"
    df[["plot_datetime", "normalized_CV_increase_C10_C1"]].to_csv(out_metric, index=False)
    print(f"Saved: {out_metric}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        df["normalized_CV_increase_C10_C1"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        * 100,
        bins=80,
        alpha=0.75,
    )

    ax.axvline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Normalized CV increase (%)")
    ax.set_ylabel("Number of timesteps")
    ax.set_title(f"Distribution of normalized CV increase | {STATION_ID}")

    savefig(f"{STATION_ID}_25_normalized_cv_increase_distribution.png")

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df["plot_datetime"],
        df["normalized_CV_increase_C10_C1"] * 100,
        linewidth=1.0,
    )

    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized CV increase (%)")
    ax.set_title(f"Time series of normalized CV increase | {STATION_ID}")

    nice_time_axis(ax)

    savefig(f"{STATION_ID}_26_normalized_cv_increase_timeseries.png")


# ============================================================
# 9. SLOPE AND CURVATURE PLOTS
# ============================================================

def plot_slope_distributions_and_timeseries(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df["ratio_linear_b"].dropna(), bins=80, alpha=0.75)
    ax.axvline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Ratio linear slope")
    ax.set_ylabel("Number of timesteps")
    ax.set_title(f"Distribution of ratio-sector linear slope | {STATION_ID}")

    savefig(f"{STATION_ID}_27_ratio_linear_slope_distribution.png")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df["cv_w_linear_b"].dropna() * 100, bins=80, alpha=0.75)
    ax.axvline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("CV linear slope, percentage points per sector")
    ax.set_ylabel("Number of timesteps")
    ax.set_title(f"Distribution of CV-sector linear slope | {STATION_ID}")

    savefig(f"{STATION_ID}_28_cv_linear_slope_distribution.png")

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df["plot_datetime"], df["ratio_linear_b"], linewidth=1.0)
    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Ratio slope")
    ax.set_title(f"Time series of ratio-sector linear slope | {STATION_ID}")

    nice_time_axis(ax)

    savefig(f"{STATION_ID}_29_ratio_linear_slope_timeseries.png")

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df["plot_datetime"], df["cv_w_linear_b"] * 100, linewidth=1.0)
    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("CV slope, percentage points per sector")
    ax.set_title(f"Time series of CV-sector linear slope | {STATION_ID}")

    nice_time_axis(ax)

    savefig(f"{STATION_ID}_30_cv_linear_slope_timeseries.png")


def plot_curvature_distributions_and_timeseries(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df["ratio_quadratic_c"].dropna(), bins=80, alpha=0.75)
    ax.axvline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Ratio quadratic curvature c")
    ax.set_ylabel("Number of timesteps")
    ax.set_title(f"Distribution of ratio-sector curvature | {STATION_ID}")

    savefig(f"{STATION_ID}_31_ratio_quadratic_curvature_distribution.png")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df["cv_w_quadratic_c"].dropna() * 100, bins=80, alpha=0.75)
    ax.axvline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("CV quadratic curvature c, percentage points")
    ax.set_ylabel("Number of timesteps")
    ax.set_title(f"Distribution of CV-sector curvature | {STATION_ID}")

    savefig(f"{STATION_ID}_32_cv_quadratic_curvature_distribution.png")

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df["plot_datetime"], df["ratio_quadratic_c"], linewidth=1.0)
    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Ratio curvature c")
    ax.set_title(f"Time series of ratio-sector quadratic curvature | {STATION_ID}")

    nice_time_axis(ax)

    savefig(f"{STATION_ID}_33_ratio_quadratic_curvature_timeseries.png")

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df["plot_datetime"], df["cv_w_quadratic_c"] * 100, linewidth=1.0)
    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("CV curvature c, percentage points")
    ax.set_title(f"Time series of CV-sector quadratic curvature | {STATION_ID}")

    nice_time_axis(ax)

    savefig(f"{STATION_ID}_34_cv_quadratic_curvature_timeseries.png")


# ============================================================
# 10. OPTIONAL SEASONAL AND DAY/NIGHT BOXPLOTS
# ============================================================

def plot_seasonal_boxplots(df):
    if "season" not in df.columns:
        print("No season column found. Skipping seasonal boxplots.")
        return

    season_order = ["DJF", "MAM", "JJA", "SON"]
    available_seasons = [s for s in season_order if s in df["season"].unique()]

    if len(available_seasons) == 0:
        print("No recognized seasons found. Skipping seasonal boxplots.")
        return

    data = [
        df.loc[df["season"] == s, "R_C10_minus_R_C1"].dropna()
        for s in available_seasons
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.boxplot(data, labels=available_seasons, showfliers=False, patch_artist=True)
    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Season")
    ax.set_ylabel(r"$R_{C10} - R_{C1}$")
    ax.set_title(f"Seasonal distribution of ratio scale sensitivity | {STATION_ID}")

    savefig(f"{STATION_ID}_37_seasonal_ratio_scale_sensitivity_boxplot.png")

    data = [
        df.loc[df["season"] == s, "CV_C10_minus_CV_C1"].dropna() * 100
        for s in available_seasons
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.boxplot(data, labels=available_seasons, showfliers=False, patch_artist=True)
    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Season")
    ax.set_ylabel("CV difference, percentage points")
    ax.set_title(f"Seasonal distribution of CV scale sensitivity | {STATION_ID}")

    savefig(f"{STATION_ID}_38_seasonal_cv_scale_sensitivity_boxplot.png")


def plot_daynight_boxplots(df):
    if "day_night" not in df.columns:
        print("No day_night column found. Skipping day/night boxplots.")
        return

    daynight_order = ["day", "night"]
    available_dn = [d for d in daynight_order if d in df["day_night"].unique()]

    if len(available_dn) == 0:
        print("No recognized day/night values found. Skipping day/night boxplots.")
        return

    data = [
        df.loc[df["day_night"] == d, "R_C10_minus_R_C1"].dropna()
        for d in available_dn
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.boxplot(data, labels=available_dn, showfliers=False, patch_artist=True)
    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Period")
    ax.set_ylabel(r"$R_{C10} - R_{C1}$")
    ax.set_title(f"Day/night distribution of ratio scale sensitivity | {STATION_ID}")

    savefig(f"{STATION_ID}_39_daynight_ratio_scale_sensitivity_boxplot.png")

    data = [
        df.loc[df["day_night"] == d, "CV_C10_minus_CV_C1"].dropna() * 100
        for d in available_dn
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.boxplot(data, labels=available_dn, showfliers=False, patch_artist=True)
    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Period")
    ax.set_ylabel("CV difference, percentage points")
    ax.set_title(f"Day/night distribution of CV scale sensitivity | {STATION_ID}")

    savefig(f"{STATION_ID}_40_daynight_cv_scale_sensitivity_boxplot.png")

def plot_linear_r2_histogram_ratio_vs_cv(df,color_cv,color_ratio,alpha):
    """
    Histogram of timestep-specific R² values for the linear fits.

    One distribution is for:
        ratio_linear_r2

    The other distribution is for:
        cv_w_linear_r2
    """

    ratio_r2 = (
        df["ratio_linear_r2"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    cv_r2 = (
        df["cv_w_linear_r2"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 51)

    ax.hist(
        ratio_r2,
        bins=bins,histtype="step",color=color_ratio,
        alpha=alpha,
        label=f"Ratio, mean R²={ratio_r2.mean():.3f}",
    )

    ax.hist(
        cv_r2,
        bins=bins,color=color_cv,
        alpha=alpha,histtype="step",
        label=f"CV, mean R²={cv_r2.mean():.3f}",
    )

    ax.axvline(
        ratio_r2.mean(),color=color_ratio,
        linestyle="--",
        linewidth=1.8,
        label="Ratio mean",
    )

    ax.axvline(
        cv_r2.mean(),color=color_cv,
        linestyle=":",
        linewidth=2.0,
        label="CV mean",
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$R^2$")
    ax.set_ylabel("Number of timesteps")
    ax.set_title(
        f"Distribution of linear-fit $R^2$ values | {STATION_ID}"
    )

    ax.legend(frameon=True)

    savefig(f"{STATION_ID}_41_linear_r2_histogram_ratio_vs_cv.png")

def plot_quadratic_r2_histogram_ratio_vs_cv(df,color_cv,color_ratio,alpha):
    """
    Histogram of timestep-specific R² values for the quadratic fits.

    One distribution is for:
        ratio_quadratic_r2

    The other distribution is for:
        cv_w_quadratic_r2
    """

    ratio_r2 = (
        df["ratio_quadratic_r2"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    cv_r2 = (
        df["cv_w_quadratic_r2"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 51)

    ax.hist(
        ratio_r2,
        bins=bins,histtype="step",color=color_ratio,
        alpha=alpha,
        label=f"Ratio, mean R²={ratio_r2.mean():.3f}",
    )

    ax.hist(
        cv_r2,
        bins=bins,histtype="step",color=color_cv,
        alpha=alpha,
        label=f"CV, mean R²={cv_r2.mean():.3f}",
    )

    ax.axvline(
        ratio_r2.mean(),color=color_ratio,
        linestyle="--",
        linewidth=1.8,
        label="Ratio mean",
    )

    ax.axvline(
        cv_r2.mean(),color=color_cv,
        linestyle=":",
        linewidth=2.0,
        label="CV mean",
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$R^2$")
    ax.set_ylabel("Number of timesteps")
    ax.set_title(
        f"Distribution of quadratic-fit $R^2$ values | {STATION_ID}"
    )

    ax.legend(frameon=True)

    savefig(f"{STATION_ID}_42_quadratic_r2_histogram_ratio_vs_cv.png")   

def plot_exponential_r2_histogram_ratio_vs_cv(df,color_cv,color_ratio,alpha):
    """
    Histogram of timestep-specific R² values for the exponential fits.

    One distribution is for:
        ratio_exponential_r2

    The other distribution is for:
        cv_w_exponential_r2
    """

    ratio_r2 = (
        df["ratio_exponential_r2"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    cv_r2 = (
        df["cv_w_exponential_r2"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 51)

    ax.hist(
        ratio_r2,color=color_ratio,
        bins=bins,histtype="step",
        alpha=alpha,
        label=f"Ratio, mean R²={ratio_r2.mean():.3f}",
    )

    ax.hist(
        cv_r2,color=color_cv,
        bins=bins,histtype="step",
        alpha=alpha,
        label=f"CV, mean R²={cv_r2.mean():.3f}",
    )

    ax.axvline(
        ratio_r2.mean(),color=color_ratio,
        linestyle="--",
        linewidth=1.8,
        label="Ratio mean",
    )

    ax.axvline(
        cv_r2.mean(),color=color_cv,
        linestyle=":",
        linewidth=2.0,
        label="CV mean",
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$R^2$")
    ax.set_ylabel("Number of timesteps")
    ax.set_title(
        f"Distribution of exponential-fit $R^2$ values | {STATION_ID}"
    )

    ax.legend(frameon=True)

    savefig(f"{STATION_ID}_43_exponential_r2_histogram_ratio_vs_cv.png")

# ============================================================
# 11. SUMMARY CSV
# ============================================================

def save_extended_metric_summary(df):
    out = {}

    out["station"] = STATION_ID
    out["n_timesteps"] = len(df)

    for s in SECTORS:
        r = df[f"ratio_C{s}"]

        out[f"C{s}_mean_abs_R_minus_1_pct"] = (r - 1.0).abs().mean() * 100
        out[f"C{s}_bias_R_minus_1_pct"] = (r - 1.0).mean() * 100
        out[f"C{s}_freq_abs_R_minus_1_gt_5pct"] = ((r - 1.0).abs() > 0.05).mean()
        out[f"C{s}_freq_abs_R_minus_1_gt_10pct"] = ((r - 1.0).abs() > 0.10).mean()
        out[f"C{s}_freq_R_above_1"] = (r > 1.0).mean()
        out[f"C{s}_freq_R_below_1"] = (r < 1.0).mean()

    valid_cv = df["cv_w_C1"] > 1e-8

    cv_growth_factor = pd.Series(np.nan, index=df.index)
    cv_growth_factor.loc[valid_cv] = (
        df.loc[valid_cv, "cv_w_C10"] / df.loc[valid_cv, "cv_w_C1"]
    )

    normalized_cv_increase = pd.Series(np.nan, index=df.index)
    normalized_cv_increase.loc[valid_cv] = (
        (df.loc[valid_cv, "cv_w_C10"] - df.loc[valid_cv, "cv_w_C1"])
        / df.loc[valid_cv, "cv_w_C1"]
    )

    out["mean_CV_growth_factor_C10_C1"] = cv_growth_factor.mean()
    out["median_CV_growth_factor_C10_C1"] = cv_growth_factor.median()
    out["q25_CV_growth_factor_C10_C1"] = cv_growth_factor.quantile(0.25)
    out["q75_CV_growth_factor_C10_C1"] = cv_growth_factor.quantile(0.75)

    out["mean_normalized_CV_increase_pct"] = normalized_cv_increase.mean() * 100
    out["median_normalized_CV_increase_pct"] = normalized_cv_increase.median() * 100
    out["q25_normalized_CV_increase_pct"] = normalized_cv_increase.quantile(0.25) * 100
    out["q75_normalized_CV_increase_pct"] = normalized_cv_increase.quantile(0.75) * 100

    out["mean_ratio_linear_slope"] = df["ratio_linear_b"].mean()
    out["median_ratio_linear_slope"] = df["ratio_linear_b"].median()
    out["freq_ratio_linear_slope_positive"] = (df["ratio_linear_b"] > 0).mean()
    out["freq_ratio_linear_slope_negative"] = (df["ratio_linear_b"] < 0).mean()

    out["mean_cv_linear_slope_pct_points_per_sector"] = df["cv_w_linear_b"].mean() * 100
    out["median_cv_linear_slope_pct_points_per_sector"] = df["cv_w_linear_b"].median() * 100
    out["freq_cv_linear_slope_positive"] = (df["cv_w_linear_b"] > 0).mean()
    out["freq_cv_linear_slope_negative"] = (df["cv_w_linear_b"] < 0).mean()

    out["mean_ratio_quadratic_curvature"] = df["ratio_quadratic_c"].mean()
    out["median_ratio_quadratic_curvature"] = df["ratio_quadratic_c"].median()
    out["freq_ratio_curvature_positive"] = (df["ratio_quadratic_c"] > 0).mean()
    out["freq_ratio_curvature_negative"] = (df["ratio_quadratic_c"] < 0).mean()

    out["mean_cv_quadratic_curvature_pct_points"] = df["cv_w_quadratic_c"].mean() * 100
    out["median_cv_quadratic_curvature_pct_points"] = df["cv_w_quadratic_c"].median() * 100
    out["freq_cv_curvature_positive"] = (df["cv_w_quadratic_c"] > 0).mean()
    out["freq_cv_curvature_negative"] = (df["cv_w_quadratic_c"] < 0).mean()

    out["mean_ratio_linear_r2"] = df["ratio_linear_r2"].mean()
    out["mean_ratio_quadratic_r2"] = df["ratio_quadratic_r2"].mean()
    out["mean_ratio_exponential_r2"] = df["ratio_exponential_r2"].mean()

    out["mean_cv_linear_r2"] = df["cv_w_linear_r2"].mean()
    out["mean_cv_quadratic_r2"] = df["cv_w_quadratic_r2"].mean()
    out["mean_cv_exponential_r2"] = df["cv_w_exponential_r2"].mean()

    summary_df = pd.DataFrame([out])

    out_csv = OUT_DIR / f"{STATION_ID}_extended_metric_summary.csv"
    summary_df.to_csv(out_csv, index=False)

    print(f"Saved extended metric summary: {out_csv}")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    df = load_and_prepare_data()
    '''
    # 1. Main profile plots
    plot_mean_ratio_profile(df)
    plot_mean_cv_profile(df)

    # 2. Scale-sensitivity plots
    plot_ratio_scale_sensitivity_timeseries(df)
    plot_cv_scale_sensitivity_timeseries(df)
    plot_ratio_metric_distributions(df)
    plot_cv_metric_distribution(df)

    # 3. Heatmaps
    plot_ratio_heatmap(df)
    plot_cv_heatmap(df)

    # 4. Fit summaries with mean parameters
    plot_time_average_ratio_fit_parameters(df)
    plot_time_average_cv_fit_parameters(df)

    # 5. Example timestep fits with parameters
    plot_example_ratio_fit_with_parameters(df)
    plot_example_cv_fit_with_parameters(df)
    '''
    # 6. R² distributions
    plot_r2_distributions_for_ratio_and_cv(df)
    # 6b. R² histograms: one plot per model type, ratio vs CV
    plot_linear_r2_histogram_ratio_vs_cv(df,color_ratio=color_ratio,color_cv=color_cv,alpha=alpha)
    plot_quadratic_r2_histogram_ratio_vs_cv(df,color_ratio=color_ratio,color_cv=color_cv,alpha=alpha)
    plot_exponential_r2_histogram_ratio_vs_cv(df,color_ratio=color_ratio,color_cv=color_cv,alpha=alpha)
    '''
    # 7. Extra ratio metrics
    plot_mean_absolute_ratio_deviation(df)
    plot_frequency_large_ratio_deviations(df, thresholds=(0.05, 0.10))
    plot_sector_bias_relative_to_center(df)
    plot_positive_negative_ratio_balance(df)

    # 8. Extra CV metrics
    plot_cv_growth_factor(df)
    plot_normalized_cv_increase(df)

    # 9. Slope and curvature
    plot_slope_distributions_and_timeseries(df)
    plot_curvature_distributions_and_timeseries(df)

    # 10. Optional seasonal/day-night plots
    plot_seasonal_boxplots(df)
    plot_daynight_boxplots(df)
'''
    # 11. Summary CSV
    save_extended_metric_summary(df)

    print("\nAll plots completed.")
    print(f"Output directory: {OUT_DIR}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
# %%
