#%%
import os
from pathlib import Path
import calendar
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
#%%
# ============================================================
# ============================================================
# USER INPUTS
# ============================================================
RUN_MODE = "one"          # "one", "multi", "both"
species = "O3"
mode = "A"
units = "ppb"

station = "2629A"
stations = ["1006A","2629A"]

SECTOR_TYPE = "CUM"
CENTER_COL = "center_ppb"
CV_COL = "cv_w"
SECTOR_COL = "sector"
AGGREGATE_CV_OVER_SECTORS = False
CV_SECTOR="C1"
MERGE_SEASONAL_YEARS = True
SP_LIM=(20,100)
CV_LIM=(0,10)
RATIO_LIM=(0.75,1.25)
PLOTS_DIR = "/home/agkiokas/CAMS/plots"
STATIONS_PATH = "/home/agkiokas/CAMS/CHINESE_STATIONS_INFO_2015_2023.txt"
out_dir_one = f"{PLOTS_DIR}/full_analysis_one_station"
out_dir_multi = f"{PLOTS_DIR}/full_analysis_multi_station"
showfliers=False
UTC_OFFSET_LOCAL = 8
LINE_ALPHA=0.8
#%%
# 1. STATIONS LOADER
# ============================================================
def load_stations_file(stations_path: str) -> pd.DataFrame:
    df = pd.read_csv(stations_path, sep="\t")
    df = df.reset_index().rename(columns={"index": "idx"})

    rename_map = {}
    for col in df.columns:
        c = col.lower()
        if c.startswith("station"):
            rename_map[col] = "Station_Name"
        elif c.startswith("lat"):
            rename_map[col] = "Latitude"
        elif c.startswith("lon"):
            rename_map[col] = "Longitude"
        elif c.startswith("alt"):
            rename_map[col] = "Altitude"

    df = df.rename(columns=rename_map)

    expected = ["idx", "Station_Name", "Latitude", "Longitude", "Altitude"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Stations file missing expected columns: {missing}")

    for col in ["Latitude", "Longitude", "Altitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_valid"] = df[["Latitude", "Longitude", "Altitude"]].notna().all(axis=1)

    return df[expected + ["is_valid"]]
#---------------------------------
#LOCAL TIME UTC HELPER
#-----------------------------
def add_local_time_top_axis(ax, utc_offset=8):
    """
    Add a second x-axis showing local time from UTC hour ticks.
    Assumes the main x-axis is hour of day in UTC (0..23).
    """
    secax = ax.secondary_xaxis("top")

    ticks = np.arange(24)
    labels = [str((h + utc_offset) % 24) for h in ticks]

    secax.set_xticks(ticks)
    secax.set_xticklabels(labels)
    secax.set_xlabel(f"Local Time (UTC+{utc_offset})")

    return secax
# ============================================================
# 2. DATA LOADER
# ============================================================
def load_station_timeseries_csv(csv_path: str, station_name: str = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")

        if ts.isna().all() and {"date", "time"}.issubset(df.columns):
            date_str = df["date"].astype(str).str.zfill(8)
            time_str = df["time"].astype(str).str.zfill(4)
            ts = pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M", errors="coerce")

        df["timestamp"] = ts

    elif {"date", "time"}.issubset(df.columns):
        date_str = df["date"].astype(str).str.zfill(8)
        time_str = df["time"].astype(str).str.zfill(4)
        df["timestamp"] = pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M", errors="coerce")
    else:
        raise ValueError("Need either 'timestamp' or both 'date' and 'time' columns.")

    if df["timestamp"].isna().all():
        raise ValueError(f"All timestamps failed to parse for file: {csv_path}")

    if "station" not in df.columns:
        if station_name is None:
            stem = Path(csv_path).stem
            station_name = stem.split("_")[0]
        df["station"] = station_name

    return df


def load_many_station_csvs(
    input_dir: str,
    species: str,
    mode: str,
    station_names,
    pattern_template: str = "{station}_{species}_{mode}_30min.csv"
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    all_dfs = []

    for st in station_names:
        filename = pattern_template.format(station=st, species=species, mode=mode)
        csv_path = input_dir / filename

        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue

        tmp = load_station_timeseries_csv(str(csv_path), station_name=st)
        all_dfs.append(tmp)

    if not all_dfs:
        raise ValueError("No station CSVs were loaded.")

    return pd.concat(all_dfs, ignore_index=True)


# ============================================================
# 3. TIME FEATURES
# ============================================================
def month_to_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    return np.nan


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).copy()

    out["year"] = out["timestamp"].dt.year
    out["month"] = out["timestamp"].dt.month
    out["month_name"] = out["month"].map(lambda x: calendar.month_abbr[x])
    out["season"] = out["month"].map(month_to_season)
    out["day"] = out["timestamp"].dt.day
    out["hour"] = out["timestamp"].dt.hour
    out["minute"] = out["timestamp"].dt.minute
    return out
def format_season_time_axis(ax, season, merge_years=False, axis_info=None):
    if merge_years:
        if axis_info is not None and axis_info["tick_positions"] is not None:
            ax.set_xticks(axis_info["tick_positions"])
            ax.set_xticklabels(axis_info["tick_labels"], rotation=45, ha="right")
        return

    season_months = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11],
    }

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=season_months[season]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", rotation=45)
# ============================================================
# 4. MERGE STATION METADATA
# ============================================================
def attach_station_metadata(df: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
    meta = stations_df.rename(columns={"Station_Name": "station"})
    return df.merge(
        meta[["station", "Latitude", "Longitude", "Altitude", "is_valid"]],
        on="station",
        how="left"
    )


# ============================================================
# 5. HELPERS
# ============================================================
def ensure_dir(path: str):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def compute_basic_stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "median": np.nan, "max": np.nan}
    return {
        "count": int(s.count()),
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
        "median": s.median(),
        "max": s.max(),
    }


def format_stats_text(stats: dict) -> str:
    return (
        f"n={stats['count']}\n"
        f"mean={stats['mean']:.3f}\n"
        f"std={stats['std']:.3f}\n"
        f"min={stats['min']:.3f}\n"
        f"median={stats['median']:.3f}\n"
        f"max={stats['max']:.3f}"
    )


def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 2:
        return np.nan
    return tmp["x"].corr(tmp["y"], method=method)


def get_sector_order(sectors):
    def sector_key(x):
        s = str(x)
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else 9999
    return sorted(sectors, key=sector_key)


def make_red_pink_gradient(n=10):
    colors = []
    red_vals = np.linspace(0.85, 1.0, n)
    gb_vals = np.linspace(0.85, 0.25, n)
    for r, gb in zip(red_vals, gb_vals):
        colors.append((r, gb, gb))
    return [to_hex(c) for c in colors]

# ============================================================
# 5b. HELPER FOR SEASONAL X-AXIS
# ============================================================
def build_season_plot_axis(sub: pd.DataFrame, merge_years: bool = False, tick_every_month: bool = True):
    """
    For seasonal plots:
    - merge_years=False: use real timestamps
    - merge_years=True: concatenate all rows of the same season on a synthetic axis
      and provide month/year tick positions for the synthetic axis.
    """
    sub = sub.sort_values("timestamp").copy()

    if not merge_years:
        return {
            "x": sub["timestamp"].values,
            "tick_positions": None,
            "tick_labels": None
        }

    sub["__plot_x"] = np.arange(len(sub))

    # tick at first point of each month-year chunk
    sub["year_month"] = sub["timestamp"].dt.to_period("M").astype(str)
    month_starts = sub.groupby("year_month", as_index=False)["__plot_x"].min()

    tick_positions = month_starts["__plot_x"].tolist()
    tick_labels = month_starts["year_month"].tolist()

    # optional thinning if too many ticks
    if len(tick_positions) > 12:
        step = int(np.ceil(len(tick_positions) / 12))
        tick_positions = tick_positions[::step]
        tick_labels = tick_labels[::step]

    return {
        "x": sub["__plot_x"].values,
        "tick_positions": tick_positions,
        "tick_labels": tick_labels
    }
# ============================================================
# 6. DATA PREPARATION
# ============================================================
def prepare_center_timeseries(df: pd.DataFrame, stations=None, center_col="center_ppb", mode=None) -> pd.DataFrame:
    out = df.copy()

    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        out = out[out["station"].isin(stations)]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    out = out.dropna(subset=["timestamp", center_col]).copy()

    # one center value per station/timestamp
    out = (
        out.groupby(["station", "timestamp"], as_index=False)[center_col]
        .mean()
        .sort_values(["station", "timestamp"])
    )
    out = add_time_features(out)
    return out


def prepare_cv_by_sector(df: pd.DataFrame, station: str, cv_col="cv_w", sector_col="sector", sector_type=None, mode=None):
    out = df.copy()
    out = out[out["station"] == station]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    out = out.dropna(subset=["timestamp", cv_col, sector_col]).copy()
    out = add_time_features(out)
    return out.sort_values(["season", "timestamp", sector_col])


def prepare_center_cv_pairs(
    df,
    stations=None,
    center_col="center_ppb",
    cv_col="cv_w",
    sector_col="sector",
    mode=None,
    sector_type=None,
    cv_sector=None,            # NEW
    aggregate_cv_over_sectors=True
):

    out = df.copy()

    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        out = out[out["station"].isin(stations)]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    out = out.dropna(subset=["timestamp", center_col, cv_col]).copy()

    # OPTION 1: specific sector
    if cv_sector is not None:
        out = out[out[sector_col] == cv_sector]

        out = (
            out.groupby(["station", "timestamp"], as_index=False)
            .agg(
                center_ppb=(center_col, "mean"),
                cv_w=(cv_col, "mean")
            )
        )

    # OPTION 2: aggregate sectors
    elif aggregate_cv_over_sectors:

        out = (
            out.groupby(["station", "timestamp"], as_index=False)
            .agg(
                center_ppb=(center_col, "mean"),
                cv_w=(cv_col, "mean")
            )
        )

    # OPTION 3: keep all sectors
    else:
        out = out.rename(columns={center_col: "center_ppb", cv_col: "cv_w"})

    out = add_time_features(out)

    return out

def prepare_cv_for_boxplots(
    df,
    stations=None,
    cv_col="cv_w",
    sector_col="sector",
    mode=None,
    sector_type=None,
    aggregate_cv_over_sectors=True,
    cv_sector=None
):
    out = df.copy()

    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        out = out[out["station"].isin(stations)]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    out = out.dropna(subset=["timestamp", cv_col]).copy()

    # one selected sector
    if cv_sector is not None:
        out = out[out[sector_col] == cv_sector].copy()

    # aggregate over sectors
    elif aggregate_cv_over_sectors:
        out = (
            out.groupby(["station", "timestamp"], as_index=False)
            .agg(cv_w=(cv_col, "mean"))
        )
        cv_col = "cv_w"

    out = add_time_features(out)
    return out, cv_col
# ============================================================
# 7. STATISTICS TABLES
# ============================================================
def monthly_stats_table(df: pd.DataFrame, value_cols, station_col="station") -> pd.DataFrame:
    tmp = add_time_features(df)

    agg = {col: ["count", "mean", "std", "min", "median", "max"] for col in value_cols if col in tmp.columns}
    out = tmp.groupby([station_col, "month", "month_name"]).agg(agg).reset_index()

    flat_cols = []
    for c in out.columns:
        if isinstance(c, tuple):
            if c[1] == "":
                flat_cols.append(c[0])
            else:
                flat_cols.append(f"{c[0]}_{c[1]}")
        else:
            flat_cols.append(c)
    out.columns = flat_cols

    return out.sort_values([station_col, "month"]).reset_index(drop=True)


def seasonal_stats_table(df: pd.DataFrame, value_cols, station_col="station") -> pd.DataFrame:
    tmp = add_time_features(df)

    agg = {col: ["count", "mean", "std", "min", "median", "max"] for col in value_cols if col in tmp.columns}
    out = tmp.groupby([station_col, "season"]).agg(agg).reset_index()

    flat_cols = []
    for c in out.columns:
        if isinstance(c, tuple):
            if c[1] == "":
                flat_cols.append(c[0])
            else:
                flat_cols.append(f"{c[0]}_{c[1]}")
        else:
            flat_cols.append(c)
    out.columns = flat_cols

    season_order = pd.CategoricalDtype(["Winter", "Spring", "Summer", "Autumn"], ordered=True)
    out["season"] = out["season"].astype(season_order)
    out = out.sort_values([station_col, "season"]).reset_index(drop=True)
    return out


def correlation_by_month(df_pairs: pd.DataFrame, x_col="center_ppb", y_col="cv_w", station_col="station", method="pearson"):
    out = (
        df_pairs.groupby([station_col, "month", "month_name"])
        .apply(lambda g: pd.Series({
            "n": len(g[[x_col, y_col]].dropna()),
            "corr": safe_corr(g[x_col], g[y_col], method=method)
        }))
        .reset_index()
        .sort_values([station_col, "month"])
        .reset_index(drop=True)
    )
    return out


def correlation_by_season(df_pairs: pd.DataFrame, x_col="center_ppb", y_col="cv_w", station_col="station", method="pearson"):
    out = (
        df_pairs.groupby([station_col, "season"])
        .apply(lambda g: pd.Series({
            "n": len(g[[x_col, y_col]].dropna()),
            "corr": safe_corr(g[x_col], g[y_col], method=method)
        }))
        .reset_index()
    )
    season_order = pd.CategoricalDtype(["Winter", "Spring", "Summer", "Autumn"], ordered=True)
    out["season"] = out["season"].astype(season_order)
    out = out.sort_values([station_col, "season"]).reset_index(drop=True)
    return out


def altitude_relation_table(
    df: pd.DataFrame,
    stations=None,
    center_col="center_ppb",
    cv_col="cv_w",
    mode=None,
    sector_type=None
) -> pd.DataFrame:
    out = df.copy()

    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        out = out[out["station"].isin(stations)]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    grouped = (
        out.groupby("station", as_index=False)
        .agg(
            Altitude=("Altitude", "first"),
            center_mean=(center_col, "mean"),
            center_std=(center_col, "std"),
            center_cv=(center_col, lambda s: np.nan if pd.to_numeric(s, errors="coerce").dropna().mean() == 0 else pd.to_numeric(s, errors="coerce").dropna().std() / pd.to_numeric(s, errors="coerce").dropna().mean()),
            cv_w_mean=(cv_col, "mean"),
            cv_w_std=(cv_col, "std"),
        )
    )
    return grouped

#HELPER HOURLY STATS/FUNCTIONS
#-----------------------------------------------------------
def prepare_diurnal_cycle(df, stations=None, center_col="center_ppb", mode=None):
    out = df.copy()

    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        out = out[out["station"].isin(stations)]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    out = out.dropna(subset=["timestamp", center_col]).copy()

    # one center value per station/timestamp
    out = (
        out.groupby(["station", "timestamp"], as_index=False)[center_col]
        .mean()
    )

    out = add_time_features(out)
    return out
def plot_one_station_diurnal_cycle(
    df, station, center_col="center_ppb", mode=None,
    species="O3", units="ppb", out_path=None, figsize=(10, 5),
    sp_lim=None
):
    out = prepare_diurnal_cycle(df, stations=station, center_col=center_col, mode=mode)

    grp = (
        out.groupby("hour")[center_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grp["hour"], grp["mean"], marker="o")
    ax.set_xticks(range(24))
    ax.set_xlabel("UTC Hour")
    add_local_time_top_axis(ax, utc_offset=8)
    ax.set_ylabel(f"{species} ({units})")
    ax.set_title(f"Diurnal cycle of {center_col} - {station}")
    ax.grid(True, alpha=0.3)

    if sp_lim is not None:
        ax.set_ylim(sp_lim)

    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax, grp
def plot_one_station_diurnal_cycle_by_season(
    df, station, center_col="center_ppb", mode=None,
    species="O3", units="ppb", out_path=None, figsize=(12, 8),
    sp_lim=None
):
    out = prepare_diurnal_cycle(df, stations=station, center_col=center_col, mode=mode)

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, season in zip(axes, seasons):
        sub = out[out["season"] == season]
        grp = sub.groupby("hour")[center_col].mean().reset_index()

        ax.plot(grp["hour"], grp[center_col], marker="o")
        ax.set_title(season)
        ax.set_xticks(range(24))
        ax.set_xlabel("UTC Hour")
        add_local_time_top_axis(ax, utc_offset=UTC_OFFSET_LOCAL)
        ax.set_ylabel(f"{species} ({units})")
        ax.grid(True, alpha=0.3)

        if sp_lim is not None:
            ax.set_ylim(sp_lim)

    fig.suptitle(f"Diurnal cycle of {center_col} by season - {station}", y=1.02)
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, axes
def plot_multi_station_diurnal_cycle(
    df, stations, center_col="center_ppb", mode=None,
    species="O3", units="ppb", out_path=None, figsize=(10, 5),
    sp_lim=None
):
    out = prepare_diurnal_cycle(df, stations=stations, center_col=center_col, mode=mode)

    fig, ax = plt.subplots(figsize=figsize)

    for st, g in out.groupby("station"):
        grp = g.groupby("hour")[center_col].mean().reset_index()
        ax.plot(grp["hour"], grp[center_col], marker="o", label=st)

    ax.set_xticks(range(24))
    ax.set_xlabel("UTC Hour")
    add_local_time_top_axis(ax, utc_offset=UTC_OFFSET_LOCAL)
    ax.set_ylabel(f"O3 ({units})")
    ax.set_title(f"Diurnal cycle of O3 - multiple stations")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if sp_lim is not None:
        ax.set_ylim(sp_lim)

    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax
def plot_one_station_diurnal_boxplot(
    df, station, center_col="center_ppb", mode=None,
    species="O3", units="ppb", out_path=None, figsize=(12, 5),
    sp_lim=None
):
    out = prepare_diurnal_cycle(df, stations=station, center_col=center_col, mode=mode)

    data = [out.loc[out["hour"] == h, center_col].dropna().values for h in range(24)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, tick_labels=list(range(24)))
    ax.set_xticks(range(24))
    ax.set_xlabel("UTC Hour")
    add_local_time_top_axis(ax, utc_offset=UTC_OFFSET_LOCAL)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(f"{species} ({units})")
    ax.set_title(f"Diurnal distribution of O3 - {station}")
    ax.grid(True, alpha=0.3)

    if sp_lim is not None:
        ax.set_ylim(sp_lim)

    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax
# ============================================================
# 8. PLOTS: SEASONAL TIMESERIES
# ============================================================
def plot_one_station_seasonal_center(
    df, station, stations_df=None, center_col="center_ppb", mode=None,
    species="O3", units="ppb", out_path=None, figsize=(16, 10),sp_lim=None,
    merge_years=False
):
    data = prepare_center_timeseries(df, stations=station, center_col=center_col, mode=mode)

    altitude = np.nan
    if stations_df is not None:
        row = stations_df.loc[stations_df["Station_Name"] == station]
        if not row.empty:
            altitude = row["Altitude"].iloc[0]

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    axes = axes.ravel()

    for ax, season in zip(axes, seasons):
        sub = data[data["season"] == season].sort_values("timestamp").copy()

        if sub.empty:
            ax.set_title(f"{season} (no data)")
            ax.grid(True, alpha=0.3)
            continue

        axis_info = build_season_plot_axis(sub, merge_years=merge_years)
        x = axis_info["x"]
        if merge_years and len(x) > 0:
            ax.set_xlim(x.min(), x.max())
        ax.plot(x, sub[center_col].values, label=station)
        stats = compute_basic_stats(sub[center_col])

        ax.text(
            0.02, 0.98, format_stats_text(stats),
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        title = f"{season} | {station}"
        if pd.notna(altitude):
            title += f" ({altitude:.0f} m)"

        ax.set_title(title)
        ax.set_ylabel(f"{species} ({units})")
        ax.grid(True, alpha=0.3)

        if merge_years:
            ax.set_xlabel("Concatenated seasonal timeline")
            format_season_time_axis(ax, season, merge_years=True, axis_info=axis_info)
        else:
            ax.set_xlabel("Time")
            format_season_time_axis(ax, season, merge_years=False)
            ax.tick_params(axis="x", rotation=45)
    if sp_lim is not None:
        ax.set_ylim(sp_lim)
    suffix = "merged by season" if merge_years else "real timestamps"
    fig.suptitle(f"{species} center pixel seasonal timeseries - {station} ({suffix})", y=1.02)
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.show()
    return fig, axes

def plot_multi_station_seasonal_center(
    df, stations, stations_df=None, center_col="center_ppb", mode=None,
    species="O3", units="ppb", out_path=None, figsize=(16, 10),sp_lim=None,
    merge_years=False
):
    data = prepare_center_timeseries(df, stations=stations, center_col=center_col, mode=mode)

    alt_map = {}
    if stations_df is not None:
        alt_map = dict(zip(stations_df["Station_Name"], stations_df["Altitude"]))

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    axes = axes.ravel()

    for ax, season in zip(axes, seasons):
        sub = data[data["season"] == season].copy()
        shared_axis_info = None
        if merge_years:
            shared_axis_info = build_season_plot_axis(sub.sort_values("timestamp").copy(), merge_years=True)
        if sub.empty:
            ax.set_title(f"{season} (no data)")
            ax.grid(True, alpha=0.3)
            continue

        for st, g in sub.groupby("station"):
            g = g.sort_values("timestamp").copy()
            if merge_years:
                # use season-wide synthetic axis positions based on sorted order within this station
                g["__plot_x"] = np.arange(len(g))
                x = g["__plot_x"].values
            else:
                x = g["timestamp"].values
            label = f"{st}"
            alt = alt_map.get(st, np.nan)
            if pd.notna(alt):
                label += f" ({alt:.0f} m)"

            ax.plot(x, g[center_col].values, label=label)
            stats = compute_basic_stats(g[center_col])

            # stack stats boxes vertically, one per station
            station_index = list(sorted(sub["station"].unique())).index(st)
            y_pos = 0.98 - station_index * 0.18

            ax.text(
                0.02, y_pos,
                f"{st}\nmean={stats['mean']:.2f}\nstd={stats['std']:.2f}",
                transform=ax.transAxes,
                va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                fontsize=8
            )
        ax.set_title(season)
        ax.set_ylabel(f"{species} ({units})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if sp_lim is not None:
            ax.set_ylim(sp_lim)
        if merge_years:
            ax.set_xlabel("Concatenated seasonal timeline")
            if shared_axis_info is not None and len(shared_axis_info["x"]) > 0:
                ax.set_xlim(shared_axis_info["x"].min(), shared_axis_info["x"].max())
            format_season_time_axis(ax, season, merge_years=True, axis_info=shared_axis_info)
        else:
            ax.set_xlabel("Time")
            format_season_time_axis(ax, season, merge_years=False)

    suffix = "merged by season" if merge_years else "real timestamps"
    fig.suptitle(f"{species} center pixel seasonal timeseries - multiple stations ({suffix})", y=1.02)
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.show()
    return fig, axes
# ============================================================
# 9. PLOT: CV_W BY SECTOR PER SEASON
# ============================================================
def plot_one_station_seasonal_cv_by_sector(
    df, station, stations_df=None, cv_col="cv_w", sector_col="sector",
    sector_type="CUM", mode=None, species="O3", out_path=None, figsize=(16, 10),
    merge_years=False, cv_lim=None
):
    data = prepare_cv_by_sector(
        df, station=station, cv_col=cv_col, sector_col=sector_col,
        sector_type=sector_type, mode=mode
    )

    altitude = np.nan
    if stations_df is not None:
        row = stations_df.loc[stations_df["Station_Name"] == station]
        if not row.empty:
            altitude = row["Altitude"].iloc[0]

    all_sectors = get_sector_order(data[sector_col].dropna().unique())
    colors = make_red_pink_gradient(len(all_sectors))
    color_map = dict(zip(all_sectors, colors))

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    axes = axes.ravel()

    for ax, season in zip(axes, seasons):
        sub = data[data["season"] == season].copy()

        if sub.empty:
            ax.set_title(f"{season} (no data)")
            ax.grid(True, alpha=0.3)
            continue

        merged_axis_info = None
        timestamp_to_x = None

        if merge_years:
            # build ONE shared synthetic axis using unique timestamps of this season
            ts_df = (
                sub[["timestamp"]]
                .drop_duplicates()
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            ts_df["__plot_x"] = np.arange(len(ts_df))
            timestamp_to_x = dict(zip(ts_df["timestamp"], ts_df["__plot_x"]))

            # month/year tick positions
            ts_df["year_month"] = ts_df["timestamp"].dt.to_period("M").astype(str)
            month_starts = ts_df.groupby("year_month", as_index=False)["__plot_x"].min()

            tick_positions = month_starts["__plot_x"].tolist()
            tick_labels = month_starts["year_month"].tolist()

            if len(tick_positions) > 12:
                step = int(np.ceil(len(tick_positions) / 12))
                tick_positions = tick_positions[::step]
                tick_labels = tick_labels[::step]

            merged_axis_info = {
                "x": ts_df["__plot_x"].values,
                "tick_positions": tick_positions,
                "tick_labels": tick_labels
            }

        for sec in all_sectors:
            ss = sub[sub[sector_col] == sec].sort_values("timestamp").copy()
            if ss.empty:
                continue

            if merge_years:
                x = ss["timestamp"].map(timestamp_to_x).values
            else:
                x = ss["timestamp"].values

            ax.plot(
                x,
                ss[cv_col].values * 100.0,
                label=str(sec),
                color=color_map[sec]
            )

        title = f"{season} | {station}"
        if pd.notna(altitude):
            title += f" ({altitude:.0f} m)"

        ax.set_title(title)
        ax.set_ylabel(f"{cv_col} (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Sector", fontsize=8, ncol=2)

        if merge_years:
            ax.set_xlabel("Concatenated seasonal timeline")
            if len(merged_axis_info["x"]) > 0:
                ax.set_xlim(merged_axis_info["x"].min(), merged_axis_info["x"].max())
            format_season_time_axis(ax, season, merge_years=True, axis_info=merged_axis_info)
        else:
            ax.set_xlabel("Time")
            format_season_time_axis(ax, season, merge_years=False)

        if cv_lim is not None:
            ax.set_ylim(cv_lim)

    suffix = "merged by season" if merge_years else "real timestamps"
    fig.suptitle(
        f"{species} seasonal Coefficient of variation (%) by sector - {station} - sector_type={sector_type} ({suffix})",
        y=1.02
    )
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.show()
    return fig, axes
# ============================================================
# 10. BOXPLOTS: MONTH / SEASON
# ============================================================
def plot_monthly_boxplot(df, value_col, stations=None, mode=None, sector_type=None, title=None, 
                         ylabel=None,sp_lim=None,cv_lim=None, out_path=None,
                         aggregate_cv_over_sectors=True, cv_sector=None, sector_col="sector", figsize=(12, 6)):
    out = df.copy()

    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        out = out[out["station"].isin(stations)]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    if value_col == "cv_w":
            out, value_col = prepare_cv_for_boxplots(
            out,
            stations=None,
            cv_col=value_col,
            sector_col=sector_col,
            mode=None,
            sector_type=sector_type,
            aggregate_cv_over_sectors=aggregate_cv_over_sectors,
            cv_sector=cv_sector
        )
    else:
        out = add_time_features(out)
        out = out.dropna(subset=[value_col])

    months = list(range(1, 13))
    if value_col == "cv_w":
        data = [out.loc[out["month"] == m, value_col].dropna().values * 100.0 for m in months]
    else:
        data = [out.loc[out["month"] == m, value_col].dropna().values for m in months]   
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, tick_labels=[calendar.month_abbr[m] for m in months],showfliers=False)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel or f"{species} ({units})" )
    ax.set_title(title or f"Monthly boxplot of {species} ({units}) ")
    ax.grid(True, alpha=0.3)
    if sp_lim is not None:
        ax.set_ylim(sp_lim)
    else:
        ax.set_ylim(cv_lim)    
    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_seasonal_boxplot(df, value_col, stations=None, mode=None, sector_type=None, title=None,
                           ylabel=None,sp_lim=None,cv_lim=None,aggregate_cv_over_sectors=True, 
                           cv_sector=None, sector_col="sector", out_path=None, figsize=(10, 6)):
    out = df.copy()

    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        out = out[out["station"].isin(stations)]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    if value_col == "cv_w":
        out, value_col = prepare_cv_for_boxplots(
            out,
            stations=None,
            cv_col=value_col,
            sector_col=sector_col,
            mode=None,
            sector_type=sector_type,
            aggregate_cv_over_sectors=aggregate_cv_over_sectors,
            cv_sector=cv_sector
        )
    else:
        out = add_time_features(out)
        out = out.dropna(subset=[value_col])

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    if value_col == "cv_w":
        data = [out.loc[out["season"] == s, value_col].dropna().values * 100.0 for s in seasons]
    else:
        data = [out.loc[out["season"] == s, value_col].dropna().values for s in seasons]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, tick_labels=seasons,showfliers=False)
    ax.set_xlabel("Season")
    ax.set_ylabel(ylabel or 'CV(%)')
    if sp_lim is not None:
        ax.set_ylim(sp_lim)   
    if cv_lim is not None:  
        ax.set_ylim(cv_lim)  
    ax.set_title(title or f"Seasonal boxplot of CV")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax

def plot_monthly_boxplot_by_station(
    df, value_col, stations, mode=None, sector_type=None,
    title=None, ylabel=None, out_path=None,sp_lim=None,cv_lim=None,
    aggregate_cv_over_sectors=True, cv_sector=None, sector_col="sector", figsize=(14, 6)
):
    out = df.copy()

    if isinstance(stations, str):
        stations = [stations]
    out = out[out["station"].isin(stations)]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    if value_col == "cv_w":
        out, value_col = prepare_cv_for_boxplots(
            out,
            stations=stations,
            cv_col=value_col,
            sector_col=sector_col,
            mode=mode,
            sector_type=sector_type,
            aggregate_cv_over_sectors=aggregate_cv_over_sectors,
            cv_sector=cv_sector
        )
    else:
        out = add_time_features(out)
        out = out.dropna(subset=[value_col])

    months = list(range(1, 13))
    n_st = len(stations)
    width = 0.8 / max(n_st, 1)

    fig, ax = plt.subplots(figsize=figsize)

    for i, st in enumerate(stations):
        data = []
        positions = []
        st_df = out[out["station"] == st]

        for m in months:
            vals = st_df.loc[st_df["month"] == m, value_col].dropna().values
            if value_col == "cv_w":
                vals = vals * 100.0
            data.append(vals)
            positions.append(m + (i - (n_st - 1) / 2) * width)
        color = f"C{i}"
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            manage_ticks=False,showfliers=False
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.4)

            
    ax.set_xticks(months)
    ax.set_xticklabels([calendar.month_abbr[m] for m in months])
    ax.set_xlabel("Month")
    if value_col == "cv_w":
        ax.set_ylim(cv_lim)
        ax.set_ylabel(ylabel or "CV (%)")
        ax.set_title("Monthly boxplot of CV by station")
    else:
        ax.set_ylabel(ylabel or "O3 (ppb)")
        ax.set_ylim(sp_lim)
        ax.set_title("Monthly boxplot of O3 by station")
    ax.grid(True, alpha=0.3)

    # simple legend
    handles = [plt.Line2D([0], [0], color=f"C{i}", lw=6, alpha=0.5) for i in range(n_st)]
    ax.legend(handles, stations, title="Station", loc="best")

    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax

def plot_seasonal_boxplot_by_station(
    df, value_col, stations, mode=None, sector_type=None,
    title=None, ylabel=None, out_path=None,cv_lim=None,sp_lim=None,
     aggregate_cv_over_sectors=True, cv_sector=None, sector_col="sector", figsize=(12, 6)
):
    out = df.copy()

    if isinstance(stations, str):
        stations = [stations]
    out = out[out["station"].isin(stations)]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    if value_col == "cv_w":
        out, value_col = prepare_cv_for_boxplots(
            out,
            stations=stations,
            cv_col=value_col,
            sector_col=sector_col,
            mode=mode,
            sector_type=sector_type,
            aggregate_cv_over_sectors=aggregate_cv_over_sectors,
            cv_sector=cv_sector
        )
    else:
        out = add_time_features(out)
        out = out.dropna(subset=[value_col])

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    n_st = len(stations)
    width = 0.8 / max(n_st, 1)

    fig, ax = plt.subplots(figsize=figsize)

    for i, st in enumerate(stations):
        data = []
        positions = []
        st_df = out[out["station"] == st]

        for j, s in enumerate(seasons, start=1):
            vals = st_df.loc[st_df["season"] == s, value_col].dropna().values
            if value_col == "cv_w":
                vals = vals * 100.0
            data.append(vals)
            positions.append(j + (i - (n_st - 1) / 2) * width)
        color=f"C{i}"
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            manage_ticks=False,showfliers=False
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.4)
        
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            manage_ticks=False,showfliers=False
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.4)
    ax.set_xticks(range(1, len(seasons) + 1))
    ax.set_xticklabels(seasons)
    ax.set_xlabel("Season")
    if value_col == "cv_w":
        ax.set_ylabel(ylabel or "CV (%)")
        ax.set_ylim(cv_lim)
    else:
        ax.set_ylabel(ylabel or 'O3(ppb)')
        ax.set_ylim(sp_lim)
    ax.set_title(title or f"Seasonal boxplot of {value_col} by station")
    ax.grid(True, alpha=0.3)

    handles = [plt.Line2D([0], [0], color=f"C{i}", lw=6, alpha=0.5) for i in range(n_st)]
    ax.legend(handles, stations, title="Station", loc="best")

    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax
# ============================================================
# 11. SCATTER: CENTER_PPB VS CV_W
# ============================================================
def plot_center_vs_cv_scatter(
    df, stations=None, mode=None, sector_type=None,
    aggregate_cv_over_sectors=True,
    out_path=None, figsize=(8, 6),sp_lim=None,cv_lim=None
):
    pairs = prepare_center_cv_pairs(
        df=df,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors
    )

    corr = safe_corr(pairs["center_ppb"], pairs["cv_w"], method="pearson")

    fig, ax = plt.subplots(figsize=figsize)
    if "station" in pairs.columns:
        for st, g in pairs.groupby("station"):
            ax.scatter(g["center_ppb"], g["cv_w"]*100.0, alpha=0.6, label=st)
        ax.legend(fontsize=8)
    else:
        ax.scatter(pairs["center_ppb"], pairs["cv_w"]*100.0, alpha=0.6)

    ax.set_xlabel("O3 (ppb)")
    ax.set_xlim(sp_lim)
    ax.set_ylabel("Coefficient of variation (%)")
    ax.set_ylim(cv_lim)
    ax.set_title(f"Scatter: O3 (central cell) vs CV | Pearson r={corr:.3f}" if pd.notna(corr) else "Scatter: center_ppb vs cv_w")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return pairs, corr, fig, ax


def plot_center_vs_cv_scatter_by_season(
    df, stations=None, mode=None, sector_type=None,
    aggregate_cv_over_sectors=True,
    out_path=None, figsize=(14, 10),sp_lim=None,cv_lim=None
):
    pairs = prepare_center_cv_pairs(
        df=df,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors
    )

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, season in zip(axes, seasons):
        sub = pairs[pairs["season"] == season]
        corr = safe_corr(sub["center_ppb"], sub["cv_w"])

        if not sub.empty:
            if "station" in sub.columns:
                for st, g in sub.groupby("station"):
                    ax.scatter(g["center_ppb"], g["cv_w"]*100.0, alpha=0.6, label=st)
            else:
                ax.scatter(sub["center_ppb"], sub["cv_w"]*100.0, alpha=0.6)

        ax.set_title(f"{season} | r={corr:.3f}" if pd.notna(corr) else f"{season} | r=nan")
        ax.set_xlabel("O3 central cell (ppb)")
        ax.set_xlim(sp_lim)
        ax.set_ylabel("Coefficient of Variation (%)")
        ax.set_ylim(cv_lim)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle("O3(central cell) vs CV by season", y=1.02)
    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return pairs, fig, axes


def plot_center_vs_cv_scatter_by_month(
    df, stations=None, mode=None, sector_type=None,
    aggregate_cv_over_sectors=True,
    out_path=None, figsize=(18, 12),sp_lim=None,cv_lim=None
):
    pairs = prepare_center_cv_pairs(
        df=df,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors
    )

    fig, axes = plt.subplots(3, 4, figsize=figsize, sharex=True, sharey=True)
    axes = axes.ravel()

    for i, m in enumerate(range(1, 13)):
        ax = axes[i]
        sub = pairs[pairs["month"] == m]
        corr = safe_corr(sub["center_ppb"], sub["cv_w"])

        if not sub.empty:
            if "station" in sub.columns:
                for st, g in sub.groupby("station"):
                    ax.scatter(g["center_ppb"], g["cv_w"]*100.0, alpha=0.6, label=st)
            else:
                ax.scatter(sub["center_ppb"], sub["cv_w"]*100.0, alpha=0.6)

        ax.set_title(f"{calendar.month_abbr[m]} | r={corr:.3f}" if pd.notna(corr) else f"{calendar.month_abbr[m]} | r=nan")
        ax.set_xlabel("O3 (ppb)")
        ax.set_xlim(sp_lim)
        ax.set_ylabel("Coefficient of Variation (%)")
        ax.set_ylim(cv_lim)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle("O3 (central cell) vs CV by month", y=1.02)
    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return pairs, fig, axes


# ============================================================
# ============================================================
# 12. ALTITUDE RELATIONS FOR MULTIPLE STATIONS
# ============================================================
def plot_altitude_relations(
    df, stations=None, mode=None, sector_type=None,
    center_col="center_ppb", cv_col="cv_w",
    out_prefix=None,sp_lim=None,cv_lim=None, figsize=(8, 6)
):
    tbl = altitude_relation_table(
        df=df,
        stations=stations,
        center_col=center_col,
        cv_col=cv_col,
        mode=mode,
        sector_type=sector_type
    )

    results = {}

    # center_mean vs altitude
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.scatter(tbl["Altitude"], tbl["center_mean"], alpha=0.8)
    for _, r in tbl.iterrows():
        ax1.text(r["Altitude"], r["center_mean"], str(r["station"]), fontsize=8)
    r1 = safe_corr(tbl["Altitude"], tbl["center_mean"])
    ax1.set_xlabel("Altitude (m)")
    ax1.set_ylabel("Mean center_ppb")
    ax1.set_ylim(sp_lim)
    ax1.set_title(
        f"Mean center_ppb vs Altitude | r={r1:.3f}"
        if pd.notna(r1) else "Mean center_ppb vs Altitude"
    )
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    if out_prefix:
        p = f"{out_prefix}_center_mean_vs_altitude.png"
        ensure_dir(p)
        fig1.savefig(p, dpi=200, bbox_inches="tight")
    plt.show()
    results["center_mean_vs_altitude_corr"] = r1

    # cv_w_mean vs altitude
    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.scatter(tbl["Altitude"], tbl["cv_w_mean"]*100, alpha=0.8)
    for _, r in tbl.iterrows():
        ax2.text(r["Altitude"], r["cv_w_mean"], str(r["station"]), fontsize=8)
    r2 = safe_corr(tbl["Altitude"], tbl["cv_w_mean"])
    ax2.set_xlabel("Altitude (m)")
    ax2.set_ylabel("Mean CV(%)")
    ax2.set_ylim(cv_lim)
    ax2.set_title(
        f"Mean CV vs Altitude | r={r2:.3f}"
        if pd.notna(r2) else "Mean cv_w vs Altitude"
    )
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    if out_prefix:
        p = f"{out_prefix}_cvw_mean_vs_altitude.png"
        ensure_dir(p)
        fig2.savefig(p, dpi=200, bbox_inches="tight")
    plt.show()
    results["cvw_mean_vs_altitude_corr"] = r2

    # variability of center_ppb vs altitude
    fig3, ax3 = plt.subplots(figsize=figsize)
    ax3.scatter(tbl["Altitude"], tbl["center_std"], alpha=0.8)
    for _, r in tbl.iterrows():
        ax3.text(r["Altitude"], r["center_std"], str(r["station"]), fontsize=8)
    r3 = safe_corr(tbl["Altitude"], tbl["center_std"])
    ax3.set_xlabel("Altitude (m)")
    ax3.set_xlim(sp_lim)
    ax3.set_ylabel(f"Std of {species}")
    ax3.set_ylim(cv_lim)
    ax3.set_title(
        f"Variability of center_ppb vs Altitude | r={r3:.3f}"
        if pd.notna(r3) else "Variability of center_ppb vs Altitude"
    )
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    if out_prefix:
        p = f"{out_prefix}_center_std_vs_altitude.png"
        ensure_dir(p)
        fig3.savefig(p, dpi=200, bbox_inches="tight")
    plt.show()
    results["center_std_vs_altitude_corr"] = r3

    return tbl, results


# ============================================================
# 12b. FULL-PERIOD TIMESERIES OF center_ppb,and CV
# ============================================================
def plot_center_timeseries_full_period(
    df,
    stations=None,
    stations_df=None,
    center_col="center_ppb",
    mode=None,
    species="O3",
    units="ppb",
    out_path=None,sp_lim=None,
    figsize=(14, 6)
):
    data = prepare_center_timeseries(
        df=df,
        stations=stations,
        center_col=center_col,
        mode=mode
    )

    alt_map = {}
    if stations_df is not None:
        alt_map = dict(zip(stations_df["Station_Name"], stations_df["Altitude"]))

    fig, ax = plt.subplots(figsize=figsize)

    for st, sub in data.groupby("station"):
        label = st
        alt = alt_map.get(st, np.nan)
        if pd.notna(alt):
            label = f"{st} ({alt:.0f} m)"
        ax.plot(sub["timestamp"], sub[center_col], label=label)

    title = f"{species} full-period timeseries of the central grid cell"
    if stations is not None:
        if isinstance(stations, str):
            title += f" - {stations}"
        else:
            title += " - selected stations"

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{species} ({units})")
    ax.set_ylim(sp_lim)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.show()
    return fig, ax
def plot_one_station_cv_timeseries_full_period(
    df,
    station,
    stations_df=None,
    cv_col="cv_w",
    sector_col="sector",
    sector_type="CUM",
    mode=None,
    species="O3",
    out_path=None,
    figsize=(14, 6),
    cv_lim=None
):
    out = df.copy()
    out = out[out["station"] == station]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    out = out.dropna(subset=["timestamp", cv_col, sector_col]).copy()
    out = out.sort_values("timestamp")

    altitude = np.nan
    if stations_df is not None:
        row = stations_df.loc[stations_df["Station_Name"] == station]
        if not row.empty:
            altitude = row["Altitude"].iloc[0]

    all_sectors = get_sector_order(out[sector_col].dropna().unique())
    colors = make_red_pink_gradient(len(all_sectors))
    color_map = dict(zip(all_sectors, colors))

    fig, ax = plt.subplots(figsize=figsize)

    for sec in all_sectors:
        ss = out[out[sector_col] == sec].sort_values("timestamp").copy()
        if ss.empty:
            continue

        ax.plot(
            ss["timestamp"],
            ss[cv_col].values * 100.0,
            label=str(sec),
            color=color_map[sec],
            alpha=LINE_ALPHA
        )

    title = f"{species} full-period Coefficient of variation (%) by sector | {station}"
    if pd.notna(altitude):
        title += f" ({altitude:.0f} m)"
    if sector_type is not None:
        title += f" | sector_type={sector_type}"

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{cv_col} (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Sector", fontsize=8, ncol=2)
    ax.tick_params(axis="x", rotation=45)

    if cv_lim is not None:
        ax.set_ylim(cv_lim)

    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.show()
    return fig, ax
#12c.RATIO PLOTTING
def plot_one_station_ratio_meanw_to_center(
    df,
    station,
    stations_df=None,
    center_col="center_ppb",
    mean_col="mean_w",
    sector_col="sector",
    sector_type="CUM",
    mode=None,
    species="O3",
    out_path=None,
    figsize=(14, 6),
    merge_years=False,
    ratio_lim=None
):
    out = df.copy()
    out = out[out["station"] == station]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    out = out.dropna(subset=["timestamp", center_col, mean_col, sector_col]).copy()

    # avoid division by zero
    out = out[pd.to_numeric(out[mean_col], errors="coerce") != 0].copy()

    out["ratio_meanw_to_center"] = out[mean_col] / out[center_col]
    out = add_time_features(out)

    altitude = np.nan
    if stations_df is not None:
        row = stations_df.loc[stations_df["Station_Name"] == station]
        if not row.empty:
            altitude = row["Altitude"].iloc[0]

    all_sectors = get_sector_order(out[sector_col].dropna().unique())
    colors = make_red_pink_gradient(len(all_sectors))
    color_map = dict(zip(all_sectors, colors))

    fig, ax = plt.subplots(figsize=figsize)

    for sec in all_sectors:
        ss = out[out[sector_col] == sec].sort_values("timestamp").copy()
        if ss.empty:
            continue

        if merge_years:
            axis_info = build_season_plot_axis(ss, merge_years=True)
            x = axis_info["x"]
        else:
            x = ss["timestamp"].values

        ax.plot(
            x,
            ss["ratio_meanw_to_center"].values,
            label=str(sec),
            color=color_map[sec]
        )

    title = f"{species} Ratio mean to central gridcell by sector | {station}"
    if pd.notna(altitude):
        title += f" ({altitude:.0f} m)"
    if sector_type is not None:
        title += f" | sector_type={sector_type}"

    ax.set_title(title)
    ax.set_xlabel("Concatenated timeline" if merge_years else "Time")
    ax.set_ylabel("Mean/central grid cell")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Sector", fontsize=8, ncol=2)

    if merge_years:
        first_sector = out[out[sector_col] == all_sectors[0]].sort_values("timestamp").copy()
        if not first_sector.empty:
            axis_info = build_season_plot_axis(first_sector, merge_years=True)
            if axis_info["tick_positions"] is not None:
                ax.set_xticks(axis_info["tick_positions"])
                ax.set_xticklabels(axis_info["tick_labels"], rotation=45)
    else:
        ax.tick_params(axis="x", rotation=45)

    if ratio_lim is not None:
        ax.set_ylim(ratio_lim)

    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.show()
    return fig, ax
def plot_one_station_seasonal_boxplot_ratio_meanw_to_center(
    df,
    station,
    center_col="center_ppb",
    mean_col="mean_w",
    sector_col="sector",
    sector_type="CUM",
    mode=None,
    species="O3",
    out_path=None,
    figsize=(12, 6),
    ratio_lim=None
):
    out = df.copy()
    out = out[out["station"] == station]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    out = out.dropna(subset=["timestamp", center_col, mean_col, sector_col]).copy()
    out = out[pd.to_numeric(out[center_col], errors="coerce") != 0].copy()

    out["ratio_meanw_to_center"] = out[mean_col] / out[center_col]
    out = add_time_features(out)

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    data = [out.loc[out["season"] == s, "ratio_meanw_to_center"].dropna().values for s in seasons]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, tick_labels=seasons)
    ax.set_xlabel("Season")
    ax.set_ylim(0.99,1.01)
    ax.set_ylabel("mean_w / center_ppb")
    ax.set_title(f"{species} seasonal boxplot of mean / center value - {station}")
    ax.grid(True, alpha=0.3)

    if ratio_lim is not None:
        ax.set_ylim(ratio_lim)

    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax
# ============================================================
# 13. WRAPPERS: RUN EVERYTHING AUTOMATICALLY
# ============================================================
def run_full_station_analysis(
    df,
    station,
    stations_df=None,
    species="O3",
    units="ppb",
    mode=None,
    sector_type="CUM",
    out_dir=".",
    center_col="center_ppb",
    cv_col="cv_w",
    sector_col="sector",
    aggregate_cv_over_sectors=True,merge_years=False,sp_lim=None,
    cv_lim=None,ratio_lim=None,cv_sector=None
):
    os.makedirs(out_dir, exist_ok=True)

    plot_center_timeseries_full_period(
        df=df,
        stations=station,
        stations_df=stations_df,
        center_col=center_col,
        mode=mode,
        species=species,
        units=units,sp_lim=SP_LIM,
        out_path=os.path.join(out_dir, f"{station}_{species}_full_period_{center_col}.png")
    )
    plot_one_station_ratio_meanw_to_center(
        df=df_one,
        station=station,
        stations_df=stations_df,
        center_col="center_ppb",
        mean_col="mean_w",
        sector_col="sector",
        sector_type="CUM",
        mode=mode,ratio_lim=RATIO_LIM,
        species=species,
        out_path=f"{out_dir_one}/{station}_{species}_ratio_center_to_meanw_by_sector.png",
        merge_years=False
)
    plot_one_station_cv_timeseries_full_period(
    df=df_one,
    station=station,
    stations_df=stations_df,
    cv_col="cv_w",
    sector_col="sector",
    sector_type="CUM",
    mode=mode,
    species=species,
    out_path=f"{out_dir_one}/{station}_{species}_full_period_cvw_by_sector.png",
    cv_lim=CV_LIM
)
    plot_one_station_seasonal_boxplot_ratio_meanw_to_center(
        df=df,
        station=station,
        center_col=center_col,
        mean_col="mean_w",
        sector_col=sector_col,
        sector_type=sector_type,
        mode=mode,
        species=species,
        out_path=os.path.join(out_dir, f"{station}_{species}_seasonal_boxplot_ratio_meanw_to_center.png"),
        ratio_lim=None
    )
    plot_one_station_seasonal_center(
        df=df,
        station=station,
        stations_df=stations_df,
        center_col=center_col,
        mode=mode,
        species=species,sp_lim=SP_LIM,
        units=units,merge_years=merge_years,
        out_path=os.path.join(out_dir, f"{station}_{species}_seasonal_center.png")
    )

    plot_one_station_seasonal_cv_by_sector(
        df=df,
        station=station,
        stations_df=stations_df,
        cv_col=cv_col,
        sector_col=sector_col,
        sector_type=sector_type,
        mode=mode,cv_lim=CV_LIM,
        species=species,merge_years=merge_years,
        out_path=os.path.join(out_dir, f"{station}_{species}_seasonal_{cv_col}_by_sector.png")
    )

    plot_monthly_boxplot(
        df=df,
        value_col=center_col,
        stations=station,
        mode=mode,
        title=f"{station} monthly boxplot of {species} {units}",
        ylabel=f"{species} {units}",sp_lim=SP_LIM,
        out_path=os.path.join(out_dir, f"{station}_{species}_monthly_box_{center_col}.png")
    )

    plot_monthly_boxplot(
        df=df,
        value_col=cv_col,
        stations=station,
        mode=mode,
        sector_type=sector_type,
        title=f"{station} monthly boxplot of coefficient of variation (%) {cv_sector}",
        ylabel='CV(%)',cv_lim=CV_LIM,aggregate_cv_over_sectors=aggregate_cv_over_sectors,
        cv_sector=cv_sector,
        sector_col=sector_col,
        out_path=os.path.join(out_dir, f"{station}_{species}_monthly_box_{cv_col}.png")
    )

    plot_seasonal_boxplot(
        df=df,
        value_col=center_col,
        stations=station,
        mode=mode,
        title=f"{station} seasonal boxplot of {species} ({units})",
        ylabel=f"{species} ({units})",sp_lim=SP_LIM,
        out_path=os.path.join(out_dir, f"{station}_{species}_seasonal_box_{center_col}.png")
    )

    plot_seasonal_boxplot(
        df=df,
        value_col=cv_col,
        stations=station,
        mode=mode,
        sector_type=sector_type,
        title=f"{station} seasonal boxplot of Coefficient of variation {cv_sector} %",
        ylabel='CV (%)',cv_lim=CV_LIM,aggregate_cv_over_sectors=aggregate_cv_over_sectors,
        cv_sector=cv_sector,
        sector_col=sector_col,
        out_path=os.path.join(out_dir, f"{station}_{species}_seasonal_box_{cv_col}.png")
    )
    plot_one_station_diurnal_cycle(
    df=df_one,
    station=station,
    center_col="center_ppb",
    mode=mode,
    species=species,
    units=units,
    out_path=f"{out_dir_one}/{station}_{species}_diurnal_mean.png",
    sp_lim=SP_LIM
)
    plot_one_station_diurnal_cycle_by_season(
    df=df_one,
    station=station,
    center_col="center_ppb",
    mode=mode,
    species=species,
    units=units,
    out_path=f"{out_dir_one}/{station}_{species}_diurnal_by_season.png",
    sp_lim=SP_LIM
)
    plot_one_station_diurnal_boxplot(
    df=df_one,
    station=station,
    center_col="center_ppb",
    mode=mode,
    species=species,
    units=units,
    out_path=f"{out_dir_one}/{station}_{species}_diurnal_boxplot.png",
    sp_lim=SP_LIM
)
    
    pairs, corr_all, _, _ = plot_center_vs_cv_scatter(
        df=df,
        stations=station,
        mode=mode,
        sector_type=sector_type,cv_lim=cv_lim,sp_lim=sp_lim,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors,
        out_path=os.path.join(out_dir, f"{station}_{species}_scatter_{center_col}_vs_{cv_col}.png")
    )

    pairs_season, _, _ = plot_center_vs_cv_scatter_by_season(
        df=df,
        stations=station,
        mode=mode,
        sector_type=sector_type,sp_lim=sp_lim,cv_lim=cv_lim,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors,
        out_path=os.path.join(out_dir, f"{station}_{species}_scatter_{center_col}_vs_{cv_col}_by_season.png")
    )
    
    pairs_month, _, _ = plot_center_vs_cv_scatter_by_month(
        df=df,
        stations=station,
        mode=mode,
        sector_type=sector_type,sp_lim=sp_lim,cv_lim=cv_lim,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors,
        out_path=os.path.join(out_dir, f"{station}_{species}_scatter_{center_col}_vs_{cv_col}_by_month.png")
    )

    monthly_stats = monthly_stats_table(
        df[df["station"] == station],
        value_cols=[center_col, cv_col]
    )
    seasonal_stats = seasonal_stats_table(
        df[df["station"] == station],
        value_cols=[center_col, cv_col]
    )

    monthly_stats.to_csv(
        os.path.join(out_dir, f"{station}_{species}_monthly_stats.csv"),
        index=False
    )
    seasonal_stats.to_csv(
        os.path.join(out_dir, f"{station}_{species}_seasonal_stats.csv"),
        index=False
    )

    corr_month_tbl = correlation_by_month(pairs_month, x_col="center_ppb", y_col="cv_w")
    corr_season_tbl = correlation_by_season(pairs_season, x_col="center_ppb", y_col="cv_w")

    corr_month_tbl.to_csv(
        os.path.join(out_dir, f"{station}_{species}_corr_by_month.csv"),
        index=False
    )
    corr_season_tbl.to_csv(
        os.path.join(out_dir, f"{station}_{species}_corr_by_season.csv"),
        index=False
    )

    print(f"\nSaved full one-station analysis for {station} in: {out_dir}")
    print(f"Overall Pearson correlation ({center_col} vs {cv_col}): {corr_all}")

    return {
        "monthly_stats": monthly_stats,
        "seasonal_stats": seasonal_stats,
        "corr_month": corr_month_tbl,
        "corr_season": corr_season_tbl,
        "overall_corr": corr_all,
        "pairs": pairs,
    }


def run_full_multi_station_analysis(
    df,
    stations,
    stations_df=None,
    species="O3",
    units="ppb",
    mode=None,
    sector_type="CUM",
    out_dir=".",
    center_col="center_ppb",
    cv_col="cv_w",
    aggregate_cv_over_sectors=True,merge_years=False,
    sp_lim=None,cv_lim=None,cv_sector=None,sector_col="sector"
):
    os.makedirs(out_dir, exist_ok=True)

    plot_center_timeseries_full_period(
        df=df,
        stations=stations,
        stations_df=stations_df,
        center_col=center_col,
        mode=mode,
        species=species,
        units=units,sp_lim=SP_LIM,
        out_path=os.path.join(out_dir, f"multi_station_{species}_full_period_{center_col}.png")
    )

    plot_multi_station_seasonal_center(
        df=df,
        stations=stations,
        stations_df=stations_df,
        center_col=center_col,
        mode=mode,
        species=species,
        units=units,merge_years=merge_years,sp_lim=SP_LIM,
        out_path=os.path.join(out_dir, f"multi_station_{species}_seasonal_center.png")
    )

    plot_monthly_boxplot_by_station(
    df=df,
    value_col=center_col,
    stations=stations,
    mode=mode,
    title=f"Multi-station monthly boxplot of 03(ppb)",
    ylabel=center_col,sp_lim=SP_LIM,
    out_path=os.path.join(out_dir, f"multi_station_{species}_monthly_box_{center_col}.png")
)

    plot_monthly_boxplot_by_station(
    df=df,
    value_col=cv_col,
    stations=stations,
    mode=mode,
    sector_type=sector_type,
    title=f"Multi-station monthly boxplot of CV",
    ylabel="CV (%)",cv_lim=CV_LIM,aggregate_cv_over_sectors=aggregate_cv_over_sectors,
    cv_sector=cv_sector,sector_col=sector_col,
    out_path=os.path.join(out_dir, f"multi_station_{species}_monthly_box_{cv_col}.png")
)

    plot_seasonal_boxplot_by_station(
    df=df,
    value_col=center_col,
    stations=stations,
    mode=mode,
    title=f"Multi-station seasonal boxplot of O3 (ppb) by station",
    ylabel='O3 (ppb)',sp_lim=SP_LIM,
    out_path=os.path.join(out_dir, f"multi_station_{species}_seasonal_box_{center_col}.png")
)

    plot_seasonal_boxplot_by_station(
    df=df,
    value_col=cv_col,
    stations=stations,
    mode=mode,
    sector_type=sector_type,
    title=f"Multi-station seasonal boxplot of CV (%) by station",
    ylabel="CV (%)",cv_lim=CV_LIM,sp_lim=SP_LIM,aggregate_cv_over_sectors=aggregate_cv_over_sectors,
        cv_sector=cv_sector,sector_col=sector_col,
    out_path=os.path.join(out_dir, f"multi_station_{species}_seasonal_box_{cv_col}.png")
)

    pairs, corr_all, _, _ = plot_center_vs_cv_scatter(
        df=df,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors,cv_lim=CV_LIM,
        out_path=os.path.join(out_dir, f"multi_station_{species}_scatter_{center_col}_vs_{cv_col}.png")
    )

    pairs_season, _, _ = plot_center_vs_cv_scatter_by_season(
        df=df,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors,cv_lim=CV_LIM,
        out_path=os.path.join(out_dir, f"multi_station_{species}_scatter_{center_col}_vs_{cv_col}_by_season.png")
    )

    pairs_month, _, _ = plot_center_vs_cv_scatter_by_month(
        df=df,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors,cv_lim=CV_LIM,
        out_path=os.path.join(out_dir, f"multi_station_{species}_scatter_{center_col}_vs_{cv_col}_by_month.png")
    )
    plot_multi_station_diurnal_cycle(
    df=df_all,
    stations=stations,
    center_col="center_ppb",
    mode=mode,
    species=species,
    units=units,
    out_path=f"{out_dir_multi}/multi_station_{species}_diurnal_mean.png",
    sp_lim=SP_LIM
     )
    monthly_stats = monthly_stats_table(
        df[df["station"].isin(stations)],
        value_cols=[center_col, cv_col]
    )
    seasonal_stats = seasonal_stats_table(
        df[df["station"].isin(stations)],
        value_cols=[center_col, cv_col]
    )

    monthly_stats.to_csv(
        os.path.join(out_dir, f"multi_station_{species}_monthly_stats.csv"),
        index=False
    )
    seasonal_stats.to_csv(
        os.path.join(out_dir, f"multi_station_{species}_seasonal_stats.csv"),
        index=False
    )

    corr_month_tbl = correlation_by_month(pairs_month, x_col="center_ppb", y_col="cv_w")
    corr_season_tbl = correlation_by_season(pairs_season, x_col="center_ppb", y_col="cv_w")

    corr_month_tbl.to_csv(
        os.path.join(out_dir, f"multi_station_{species}_corr_by_month.csv"),
        index=False
    )
    corr_season_tbl.to_csv(
        os.path.join(out_dir, f"multi_station_{species}_corr_by_season.csv"),
        index=False
    )

    alt_tbl, alt_corrs = plot_altitude_relations(
        df=df,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        center_col=center_col,
        cv_col=cv_col,sp_lim=sp_lim,cv_lim=cv_lim,
        out_prefix=os.path.join(out_dir, f"multi_station_{species}_altitude")
    )

    alt_tbl.to_csv(
        os.path.join(out_dir, f"multi_station_{species}_altitude_relation_table.csv"),
        index=False
    )

    print(f"\nSaved full multi-station analysis in: {out_dir}")
    print(f"Overall Pearson correlation ({center_col} vs {cv_col}): {corr_all}")
    print("Altitude correlations:", alt_corrs)

    return {
        "monthly_stats": monthly_stats,
        "seasonal_stats": seasonal_stats,
        "corr_month": corr_month_tbl,
        "corr_season": corr_season_tbl,
        "overall_corr": corr_all,
        "altitude_table": alt_tbl,
        "altitude_corrs": alt_corrs,
        "pairs": pairs,
    }


# ============================================================
# FINAL EXECUTION BLOCK
# ============================================================
if __name__ == "__main__":
    stations_df = load_stations_file(STATIONS_PATH)

    if RUN_MODE in ["one", "both"]:
        df_one = load_station_timeseries_csv(
            f"{PLOTS_DIR}/{station}_{species}_{mode}_30min.csv",
            station_name=station
        )
        df_one = attach_station_metadata(df_one, stations_df)

        res_one = run_full_station_analysis(
            df=df_one,
            station=station,
            stations_df=stations_df,
            species=species,
            units=units,
            mode=mode,
            sector_type=SECTOR_TYPE,
            out_dir=out_dir_one,
            center_col=CENTER_COL,
            cv_col=CV_COL,
            sector_col=SECTOR_COL,ratio_lim=RATIO_LIM,
            aggregate_cv_over_sectors=AGGREGATE_CV_OVER_SECTORS,
             merge_years=MERGE_SEASONAL_YEARS,sp_lim=SP_LIM,cv_lim=CV_LIM, cv_sector=CV_SECTOR
        )

    if RUN_MODE in ["multi", "both"]:
        df_all = load_many_station_csvs(
            input_dir=PLOTS_DIR,
            species=species,
            mode=mode,
            station_names=stations
        )
        df_all = attach_station_metadata(df_all, stations_df)

        res_multi = run_full_multi_station_analysis(
            df=df_all,
            stations=stations,
            stations_df=stations_df,
            species=species,
            units=units,
            mode=mode,
            sector_type=SECTOR_TYPE,
            out_dir=out_dir_multi,
            center_col=CENTER_COL,
            cv_col=CV_COL,
            aggregate_cv_over_sectors=AGGREGATE_CV_OVER_SECTORS,sector_col=SECTOR_COL,
            merge_years=MERGE_SEASONAL_YEARS,sp_lim=SP_LIM,cv_lim=CV_LIM)
        
# %%
