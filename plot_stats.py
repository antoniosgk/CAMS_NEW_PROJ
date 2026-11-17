#%%
import os
from pathlib import Path
import calendar
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import datetime as dt
import time

# ============================================================
# USER INPUTS
# ============================================================
# To analyze another species, change `species` only — every path,
# filename, axis label and title is derived from it below.
RUN_MODE = "one"             # "one", "multi", "both"
species = "O3"
mode = "A"
units = "ppb"
plt.rcParams['font.weight'] = 'bold'

TIME_PERIOD_MODE = "all"     # "all" or "custom"
START_DATE = "2005-06-09"    # used only if TIME_PERIOD_MODE = "custom"
END_DATE = "2005-06-16"      # used only if TIME_PERIOD_MODE = "custom"

station = "1006A"
stations = ["1001A", "1002A", "1003A", "1004A", "1005A"]
# stations = "all"           # auto-discover every station in the species folder

SECTOR_TYPE = "CUM"
CENTER_COL = "center_ppb"
CV_COL = "cv_w"
SECTOR_COL = "sector"
AGGREGATE_CV_OVER_SECTORS = False
CV_SECTOR = "C10"
MERGE_SEASONAL_YEARS = True

# axis limits (None = autoscale)
SP_LIM = None
CV_LIM = (0, None)
RATIO_LIM = (0.001, 3)
SCATTER_COLOR = "tab:blue"
SHOWFLIERS = False
LINE_ALPHA = 0.3

# ------------------------------------------------------------
# PLOT TOGGLES  (True = produce, False = skip)
# ------------------------------------------------------------
PLOT_CENTER_TS          = True   # full-period center timeseries            [LOCKED]
PLOT_SEASONAL_CENTER    = True   # seasonal 2x2 panel of center             [LOCKED]
PLOT_RATIO              = True   # mean/center ratio by sector              [LOCKED]
PLOT_CV_TS              = True   # full-period cv_w by sector               [LOCKED]
PLOT_SEASONAL_CV_SECTOR = True   # seasonal 2x2 panel of cv_w by sector
PLOT_BOXPLOTS           = False   # monthly + seasonal boxplots (conc & CV)
PLOT_SCATTER            = False   # center-vs-CV scatter (overall/season/month)
PLOT_ALTITUDE           = False   # altitude relations (multi only)
SAVE_STATS_CSV          = False   # monthly/seasonal stats + correlation CSVs
RATIO_SEASONAL_BOX      = False  # optional seasonal boxplot of the ratio

# ------------------------------------------------------------
# PATHS  (species-driven)
# ------------------------------------------------------------
BASE_DIR = "/mnt/store01/agkiokas/CAMS/stations_parquet"
SPECIES_DIR = f"{BASE_DIR}/{species}"            # inputs live here (per species)
PLOTS_DIR = BASE_DIR                             # output root
STATIONS_PATH = "/home/agkiokas/CAMS/CHINESE_STATIONS_INFO_2015_2023.txt"
out_dir_one = f"{PLOTS_DIR}/full_analysis_one_station"
out_dir_multi = f"{PLOTS_DIR}/full_analysis_multi_station"


# ============================================================
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


# ============================================================
# 2. DATA LOADER  (reads .csv OR .parquet/.pq, prefers CSV)
# ============================================================
def read_station_table(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .csv, .parquet, .pq")


def load_station_timeseries_csv(csv_path: str, station_name: str = None) -> pd.DataFrame:
    df = read_station_table(csv_path)

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
            station_name = Path(csv_path).stem.split("_")[0]
        df["station"] = station_name

    return df


def load_many_station_csvs(
    input_dir: str, species: str, mode: str, station_names,
    pattern_template_csv: str = "{station}_{species}_{mode}_30min.csv",
    pattern_template_parquet: str = "{station}_{species}_{mode}_30min.parquet"
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    all_dfs = []

    for st in station_names:
        csv_path = input_dir / pattern_template_csv.format(station=st, species=species, mode=mode)
        parquet_path = input_dir / pattern_template_parquet.format(station=st, species=species, mode=mode)

        if csv_path.exists():
            file_path = csv_path
        elif parquet_path.exists():
            file_path = parquet_path
        else:
            print(f"Skipping missing file for station {st}: neither CSV nor Parquet found")
            continue

        all_dfs.append(load_station_timeseries_csv(str(file_path), station_name=st))

    if not all_dfs:
        raise ValueError("No station files were loaded.")

    return pd.concat(all_dfs, ignore_index=True)


def discover_station_names(
    input_dir: str, species: str, mode: str,
    pattern_csv: str = "*_{species}_{mode}_30min.csv",
    pattern_parquet: str = "*_{species}_{mode}_30min.parquet"
):
    input_dir = Path(input_dir)
    files = list(input_dir.glob(pattern_csv.format(species=species, mode=mode)))
    files += list(input_dir.glob(pattern_parquet.format(species=species, mode=mode)))

    stations = sorted({p.name.split(f"_{species}_{mode}_30min")[0] for p in files})
    if not stations:
        raise ValueError(f"No station files found in {input_dir} for species={species}, mode={mode}")
    return stations


def resolve_stations(stations, input_dir, species, mode):
    if isinstance(stations, str) and stations.lower() == "all":
        return discover_station_names(input_dir, species, mode)
    return stations


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


def filter_time_period(df, time_period_mode="all", start_date=None, end_date=None):
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).copy()

    if time_period_mode == "all":
        return out
    if time_period_mode == "custom":
        if start_date is not None:
            out = out[out["timestamp"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            out = out[out["timestamp"] <= pd.to_datetime(end_date)]
        return out
    raise ValueError("TIME_PERIOD_MODE must be either 'all' or 'custom'")


def format_season_time_axis(ax, season, merge_years=False, axis_info=None):
    if merge_years:
        if axis_info is not None and axis_info["tick_positions"] is not None:
            ax.set_xticks(axis_info["tick_positions"])
            ax.set_xticklabels(axis_info["tick_labels"], rotation=45, ha="right")
        return

    season_months = {
        "Winter": [12, 1, 2], "Spring": [3, 4, 5],
        "Summer": [6, 7, 8], "Autumn": [9, 10, 11],
    }
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=season_months[season]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", rotation=45)


def style_time_axis(ax, title_size=13, label_size=11, tick_size=9):
    ax.margins(x=0)
    if ax.lines:
        ax.set_xlim(ax.lines[0].get_xdata()[0], ax.lines[0].get_xdata()[-1])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.title.set_fontsize(title_size)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(label_size)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontweight("bold")
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.tick_params(axis="x", rotation=35)


# ============================================================
# 4. MERGE STATION METADATA
# ============================================================
def attach_station_metadata(df: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
    meta = stations_df.rename(columns={"Station_Name": "station"})
    return df.merge(
        meta[["station", "Latitude", "Longitude", "Altitude", "is_valid"]],
        on="station", how="left"
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
        "count": int(s.count()), "mean": s.mean(), "std": s.std(),
        "min": s.min(), "median": s.median(), "max": s.max(),
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
        digits = "".join(ch for ch in str(x) if ch.isdigit())
        return int(digits) if digits else 9999
    return sorted(sectors, key=sector_key)


def make_red_pink_gradient(n=10):
    colors = [plt.cm.Reds(0.25 + 0.70 * i / (n - 1)) for i in range(n)]
    return [to_hex(c) for c in colors]


def build_season_plot_axis(sub: pd.DataFrame, merge_years: bool = False):
    """Seasonal x-axis. merge_years=False -> real timestamps;
       merge_years=True -> concatenate same-season rows on a synthetic axis."""
    sub = sub.sort_values("timestamp").copy()
    if not merge_years:
        return {"x": sub["timestamp"].values, "tick_positions": None, "tick_labels": None}

    sub["__plot_x"] = np.arange(len(sub))
    sub["year_month"] = sub["timestamp"].dt.to_period("M").astype(str)
    month_starts = sub.groupby("year_month", as_index=False)["__plot_x"].min()
    tick_positions = month_starts["__plot_x"].tolist()
    tick_labels = month_starts["year_month"].tolist()

    if len(tick_positions) > 12:
        step = int(np.ceil(len(tick_positions) / 12))
        tick_positions = tick_positions[::step]
        tick_labels = tick_labels[::step]

    return {"x": sub["__plot_x"].values, "tick_positions": tick_positions, "tick_labels": tick_labels}


# ============================================================
# 6. DATA PREPARATION
# ============================================================
def prepare_center_timeseries(df, stations=None, center_col="center_ppb", mode=None):
    out = df.copy()
    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        out = out[out["station"].isin(stations)]
    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    out = out.dropna(subset=["timestamp", center_col]).copy()
    # one center value per station/timestamp (collapse sector rows)
    out = (out.groupby(["station", "timestamp"], as_index=False)[center_col]
           .mean().sort_values(["station", "timestamp"]))
    return add_time_features(out)


def prepare_cv_by_sector(df, station, cv_col="cv_w", sector_col="sector", sector_type=None, mode=None):
    out = df.copy()
    out = out[out["station"] == station]
    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]
    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    out = out.dropna(subset=["timestamp", cv_col, sector_col]).copy()
    out = add_time_features(out)
    return out.sort_values(["season", "timestamp", sector_col])


def prepare_center_cv_pairs(df, stations=None, center_col="center_ppb", cv_col="cv_w",
                            sector_col="sector", mode=None, sector_type=None,
                            cv_sector=None, aggregate_cv_over_sectors=True):
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

    if cv_sector is not None:                       # one chosen sector
        out = out[out[sector_col] == cv_sector]
        out = (out.groupby(["station", "timestamp"], as_index=False)
               .agg(center_ppb=(center_col, "mean"), cv_w=(cv_col, "mean")))
    elif aggregate_cv_over_sectors:                 # mean over sectors
        out = (out.groupby(["station", "timestamp"], as_index=False)
               .agg(center_ppb=(center_col, "mean"), cv_w=(cv_col, "mean")))
    else:                                           # keep all sectors
        out = out.rename(columns={center_col: "center_ppb", cv_col: "cv_w"})

    return add_time_features(out)


def prepare_cv_for_boxplots(df, stations=None, cv_col="cv_w", sector_col="sector",
                            mode=None, sector_type=None, aggregate_cv_over_sectors=True, cv_sector=None):
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

    if cv_sector is not None:
        out = out[out[sector_col] == cv_sector].copy()
    elif aggregate_cv_over_sectors:
        out = (out.groupby(["station", "timestamp"], as_index=False).agg(cv_w=(cv_col, "mean")))
        cv_col = "cv_w"

    out = add_time_features(out)
    return out, cv_col


def build_stats_frame(df, stations=None, center_col="center_ppb", cv_col="cv_w",
                      sector_col="sector", mode=None, sector_type=None,
                      cv_sector=None, aggregate_cv_over_sectors=True):
    """One row per (station, timestamp): de-duplicated center + a single cv value,
       so stats counts/std are not inflated by the per-sector rows."""
    center = prepare_center_timeseries(df, stations=stations, center_col=center_col, mode=mode)
    center = center[["station", "timestamp", center_col]]

    agg_flag = aggregate_cv_over_sectors
    if cv_sector is None and not agg_flag:
        agg_flag = True  # force one cv row per (station, timestamp)

    cv, cvname = prepare_cv_for_boxplots(
        df, stations=stations, cv_col=cv_col, sector_col=sector_col, mode=mode,
        sector_type=sector_type, aggregate_cv_over_sectors=agg_flag, cv_sector=cv_sector
    )
    cv = cv[["station", "timestamp", cvname]].rename(columns={cvname: cv_col})

    return center.merge(cv, on=["station", "timestamp"], how="outer")


# ============================================================
# 7. STATISTICS TABLES
# ============================================================
def _flatten(out):
    flat = []
    for c in out.columns:
        if isinstance(c, tuple):
            flat.append(c[0] if c[1] == "" else f"{c[0]}_{c[1]}")
        else:
            flat.append(c)
    out.columns = flat
    return out


def monthly_stats_table(df, value_cols, station_col="station"):
    tmp = add_time_features(df)
    agg = {col: ["count", "mean", "std", "min", "median", "max"] for col in value_cols if col in tmp.columns}
    out = _flatten(tmp.groupby([station_col, "month", "month_name"]).agg(agg).reset_index())
    return out.sort_values([station_col, "month"]).reset_index(drop=True)


def seasonal_stats_table(df, value_cols, station_col="station"):
    tmp = add_time_features(df)
    agg = {col: ["count", "mean", "std", "min", "median", "max"] for col in value_cols if col in tmp.columns}
    out = _flatten(tmp.groupby([station_col, "season"]).agg(agg).reset_index())
    season_order = pd.CategoricalDtype(["Winter", "Spring", "Summer", "Autumn"], ordered=True)
    out["season"] = out["season"].astype(season_order)
    return out.sort_values([station_col, "season"]).reset_index(drop=True)


def correlation_by_month(df_pairs, x_col="center_ppb", y_col="cv_w", station_col="station", method="pearson"):
    rows = []
    for (st, m, mn), g in df_pairs.groupby([station_col, "month", "month_name"]):
        sub = g[[x_col, y_col]].dropna()
        rows.append({station_col: st, "month": m, "month_name": mn,
                     "n": len(sub), "corr": safe_corr(g[x_col], g[y_col], method=method)})
    return pd.DataFrame(rows).sort_values([station_col, "month"]).reset_index(drop=True)


def correlation_by_season(df_pairs, x_col="center_ppb", y_col="cv_w", station_col="station", method="pearson"):
    rows = []
    for (st, s), g in df_pairs.groupby([station_col, "season"]):
        sub = g[[x_col, y_col]].dropna()
        rows.append({station_col: st, "season": s,
                     "n": len(sub), "corr": safe_corr(g[x_col], g[y_col], method=method)})
    out = pd.DataFrame(rows)
    season_order = pd.CategoricalDtype(["Winter", "Spring", "Summer", "Autumn"], ordered=True)
    out["season"] = out["season"].astype(season_order)
    return out.sort_values([station_col, "season"]).reset_index(drop=True)


def altitude_relation_table(df, stations=None, center_col="center_ppb", cv_col="cv_w",
                            mode=None, sector_type=None):
    out = df.copy()
    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        out = out[out["station"].isin(stations)]
    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]
    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    return (out.groupby("station", as_index=False).agg(
        Altitude=("Altitude", "first"),
        center_mean=(center_col, "mean"),
        center_std=(center_col, "std"),
        center_cv=(center_col, lambda s: np.nan if pd.to_numeric(s, errors="coerce").dropna().mean() == 0
                   else pd.to_numeric(s, errors="coerce").dropna().std() / pd.to_numeric(s, errors="coerce").dropna().mean()),
        cv_w_mean=(cv_col, "mean"),
        cv_w_std=(cv_col, "std"),
    ))


# ============================================================
# 8. FULL-PERIOD TIMESERIES: center & cv_w
# ============================================================
def plot_center_timeseries_full_period(df, stations=None, stations_df=None, center_col="center_ppb",
                                        mode=None, species="O3", units="ppb", out_path=None,
                                        sp_lim=None, figsize=(14, 6)):
    data = prepare_center_timeseries(df, stations=stations, center_col=center_col, mode=mode)

    alt_map = dict(zip(stations_df["Station_Name"], stations_df["Altitude"])) if stations_df is not None else {}

    fig, ax = plt.subplots(figsize=figsize, dpi=400)
    for st, sub in data.groupby("station"):
        label = st
        alt = alt_map.get(st, np.nan)
        if pd.notna(alt):
            label = f"{st} ({alt:.0f} m)"
        ax.plot(sub["timestamp"], sub[center_col], label=label)

    title = f"{species} full-period timeseries of the central grid cell"
    if stations is not None:
        title += f" - {stations}" if isinstance(stations, str) else " - selected stations"

    ax.set_title(title)
    ax.set_xlabel("Time", fontsize=11, fontweight='bold')
    ax.set_ylabel(f"{species} ({units})", fontsize=11, fontweight='bold')
    ax.set_ylim(sp_lim)
    style_time_axis(ax)
    ax.grid(True, alpha=0.3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_weight('bold')
    ax.legend()
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_one_station_cv_timeseries_full_period(df, station, stations_df=None, cv_col="cv_w",
                                               sector_col="sector", sector_type="CUM", mode=None,
                                               species="O3", out_path=None, figsize=(14, 6), cv_lim=None):
    out = df.copy()
    out = out[out["station"] == station]
    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]
    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    out = out.dropna(subset=["timestamp", cv_col, sector_col]).copy().sort_values("timestamp")

    altitude = np.nan
    if stations_df is not None:
        row = stations_df.loc[stations_df["Station_Name"] == station]
        if not row.empty:
            altitude = row["Altitude"].iloc[0]

    all_sectors = get_sector_order(out[sector_col].dropna().unique())
    color_map = dict(zip(all_sectors, make_red_pink_gradient(len(all_sectors))))

    fig, ax = plt.subplots(figsize=figsize)
    for sec in all_sectors:
        ss = out[out[sector_col] == sec].sort_values("timestamp")
        if ss.empty:
            continue
        ax.plot(ss["timestamp"], ss[cv_col].values * 100.0, label=str(sec),
                color=color_map[sec], alpha=LINE_ALPHA)

    title = f"{species} full-period Coefficient of variation (%) by sector | {station}"
    if pd.notna(altitude):
        title += f" ({altitude:.0f} m)"

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Time", fontsize=11, fontweight='bold')
    ax.set_ylabel("CV (%)", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(title="Sector", fontsize=8, ncol=2)
    style_time_axis(ax)
    if cv_lim is not None:
        ax.set_ylim(cv_lim)
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax


# ============================================================
# 9. RATIO: mean_w / center_ppb
# ============================================================
def plot_one_station_ratio_meanw_to_center(df, station, stations_df=None, center_col="center_ppb",
                                           mean_col="mean_w", sector_col="sector", sector_type="CUM",
                                           mode=None, species="O3", out_path=None, figsize=(14, 6),
                                           merge_years=False, ratio_lim=None):
    out = df.copy()
    out = out[out["station"] == station]
    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]
    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    out = out.dropna(subset=["timestamp", center_col, mean_col, sector_col]).copy()
    out = out[pd.to_numeric(out[mean_col], errors="coerce") != 0].copy()
    out["ratio_meanw_to_center"] = out[mean_col] / out[center_col]
    out = add_time_features(out)

    altitude = np.nan
    if stations_df is not None:
        row = stations_df.loc[stations_df["Station_Name"] == station]
        if not row.empty:
            altitude = row["Altitude"].iloc[0]

    all_sectors = get_sector_order(out[sector_col].dropna().unique())
    color_map = dict(zip(all_sectors, make_red_pink_gradient(len(all_sectors))))

    fig, ax = plt.subplots(figsize=figsize)
    for sec in all_sectors:
        ss = out[out[sector_col] == sec].sort_values("timestamp")
        if ss.empty:
            continue
        x = build_season_plot_axis(ss, merge_years=True)["x"] if merge_years else ss["timestamp"].values
        ax.plot(x, ss["ratio_meanw_to_center"].values, label=str(sec), color=color_map[sec])

    title = f"{species} ratio mean/central gridcell by sector | {station}"
    if pd.notna(altitude):
        title += f" ({altitude:.0f} m)"

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Concatenated timeline" if merge_years else "Time", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mean / central grid cell", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(title="Sector", fontsize=8, ncol=2)

    if merge_years:
        first = out[out[sector_col] == all_sectors[0]].sort_values("timestamp")
        if not first.empty:
            info = build_season_plot_axis(first, merge_years=True)
            if info["tick_positions"] is not None:
                ax.set_xticks(info["tick_positions"])
                ax.set_xticklabels(info["tick_labels"], rotation=45)
    else:
        style_time_axis(ax)

    if ratio_lim is not None:
        ax.set_ylim(ratio_lim)
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_one_station_seasonal_boxplot_ratio_meanw_to_center(df, station, center_col="center_ppb",
                                                            mean_col="mean_w", sector_col="sector",
                                                            sector_type="CUM", mode=None, species="O3",
                                                            out_path=None, figsize=(12, 6), ratio_lim=None):
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
    ax.boxplot(data, tick_labels=seasons, showfliers=SHOWFLIERS)
    ax.set_xlabel("Season")
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
# 10. SEASONAL PANELS
# ============================================================
def plot_one_station_seasonal_center(df, station, stations_df=None, center_col="center_ppb", mode=None,
                                     species="O3", units="ppb", out_path=None, figsize=(16, 10),
                                     sp_lim=None, merge_years=False):
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
        ax.text(0.02, 0.98, format_stats_text(stats), transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        title = f"{season} | {station}"
        if pd.notna(altitude):
            title += f" ({altitude:.0f} m)"
        ax.set_title(title)
        ax.set_ylabel(f"{species} ({units})")
        ax.grid(True, alpha=0.3)
        if sp_lim is not None:
            ax.set_ylim(sp_lim)

        if merge_years:
            ax.set_xlabel("Concatenated seasonal timeline")
            format_season_time_axis(ax, season, merge_years=True, axis_info=axis_info)
        else:
            ax.set_xlabel("Time")
            format_season_time_axis(ax, season, merge_years=False)
            ax.tick_params(axis="x", rotation=45)

    suffix = "merged by season" if merge_years else "real timestamps"
    fig.suptitle(f"{species} center pixel seasonal timeseries - {station} ({suffix})", y=1.02)
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, axes


def plot_multi_station_seasonal_center(df, stations, stations_df=None, center_col="center_ppb", mode=None,
                                       species="O3", units="ppb", out_path=None, figsize=(16, 10),
                                       sp_lim=None, merge_years=False):
    data = prepare_center_timeseries(df, stations=stations, center_col=center_col, mode=mode)
    alt_map = dict(zip(stations_df["Station_Name"], stations_df["Altitude"])) if stations_df is not None else {}

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    axes = axes.ravel()

    for ax, season in zip(axes, seasons):
        sub = data[data["season"] == season].copy()
        if sub.empty:
            ax.set_title(f"{season} (no data)")
            ax.grid(True, alpha=0.3)
            continue

        shared_axis_info = None
        timestamp_to_x = None
        if merge_years:
            ts_df = sub[["timestamp"]].drop_duplicates().sort_values("timestamp").reset_index(drop=True)
            ts_df["__plot_x"] = np.arange(len(ts_df))
            timestamp_to_x = dict(zip(ts_df["timestamp"], ts_df["__plot_x"]))
            ts_df["year_month"] = ts_df["timestamp"].dt.to_period("M").astype(str)
            month_starts = ts_df.groupby("year_month", as_index=False)["__plot_x"].min()
            tick_positions = month_starts["__plot_x"].tolist()
            tick_labels = month_starts["year_month"].tolist()
            if len(tick_positions) > 12:
                step = int(np.ceil(len(tick_positions) / 12))
                tick_positions = tick_positions[::step]
                tick_labels = tick_labels[::step]
            shared_axis_info = {"x": ts_df["__plot_x"].values,
                                "tick_positions": tick_positions, "tick_labels": tick_labels}

        station_order = sorted(sub["station"].unique())
        for st in station_order:
            g = sub[sub["station"] == st].sort_values("timestamp")
            x = g["timestamp"].map(timestamp_to_x).values if merge_years else g["timestamp"].values
            label = st
            alt = alt_map.get(st, np.nan)
            if pd.notna(alt):
                label += f" ({alt:.0f} m)"
            ax.plot(x, g[center_col].values, label=label)

            stats = compute_basic_stats(g[center_col])
            y_pos = 0.98 - station_order.index(st) * 0.18
            ax.text(0.02, y_pos, f"{st}\nmean={stats['mean']:.2f}\nstd={stats['std']:.2f}",
                    transform=ax.transAxes, va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7), fontsize=8)

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


def plot_one_station_seasonal_cv_by_sector(df, station, stations_df=None, cv_col="cv_w", sector_col="sector",
                                           sector_type="CUM", mode=None, species="O3", out_path=None,
                                           figsize=(16, 10), merge_years=False, cv_lim=None):
    data = prepare_cv_by_sector(df, station=station, cv_col=cv_col, sector_col=sector_col,
                                sector_type=sector_type, mode=mode)

    altitude = np.nan
    if stations_df is not None:
        row = stations_df.loc[stations_df["Station_Name"] == station]
        if not row.empty:
            altitude = row["Altitude"].iloc[0]

    all_sectors = get_sector_order(data[sector_col].dropna().unique())
    color_map = dict(zip(all_sectors, make_red_pink_gradient(len(all_sectors))))

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
            ts_df = sub[["timestamp"]].drop_duplicates().sort_values("timestamp").reset_index(drop=True)
            ts_df["__plot_x"] = np.arange(len(ts_df))
            timestamp_to_x = dict(zip(ts_df["timestamp"], ts_df["__plot_x"]))
            ts_df["year_month"] = ts_df["timestamp"].dt.to_period("M").astype(str)
            month_starts = ts_df.groupby("year_month", as_index=False)["__plot_x"].min()
            tick_positions = month_starts["__plot_x"].tolist()
            tick_labels = month_starts["year_month"].tolist()
            if len(tick_positions) > 12:
                step = int(np.ceil(len(tick_positions) / 12))
                tick_positions = tick_positions[::step]
                tick_labels = tick_labels[::step]
            merged_axis_info = {"x": ts_df["__plot_x"].values,
                                "tick_positions": tick_positions, "tick_labels": tick_labels}

        for sec in all_sectors:
            ss = sub[sub[sector_col] == sec].sort_values("timestamp")
            if ss.empty:
                continue
            x = ss["timestamp"].map(timestamp_to_x).values if merge_years else ss["timestamp"].values
            ax.plot(x, ss[cv_col].values * 100.0, label=str(sec), color=color_map[sec])

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
    fig.suptitle(f"{species} seasonal CV (%) by sector - {station} - sector_type={sector_type} ({suffix})", y=1.02)
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, axes


# ============================================================
# 11. BOXPLOTS
# ============================================================
def plot_monthly_boxplot(df, value_col, stations=None, mode=None, sector_type=None, title=None,
                         ylabel=None, sp_lim=None, cv_lim=None, out_path=None,
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

    is_cv = (value_col == "cv_w")
    if is_cv:
        out, value_col = prepare_cv_for_boxplots(out, stations=None, cv_col=value_col, sector_col=sector_col,
                                                 mode=None, sector_type=sector_type,
                                                 aggregate_cv_over_sectors=aggregate_cv_over_sectors,
                                                 cv_sector=cv_sector)
    else:
        out = add_time_features(out).dropna(subset=[value_col])

    months = list(range(1, 13))
    if is_cv:
        data = [out.loc[out["month"] == m, value_col].dropna().values * 100.0 for m in months]
    else:
        data = [out.loc[out["month"] == m, value_col].dropna().values for m in months]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, tick_labels=[calendar.month_abbr[m] for m in months], showfliers=SHOWFLIERS)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel or (f"{species} ({units})" if not is_cv else "CV (%)"))
    ax.set_title(title or f"Monthly boxplot of {species} ({units})")
    ax.grid(True, alpha=0.3)
    if is_cv:
        if cv_lim is not None:
            ax.set_ylim(cv_lim)
    elif sp_lim is not None:
        ax.set_ylim(sp_lim)
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_seasonal_boxplot(df, value_col, stations=None, mode=None, sector_type=None, title=None,
                          ylabel=None, sp_lim=None, cv_lim=None, aggregate_cv_over_sectors=True,
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

    is_cv = (value_col == "cv_w")
    if is_cv:
        out, value_col = prepare_cv_for_boxplots(out, stations=None, cv_col=value_col, sector_col=sector_col,
                                                 mode=None, sector_type=sector_type,
                                                 aggregate_cv_over_sectors=aggregate_cv_over_sectors,
                                                 cv_sector=cv_sector)
    else:
        out = add_time_features(out).dropna(subset=[value_col])

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    if is_cv:
        data = [out.loc[out["season"] == s, value_col].dropna().values * 100.0 for s in seasons]
    else:
        data = [out.loc[out["season"] == s, value_col].dropna().values for s in seasons]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, tick_labels=seasons, showfliers=SHOWFLIERS)
    ax.set_xlabel("Season")
    ax.set_ylabel(ylabel or ("CV (%)" if is_cv else f"{species} ({units})"))
    if is_cv:
        if cv_lim is not None:
            ax.set_ylim(cv_lim)
    elif sp_lim is not None:
        ax.set_ylim(sp_lim)
    ax.set_title(title or f"Seasonal boxplot of {value_col}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_monthly_boxplot_by_station(df, value_col, stations, mode=None, sector_type=None, title=None,
                                    ylabel=None, out_path=None, sp_lim=None, cv_lim=None,
                                    aggregate_cv_over_sectors=True, cv_sector=None, sector_col="sector",
                                    species="O3", units="ppb", figsize=(14, 6)):
    out = df.copy()
    if isinstance(stations, str):
        stations = [stations]
    out = out[out["station"].isin(stations)]
    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]
    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    is_cv = (value_col == "cv_w")
    if is_cv:
        out, value_col = prepare_cv_for_boxplots(out, stations=stations, cv_col=value_col, sector_col=sector_col,
                                                 mode=mode, sector_type=sector_type,
                                                 aggregate_cv_over_sectors=aggregate_cv_over_sectors,
                                                 cv_sector=cv_sector)
    else:
        out = add_time_features(out).dropna(subset=[value_col])

    months = list(range(1, 13))
    n_st = len(stations)
    width = 0.8 / max(n_st, 1)

    fig, ax = plt.subplots(figsize=figsize)
    for i, st in enumerate(stations):
        st_df = out[out["station"] == st]
        data, positions = [], []
        for m in months:
            vals = st_df.loc[st_df["month"] == m, value_col].dropna().values
            if is_cv:
                vals = vals * 100.0
            data.append(vals)
            positions.append(m + (i - (n_st - 1) / 2) * width)
        color = f"C{i}"
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                        patch_artist=True, manage_ticks=False, showfliers=SHOWFLIERS)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.4)

    ax.set_xticks(months)
    ax.set_xticklabels([calendar.month_abbr[m] for m in months])
    ax.set_xlabel("Month")
    if is_cv:
        ax.set_ylim(cv_lim)
        ax.set_ylabel(ylabel or "CV (%)")
        ax.set_title(title or "Monthly boxplot of CV (%) by station")
    else:
        ax.set_ylabel(ylabel or f"{species} ({units})")
        ax.set_ylim(sp_lim)
        ax.set_title(title or f"Monthly boxplot of {species} ({units}) by station")
    ax.grid(True, alpha=0.3)

    handles = [plt.Line2D([0], [0], color=f"C{i}", lw=6, alpha=0.5) for i in range(n_st)]
    ax.legend(handles, stations, title="Station", loc="best")
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_seasonal_boxplot_by_station(df, value_col, stations, mode=None, sector_type=None, title=None,
                                     ylabel=None, out_path=None, cv_lim=None, sp_lim=None,
                                     aggregate_cv_over_sectors=True, cv_sector=None, sector_col="sector",
                                     species="O3", units="ppb", figsize=(12, 6)):
    out = df.copy()
    if isinstance(stations, str):
        stations = [stations]
    out = out[out["station"].isin(stations)]
    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]
    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    is_cv = (value_col == "cv_w")
    if is_cv:
        out, value_col = prepare_cv_for_boxplots(out, stations=stations, cv_col=value_col, sector_col=sector_col,
                                                 mode=mode, sector_type=sector_type,
                                                 aggregate_cv_over_sectors=aggregate_cv_over_sectors,
                                                 cv_sector=cv_sector)
    else:
        out = add_time_features(out).dropna(subset=[value_col])

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    n_st = len(stations)
    width = 0.8 / max(n_st, 1)

    fig, ax = plt.subplots(figsize=figsize)
    for i, st in enumerate(stations):
        st_df = out[out["station"] == st]
        data, positions = [], []
        for j, s in enumerate(seasons, start=1):
            vals = st_df.loc[st_df["season"] == s, value_col].dropna().values
            if is_cv:
                vals = vals * 100.0
            data.append(vals)
            positions.append(j + (i - (n_st - 1) / 2) * width)
        color = f"C{i}"
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                        patch_artist=True, manage_ticks=False, showfliers=SHOWFLIERS)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.4)

    ax.set_xticks(range(1, len(seasons) + 1))
    ax.set_xticklabels(seasons)
    ax.set_xlabel("Season")
    if is_cv:
        ax.set_ylabel(ylabel or "CV (%)")
        ax.set_ylim(cv_lim)
        ax.set_title(title or "Seasonal boxplot of CV (%) by station")
    else:
        ax.set_ylabel(ylabel or f"{species} ({units})")
        ax.set_ylim(sp_lim)
        ax.set_title(title or f"Seasonal boxplot of {species} ({units}) by station")
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
# 12. SCATTER: center vs cv
# ============================================================
def plot_center_vs_cv_scatter(df, stations=None, mode=None, sector_type=None,
                              aggregate_cv_over_sectors=True, out_path=None, figsize=(8, 6),
                              sp_lim=None, cv_lim=None, scatter_color="tab:blue",
                              species="O3", units="ppb"):
    pairs = prepare_center_cv_pairs(df=df, stations=stations, mode=mode, sector_type=sector_type,
                                    aggregate_cv_over_sectors=aggregate_cv_over_sectors)
    corr = safe_corr(pairs["center_ppb"], pairs["cv_w"], method="pearson")

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(pairs["center_ppb"], pairs["cv_w"] * 100.0, alpha=0.6, color=scatter_color)
    ax.set_xlabel(f"{species} ({units})")
    ax.set_xlim(sp_lim)
    ax.set_ylabel("Coefficient of variation (%)")
    ax.set_ylim(cv_lim)
    ax.set_title(f"{species} central cell vs CV | Pearson r={corr:.3f}" if pd.notna(corr)
                 else f"{species} central cell vs CV")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return pairs, corr, fig, ax


def plot_center_vs_cv_scatter_by_season(df, stations=None, mode=None, sector_type=None,
                                        aggregate_cv_over_sectors=True, out_path=None, figsize=(14, 10),
                                        sp_lim=None, cv_lim=None, scatter_color="tab:blue",
                                        species="O3", units="ppb"):
    pairs = prepare_center_cv_pairs(df=df, stations=stations, mode=mode, sector_type=sector_type,
                                    aggregate_cv_over_sectors=aggregate_cv_over_sectors)

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, season in zip(axes, seasons):
        sub = pairs[pairs["season"] == season]
        corr = safe_corr(sub["center_ppb"], sub["cv_w"])
        if not sub.empty:
            ax.scatter(sub["center_ppb"], sub["cv_w"] * 100.0, alpha=0.6, color=scatter_color)
        ax.set_title(f"{season} | r={corr:.3f}" if pd.notna(corr) else f"{season} | r=nan")
        ax.set_xlabel(f"{species} central cell ({units})")
        ax.set_xlim(sp_lim)
        ax.set_ylabel("Coefficient of Variation (%)")
        ax.set_ylim(cv_lim)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{species} central cell vs CV by season", y=1.02)
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return pairs, fig, axes


def plot_center_vs_cv_scatter_by_month(df, stations=None, mode=None, sector_type=None,
                                       aggregate_cv_over_sectors=True, out_path=None, figsize=(18, 12),
                                       sp_lim=None, cv_lim=None, scatter_color="tab:blue",
                                       species="O3", units="ppb"):
    pairs = prepare_center_cv_pairs(df=df, stations=stations, mode=mode, sector_type=sector_type,
                                    aggregate_cv_over_sectors=aggregate_cv_over_sectors)

    fig, axes = plt.subplots(3, 4, figsize=figsize, sharex=True, sharey=True)
    axes = axes.ravel()

    for i, m in enumerate(range(1, 13)):
        ax = axes[i]
        sub = pairs[pairs["month"] == m]
        corr = safe_corr(sub["center_ppb"], sub["cv_w"])
        if not sub.empty:
            ax.scatter(sub["center_ppb"], sub["cv_w"] * 100.0, alpha=0.6, color=scatter_color)
        ax.set_title(f"{calendar.month_abbr[m]} | r={corr:.3f}" if pd.notna(corr)
                     else f"{calendar.month_abbr[m]} | r=nan")
        ax.set_xlabel(f"{species} ({units})")
        ax.set_xlim(sp_lim)
        ax.set_ylabel("Coefficient of Variation (%)")
        ax.set_ylim(cv_lim)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{species} central cell vs CV by month", y=1.02)
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return pairs, fig, axes


# ============================================================
# 13. ALTITUDE RELATIONS (multi only)
# ============================================================
def plot_altitude_relations(df, stations=None, mode=None, sector_type=None,
                            center_col="center_ppb", cv_col="cv_w", out_prefix=None,
                            sp_lim=None, cv_lim=None, species="O3", units="ppb", figsize=(8, 6)):
    tbl = altitude_relation_table(df=df, stations=stations, center_col=center_col, cv_col=cv_col,
                                  mode=mode, sector_type=sector_type)
    results = {}

    # mean vs altitude
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.scatter(tbl["Altitude"], tbl["center_mean"], alpha=0.8)
    r1 = safe_corr(tbl["Altitude"], tbl["center_mean"])
    ax1.set_xlabel("Altitude (m)")
    ax1.set_ylabel(f"Mean {species} ({units})")
    ax1.set_ylim(sp_lim)
    ax1.set_title(f"Mean {species} vs Altitude | r={r1:.3f}" if pd.notna(r1) else f"Mean {species} vs Altitude")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    if out_prefix:
        p = f"{out_prefix}_center_mean_vs_altitude.png"
        ensure_dir(p)
        fig1.savefig(p, dpi=200, bbox_inches="tight")
    plt.show()
    results["center_mean_vs_altitude_corr"] = r1

    # mean CV vs altitude
    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.scatter(tbl["Altitude"], tbl["cv_w_mean"] * 100, alpha=0.8)
    r2 = safe_corr(tbl["Altitude"], tbl["cv_w_mean"])
    ax2.set_xlabel("Altitude (m)")
    ax2.set_ylabel("Mean CV (%)")
    ax2.set_ylim(cv_lim)
    ax2.set_title(f"Mean CV vs Altitude | r={r2:.3f}" if pd.notna(r2) else "Mean CV vs Altitude")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    if out_prefix:
        p = f"{out_prefix}_cvw_mean_vs_altitude.png"
        ensure_dir(p)
        fig2.savefig(p, dpi=200, bbox_inches="tight")
    plt.show()
    results["cvw_mean_vs_altitude_corr"] = r2

    # variability (std) vs altitude  (autoscaled — std has its own range)
    fig3, ax3 = plt.subplots(figsize=figsize)
    ax3.scatter(tbl["Altitude"], tbl["center_std"], alpha=0.8)
    r3 = safe_corr(tbl["Altitude"], tbl["center_std"])
    ax3.set_xlabel("Altitude (m)")
    ax3.set_ylabel(f"Std of {species} ({units})")
    ax3.set_title(f"Variability of {species} vs Altitude | r={r3:.3f}" if pd.notna(r3)
                  else f"Variability of {species} vs Altitude")
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
# 14. RUNNERS
# ============================================================
def run_full_station_analysis(df, station, stations_df=None, species="O3", units="ppb", mode=None,
                              sector_type="CUM", out_dir=".", center_col="center_ppb", cv_col="cv_w",
                              sector_col="sector", aggregate_cv_over_sectors=True, merge_years=False,
                              sp_lim=None, cv_lim=None, ratio_lim=None, cv_sector=None):
    os.makedirs(out_dir, exist_ok=True)

    if PLOT_CENTER_TS:
        plot_center_timeseries_full_period(
            df=df, stations=station, stations_df=stations_df, center_col=center_col, mode=mode,
            species=species, units=units, sp_lim=sp_lim,
            out_path=os.path.join(out_dir, f"{station}_{species}_full_period_{center_col}.png"))

    if PLOT_RATIO:
        plot_one_station_ratio_meanw_to_center(
            df=df, station=station, stations_df=stations_df, center_col=center_col, mean_col="mean_w",
            sector_col=sector_col, sector_type=sector_type, mode=mode, ratio_lim=ratio_lim, species=species,
            out_path=os.path.join(out_dir, f"{station}_{species}_ratio_meanw_to_center_by_sector.png"),
            merge_years=False)
        if RATIO_SEASONAL_BOX:
            plot_one_station_seasonal_boxplot_ratio_meanw_to_center(
                df=df, station=station, center_col=center_col, mean_col="mean_w", sector_col=sector_col,
                sector_type=sector_type, mode=mode, species=species, ratio_lim=ratio_lim,
                out_path=os.path.join(out_dir, f"{station}_{species}_seasonal_box_ratio.png"))

    if PLOT_CV_TS:
        plot_one_station_cv_timeseries_full_period(
            df=df, station=station, stations_df=stations_df, cv_col=cv_col, sector_col=sector_col,
            sector_type=sector_type, mode=mode, species=species, cv_lim=cv_lim,
            out_path=os.path.join(out_dir, f"{station}_{species}_full_period_{cv_col}_by_sector.png"))

    if PLOT_SEASONAL_CENTER:
        plot_one_station_seasonal_center(
            df=df, station=station, stations_df=stations_df, center_col=center_col, mode=mode,
            species=species, units=units, sp_lim=sp_lim, merge_years=merge_years,
            out_path=os.path.join(out_dir, f"{station}_{species}_seasonal_{center_col}.png"))

    if PLOT_SEASONAL_CV_SECTOR:
        plot_one_station_seasonal_cv_by_sector(
            df=df, station=station, stations_df=stations_df, cv_col=cv_col, sector_col=sector_col,
            sector_type=sector_type, mode=mode, cv_lim=cv_lim, species=species, merge_years=merge_years,
            out_path=os.path.join(out_dir, f"{station}_{species}_seasonal_{cv_col}_by_sector.png"))

    if PLOT_BOXPLOTS:
        plot_monthly_boxplot(
            df=df, value_col=center_col, stations=station, mode=mode, sp_lim=sp_lim,
            title=f"{station} monthly boxplot of {species} ({units})", ylabel=f"{species} ({units})",
            out_path=os.path.join(out_dir, f"{station}_{species}_monthly_box_{center_col}.png"))
        plot_monthly_boxplot(
            df=df, value_col=cv_col, stations=station, mode=mode, sector_type=sector_type, cv_lim=cv_lim,
            title=f"{station} monthly boxplot of CV (%)", ylabel="CV (%)",
            aggregate_cv_over_sectors=aggregate_cv_over_sectors, cv_sector=cv_sector, sector_col=sector_col,
            out_path=os.path.join(out_dir, f"{station}_{species}_monthly_box_{cv_col}.png"))
        plot_seasonal_boxplot(
            df=df, value_col=center_col, stations=station, mode=mode, sp_lim=sp_lim,
            title=f"{station} seasonal boxplot of {species} ({units})", ylabel=f"{species} ({units})",
            out_path=os.path.join(out_dir, f"{station}_{species}_seasonal_box_{center_col}.png"))
        plot_seasonal_boxplot(
            df=df, value_col=cv_col, stations=station, mode=mode, sector_type=sector_type, cv_lim=cv_lim,
            title=f"{station} seasonal boxplot of CV (%)", ylabel="CV (%)",
            aggregate_cv_over_sectors=aggregate_cv_over_sectors, cv_sector=cv_sector, sector_col=sector_col,
            out_path=os.path.join(out_dir, f"{station}_{species}_seasonal_box_{cv_col}.png"))

    pairs = pairs_season = pairs_month = None
    corr_all = np.nan
    if PLOT_SCATTER:
        pairs, corr_all, _, _ = plot_center_vs_cv_scatter(
            df=df, stations=station, mode=mode, sector_type=sector_type, cv_lim=cv_lim, sp_lim=sp_lim,
            species=species, units=units, aggregate_cv_over_sectors=aggregate_cv_over_sectors,
            out_path=os.path.join(out_dir, f"{station}_{species}_scatter_{center_col}_vs_{cv_col}.png"))
        pairs_season, _, _ = plot_center_vs_cv_scatter_by_season(
            df=df, stations=station, mode=mode, sector_type=sector_type, sp_lim=sp_lim, cv_lim=cv_lim,
            species=species, units=units, aggregate_cv_over_sectors=aggregate_cv_over_sectors,
            out_path=os.path.join(out_dir, f"{station}_{species}_scatter_by_season.png"))
        pairs_month, _, _ = plot_center_vs_cv_scatter_by_month(
            df=df, stations=station, mode=mode, sector_type=sector_type, sp_lim=sp_lim, cv_lim=cv_lim,
            species=species, units=units, aggregate_cv_over_sectors=aggregate_cv_over_sectors,
            out_path=os.path.join(out_dir, f"{station}_{species}_scatter_by_month.png"))

    if SAVE_STATS_CSV:
        sf = build_stats_frame(df, stations=station, center_col=center_col, cv_col=cv_col,
                               sector_col=sector_col, mode=mode, sector_type=sector_type,
                               cv_sector=cv_sector, aggregate_cv_over_sectors=aggregate_cv_over_sectors)
        monthly_stats_table(sf, value_cols=[center_col, cv_col]).to_csv(
            os.path.join(out_dir, f"{station}_{species}_monthly_stats.csv"), index=False)
        seasonal_stats_table(sf, value_cols=[center_col, cv_col]).to_csv(
            os.path.join(out_dir, f"{station}_{species}_seasonal_stats.csv"), index=False)
        if pairs_month is not None:
            correlation_by_month(pairs_month).to_csv(
                os.path.join(out_dir, f"{station}_{species}_corr_by_month.csv"), index=False)
        if pairs_season is not None:
            correlation_by_season(pairs_season).to_csv(
                os.path.join(out_dir, f"{station}_{species}_corr_by_season.csv"), index=False)

    print(f"Saved one-station analysis for {station} in: {out_dir}")
    print(f"Overall Pearson correlation ({center_col} vs {cv_col}): {corr_all}")
    return {"pairs": pairs, "overall_corr": corr_all}


def run_full_multi_station_analysis(df, stations, stations_df=None, species="O3", units="ppb", mode=None,
                                    sector_type="CUM", out_dir=".", center_col="center_ppb", cv_col="cv_w",
                                    aggregate_cv_over_sectors=True, merge_years=False, scatter_color="tab:blue",
                                    sp_lim=None, cv_lim=None, cv_sector=None, sector_col="sector"):
    os.makedirs(out_dir, exist_ok=True)

    if PLOT_CENTER_TS:
        plot_center_timeseries_full_period(
            df=df, stations=stations, stations_df=stations_df, center_col=center_col, mode=mode,
            species=species, units=units, sp_lim=sp_lim,
            out_path=os.path.join(out_dir, f"multi_station_{species}_full_period_{center_col}.png"))

    if PLOT_SEASONAL_CENTER:
        plot_multi_station_seasonal_center(
            df=df, stations=stations, stations_df=stations_df, center_col=center_col, mode=mode,
            species=species, units=units, merge_years=merge_years, sp_lim=sp_lim,
            out_path=os.path.join(out_dir, f"multi_station_{species}_seasonal_center.png"))

    if PLOT_BOXPLOTS:
        plot_monthly_boxplot_by_station(
            df=df, value_col=center_col, stations=stations, mode=mode, sp_lim=sp_lim,
            species=species, units=units,
            out_path=os.path.join(out_dir, f"multi_station_{species}_monthly_box_{center_col}.png"))
        plot_monthly_boxplot_by_station(
            df=df, value_col=cv_col, stations=stations, mode=mode, sector_type=sector_type, cv_lim=cv_lim,
            aggregate_cv_over_sectors=aggregate_cv_over_sectors, cv_sector=cv_sector, sector_col=sector_col,
            species=species, units=units,
            out_path=os.path.join(out_dir, f"multi_station_{species}_monthly_box_{cv_col}.png"))
        plot_seasonal_boxplot_by_station(
            df=df, value_col=center_col, stations=stations, mode=mode, sp_lim=sp_lim,
            species=species, units=units,
            out_path=os.path.join(out_dir, f"multi_station_{species}_seasonal_box_{center_col}.png"))
        plot_seasonal_boxplot_by_station(
            df=df, value_col=cv_col, stations=stations, mode=mode, sector_type=sector_type, cv_lim=cv_lim,
            sp_lim=sp_lim, aggregate_cv_over_sectors=aggregate_cv_over_sectors, cv_sector=cv_sector,
            sector_col=sector_col, species=species, units=units,
            out_path=os.path.join(out_dir, f"multi_station_{species}_seasonal_box_{cv_col}.png"))

    pairs = pairs_season = pairs_month = None
    corr_all = np.nan
    if PLOT_SCATTER:
        pairs, corr_all, _, _ = plot_center_vs_cv_scatter(
            df=df, stations=stations, mode=mode, sector_type=sector_type, cv_lim=cv_lim, sp_lim=sp_lim,
            species=species, units=units, scatter_color=scatter_color,
            aggregate_cv_over_sectors=aggregate_cv_over_sectors,
            out_path=os.path.join(out_dir, f"multi_station_{species}_scatter_{center_col}_vs_{cv_col}.png"))
        pairs_season, _, _ = plot_center_vs_cv_scatter_by_season(
            df=df, stations=stations, mode=mode, sector_type=sector_type, cv_lim=cv_lim, sp_lim=sp_lim,
            species=species, units=units, scatter_color=scatter_color,
            aggregate_cv_over_sectors=aggregate_cv_over_sectors,
            out_path=os.path.join(out_dir, f"multi_station_{species}_scatter_by_season.png"))
        pairs_month, _, _ = plot_center_vs_cv_scatter_by_month(
            df=df, stations=stations, mode=mode, sector_type=sector_type, cv_lim=cv_lim, sp_lim=sp_lim,
            species=species, units=units, scatter_color=scatter_color,
            aggregate_cv_over_sectors=aggregate_cv_over_sectors,
            out_path=os.path.join(out_dir, f"multi_station_{species}_scatter_by_month.png"))

    if PLOT_ALTITUDE:
        alt_tbl, alt_corrs = plot_altitude_relations(
            df=df, stations=stations, mode=mode, sector_type=sector_type, center_col=center_col,
            cv_col=cv_col, species=species, units=units, sp_lim=sp_lim, cv_lim=cv_lim,
            out_prefix=os.path.join(out_dir, f"multi_station_{species}_altitude"))
        print("Altitude correlations:", alt_corrs)
        if SAVE_STATS_CSV:
            alt_tbl.to_csv(os.path.join(out_dir, f"multi_station_{species}_altitude_relation_table.csv"), index=False)

    if SAVE_STATS_CSV:
        sf = build_stats_frame(df, stations=stations, center_col=center_col, cv_col=cv_col,
                               sector_col=sector_col, mode=mode, sector_type=sector_type,
                               cv_sector=cv_sector, aggregate_cv_over_sectors=aggregate_cv_over_sectors)
        monthly_stats_table(sf, value_cols=[center_col, cv_col]).to_csv(
            os.path.join(out_dir, f"multi_station_{species}_monthly_stats.csv"), index=False)
        seasonal_stats_table(sf, value_cols=[center_col, cv_col]).to_csv(
            os.path.join(out_dir, f"multi_station_{species}_seasonal_stats.csv"), index=False)
        if pairs_month is not None:
            correlation_by_month(pairs_month).to_csv(
                os.path.join(out_dir, f"multi_station_{species}_corr_by_month.csv"), index=False)
        if pairs_season is not None:
            correlation_by_season(pairs_season).to_csv(
                os.path.join(out_dir, f"multi_station_{species}_corr_by_season.csv"), index=False)

    print(f"Saved multi-station analysis in: {out_dir}")
    print(f"Overall Pearson correlation ({center_col} vs {cv_col}): {corr_all}")
    return {"pairs": pairs, "overall_corr": corr_all}


# ============================================================
# FINAL EXECUTION BLOCK
# ============================================================
if __name__ == "__main__":
    start = time.time()
    print("\n===== PRODUCTION RUN =====")
    print(f"Species: {species} | Mode: {mode} | RUN_MODE: {RUN_MODE}")
    print("Start:", dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"))

    stations_df = load_stations_file(STATIONS_PATH)
    res_one = res_multi = None

    if RUN_MODE in ("one", "both"):
        csv_path = Path(SPECIES_DIR) / f"{station}_{species}_{mode}_30min.csv"
        parquet_path = Path(SPECIES_DIR) / f"{station}_{species}_{mode}_30min.parquet"
        if csv_path.exists():
            file_path = csv_path
        elif parquet_path.exists():
            file_path = parquet_path
        else:
            raise FileNotFoundError(
                f"No input file found for station {station}. Checked: {csv_path} and {parquet_path}")

        df_one = load_station_timeseries_csv(str(file_path), station_name=station)
        df_one = attach_station_metadata(df_one, stations_df)
        df_one = filter_time_period(df_one, TIME_PERIOD_MODE, START_DATE, END_DATE)

        res_one = run_full_station_analysis(
            df=df_one, station=station, stations_df=stations_df, species=species, units=units, mode=mode,
            sector_type=SECTOR_TYPE, out_dir=out_dir_one, center_col=CENTER_COL, cv_col=CV_COL,
            sector_col=SECTOR_COL, ratio_lim=RATIO_LIM, aggregate_cv_over_sectors=AGGREGATE_CV_OVER_SECTORS,
            merge_years=MERGE_SEASONAL_YEARS, sp_lim=SP_LIM, cv_lim=CV_LIM, cv_sector=CV_SECTOR)

    if RUN_MODE in ("multi", "both", "all"):
        stations_resolved = resolve_stations(stations=stations, input_dir=SPECIES_DIR, species=species, mode=mode)
        print(f"Stations selected: {stations_resolved}")

        df_all = load_many_station_csvs(input_dir=SPECIES_DIR, species=species, mode=mode,
                                        station_names=stations_resolved)
        df_all = attach_station_metadata(df_all, stations_df)
        df_all = filter_time_period(df_all, TIME_PERIOD_MODE, START_DATE, END_DATE)

        res_multi = run_full_multi_station_analysis(
            df=df_all, stations=stations_resolved, stations_df=stations_df, species=species, units=units,
            mode=mode, sector_type=SECTOR_TYPE, out_dir=out_dir_multi, center_col=CENTER_COL, cv_col=CV_COL,
            aggregate_cv_over_sectors=AGGREGATE_CV_OVER_SECTORS, sector_col=SECTOR_COL,
            merge_years=MERGE_SEASONAL_YEARS, sp_lim=SP_LIM, cv_lim=CV_LIM, cv_sector=CV_SECTOR,
            scatter_color=SCATTER_COLOR)

    end = time.time()
    print("End:", dt.datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Execution time: {(end - start) / 60:.2f} minutes")
        
# %%
