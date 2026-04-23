#%%
from __future__ import annotations

from pathlib import Path
import math
import datetime as dt
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================
BASE_DIR = Path("/home/agkiokas/CAMS")

# Input files
ZARR_PATH =Path("/mnt/store01/agkiokas/CAMS/O3_reduced_archive.zarr/")
HLOOKUP_PATH = BASE_DIR /"lookups/station_horizontal_lookup_all.parquet"
LEVEL_TS_PATH = BASE_DIR /"lookups/station_level_timeseries_all.parquet"
DECISION_TABLE_PATH = BASE_DIR / "lookups/station_k_decision_table.csv"
CONSTANT_K_PATH = BASE_DIR / "lookups/station_constant_k_only_lookup.csv"

# Output directories
OUT_DIR = BASE_DIR / "climatological_outputs"
MAP_DIR = OUT_DIR / "maps"
NC_DIR = OUT_DIR / "nc"
CSV_DIR = OUT_DIR / "csv"

# Variable in Zarr
VAR_NAME = "O3"

# One station, many stations, or "ALL"
SELECTED_STATIONS: Sequence[str] | str = ["1001A", "1002A", "1003A", "1004A", "1005A"]

# Sectors C1..C10 -> box sizes 3x3, 5x5, ..., 21x21
MAX_SECTOR = 10

# Which sectors to plot/export:
# - "ALL" -> C1..C10
# - [1] -> only C1
# - [1, 3, 5] -> selected sectors
SELECTED_SECTORS: Sequence[int] | str = "ALL"

# Weighted statistics with cos(phi)
USE_AREA_WEIGHTING = True

# Plot only weighted CV maps (cv_w shown as percent)
PLOT_ONLY_WEIGHTED_CV = True

# Color scale options for plots:
# - True  -> same color scale across all requested sectors / seasons / day-night groups
# - False -> each plot gets its own scale
USE_COMMON_COLOR_SCALE = True

# Optional manual limits for CV(%) color scale.
# Set to None for automatic behavior.
CV_PERCENT_VMIN = None
CV_PERCENT_VMAX = None

# Day/night threshold
SOLAR_THRESHOLD_DEG = -0.833

# Plot size / DPI
FIGSIZE = (10, 7)
DPI = 150

# Map extent padding in degrees
MAP_PAD_DEG = 1.5

# Save NetCDF summary file
SAVE_NETCDF = True
SAVE_CSV = True
MAKE_PLOTS = True


# ============================================================
# DAY / NIGHT FUNCTION PROVIDED BY USER
# ============================================================
def solar_day_mask(lat2d, lon2d, ts, threshold_deg=-0.833):
    """
    Vectorized day/night mask over the whole grid using solar elevation angle.
    Returns True where it is day.
    """
    lon2d = ((lon2d + 180.0) % 360.0) - 180.0
    lat_rad = np.deg2rad(lat2d.astype(float))

    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()

    if ts.tzinfo is not None:
        ts_utc = ts.astimezone(dt.timezone.utc)
    else:
        ts_utc = ts

    doy = ts_utc.timetuple().tm_yday
    hour = ts_utc.hour + ts_utc.minute / 60.0 + ts_utc.second / 3600.0

    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (hour - 12.0) / 24.0)

    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.001480 * np.sin(3 * gamma)
    )

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    tst = (hour * 60.0 + eqtime + 4.0 * lon2d) % 1440.0
    ha = np.deg2rad((tst / 4.0) - 180.0)

    cosz = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, -1.0, 1.0)
    zen = np.arccos(cosz)

    elev_deg = np.rad2deg(np.pi / 2.0 - zen)

    return elev_deg > float(threshold_deg)


# ============================================================
# HELPERS
# ============================================================
def ensure_dirs():
    for d in [OUT_DIR, MAP_DIR, NC_DIR, CSV_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def normalize_sector_selection(sectors: Sequence[int] | str, max_sector: int = 10) -> list[int]:
    if isinstance(sectors, str):
        if sectors.upper() == "ALL":
            return list(range(1, max_sector + 1))
        raise ValueError("SELECTED_SECTORS must be 'ALL' or a list like [1, 3, 5]")

    out = sorted({int(s) for s in sectors})
    bad = [s for s in out if s < 1 or s > max_sector]
    if bad:
        raise ValueError(f"Invalid sector numbers: {bad}. Valid range is 1..{max_sector}")
    return out


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def normalize_station_selection(stations: Sequence[str] | str, available: Iterable[str]) -> list[str]:
    available = list(pd.Index(available).astype(str))
    if isinstance(stations, str):
        if stations.upper() == "ALL":
            return sorted(available)
        return [stations]
    return [str(s) for s in stations]


def infer_time_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.lower() in {"time", "datetime", "timestamp", "date"}]
    if not candidates:
        raise ValueError(f"Could not find a time column in columns: {list(df.columns)}")
    return candidates[0]


def infer_station_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.lower() in {"station_name", "station", "station_id", "name"}]
    if not candidates:
        raise ValueError(f"Could not find a station column in columns: {list(df.columns)}")
    return candidates[0]


def infer_level_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.lower() in {"level_idx", "k", "lev_idx", "level"}]
    if not candidates:
        raise ValueError(f"Could not find a level column in columns: {list(df.columns)}")
    return candidates[0]


def build_original_to_local_level_map(ds: xr.Dataset, decision_df: pd.DataFrame, level_ts_df: pd.DataFrame) -> dict[int, int]:
    """
    Maps original station level indices (e.g. 16..22) to local Zarr level positions (0..6).

    Assumption used here:
    - reduced Zarr contains only the kept levels in ascending order
    - original indices appearing in CSV/parquet correspond in order to the reduced Zarr level axis
    """
    zarr_nlev = ds.sizes["lev"]

    level_candidates = []
    for df in [decision_df, level_ts_df]:
        for c in df.columns:
            if c.lower() in {"k_min", "k_max", "level_idx", "k", "lev_idx", "level", "level_idx_constant"}:
                vals = pd.to_numeric(df[c], errors="coerce").dropna().astype(int).tolist()
                level_candidates.extend(vals)

    uniq = sorted(set(level_candidates))
    if len(uniq) < zarr_nlev:
        raise ValueError(
            f"Found only {len(uniq)} original station levels in metadata, but Zarr has {zarr_nlev} levels. "
            f"Original levels found: {uniq}"
        )

    kept_original = uniq[:zarr_nlev] if len(uniq) == zarr_nlev else uniq[-zarr_nlev:]
    return {orig_k: local_k for local_k, orig_k in enumerate(kept_original)}


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan, np.nan
    v = values[mask]
    w = weights[mask]
    wsum = w.sum()
    if wsum <= 0:
        return np.nan, np.nan
    mean = np.sum(w * v) / wsum
    var = np.sum(w * (v - mean) ** 2) / wsum
    return float(mean), float(np.sqrt(var))


def unweighted_mean_std(values: np.ndarray) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=0))


def sector_bounds(i0: int, j0: int, sector: int, ny: int, nx: int) -> tuple[slice, slice]:
    r = sector
    i1 = max(0, i0 - r)
    i2 = min(ny, i0 + r + 1)
    j1 = max(0, j0 - r)
    j2 = min(nx, j0 + r + 1)
    return slice(i1, i2), slice(j1, j2)


def classify_group(ts: pd.Timestamp, is_day: bool) -> dict[str, str]:
    season = season_from_month(ts.month)
    daynight = "day" if is_day else "night"
    return {
        "all": "ALL",
        "season": season,
        "daynight": daynight,
        "season_daynight": f"{season}_{daynight}",
    }


def make_group_index(times: pd.DatetimeIndex, central_lat: np.ndarray, central_lon: np.ndarray) -> pd.DataFrame:
    rows = []
    for ts in times:
        is_day = bool(solar_day_mask(central_lat, central_lon, ts, threshold_deg=SOLAR_THRESHOLD_DEG)[0, 0])
        rows.append({"time": ts, **classify_group(ts, is_day)})
    return pd.DataFrame(rows)


# ============================================================
# CORE PROCESSING
# ============================================================
def load_inputs():
    ds = xr.open_zarr(ZARR_PATH, consolidated=False)

    hlookup = pd.read_parquet(HLOOKUP_PATH)
    level_ts = pd.read_parquet(LEVEL_TS_PATH)
    decision = pd.read_csv(DECISION_TABLE_PATH)
    constant_k = pd.read_csv(CONSTANT_K_PATH)

    return ds, hlookup, level_ts, decision, constant_k


def prepare_station_table(hlookup: pd.DataFrame, selected_stations: list[str]) -> pd.DataFrame:
    stcol = infer_station_column(hlookup)
    out = hlookup.copy()
    out[stcol] = out[stcol].astype(str)
    out = out[out[stcol].isin(selected_stations)].copy()
    if out.empty:
        raise ValueError("No selected stations found in station_horizontal_lookup_all.parquet")
    return out.rename(columns={stcol: "Station_Name"})


def prepare_level_timeseries(level_ts: pd.DataFrame, selected_stations: list[str]) -> pd.DataFrame:
    stcol = infer_station_column(level_ts)
    tcol = infer_time_column(level_ts)
    lcol = infer_level_column(level_ts)

    out = level_ts.copy()
    out[stcol] = out[stcol].astype(str)
    out = out[out[stcol].isin(selected_stations)].copy()
    out[tcol] = pd.to_datetime(out[tcol])
    out = out.rename(columns={stcol: "Station_Name", tcol: "time", lcol: "orig_level_idx"})
    return out[["Station_Name", "time", "orig_level_idx"]]


def compute_station_sector_timeseries(
    ds: xr.Dataset,
    station_row: pd.Series,
    level_ts_station: pd.DataFrame,
    orig_to_local_level: dict[int, int],
) -> pd.DataFrame:
    station = str(station_row["Station_Name"])
    i0 = int(station_row["i"])
    j0 = int(station_row["j"])

    times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    ny = ds.sizes["lat"]
    nx = ds.sizes["lon"]

    # Central grid lat/lon only for day/night classification of the station cell
    central_lat = np.array([[float(ds["lat"].values[i0])]])
    central_lon = np.array([[float(ds["lon"].values[j0])]])
    group_df = make_group_index(times, central_lat, central_lon)

    # Map station-specific level time series to local zarr level positions
    level_map_df = pd.DataFrame({"time": times}).merge(level_ts_station, on="time", how="left")
    level_map_df["orig_level_idx"] = level_map_df["orig_level_idx"].astype("Int64")
    level_map_df["local_lev"] = level_map_df["orig_level_idx"].map(orig_to_local_level)

    if level_map_df["local_lev"].isna().all():
        raise ValueError(f"All local levels are NaN after mapping for station {station}. Check level index mapping.")

    lat_vals = ds["lat"].values.astype(float)

    rows = []
    for sector in range(1, MAX_SECTOR + 1):
        islc, jslc = sector_bounds(i0, j0, sector, ny, nx)
        lat_sector = lat_vals[islc]
        cosw_2d = np.cos(np.deg2rad(lat_sector))[:, None] * np.ones((islc.stop - islc.start, jslc.stop - jslc.start))

        for t_idx, ts in enumerate(times):
            lev_local = level_map_df.iloc[t_idx]["local_lev"]
            if pd.isna(lev_local):
                continue
            lev_local = int(lev_local)

            arr = ds[VAR_NAME].isel(time=t_idx, lev=lev_local, lat=islc, lon=jslc).values.astype(float)

            mean_u, std_u = unweighted_mean_std(arr)
            cv_u = np.nan if (not np.isfinite(mean_u) or mean_u == 0) else std_u / mean_u

            mean_w, std_w = weighted_mean_std(arr, cosw_2d)
            cv_w = np.nan if (not np.isfinite(mean_w) or mean_w == 0) else std_w / mean_w

            g = group_df.iloc[t_idx]
            rows.append(
                {
                    "Station_Name": station,
                    "time": ts,
                    "sector": f"C{sector}",
                    "sector_num": sector,
                    "mean": mean_u,
                    "std": std_u,
                    "cv": cv_u,
                    "mean_w": mean_w,
                    "std_w": std_w,
                    "cv_w": cv_w,
                    "group_all": g["all"],
                    "group_season": g["season"],
                    "group_daynight": g["daynight"],
                    "group_season_daynight": g["season_daynight"],
                }
            )

    return pd.DataFrame(rows)


def aggregate_climatologies(ts_df: pd.DataFrame, station_meta: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    group_specs = [
        ("all", "group_all"),
        ("season", "group_season"),
        ("daynight", "group_daynight"),
        ("season_daynight", "group_season_daynight"),
    ]

    for group_type, gcol in group_specs:
        tmp = (
            ts_df.groupby(["Station_Name", "sector", "sector_num", gcol], dropna=False)[
                ["mean", "std", "cv", "mean_w", "std_w", "cv_w"]
            ]
            .mean()
            .reset_index()
            .rename(columns={gcol: "group_value"})
        )
        tmp["group_type"] = group_type
        pieces.append(tmp)

    out = pd.concat(pieces, axis=0, ignore_index=True)
    out = out.merge(
        station_meta[["Station_Name", "Latitude", "Longitude", "Altitude", "i", "j", "model_lat", "model_lon"]],
        on="Station_Name",
        how="left",
    )
    return out


def to_xarray_dataset(clim_df: pd.DataFrame) -> xr.Dataset:
    df = clim_df.copy()
    df["sector_num"] = df["sector_num"].astype(int)

    ds = df.set_index(["group_type", "group_value", "sector_num", "Station_Name"]).to_xarray()
    return ds


# ============================================================
# PLOTTING
# ============================================================
def compute_global_cv_percent_limits(clim_df: pd.DataFrame, selected_sectors: list[int]) -> tuple[float | None, float | None]:
    sub = clim_df[
        clim_df["sector_num"].isin(selected_sectors) & np.isfinite(clim_df["cv_w"])
    ].copy()
    if sub.empty:
        return None, None

    vals = 100.0 * sub["cv_w"].astype(float).values
    if vals.size == 0:
        return None, None

    return float(np.nanmin(vals)), float(np.nanmax(vals))



def plot_station_network_map_cv(
    clim_df: pd.DataFrame,
    group_type: str,
    group_value: str,
    sector_num: int,
    vmin: float | None = None,
    vmax: float | None = None,
):
    metric_col = "cv_w"
    sub = clim_df[
        (clim_df["group_type"] == group_type)
        & (clim_df["group_value"] == group_value)
        & (clim_df["sector_num"] == sector_num)
    ].copy()

    sub = sub[np.isfinite(sub[metric_col])].copy()
    if sub.empty:
        return

    # Convert CV to percent for plotting
    sub["cv_percent"] = 100.0 * sub[metric_col].astype(float)

    x = sub["model_lon"].values
    y = sub["model_lat"].values
    c = sub["cv_percent"].values

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sc = ax.scatter(x, y, c=c, s=140, marker="o", vmin=vmin, vmax=vmax)

    for _, r in sub.iterrows():
        ax.text(
            r["model_lon"] + 0.03,
            r["model_lat"] + 0.03,
            str(r["Station_Name"]),
            fontsize=7,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{VAR_NAME} climatology | CV (%) | {group_type}={group_value} | C{sector_num}"
    )
    ax.set_xlim(sub["model_lon"].min() - MAP_PAD_DEG, sub["model_lon"].max() + MAP_PAD_DEG)
    ax.set_ylim(sub["model_lat"].min() - MAP_PAD_DEG, sub["model_lat"].max() + MAP_PAD_DEG)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Coefficient of Variation (%)")
    fig.tight_layout()

    out = MAP_DIR / f"map_cvw_percent_{group_type}_{group_value}_C{sector_num}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)



def make_all_maps(clim_df: pd.DataFrame, selected_sectors: list[int]):
    group_order = {
        "all": ["ALL"],
        "season": ["DJF", "MAM", "JJA", "SON"],
        "daynight": ["day", "night"],
        "season_daynight": [
            "DJF_day", "DJF_night", "MAM_day", "MAM_night",
            "JJA_day", "JJA_night", "SON_day", "SON_night",
        ],
    }

    if CV_PERCENT_VMIN is not None or CV_PERCENT_VMAX is not None:
        global_vmin = CV_PERCENT_VMIN
        global_vmax = CV_PERCENT_VMAX
    elif USE_COMMON_COLOR_SCALE:
        global_vmin, global_vmax = compute_global_cv_percent_limits(clim_df, selected_sectors)
    else:
        global_vmin, global_vmax = None, None

    for group_type, group_values in group_order.items():
        for group_value in group_values:
            for sector_num in selected_sectors:
                vmin = global_vmin if USE_COMMON_COLOR_SCALE or CV_PERCENT_VMIN is not None else None
                vmax = global_vmax if USE_COMMON_COLOR_SCALE or CV_PERCENT_VMAX is not None else None
                plot_station_network_map_cv(
                    clim_df=clim_df,
                    group_type=group_type,
                    group_value=group_value,
                    sector_num=sector_num,
                    vmin=vmin,
                    vmax=vmax,
                )


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_dirs()

    ds, hlookup, level_ts, decision, constant_k = load_inputs()

    station_name_col = infer_station_column(hlookup)
    selected_stations = normalize_station_selection(SELECTED_STATIONS, hlookup[station_name_col].astype(str).unique())
    selected_sectors = normalize_sector_selection(SELECTED_SECTORS, MAX_SECTOR)

    station_meta = prepare_station_table(hlookup, selected_stations)
    level_ts_sel = prepare_level_timeseries(level_ts, selected_stations)

    orig_to_local_level = build_original_to_local_level_map(ds, decision, level_ts_sel)
    print("Original station level -> local Zarr level mapping:")
    print(orig_to_local_level)
    print("Zarr lev values:", ds["lev"].values)

    all_station_ts = []
    for _, srow in station_meta.iterrows():
        station = str(srow["Station_Name"])
        print(f"Processing station {station} ...")
        level_ts_station = level_ts_sel[level_ts_sel["Station_Name"] == station][["time", "orig_level_idx"]].copy()
        ts_df = compute_station_sector_timeseries(
            ds=ds,
            station_row=srow,
            level_ts_station=level_ts_station,
            orig_to_local_level=orig_to_local_level,
        )
        all_station_ts.append(ts_df)

    all_station_ts_df = pd.concat(all_station_ts, axis=0, ignore_index=True)
    clim_df = aggregate_climatologies(all_station_ts_df, station_meta)

    if SAVE_CSV:
        all_station_ts_df.to_csv(CSV_DIR / f"{VAR_NAME}_station_sector_timeseries.csv", index=False)
        clim_df.to_csv(CSV_DIR / f"{VAR_NAME}_station_sector_climatology.csv", index=False)

    if SAVE_NETCDF:
        clim_ds = to_xarray_dataset(clim_df)
        clim_ds.to_netcdf(NC_DIR / f"{VAR_NAME}_station_sector_climatology.nc")

    if MAKE_PLOTS:
        make_all_maps(clim_df, selected_sectors)

    print("Done.")
    print(f"Selected stations: {selected_stations}")
    print(f"Selected sectors: {selected_sectors}")
    print(f"CSV output: {CSV_DIR}")
    print(f"NetCDF output: {NC_DIR}")
    print(f"Map output: {MAP_DIR}")


if __name__ == "__main__":
    main()

# %%
