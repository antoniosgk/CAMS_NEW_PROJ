#THIS IS THE NEW COMPACT VERSION
#SUPORTS ONLY MODE='A' AND TIME_INTERVAL_ANALYSIS
#PRODUCES NO PLOTS.ONLY OBJECTIVE IS TO CREATE CSV/PARQUET FILES FOR EACH STATION
#
# v2 changes (2026-11):
#   - Autodetects columnar (2D) vs 3D fields from the variable's dims.
#     For a columnar field (e.g. AOT / TOTEXTTAU, dims time,lat,lon):
#       * only horizontal collocation is done (no vertical level)
#       * the constant-k and level-timestep lookups are NOT loaded/needed
#       * no MMR->ppb conversion (value stored raw into center_ppb / stats)
#       * k_star_center and z_target_m are NaN
#     Just set product="inst2d", species="AOT" and the script handles it.
#   - Timestep-first engine now reads each distinct level-plane ONCE per file
#     and slices every station's box from it in memory (instead of one strided
#     disk read per station). Identical numbers, much faster for station lists.
#     For columnar fields this is a single plane per file.
#   - C0 is still INCLUDED in the cumulative sector masks (unchanged, by design).
#   - center_ppb schema unchanged.

#%%
# ============================================================
# Compact period-only script with:
#   - baseline station-first run_time_interval
#   - faster timestep-first run_time_interval_timestep_first
#   - hybrid constant-k / timestep-k logic (3D only)
#   - columnar (2D) horizontal-only path
#   - old-vs-new comparison helpers
# ============================================================

from __future__ import annotations

import os
import time
import math
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:  # tqdm optional; falls back to periodic text progress
    _HAVE_TQDM = False


# ============================================================
# USER SETTINGS
# ============================================================

RUN_PERIOD = True
RUN_COMPARE_OLD_NEW = False   # compare baseline vs timestep-first on a short test window
RUN_ENGINE = "new"           # "old" or "new" ,new is the compact(fast) version,old is the one used in main.py

START_DT = dt.datetime(2005, 5, 16, 0, 0)
END_DT   = dt.datetime(2007, 6, 16, 0, 0)

MODE = "A"                   # only "A" implemented in this compact script

STATION_SELECTION = "all"      # "single", "list", "all"
STATION_IDX = 1
STATION_IDXS = [i for i in range(1800, 2050)]

STEP_MINUTES = 30
CELL_NUMS = 10
DIST_BINS_KM = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

WEIGHTED = True
SAVE_CSV = False
SAVE_PARQUET = True

# Resume: when True, skip any station whose *_30min.parquet already exists AND
# opens as a valid parquet (readable footer). Half-written files from a killed
# run are treated as not-done and reprocessed. Default False = always overwrite.
SKIP_EXISTING = True

# Streaming output (bounded memory): the fast engine writes one parquet per
# station incrementally, flushing all station buffers once this many rows are
# buffered in total. Keeps peak RAM ~ MAX_BUFFER_ROWS * ~3 KB per process.
# Recommended run pattern: parallel tmux batches of <= ~200 stations each
# (one open writer per station; stays well under the default 1024 FD limit).
STREAMING_FLUSH = True
# Primary flush cadence: flush ALL station buffers to parquet every this many
# processed timesteps, so files grow incrementally and progress is visible.
FLUSH_EVERY_TIMESTEPS = 2000
# Safety cap: flush earlier if total buffered rows ever exceed this (guards RAM
# when many stations are run in one process). Whichever triggers first wins.
MAX_BUFFER_ROWS = 3_000_000
SHOW_PROGRESS_BAR = True       # tqdm bar over timesteps (falls back to text if tqdm absent)
PROGRESS_EVERY_FILES = 2000    # text-fallback: print a progress line every N files

# Hybrid k lookup (ignored for columnar / 2D fields)
USE_HYBRID_K_LOOKUP = True
VARIABLE_K_STATIONS = [474, 732, 1001, 1120, 1209, 1324, 1784]

# Paths
# 3D example : product="inst3d", species="SO2" / "O3"
# 2D example : product="inst2d", species="AOT"  (variable inside is TOTEXTTAU;
#              resolved automatically; same horizontal lookup, same cadence)
base_path = "/mnt/store01/agkiokas/CAMS/inst/subsets"
product = "inst2d"
species = "AOT"

stations_path = "/home/agkiokas/CAMS/CHINESE_STATIONS_INFO_2015_2023.txt"
out_dir = f"/mnt/store01/agkiokas/CAMS/stations_parquet/{species}"
lookup_dir = "/home/agkiokas/CAMS/lookups/"

HORIZONTAL_LOOKUP_PATH = f"{lookup_dir}/station_horizontal_lookup_all.parquet"
LEVEL_LOOKUP_PATH = f"{lookup_dir}/station_level_timeseries_all.parquet"
CONSTANT_K_LOOKUP_PATH = f"{lookup_dir}/station_constant_k_only_lookup.csv"

# Optional quick comparison window
# NOTE: set these to a SHORT window (e.g. a few days) when RUN_COMPARE_OLD_NEW=True,
# otherwise the validation runs both engines over the full period.
COMPARE_START_DT = START_DT
COMPARE_END_DT = END_DT


# ============================================================
# CONSTANTS / OUTPUT SCHEMA
# ============================================================

EARTH_RADIUS_KM = 6371.0

EXPECTED_COLS = [
    "station", "station_idx", "station_lat", "station_lon", "station_alt",
    "model_lat", "model_lon", "i_center", "j_center",
    "date", "time", "timestamp", "datetime",
    "season", "day_night",
    "mode", "sector_type", "sector", "radius",
    "k_star_center", "z_target_m", "center_ppb",
    "n_total", "n_valid", "n_excluded", "frac_excluded",
    "n", "mean", "std", "cv", "median", "q25", "q75", "iqr",
    "n_w", "mean_w", "std_w", "cv_w", "median_w", "q1_w", "q3_w", "iqr_w",
]

# vertical-dim name candidates used for columnar detection
VERT_DIM_CANDIDATES = {"lev", "level", "ilev", "nhym", "z", "height"}


# ============================================================
# BASIC HELPERS
# ===========================================================
def debug_species_file_structure():
    for d0, t0 in iter_timestamps(START_DT, END_DT, STEP_MINUTES):
        spf, _, _, _, _ = build_paths(base_path, product, species, d0, t0)
        if spf.exists():
            ds = xr.open_dataset(spf, decode_times=False)
            print("First species file:", spf)
            print("Variables:", list(ds.data_vars))
            for v in ds.data_vars:
                print(f"Variable {v}: dims={ds[v].dims}, shape={ds[v].shape}")
            ds.close()
            return
    print("No species file found in requested period.")


def iter_timestamps(start_dt: dt.datetime, end_dt: dt.datetime, step_minutes: int):
    cur = start_dt
    delta = dt.timedelta(minutes=step_minutes)
    while cur < end_dt:
        yield cur.strftime("%Y%m%d"), cur.strftime("%H%M")
        cur += delta


def build_paths(base_path: str, product: str, species: str, date: str, time_: str):
    spf = Path(f"{base_path}/{species}/{product}_{species}_{date}_{time_}.nc4")
    Tf = Path(f"{base_path}/T/{product}_T_{date}_{time_}.nc4")
    PLf = Path(f"{base_path}/PL/{product}_PL_{date}_{time_}.nc4")
    RHf = Path(f"{base_path}/RH/{product}_RH_{date}_{time_}.nc4")
    orogf = Path(f"{base_path}/const/const_2d_asm_Nx_{date}.nc4")
    return spf, Tf, PLf, RHf, orogf


def season_from_datetime(ts: dt.datetime) -> str:
    m = ts.month
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"


def day_night_label(lat, lon, ts: dt.datetime, threshold_deg=-0.833, return_elev=False):
    """
    Day/Night from solar elevation angle.
    - threshold_deg = -0.833 is a common sunrise/sunset threshold (refraction + solar radius).
    - Works with naive datetimes (assumed UTC) or aware datetimes.

    Returns:
      'day' or 'night'
      If return_elev=True -> (label, elev_deg)
    """

    # normalize lon to [-180, 180]
    lon = float(lon)
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat_rad = np.deg2rad(float(lat))

    # treat naive timestamps as UTC
    if ts.tzinfo is not None:
        ts_utc = ts.astimezone(dt.timezone.utc)
    else:
        ts_utc = ts

    doy = ts_utc.timetuple().tm_yday
    hour = ts_utc.hour + ts_utc.minute / 60.0 + ts_utc.second / 3600.0

    # fractional year
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (hour - 12.0) / 24.0)

    # solar declination (rad)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.001480 * np.sin(3 * gamma)
    )

    # equation of time (minutes)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    # true solar time (minutes)
    tst = (hour * 60.0 + eqtime + 4.0 * lon) % 1440.0

    # hour angle (rad)
    ha = np.deg2rad((tst / 4.0) - 180.0)

    # solar zenith cos
    cosz = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, -1.0, 1.0)
    zen = np.arccos(cosz)

    elev_deg = np.rad2deg(np.pi / 2.0 - zen)

    label = "day" if elev_deg > float(threshold_deg) else "night"
    return (label, float(elev_deg)) if return_elev else label


def load_stations(stations_path: str) -> pd.DataFrame:
    df = pd.read_csv(stations_path, sep="\t")
    df = df.reset_index().rename(columns={"index": "idx"})

    for col in list(df.columns):
        low = col.lower()
        if low.startswith("station"):
            df = df.rename(columns={col: "Station_Name"})
        elif low.startswith("lat"):
            df = df.rename(columns={col: "Latitude"})
        elif low.startswith("lon"):
            df = df.rename(columns={col: "Longitude"})
        elif low.startswith("alt"):
            df = df.rename(columns={col: "Altitude"})

    expected = ["idx", "Station_Name", "Latitude", "Longitude", "Altitude"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Stations file missing expected columns: {missing}")

    for col in ["Latitude", "Longitude", "Altitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_valid"] = df[["Latitude", "Longitude", "Altitude"]].notna().all(axis=1)
    return df[expected + ["is_valid"]]


def get_target_stations(
    stations_df: pd.DataFrame,
    selection: str = "single",
    idx: Optional[int] = None,
    idxs: Optional[List[int]] = None,
) -> pd.DataFrame:
    valid = stations_df[stations_df["is_valid"]].copy()

    if selection == "single":
        if idx is None:
            raise ValueError("For selection='single', idx must be provided.")
        out = valid[valid["idx"] == int(idx)].copy()
        if out.empty:
            raise ValueError(f"Station idx={idx} not found or invalid.")
        return out.reset_index(drop=True)

    if selection == "list":
        if not idxs:
            raise ValueError("For selection='list', idxs must be a non-empty list.")
        out = valid[valid["idx"].isin([int(x) for x in idxs])].copy()
        if out.empty:
            raise ValueError("No valid stations found for the requested station list.")
        return out.sort_values("idx").reset_index(drop=True)

    if selection == "all":
        return valid.sort_values("idx").reset_index(drop=True)

    raise ValueError("selection must be one of: 'single', 'list', 'all'")


def load_horizontal_lookup(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Horizontal lookup not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported horizontal lookup format: {path.suffix}")

    required = {"idx", "i", "j", "model_lat", "model_lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Horizontal lookup missing columns: {sorted(missing)}")

    df["idx"] = df["idx"].astype(int)
    df["i"] = df["i"].astype(int)
    df["j"] = df["j"].astype(int)
    return df


def load_constant_k_lookup(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Constant-k lookup not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = {"idx", "level_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Constant-k lookup missing columns: {sorted(missing)}")

    df["idx"] = df["idx"].astype(int)
    df["level_idx"] = df["level_idx"].astype(int)
    return df


def load_level_lookup(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Timestep lookup not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported timestep lookup format: {path.suffix}")

    required = {"idx", "time", "level_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Timestep lookup missing columns: {sorted(missing)}")

    df["idx"] = df["idx"].astype(int)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["level_idx"] = df["level_idx"].astype(int)
    return df


# ============================================================
# VARIABLE / DIMENSION RESOLUTION + GEOMETRY DETECTION
# ============================================================

def get_species_var_name(ds: xr.Dataset, species: str) -> str:
    """
    Resolve the main data variable.
    Priority:
      1) exact / case variants of `species`
      2) if only one data var, use it
      3) the data var that has both lat and lon dims with the most dims
         (covers AOT files whose variable is TOTEXTTAU, not 'AOT')
    """
    candidates = [species, species.upper(), species.lower()]
    for c in candidates:
        if c in ds.data_vars:
            return c

    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        return data_vars[0]

    def has_latlon(v):
        dl = [d.lower() for d in ds[v].dims]
        return any(d in {"lat", "latitude"} for d in dl) and any(d in {"lon", "longitude"} for d in dl)

    latlon_vars = [v for v in data_vars if has_latlon(v)]
    if latlon_vars:
        latlon_vars.sort(key=lambda v: ds[v].ndim, reverse=True)
        return latlon_vars[0]

    raise KeyError(
        f"Could not infer species variable for '{species}'. "
        f"Available variables: {data_vars}"
    )


def resolve_dims(da: xr.DataArray) -> dict:
    """Return {'time','lat','lon','vert'} dim names; vert=None means columnar (2D)."""
    dims = list(da.dims)
    time_dim = next((d for d in dims if d.lower() == "time"), None)
    lat_dim = next((d for d in dims if d.lower() in {"lat", "latitude"}), None)
    lon_dim = next((d for d in dims if d.lower() in {"lon", "longitude"}), None)
    vert_dim = next((d for d in dims if d.lower() in VERT_DIM_CANDIDATES), None)

    if lat_dim is None or lon_dim is None:
        raise ValueError(f"Could not identify lat/lon dims. Found dims: {dims}")

    return {"time": time_dim, "lat": lat_dim, "lon": lon_dim, "vert": vert_dim}


def load_grid_info_once(base_path, product, species, start_dt, end_dt, step_minutes):
    """
    Open the first available file to grab the grid AND detect geometry
    (variable name, columnar vs 3D, and dim names). Cached and reused.
    """
    for d0, t0 in iter_timestamps(start_dt, end_dt, step_minutes):
        spf, _, _, _, _ = build_paths(base_path, product, species, d0, t0)
        if not spf.exists():
            continue
        try:
            ds = xr.open_dataset(spf, decode_times=False)
            var = get_species_var_name(ds, species)
            dim_info = resolve_dims(ds[var])
            is_columnar = dim_info["vert"] is None
            lats = ds[dim_info["lat"]].values
            lons = ds[dim_info["lon"]].values
            ds.close()
            print(
                f"[grid-info] var='{var}' "
                f"{'COLUMNAR (2D, horizontal-only)' if is_columnar else '3D (vertical level used)'} "
                f"dims={dim_info}"
            )
            return {
                "lats": lats,
                "lons": lons,
                "first_found": (d0, t0),
                "var": var,
                "is_columnar": is_columnar,
                "dim_info": dim_info,
            }
        except Exception:
            continue

    raise FileNotFoundError("No species files found in requested period.")


# ============================================================
# GEOMETRY / MASKS / STATS
# ============================================================

def make_small_box_indices(i: int, j: int, Ny: int, Nx: int, cell_nums: int):
    i1 = max(0, i - cell_nums)
    i2 = min(Ny - 1, i + cell_nums)
    j1 = max(0, j - cell_nums)
    j2 = min(Nx - 1, j + cell_nums)
    ii = i - i1
    jj = j - j1
    return i1, i2, j1, j2, ii, jj


def cumulative_square_masks(ii: int, jj: int, Ny_s: int, Nx_s: int, cell_nums: int):
    # NOTE: center cell (C0) is intentionally INCLUDED in every mask (unchanged).
    yy, xx = np.indices((Ny_s, Nx_s))
    masks = []
    for k in range(1, cell_nums + 1):
        m = (np.abs(yy - ii) <= k) & (np.abs(xx - jj) <= k)
        masks.append(m)
    return masks


def compute_w_area_small(lats_small, lons_small, earth_radius_km=EARTH_RADIUS_KM):
    lats_small = np.asarray(lats_small, dtype=float)
    lons_small = np.asarray(lons_small, dtype=float)

    if lats_small.ndim != 1 or lons_small.ndim != 1:
        raise ValueError("compute_w_area_small expects 1D lats_small and lons_small")

    if len(lats_small) < 2 or len(lons_small) < 2:
        return np.ones((len(lats_small), len(lons_small)), dtype=float)

    # approximate cell-area weights on sphere
    dlat = np.abs(np.diff(lats_small).mean())
    dlon = np.abs(np.diff(lons_small).mean())
    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)

    lat_rad = np.deg2rad(lats_small)
    weights_lat = np.cos(lat_rad) * dlat_rad * dlon_rad * (earth_radius_km ** 2)

    w = np.repeat(weights_lat[:, None], len(lons_small), axis=1)
    w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
    return w


def weighted_quantile(x, w, q):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[m]
    w = w[m]
    if x.size == 0:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    w = w[order]

    cw = np.cumsum(w)
    cw = cw / cw[-1]

    return float(np.interp(q, cw, x))


def stats_unw(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "n": 0, "mean": np.nan, "std": np.nan, "cv": np.nan,
            "median": np.nan, "q25": np.nan, "q75": np.nan, "iqr": np.nan
        }
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=0))
    cv = float(std / mean) if mean != 0 else np.nan
    q25 = float(np.quantile(vals, 0.25))
    med = float(np.quantile(vals, 0.50))
    q75 = float(np.quantile(vals, 0.75))
    return {
        "n": int(vals.size), "mean": mean, "std": std, "cv": cv,
        "median": med, "q25": q25, "q75": q75, "iqr": float(q75 - q25)
    }


def stats_w(vals, w):
    vals = np.asarray(vals, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    vals = vals[m]
    w = w[m]
    if vals.size == 0:
        return {
            "n_w": 0, "mean_w": np.nan, "std_w": np.nan, "cv_w": np.nan,
            "median_w": np.nan, "q1_w": np.nan, "q3_w": np.nan, "iqr_w": np.nan
        }

    wsum = float(np.sum(w))
    mean = float(np.sum(w * vals) / wsum)
    var = float(np.sum(w * (vals - mean) ** 2) / wsum)
    std = float(np.sqrt(var))
    cv = float(std / mean) if mean != 0 else np.nan
    q1 = float(weighted_quantile(vals, w, 0.25))
    med = float(weighted_quantile(vals, w, 0.50))
    q3 = float(weighted_quantile(vals, w, 0.75))

    return {
        "n_w": int(vals.size), "mean_w": mean, "std_w": std, "cv_w": cv,
        "median_w": med, "q1_w": q1, "q3_w": q3, "iqr_w": float(q3 - q1)
    }


# ============================================================
# SPECIES EXTRACTION / CONVERSION
# ============================================================

def to_ppb_mmr(data_arr, species):
    """
    Convert mass mixing ratio (kg/kg) to ppb using MW_air/MW_species * 1e9.
    Extend MW_map if you add gas species. Columnar fields (e.g. AOT) are
    dimensionless and bypass this entirely.
    """
    MW_air = 28.9647
    MW_map = {"O3": 48.0, "SO2": 64.07}
    if species not in MW_map:
        raise ValueError(f"No MW defined for species={species}. Add it to MW_map.")
    return np.asarray(data_arr, dtype=float) * (MW_air / MW_map[species]) * 1e9


def read_full_plane(ds_species, var, dim_info, is_columnar, k_star):
    """
    Read ONE full horizontal plane (lat,lon) for the given level.
    For columnar fields k_star is ignored. Used by the fast timestep-first engine.
    Returns a 2D ndarray ordered (lat, lon).
    """
    sel = {}
    if dim_info["time"] is not None:
        sel[dim_info["time"]] = 0
    if not is_columnar:
        sel[dim_info["vert"]] = int(k_star)
    da = ds_species[var].isel(sel).transpose(dim_info["lat"], dim_info["lon"])
    return da.values


def extract_box_given_k(ds_species, var, dim_info, is_columnar, k_star, i1_s, i2_s, j1_s, j2_s):
    """
    Read a single station's box directly (box-level isel). Used by the OLD
    (reference) engine. Returns a 2D ndarray ordered (lat, lon), NOT converted.
    """
    sel = {
        dim_info["lat"]: slice(i1_s, i2_s + 1),
        dim_info["lon"]: slice(j1_s, j2_s + 1),
    }
    if dim_info["time"] is not None:
        sel[dim_info["time"]] = 0
    if not is_columnar:
        sel[dim_info["vert"]] = int(k_star)
    da = ds_species[var].isel(sel).transpose(dim_info["lat"], dim_info["lon"])
    return da.values


# ============================================================
# LOOKUP PREPARATION
# ============================================================

def prepare_station_configs(
    target_stations: pd.DataFrame,
    hlookup_df: pd.DataFrame,
    level_lookup_df: Optional[pd.DataFrame],
    constant_k_df: Optional[pd.DataFrame],
    variable_k_stations: List[int],
    use_hybrid_k_lookup: bool,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    weighted: bool,
    cell_nums: int,
    grid_info: dict,
    is_columnar: bool,
):
    lats = grid_info["lats"]
    lons = grid_info["lons"]
    Ny, Nx = (lats.shape[0], lons.shape[0]) if np.ndim(lats) == 1 else lats.shape

    variable_k_stations = set(int(x) for x in variable_k_stations)

    hmap = {int(r["idx"]): r.to_dict() for _, r in hlookup_df.iterrows()}
    cmap = {}
    if (not is_columnar) and (constant_k_df is not None):
        cmap = {int(r["idx"]): r.to_dict() for _, r in constant_k_df.iterrows()}

    # Prepare timestep lookup maps only for 3D stations that need them
    timestep_lookup_maps: Dict[int, Dict[dt.datetime, dict]] = {}
    if (not is_columnar) and (level_lookup_df is not None):
        timestep_station_ids = []
        for _, st in target_stations.iterrows():
            st_idx = int(st["idx"])
            if use_hybrid_k_lookup and (st_idx not in variable_k_stations):
                continue
            timestep_station_ids.append(st_idx)

        if timestep_station_ids:
            sub = level_lookup_df[level_lookup_df["idx"].isin(timestep_station_ids)].copy()
            sub = sub[
                (sub["time"] >= pd.Timestamp(start_dt)) &
                (sub["time"] <= pd.Timestamp(end_dt))
            ].copy()

            for st_idx, grp in sub.groupby("idx"):
                timestep_lookup_maps[int(st_idx)] = {
                    pd.Timestamp(r["time"]).to_pydatetime(): r.to_dict()
                    for _, r in grp.iterrows()
                }

    station_cfgs = {}
    n_skipped = 0

    for _, st in target_stations.iterrows():
        st_idx = int(st["idx"])
        name = st["Station_Name"]
        lat_s = float(st["Latitude"])
        lon_s = float(st["Longitude"])
        alt_s = float(st["Altitude"])

        hrow = hmap.get(st_idx)
        if hrow is None:
            print(f"[skip-config] No horizontal lookup row for station {name} (idx={st_idx}).")
            n_skipped += 1
            continue

        i = int(hrow["i"])
        j = int(hrow["j"])
        model_lat = float(hrow["model_lat"])
        model_lon = float(hrow["model_lon"])

        if is_columnar:
            # horizontal-only: no vertical lookup needed
            lookup_mode = "columnar"
            constant_k_row = None
            lookup_map = None
        elif use_hybrid_k_lookup and (st_idx not in variable_k_stations):
            lookup_mode = "constant"
            constant_k_row = cmap.get(st_idx)
            lookup_map = None
            if constant_k_row is None:
                print(f"[skip-config] No constant-k row for station {name} (idx={st_idx}).")
                n_skipped += 1
                continue
        else:
            lookup_mode = "timestep"
            constant_k_row = None
            lookup_map = timestep_lookup_maps.get(st_idx)
            if not lookup_map:
                print(f"[skip-config] No timestep-k rows for station {name} (idx={st_idx}).")
                n_skipped += 1
                continue

        i1_s, i2_s, j1_s, j2_s, ii, jj = make_small_box_indices(i, j, Ny, Nx, cell_nums)

        lats_small = lats[i1_s:i2_s + 1]
        lons_small = lons[j1_s:j2_s + 1]
        Ny_s, Nx_s = len(lats_small), len(lons_small)

        w_area_small = compute_w_area_small(lats_small, lons_small) if weighted else None
        cum_masks = cumulative_square_masks(ii, jj, Ny_s, Nx_s, cell_nums)

        station_cfgs[st_idx] = {
            "station": st,
            "station_idx": st_idx,
            "name": name,
            "lat_s": lat_s,
            "lon_s": lon_s,
            "alt_s": alt_s,
            "i": i,
            "j": j,
            "model_lat": model_lat,
            "model_lon": model_lon,
            "i1_s": i1_s,
            "i2_s": i2_s,
            "j1_s": j1_s,
            "j2_s": j2_s,
            "ii": ii,
            "jj": jj,
            "w_area_small": w_area_small,
            "cum_masks": cum_masks,
            "radii": list(range(1, cell_nums + 1)),
            "lookup_mode": lookup_mode,
            "constant_k_row": constant_k_row,
            "lookup_map": lookup_map,
            "rows": [],
        }

    print(f"[configs] prepared {len(station_cfgs)} station(s), skipped {n_skipped}.")
    return station_cfgs


# ============================================================
# ROW BUILDING (shared math, data-agnostic)
# ============================================================

def resolve_k_for_timestep(cfg: dict, ts: dt.datetime, is_columnar: bool):
    """
    Return (k_star, z_target, ok) for a station at a timestep.
    - columnar : (None, nan, True)
    - constant : (k, z, True)
    - timestep : (k, z, True) or (None, nan, False) if no row for this ts
    """
    if is_columnar or cfg["lookup_mode"] == "columnar":
        return None, np.nan, True

    if cfg["lookup_mode"] == "constant":
        row = cfg["constant_k_row"]
    else:
        row = cfg["lookup_map"].get(ts)
        if row is None:
            return None, np.nan, False

    k = int(row["level_idx"])
    z = float(row["level_height_m"]) if "level_height_m" in row else np.nan
    return k, z, True


def build_rows_from_box(
    grid_box,
    cfg: dict,
    date: str,
    time_: str,
    ts: dt.datetime,
    weighted: bool,
    k_star_t,
    z_star_t,
):
    """
    Build the C1..C10 records from an already-read (and already-converted) box.
    `grid_box` is a 2D ndarray (lat,lon) in the final stored units
    (ppb for gases, raw value for columnar fields).
    k_star_t is None for columnar.
    """
    valid = np.isfinite(grid_box)
    center_val = grid_box[cfg["ii"], cfg["jj"]]
    center_ppb = float(center_val) if np.isfinite(center_val) else np.nan

    season = season_from_datetime(ts)
    dn = day_night_label(cfg["lat_s"], cfg["lon_s"], ts)

    out = []
    for k, mask_total in enumerate(cfg["cum_masks"], start=1):
        mask_total = np.asarray(mask_total, dtype=bool)
        mask_valid = mask_total & valid

        n_total = int(mask_total.sum())
        n_valid = int(mask_valid.sum())
        n_excl = int(n_total - n_valid)
        frac_excl = float(n_excl / n_total) if n_total > 0 else np.nan

        vals = grid_box[mask_valid]

        rec = {c: np.nan for c in EXPECTED_COLS}
        rec.update({
            "station": cfg["name"],
            "station_idx": cfg["station_idx"],
            "station_lat": cfg["lat_s"],
            "station_lon": cfg["lon_s"],
            "station_alt": cfg["alt_s"],
            "model_lat": cfg["model_lat"],
            "model_lon": cfg["model_lon"],
            "i_center": int(cfg["i"]),
            "j_center": int(cfg["j"]),
            "date": date,
            "time": time_,
            "timestamp": f"{date} {time_}",
            "datetime": ts,
            "season": season,
            "day_night": dn,
            "mode": "A",
            "sector_type": "CUM",
            "sector": f"C{k}",
            "radius": cfg["radii"][k - 1],
            "k_star_center": (int(k_star_t) if k_star_t is not None else np.nan),
            "z_target_m": float(z_star_t) if np.isfinite(z_star_t) else np.nan,
            "center_ppb": center_ppb,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_excluded": n_excl,
            "frac_excluded": frac_excl,
        })

        rec.update(stats_unw(vals))

        if weighted and (cfg["w_area_small"] is not None):
            w = cfg["w_area_small"][mask_valid]
            rec.update(stats_w(vals, w))
        else:
            rec.update({
                "n_w": np.nan,
                "mean_w": np.nan,
                "std_w": np.nan,
                "cv_w": np.nan,
                "median_w": np.nan,
                "q1_w": np.nan,
                "q3_w": np.nan,
                "iqr_w": np.nan,
            })

        out.append(rec)

    return out


def build_records_for_one_station_and_timestep(
    ds_species: xr.Dataset,
    cfg: dict,
    date: str,
    time_: str,
    ts: dt.datetime,
    weighted: bool,
    var: str,
    dim_info: dict,
    is_columnar: bool,
):
    """Reference path (OLD engine): read this station's box directly, then build rows."""
    k_star_t, z_star_t, ok = resolve_k_for_timestep(cfg, ts, is_columnar)
    if not ok:
        return []

    try:
        grid_box = extract_box_given_k(
            ds_species, var, dim_info, is_columnar, k_star_t,
            cfg["i1_s"], cfg["i2_s"], cfg["j1_s"], cfg["j2_s"],
        )
    except Exception as e:
        print(
            f"[extract-error] station={cfg['name']} idx={cfg['station_idx']} "
            f"date={date} time={time_} k={k_star_t} error={e}"
        )
        return []

    if not is_columnar:
        grid_box = to_ppb_mmr(grid_box, species)

    return build_rows_from_box(grid_box, cfg, date, time_, ts, weighted, k_star_t, z_star_t)


# ============================================================
# SUMMARY / SAVE
# ============================================================

def build_summary(df_per_timestep: pd.DataFrame) -> pd.DataFrame:
    if df_per_timestep.empty:
        return pd.DataFrame()

    stat_cols = [
        "n", "mean", "std", "cv", "median", "q25", "q75", "iqr",
        "n_w", "mean_w", "std_w", "cv_w", "median_w", "q1_w", "q3_w", "iqr_w"
    ]
    stat_cols = [c for c in stat_cols if c in df_per_timestep.columns]

    summary = (
        df_per_timestep
        .groupby(["station", "mode", "sector_type", "sector"], as_index=False)[stat_cols]
        .agg(["mean", "std", "median"])
    )
    summary.columns = [f"{a}_{b}" if b else a for (a, b) in summary.columns.to_flat_index()]
    return summary


def save_station_outputs(
    station_name: str,
    species: str,
    mode: str,
    df_30min: pd.DataFrame,
    df_summary: pd.DataFrame,
    out_dir: str,
    save_csv: bool,
    save_parquet: bool,
):
    base = f"{out_dir}/{station_name}_{species}_{mode}"

    if save_csv:
        df_30min.to_csv(f"{base}_30min.csv", index=False)
        df_summary.to_csv(f"{base}_summary.csv", index=False)

    if save_parquet:
        df_30min.to_parquet(f"{base}_30min.parquet", index=False)
        df_summary.to_parquet(f"{base}_summary.parquet", index=False)


# ============================================================
# BASELINE: STATION-FIRST  (reference engine)
# ============================================================

def run_time_interval(
    mode: str = "A",
    weighted: bool = True,
    start_dt: Optional[dt.datetime] = None,
    end_dt: Optional[dt.datetime] = None,
    step_minutes: int = 30,
    station_selection: str = "single",
    station_idx: Optional[int] = None,
    station_idxs: Optional[List[int]] = None,
    save_csv: bool = True,
    save_parquet: bool = True,
) -> Dict[int, dict]:
    """
    Baseline station-first implementation.
    Keeps the scientific logic simple and is used as the reference version.
    """
    if mode.upper() != "A":
        raise NotImplementedError("This compact script implements only MODE='A'.")

    if start_dt is None:
        start_dt = START_DT
    if end_dt is None:
        end_dt = END_DT

    os.makedirs(out_dir, exist_ok=True)

    stations_df = load_stations(stations_path)
    target_stations = get_target_stations(stations_df, station_selection, station_idx, station_idxs)

    hlookup_df = load_horizontal_lookup(HORIZONTAL_LOOKUP_PATH)
    grid_info = load_grid_info_once(base_path, product, species, start_dt, end_dt, step_minutes)
    is_columnar = grid_info["is_columnar"]
    var = grid_info["var"]
    dim_info = grid_info["dim_info"]

    if is_columnar:
        level_lookup_df = None
        constant_k_df = None
    else:
        level_lookup_df = load_level_lookup(LEVEL_LOOKUP_PATH)
        constant_k_df = load_constant_k_lookup(CONSTANT_K_LOOKUP_PATH) if USE_HYBRID_K_LOOKUP else None

    station_cfgs = prepare_station_configs(
        target_stations=target_stations,
        hlookup_df=hlookup_df,
        level_lookup_df=level_lookup_df,
        constant_k_df=constant_k_df,
        variable_k_stations=VARIABLE_K_STATIONS,
        use_hybrid_k_lookup=USE_HYBRID_K_LOOKUP,
        start_dt=start_dt,
        end_dt=end_dt,
        weighted=weighted,
        cell_nums=CELL_NUMS,
        grid_info=grid_info,
        is_columnar=is_columnar,
    )

    if SKIP_EXISTING and save_parquet:
        station_cfgs = _filter_existing(station_cfgs, species, mode, out_dir)

    results = {}
    for st_idx, cfg in station_cfgs.items():
        print(f"\n[OLD] Running station {cfg['name']} (idx={st_idx})")
        rows = []

        for date, time_ in iter_timestamps(start_dt, end_dt, step_minutes):
            ts = dt.datetime.strptime(date + time_, "%Y%m%d%H%M")
            spf, _, _, _, _ = build_paths(base_path, product, species, date, time_)
            if not spf.exists():
                continue

            ds_species = None
            try:
                ds_species = xr.open_dataset(spf, decode_times=False)
                rows.extend(build_records_for_one_station_and_timestep(
                    ds_species=ds_species,
                    cfg=cfg,
                    date=date,
                    time_=time_,
                    ts=ts,
                    weighted=weighted,
                    var=var,
                    dim_info=dim_info,
                    is_columnar=is_columnar,
                ))
            finally:
                if ds_species is not None:
                    try:
                        ds_species.close()
                    except Exception:
                        pass

        df_30min = pd.DataFrame.from_records(rows, columns=EXPECTED_COLS)
        if df_30min.empty:
            continue

        df_summary = build_summary(df_30min)

        save_station_outputs(
            station_name=cfg["name"],
            species=species,
            mode=mode,
            df_30min=df_30min,
            df_summary=df_summary,
            out_dir=out_dir,
            save_csv=save_csv,
            save_parquet=save_parquet,
        )

        results[st_idx] = {
            "station_name": cfg["name"],
            "df_30min": df_30min,
            "df_summary": df_summary,
        }

    return results


# ============================================================
# STREAMING WRITE HELPER
# ============================================================

def _parquet_is_complete(path: str) -> bool:
    """True if `path` exists and opens as a valid parquet (footer readable)."""
    if not os.path.exists(path):
        return False
    try:
        pf = pq.ParquetFile(path)
        _ = pf.metadata.num_rows   # touching metadata forces a footer read
        return True
    except Exception:
        return False


def _filter_existing(station_cfgs: dict, species: str, mode: str, out_dir: str) -> dict:
    """Drop stations whose _30min.parquet is already complete (resume support)."""
    kept, skipped = {}, 0
    for st_idx, cfg in station_cfgs.items():
        path = f"{out_dir}/{cfg['name']}_{species}_{mode}_30min.parquet"
        if _parquet_is_complete(path):
            skipped += 1
        else:
            kept[st_idx] = cfg
    if skipped:
        print(f"[skip-existing] skipped {skipped} station(s) with valid parquet; "
              f"{len(kept)} to run.")
    return kept


def _flush_station_buffer(cfg: dict, species: str) -> int:
    """
    Write the station's currently buffered rows as one parquet row-group chunk,
    then clear the buffer. Creates the per-station writer on first call and keeps
    it open for subsequent chunks. Schema is fixed from the first chunk (dtypes
    are stable within a run), so later chunks are cast to it. Returns rows written.
    """
    rows = cfg.get("rows")
    if not rows:
        return 0

    df = pd.DataFrame.from_records(rows, columns=EXPECTED_COLS)
    cfg["rows"] = []
    n = len(df)

    table = pa.Table.from_pandas(df, preserve_index=False)
    if cfg.get("writer") is None:
        cfg["schema"] = table.schema
        cfg["writer"] = pq.ParquetWriter(cfg["out_path"], cfg["schema"])
    else:
        table = table.cast(cfg["schema"], safe=False)

    cfg["writer"].write_table(table)
    cfg["n_written"] = cfg.get("n_written", 0) + n
    return n


# ============================================================
# FASTER VERSION: TIMESTEP-FIRST MULTI-STATION
#   - reads each distinct level-plane ONCE per file, slices boxes in memory
#   - columnar fields => single plane per file
#   - streaming per-station parquet with bounded memory (production)
# ============================================================

def run_time_interval_timestep_first(
    mode: str = "A",
    weighted: bool = True,
    start_dt: Optional[dt.datetime] = None,
    end_dt: Optional[dt.datetime] = None,
    step_minutes: int = 30,
    station_selection: str = "single",
    station_idx: Optional[int] = None,
    station_idxs: Optional[List[int]] = None,
    save_csv: bool = True,
    save_parquet: bool = True,
    stream: bool = True,
) -> Dict[int, dict]:
    """
    Faster timestep-first multi-station implementation.
    Opens each timestep file once and processes all selected stations from it,
    reading each needed horizontal plane a single time.

    When stream=True AND save_parquet=True (production), each station's parquet is
    written incrementally in row-group chunks and all buffers are flushed whenever
    MAX_BUFFER_ROWS rows are buffered across stations, so peak RAM stays bounded
    regardless of station count or period length. The per-station summary is built
    at the end by reading each finished file back (one station at a time). Output
    numbers and files are identical to the non-streaming path.

    When stream=False or save_parquet=False (e.g. the old-vs-new harness), rows are
    accumulated in memory and the station DataFrames are returned for comparison.
    """
    if mode.upper() != "A":
        raise NotImplementedError("This compact script implements only MODE='A'.")

    if start_dt is None:
        start_dt = START_DT
    if end_dt is None:
        end_dt = END_DT

    os.makedirs(out_dir, exist_ok=True)

    stations_df = load_stations(stations_path)
    target_stations = get_target_stations(stations_df, station_selection, station_idx, station_idxs)

    hlookup_df = load_horizontal_lookup(HORIZONTAL_LOOKUP_PATH)
    grid_info = load_grid_info_once(base_path, product, species, start_dt, end_dt, step_minutes)
    is_columnar = grid_info["is_columnar"]
    var = grid_info["var"]
    dim_info = grid_info["dim_info"]

    if is_columnar:
        level_lookup_df = None
        constant_k_df = None
    else:
        level_lookup_df = load_level_lookup(LEVEL_LOOKUP_PATH)
        constant_k_df = load_constant_k_lookup(CONSTANT_K_LOOKUP_PATH) if USE_HYBRID_K_LOOKUP else None

    station_cfgs = prepare_station_configs(
        target_stations=target_stations,
        hlookup_df=hlookup_df,
        level_lookup_df=level_lookup_df,
        constant_k_df=constant_k_df,
        variable_k_stations=VARIABLE_K_STATIONS,
        use_hybrid_k_lookup=USE_HYBRID_K_LOOKUP,
        start_dt=start_dt,
        end_dt=end_dt,
        weighted=weighted,
        cell_nums=CELL_NUMS,
        grid_info=grid_info,
        is_columnar=is_columnar,
    )

    if SKIP_EXISTING and save_parquet:
        station_cfgs = _filter_existing(station_cfgs, species, mode, out_dir)
        if not station_cfgs:
            print("[skip-existing] nothing to do — all selected stations already complete.")
            return {}

    use_stream = bool(stream and save_parquet and STREAMING_FLUSH)
    if use_stream:
        for cfg in station_cfgs.values():
            cfg["writer"] = None
            cfg["schema"] = None
            cfg["n_written"] = 0
            cfg["out_path"] = f"{out_dir}/{cfg['name']}_{species}_{mode}_30min.parquet"
        print(f"[new-engine] STREAMING: 1 parquet/station, flush at "
              f"{MAX_BUFFER_ROWS:,} buffered rows, {len(station_cfgs)} open writer(s).")

    all_ts = list(iter_timestamps(start_dt, end_dt, step_minutes))
    total_ts = len(all_ts)
    t_start = time.time()
    n_files = 0
    n_station_ts_skipped = 0   # 3D stations with no level row at a timestep
    buffered = 0
    since_flush = 0            # processed timesteps since last flush

    use_bar = bool(SHOW_PROGRESS_BAR and _HAVE_TQDM)
    bar = tqdm(total=total_ts, desc=f"{species} {len(station_cfgs)}st",
               unit="ts", dynamic_ncols=True) if use_bar else None

    def _log(msg):
        if bar is not None:
            bar.write(msg)
        else:
            print(msg)

    def _progress(tag):
        el = time.time() - t_start
        rate = n_files / el if el > 0 else 0.0
        pct = (100.0 * n_files / total_ts) if total_ts else 0.0
        _log(f"[progress] {tag} files={n_files}/{total_ts} ({pct:.1f}%) "
             f"elapsed={el/60:.1f}min rate={rate:.1f} files/s buffered={buffered:,}")

    def _flush_all(reason):
        nonlocal buffered, since_flush
        if not use_stream:
            return
        wrote = 0
        for cfg in station_cfgs.values():
            wrote += _flush_station_buffer(cfg, species)
        _log(f"[flush:{reason}] wrote {wrote:,} rows at ts {n_files}/{total_ts} "
             f"(elapsed {(time.time() - t_start)/60:.1f} min)")
        buffered = 0
        since_flush = 0

    try:
        for (date, time_) in all_ts:
            ts = dt.datetime.strptime(date + time_, "%Y%m%d%H%M")
            spf, _, _, _, _ = build_paths(base_path, product, species, date, time_)

            if spf.exists():
                ds_species = None
                try:
                    ds_species = xr.open_dataset(spf, decode_times=False)
                    n_files += 1
                    since_flush += 1

                    # 1) resolve each station's (k, z) for this timestep
                    resolved: Dict[int, Tuple[Optional[int], float]] = {}
                    for st_idx, cfg in station_cfgs.items():
                        k_star_t, z_star_t, ok = resolve_k_for_timestep(cfg, ts, is_columnar)
                        if not ok:
                            n_station_ts_skipped += 1
                            continue
                        resolved[st_idx] = (k_star_t, z_star_t)

                    if resolved:
                        # 2) read each distinct plane once
                        if is_columnar:
                            planes = {None: read_full_plane(ds_species, var, dim_info, True, None)}
                        else:
                            planes = {}
                            for (k_star_t, _z) in resolved.values():
                                if k_star_t not in planes:
                                    planes[k_star_t] = read_full_plane(
                                        ds_species, var, dim_info, False, k_star_t
                                    )

                        # 3) per station: slice box from its plane, convert, build rows
                        for st_idx, (k_star_t, z_star_t) in resolved.items():
                            cfg = station_cfgs[st_idx]
                            plane_key = None if is_columnar else k_star_t
                            plane = planes[plane_key]
                            grid_box = plane[cfg["i1_s"]:cfg["i2_s"] + 1, cfg["j1_s"]:cfg["j2_s"] + 1]

                            if not is_columnar:
                                grid_box = to_ppb_mmr(grid_box, species)

                            recs = build_rows_from_box(
                                grid_box, cfg, date, time_, ts, weighted, k_star_t, z_star_t
                            )
                            if recs:
                                cfg["rows"].extend(recs)
                                buffered += len(recs)
                finally:
                    if ds_species is not None:
                        try:
                            ds_species.close()
                        except Exception:
                            pass

                # flush on timestep cadence, or earlier if the safety cap is hit
                if use_stream and (since_flush >= FLUSH_EVERY_TIMESTEPS or buffered >= MAX_BUFFER_ROWS):
                    _flush_all("cap" if buffered >= MAX_BUFFER_ROWS else f"{FLUSH_EVERY_TIMESTEPS}ts")

                if (not use_bar) and (n_files % PROGRESS_EVERY_FILES == 0) and n_files > 0:
                    _progress("tick")

            if bar is not None:
                bar.update(1)
                bar.set_postfix(files=n_files, buf=f"{buffered:,}", refresh=False)
    finally:
        if bar is not None:
            bar.close()
        if use_stream:
            # final flush + close all writers (also runs on error/interrupt)
            for cfg in station_cfgs.values():
                try:
                    _flush_station_buffer(cfg, species)
                finally:
                    if cfg.get("writer") is not None:
                        try:
                            cfg["writer"].close()
                        except Exception:
                            pass

    print(f"[new-engine] processed {n_files} files; "
          f"station-timestep skips (no level row): {n_station_ts_skipped}")

    results = {}

    if use_stream:
        # summary per station from the finished parquet (bounded: one station at a time)
        for st_idx, cfg in station_cfgs.items():
            path = cfg["out_path"]
            if cfg.get("n_written", 0) == 0 or not os.path.exists(path):
                continue
            df_full = pd.read_parquet(path)
            df_summary = build_summary(df_full)
            base = f"{out_dir}/{cfg['name']}_{species}_{mode}"
            df_summary.to_parquet(f"{base}_summary.parquet", index=False)
            if save_csv:
                df_full.to_csv(f"{base}_30min.csv", index=False)
                df_summary.to_csv(f"{base}_summary.csv", index=False)
            results[st_idx] = {
                "station_name": cfg["name"],
                "n_rows": int(cfg["n_written"]),
                "path": path,
            }
            del df_full
        return results

    # non-streaming (in-memory) path: accumulate and return DataFrames
    for st_idx, cfg in station_cfgs.items():
        df_30min = pd.DataFrame.from_records(cfg["rows"], columns=EXPECTED_COLS)
        if df_30min.empty:
            continue

        df_summary = build_summary(df_30min)

        save_station_outputs(
            station_name=cfg["name"],
            species=species,
            mode=mode,
            df_30min=df_30min,
            df_summary=df_summary,
            out_dir=out_dir,
            save_csv=save_csv,
            save_parquet=save_parquet,
        )

        results[st_idx] = {
            "station_name": cfg["name"],
            "df_30min": df_30min,
            "df_summary": df_summary,
        }

    return results


# ============================================================
# OLD VS NEW COMPARISON
# ============================================================

def compare_old_vs_new_results(
    results_old: Dict[int, dict],
    results_new: Dict[int, dict],
    rtol: float = 1e-10,
    atol: float = 1e-12,
    check_summary: bool = True,
    verbose: bool = True,
):
    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = sorted(df.columns.tolist())
        df = df[cols]

        for c in df.columns:
            if c == "datetime" or c == "time":
                try:
                    df[c] = pd.to_datetime(df[c], errors="ignore")
                except Exception:
                    pass

        sort_cols = [c for c in ["station", "station_idx", "date", "time", "sector_type", "sector", "radius", "datetime"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        return df

    def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, name: str):
        rep = {
            "match": True,
            "shape_equal": True,
            "columns_equal": True,
            "different_columns": [],
            "different_numeric_columns": [],
            "different_non_numeric_columns": [],
            "message": "",
        }

        d1 = normalize_df(df1)
        d2 = normalize_df(df2)

        if list(d1.columns) != list(d2.columns):
            rep["match"] = False
            rep["columns_equal"] = False
            rep["different_columns"] = sorted(list(set(d1.columns).symmetric_difference(set(d2.columns))))
            rep["message"] = f"{name}: columns differ"
            return rep

        if d1.shape != d2.shape:
            rep["match"] = False
            rep["shape_equal"] = False
            rep["message"] = f"{name}: shape differs {d1.shape} vs {d2.shape}"
            return rep

        for c in d1.columns:
            s1 = d1[c]
            s2 = d2[c]

            if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
                ok = np.isclose(s1.to_numpy(), s2.to_numpy(), rtol=rtol, atol=atol, equal_nan=True)
                if not np.all(ok):
                    rep["match"] = False
                    rep["different_numeric_columns"].append(c)
            else:
                ok = (s1.fillna("__NA__").astype(str).to_numpy() ==
                      s2.fillna("__NA__").astype(str).to_numpy())
                if not np.all(ok):
                    rep["match"] = False
                    rep["different_non_numeric_columns"].append(c)

        rep["message"] = f"{name}: match" if rep["match"] else f"{name}: differences found"
        return rep

    report = {}

    all_ids = sorted(set(results_old.keys()) | set(results_new.keys()))
    for st_idx in all_ids:
        item = {
            "exists_in_old": st_idx in results_old,
            "exists_in_new": st_idx in results_new,
            "df_30min": None,
            "df_summary": None,
            "match": True,
        }

        if st_idx not in results_old or st_idx not in results_new:
            item["match"] = False
            report[st_idx] = item
            continue

        rep_30 = compare_dataframes(results_old[st_idx]["df_30min"], results_new[st_idx]["df_30min"], "df_30min")
        item["df_30min"] = rep_30
        item["match"] = item["match"] and rep_30["match"]

        if check_summary:
            rep_sum = compare_dataframes(results_old[st_idx]["df_summary"], results_new[st_idx]["df_summary"], "df_summary")
            item["df_summary"] = rep_sum
            item["match"] = item["match"] and rep_sum["match"]

        report[st_idx] = item

    if verbose:
        print("\n===== OLD VS NEW COMPARISON =====")
        for st_idx, rep in report.items():
            print(f"\nStation idx={st_idx}")
            print(f"  exists_in_old: {rep['exists_in_old']}")
            print(f"  exists_in_new: {rep['exists_in_new']}")
            print(f"  overall_match: {rep['match']}")
            if rep["df_30min"] is not None:
                print(f"  df_30min: {rep['df_30min']['message']}")
                if rep["df_30min"]["different_numeric_columns"]:
                    print("    numeric diffs:", rep["df_30min"]["different_numeric_columns"])
                if rep["df_30min"]["different_non_numeric_columns"]:
                    print("    non-numeric diffs:", rep["df_30min"]["different_non_numeric_columns"])
            if rep["df_summary"] is not None:
                print(f"  df_summary: {rep['df_summary']['message']}")
                if rep["df_summary"]["different_numeric_columns"]:
                    print("    numeric diffs:", rep["df_summary"]["different_numeric_columns"])
                if rep["df_summary"]["different_non_numeric_columns"]:
                    print("    non-numeric diffs:", rep["df_summary"]["different_non_numeric_columns"])

    return report


def comparison_all_match(report: dict) -> bool:
    return all(rep.get("match", False) for rep in report.values())


def test_old_vs_new_on_small_period(
    test_start_dt: dt.datetime,
    test_end_dt: dt.datetime,
    test_station_selection: str = "single",
    test_station_idx: Optional[int] = None,
    test_station_idxs: Optional[List[int]] = None,
    weighted: bool = True,
):
    print("\nRunning OLD station-first implementation...")
    t0 = time.time()
    results_old = run_time_interval(
        mode="A",
        weighted=weighted,
        start_dt=test_start_dt,
        end_dt=test_end_dt,
        step_minutes=STEP_MINUTES,
        station_selection=test_station_selection,
        station_idx=test_station_idx,
        station_idxs=test_station_idxs,
        save_csv=False,
        save_parquet=False,
    )
    t1 = time.time()

    print("\nRunning NEW timestep-first implementation...")
    results_new = run_time_interval_timestep_first(
        mode="A",
        weighted=weighted,
        start_dt=test_start_dt,
        end_dt=test_end_dt,
        step_minutes=STEP_MINUTES,
        station_selection=test_station_selection,
        station_idx=test_station_idx,
        station_idxs=test_station_idxs,
        save_csv=False,
        save_parquet=False,
    )
    t2 = time.time()

    report = compare_old_vs_new_results(
        results_old=results_old,
        results_new=results_new,
        rtol=1e-10,
        atol=1e-12,
        check_summary=True,
        verbose=True,
    )

    print("\nTiming")
    print(f"  OLD: {(t1 - t0):.2f} s")
    print(f"  NEW: {(t2 - t1):.2f} s")
    print("  FINAL RESULT:", "MATCH" if comparison_all_match(report) else "DIFFERENCES FOUND")

    return results_old, results_new, report


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(lookup_dir, exist_ok=True)

    if MODE.upper() != "A":
        raise NotImplementedError("This compact script currently supports only MODE='A'.")

    if RUN_COMPARE_OLD_NEW:
        # Use single or small list first for validation
        test_old_vs_new_on_small_period(
            test_start_dt=COMPARE_START_DT,
            test_end_dt=COMPARE_END_DT,
            test_station_selection=STATION_SELECTION,
            test_station_idx=STATION_IDX,
            test_station_idxs=STATION_IDXS,
            weighted=WEIGHTED,
        )

    if RUN_PERIOD:
        start = time.time()
        print("\n===== PRODUCTION RUN =====")
        print("Start:", dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"))

        if RUN_ENGINE == "old":
            results = run_time_interval(
                mode=MODE,
                weighted=WEIGHTED,
                start_dt=START_DT,
                end_dt=END_DT,
                step_minutes=STEP_MINUTES,
                station_selection=STATION_SELECTION,
                station_idx=STATION_IDX,
                station_idxs=STATION_IDXS,
                save_csv=SAVE_CSV,
                save_parquet=SAVE_PARQUET,
            )
        elif RUN_ENGINE == "new":
            results = run_time_interval_timestep_first(
                mode=MODE,
                weighted=WEIGHTED,
                start_dt=START_DT,
                end_dt=END_DT,
                step_minutes=STEP_MINUTES,
                station_selection=STATION_SELECTION,
                station_idx=STATION_IDX,
                station_idxs=STATION_IDXS,
                save_csv=SAVE_CSV,
                save_parquet=SAVE_PARQUET,
            )
        else:
            raise ValueError("RUN_ENGINE must be 'old' or 'new'")

        end = time.time()
        print("End:", dt.datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"))
        print(f"Execution time: {(end - start) / 60:.2f} minutes")
        print(f"Stations with output: {len(results)}")

        return results


if __name__ == "__main__":
    main()
# %%
# %%
