#THIS IS THE NEW COMPACT VERSION
#SUPORTS ONLY MODE='A' AND TIME_INTERVAL_ANALYSIS
#PRODUCES NO PLOTS.ONLY OBJECTIVE IS TO CREATE CSV/PARQUET FILES FOR EACH STATION

#%%
# ============================================================
# Compact period-only script with:
#   - baseline station-first run_time_interval
#   - faster timestep-first run_time_interval_timestep_first
#   - hybrid constant-k / timestep-k logic
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


# ============================================================
# USER SETTINGS
# ============================================================

RUN_PERIOD = True
RUN_COMPARE_OLD_NEW = False   # compare baseline vs timestep-first on a short test window
RUN_ENGINE = "new"           # "old" or "new" ,new is the compact(fast) version,old is the one used in main.py

START_DT = dt.datetime(2005, 5, 16, 0, 0)
END_DT   = dt.datetime(2007, 6, 16, 0, 0)

MODE = "A"                   # only "A" implemented in this compact script

STATION_SELECTION = "list"    # "single", "list", "all"
STATION_IDX = 14
STATION_IDXS = [i for i in range(400, 500)]

STEP_MINUTES = 30
CELL_NUMS = 10
DIST_BINS_KM = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

WEIGHTED = True
SAVE_CSV = False
SAVE_PARQUET = True

# Hybrid k lookup
USE_HYBRID_K_LOOKUP = True
VARIABLE_K_STATIONS = [474, 732, 1001, 1120, 1209, 1324, 1784]

# Paths
base_path = "/mnt/store01/agkiokas/CAMS/inst/subsets"
product = "inst3d"
species = "O3"

stations_path = "/home/agkiokas/CAMS/CHINESE_STATIONS_INFO_2015_2023.txt"
out_dir = "/mnt/store01/agkiokas/CAMS/stations_parquet/"
lookup_dir = "/home/agkiokas/CAMS/lookups/"

HORIZONTAL_LOOKUP_PATH = f"{lookup_dir}/station_horizontal_lookup_all.parquet"
LEVEL_LOOKUP_PATH = f"{lookup_dir}/station_level_timeseries_all.parquet"
CONSTANT_K_LOOKUP_PATH = f"{lookup_dir}/station_constant_k_only_lookup.csv"

# Optional quick comparison window
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


def load_grid_info_once(base_path: str, product: str, species: str, start_dt: dt.datetime, end_dt: dt.datetime, step_minutes: int):
    for d0, t0 in iter_timestamps(start_dt, end_dt, step_minutes):
        spf, _, _, _, _ = build_paths(base_path, product, species, d0, t0)
        if not spf.exists():
            continue
        try:
            ds = xr.open_dataset(spf, decode_times=False)
            lats = ds["lat"].values
            lons = ds["lon"].values
            ds.close()
            return {"lats": lats, "lons": lons, "first_found": (d0, t0)}
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
        return np.ones((len(lats_small), len(lons_small)),dtype=float)

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
# SPECIES EXTRACTION
# ============================================================

def to_ppb_mmr(data_arr, species):
    """
    Convert mass mixing ratio (kg/kg) to ppb using MW_air/MW_species * 1e9.
    Extend MW_map if you add species.
    """
    MW_air = 28.9647
    MW_map = {"O3": 48.0}
    if species not in MW_map:
        raise ValueError(f"No MW defined for species={species}. Add it to MW_map.")
    return np.asarray(data_arr, dtype=float) * (MW_air / MW_map[species]) * 1e9


def get_species_var_name(ds: xr.Dataset, species: str) -> str:
    candidates = [
        species,
        species.upper(),
        species.lower(),
    ]
    for c in candidates:
        if c in ds.data_vars:
            return c

    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]

    raise KeyError(
        f"Could not infer species variable for '{species}'. "
        f"Available variables: {list(ds.data_vars)}"
    )


def extract_smallbox_ppb_optionA_given_k(
    ds_species: xr.Dataset,
    species: str,
    k_star: int,
    i1_s: int,
    i2_s: int,
    j1_s: int,
    j2_s: int,
    to_ppb_fn=to_ppb_mmr,
):
    var = get_species_var_name(ds_species, species)
    da = ds_species[var]

    dims = list(da.dims)

    # detect dimensions safely
    time_dim = next((d for d in dims if d.lower() == "time"), None)
    lat_dim = next((d for d in dims if d.lower() in {"lat", "latitude"}), None)
    lon_dim = next((d for d in dims if d.lower() in {"lon", "longitude"}), None)

    vert_candidates = ["lev", "level", "ilev", "nhym", "z", "height"]
    vert_dim = next((d for d in dims if d.lower() in vert_candidates), None)

    if time_dim is None or lat_dim is None or lon_dim is None or vert_dim is None:
        raise ValueError(
            f"Could not identify required dims for variable '{var}'. "
            f"Found dims: {dims}"
        )

    arr = da.isel(
        {
            time_dim: 0,
            vert_dim: int(k_star),
            lat_dim: slice(i1_s, i2_s + 1),
            lon_dim: slice(j1_s, j2_s + 1),
        }
    ).values

    return to_ppb_fn(arr,species)


# ============================================================
# LOOKUP PREPARATION
# ============================================================

def prepare_station_configs(
    target_stations: pd.DataFrame,
    hlookup_df: pd.DataFrame,
    level_lookup_df: pd.DataFrame,
    constant_k_df: Optional[pd.DataFrame],
    variable_k_stations: List[int],
    use_hybrid_k_lookup: bool,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    weighted: bool,
    cell_nums: int,
    grid_info: dict,
):
    lats = grid_info["lats"]
    lons = grid_info["lons"]
    Ny, Nx = (lats.shape[0], lons.shape[0]) if np.ndim(lats) == 1 else lats.shape

    variable_k_stations = set(int(x) for x in variable_k_stations)

    hmap = {int(r["idx"]): r.to_dict() for _, r in hlookup_df.iterrows()}
    cmap = {}
    if constant_k_df is not None:
        cmap = {int(r["idx"]): r.to_dict() for _, r in constant_k_df.iterrows()}

    # Prepare timestep lookup maps only for stations that need them
    timestep_station_ids = []
    for _, st in target_stations.iterrows():
        st_idx = int(st["idx"])
        if use_hybrid_k_lookup and (st_idx not in variable_k_stations):
            continue
        timestep_station_ids.append(st_idx)

    timestep_lookup_maps: Dict[int, Dict[dt.datetime, dict]] = {}
    if timestep_station_ids:
        sub = level_lookup_df[
            level_lookup_df["idx"].isin(timestep_station_ids)
        ].copy()
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

    for _, st in target_stations.iterrows():
        st_idx = int(st["idx"])
        name = st["Station_Name"]
        lat_s = float(st["Latitude"])
        lon_s = float(st["Longitude"])
        alt_s = float(st["Altitude"])

        hrow = hmap.get(st_idx)
        if hrow is None:
            print(f"[skip-config] No horizontal lookup row for station {name} (idx={st_idx}).")
            continue

        i = int(hrow["i"])
        j = int(hrow["j"])
        model_lat = float(hrow["model_lat"])
        model_lon = float(hrow["model_lon"])

        if use_hybrid_k_lookup and (st_idx not in variable_k_stations):
            lookup_mode = "constant"
            constant_k_row = cmap.get(st_idx)
            lookup_map = None
            if constant_k_row is None:
                print(f"[skip-config] No constant-k row for station {name} (idx={st_idx}).")
                continue
        else:
            lookup_mode = "timestep"
            constant_k_row = None
            lookup_map = timestep_lookup_maps.get(st_idx)
            if not lookup_map:
                print(f"[skip-config] No timestep-k rows for station {name} (idx={st_idx}).")
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

    return station_cfgs


# ============================================================
# PER-STATION ROW BUILDER
# ============================================================

def build_records_for_one_station_and_timestep(
    ds_species: xr.Dataset,
    cfg: dict,
    date: str,
    time_: str,
    ts: dt.datetime,
    weighted: bool,
):
    if cfg["lookup_mode"] == "constant":
        lookup_row = cfg["constant_k_row"]
    else:
        lookup_row = cfg["lookup_map"].get(ts)

    if lookup_row is None:
        return []

    k_star_t = int(lookup_row["level_idx"])
    z_star_t = float(lookup_row["level_height_m"]) if "level_height_m" in lookup_row else np.nan

    try:
        grid_ppb = extract_smallbox_ppb_optionA_given_k(
            ds_species=ds_species,
            species=species,
            k_star=k_star_t,
            i1_s=cfg["i1_s"],
            i2_s=cfg["i2_s"],
            j1_s=cfg["j1_s"],
            j2_s=cfg["j2_s"],
            to_ppb_fn=to_ppb_mmr,
        )
    except Exception as e:
        print(
            f"[extract-error] station={cfg['name']} idx={cfg['station_idx']} "
            f"date={date} time={time_} k={k_star_t} error={e}"
        )
        return []

    valid = np.isfinite(grid_ppb)
    center_ppb = float(grid_ppb[cfg["ii"], cfg["jj"]]) if np.isfinite(grid_ppb[cfg["ii"], cfg["jj"]]) else np.nan

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

        vals = grid_ppb[mask_valid]

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
            "k_star_center": int(k_star_t),
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
# BASELINE: STATION-FIRST
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
    level_lookup_df = load_level_lookup(LEVEL_LOOKUP_PATH)
    constant_k_df = load_constant_k_lookup(CONSTANT_K_LOOKUP_PATH) if USE_HYBRID_K_LOOKUP else None

    grid_info = load_grid_info_once(base_path, product, species, start_dt, end_dt, step_minutes)

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
    )

    results = {}
    debug_species_file_structure()
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
# FASTER VERSION: TIMESTEP-FIRST MULTI-STATION
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
) -> Dict[int, dict]:
    """
    Faster timestep-first multi-station implementation.
    Opens each timestep file once and processes all selected stations from it.
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
    level_lookup_df = load_level_lookup(LEVEL_LOOKUP_PATH)
    constant_k_df = load_constant_k_lookup(CONSTANT_K_LOOKUP_PATH) if USE_HYBRID_K_LOOKUP else None

    grid_info = load_grid_info_once(base_path, product, species, start_dt, end_dt, step_minutes)

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
    )

    for date, time_ in iter_timestamps(start_dt, end_dt, step_minutes):
        ts = dt.datetime.strptime(date + time_, "%Y%m%d%H%M")
        spf, _, _, _, _ = build_paths(base_path, product, species, date, time_)
        if not spf.exists():
            continue

        ds_species = None
        try:
            ds_species = xr.open_dataset(spf, decode_times=False)

            for st_idx, cfg in station_cfgs.items():
                recs = build_records_for_one_station_and_timestep(
                    ds_species=ds_species,
                    cfg=cfg,
                    date=date,
                    time_=time_,
                    ts=ts,
                    weighted=weighted,
                )
                if recs:
                    cfg["rows"].extend(recs)

        finally:
            if ds_species is not None:
                try:
                    ds_species.close()
                except Exception:
                    pass

    results = {}
    debug_species_file_structure()
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
