#same code logic as the build_gridded_pooled_cv_nc.py but with parallelized processing
#%%
from pathlib import Path
import re
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import time

warnings.filterwarnings("ignore")

# ============================================================
# USER SETTINGS
# ============================================================

NC_DIR = Path("/mnt/store01/agkiokas/CAMS/inst/subsets/O3/")
NC_GLOB = "*.nc*"
VAR_NAME = "O3"

OUT_FILE = Path("/mnt/store01/agkiokas/CAMS/O3_gridded_pooled_cv_parallel.nc")

SECTORS = [f"C{i}" for i in range(1, 11)]
LEVELS_TO_KEEP = None

AREA_WEIGHTED = True
SAVE_MEAN_STD_COUNT = True

THRESHOLD_DEG = -0.833
PRINT_EVERY = 200

from concurrent.futures import ProcessPoolExecutor, as_completed
import os


N_WORKERS = 2          # try 4, 8, 12 depending on server
CHUNK_SIZE = 50       # each worker processes 200 files


def process_file_chunk(file_chunk, shape3, lat2d, lon2d):
    sin_lat, cos_lat, lon_norm = prepare_solar_grid(lat2d, lon2d)

    sum_all = np.zeros(shape3, dtype=np.float64)
    sumsq_all = np.zeros(shape3, dtype=np.float64)
    count_all = np.zeros(shape3, dtype=np.float64)

    sum_season = np.zeros((4, *shape3), dtype=np.float64)
    sumsq_season = np.zeros((4, *shape3), dtype=np.float64)
    count_season = np.zeros((4, *shape3), dtype=np.float64)

    sum_dn = np.zeros((2, *shape3), dtype=np.float64)
    sumsq_dn = np.zeros((2, *shape3), dtype=np.float64)
    count_dn = np.zeros((2, *shape3), dtype=np.float64)

    sum_sd = np.zeros((4, 2, *shape3), dtype=np.float64)
    sumsq_sd = np.zeros((4, 2, *shape3), dtype=np.float64)
    count_sd = np.zeros((4, 2, *shape3), dtype=np.float64)

    for fp in file_chunk:
        ts = infer_timestamp(fp)
        sidx = month_to_season_idx(ts.month)

        with xr.open_dataset(fp) as ds:
            vals = extract_field_3d(ds, VAR_NAME)

        day_mask = solar_day_mask_fast(
            sin_lat,
            cos_lat,
            lon_norm,
            ts,
            threshold_deg=THRESHOLD_DEG,
        )
        night_mask = ~day_mask

        add_values(sum_all, sumsq_all, count_all, vals)
        add_values(sum_season[sidx], sumsq_season[sidx], count_season[sidx], vals)

        add_values(sum_dn[0], sumsq_dn[0], count_dn[0], vals, day_mask)
        add_values(sum_dn[1], sumsq_dn[1], count_dn[1], vals, night_mask)

        add_values(sum_sd[sidx, 0], sumsq_sd[sidx, 0], count_sd[sidx, 0], vals, day_mask)
        add_values(sum_sd[sidx, 1], sumsq_sd[sidx, 1], count_sd[sidx, 1], vals, night_mask)

    return (
        sum_all, sumsq_all, count_all,
        sum_season, sumsq_season, count_season,
        sum_dn, sumsq_dn, count_dn,
        sum_sd, sumsq_sd, count_sd,
        len(file_chunk),
    )


def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]
# ============================================================
# TIME HELPERS
# ============================================================

def infer_timestamp(fp: Path):
    m = re.search(r"(\d{8})[_-]?(\d{4})", fp.stem)
    return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M")


def month_to_season_idx(month: int):
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


# ============================================================
# SOLAR DAY/NIGHT
# ============================================================

def prepare_solar_grid(lat2d, lon2d):
    lon_norm = ((lon2d + 180.0) % 360.0) - 180.0
    lat_rad = np.deg2rad(lat2d.astype(np.float64))
    return np.sin(lat_rad), np.cos(lat_rad), lon_norm


def solar_day_mask_fast(sin_lat, cos_lat, lon_norm, ts, threshold_deg=-0.833):
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()

    if ts.tzinfo is not None:
        ts = ts.astimezone(dt.timezone.utc)

    doy = ts.timetuple().tm_yday
    hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0

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

    tst = (hour * 60.0 + eqtime + 4.0 * lon_norm) % 1440.0
    ha = np.deg2rad((tst / 4.0) - 180.0)

    cosz = sin_lat * np.sin(decl) + cos_lat * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, -1.0, 1.0)

    elev = np.rad2deg(np.pi / 2.0 - np.arccos(cosz))
    return elev > threshold_deg


# ============================================================
# NC HELPERS
# ============================================================

def extract_field_3d(ds, var_name):
    da = ds[var_name].squeeze()

    if LEVELS_TO_KEEP is not None:
        da = da.isel({da.dims[0]: LEVELS_TO_KEEP})

    vals = da.to_numpy().astype(np.float64, copy=False)
    return vals


def read_lat_lon(ds):
    lat_name = "lat" if "lat" in ds else "latitude"
    lon_name = "lon" if "lon" in ds else "longitude"

    lat = ds[lat_name].to_numpy()
    lon = ds[lon_name].to_numpy()

    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d, lon2d = lat, lon

    return lat, lon, lat2d, lon2d


# ============================================================
# FAST WINDOW SUM, VECTORIZED OVER LEADING DIMS
# ============================================================

def window_sum_nd(arr, radius):
    """
    Sum over square spatial window for arrays shaped (..., lat, lon).
    """
    arr = np.asarray(arr, dtype=np.float64)

    padded = np.pad(
        arr,
        [(0, 0)] * (arr.ndim - 2) + [(1, 0), (1, 0)],
        mode="constant",
        constant_values=0.0,
    )

    cs = padded.cumsum(axis=-2).cumsum(axis=-1)

    ny, nx = arr.shape[-2:]
    i = np.arange(ny)
    j = np.arange(nx)

    i0 = np.maximum(0, i - radius)
    i1 = np.minimum(ny - 1, i + radius) + 1
    j0 = np.maximum(0, j - radius)
    j1 = np.minimum(nx - 1, j + radius) + 1

    return (
        cs[..., i1[:, None], j1[None, :]]
        - cs[..., i0[:, None], j1[None, :]]
        - cs[..., i1[:, None], j0[None, :]]
        + cs[..., i0[:, None], j0[None, :]]
    )


def pooled_stats_from_accumulators(sum_arr, sumsq_arr, count_arr, weights2d, radius):
    weighted_sum = window_sum_nd(sum_arr * weights2d, radius)
    weighted_sumsq = window_sum_nd(sumsq_arr * weights2d, radius)
    weighted_count = window_sum_nd(count_arr * weights2d, radius)

    mean = np.divide(
        weighted_sum,
        weighted_count,
        out=np.full_like(weighted_sum, np.nan),
        where=weighted_count > 0,
    )

    mean_sq = np.divide(
        weighted_sumsq,
        weighted_count,
        out=np.full_like(weighted_sumsq, np.nan),
        where=weighted_count > 0,
    )

    var = mean_sq - mean**2
    var = np.maximum(var, 0.0)

    std = np.sqrt(var)

    cv = np.divide(
        100.0 * std,
        mean,
        out=np.full_like(std, np.nan),
        where=np.isfinite(mean) & (mean != 0),
    )

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
        cv.astype(np.float32),
        weighted_count.astype(np.float32),
    )


# ============================================================
# ACCUMULATOR HELPER
# ============================================================

def add_values(sum_arr, sumsq_arr, count_arr, vals, mask=None):
    finite = np.isfinite(vals)

    if mask is not None:
        finite &= mask[None, :, :]

    clean = np.where(finite, vals, 0.0)

    sum_arr += clean
    sumsq_arr += clean * clean
    count_arr += finite


# ============================================================
# MAIN
# ============================================================

def main():
    start = time.time()
    print("\n===== PRODUCTION RUN =====")
    print("Start:", dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"))
    files = sorted(NC_DIR.glob(NC_GLOB))
    print(f"Found {len(files)} NC files")

    with xr.open_dataset(files[0]) as ds0:
        vals0 = extract_field_3d(ds0, VAR_NAME)
        lat, lon, lat2d, lon2d = read_lat_lon(ds0)

    nlev, ny, nx = vals0.shape
    shape3 = (nlev, ny, nx)

    used_levels = np.arange(nlev) if LEVELS_TO_KEEP is None else np.array(LEVELS_TO_KEEP)

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    dn_vals = ["day", "night"]

    weights2d = (
        np.cos(np.deg2rad(lat2d)).astype(np.float64)
        if AREA_WEIGHTED
        else np.ones_like(lat2d, dtype=np.float64)
    )

    sin_lat, cos_lat, lon_norm = prepare_solar_grid(lat2d, lon2d)

    sum_all = np.zeros(shape3, dtype=np.float64)
    sumsq_all = np.zeros(shape3, dtype=np.float64)
    count_all = np.zeros(shape3, dtype=np.float64)

    sum_season = np.zeros((4, *shape3), dtype=np.float64)
    sumsq_season = np.zeros((4, *shape3), dtype=np.float64)
    count_season = np.zeros((4, *shape3), dtype=np.float64)

    sum_dn = np.zeros((2, *shape3), dtype=np.float64)
    sumsq_dn = np.zeros((2, *shape3), dtype=np.float64)
    count_dn = np.zeros((2, *shape3), dtype=np.float64)

    sum_sd = np.zeros((4, 2, *shape3), dtype=np.float64)
    sumsq_sd = np.zeros((4, 2, *shape3), dtype=np.float64)
    count_sd = np.zeros((4, 2, *shape3), dtype=np.float64)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    file_chunks = list(chunks(files, CHUNK_SIZE))

    processed = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [
        executor.submit(process_file_chunk, chunk, shape3, lat2d, lon2d)
        for chunk in file_chunks
        ]

        for future in as_completed(futures):
            (
             p_sum_all, p_sumsq_all, p_count_all,
             p_sum_season, p_sumsq_season, p_count_season,
             p_sum_dn, p_sumsq_dn, p_count_dn,
             p_sum_sd, p_sumsq_sd, p_count_sd,
             n_done,
            ) = future.result()

            sum_all += p_sum_all
            sumsq_all += p_sumsq_all
            count_all += p_count_all

            sum_season += p_sum_season
            sumsq_season += p_sumsq_season
            count_season += p_count_season

            sum_dn += p_sum_dn
            sumsq_dn += p_sumsq_dn
            count_dn += p_count_dn

            sum_sd += p_sum_sd
            sumsq_sd += p_sumsq_sd
            count_sd += p_count_sd

            processed += n_done

            if processed % PRINT_EVERY == 0 or processed == len(files):
                print(f"Processed {processed}/{len(files)} files")

    sector_names = SECTORS
    radii = np.array([int("".join(ch for ch in s if ch.isdigit())) for s in sector_names])

    cv_all = []
    cv_season = []
    cv_dn = []
    cv_sd = []

    if SAVE_MEAN_STD_COUNT:
        mean_all_list, std_all_list, count_all_list = [], [], []
        mean_season_list, std_season_list, count_season_list = [], [], []
        mean_dn_list, std_dn_list, count_dn_list = [], [], []
        mean_sd_list, std_sd_list, count_sd_list = [], [], []

    for sector, radius in zip(sector_names, radii):
        print(f"Computing {sector}, radius={radius}")

        mean_, std_, cv_, cnt_ = pooled_stats_from_accumulators(
            sum_all, sumsq_all, count_all, weights2d, radius
        )
        cv_all.append(cv_)

        if SAVE_MEAN_STD_COUNT:
            mean_all_list.append(mean_)
            std_all_list.append(std_)
            count_all_list.append(cnt_)

        mean_, std_, cv_, cnt_ = pooled_stats_from_accumulators(
            sum_season, sumsq_season, count_season, weights2d, radius
        )
        cv_season.append(cv_)

        if SAVE_MEAN_STD_COUNT:
            mean_season_list.append(mean_)
            std_season_list.append(std_)
            count_season_list.append(cnt_)

        mean_, std_, cv_, cnt_ = pooled_stats_from_accumulators(
            sum_dn, sumsq_dn, count_dn, weights2d, radius
        )
        cv_dn.append(cv_)

        if SAVE_MEAN_STD_COUNT:
            mean_dn_list.append(mean_)
            std_dn_list.append(std_)
            count_dn_list.append(cnt_)

        mean_, std_, cv_, cnt_ = pooled_stats_from_accumulators(
            sum_sd, sumsq_sd, count_sd, weights2d, radius
        )
        cv_sd.append(cv_)

        if SAVE_MEAN_STD_COUNT:
            mean_sd_list.append(mean_)
            std_sd_list.append(std_)
            count_sd_list.append(cnt_)

    cv_all = np.stack(cv_all, axis=0)
    cv_season = np.stack(cv_season, axis=1)
    cv_dn = np.stack(cv_dn, axis=1)
    cv_sd = np.stack(cv_sd, axis=2)

    ds_out = xr.Dataset()

    ds_out["sector"] = ("sector", sector_names)
    ds_out["radius"] = ("sector", radii)
    ds_out["season"] = ("season", seasons)
    ds_out["day_night"] = ("day_night", dn_vals)
    ds_out["lev"] = ("lev", used_levels)

    if lat.ndim == 1 and lon.ndim == 1:
        ds_out["lat"] = ("lat", lat)
        ds_out["lon"] = ("lon", lon)
    else:
        ds_out["lat"] = (("lat", "lon"), lat)
        ds_out["lon"] = (("lat", "lon"), lon)

    ds_out["cv_pct_all"] = xr.DataArray(
        cv_all,
        dims=("sector", "lev", "lat", "lon"),
    )

    ds_out["cv_pct_season"] = xr.DataArray(
        cv_season,
        dims=("season", "sector", "lev", "lat", "lon"),
    )

    ds_out["cv_pct_daynight"] = xr.DataArray(
        cv_dn,
        dims=("day_night", "sector", "lev", "lat", "lon"),
    )

    ds_out["cv_pct_season_daynight"] = xr.DataArray(
        cv_sd,
        dims=("season", "day_night", "sector", "lev", "lat", "lon"),
    )

    if SAVE_MEAN_STD_COUNT:
        ds_out["mean_all"] = xr.DataArray(
            np.stack(mean_all_list, axis=0),
            dims=("sector", "lev", "lat", "lon"),
        )
        ds_out["std_all"] = xr.DataArray(
            np.stack(std_all_list, axis=0),
            dims=("sector", "lev", "lat", "lon"),
        )
        ds_out["count_all"] = xr.DataArray(
            np.stack(count_all_list, axis=0),
            dims=("sector", "lev", "lat", "lon"),
        )

        ds_out["mean_season"] = xr.DataArray(
            np.stack(mean_season_list, axis=1),
            dims=("season", "sector", "lev", "lat", "lon"),
        )
        ds_out["std_season"] = xr.DataArray(
            np.stack(std_season_list, axis=1),
            dims=("season", "sector", "lev", "lat", "lon"),
        )
        ds_out["count_season"] = xr.DataArray(
            np.stack(count_season_list, axis=1),
            dims=("season", "sector", "lev", "lat", "lon"),
        )

        ds_out["mean_daynight"] = xr.DataArray(
            np.stack(mean_dn_list, axis=1),
            dims=("day_night", "sector", "lev", "lat", "lon"),
        )
        ds_out["std_daynight"] = xr.DataArray(
            np.stack(std_dn_list, axis=1),
            dims=("day_night", "sector", "lev", "lat", "lon"),
        )
        ds_out["count_daynight"] = xr.DataArray(
            np.stack(count_dn_list, axis=1),
            dims=("day_night", "sector", "lev", "lat", "lon"),
        )

        ds_out["mean_season_daynight"] = xr.DataArray(
            np.stack(mean_sd_list, axis=2),
            dims=("season", "day_night", "sector", "lev", "lat", "lon"),
        )
        ds_out["std_season_daynight"] = xr.DataArray(
            np.stack(std_sd_list, axis=2),
            dims=("season", "day_night", "sector", "lev", "lat", "lon"),
        )
        ds_out["count_season_daynight"] = xr.DataArray(
            np.stack(count_sd_list, axis=2),
            dims=("season", "day_night", "sector", "lev", "lat", "lon"),
        )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in ds_out.data_vars
        if ds_out[var].dtype.kind == "f"
    }

    ds_out.to_netcdf(OUT_FILE, encoding=encoding)

    print(f"Saved gridded pooled CV file: {OUT_FILE}")
    end = time.time()
    print("End:", dt.datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Execution time: {(end - start) / 60:.2f} minutes")

if __name__ == "__main__":
    main()
# %%
