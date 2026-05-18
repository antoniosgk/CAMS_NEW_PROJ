#%%
from pathlib import Path
import re
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import shutil
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# USER SETTINGS
# ============================================================

RAW_NC_DIR = Path("/mnt/store01/agkiokas/CAMS/inst/subsets/O3/")
RAW_NC_GLOB = "*.nc*"
VAR_NAME = "O3"

EXISTING_FILE = Path("/mnt/store01/agkiokas/CAMS/O3_gridded_pooled_cv.nc")

C0_ONLY_FILE = Path("/mnt/store01/agkiokas/CAMS/O3_gridded_pooled_cv_C0_only.nc")

COMBINED_FILE = Path("/mnt/store01/agkiokas/CAMS/O3_gridded_pooled_cv_with_C0.nc")

BACKUP_EXISTING_FILE = Path("/mnt/store01/agkiokas/CAMS/O3_gridded_pooled_cv_original_backup.nc")

LEVELS_TO_KEEP = None
THRESHOLD_DEG = -0.833
PRINT_EVERY = 100

SAVE_MEAN_STD_COUNT = True


# ============================================================
# HELPERS
# ============================================================

def infer_timestamp(fp: Path):
    m = re.search(r"(\d{8})[_-]?(\d{4})", fp.stem)
    return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M")


def month_to_season_idx(month):
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


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


def extract_field_3d(ds, var_name):
    da = ds[var_name].squeeze()

    if LEVELS_TO_KEEP is not None:
        da = da.isel({da.dims[0]: LEVELS_TO_KEEP})

    return da.to_numpy().astype(np.float64, copy=False)


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


def add_values(sum_arr, sumsq_arr, count_arr, vals, mask=None):
    finite = np.isfinite(vals)

    if mask is not None:
        finite &= mask[None, :, :]

    clean = np.where(finite, vals, 0.0)

    sum_arr += clean
    sumsq_arr += clean * clean
    count_arr += finite


def stats_from_accumulators(sum_arr, sumsq_arr, count_arr):
    mean = np.divide(
        sum_arr,
        count_arr,
        out=np.full_like(sum_arr, np.nan),
        where=count_arr > 0,
    )

    mean_sq = np.divide(
        sumsq_arr,
        count_arr,
        out=np.full_like(sumsq_arr, np.nan),
        where=count_arr > 0,
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
        count_arr.astype(np.float32),
    )


# ============================================================
# BUILD C0 ONLY FILE
# ============================================================

def build_c0_only_file():
    files = sorted(RAW_NC_DIR.glob(RAW_NC_GLOB))

    if not files:
        raise FileNotFoundError(f"No raw NC files found in {RAW_NC_DIR}")

    print(f"Found {len(files)} raw NC files")

    with xr.open_dataset(files[0]) as ds0:
        vals0 = extract_field_3d(ds0, VAR_NAME)
        lat, lon, lat2d, lon2d = read_lat_lon(ds0)

    nlev, ny, nx = vals0.shape
    shape3 = (nlev, ny, nx)

    used_levels = np.arange(nlev) if LEVELS_TO_KEEP is None else np.array(LEVELS_TO_KEEP)

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    dn_vals = ["day", "night"]

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

    for n, fp in enumerate(files, start=1):
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

        if n % PRINT_EVERY == 0 or n == len(files):
            print(f"Processed {n}/{len(files)} files")

    mean_all, std_all, cv_all, cnt_all = stats_from_accumulators(
        sum_all, sumsq_all, count_all
    )

    mean_season, std_season, cv_season, cnt_season = stats_from_accumulators(
        sum_season, sumsq_season, count_season
    )

    mean_dn, std_dn, cv_dn, cnt_dn = stats_from_accumulators(
        sum_dn, sumsq_dn, count_dn
    )

    mean_sd, std_sd, cv_sd, cnt_sd = stats_from_accumulators(
        sum_sd, sumsq_sd, count_sd
    )

    ds_out = xr.Dataset()

    ds_out["sector"] = ("sector", ["C0"])
    ds_out["radius"] = ("sector", np.array([0], dtype=np.int32))
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
        cv_all[None, ...],
        dims=("sector", "lev", "lat", "lon"),
    )

    ds_out["cv_pct_season"] = xr.DataArray(
        cv_season[:, None, ...],
        dims=("season", "sector", "lev", "lat", "lon"),
    )

    ds_out["cv_pct_daynight"] = xr.DataArray(
        cv_dn[:, None, ...],
        dims=("day_night", "sector", "lev", "lat", "lon"),
    )

    ds_out["cv_pct_season_daynight"] = xr.DataArray(
        cv_sd[:, :, None, ...],
        dims=("season", "day_night", "sector", "lev", "lat", "lon"),
    )

    if SAVE_MEAN_STD_COUNT:
        ds_out["mean_all"] = xr.DataArray(
            mean_all[None, ...],
            dims=("sector", "lev", "lat", "lon"),
        )
        ds_out["std_all"] = xr.DataArray(
            std_all[None, ...],
            dims=("sector", "lev", "lat", "lon"),
        )
        ds_out["count_all"] = xr.DataArray(
            cnt_all[None, ...],
            dims=("sector", "lev", "lat", "lon"),
        )

        ds_out["mean_season"] = xr.DataArray(
            mean_season[:, None, ...],
            dims=("season", "sector", "lev", "lat", "lon"),
        )
        ds_out["std_season"] = xr.DataArray(
            std_season[:, None, ...],
            dims=("season", "sector", "lev", "lat", "lon"),
        )
        ds_out["count_season"] = xr.DataArray(
            cnt_season[:, None, ...],
            dims=("season", "sector", "lev", "lat", "lon"),
        )

        ds_out["mean_daynight"] = xr.DataArray(
            mean_dn[:, None, ...],
            dims=("day_night", "sector", "lev", "lat", "lon"),
        )
        ds_out["std_daynight"] = xr.DataArray(
            std_dn[:, None, ...],
            dims=("day_night", "sector", "lev", "lat", "lon"),
        )
        ds_out["count_daynight"] = xr.DataArray(
            cnt_dn[:, None, ...],
            dims=("day_night", "sector", "lev", "lat", "lon"),
        )

        ds_out["mean_season_daynight"] = xr.DataArray(
            mean_sd[:, :, None, ...],
            dims=("season", "day_night", "sector", "lev", "lat", "lon"),
        )
        ds_out["std_season_daynight"] = xr.DataArray(
            std_sd[:, :, None, ...],
            dims=("season", "day_night", "sector", "lev", "lat", "lon"),
        )
        ds_out["count_season_daynight"] = xr.DataArray(
            cnt_sd[:, :, None, ...],
            dims=("season", "day_night", "sector", "lev", "lat", "lon"),
        )

    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in ds_out.data_vars
        if ds_out[var].dtype.kind == "f"
    }

    C0_ONLY_FILE.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(C0_ONLY_FILE, encoding=encoding)

    print(f"Saved C0-only file: {C0_ONLY_FILE}")


# ============================================================
# CONCATENATE C0 + EXISTING C1-C10
# ============================================================

def concatenate_c0_with_existing():
    if not BACKUP_EXISTING_FILE.exists():
        shutil.copy2(EXISTING_FILE, BACKUP_EXISTING_FILE)
        print(f"Backup saved: {BACKUP_EXISTING_FILE}")

    ds_c0 = xr.open_dataset(C0_ONLY_FILE)
    ds_old = xr.open_dataset(EXISTING_FILE)

    ds_combined = xr.concat(
        [ds_c0, ds_old],
        dim="sector",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="exact",
    )

    ds_combined = ds_combined.assign_coords(
        sector=("sector", ["C0"] + [str(s) for s in ds_old["sector"].values])
    )

    ds_combined["radius"] = ("sector", np.array([0] + list(ds_old["radius"].values), dtype=np.int32))

    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in ds_combined.data_vars
        if ds_combined[var].dtype.kind == "f"
    }

    ds_combined.to_netcdf(COMBINED_FILE, encoding=encoding)

    ds_c0.close()
    ds_old.close()
    ds_combined.close()

    print(f"Saved combined file: {COMBINED_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():
    build_c0_only_file()
    concatenate_c0_with_existing()


if __name__ == "__main__":
    main()