#%%
from pathlib import Path
import re
import numpy as np
import xarray as xr
import pandas as pd
import time
import datetime as dt

# ============================================================
# USER SETTINGS
# ============================================================

NC_DIR = Path("/mnt/store01/agkiokas/CAMS/inst/subsets/O3/")
NC_GLOB = "*.nc*"
VAR_NAME = "O3"

OUT_FILE = Path("/mnt/store01/agkiokas/CAMS/O3_temporal_mean_std_season_daynight.nc")

PRINT_EVERY = 200

# Standard sunrise/sunset solar elevation threshold
# -0.833 degrees accounts approximately for refraction + solar disk radius
DAY_THRESHOLD_DEG = -0.833

# std definition:
# 0 = population std, 1 = sample std
STD_DDOF = 0

SEASONS = ["DJF", "MAM", "JJA", "SON"]
DAYNIGHT = ["day", "night"]


# ============================================================
# HELPERS
# ============================================================

def guess_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_candidates = ["lat", "latitude", "Latitude", "LAT"]
    lon_candidates = ["lon", "longitude", "Longitude", "LON"]

    lat_name = next((x for x in lat_candidates if x in ds.coords or x in ds.variables), None)
    lon_name = next((x for x in lon_candidates if x in ds.coords or x in ds.variables), None)

    if lat_name is None or lon_name is None:
        raise ValueError("Could not identify lat/lon names in dataset")

    return lat_name, lon_name


def remove_singleton_dims(da: xr.DataArray) -> xr.DataArray:
    for dim in list(da.dims):
        if da.sizes[dim] == 1:
            da = da.isel({dim: 0})
    return da


def get_file_time(ds: xr.Dataset, fp: Path) -> pd.Timestamp:
    """
    Prefer time coordinate from the NetCDF file.
    If not available, try to infer YYYYMMDD or YYYYMMDDHH from filename.
    """

    time_candidates = ["time", "valid_time", "datetime", "date"]

    for name in time_candidates:
        if name in ds.coords or name in ds.variables:
            vals = ds[name].values
            vals = np.atleast_1d(vals)
            if len(vals) > 0:
                return pd.to_datetime(vals[0])

    # fallback from filename, e.g. 2024010100, 20240101
    m = re.search(r"(20\d{6})(\d{2})?", fp.name)
    if m:
        ymd = m.group(1)
        hh = m.group(2) if m.group(2) is not None else "00"
        return pd.to_datetime(ymd + hh, format="%Y%m%d%H")

    raise ValueError(f"Could not determine timestamp for file: {fp.name}")


def month_to_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "DJF"
    if month in [3, 4, 5]:
        return "MAM"
    if month in [6, 7, 8]:
        return "JJA"
    return "SON"


def get_lat_lon_2d(ds: xr.Dataset, lat_name: str, lon_name: str):
    lat = ds[lat_name].values
    lon = ds[lon_name].values

    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    elif lat.ndim == 2 and lon.ndim == 2:
        lat2d, lon2d = lat, lon
    else:
        raise ValueError(
            f"Unsupported lat/lon shapes: lat={lat.shape}, lon={lon.shape}"
        )

    return lat2d.astype(float), lon2d.astype(float)


def astronomical_day_mask(timestamp: pd.Timestamp, lat2d: np.ndarray, lon2d: np.ndarray):
    """
    Returns True for day and False for night for each grid cell.

    Assumption:
    timestamp is UTC. CAMS files are usually UTC, but verify this for your files.
    """

    ts = pd.Timestamp(timestamp).tz_localize(None)

    doy = ts.dayofyear
    hour = ts.hour + ts.minute / 60 + ts.second / 3600

    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (hour - 12.0) / 24.0)

    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    utc_minutes = ts.hour * 60 + ts.minute + ts.second / 60

    # longitude correction: 4 minutes per degree
    true_solar_time = (utc_minutes + eqtime + 4.0 * lon2d) % 1440.0

    hour_angle_deg = true_solar_time / 4.0 - 180.0
    hour_angle = np.deg2rad(hour_angle_deg)

    lat_rad = np.deg2rad(lat2d)

    cos_zenith = (
        np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    )

    threshold = np.sin(np.deg2rad(DAY_THRESHOLD_DEG))

    return cos_zenith > threshold


def update_stats(vals, sum_arr, sumsq_arr, count_arr, mask_extra=None):
    vals = np.asarray(vals, dtype=np.float64)
    mask = np.isfinite(vals)

    if mask_extra is not None:
        mask = mask & mask_extra

    sum_arr[mask] += vals[mask]
    sumsq_arr[mask] += vals[mask] ** 2
    count_arr[mask] += 1


def finalize_mean_std(sum_arr, sumsq_arr, count_arr, ddof=0):
    mean = np.divide(
        sum_arr,
        count_arr,
        out=np.full_like(sum_arr, np.nan, dtype=np.float64),
        where=count_arr > 0,
    )

    denom = count_arr - ddof

    variance = np.divide(
        sumsq_arr - count_arr * mean ** 2,
        denom,
        out=np.full_like(sum_arr, np.nan, dtype=np.float64),
        where=denom > 0,
    )

    variance = np.where(variance < 0, 0, variance)
    std = np.sqrt(variance)

    return mean, std


def broadcast_day_mask(day_mask_2d, data_dims, data_shape, lat_name, lon_name):
    """
    Converts 2D lat/lon day mask to the same shape as the data variable.
    Works for:
    - (lat, lon)
    - (lev, lat, lon)
    """

    if len(data_shape) == 2:
        return day_mask_2d

    if len(data_shape) == 3:
        lat_axis = data_dims.index(lat_name)
        lon_axis = data_dims.index(lon_name)

        mask = day_mask_2d

        for axis in range(3):
            if axis not in [lat_axis, lon_axis]:
                mask = np.expand_dims(mask, axis=axis)

        return np.broadcast_to(mask, data_shape)

    raise ValueError(f"Unsupported data shape for day/night mask: {data_shape}")


# ============================================================
# MAIN
# ============================================================

def main():
    start = time.time()
    print("Start:", dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"))

    files = sorted(NC_DIR.glob(NC_GLOB))
    if not files:
        raise FileNotFoundError(f"No files found in {NC_DIR} matching {NC_GLOB}")

    with xr.open_dataset(files[0]) as ds0:
        if VAR_NAME not in ds0.variables:
            raise ValueError(f"{VAR_NAME} not found in first file: {files[0].name}")

        da0 = remove_singleton_dims(ds0[VAR_NAME])

        if da0.ndim not in [2, 3]:
            raise ValueError(
                f"Unsupported ndim for {VAR_NAME}: {da0.ndim}, dims={da0.dims}. "
                "Expected 2D or 3D after removing singleton dims."
            )

        data_shape = da0.shape
        data_dims = da0.dims

        lat_name, lon_name = guess_lat_lon_names(ds0)
        lat2d, lon2d = get_lat_lon_2d(ds0, lat_name, lon_name)

        if lat_name not in data_dims or lon_name not in data_dims:
            raise ValueError(
                f"lat/lon dimensions must be part of data dims. "
                f"data_dims={data_dims}, lat={lat_name}, lon={lon_name}"
            )

        coords_to_keep = {}
        for name in data_dims:
            if name in ds0.coords:
                coords_to_keep[name] = ds0.coords[name].values
            elif name in ds0.variables:
                coords_to_keep[name] = ds0[name].values

    print(f"Found {len(files)} files")
    print(f"Variable: {VAR_NAME}")
    print(f"Dims after squeeze: {data_dims}")
    print(f"Shape: {data_shape}")
    print(f"Lat/Lon names: {lat_name}, {lon_name}")

    # Overall temporal stats
    total_sum = np.zeros(data_shape, dtype=np.float64)
    total_sumsq = np.zeros(data_shape, dtype=np.float64)
    total_count = np.zeros(data_shape, dtype=np.int64)

    # Seasonal stats
    seasonal_sum = {
        s: np.zeros(data_shape, dtype=np.float64) for s in SEASONS
    }
    seasonal_sumsq = {
        s: np.zeros(data_shape, dtype=np.float64) for s in SEASONS
    }
    seasonal_count = {
        s: np.zeros(data_shape, dtype=np.int64) for s in SEASONS
    }

    # Day/night stats
    daynight_sum = {
        k: np.zeros(data_shape, dtype=np.float64) for k in DAYNIGHT
    }
    daynight_sumsq = {
        k: np.zeros(data_shape, dtype=np.float64) for k in DAYNIGHT
    }
    daynight_count = {
        k: np.zeros(data_shape, dtype=np.int64) for k in DAYNIGHT
    }

    for idx, fp in enumerate(files, start=1):
        try:
            with xr.open_dataset(fp) as ds:
                if VAR_NAME not in ds.variables:
                    print(f"Skipping {fp.name}: variable {VAR_NAME} not found")
                    continue

                timestamp = get_file_time(ds, fp)
                season = month_to_season(timestamp.month)

                da = remove_singleton_dims(ds[VAR_NAME])

                if da.dims != data_dims or da.shape != data_shape:
                    print(
                        f"Skipping {fp.name}: incompatible shape/dims. "
                        f"Expected {data_dims} {data_shape}, got {da.dims} {da.shape}"
                    )
                    continue

                vals = np.asarray(da.values, dtype=np.float64)

                # overall
                update_stats(vals, total_sum, total_sumsq, total_count)

                # seasonal
                update_stats(
                    vals,
                    seasonal_sum[season],
                    seasonal_sumsq[season],
                    seasonal_count[season],
                )

                # astronomical day/night
                day_mask_2d = astronomical_day_mask(timestamp, lat2d, lon2d)

                day_mask = broadcast_day_mask(
                    day_mask_2d,
                    data_dims,
                    data_shape,
                    lat_name,
                    lon_name,
                )

                night_mask = ~day_mask

                update_stats(
                    vals,
                    daynight_sum["day"],
                    daynight_sumsq["day"],
                    daynight_count["day"],
                    mask_extra=day_mask,
                )

                update_stats(
                    vals,
                    daynight_sum["night"],
                    daynight_sumsq["night"],
                    daynight_count["night"],
                    mask_extra=night_mask,
                )

                if idx % PRINT_EVERY == 0 or idx == len(files):
                    print(f"Processed {idx}/{len(files)} files")

        except Exception as e:
            print(f"Skipping {fp.name}: {e}")

    # Finalize overall
    total_mean, total_std = finalize_mean_std(
        total_sum, total_sumsq, total_count, ddof=STD_DDOF
    )

    # Finalize seasonal
    seasonal_mean_list = []
    seasonal_std_list = []
    seasonal_count_list = []

    for s in SEASONS:
        m, st = finalize_mean_std(
            seasonal_sum[s],
            seasonal_sumsq[s],
            seasonal_count[s],
            ddof=STD_DDOF,
        )
        seasonal_mean_list.append(m)
        seasonal_std_list.append(st)
        seasonal_count_list.append(seasonal_count[s])

    seasonal_mean_arr = np.stack(seasonal_mean_list, axis=0)
    seasonal_std_arr = np.stack(seasonal_std_list, axis=0)
    seasonal_count_arr = np.stack(seasonal_count_list, axis=0)

    # Finalize day/night
    daynight_mean_list = []
    daynight_std_list = []
    daynight_count_list = []

    for k in DAYNIGHT:
        m, st = finalize_mean_std(
            daynight_sum[k],
            daynight_sumsq[k],
            daynight_count[k],
            ddof=STD_DDOF,
        )
        daynight_mean_list.append(m)
        daynight_std_list.append(st)
        daynight_count_list.append(daynight_count[k])

    daynight_mean_arr = np.stack(daynight_mean_list, axis=0)
    daynight_std_arr = np.stack(daynight_std_list, axis=0)
    daynight_count_arr = np.stack(daynight_count_list, axis=0)

    # ========================================================
    # BUILD OUTPUT DATASET
    # ========================================================

    coord_dict = {
        dim: coords_to_keep[dim]
        for dim in data_dims
        if dim in coords_to_keep
    }

    out_ds = xr.Dataset()

    out_ds[f"{VAR_NAME}_temporal_mean"] = xr.DataArray(
        total_mean,
        dims=data_dims,
        coords=coord_dict,
        attrs={
            "long_name": f"Temporal mean of {VAR_NAME}",
            "note": "Arithmetic mean over all valid timestep files",
        },
    )

    out_ds[f"{VAR_NAME}_temporal_std"] = xr.DataArray(
        total_std,
        dims=data_dims,
        coords=coord_dict,
        attrs={
            "long_name": f"Temporal standard deviation of {VAR_NAME}",
            "ddof": STD_DDOF,
        },
    )

    out_ds[f"{VAR_NAME}_valid_count"] = xr.DataArray(
        total_count,
        dims=data_dims,
        coords=coord_dict,
        attrs={
            "long_name": f"Number of valid samples used for temporal statistics of {VAR_NAME}",
        },
    )

    seasonal_dims = ("season",) + data_dims
    seasonal_coords = {"season": SEASONS}
    seasonal_coords.update(coord_dict)

    out_ds[f"{VAR_NAME}_seasonal_mean"] = xr.DataArray(
        seasonal_mean_arr,
        dims=seasonal_dims,
        coords=seasonal_coords,
        attrs={
            "long_name": f"Seasonal mean of {VAR_NAME}",
            "seasons": "DJF, MAM, JJA, SON",
        },
    )

    out_ds[f"{VAR_NAME}_seasonal_std"] = xr.DataArray(
        seasonal_std_arr,
        dims=seasonal_dims,
        coords=seasonal_coords,
        attrs={
            "long_name": f"Seasonal standard deviation of {VAR_NAME}",
            "ddof": STD_DDOF,
        },
    )

    out_ds[f"{VAR_NAME}_seasonal_valid_count"] = xr.DataArray(
        seasonal_count_arr,
        dims=seasonal_dims,
        coords=seasonal_coords,
        attrs={
            "long_name": f"Number of valid samples used for seasonal statistics of {VAR_NAME}",
        },
    )

    daynight_dims = ("daynight",) + data_dims
    daynight_coords = {"daynight": DAYNIGHT}
    daynight_coords.update(coord_dict)

    out_ds[f"{VAR_NAME}_daynight_mean"] = xr.DataArray(
        daynight_mean_arr,
        dims=daynight_dims,
        coords=daynight_coords,
        attrs={
            "long_name": f"Day/night mean of {VAR_NAME}",
            "day_definition": f"Solar elevation greater than {DAY_THRESHOLD_DEG} degrees",
        },
    )

    out_ds[f"{VAR_NAME}_daynight_std"] = xr.DataArray(
        daynight_std_arr,
        dims=daynight_dims,
        coords=daynight_coords,
        attrs={
            "long_name": f"Day/night standard deviation of {VAR_NAME}",
            "ddof": STD_DDOF,
            "day_definition": f"Solar elevation greater than {DAY_THRESHOLD_DEG} degrees",
        },
    )

    out_ds[f"{VAR_NAME}_daynight_valid_count"] = xr.DataArray(
        daynight_count_arr,
        dims=daynight_dims,
        coords=daynight_coords,
        attrs={
            "long_name": f"Number of valid samples used for day/night statistics of {VAR_NAME}",
        },
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(OUT_FILE)

    print(f"Saved file to: {OUT_FILE}")

    end = time.time()
    print("End:", dt.datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Execution time: {(end - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
# %%
