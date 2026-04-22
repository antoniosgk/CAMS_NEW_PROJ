#%%
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import re
import time

# ============================================================
# USER SETTINGS
# ============================================================

NC_DIR = Path("/mnt/store01/agkiokas/CAMS/inst/subsets/O3/")
NC_GLOB = "*.nc*"
VAR_NAME = "O3"

OUT_DIR = Path("/mnt/store01/agkiokas/CAMS/output_climatologies")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "O3_climatology.nc"

THRESHOLD_DEG = -0.833
PRINT_EVERY = 1000

# Optional: if you want to keep only a subset of levels, set here.
# Example: LEVELS_TO_KEEP = [16, 17, 18, 19, 20, 21, 22]
LEVELS_TO_KEEP = None

# Optional: stop after first bad file for debugging
STOP_ON_FIRST_BAD_FILE = False


# ============================================================
# TIME / LABEL HELPERS
# ============================================================

def infer_timestamp(fp: Path):
    """
    Try to infer timestamp from filename using patterns like:
    YYYYMMDD_HHMM or YYYYMMDDHHMM
    """
    m = re.search(r"(\d{8})[_-]?(\d{4})", fp.stem)
    if m:
        return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M", errors="coerce")
    return pd.NaT


def month_to_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


# ============================================================
# SOLAR DAY/NIGHT (VECTORIZED)
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
# DIMENSION SAFETY HELPERS
# ============================================================

def identify_dims(da: xr.DataArray):
    """
    Identify lev, lat, lon dims robustly.
    """
    lat_dim = next((d for d in da.dims if d.lower() in ["lat", "latitude"]), None)
    lon_dim = next((d for d in da.dims if d.lower() in ["lon", "longitude"]), None)
    lev_dim = next((d for d in da.dims if d.lower() in ["lev", "level", "plev", "layer", "z", "alt"]), None)

    if lat_dim is None or lon_dim is None or lev_dim is None:
        raise ValueError(f"Could not identify lev/lat/lon dims in {da.dims}")

    return lev_dim, lat_dim, lon_dim


def extract_field_3d(ds: xr.Dataset, var_name: str, levels_to_keep=None):
    """
    Return variable as numpy array with shape (lev, lat, lon),
    after safely removing only singleton extra dims.

    If a non-singleton extra dim exists, raise a clear error.
    """
    if var_name not in ds.variables:
        raise ValueError(f"Variable {var_name} not found")

    da = ds[var_name]
    lev_dim, lat_dim, lon_dim = identify_dims(da)

    extra_dims = [d for d in da.dims if d not in [lev_dim, lat_dim, lon_dim]]

    # only remove extra dims if they are singleton
    for d in extra_dims:
        if da.sizes[d] != 1:
            raise ValueError(
                f"Unexpected non-singleton extra dimension '{d}' with size {da.sizes[d]} "
                f"in dims {da.dims}"
            )
        da = da.isel({d: 0})

    # optionally subset levels
    if levels_to_keep is not None:
        da = da.isel({lev_dim: levels_to_keep})

    # force order
    da = da.transpose(lev_dim, lat_dim, lon_dim)

    vals = np.asarray(da.values, dtype=np.float64)
    return vals, lev_dim, lat_dim, lon_dim


def get_lat_lon(ds: xr.Dataset, lat_dim: str, lon_dim: str):
    lat = np.asarray(ds[lat_dim].values)
    lon = np.asarray(ds[lon_dim].values)

    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d, lon2d = lat, lon

    return lat, lon, lat2d, lon2d


# ============================================================
# ENCODING
# ============================================================

def make_encoding(ds_out: xr.Dataset):
    enc = {}
    for v in ds_out.data_vars:
        enc[v] = {"zlib": True, "complevel": 4}
    return enc


# ============================================================
# MAIN
# ============================================================

def main():
    files = sorted(NC_DIR.glob(NC_GLOB))
    if not files:
        raise FileNotFoundError(f"No files found in {NC_DIR} matching {NC_GLOB}")

    print(f"Scanning {len(files)} files from {NC_DIR}")

    # --------------------------------------------------------
# FAST initialization from FIRST file (no inspection loop)
# --------------------------------------------------------
    fp0 = files[0]

    with xr.open_dataset(fp0) as ds0:
        da0 = ds0[VAR_NAME].squeeze()

    # optionally subset levels
        if LEVELS_TO_KEEP is not None:
            da0 = da0.isel({da0.dims[0]: LEVELS_TO_KEEP})

        vals0 = da0.values

        # assume order is already (lev, lat, lon)
        nlev, ny, nx = vals0.shape

    # coordinates
        lat = ds0["lat"].values if "lat" in ds0 else ds0["latitude"].values
        lon = ds0["lon"].values if "lon" in ds0 else ds0["longitude"].values

        # build 2D grid
        if lat.ndim == 1 and lon.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
        else:
            lat2d, lon2d = lat, lon

    # levels used
        if LEVELS_TO_KEEP is None:
            used_levels = np.arange(nlev)
        else:
            used_levels = np.array(LEVELS_TO_KEEP)

    print(f"Initialized from: {fp0.name}")
    print(f"Expected shape: {(nlev, ny, nx)}")

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    dn_vals = ["day", "night"]

    season_to_idx = {s: i for i, s in enumerate(seasons)}
    dn_to_idx = {"day": 0, "night": 1}

    # full-period
    sum_all = np.zeros((nlev, ny, nx), dtype=np.float64)
    cnt_all = np.zeros((nlev, ny, nx), dtype=np.int32)

    # seasonal
    sum_season = np.zeros((4, nlev, ny, nx), dtype=np.float64)
    cnt_season = np.zeros((4, nlev, ny, nx), dtype=np.int32)

    # day/night
    sum_dn = np.zeros((2, nlev, ny, nx), dtype=np.float64)
    cnt_dn = np.zeros((2, nlev, ny, nx), dtype=np.int32)

    # season x day/night
    sum_sd = np.zeros((4, 2, nlev, ny, nx), dtype=np.float64)
    cnt_sd = np.zeros((4, 2, nlev, ny, nx), dtype=np.int32)

    bad_files = []
    processed = 0

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------
    for i, fp in enumerate(files, start=1):
        try:
            ts = infer_timestamp(fp)
            if pd.isna(ts):
                raise ValueError("Could not infer timestamp from filename")

            season = month_to_season(ts.month)
            sidx = season_to_idx[season]

            with xr.open_dataset(fp) as ds:
                vals, _, _, _ = extract_field_3d(ds, VAR_NAME, levels_to_keep=LEVELS_TO_KEEP)

            if vals.shape != (nlev, ny, nx):
                raise ValueError(f"Unexpected shape {vals.shape}, expected {(nlev, ny, nx)}")

            mask = np.isfinite(vals)

            # ---- FULL PERIOD
            sum_all[mask] += vals[mask]
            cnt_all[mask] += 1

            # ---- SEASON
            sum_season[sidx][mask] += vals[mask]
            cnt_season[sidx][mask] += 1

            # ---- DAY/NIGHT
            mask_day_2d = solar_day_mask(lat2d, lon2d, ts, threshold_deg=THRESHOLD_DEG)
            mask_night_2d = ~mask_day_2d

            day_valid = mask & mask_day_2d[None, :, :]
            night_valid = mask & mask_night_2d[None, :, :]

            sum_dn[0][day_valid] += vals[day_valid]
            cnt_dn[0][day_valid] += 1

            sum_dn[1][night_valid] += vals[night_valid]
            cnt_dn[1][night_valid] += 1

            # ---- SEASON x DAY/NIGHT
            sum_sd[sidx, 0][day_valid] += vals[day_valid]
            cnt_sd[sidx, 0][day_valid] += 1

            sum_sd[sidx, 1][night_valid] += vals[night_valid]
            cnt_sd[sidx, 1][night_valid] += 1

            processed += 1

            if i % PRINT_EVERY == 0 or i == len(files):
                print(f"Processed {i}/{len(files)} files | valid={processed} | bad={len(bad_files)}")

        except Exception as e:
            bad_files.append((fp.name, str(e)))
            print(f"[BAD] {fp.name} -> {e}")

            if STOP_ON_FIRST_BAD_FILE:
                break

    # --------------------------------------------------------
    # Means
    # --------------------------------------------------------
    mean_all = np.divide(sum_all, cnt_all, out=np.full_like(sum_all, np.nan), where=cnt_all > 0)
    mean_season = np.divide(sum_season, cnt_season, out=np.full_like(sum_season, np.nan), where=cnt_season > 0)
    mean_dn = np.divide(sum_dn, cnt_dn, out=np.full_like(sum_dn, np.nan), where=cnt_dn > 0)
    mean_sd = np.divide(sum_sd, cnt_sd, out=np.full_like(sum_sd, np.nan), where=cnt_sd > 0)

    # --------------------------------------------------------
    # Save one compact climatology file
    # --------------------------------------------------------
    ds_out = xr.Dataset()

    ds_out["season"] = ("season", seasons)
    ds_out["day_night"] = ("day_night", dn_vals)
    ds_out["lev"] = ("lev", used_levels)

    if lat.ndim == 1 and lon.ndim == 1:
        ds_out["lat"] = ("lat", lat)
        ds_out["lon"] = ("lon", lon)

        ds_out[f"{VAR_NAME}_mean_all"] = xr.DataArray(
            mean_all.astype(np.float32),
            dims=("lev", "lat", "lon")
        )

        ds_out[f"{VAR_NAME}_mean_season"] = xr.DataArray(
            mean_season.astype(np.float32),
            dims=("season", "lev", "lat", "lon")
        )

        ds_out[f"{VAR_NAME}_mean_daynight"] = xr.DataArray(
            mean_dn.astype(np.float32),
            dims=("day_night", "lev", "lat", "lon")
        )

        ds_out[f"{VAR_NAME}_mean_season_daynight"] = xr.DataArray(
            mean_sd.astype(np.float32),
            dims=("season", "day_night", "lev", "lat", "lon")
        )

        ds_out[f"{VAR_NAME}_count_all"] = xr.DataArray(
            cnt_all,
            dims=("lev", "lat", "lon")
        )

        ds_out[f"{VAR_NAME}_count_season"] = xr.DataArray(
            cnt_season,
            dims=("season", "lev", "lat", "lon")
        )

        ds_out[f"{VAR_NAME}_count_daynight"] = xr.DataArray(
            cnt_dn,
            dims=("day_night", "lev", "lat", "lon")
        )

        ds_out[f"{VAR_NAME}_count_season_daynight"] = xr.DataArray(
            cnt_sd,
            dims=("season", "day_night", "lev", "lat", "lon")
        )

    else:
        ds_out["lat"] = (("lat", "lon"), lat)
        ds_out["lon"] = (("lat", "lon"), lon)

        ds_out[f"{VAR_NAME}_mean_all"] = xr.DataArray(
            mean_all.astype(np.float32),
            dims=("lev", "lat", "lon")
        )

        ds_out[f"{VAR_NAME}_mean_season"] = xr.DataArray(
            mean_season.astype(np.float32),
            dims=("season", "lev", "lat", "lon")
        )

        ds_out[f"{VAR_NAME}_mean_daynight"] = xr.DataArray(
            mean_dn.astype(np.float32),
            dims=("day_night", "lev", "lat", "lon")
        )

        ds_out[f"{VAR_NAME}_mean_season_daynight"] = xr.DataArray(
            mean_sd.astype(np.float32),
            dims=("season", "day_night", "lev", "lat", "lon")
        )

        ds_out[f"{VAR_NAME}_count_all"] = xr.DataArray(
            cnt_all,
            dims=("lev", "lat", "lon")
        )

        ds_out[f"{VAR_NAME}_count_season"] = xr.DataArray(
            cnt_season,
            dims=("season", "lev", "lat", "lon")
        )

        ds_out[f"{VAR_NAME}_count_daynight"] = xr.DataArray(
            cnt_dn,
            dims=("day_night", "lev", "lat", "lon")
        )

        ds_out[f"{VAR_NAME}_count_season_daynight"] = xr.DataArray(
            cnt_sd,
            dims=("season", "day_night", "lev", "lat", "lon")
        )

    ds_out.to_netcdf(OUT_FILE, encoding=make_encoding(ds_out))

    print(f"\nSaved climatology file: {OUT_FILE}")
    print(f"Processed valid files: {processed}")
    print(f"Skipped bad files: {len(bad_files)}")

    if bad_files:
        bad_report = OUT_DIR / "bad_nc_files_report.csv"
        pd.DataFrame(bad_files, columns=["file", "error"]).to_csv(bad_report, index=False)
        print(f"Bad-file report saved to: {bad_report}")


if __name__ == "__main__":
    start = time.time()
    print("\n===== PRODUCTION RUN =====")
    print("Start:", dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"))
    main()
    end = time.time()
    print("End:", dt.datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Execution time: {(end - start) / 60:.2f} minutes")