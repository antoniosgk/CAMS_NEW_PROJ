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

OUT_SEASON = OUT_DIR / "O3_mean_season.nc"
OUT_DAYNIGHT = OUT_DIR / "O3_mean_daynight.nc"

THRESHOLD_DEG = -0.833
PRINT_EVERY = 200


# ============================================================
# TIME PARSING
# ============================================================

def infer_timestamp(fp):
    m = re.search(r"(\d{8})[_-]?(\d{4})", fp.stem)
    if m:
        return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M", errors="coerce")
    return pd.NaT


def month_to_season(month):
    if month in [12,1,2]: return "DJF"
    if month in [3,4,5]: return "MAM"
    if month in [6,7,8]: return "JJA"
    return "SON"


# ============================================================
# SOLAR FUNCTION (VECTORIZED)
# ============================================================

def solar_day_mask(lat2d, lon2d, ts):
    lon2d = ((lon2d + 180) % 360) - 180
    lat_rad = np.deg2rad(lat2d)

    doy = ts.timetuple().tm_yday
    hour = ts.hour + ts.minute/60 + ts.second/3600

    gamma = 2*np.pi/365 * (doy - 1 + (hour - 12)/24)

    decl = (
        0.006918
        - 0.399912*np.cos(gamma)
        + 0.070257*np.sin(gamma)
        - 0.006758*np.cos(2*gamma)
        + 0.000907*np.sin(2*gamma)
        - 0.002697*np.cos(3*gamma)
        + 0.001480*np.sin(3*gamma)
    )

    eqtime = 229.18 * (
        0.000075
        + 0.001868*np.cos(gamma)
        - 0.032077*np.sin(gamma)
        - 0.014615*np.cos(2*gamma)
        - 0.040849*np.sin(2*gamma)
    )

    tst = (hour*60 + eqtime + 4*lon2d) % 1440
    ha = np.deg2rad(tst/4 - 180)

    cosz = np.sin(lat_rad)*np.sin(decl) + np.cos(lat_rad)*np.cos(decl)*np.cos(ha)
    cosz = np.clip(cosz, -1, 1)

    elev = np.rad2deg(np.pi/2 - np.arccos(cosz))

    return elev > THRESHOLD_DEG


# ============================================================
# MAIN
# ============================================================

def main():

    files = sorted(NC_DIR.glob(NC_GLOB))
    if not files:
        raise FileNotFoundError("No files found")

    # ---- inspect first file ----
    with xr.open_dataset(files[0]) as ds0:
        da0 = ds0[VAR_NAME]

        # remove singleton dims
        for d in list(da0.dims):
            if da0.sizes[d] == 1:
                da0 = da0.isel({d: 0})

        level_dim = [d for d in da0.dims if d not in ["lat","lon"]][0]

        lat = ds0["lat"].values
        lon = ds0["lon"].values

        lon2d, lat2d = np.meshgrid(lon, lat)

        nlev = da0.sizes[level_dim]
        ny, nx = lat2d.shape

    # ---- accumulators ----

    seasons = ["DJF","MAM","JJA","SON"]
    dn_vals = ["day","night"]

    sum_season = np.zeros((4,nlev,ny,nx))
    cnt_season = np.zeros((4,nlev,ny,nx))

    sum_dn = np.zeros((2,nlev,ny,nx))
    cnt_dn = np.zeros((2,nlev,ny,nx))

    season_map = {s:i for i,s in enumerate(seasons)}
    dn_map = {"day":0, "night":1}

    # ---- loop ----
    for i, fp in enumerate(files):

        ts = infer_timestamp(fp)
        if pd.isna(ts):
            print(f"Skipping {fp.name}")
            continue

        season = month_to_season(ts.month)
        sidx = season_map[season]

        with xr.open_dataset(fp) as ds:
            da = ds[VAR_NAME]

            for d in list(da.dims):
                if da.sizes[d] == 1:
                    da = da.isel({d:0})

            vals = da.values.astype(float)

        mask = np.isfinite(vals)

        # ---- SEASON ----
        sum_season[sidx][mask] += vals[mask]
        cnt_season[sidx][mask] += 1

        # ---- DAY/NIGHT ----
        mask_day = solar_day_mask(lat2d, lon2d, ts)
        mask_night = ~mask_day

        day_mask = mask & mask_day[None,:,:]
        night_mask = mask & mask_night[None,:,:]

        sum_dn[0][day_mask] += vals[day_mask]
        cnt_dn[0][day_mask] += 1

        sum_dn[1][night_mask] += vals[night_mask]
        cnt_dn[1][night_mask] += 1

        if (i+1) % PRINT_EVERY == 0:
            print(f"{i+1}/{len(files)} processed")

    # ---- means ----
    mean_season = sum_season / np.maximum(cnt_season,1)
    mean_dn = sum_dn / np.maximum(cnt_dn,1)

    # ---- SAVE ----
    ds_out = xr.Dataset()

    ds_out["season"] = ("season", seasons)
    ds_out["day_night"] = ("day_night", dn_vals)
    ds_out["lev"] = ("lev", range(nlev))
    ds_out["lat"] = ("lat", lat)
    ds_out["lon"] = ("lon", lon)

    ds_out["O3_mean_season"] = (("season","lev","lat","lon"), mean_season.astype(np.float32))
    ds_out["O3_mean_daynight"] = (("day_night","lev","lat","lon"), mean_dn.astype(np.float32))

    ds_out.to_netcdf(OUT_DIR/"O3_climatology.nc")

    print("DONE")


if __name__ == "__main__":
    start = time.time()
    print("\n===== PRODUCTION RUN =====")
    print("Start:", dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"))
    main()
    end = time.time()
    print("End:", dt.datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Execution time: {(end - start) / 60:.2f} minutes")
# %%
