from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import re
import dask


# ============================================================
# USER SETTINGS
# ============================================================

NC_DIR = Path("/mnt/store01/agkiokas/CAMS/inst/subsets/O3/")
NC_GLOB = "*.nc*"
VAR_NAME = "O3"

OUT_ZARR = Path("/mnt/store01/agkiokas/CAMS/O3_reduced_archive.zarr")
# metadata files
CONST_K_FILE = Path("/home/agkiokas/CAMS/lookups/station_constant_k_only_lookup.csv")
K_TS_FILE = Path("/home/agkiokas/CAMS/lookups/station_level_timeseries_all.parquet")
K_DECISION_FILE = Path("/home/agkiokas/CAMS/lookups/station_k_decision_table.csv")

STATION_LIST = None
# Example:
# STATION_LIST = ["1001A", "1006A", "2209A"]

# chunking
CHUNKS = {"time": 48, "lev": 1, "lat": 256, "lon": 256}

# Dask scheduler: "threads", "processes", or "single-threaded"
DASK_SCHEDULER = "threads"


# ============================================================
# LEVEL SELECTION
# ============================================================

def get_required_levels():
    df_const = pd.read_csv(CONST_K_FILE)
    df_decision = pd.read_csv(K_DECISION_FILE)
    df_ts = pd.read_parquet(K_TS_FILE)

    if STATION_LIST is not None:
        wanted = set(STATION_LIST)
        if "Station_Name" in df_const.columns:
            df_const = df_const[df_const["Station_Name"].isin(wanted)]
        if "Station_Name" in df_decision.columns:
            df_decision = df_decision[df_decision["Station_Name"].isin(wanted)]
        if "Station_Name" in df_ts.columns:
            df_ts = df_ts[df_ts["Station_Name"].isin(wanted)]

    const_levels = set(df_const["level_idx"].dropna().astype(int).unique())

    variable_ids = df_decision[df_decision["k_min"] != df_decision["k_max"]]["idx"].values
    df_var = df_ts[df_ts["idx"].isin(variable_ids)]
    var_levels = set(df_var["level_idx"].dropna().astype(int).unique())

    all_levels = sorted(const_levels.union(var_levels))

    print("Constant levels:", sorted(const_levels))
    print("Variable levels:", sorted(var_levels))
    print("Final levels:", all_levels)

    return all_levels


# ============================================================
# HELPERS
# ============================================================

def infer_timestamp_from_name(path: Path):
    m = re.search(r"(\d{8})[_-]?(\d{4})", path.stem)
    if m:
        return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M", errors="coerce")
    return pd.NaT


def guess_level_dim(da):
    for d in ["lev", "level", "plev", "layer", "z", "alt"]:
        if d in da.dims:
            return d
    raise ValueError(f"Could not find vertical dimension in {da.dims}")


def guess_lat_lon_names(ds):
    lat_name = next((x for x in ["lat", "latitude", "Latitude"] if x in ds.coords or x in ds.variables), None)
    lon_name = next((x for x in ["lon", "longitude", "Longitude"] if x in ds.coords or x in ds.variables), None)
    if lat_name is None or lon_name is None:
        raise ValueError("Could not identify lat/lon names")
    return lat_name, lon_name


def preprocess_one_file(ds, var_name, levels):
    if var_name not in ds.variables:
        raise ValueError(f"{var_name} not found")

    da = ds[var_name]

    # remove singleton dims except lat/lon/lev
    for dim in list(da.dims):
        if ds[var_name].sizes[dim] == 1 and dim not in ["lat", "lon", "latitude", "longitude", "lev", "level", "plev"]:
            da = da.isel({dim: 0})

    level_dim = guess_level_dim(da)
    da = da.isel({level_dim: levels})

    # normalize level dim name to "lev"
    if level_dim != "lev":
        da = da.rename({level_dim: "lev"})

    return da.to_dataset(name=var_name)


# ============================================================
# MAIN
# ============================================================

def build_reduced_zarr():
    levels = get_required_levels()

    files = sorted(NC_DIR.glob(NC_GLOB))
    if not files:
        raise FileNotFoundError(f"No files found in {NC_DIR} matching {NC_GLOB}")

    datasets = []

    for fp in files:
        try:
            ds = xr.open_dataset(fp, chunks={})
            ds_proc = preprocess_one_file(ds, VAR_NAME, levels)

            if "time" not in ds_proc.coords:
                ts = infer_timestamp_from_name(fp)
                if pd.isna(ts):
                    raise ValueError(f"Could not infer time from filename {fp.name}")
                ds_proc = ds_proc.expand_dims(time=[ts])

            datasets.append(ds_proc)

        except Exception as e:
            print(f"Skipping {fp.name}: {e}")

    if not datasets:
        raise RuntimeError("No valid datasets were prepared")

    combined = xr.concat(datasets, dim="time").sortby("time")

    # normalize chunking
    chunk_dict = {k: v for k, v in CHUNKS.items() if k in combined[VAR_NAME].dims}
    combined = combined.chunk(chunk_dict)

    print(combined)
    print("Writing Zarr archive...")

    with dask.config.set(scheduler=DASK_SCHEDULER):
        combined.to_zarr(OUT_ZARR, mode="w")

    print(f"Saved reduced archive to {OUT_ZARR}")


if __name__ == "__main__":
    build_reduced_zarr()