#%%
from pathlib import Path
import numpy as np
import xarray as xr
import time
import datetime as dt

# ============================================================
# USER SETTINGS
# ============================================================

NC_DIR = Path("/mnt/store01/agkiokas/CAMS/inst/subsets/O3/")
NC_GLOB = "*.nc*"                  # e.g. "*.nc4"
VAR_NAME = "O3"                    # species variable name in the nc files

OUT_FILE = Path("/mnt/store01/agkiokas/CAMS/O3_temporal_mean.nc")

# If True, keep the variable exactly as it exists after removing any singleton time dim.
# This is the safest choice for your use case.
KEEP_VERTICAL_DIM = True

# Optional chunk size for progress printing
PRINT_EVERY = 200


# ============================================================
# HELPERS
# ============================================================

def guess_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_candidates = ["lat", "latitude", "Latitude"]
    lon_candidates = ["lon", "longitude", "Longitude"]

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


def open_first_valid_file(files: list[Path], var_name: str) -> tuple[xr.Dataset, xr.DataArray]:
    for fp in files:
        with xr.open_dataset(fp) as ds:
            if var_name in ds.variables:
                da = remove_singleton_dims(ds[var_name])
                return ds, da
    raise ValueError(f"Variable {var_name} not found in any files")


# ============================================================
# MAIN
# ============================================================

def main():
    start = time.time()
    print("Start:", dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"))
    files = sorted(NC_DIR.glob(NC_GLOB))
    if not files:
        raise FileNotFoundError(f"No files found in {NC_DIR} matching {NC_GLOB}")

    # Inspect first valid file to determine variable shape/dims
    with xr.open_dataset(files[0]) as ds0:
        if VAR_NAME not in ds0.variables:
            raise ValueError(f"{VAR_NAME} not found in first file: {files[0].name}")

        da0 = remove_singleton_dims(ds0[VAR_NAME])

        # Expected dims after removing singleton time dim:
        # either (lev, lat, lon) or (lat, lon)
        if da0.ndim not in [2, 3]:
            raise ValueError(
                f"Unsupported ndim for {VAR_NAME}: {da0.ndim}, dims={da0.dims}. "
                "Expected 2D or 3D after removing singleton dims."
            )

        data_shape = da0.shape
        data_dims = da0.dims
        data_dtype = np.float64

        # keep coordinate variables if possible
        coords_to_keep = {}
        for name in da0.dims:
            if name in ds0.coords:
                coords_to_keep[name] = ds0.coords[name].values
            elif name in ds0.variables:
                coords_to_keep[name] = ds0[name].values

        # also keep lat/lon if they exist separately
        try:
            lat_name, lon_name = guess_lat_lon_names(ds0)
            if lat_name not in coords_to_keep and lat_name in ds0:
                coords_to_keep[lat_name] = ds0[lat_name].values
            if lon_name not in coords_to_keep and lon_name in ds0:
                coords_to_keep[lon_name] = ds0[lon_name].values
        except Exception:
            lat_name, lon_name = None, None

    print(f"Found {len(files)} files")
    print(f"Variable: {VAR_NAME}")
    print(f"Dims after squeeze: {data_dims}")
    print(f"Shape: {data_shape}")

    sum_arr = np.zeros(data_shape, dtype=data_dtype)
    count_arr = np.zeros(data_shape, dtype=np.int64)

    for idx, fp in enumerate(files, start=1):
        try:
            with xr.open_dataset(fp) as ds:
                if VAR_NAME not in ds.variables:
                    print(f"Skipping {fp.name}: variable {VAR_NAME} not found")
                    continue

                da = remove_singleton_dims(ds[VAR_NAME])

                if da.dims != data_dims or da.shape != data_shape:
                    print(
                        f"Skipping {fp.name}: incompatible shape/dims. "
                        f"Expected {data_dims} {data_shape}, got {da.dims} {da.shape}"
                    )
                    continue

                vals = np.asarray(da.values, dtype=np.float64)
                mask = np.isfinite(vals)

                sum_arr[mask] += vals[mask]
                count_arr[mask] += 1

                if idx % PRINT_EVERY == 0 or idx == len(files):
                    print(f"Processed {idx}/{len(files)} files")

        except Exception as e:
            print(f"Skipping {fp.name}: {e}")

    mean_arr = np.divide(
        sum_arr,
        count_arr,
        out=np.full_like(sum_arr, np.nan, dtype=np.float64),
        where=count_arr > 0
    )

    # Build output dataset
    out_ds = xr.Dataset()

    out_ds[f"{VAR_NAME}_temporal_mean"] = xr.DataArray(
        mean_arr,
        dims=data_dims,
        coords={dim: coords_to_keep[dim] for dim in data_dims if dim in coords_to_keep},
        attrs={
            "long_name": f"Temporal mean of {VAR_NAME}",
            "source_files": str(len(files)),
            "note": "Computed as arithmetic mean over timestep files"
        }
    )

    out_ds[f"{VAR_NAME}_valid_count"] = xr.DataArray(
        count_arr,
        dims=data_dims,
        coords={dim: coords_to_keep[dim] for dim in data_dims if dim in coords_to_keep},
        attrs={
            "long_name": f"Number of valid samples used for temporal mean of {VAR_NAME}"
        }
    )

    # Optionally keep explicit lat/lon variables if they were not dims
    for name, vals in coords_to_keep.items():
        if name not in out_ds.coords and name not in out_ds.variables:
            try:
                out_ds[name] = xr.DataArray(vals)
            except Exception:
                pass

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(OUT_FILE)

    print(f"Saved temporal mean file to: {OUT_FILE}")
    end = time.time()
    print("End:", dt.datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Execution time: {(end - start) / 60:.2f} minutes")

if __name__ == "__main__":
    main()
# %%
