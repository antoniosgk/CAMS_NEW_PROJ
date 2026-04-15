#%%
#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr

# ============================================================
# USER SETTINGS
# ============================================================

# Directory containing per-station parquet files
PARQUET_DIR = Path("/home/agkiokas/CAMS/stations_csv_parquet/")

# Directory where NetCDF files will be written
OUT_DIR = Path("/home/agkiokas/CAMS/stations_nc/")

# Optional species tag for output filenames
SPECIES = "O3"

# Station selection:
#   None              -> convert all stations found
#   [1001]            -> convert one station
#   [474, 732, 1001]  -> convert multiple stations
STATION_IDXS = None

# Skip unreadable parquet files instead of stopping
SKIP_BAD_FILES = True

# Summary parquet files are skipped automatically if filename contains this text
SUMMARY_TOKEN = "_summary"


# ============================================================
# DEFAULT VARIABLE GROUPS
# ============================================================

DEFAULT_2D_VARS = [
    "n_total", "n_valid", "n_excluded", "frac_excluded",
    "n", "mean", "std", "cv", "median", "q25", "q75", "iqr",
    "n_w", "mean_w", "std_w", "cv_w", "median_w", "q1_w", "q3_w", "iqr_w",
]

DEFAULT_1D_VARS = [
    "center_ppb", "k_star_center", "z_target_m",
]


# ============================================================
# HELPERS
# ============================================================

def sanitize_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^\w\-\.]+", "_", name)
    return name


def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif {"date", "time"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + df["time"].astype(str).str.zfill(4),
            format="%Y%m%d%H%M",
            errors="coerce",
        )
    else:
        raise ValueError(
            f"{path.name}: parquet must contain 'datetime' or both 'date' and 'time'."
        )

    required = {"station", "station_idx", "sector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns {sorted(missing)}")

    return df


def station_df_to_xarray(
    sdf: pd.DataFrame,
    two_d_vars=None,
    one_d_vars=None,
) -> xr.Dataset:
    if two_d_vars is None:
        two_d_vars = DEFAULT_2D_VARS
    if one_d_vars is None:
        one_d_vars = DEFAULT_1D_VARS

    sdf = sdf.copy()
    sdf = sdf.sort_values(["datetime", "sector"]).reset_index(drop=True)

    # sector ordering C1, C2, ...
    sectors = sorted(
        sdf["sector"].dropna().unique().tolist(),
        key=lambda s: int(str(s).replace("C", "")) if str(s).startswith("C") and str(s).replace("C", "").isdigit() else str(s)
    )

    times = np.array(sorted(sdf["datetime"].dropna().unique()), dtype="datetime64[ns]")

    ds = xr.Dataset(coords={
        "time": ("time", times),
        "sector": ("sector", np.array(sectors, dtype=object)),
    })

    # radius as sector coordinate if uniquely mapped
    if "radius" in sdf.columns:
        radius_map = (
            sdf[["sector", "radius"]]
            .dropna()
            .drop_duplicates()
            .sort_values("sector")
        )
        if radius_map["sector"].nunique() == len(sectors):
            radius_map = radius_map.set_index("sector").reindex(sectors)
            ds = ds.assign_coords(radius=("sector", radius_map["radius"].to_numpy(dtype=float)))

    # 2D variables: (time, sector)
    for var in two_d_vars:
        if var not in sdf.columns:
            continue

        piv = sdf.pivot(index="datetime", columns="sector", values=var)
        piv = piv.reindex(index=pd.to_datetime(times), columns=sectors)

        arr = piv.to_numpy()
        if np.issubdtype(arr.dtype, np.number):
            ds[var] = (("time", "sector"), arr.astype(float))

    # 1D variables: one value per time
    for var in one_d_vars:
        if var not in sdf.columns:
            continue

        tmp = (
            sdf[["datetime", var]]
            .dropna(subset=["datetime"])
            .groupby("datetime", as_index=False)[var]
            .first()
            .set_index("datetime")
            .reindex(pd.to_datetime(times))
        )

        arr = tmp[var].to_numpy()
        if np.issubdtype(arr.dtype, np.number):
            ds[var] = (("time",), arr.astype(float))
        else:
            ds[var] = (("time",), arr.astype(str))

    # season/day_night as time-dependent strings
    for var in ["season", "day_night"]:
        if var in sdf.columns:
            tmp = (
                sdf[["datetime", var]]
                .dropna(subset=["datetime"])
                .groupby("datetime", as_index=False)[var]
                .first()
                .set_index("datetime")
                .reindex(pd.to_datetime(times))
            )
            ds[var] = (("time",), tmp[var].fillna("").astype(str).to_numpy())

    # station metadata as global attrs
    meta_cols = [
        "station", "station_idx", "station_lat", "station_lon", "station_alt",
        "model_lat", "model_lon", "mode"
    ]
    for col in meta_cols:
        if col in sdf.columns:
            vals = sdf[col].dropna().unique()
            if len(vals) > 0:
                v = vals[0]
                ds.attrs[col] = v.item() if hasattr(v, "item") else v

    ds.attrs["created_from"] = "parquet"
    ds.attrs["dimensions"] = "time, sector"

    return ds


def write_station_netcdf(sdf: pd.DataFrame, out_dir: Path, species: str = "") -> Path:
    ds = station_df_to_xarray(sdf)

    station_name = str(ds.attrs.get("station", "station"))
    station_idx = ds.attrs.get("station_idx", "unknown")
    mode = str(ds.attrs.get("mode", "A"))

    safe_station = sanitize_name(station_name)

    parts = [safe_station, f"idx{station_idx}"]
    if species:
        parts.append(species)
    parts.append(mode)

    out_path = out_dir / ("_".join(map(str, parts)) + ".nc")

    encoding = {}
    for var in ds.data_vars:
        if ds[var].dtype.kind in {"f", "i", "u"}:
            encoding[var] = {"zlib": True, "complevel": 4}

    ds.to_netcdf(out_path, encoding=encoding)
    ds.close()
    return out_path


def is_big_station_parquet(path: Path) -> bool:
    """
    Keep only the big parquet, skip:
    - non-parquet files
    - summary parquet files
    """
    if path.suffix.lower() != ".parquet":
        return False

    if SUMMARY_TOKEN.lower() in path.name.lower():
        return False

    return True


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PARQUET_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {PARQUET_DIR}")

    files = sorted(PARQUET_DIR.iterdir())

    outputs = []
    n_skipped_non_parquet = 0
    n_skipped_summary = 0
    n_skipped_bad = 0
    n_skipped_empty = 0

    for path in files:
        if path.suffix.lower() != ".parquet":
            print(f"Skipping non-parquet file: {path.name}")
            n_skipped_non_parquet += 1
            continue

        if SUMMARY_TOKEN.lower() in path.name.lower():
            print(f"Skipping summary parquet: {path.name}")
            n_skipped_summary += 1
            continue

        try:
            df = load_parquet(path)
        except Exception as e:
            if SKIP_BAD_FILES:
                print(f"Skipping unreadable parquet: {path.name} | error: {e}")
                n_skipped_bad += 1
                continue
            raise

        # filter requested station(s)
        if STATION_IDXS is not None:
            df = df[df["station_idx"].isin(STATION_IDXS)].copy()

        if df.empty:
            print(f"No matching rows in: {path.name}")
            n_skipped_empty += 1
            continue

        # since you have one parquet per station, this should usually be one group
        for station_idx, sdf in df.groupby("station_idx", sort=True):
            if sdf.empty:
                continue

            out_path = write_station_netcdf(
                sdf=sdf,
                out_dir=OUT_DIR,
                species=SPECIES,
            )
            outputs.append(out_path)
            print(f"Saved: {out_path}")

    print("\nDone.")
    print(f"Created NetCDF files: {len(outputs)}")
    print(f"Skipped non-parquet files: {n_skipped_non_parquet}")
    print(f"Skipped summary parquet files: {n_skipped_summary}")
    print(f"Skipped unreadable parquet files: {n_skipped_bad}")
    print(f"Skipped files with no matching station rows: {n_skipped_empty}")


if __name__ == "__main__":
    main()
# %%
