#%%
import math
from pathlib import Path
from typing import Optional, Iterable
import time
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# ============================================================
# USER SETTINGS
# ============================================================
#STATION_LIST = None
# Example:
STATION_LIST = ["1001A", "1002A"]
MAP_EXTENT = [110, 130, 30, 45]   # [lon_min, lon_max, lat_min, lat_max] or None
PARQUET_DIR = Path("/home/agkiokas/CAMS/stations_csv_parquet")
NC_DIR = Path("/mnt/store01/agkiokas/CAMS/inst/subsets/O3/")
NC_GLOB = "*.nc4"                     # e.g. "*.nc4" or "*.nc"
FIELD_VAR = "O3"                      # change to your model variable name

OUT_DIR = Path("/home/agkiokas/CAMS/climatological_plots/")
#FIELD_LEVEL_INDEX = 22   # choose the vertical level index you want
# General filters for parquet-based approaches
MODE_FILTER = "A"                     # e.g. "A" or None
DAY_NIGHT_FILTER = None               # "day", "night", or None
SEASON_FILTER = None                  # "Winter", "Spring", ... or None

# Approaches 3 and 5 controls
A35_SECTOR_TYPE = "CUM"               # e.g. "CUM"
A35_SECTOR = "C1"                     # e.g. "C1"; set None to aggregate all sectors
A35_AGGREGATE_OVER_SECTORS = False    # True => average over sectors per timestamp

# Approach 4 controls
A4_SECTOR_TYPE = "CUM"
A4_SECTOR = "C1"
A4_WEIGHTED = True                    # weight by cos(lat)
A4_USE_RADIUS_FROM_PARQUET = True     # radius for chosen sector is taken from parquet

# Plot settings
FIGSIZE = (10, 6)
LABEL_STATIONS = True


# ============================================================
# HELPERS
# ============================================================

def load_station_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # ensure timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif {"date", "time"}.issubset(df.columns):
        date_str = df["date"].astype(str).str.zfill(8)
        time_str = df["time"].astype(str).str.zfill(4)
        df["timestamp"] = pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M", errors="coerce")
    else:
        raise ValueError(f"{path.name}: no timestamp/date+time found")

    return df


def apply_common_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if MODE_FILTER is not None and "mode" in out.columns:
        out = out[out["mode"] == MODE_FILTER]

    if DAY_NIGHT_FILTER is not None and "day_night" in out.columns:
        out = out[out["day_night"] == DAY_NIGHT_FILTER]

    if SEASON_FILTER is not None and "season" in out.columns:
        out = out[out["season"] == SEASON_FILTER]

    out = out.dropna(subset=["timestamp"]).copy()
    return out


def get_station_meta(df: pd.DataFrame) -> dict:
    row = df.iloc[0]
    return {
        "station": row["station"],
        "lat": float(row["station_lat"]),
        "lon": float(row["station_lon"]),
        "alt": float(row["station_alt"]),
        "i_center": int(row["i_center"]),
        "j_center": int(row["j_center"]),
        "k_star_center": int(row["k_star_center"]) if pd.notna(row["k_star_center"]) else np.nan,
    }


def get_center_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    center_ppb repeats across sectors for the same timestamp, so reduce to one row per timestamp.
    """
    out = (
        df.dropna(subset=["timestamp", "center_ppb"])
          .groupby("timestamp", as_index=False)["center_ppb"]
          .mean()
          .sort_values("timestamp")
    )
    return out


def get_sector_stat_timeseries(
    df: pd.DataFrame,
    value_cols: Iterable[str],
    sector_type: Optional[str],
    sector: Optional[str],
    aggregate_over_sectors: bool,
) -> pd.DataFrame:
    out = df.copy()

    if sector_type is not None and "sector_type" in out.columns:
        out = out[out["sector_type"] == sector_type]

    out = out.dropna(subset=["timestamp"]).copy()

    if sector is not None:
        out = out[out["sector"] == sector].copy()
    elif aggregate_over_sectors:
        agg_map = {c: "mean" for c in value_cols}
        out = out.groupby("timestamp", as_index=False).agg(agg_map)

    cols = ["timestamp"] + list(value_cols)
    cols = [c for c in cols if c in out.columns]
    out = out[cols].dropna()
    return out.sort_values("timestamp")


def weighted_mean_and_std(values: np.ndarray, weights: Optional[np.ndarray] = None) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v)
    v = v[mask]

    if v.size == 0:
        return np.nan, np.nan

    if weights is None:
        return float(np.mean(v)), float(np.std(v, ddof=0))

    w = np.asarray(weights, dtype=float)[mask]
    if w.size == 0 or np.sum(w) == 0:
        return np.nan, np.nan

    mean = np.sum(w * v) / np.sum(w)
    var = np.sum(w * (v - mean) ** 2) / np.sum(w)
    return float(mean), float(np.sqrt(var))


def plot_station_map(
    df_summary: pd.DataFrame,
    metric_col: str,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
    outfile: Optional[Path] = None,
    extent: Optional[list[float]] = None,
):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=FIGSIZE)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.2)

        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        sc = ax.scatter(
            df_summary["lon"],
            df_summary["lat"],
            c=df_summary[metric_col],
            cmap=cmap,
            s=70,
            edgecolor="k",
            transform=ccrs.PlateCarree(),
        )

        if LABEL_STATIONS:
            for _, r in df_summary.iterrows():
                ax.text(
                    r["lon"], r["lat"], str(r["station"]),
                    fontsize=8, transform=ccrs.PlateCarree()
                )

        ax.set_title(title)
        plt.colorbar(sc, ax=ax, label=cbar_label)

    except Exception:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sc = ax.scatter(
            df_summary["lon"],
            df_summary["lat"],
            c=df_summary[metric_col],
            cmap=cmap,
            s=70,
            edgecolor="k",
        )
        if LABEL_STATIONS:
            for _, r in df_summary.iterrows():
                ax.text(r["lon"], r["lat"], str(r["station"]), fontsize=8)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        ax.set_title(title)
        plt.colorbar(sc, ax=ax, label=cbar_label)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.show()

# ============================================================
# APPROACHES 1, 2, 3, 5 FROM PARQUET
# ============================================================

def compute_station_metrics_from_parquet(path: Path) -> dict:
    df = load_station_parquet(path)
    df = apply_common_filters(df)

    if df.empty:
        raise ValueError(f"{path.name}: no data after filters")

    meta = get_station_meta(df)

    # Approach 1: temporal mean of center_ppb
    center_ts = get_center_timeseries(df)
    center_mean_time = float(center_ts["center_ppb"].mean())

    # Approach 2: temporal CV of center_ppb
    center_std_time = float(center_ts["center_ppb"].std(ddof=0))
    center_cv_time_pct = float(100.0 * center_std_time / center_mean_time) if center_mean_time != 0 else np.nan

    # Approach 3: mean(std_w over time) / mean(mean_w over time)
    s35 = get_sector_stat_timeseries(
        df,
        value_cols=["mean_w", "std_w"],
        sector_type=A35_SECTOR_TYPE,
        sector=A35_SECTOR,
        aggregate_over_sectors=A35_AGGREGATE_OVER_SECTORS,
    )
    mean_of_meanw = float(s35["mean_w"].mean()) if not s35.empty else np.nan
    mean_of_stdw = float(s35["std_w"].mean()) if not s35.empty else np.nan
    approach3_ratio_pct = float(100.0 * mean_of_stdw / mean_of_meanw) if pd.notna(mean_of_meanw) and mean_of_meanw != 0 else np.nan

    # Approach 5: time average of cv_w
    s5 = get_sector_stat_timeseries(
        df,
        value_cols=["cv_w"],
        sector_type=A35_SECTOR_TYPE,
        sector=A35_SECTOR,
        aggregate_over_sectors=A35_AGGREGATE_OVER_SECTORS,
    )
    approach5_mean_cvw_pct = float(100.0 * s5["cv_w"].mean()) if not s5.empty else np.nan

    out = {
        **meta,
        "center_mean_time_ppb": center_mean_time,
        "center_temporal_cv_pct": center_cv_time_pct,
        "approach3_ratio_pct": approach3_ratio_pct,
        "approach5_mean_cvw_pct": approach5_mean_cvw_pct,
    }

    # store sector radius for approach 4
    if A4_USE_RADIUS_FROM_PARQUET:
        a4 = df.copy()
        if A4_SECTOR_TYPE is not None and "sector_type" in a4.columns:
            a4 = a4[a4["sector_type"] == A4_SECTOR_TYPE]
        if A4_SECTOR is not None and "sector" in a4.columns:
            a4 = a4[a4["sector"] == A4_SECTOR]
        if not a4.empty and "radius" in a4.columns:
            out["a4_radius"] = int(a4["radius"].dropna().iloc[0])
        else:
            out["a4_radius"] = np.nan

    return out


# ============================================================
# APPROACH 4 FROM NETCDF
# ============================================================

def guess_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_candidates = ["lat", "latitude", "Latitude"]
    lon_candidates = ["lon", "longitude", "Longitude"]

    lat_name = next((x for x in lat_candidates if x in ds.coords or x in ds.variables), None)
    lon_name = next((x for x in lon_candidates if x in ds.coords or x in ds.variables), None)

    if lat_name is None or lon_name is None:
        raise ValueError("Could not identify lat/lon names in dataset")

    return lat_name, lon_name




def extract_2d_field(ds: xr.Dataset, var_name: str, level_index: Optional[int] = None) -> xr.DataArray:
    da = ds[var_name]

    # already 2D
    if da.ndim == 2:
        return da

    # squeeze singleton dims
    for dim in list(da.dims):
        if da.sizes[dim] == 1:
            da = da.isel({dim: 0})

    if da.ndim == 2:
        return da

    # handle vertical dimension
    if da.ndim == 3:
        level_dim_candidates = ["lev", "level", "plev", "layer", "alt", "z"]
        level_dim = next((d for d in level_dim_candidates if d in da.dims), None)

        if level_dim is None:
            non_xy_dims = [d for d in da.dims if d.lower() not in ["lat", "latitude", "lon", "longitude"]]
            if len(non_xy_dims) == 1:
                level_dim = non_xy_dims[0]

        if level_dim is None:
            raise ValueError(
                f"Variable {var_name} is 3D with dims {da.dims}, but vertical dimension could not be identified"
            )

        if level_index is None:
            raise ValueError(
                f"Variable {var_name} is 3D with dims {da.dims}. A station-specific level index is required."
            )

        if level_index < 0 or level_index >= da.sizes[level_dim]:
            raise ValueError(
                f"Requested level index {level_index} is out of bounds for dim {level_dim} with size {da.sizes[level_dim]}"
            )

        da = da.isel({level_dim: int(level_index)})

        if da.ndim == 2:
            return da

    raise ValueError(
        f"Variable {var_name} is not reducible to 2D automatically. Current dims: {da.dims}."
    )




def extract_cumulative_square_window(arr2d: np.ndarray, i_center: int, j_center: int, radius: int) -> np.ndarray:
    i0 = max(0, int(i_center) - int(radius))
    i1 = min(arr2d.shape[0], int(i_center) + int(radius) + 1)
    j0 = max(0, int(j_center) - int(radius))
    j1 = min(arr2d.shape[1], int(j_center) + int(radius) + 1)
    return arr2d[i0:i1, j0:j1]


def list_nc_files(nc_dir: Path, pattern: str) -> list[Path]:
    files = sorted(nc_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {nc_dir / pattern}")
    return files

def build_level_to_stations(summary: pd.DataFrame) -> dict[int, list[int]]:
    """
    Returns dict:
        level_index -> list of row indices in summary
    """
    level_map = {}
    for idx, row in summary.iterrows():
        if pd.isna(row.get("k_star_center", np.nan)):
            continue
        k = int(row["k_star_center"])
        level_map.setdefault(k, []).append(idx)
    return level_map


def compute_climatological_mean_map_for_level(
    nc_files: list[Path],
    var_name: str,
    level_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute one climatological 2D mean map for a given vertical level.
    Reused for all stations sharing the same k_star_center.
    """
    sum_field = None
    count_field = None
    lat2d = None
    lon2d = None

    for fp in nc_files:
        with xr.open_dataset(fp) as ds:
            lat_name, lon_name = guess_lat_lon_names(ds)
            da2d = extract_2d_field(ds, var_name, level_index=level_index)

            vals = np.asarray(da2d.values, dtype=float)

            lat = np.asarray(ds[lat_name].values)
            lon = np.asarray(ds[lon_name].values)

            if lat.ndim == 1 and lon.ndim == 1:
                lon2d_, lat2d_ = np.meshgrid(lon, lat)
            else:
                lat2d_, lon2d_ = lat, lon

            if sum_field is None:
                sum_field = np.zeros_like(vals, dtype=float)
                count_field = np.zeros_like(vals, dtype=float)
                lat2d = lat2d_
                lon2d = lon2d_

            mask = np.isfinite(vals)
            sum_field[mask] += vals[mask]
            count_field[mask] += 1.0

    clim_mean = np.divide(
        sum_field,
        count_field,
        out=np.full_like(sum_field, np.nan, dtype=float),
        where=count_field > 0
    )

    return clim_mean, lat2d, lon2d


def compute_approach4_grouped_by_level(
    summary: pd.DataFrame,
    nc_files: list[Path],
) -> pd.Series:
    """
    Fast Approach 4:
    compute one climatological mean map per unique k_star_center,
    then evaluate all stations that belong to that level.
    """
    result = pd.Series(np.nan, index=summary.index, dtype=float)

    level_map = build_level_to_stations(summary)

    for k_level, row_indices in level_map.items():
        print(f"Approach 4: computing climatological map for level {k_level} ({len(row_indices)} stations)")

        clim_mean_map, lat2d, lon2d = compute_climatological_mean_map_for_level(
            nc_files=nc_files,
            var_name=FIELD_VAR,
            level_index=k_level
        )

        for idx in row_indices:
            row = summary.loc[idx]

            if pd.isna(row.get("a4_radius", np.nan)):
                result.loc[idx] = np.nan
                continue

            radius = int(row["a4_radius"])
            i_center = int(row["i_center"])
            j_center = int(row["j_center"])

            vals = extract_cumulative_square_window(clim_mean_map, i_center, j_center, radius)

            if A4_WEIGHTED:
                lat_win = extract_cumulative_square_window(lat2d, i_center, j_center, radius)
                weights = np.cos(np.deg2rad(lat_win))
                mean_, std_ = weighted_mean_and_std(vals, weights)
            else:
                mean_, std_ = weighted_mean_and_std(vals, None)

            if pd.isna(mean_) or mean_ == 0:
                result.loc[idx] = np.nan
            else:
                result.loc[idx] = 100.0 * std_ / mean_

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    start = time.time()
    print("Start:", dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"))
    parquet_files = sorted([p for p in PARQUET_DIR.glob("*.parquet") if "_summary" not in p.name])

    if STATION_LIST is not None:
        wanted = set(STATION_LIST)
        parquet_files = [p for p in parquet_files if p.stem.split("_")[0] in wanted]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {PARQUET_DIR}")

    rows = []
    for fp in parquet_files:
        try:
            rows.append(compute_station_metrics_from_parquet(fp))
        except Exception as e:
            print(f"Skipping {fp.name}: {e}")

    if not rows:
        raise RuntimeError("No station summaries were computed")

    summary = pd.DataFrame(rows).sort_values("station").reset_index(drop=True)

    # Approach 4 from NC files
    try:
        nc_files = list_nc_files(NC_DIR, NC_GLOB)
        summary["approach4_clim_map_cv_pct"] = compute_approach4_grouped_by_level(
            summary=summary,
            nc_files=nc_files
        )
    except Exception as e:
        print(f"Approach 4 not computed: {e}")
        summary["approach4_clim_map_cv_pct"] = np.nan

    summary.to_csv(OUT_DIR / "station_climatology_summary.csv", index=False)
    print(summary)

    # Maps
    plot_station_map(
        summary,
        metric_col="center_mean_time_ppb",
        title="Approach 1: Temporal mean of center_ppb",
        cbar_label="ppb",
        cmap="viridis",extent=MAP_EXTENT,
        outfile=OUT_DIR / "map_approach1_center_mean.png",
    )

    plot_station_map(
        summary,
        metric_col="center_temporal_cv_pct",
        title="Approach 2: Temporal CV of center_ppb",
        cbar_label="CV (%)",
        cmap="magma",extent=MAP_EXTENT,
        outfile=OUT_DIR / "map_approach2_center_temporal_cv.png",
    )

    plot_station_map(
        summary,
        metric_col="approach3_ratio_pct",
        title=f"Approach 3: mean(std_w)/mean(mean_w) | sector_type={A35_SECTOR_TYPE}, sector={A35_SECTOR}",
        cbar_label="Ratio (%)",
        cmap="plasma",extent=MAP_EXTENT,
        outfile=OUT_DIR / "map_approach3_ratio.png",
    )

    plot_station_map(
        summary,
        metric_col="approach5_mean_cvw_pct",
        title=f"Approach 5: time-mean cv_w | sector_type={A35_SECTOR_TYPE}, sector={A35_SECTOR}",
        cbar_label="CV (%)",
        cmap="inferno",extent=MAP_EXTENT,
        outfile=OUT_DIR / "map_approach5_mean_cvw.png",
    )
    
    if summary["approach4_clim_map_cv_pct"].notna().any():
        plot_station_map(
            summary,
            metric_col="approach4_clim_map_cv_pct",
            title=f"Approach 4: CV from climatological mean map | sector_type={A4_SECTOR_TYPE}, sector={A4_SECTOR}",
            cbar_label="CV (%)",
            cmap="cividis",extent=MAP_EXTENT,
            outfile=OUT_DIR / "map_approach4_clim_map_cv.png",
        )

    end = time.time()
    print("End:", dt.datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Execution time: {(end - start) / 60:.2f} minutes")
if __name__ == "__main__":
    main()
# %%
