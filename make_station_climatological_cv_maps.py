#%%
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

PARQUET_DIR = Path("/path/to/full_station_parquets")
STATION_LIST = ["1001A", "1006A", "2209A"]   # set None for all

# choose which climatology to use
CLIM_KIND = "all"        # "all", "season", "day_night"
SEASON_NAME = "Winter"   # used if CLIM_KIND == "season"
DAY_NIGHT_NAME = "day"   # used if CLIM_KIND == "day_night"

MEAN_ALL_FILE = Path("/path/to/output_climatologies/O3_mean_all.nc")
MEAN_SEASON_FILE = Path("/path/to/output_climatologies/O3_mean_season.nc")
MEAN_DAYNIGHT_FILE = Path("/path/to/output_climatologies/O3_mean_daynight.nc")

VAR_ALL = "O3_mean_all"
VAR_SEASON = "O3_mean_season"
VAR_DAYNIGHT = "O3_mean_daynight"

# sectors to map
SECTORS = ["C1", "C2", "C3", "C4", "C5", "C6","C7", "C8","C9","C10"]

MAP_EXTENT = None
# Example: [70, 140, 15, 55]

OUT_DIR = Path("/path/to/output_station_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_STATIONS = True
WEIGHTED = True
FIGSIZE = (10, 6)


# ============================================================
# HELPERS
# ============================================================

def sector_to_radius(sector: str) -> int:
    # C1 -> 1 => 3x3, C2 -> 2 => 5x5, ...
    digits = "".join(ch for ch in str(sector) if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse sector name: {sector}")
    return int(digits)


def weighted_mean_and_std(values: np.ndarray, weights: np.ndarray | None = None):
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


def extract_window_bounds(i_center: int, j_center: int, radius: int, shape: tuple[int, int]):
    ny, nx = shape
    i0 = max(0, int(i_center) - int(radius))
    i1 = min(ny, int(i_center) + int(radius) + 1)
    j0 = max(0, int(j_center) - int(radius))
    j1 = min(nx, int(j_center) + int(radius) + 1)
    return i0, i1, j0, j1


def extract_window(arr2d: np.ndarray, bounds):
    i0, i1, j0, j1 = bounds
    return arr2d[i0:i1, j0:j1]


def load_station_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif {"date", "time"}.issubset(df.columns):
        date_str = df["date"].astype(str).str.zfill(8)
        time_str = df["time"].astype(str).str.zfill(4)
        df["timestamp"] = pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M", errors="coerce")

    return df


def get_station_files(parquet_dir: Path, station_list=None):
    files = sorted([p for p in parquet_dir.glob("*.parquet") if "_summary" not in p.name])

    if station_list is not None:
        wanted = set(station_list)
        files = [p for p in files if p.stem.split("_")[0] in wanted]

    return files


def choose_station_level_from_context(df: pd.DataFrame) -> int:
    out = df

    if CLIM_KIND == "season":
        if "season" not in out.columns:
            raise ValueError("Station parquet does not contain 'season'")
        out = out[out["season"] == SEASON_NAME]

    elif CLIM_KIND == "day_night":
        if "day_night" not in out.columns:
            raise ValueError("Station parquet does not contain 'day_night'")
        out = out[out["day_night"] == DAY_NIGHT_NAME]

    level_col = "k_star_center" if "k_star_center" in out.columns else "level_idx"

    vals = out[level_col].dropna().astype(int)
    if vals.empty:
        raise ValueError("No valid station level values after context filtering")

    # mode is robust for variable-k stations
    return int(vals.mode().iloc[0])

def compute_station_sector_stats_from_field(
    field2d: np.ndarray,
    lat2d: np.ndarray,
    i_center: int,
    j_center: int,
    sectors: list[str],
    weighted: bool = True,
):
    """
    Compute mean/std/cv for multiple cumulative square sectors around one station
    from one already-selected 2D climatological field.
    """
    out_rows = []

    for sector in sectors:
        radius = sector_to_radius(sector)
        bounds = extract_window_bounds(i_center, j_center, radius, field2d.shape)

        vals = extract_window(field2d, bounds)

        if weighted:
            lat_win = extract_window(lat2d, bounds)
            weights = np.cos(np.deg2rad(lat_win))
            mean_, std_ = weighted_mean_and_std(vals, weights)
        else:
            mean_, std_ = weighted_mean_and_std(vals, None)

        cv_pct = 100.0 * std_ / mean_ if pd.notna(mean_) and mean_ != 0 else np.nan

        out_rows.append({
            "sector": sector,
            "radius": radius,
            "mean": mean_,
            "std": std_,
            "cv_pct": cv_pct,
        })

    return out_rows


def open_climatology_field():
    if CLIM_KIND == "all":
        ds = xr.open_dataset(MEAN_ALL_FILE)
        da = ds[VAR_ALL]

    elif CLIM_KIND == "season":
        ds = xr.open_dataset(MEAN_SEASON_FILE)
        da = ds[VAR_SEASON].sel(season=SEASON_NAME)

    elif CLIM_KIND == "day_night":
        ds = xr.open_dataset(MEAN_DAYNIGHT_FILE)
        da = ds[VAR_DAYNIGHT].sel(day_night=DAY_NIGHT_NAME)

    else:
        raise ValueError("CLIM_KIND must be 'all', 'season', or 'day_night'")

    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)

    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d, lon2d = lat, lon

    return ds, da, lat2d, lon2d


def plot_station_map(df_summary, metric_col, title, cbar_label, outfile=None):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=FIGSIZE)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.2)

        if MAP_EXTENT is not None:
            ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

        sc = ax.scatter(
            df_summary["lon"], df_summary["lat"],
            c=df_summary[metric_col],
            cmap="inferno",
            s=70,
            edgecolor="k",
            transform=ccrs.PlateCarree(),
        )

        if LABEL_STATIONS:
            for _, r in df_summary.iterrows():
                ax.text(r["lon"], r["lat"], str(r["station"]), fontsize=8, transform=ccrs.PlateCarree())

        ax.set_title(title)
        plt.colorbar(sc, ax=ax, label=cbar_label)

    except Exception:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sc = ax.scatter(
            df_summary["lon"], df_summary["lat"],
            c=df_summary[metric_col],
            cmap="inferno",
            s=70,
            edgecolor="k",
        )

        if MAP_EXTENT is not None:
            ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
            ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])

        if LABEL_STATIONS:
            for _, r in df_summary.iterrows():
                ax.text(r["lon"], r["lat"], str(r["station"]), fontsize=8)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, label=cbar_label)

    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    ds_clim, da_clim, lat2d, lon2d = open_climatology_field()
    station_files = get_station_files(PARQUET_DIR, STATION_LIST)

    if not station_files:
        raise FileNotFoundError("No station parquet files found for the chosen station list")

    rows = []
    field_cache = {}

    for fp in station_files:
        df = load_station_file(fp)
        if df.empty:
            continue

        row0 = df.iloc[0]
        station_name = row0["station"]
        station_lat = float(row0["station_lat"])
        station_lon = float(row0["station_lon"])
        i_center = int(row0["i_center"])
        j_center = int(row0["j_center"])

        try:
            k_center = choose_station_level_from_context(df)
        except Exception as e:
            print(f"Skipping {station_name}: could not determine level ({e})")
            continue

        # cache one 2D field per level
        if k_center not in field_cache:
            field_cache[k_center] = np.asarray(da_clim.isel(lev=k_center).values, dtype=float)

        field2d = field_cache[k_center]
        print(f"Processed station {station_name} at level {k_center}")
        sector_rows = compute_station_sector_stats_from_field(
            field2d=field2d,
            lat2d=lat2d,
            i_center=i_center,
            j_center=j_center,
            sectors=SECTORS,
            weighted=WEIGHTED,
        )

        for r in sector_rows:
            rows.append({
                "station": station_name,
                "lat": station_lat,
                "lon": station_lon,
                "i_center": i_center,
                "j_center": j_center,
                "k_center": k_center,
                **r
            })

    ds_clim.close()

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No station CV summaries were produced")

    summary.to_csv(OUT_DIR / f"station_cv_summary_{CLIM_KIND}.csv", index=False)

    # one map per sector
    for sector in SECTORS:
        sub = summary[summary["sector"] == sector].copy()
        if sub.empty:
            continue

        if CLIM_KIND == "all":
            title = f"Climatological area-weighted CV (%) | {sector} | full period"
            outfile = OUT_DIR / f"map_cv_{sector}_all.png"
        elif CLIM_KIND == "season":
            title = f"Climatological area-weighted CV (%) | {sector} | {SEASON_NAME}"
            outfile = OUT_DIR / f"map_cv_{sector}_{SEASON_NAME}.png"
        else:
            title = f"Climatological area-weighted CV (%) | {sector} | {DAY_NIGHT_NAME}"
            outfile = OUT_DIR / f"map_cv_{sector}_{DAY_NIGHT_NAME}.png"

        plot_station_map(
            sub,
            metric_col="cv_pct",
            title=title,
            cbar_label="CV (%)",
            outfile=outfile
        )


if __name__ == "__main__":
    main()