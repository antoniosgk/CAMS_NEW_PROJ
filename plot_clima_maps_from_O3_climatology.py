#%%
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

CLIM_FILE = Path("/mnt/store01/agkiokas/CAMS/output_climatologies/O3_climatology.nc")

# station files or lookup
PARQUET_DIR = Path("/mnt/store01/agkiokas/CAMS/stations_parquet/")
LEVEL_LOOKUP_FILE = Path("/home/agkiokas/CAMS/lookups/station_level_lookup.parquet")

# choose stations
STATION_LIST = ["1001A", "1006A"]
# STATION_LIST = None   # use all stations found in PARQUET_DIR

# choose climatology context
CLIM_KIND = "all"              # "all", "season", "day_night", "season_day_night"
SEASON_NAME = "Winter"         # used for "season" and "season_day_night"
DAY_NIGHT_NAME = "day"         # used for "day_night" and "season_day_night"

# variable names inside O3_climatology.nc
VAR_ALL = "O3_mean_all"
VAR_SEASON = "O3_mean_season"
VAR_DAYNIGHT = "O3_mean_daynight"
VAR_SEASON_DAYNIGHT = "O3_mean_season_daynight"

# sectors to map
SECTORS = ["C1", "C2", "C3", "C4", "C5", "C10"]

# options
WEIGHTED = True
LABEL_STATIONS = True
FIGSIZE = (10, 6)
MAP_EXTENT = None
# Example:
# MAP_EXTENT = [70, 140, 15, 55]   # [lon_min, lon_max, lat_min, lat_max]

OUT_DIR = Path("/mnt/store01/agkiokas/CAMS/output_station_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def sector_to_radius(sector: str) -> int:
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


def compute_station_sector_stats_from_field(
    field2d: np.ndarray,
    lat2d: np.ndarray,
    i_center: int,
    j_center: int,
    sectors: list[str],
    weighted: bool = True,
):
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


def get_station_files(parquet_dir: Path, station_list=None):
    files = sorted([p for p in parquet_dir.glob("*.parquet") if "_summary" not in p.name])

    if station_list is not None:
        wanted = set(station_list)
        files = [p for p in files if p.stem.split("_")[0] in wanted]

    return files


def read_station_meta_one_row(fp: Path):
    df = pd.read_parquet(
        fp,
        columns=["station", "station_lat", "station_lon", "i_center", "j_center"]
    )
    if df.empty:
        raise ValueError(f"No rows in {fp}")
    row0 = df.iloc[0]
    return {
        "station": row0["station"],
        "lat": float(row0["station_lat"]),
        "lon": float(row0["station_lon"]),
        "i_center": int(row0["i_center"]),
        "j_center": int(row0["j_center"]),
    }


def open_climatology_field():
    ds = xr.open_dataset(CLIM_FILE)

    if CLIM_KIND == "all":
        da = ds[VAR_ALL]

    elif CLIM_KIND == "season":
        da = ds[VAR_SEASON].sel(season=SEASON_NAME)

    elif CLIM_KIND == "day_night":
        da = ds[VAR_DAYNIGHT].sel(day_night=DAY_NIGHT_NAME)

    elif CLIM_KIND == "season_day_night":
        da = ds[VAR_SEASON_DAYNIGHT].sel(season=SEASON_NAME, day_night=DAY_NIGHT_NAME)

    else:
        raise ValueError("CLIM_KIND must be 'all', 'season', 'day_night', or 'season_day_night'")

    lat_name = "lat" if "lat" in ds else "latitude"
    lon_name = "lon" if "lon" in ds else "longitude"

    lat = np.asarray(ds[lat_name].values)
    lon = np.asarray(ds[lon_name].values)

    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d, lon2d = lat, lon

    return ds, da, lat2d, lon2d


def load_level_lookup():
    lookup = pd.read_parquet(LEVEL_LOOKUP_FILE)

    station_col = "Station_Name" if "Station_Name" in lookup.columns else "station"
    lookup = lookup.set_index(station_col)

    return lookup


def get_station_level_from_lookup(station_name: str, lookup: pd.DataFrame) -> int | None:
    try:
        if CLIM_KIND == "all":
            k_center = lookup.loc[station_name, "k_all"]

        elif CLIM_KIND == "season":
            k_center = lookup.loc[station_name, f"k_{SEASON_NAME.lower()}"]

        elif CLIM_KIND == "day_night":
            k_center = lookup.loc[station_name, f"k_{DAY_NIGHT_NAME.lower()}"]

        elif CLIM_KIND == "season_day_night":
            # for season + day/night, prefer day/night-specific if desired,
            # but here we choose the seasonal level as main context.
            # You can swap this if you prefer the day/night mode instead.
            k_center = lookup.loc[station_name, f"k_{SEASON_NAME.lower()}"]

        else:
            raise ValueError("Invalid CLIM_KIND")

    except KeyError:
        return None

    if pd.isna(k_center):
        return None

    return int(k_center)


def context_suffix():
    if CLIM_KIND == "all":
        return "all"
    if CLIM_KIND == "season":
        return SEASON_NAME
    if CLIM_KIND == "day_night":
        return DAY_NIGHT_NAME
    return f"{SEASON_NAME}_{DAY_NIGHT_NAME}"


def context_title_piece():
    if CLIM_KIND == "all":
        return "full period"
    if CLIM_KIND == "season":
        return SEASON_NAME
    if CLIM_KIND == "day_night":
        return DAY_NIGHT_NAME
    return f"{SEASON_NAME} - {DAY_NIGHT_NAME}"


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
            df_summary["lon"],
            df_summary["lat"],
            c=df_summary[metric_col],
            cmap="inferno",
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
    level_lookup = load_level_lookup()
    station_files = get_station_files(PARQUET_DIR, STATION_LIST)

    if not station_files:
        raise FileNotFoundError("No station parquet files found for the chosen station list")

    rows = []
    field_cache = {}

    for n, fp in enumerate(station_files, start=1):
        try:
            meta = read_station_meta_one_row(fp)
            station_name = meta["station"]
            station_lat = meta["lat"]
            station_lon = meta["lon"]
            i_center = meta["i_center"]
            j_center = meta["j_center"]

            k_center = get_station_level_from_lookup(station_name, level_lookup)
            if k_center is None:
                print(f"Skipping {station_name}: no level in lookup for context")
                continue

            if "lev" not in da_clim.dims:
                raise ValueError("Climatology field has no 'lev' dimension")

            if k_center not in field_cache:
                if k_center < 0 or k_center >= da_clim.sizes["lev"]:
                    print(f"Skipping {station_name}: k_center={k_center} out of bounds")
                    continue
                field_cache[k_center] = np.asarray(da_clim.isel(lev=k_center).values, dtype=float)

            field2d = field_cache[k_center]

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

            print(f"{n}/{len(station_files)} processed: {station_name} (k={k_center})")

        except Exception as e:
            print(f"Skipping file {fp.name}: {e}")

    ds_clim.close()

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No station CV summaries were produced")

    suffix = context_suffix()
    summary_file = OUT_DIR / f"station_cv_summary_{suffix}.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Saved summary table: {summary_file}")

    # one map per sector
    for sector in SECTORS:
        sub = summary[summary["sector"] == sector].copy()
        if sub.empty:
            continue

        title = f"Climatological area-weighted CV (%) | {sector} | {context_title_piece()}"
        outfile = OUT_DIR / f"map_cv_{sector}_{suffix}.png"

        plot_station_map(
            sub,
            metric_col="cv_pct",
            title=title,
            cbar_label="CV (%)",
            outfile=outfile
        )


if __name__ == "__main__":
    main()
# %%
