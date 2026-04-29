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
STATION_LIST = ["1001A","1003A","1004A","1005A","1006A","1007A"]
#STATION_LIST = None   # use all stations found in PARQUET_DIR

# choose climatology context
CLIM_KIND = "season_day_night"              # "all", "season", "day_night", "season_day_night"
SEASON_NAME = "Spring"         # used for "season" and "season_day_night"
DAY_NIGHT_NAME = "day"         # used for "day_night" and "season_day_night"

# variable names inside O3_climatology.nc
VAR_ALL = "O3_mean_all"
VAR_SEASON = "O3_mean_season"
VAR_DAYNIGHT = "O3_mean_daynight"
VAR_SEASON_DAYNIGHT = "O3_mean_season_daynight"

# sectors to map
SECTORS = ["C10"]

# options
WEIGHTED = True
LABEL_STATIONS = True
FIGSIZE = (12, 6)
MAP_EXTENT = None
# Example:
#MAP_EXTENT = [116, 117, 39.6, 40.4]   # [lon_min, lon_max, lat_min, lat_max]
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


def open_climatology_field_for_context(clim_kind, season_name=None, day_night_name=None):
    ds = xr.open_dataset(CLIM_FILE)

    if clim_kind == "all":
        da = ds[VAR_ALL]

    elif clim_kind == "season":
        da = ds[VAR_SEASON].sel(season=season_name)

    elif clim_kind == "day_night":
        da = ds[VAR_DAYNIGHT].sel(day_night=day_night_name)

    elif clim_kind == "season_day_night":
        da = ds[VAR_SEASON_DAYNIGHT].sel(
            season=season_name,
            day_night=day_night_name
        )

    else:
        raise ValueError("Invalid climatology kind")

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

def get_station_level_from_lookup_for_context(
    station_name,
    lookup,
    clim_kind,
    season_name=None,
    day_night_name=None
):
    try:
        if clim_kind == "all":
            k_center = lookup.loc[station_name, "k_all"]

        elif clim_kind == "season":
            k_center = lookup.loc[station_name, f"k_{season_name.lower()}"]

        elif clim_kind == "day_night":
            k_center = lookup.loc[station_name, f"k_{day_night_name.lower()}"]

        elif clim_kind == "season_day_night":
            # Current choice: use seasonal modal level.
            # If you later create season-day/night level lookup, use that instead.
            k_center = lookup.loc[station_name, f"k_{season_name.lower()}"]

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

def context_suffix_for(clim_kind, season_name=None, day_night_name=None):
    if clim_kind == "all":
        return "all"
    if clim_kind == "season":
        return season_name
    if clim_kind == "day_night":
        return day_night_name
    return f"{season_name}_{day_night_name}"


def context_title_for(clim_kind, season_name=None, day_night_name=None):
    if clim_kind == "all":
        return "full period"
    if clim_kind == "season":
        return season_name
    if clim_kind == "day_night":
        return day_night_name
    return f"{season_name} - {day_night_name}"

def context_title_piece():
    if CLIM_KIND == "all":
        return "full period"
    if CLIM_KIND == "season":
        return SEASON_NAME
    if CLIM_KIND == "day_night":
        return DAY_NIGHT_NAME
    return f"{SEASON_NAME} - {DAY_NIGHT_NAME}"

def get_contexts_to_run():
    if CLIM_KIND == "season":
        return [
            ("season", "Winter", None),
            ("season", "Spring", None),
            ("season", "Summer", None),
            ("season", "Autumn", None),
        ]

    elif CLIM_KIND == "day_night":
        return [
            ("day_night", None, "day"),
            ("day_night", None, "night"),
        ]

    elif CLIM_KIND == "season_day_night":
        return [
            ("season_day_night", "Winter", "day"),
            ("season_day_night", "Winter", "night"),
            ("season_day_night", "Spring", "day"),
            ("season_day_night", "Spring", "night"),
            ("season_day_night", "Summer", "day"),
            ("season_day_night", "Summer", "night"),
            ("season_day_night", "Autumn", "day"),
            ("season_day_night", "Autumn", "night"),
        ]

    else:
        return [
            ("all", None, None),
        ]
    
    
def plot_station_map(df_summary, metric_col, title, cbar_label, outfile=None,vmax=None,vmin=None):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.ticker as mticker

        fig = plt.figure(figsize=FIGSIZE)
        ax = plt.axes(projection=ccrs.PlateCarree())
        gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--"
    )

        gl.top_labels = False
        gl.right_labels = False

        gl.xlabel_style = {"size": 10}
        gl.ylabel_style = {"size": 10}
        gl.xlocator = mticker.MaxNLocator(6)
        gl.ylocator = mticker.MaxNLocator(6)
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
            edgecolor="k",vmax=vmax,vmin=vmin,
            transform=ccrs.PlateCarree(),
        )

        if LABEL_STATIONS:
            for _, r in df_summary.iterrows():
                ax.text(
                    r["lon"], r["lat"], str(r["station"]),
                    fontsize=12, transform=ccrs.PlateCarree()
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
            vmax=vmax,vmin=vmin
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

def compute_summary_for_context(clim_kind, season_name=None, day_night_name=None):
    ds_clim, da_clim, lat2d, lon2d = open_climatology_field_for_context(
        clim_kind,
        season_name,
        day_night_name
    )

    level_lookup = load_level_lookup()
    station_files = get_station_files(PARQUET_DIR, STATION_LIST)

    rows = []
    field_cache = {}

    for n, fp in enumerate(station_files, start=1):
        try:
            meta = read_station_meta_one_row(fp)
            station_name = meta["station"]

            k_center = get_station_level_from_lookup_for_context(
                station_name,
                level_lookup,
                clim_kind,
                season_name,
                day_night_name
            )

            if k_center is None:
                print(f"Skipping {station_name}: no level in lookup for context")
                continue

            if k_center not in field_cache:
                if k_center < 0 or k_center >= da_clim.sizes["lev"]:
                    print(f"Skipping {station_name}: k_center={k_center} out of bounds")
                    continue

                field_cache[k_center] = np.asarray(
                    da_clim.isel(lev=k_center).values,
                    dtype=float
                )

            field2d = field_cache[k_center]

            sector_rows = compute_station_sector_stats_from_field(
                field2d=field2d,
                lat2d=lat2d,
                i_center=meta["i_center"],
                j_center=meta["j_center"],
                sectors=SECTORS,
                weighted=WEIGHTED,
            )

            for r in sector_rows:
                rows.append({
                    "station": station_name,
                    "lat": meta["lat"],
                    "lon": meta["lon"],
                    "i_center": meta["i_center"],
                    "j_center": meta["j_center"],
                    "k_center": k_center,
                    "context": context_suffix_for(clim_kind, season_name, day_night_name),
                    **r
                })

        except Exception as e:
            print(f"Skipping file {fp.name}: {e}")

    ds_clim.close()

    return pd.DataFrame(rows)
# ============================================================
# MAIN
# ============================================================

def main():
    contexts = get_contexts_to_run()

    all_summaries = []

    for clim_kind, season_name, day_night_name in contexts:
        print(f"\nRunning context: {clim_kind}, season={season_name}, day_night={day_night_name}")

        summary = compute_summary_for_context(
            clim_kind,
            season_name,
            day_night_name
        )

        if summary.empty:
            print("No data for this context")
            continue

        suffix = context_suffix_for(clim_kind, season_name, day_night_name)
        summary_file = OUT_DIR / f"station_cv_summary_{suffix}.csv"
        summary.to_csv(summary_file, index=False)
        print(f"Saved summary table: {summary_file}")

        all_summaries.append(summary)

    if not all_summaries:
        raise RuntimeError("No station CV summaries were produced")

    all_data = pd.concat(all_summaries, ignore_index=True)

    # SAME COLORBAR LIMITS ACROSS ALL CONTEXTS AND ALL SECTORS
    vmax_global = all_data["cv_pct"].max()
    vmin_global = all_data["cv_pct"].min()
    print(f"\nGlobal colorbar limits: vmin={vmin_global:.3f}, vmax={vmax_global:.3f}")

    for summary in all_summaries:
        suffix = summary["context"].iloc[0]

        for sector in SECTORS:
            sub = summary[summary["sector"] == sector].copy()
            if sub.empty:
                continue

            title = f"Climatological area-weighted CV (%) | {sector} | {suffix}"
            outfile = OUT_DIR / f"map_cv_{sector}_{suffix}.png"

            plot_station_map(
                sub,
                metric_col="cv_pct",
                title=title,
                cbar_label="CV (%)",
                vmax=vmax_global,vmin=vmin_global,
                outfile=outfile
            )
if __name__ == "__main__":
    main()
# %%
