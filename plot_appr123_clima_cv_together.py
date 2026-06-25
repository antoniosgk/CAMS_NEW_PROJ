#%%
#!/usr/bin/env python3
# =============================================================================
# unified_cv_maps.py
#
# Unified station-CV climatology maps for the three approaches.
#
#   Approach 1 (A1): spatial CV of the TEMPORAL-MEAN field.
#                    Source: O3_climatology.nc  (time-averaged O3, per level).
#                    For each station, take the climatological mean field at the
#                    station's level, extract the cumulative square window for the
#                    sector, and compute the area-weighted spatial CV.   (small ~0-2%)
#
#   Approach 2 (A2): TEMPORAL MEAN of the per-timestep spatial CV (cv_w).
#                    Source: station_climatology_summary_{seadn}_{sector}.csv
#                    (produced by plot_climatological_maps.py; column
#                    `approach5_mean_cvw_pct`, already in %). No level needed.   (~0-3%)
#
#   Approach 3 (A3): POOLED space-time CV.
#                    Source: O3_gridded_pooled_cv_with_C0.nc (cv_pct_*, already in %).
#                    For each station, read the pooled-CV value at its level/cell.  (large ~10-25%)
#
# The three approaches produce different MAGNITUDES, so colorbar LIMITS are set
# per approach. Everything else (cmap, marker, fonts/weights, gridlines, extent,
# colorbar styling) is shared so the figure set looks consistent.
#
# Vertical level per station (A1 & A3) comes from one table built here:
#   - constant-k stations  -> station_constant_k_only_lookup.csv  (fixed level)
#   - variable-k stations  -> station_level_timeseries_all.parquet (MODAL level,
#     overall and per day/night and per season)
# This matches how main_compact built the parquets, so A1/A3 stay consistent with A2.
#
# Contexts: "all", "day", "night", "Winter", "Spring", "Summer", "Autumn".
# (season x day/night combinations are intentionally OUT OF SCOPE.)
# =============================================================================
#%%
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib


# =============================================================================
# 1. PATHS
# =============================================================================

SPECIES = "O3"

# --- Inputs shared by A1 & A3 (vertical-level table) -------------------------
CONST_K_CSV         = Path("/home/agkiokas/CAMS/lookups/station_constant_k_only_lookup.csv")
TIMESERIES_PARQUET  = Path("/home/agkiokas/CAMS/lookups/station_level_timeseries_all.parquet")

# --- Approach 1 input --------------------------------------------------------
CLIM_FILE           = Path("/mnt/store01/agkiokas/CAMS/output_climatologies/O3_climatology.nc")

# --- Approach 2 input (one CSV per (seadn, sector)) --------------------------
# seadn tags: all / day / night / DJF / MAM / JJA / SON  (see CONTEXT_SPECS)
A2_CSV_TEMPLATE     = "/home/agkiokas/CAMS/climatological_plots/station_climatology_summary_{seadn}_{sector}.csv"
A2_VALUE_COL        = "approach5_mean_cvw_pct"      # already a percentage
A2_STATION_COL      = "station"

# --- Approach 3 input --------------------------------------------------------
GRIDDED_FILE        = Path("/mnt/store01/agkiokas/CAMS/O3_gridded_pooled_cv_with_C0.nc")

# --- Output ------------------------------------------------------------------
OUT_DIR             = Path("/mnt/store01/agkiokas/CAMS/unified_cv_plots")


# =============================================================================
# 2. WHAT TO RUN
# =============================================================================

SHOW_PLOTS          = True          # False -> headless (Agg backend), only saves files
SAVE_STATION_INFO   = True           # dump the per-station levels actually used
EXPORT_CSV          = True           # dump per (approach, context, sector) station values

# Which context groups to run (set True/False independently)
RUN_ALL_PERIOD  = True
RUN_DAY_NIGHT   = False
RUN_SEASONS     = False

# --- derived: do not edit ---
CONTEXTS = (
    (["all"]                                              if RUN_ALL_PERIOD else []) +
    (["day", "night"]                                     if RUN_DAY_NIGHT  else []) +
    (["Winter", "Spring", "Summer", "Autumn"]             if RUN_SEASONS    else [])
)

# Sector mode
PLOT_SECTORS_TOGETHER = True                       # True  -> one 10-sector panel
SINGLE_SECTOR         = "C10"                      # used only when False
PANEL_SECTORS         = [f"C{i}" for i in range(1, 11)]   # C1..C10 (C0 reserved for A3 single/diff)

WEIGHTED            = True            # A1 spatial stats weighted by cos(lat)

# Titles (placeholders: {approach}, {sector}, {context})
PANEL_TITLE_TEMPLATE  = "Climatological CV (%) | all sectors | {approach} | {context}"
SINGLE_TITLE_TEMPLATE = "Coefficient of Variation (%) | {sector} | {approach} | {context}"

# How each context appears inside titles
CONTEXT_LABELS = {
    "all": "full period", "day": "day", "night": "night",
    "Winter": "Winter", "Spring": "Spring", "Summer": "Summer", "Autumn": "Autumn",
}


# =============================================================================
# 3. SHARED STYLE  (identical across all three approaches)
# =============================================================================

STYLE = {
    # geography
    "map_extent":            [70, 135, 15, 52],     # [lon_min, lon_max, lat_min, lat_max]
    "coast_lw":              0.6,
    "border_lw":             0.5,
    "land_alpha":            0.2,
    "ocean_alpha":           0.1,

    # gridlines
    "grid_color":            "gray",
    "grid_alpha":            0.4,
    "grid_lw":               0.4,
    "grid_ls":               "--",
    "grid_nloc":             5,
    "grid_label_size_single": 10,
    "grid_label_size_panel":  8,

    # markers
    "single_marker_size":    100,
    "panel_marker_size":     10,
    "marker_edge_color":     "black",
    "marker_edge_width":     0.4,

    # station labels
    "show_station_names":    False,
    "station_name_size":     6,

    # fonts / weights
    "font_weight":           "bold",
    "title_size":            16,      # single-map title
    "suptitle_size":         18,      # panel suptitle
    "ax_title_size":         13,      # panel per-sector subplot title
    "cbar_label_size":       13,
    "cbar_tick_size":        12,

    # figure sizes
    "figsize_single":        (20, 10),
    "subplot_layout":        "5x2",   # "5x2" or "2x5"
    "figsize_panel_5x2":     (8, 14),
    "figsize_panel_2x5":     (14, 8),

    # colorbar geometry
    "cbar_shrink_single":    0.95,
    "cbar_pad":              0.03,
    "panel_adjust":          dict(left=0.04, right=0.92, bottom=0.05, top=0.92,
                                  wspace=0.01, hspace=0.5),
    "panel_cbar_axes":       [0.93, 0.05, 0.02, 0.87],   # [left, bottom, width, height]

    "dpi":                   400,
}


# =============================================================================
# 4. PER-APPROACH STYLE / LIMITS
#    vmin/vmax = None  ->  auto from the data of each figure
#    extend in {"neither", "both", "min", "max"}
#    over_color / under_color = None -> not set
# =============================================================================

APPROACHES = {
    "A1": {
        "enabled":     True,
        "label":       "SR 1",
        "cmap":        "jet",
        "vmin":        0.0,
        "vmax":        2.0,
        "extend":      "max",
        "over_color":  "magenta",
        "under_color": None,
        "cbar_label":  "CV (%)",
    },
    "A2": {
        "enabled":     True,
        "label":       "SR 2",
        "cmap":        "jet",
        "vmin":        0.0,
        "vmax":        3.0,
        "extend":      "max",
        "over_color":  "magenta",
        "under_color": None,
        "cbar_label":  "CV (%)",
    },
    "A3": {
        "enabled":     True,
        "label":       "SR 3",
        "cmap":        "jet",
        "vmin":        10.0,
        "vmax":        25.0,
        "extend":      "both",
        "over_color":  "magenta",
        "under_color": "black",
        "cbar_label":  "CV (%)",
    },
}


# =============================================================================
# 5. OPTIONAL DIFFERENCE MAP  (sector_a - sector_b, one approach)
#    vmin/vmax = None -> symmetric limits +/- max|diff|
# =============================================================================

DIFF = {
    "enabled":      False,
    "approach":     "A3",        # "A1" / "A2" / "A3"
    "sector_a":     "C10",
    "sector_b":     "C0",        # C0 only valid for A3
    "cmap":         "RdBu_r",
    "vmin":         None,
    "vmax":         None,
    "extend":       "both",
    "over_color":   None,
    "under_color":  None,
    "cbar_label":   "ΔCV (%)",
    "title_template": "CV difference (%) | {approach} | {sector_a} - {sector_b} | {context}",
}


# =============================================================================
# 6. CONSTANTS / CONTEXT SPECIFICATION
# =============================================================================

THRESHOLD_DEG = -0.833            # solar elevation day/night threshold
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]

# Per-context: how to slice each source, which level column to use, and the
# A2 CSV filename tag. NOTE: A2 season files use DJF/MAM/JJA/SON; the NetCDF
# season coordinate uses Winter/Spring/Summer/Autumn.
CONTEXT_SPECS = {
    "all":    {"kind": "all",      "season": None,     "day_night": None,    "lev_col": "lev_all",    "csv_tag": "all"},
    "day":    {"kind": "daynight", "season": None,     "day_night": "day",   "lev_col": "lev_day",    "csv_tag": "day"},
    "night":  {"kind": "daynight", "season": None,     "day_night": "night", "lev_col": "lev_night",  "csv_tag": "night"},
    "Winter": {"kind": "season",   "season": "Winter", "day_night": None,    "lev_col": "lev_Winter", "csv_tag": "DJF"},
    "Spring": {"kind": "season",   "season": "Spring", "day_night": None,    "lev_col": "lev_Spring", "csv_tag": "MAM"},
    "Summer": {"kind": "season",   "season": "Summer", "day_night": None,    "lev_col": "lev_Summer", "csv_tag": "JJA"},
    "Autumn": {"kind": "season",   "season": "Autumn", "day_night": None,    "lev_col": "lev_Autumn", "csv_tag": "SON"},
}

# Climatology variable names (A1) and pooled-CV variable names (A3)
A1_VARS = {
    "all":      f"{SPECIES}_mean_all",
    "season":   f"{SPECIES}_mean_season",
    "daynight": f"{SPECIES}_mean_daynight",
}
A3_VARS = {
    "all":      "cv_pct_all",
    "season":   "cv_pct_season",
    "daynight": "cv_pct_daynight",
}


# =============================================================================
# Backend selection must happen before importing pyplot
# =============================================================================
if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.ticker as mticker      # noqa: E402
import cartopy.crs as ccrs               # noqa: E402
import cartopy.feature as cfeature       # noqa: E402

plt.rcParams["font.weight"] = STYLE["font_weight"]


# =============================================================================
# 7. TIME / SEASON / DAY-NIGHT HELPERS
# =============================================================================

def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def solar_day_mask_1d(lat, lon, time_values, threshold_deg=THRESHOLD_DEG):
    """Vectorized day/night ('day'/'night') for station-time rows."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lon = ((lon + 180.0) % 360.0) - 180.0

    times = pd.to_datetime(time_values)
    doy = times.dt.dayofyear.to_numpy()
    hour = (times.dt.hour.to_numpy()
            + times.dt.minute.to_numpy() / 60.0
            + times.dt.second.to_numpy() / 3600.0)

    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (hour - 12.0) / 24.0)

    decl = (0.006918
            - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.001480 * np.sin(3 * gamma))

    eqtime = 229.18 * (0.000075
                       + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))

    tst = (hour * 60.0 + eqtime + 4.0 * lon) % 1440.0
    ha = np.deg2rad((tst / 4.0) - 180.0)
    lat_rad = np.deg2rad(lat)

    cosz = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, -1.0, 1.0)
    elev = np.rad2deg(np.pi / 2.0 - np.arccos(cosz))
    return np.where(elev > threshold_deg, "day", "night")


def mode_int(x):
    m = x.mode()
    return int(m.iloc[0]) if len(m) else int(x.iloc[0])


def sector_to_radius(sector: str) -> int:
    digits = "".join(ch for ch in str(sector) if ch.isdigit())
    if digits == "":
        raise ValueError(f"Cannot parse sector: {sector}")
    return int(digits)


# =============================================================================
# 8. STATION VERTICAL-LEVEL TABLE  (const-k + timeseries, modal for variable-k)
# =============================================================================

def load_const_k(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {c: "station" for c in df.columns if c.lower().startswith("station")}
    df = df.rename(columns=ren)
    needed = ["station", "level_idx", "i", "j", "model_lat", "model_lon"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"const-k CSV missing columns {miss}. Has: {list(df.columns)}")
    df = df[needed].copy()
    df["station"] = df["station"].astype(str)
    for c in ("level_idx", "i", "j"):
        df[c] = df[c].astype(int)
    return df


def load_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    scol = "Station_Name" if "Station_Name" in df.columns else "station"
    df = df.rename(columns={scol: "station"})
    needed = ["time", "station", "i", "j", "model_lat", "model_lon", "level_idx"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"timeseries parquet missing columns {miss}. Has: {list(df.columns)}")
    df = df[needed].copy()
    df["station"] = df["station"].astype(str)
    df["time"] = pd.to_datetime(df["time"])
    for c in ("i", "j", "level_idx"):
        df[c] = df[c].astype(int)
    df["season"] = df["time"].dt.month.map(month_to_season)
    df["day_night"] = solar_day_mask_1d(df["model_lat"], df["model_lon"], df["time"])
    return df


def build_station_level_table(ts_df: pd.DataFrame, const_df: pd.DataFrame) -> pd.DataFrame:
    """One row per station with i, j, model_lat/lon and the level used per context."""
    lev_cols = ["lev_day", "lev_night"] + [f"lev_{s}" for s in SEASONS]
    keep = (["station", "i", "j", "model_lat", "model_lon", "lev_all"]
            + lev_cols + ["source", "n_unique_k"])

    const_stations = set(const_df["station"])
    nonconst = ts_df[~ts_df["station"].isin(const_stations)].copy()

    # constant-k stations: same level everywhere
    const_info = const_df.rename(columns={"level_idx": "lev_all"}).copy()
    const_info["lev_day"] = const_info["lev_all"]
    const_info["lev_night"] = const_info["lev_all"]
    for s in SEASONS:
        const_info[f"lev_{s}"] = const_info["lev_all"]
    const_info["source"] = "const_k_csv"
    const_info["n_unique_k"] = 1

    parts = [const_info[keep]]

    # variable-k stations: modal level overall and per day/night and per season
    if len(nonconst):
        base = (nonconst.groupby("station")
                .agg(i=("i", mode_int), j=("j", mode_int),
                     model_lat=("model_lat", "first"), model_lon=("model_lon", "first"),
                     lev_all=("level_idx", mode_int),
                     n_unique_k=("level_idx", "nunique"))
                .reset_index())

        dn = (nonconst.groupby(["station", "day_night"])["level_idx"]
              .agg(mode_int).unstack("day_night")
              .rename(columns=lambda c: f"lev_{c}").reset_index())

        se = (nonconst.groupby(["station", "season"])["level_idx"]
              .agg(mode_int).unstack("season")
              .rename(columns=lambda c: f"lev_{c}").reset_index())

        nonconst_info = base.merge(dn, on="station", how="left").merge(se, on="station", how="left")
        nonconst_info["source"] = "timeseries_parquet"
        for col in lev_cols:
            if col not in nonconst_info.columns:
                nonconst_info[col] = np.nan
            nonconst_info[col] = nonconst_info[col].fillna(nonconst_info["lev_all"]).astype(int)

        parts.append(nonconst_info[keep])

    station_info = pd.concat(parts, ignore_index=True)
    station_info["i"] = station_info["i"].astype(int)
    station_info["j"] = station_info["j"].astype(int)
    for col in ["lev_all"] + lev_cols:
        station_info[col] = station_info[col].astype(int)

    station_info = station_info.sort_values("station").reset_index(drop=True)
    return station_info


# =============================================================================
# 9. SPATIAL STAT HELPERS (Approach 1)
# =============================================================================

def weighted_mean_and_std(values, weights=None):
    v = np.asarray(values, dtype=float)
    m = np.isfinite(v)
    v = v[m]
    if v.size == 0:
        return np.nan, np.nan
    if weights is None:
        return float(np.mean(v)), float(np.std(v, ddof=0))
    w = np.asarray(weights, dtype=float)[m]
    if w.size == 0 or np.sum(w) == 0:
        return np.nan, np.nan
    mean = float(np.sum(w * v) / np.sum(w))
    var = float(np.sum(w * (v - mean) ** 2) / np.sum(w))
    return mean, float(np.sqrt(var))


def window_bounds(ic, jc, radius, shape):
    ny, nx = shape
    i0 = max(0, ic - radius); i1 = min(ny, ic + radius + 1)
    j0 = max(0, jc - radius); j1 = min(nx, jc + radius + 1)
    return i0, i1, j0, j1


def get_clim_grid(clim_ds):
    lat_name = "lat" if "lat" in clim_ds else "latitude"
    lon_name = "lon" if "lon" in clim_ds else "longitude"
    lat = np.asarray(clim_ds[lat_name].values)
    lon = np.asarray(clim_ds[lon_name].values)
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d, lon2d = lat, lon
    return lat2d, lon2d


# =============================================================================
# 10. VALUE EXTRACTORS  ->  dict {sector: 1D array aligned to station_info}
# =============================================================================

def values_approach1(clim_ds, station_info, context, sectors, weighted=WEIGHTED):
    spec = CONTEXT_SPECS[context]
    da = clim_ds[A1_VARS[spec["kind"]]]
    if spec["kind"] == "season":
        da = da.sel(season=spec["season"])
    elif spec["kind"] == "daynight":
        da = da.sel(day_night=spec["day_night"])

    lat2d, _ = get_clim_grid(clim_ds)
    nlev = da.sizes["lev"]
    lev_col = spec["lev_col"]

    # cache the 2D field per needed level (levels repeat across stations & sectors)
    field_by_lev = {}
    for k in sorted(set(int(v) for v in station_info[lev_col])):
        if 0 <= k < nlev:
            field_by_lev[k] = np.asarray(da.isel(lev=k).values, dtype=float)

    n = len(station_info)
    result = {s: np.full(n, np.nan) for s in sectors}
    radii = {s: sector_to_radius(s) for s in sectors}

    for row_i, (_, row) in enumerate(station_info.iterrows()):
        k = int(row[lev_col])
        f2d = field_by_lev.get(k)
        if f2d is None:
            continue
        ic, jc = int(row["i"]), int(row["j"])
        for s in sectors:
            i0, i1, j0, j1 = window_bounds(ic, jc, radii[s], f2d.shape)
            vals = f2d[i0:i1, j0:j1]
            if weighted:
                w = np.cos(np.deg2rad(lat2d[i0:i1, j0:j1]))
                mean_, std_ = weighted_mean_and_std(vals, w)
            else:
                mean_, std_ = weighted_mean_and_std(vals, None)
            if np.isfinite(mean_) and mean_ != 0:
                result[s][row_i] = 100.0 * std_ / mean_
    return result


def values_approach2(station_info, context, sectors):
    spec = CONTEXT_SPECS[context]
    n = len(station_info)
    result = {}
    for s in sectors:
        path = Path(A2_CSV_TEMPLATE.format(seadn=spec["csv_tag"], sector=s))
        if not path.exists():
            print(f"[A2] missing CSV: {path.name} -> NaN")
            result[s] = np.full(n, np.nan)
            continue
        df = pd.read_csv(path)
        if A2_VALUE_COL not in df.columns or A2_STATION_COL not in df.columns:
            print(f"[A2] {path.name} missing '{A2_VALUE_COL}' or '{A2_STATION_COL}' -> NaN")
            result[s] = np.full(n, np.nan)
            continue
        sub = df[[A2_STATION_COL, A2_VALUE_COL]].copy()
        sub[A2_STATION_COL] = sub[A2_STATION_COL].astype(str)
        sub = sub.drop_duplicates(subset=A2_STATION_COL)
        merged = station_info[["station"]].merge(
            sub, left_on="station", right_on=A2_STATION_COL, how="left")
        result[s] = merged[A2_VALUE_COL].to_numpy(dtype=float)
    return result


def values_approach3(grid_ds, station_info, context, sectors):
    spec = CONTEXT_SPECS[context]
    base = grid_ds[A3_VARS[spec["kind"]]]
    if spec["kind"] == "season":
        base = base.sel(season=spec["season"])
    elif spec["kind"] == "daynight":
        base = base.sel(day_night=spec["day_night"])

    lev_col = spec["lev_col"]
    n = len(station_info)
    result = {}
    for s in sectors:
        arr = np.full(n, np.nan)
        try:
            da = base.sel(sector=s)
        except (KeyError, Exception):
            print(f"[A3] sector '{s}' not found -> NaN")
            result[s] = arr
            continue
        vals3 = np.asarray(da.values, dtype=float)         # (lev, lat, lon)
        nlev, nlat, nlon = vals3.shape
        for row_i, (_, row) in enumerate(station_info.iterrows()):
            k, ii, jj = int(row[lev_col]), int(row["i"]), int(row["j"])
            if 0 <= k < nlev and 0 <= ii < nlat and 0 <= jj < nlon:
                arr[row_i] = vals3[k, ii, jj]
        result[s] = arr
    return result


def compute_values(ap_key, station_info, context, sectors, clim_ds, grid_ds):
    if ap_key == "A1":
        return values_approach1(clim_ds, station_info, context, sectors)
    if ap_key == "A2":
        return values_approach2(station_info, context, sectors)
    if ap_key == "A3":
        return values_approach3(grid_ds, station_info, context, sectors)
    raise ValueError(f"Unknown approach: {ap_key}")


# =============================================================================
# 11. PLOTTING
# =============================================================================

def make_cmap(name, over=None, under=None):
    cmap = plt.colormaps[name].copy()
    if over is not None:
        cmap.set_over(over)
    if under is not None:
        cmap.set_under(under)
    return cmap


def resolve_limits(arrays, vmin_cfg, vmax_cfg):
    if vmin_cfg is not None and vmax_cfg is not None:
        return vmin_cfg, vmax_cfg
    flat = (np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
            if arrays else np.array([np.nan]))
    flat = flat[np.isfinite(flat)]
    auto_min = float(np.nanmin(flat)) if flat.size else 0.0
    auto_max = float(np.nanmax(flat)) if flat.size else 1.0
    return (vmin_cfg if vmin_cfg is not None else auto_min,
            vmax_cfg if vmax_cfg is not None else auto_max)


def _prep_geo_ax(ax, label_size):
    ax.set_extent(STYLE["map_extent"], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=STYLE["coast_lw"])
    ax.add_feature(cfeature.BORDERS, linewidth=STYLE["border_lw"])
    ax.add_feature(cfeature.LAND, alpha=STYLE["land_alpha"])
    ax.add_feature(cfeature.OCEAN, alpha=STYLE["ocean_alpha"])
    gl = ax.gridlines(draw_labels=True, linewidth=STYLE["grid_lw"],
                      color=STYLE["grid_color"], alpha=STYLE["grid_alpha"],
                      linestyle=STYLE["grid_ls"])
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": label_size}
    gl.ylabel_style = {"size": label_size}
    gl.xlocator = mticker.MaxNLocator(STYLE["grid_nloc"])
    gl.ylocator = mticker.MaxNLocator(STYLE["grid_nloc"])
    return gl


def _scatter_names(ax, station_info, size):
    if not STYLE["show_station_names"]:
        return
    for _, r in station_info.iterrows():
        ax.text(r["model_lon"], r["model_lat"], str(r["station"]),
                fontsize=size, fontweight=STYLE["font_weight"],
                ha="left", va="bottom", transform=ccrs.PlateCarree(), zorder=6)


def plot_single_map(station_info, values, ap, title, out_file):
    cmap = make_cmap(ap["cmap"], ap.get("over_color"), ap.get("under_color"))
    vmin, vmax = resolve_limits([values], ap["vmin"], ap["vmax"])

    fig = plt.figure(figsize=STYLE["figsize_single"])
    ax = plt.axes(projection=ccrs.PlateCarree())
    _prep_geo_ax(ax, STYLE["grid_label_size_single"])

    sc = ax.scatter(station_info["model_lon"], station_info["model_lat"],
                    c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=STYLE["single_marker_size"],
                    edgecolor=STYLE["marker_edge_color"],
                    linewidth=STYLE["marker_edge_width"],
                    transform=ccrs.PlateCarree(), zorder=5)
    _scatter_names(ax, station_info, STYLE["station_name_size"])

    cbar = plt.colorbar(sc, ax=ax, shrink=STYLE["cbar_shrink_single"],
                        pad=STYLE["cbar_pad"], extend=ap["extend"])
    cbar.set_label(ap["cbar_label"], fontsize=STYLE["cbar_label_size"],
                   fontweight=STYLE["font_weight"])
    cbar.ax.tick_params(labelsize=STYLE["cbar_tick_size"])

    ax.set_title(title, fontsize=STYLE["title_size"], fontweight=STYLE["font_weight"])
    fig.tight_layout()
    fig.savefig(out_file, dpi=STYLE["dpi"], bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_sector_panel(station_info, values_by_sector, sectors, ap, suptitle, out_file):
    cmap = make_cmap(ap["cmap"], ap.get("over_color"), ap.get("under_color"))
    vmin, vmax = resolve_limits([values_by_sector[s] for s in sectors], ap["vmin"], ap["vmax"])

    if STYLE["subplot_layout"] == "2x5":
        nrows, ncols, figsize = 2, 5, STYLE["figsize_panel_2x5"]
    else:
        nrows, ncols, figsize = 5, 2, STYLE["figsize_panel_5x2"]

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             constrained_layout=False)
    axes = np.ravel(axes)
    last_sc = None

    for ax, sector in zip(axes, sectors):
        _prep_geo_ax(ax, STYLE["grid_label_size_panel"])
        last_sc = ax.scatter(station_info["model_lon"], station_info["model_lat"],
                             c=values_by_sector[sector], cmap=cmap, vmin=vmin, vmax=vmax,
                             s=STYLE["panel_marker_size"],
                             edgecolor=STYLE["marker_edge_color"],
                             linewidth=STYLE["marker_edge_width"],
                             transform=ccrs.PlateCarree(), zorder=5)
        _scatter_names(ax, station_info, max(4, STYLE["station_name_size"] - 1))
        ax.set_title(sector, fontsize=STYLE["ax_title_size"], fontweight=STYLE["font_weight"])

    # hide unused axes (if any)
    for ax in axes[len(sectors):]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=STYLE["suptitle_size"], fontweight=STYLE["font_weight"])
    fig.subplots_adjust(**STYLE["panel_adjust"])

    cbar_ax = fig.add_axes(STYLE["panel_cbar_axes"])
    cbar = fig.colorbar(last_sc, cax=cbar_ax, extend=ap["extend"])
    cbar.set_label(ap["cbar_label"], fontsize=STYLE["cbar_label_size"],
                   fontweight=STYLE["font_weight"])
    cbar.ax.tick_params(labelsize=STYLE["cbar_tick_size"])

    fig.savefig(out_file, dpi=STYLE["dpi"], bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"Saved: {out_file}")


# =============================================================================
# 12. CSV EXPORT
# =============================================================================

def export_values_csv(ap_key, context, sector, station_info, values):
    spec = CONTEXT_SPECS[context]
    lev = station_info[spec["lev_col"]].values if ap_key != "A2" else np.full(len(station_info), np.nan)
    out = pd.DataFrame({
        "station":   station_info["station"].values,
        "model_lon": station_info["model_lon"].values,
        "model_lat": station_info["model_lat"].values,
        "lev":       lev,
        "cv_pct":    values,
    })
    fn = OUT_DIR / f"{ap_key}_values_{context}_{sector}.csv"
    out.to_csv(fn, index=False)


# =============================================================================
# 13. MAIN
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- station vertical-level table (shared by A1 & A3) --------------------
    const_df = load_const_k(CONST_K_CSV)
    ts_df = load_timeseries(TIMESERIES_PARQUET)
    station_info = build_station_level_table(ts_df, const_df)

    print(f"Stations: {len(station_info)}")
    print(station_info["source"].value_counts().to_string())
    print(f"Variable-k stations: {(station_info['n_unique_k'] > 1).sum()}")

    if SAVE_STATION_INFO:
        station_info.to_csv(OUT_DIR / "station_levels_used.csv", index=False)

    # --- open shared datasets only if needed ---------------------------------
    clim_ds = xr.open_dataset(CLIM_FILE) if APPROACHES["A1"]["enabled"] else None
    grid_ds = xr.open_dataset(GRIDDED_FILE) if (APPROACHES["A3"]["enabled"]
                                               or (DIFF["enabled"] and DIFF["approach"] == "A3")) else None

    sectors = PANEL_SECTORS if PLOT_SECTORS_TOGETHER else [SINGLE_SECTOR]

    try:
        # ---------------- normal approach maps ------------------------------
        for ap_key in ("A1", "A2", "A3"):
            ap = APPROACHES[ap_key]
            if not ap["enabled"]:
                continue

            if (not PLOT_SECTORS_TOGETHER) and SINGLE_SECTOR == "C0" and ap_key in ("A1", "A2"):
                print(f"[{ap_key}] C0 is only meaningful for A3; skipping.")
                continue

            for context in CONTEXTS:
                if context not in CONTEXT_SPECS:
                    print(f"[skip] unknown context '{context}'")
                    continue

                vbs = compute_values(ap_key, station_info, context, sectors, clim_ds, grid_ds)

                if EXPORT_CSV:
                    for s in sectors:
                        export_values_csv(ap_key, context, s, station_info, vbs[s])

                ctx_lbl = CONTEXT_LABELS.get(context, context)
                if PLOT_SECTORS_TOGETHER:
                    suptitle = PANEL_TITLE_TEMPLATE.format(approach=ap["label"], context=ctx_lbl)
                    out = OUT_DIR / f"{ap_key}_CV_panel_{context}_{STYLE['subplot_layout']}.png"
                    plot_sector_panel(station_info, vbs, sectors, ap, suptitle, out)
                else:
                    title = SINGLE_TITLE_TEMPLATE.format(
                        approach=ap["label"], sector=SINGLE_SECTOR, context=ctx_lbl)
                    out = OUT_DIR / f"{ap_key}_CV_{SINGLE_SECTOR}_{context}.png"
                    plot_single_map(station_info, vbs[SINGLE_SECTOR], ap, title, out)

        # ---------------- optional difference map ---------------------------
        if DIFF["enabled"]:
            ap_key = DIFF["approach"]
            sa, sb = DIFF["sector_a"], DIFF["sector_b"]
            diff_style = {
                "cmap": DIFF["cmap"], "extend": DIFF["extend"],
                "over_color": DIFF["over_color"], "under_color": DIFF["under_color"],
                "cbar_label": DIFF["cbar_label"], "vmin": DIFF["vmin"], "vmax": DIFF["vmax"],
            }
            for context in CONTEXTS:
                if context not in CONTEXT_SPECS:
                    continue
                vbs = compute_values(ap_key, station_info, context, [sa, sb], clim_ds, grid_ds)
                diff = vbs[sa] - vbs[sb]

                # symmetric limits if not fixed
                style = dict(diff_style)
                if style["vmin"] is None or style["vmax"] is None:
                    lim = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 1.0
                    style["vmin"], style["vmax"] = -lim, lim

                if EXPORT_CSV:
                    export_values_csv(ap_key, context, f"{sa}_minus_{sb}", station_info, diff)

                title = DIFF["title_template"].format(
                    approach=APPROACHES[ap_key]["label"], sector_a=sa, sector_b=sb,
                    context=CONTEXT_LABELS.get(context, context))
                out = OUT_DIR / f"{ap_key}_CV_diff_{sa}_minus_{sb}_{context}.png"
                plot_single_map(station_info, diff, style, title, out)

    finally:
        if clim_ds is not None:
            clim_ds.close()
        if grid_ds is not None:
            grid_ds.close()

    print("DONE")


if __name__ == "__main__":
    main()
# %%
