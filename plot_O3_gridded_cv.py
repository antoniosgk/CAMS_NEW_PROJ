#%%
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
plt.rcParams['font.weight'] = 'bold'
# ============================================================
# SETTINGS
# ============================================================

NC_FILE = Path("/mnt/store01/agkiokas/CAMS/O3_gridded_pooled_cv_with_C0.nc")

CONST_K_FILE = Path("/home/agkiokas/CAMS/lookups/station_constant_k_only_lookup.csv")

BIG_TIMESERIES_FILE = Path("/home/agkiokas/CAMS/lookups/station_level_timeseries_all.parquet"
)

OUT_DIR = Path("/mnt/store01/agkiokas/CAMS/O3_station_cv_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTOR = "C10"             # C0 ... C10
PLOT_ALL_SECTORS = False   # True if you want plots for all C1-C10

PLOT_SECTORS_TOGETHER = True  # True -> one subplot figure with all sectors
SUBPLOT_LAYOUT = "5x2"         # "2x5" or "5x2"

SECTORS_ALL = [f"C{i}" for i in range(1, 11)]
SHOW_NAMES = False

USE_SAME_COLORBAR_DAYNIGHT = True
USE_SAME_COLORBAR_SEASONS = True

MARKER_SIZE = 10
FIGSIZE = (20, 10)
CMAP = "inferno"
PLOT_DIFFERENCE = False

DIFF_SECTOR_A = "C10"
DIFF_SECTOR_B = "C0"

# Difference = A - B
# Examples:
# C10 - C0
# C10 - C1
# C5  - C0
DIFF_CMAP = "inferno"

THRESHOLD_DEG = -0.833


# ============================================================
# COLUMN NAMES
# ============================================================

# const_k_only.csv
CSV_STATION_COL = "Station_Name"

# big parquet
PQ_STATION_COL = "Station_Name"

TIME_COL = "time"
I_COL = "i"
J_COL = "j"
K_COL = "level_idx"
LAT_COL = "model_lat"
LON_COL = "model_lon"


# ============================================================
# TIME / SEASON / DAY-NIGHT HELPERS
# ============================================================

def month_to_season(month: int):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def solar_day_mask_1d(lat, lon, time_values, threshold_deg=-0.833):
    """
    Vectorized solar day/night calculation for station-time rows.
    lat, lon: 1D arrays
    time_values: pandas datetime Series/array
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lon = ((lon + 180.0) % 360.0) - 180.0

    times = pd.to_datetime(time_values)

    doy = times.dt.dayofyear.to_numpy()
    hour = (
        times.dt.hour.to_numpy()
        + times.dt.minute.to_numpy() / 60.0
        + times.dt.second.to_numpy() / 3600.0
    )

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

    tst = (hour * 60.0 + eqtime + 4.0 * lon) % 1440.0
    ha = np.deg2rad((tst / 4.0) - 180.0)

    lat_rad = np.deg2rad(lat)

    cosz = (
        np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.cos(ha)
    )

    cosz = np.clip(cosz, -1.0, 1.0)
    elev = np.rad2deg(np.pi / 2.0 - np.arccos(cosz))

    return np.where(elev > threshold_deg, "day", "night")


def mode_int(x):
    return int(x.mode().iloc[0])


# ============================================================
# LOAD INPUTS
# ============================================================

def load_const_k_csv():
    df = pd.read_csv(CONST_K_FILE)

    needed = [
        CSV_STATION_COL,
        K_COL,
        I_COL,
        J_COL,
        LAT_COL,
        LON_COL,
    ]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"const_k CSV missing columns: {missing}")

    df = df[needed].copy()

    df = df.rename(columns={CSV_STATION_COL: "station"})

    df["station"] = df["station"].astype(str)
    df[I_COL] = df[I_COL].astype(int)
    df[J_COL] = df[J_COL].astype(int)
    df[K_COL] = df[K_COL].astype(int)

    return df


def load_big_timeseries():
    df = pd.read_parquet(BIG_TIMESERIES_FILE)

    needed = [
        TIME_COL,
        PQ_STATION_COL,
        I_COL,
        J_COL,
        LAT_COL,
        LON_COL,
        K_COL,
    ]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"big parquet missing columns: {missing}")

    df = df[needed].copy()

    df = df.rename(columns={PQ_STATION_COL: "station"})

    df["station"] = df["station"].astype(str)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    df[I_COL] = df[I_COL].astype(int)
    df[J_COL] = df[J_COL].astype(int)
    df[K_COL] = df[K_COL].astype(int)

    df["season"] = df[TIME_COL].dt.month.map(month_to_season)

    df["day_night"] = solar_day_mask_1d(
        lat=df[LAT_COL],
        lon=df[LON_COL],
        time_values=df[TIME_COL],
        threshold_deg=THRESHOLD_DEG,
    )

    return df


# ============================================================
# BUILD STATION LEVEL TABLE
# ============================================================

def build_station_level_table(ts_df, const_df):
    const_stations = set(const_df["station"])

    nonconst_ts = ts_df[~ts_df["station"].isin(const_stations)].copy()

    # --------------------------------------------------------
    # Constant-k stations
    # --------------------------------------------------------

    const_info = (
        const_df
        .rename(columns={K_COL: "lev_all"})
        .copy()
    )

    const_info["lev_day"] = const_info["lev_all"]
    const_info["lev_night"] = const_info["lev_all"]

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        const_info[f"lev_{season}"] = const_info["lev_all"]

    const_info["source"] = "const_k_csv"
    const_info["n_unique_k"] = 1

    # --------------------------------------------------------
    # Non-constant-k stations from full parquet
    # --------------------------------------------------------

    nonconst_base = (
        nonconst_ts
        .groupby("station")
        .agg(
            i=(I_COL, mode_int),
            j=(J_COL, mode_int),
            model_lat=(LAT_COL, "first"),
            model_lon=(LON_COL, "first"),
            lev_all=(K_COL, mode_int),
            n_records=(K_COL, "size"),
            n_unique_k=(K_COL, "nunique"),
        )
        .reset_index()
    )

    dn_levels = (
        nonconst_ts
        .groupby(["station", "day_night"])[K_COL]
        .agg(mode_int)
        .unstack("day_night")
        .rename(columns=lambda c: f"lev_{c}")
        .reset_index()
    )

    season_levels = (
        nonconst_ts
        .groupby(["station", "season"])[K_COL]
        .agg(mode_int)
        .unstack("season")
        .rename(columns=lambda c: f"lev_{c}")
        .reset_index()
    )

    nonconst_info = nonconst_base.merge(dn_levels, on="station", how="left")
    nonconst_info = nonconst_info.merge(season_levels, on="station", how="left")

    nonconst_info["source"] = "timeseries_parquet"

    level_cols = [
        "lev_day",
        "lev_night",
        "lev_Winter",
        "lev_Spring",
        "lev_Summer",
        "lev_Autumn",
    ]

    for col in level_cols:
        if col not in nonconst_info.columns:
            nonconst_info[col] = np.nan

        nonconst_info[col] = (
            nonconst_info[col]
            .fillna(nonconst_info["lev_all"])
            .astype(int)
        )

    # --------------------------------------------------------
    # Harmonize constant station table
    # --------------------------------------------------------

    const_info = const_info.rename(
        columns={
            I_COL: "i",
            J_COL: "j",
            LAT_COL: "model_lat",
            LON_COL: "model_lon",
        }
    )

    keep_cols = [
        "station",
        "i",
        "j",
        "model_lat",
        "model_lon",
        "lev_all",
        "lev_day",
        "lev_night",
        "lev_Winter",
        "lev_Spring",
        "lev_Summer",
        "lev_Autumn",
        "source",
        "n_unique_k",
    ]

    station_info = pd.concat(
        [
            const_info[keep_cols],
            nonconst_info[keep_cols],
        ],
        ignore_index=True,
    )

    station_info["i"] = station_info["i"].astype(int)
    station_info["j"] = station_info["j"].astype(int)

    for col in [
        "lev_all",
        "lev_day",
        "lev_night",
        "lev_Winter",
        "lev_Spring",
        "lev_Summer",
        "lev_Autumn",
    ]:
        station_info[col] = station_info[col].astype(int)

    return station_info


# ============================================================
# EXTRACT CV VALUES
# ============================================================

def extract_station_values(
    ds,
    station_info,
    var_name,
    sector,
    level_col,
    season=None,
    day_night=None,
):
    da = ds[var_name].sel(sector=sector)

    if season is not None:
        da = da.sel(season=season)

    if day_night is not None:
        da = da.sel(day_night=day_night)

    values = []

    for _, row in station_info.iterrows():
        val = da.isel(
            lev=int(row[level_col]),
            lat=int(row["i"]),
            lon=int(row["j"]),
        ).item()

        values.append(val)

    return np.array(values, dtype=float)

def extract_station_difference(
    ds,
    station_info,
    var_name,
    sector_a,
    sector_b,
    level_col,
    season=None,
    day_night=None,
):
    vals_a = extract_station_values(
        ds=ds,
        station_info=station_info,
        var_name=var_name,
        sector=sector_a,
        level_col=level_col,
        season=season,
        day_night=day_night,
    )

    vals_b = extract_station_values(
        ds=ds,
        station_info=station_info,
        var_name=var_name,
        sector=sector_b,
        level_col=level_col,
        season=season,
        day_night=day_night,
    )

    return vals_a - vals_b
# ============================================================
# PLOTTING
# ============================================================

def plot_station_map(
    station_info,
    values,
    title,
    out_file,
    vmin=None,
    vmax=None,
    show_names=False,
    cmap=CMAP,
    cbar_label="CV (%)"
):
    fig = plt.figure(figsize=FIGSIZE)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # --------------------------------------------------------
    # Gridlines
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Map features: same style as your reference plot
    # --------------------------------------------------------

    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, alpha=0.2)
    ax.add_feature(cfeature.OCEAN, alpha=0.1)

    # --------------------------------------------------------
    # Map extent
    # --------------------------------------------------------

    ax.set_extent([70, 135, 15, 52], crs=ccrs.PlateCarree())

    # --------------------------------------------------------
    # Scatter
    # --------------------------------------------------------

    sc = ax.scatter(
        station_info["model_lon"],
        station_info["model_lat"],
        c=values,
        s=MARKER_SIZE,
        cmap=cmap,
        vmin=-0.5,
        vmax=0.5,
        edgecolor="black",
        linewidth=0.4,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    # --------------------------------------------------------
    # Station names
    # --------------------------------------------------------

    if show_names:
        for _, row in station_info.iterrows():
            ax.text(
                row["model_lon"],
                row["model_lat"],
                str(row["station"]),
                fontsize=6,
                fontweight="bold",
                ha="left",
                va="bottom",
                transform=ccrs.PlateCarree(),
                zorder=6,
            )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    cbar = plt.colorbar(
        sc,
        ax=ax,
        shrink=0.95,
        pad=0.03,extend='both'
    )

    cbar.set_label(
        cbar_label,
        fontsize=13,
        fontweight="bold"
    )

    cbar.ax.tick_params(labelsize=13)

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.tight_layout()
    plt.savefig(out_file, dpi=400, bbox_inches="tight")

    print(f"Saved: {out_file}")
    plt.show()

def plot_all_sectors_subplot(
    station_info,
    sector_values_dict,
    title,
    out_file,
    vmin=None,
    vmax=None,
    show_names=False,
):
    import matplotlib.ticker as mticker

    if SUBPLOT_LAYOUT == "2x5":
        nrows, ncols = 2, 5
        figsize = (14, 8)
    else:
        nrows, ncols = 5, 2
        figsize = (8, 14)

    # Custom colormap:
    # normal colors: jet
    # values above vmax: magenta
    cmap = plt.colormaps["jet"].copy()
    cmap.set_over("magenta")
    cmap.set_under('black')

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=False,
    )

    axes = axes.ravel()
    last_sc = None

    for ax, sector in zip(axes, SECTORS_ALL):

        values = sector_values_dict[sector]

        ax.set_extent(
            [
                135,
                70,
                15,
                52,
            ],
            crs=ccrs.PlateCarree(),
        )

        ax.coastlines(resolution="10m", linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.2)

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.4,
            color="gray",
            alpha=0.4,
            linestyle="--",
        )

        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        gl.xlocator = mticker.MaxNLocator(4)
        gl.ylocator = mticker.MaxNLocator(4)

        last_sc = ax.scatter(
            station_info["model_lon"],
            station_info["model_lat"],
            c=values,
            s=MARKER_SIZE,
            cmap=cmap,
            vmin=10,
            vmax=25 ,
            edgecolor="black",
            linewidth=0.4,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

        if show_names:
            for _, row in station_info.iterrows():
                ax.text(
                    row["model_lon"],
                    row["model_lat"],
                    str(row["station"]),
                    fontsize=5,
                    ha="left",
                    va="bottom",
                    transform=ccrs.PlateCarree(),
                    zorder=6,
                )

        ax.set_title(sector, fontsize=13, fontweight="bold")

    fig.suptitle(title, fontsize=18, fontweight="bold")

    # Manual subplot spacing - same as first script
    fig.subplots_adjust(
        left=0.04,
        right=0.92,
        bottom=0.05,
        top=0.92,
        wspace=0.01,
        hspace=0.5,
    )

    # Manual colorbar position - same as first script
    cbar_ax = fig.add_axes([0.93, 0.05, 0.02, 0.87])

    cbar = fig.colorbar(
        last_sc,
        cax=cbar_ax,
        extend="both"
    )

    cbar.set_label("CV (%)", fontsize=13, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)

    fig.savefig(out_file, dpi=400, bbox_inches="tight")
    plt.show()
    #plt.close()
def plot_for_sector(ds, station_info, sector):
    # --------------------------------------------------------
    # Whole period
    # --------------------------------------------------------

    vals_all = extract_station_values(
        ds=ds,
        station_info=station_info,
        var_name="cv_pct_all",
        sector=sector,
        level_col="lev_all",
    )
    csv_all = station_info.copy()
    csv_all["cv_pct_all"] = vals_all

    csv_all.to_csv(
    OUT_DIR / f"station_cv_all_{sector}.csv",
    index=False,
    )
    plot_station_map(
        station_info,
        vals_all,
        title=f"Coefficient of Variation  {sector} Approach 3 (%)",
        out_file=OUT_DIR / f"O3_CV_all_{sector}.png",
        show_names=SHOW_NAMES,
    )
'''
    # --------------------------------------------------------
    # Day / night
    # --------------------------------------------------------

    daynight_values = {}

    for dn in ["day", "night"]:
        vals = extract_station_values(
            ds=ds,
            station_info=station_info,
            var_name="cv_pct_daynight",
            sector=sector,
            day_night=dn,
            level_col=f"lev_{dn}",
        )

        daynight_values[dn] = vals
        csv_dn = station_info.copy()
        csv_dn["day_night"] = dn
        csv_dn["cv_pct"] = vals

        csv_dn.to_csv(
        OUT_DIR / f"station_cv_{dn}_{sector}.csv",
        index=False,
    )

    if USE_SAME_COLORBAR_DAYNIGHT:
        all_vals = np.concatenate(list(daynight_values.values()))
        dn_vmin = np.nanmin(all_vals)
        dn_vmax = np.nanmax(all_vals)
    else:
        dn_vmin = None
        dn_vmax = None

    for dn, vals in daynight_values.items():
        plot_station_map(
            station_info,
            vals,
            title=f"Coefficient of Variation  {dn} {sector} Approach 3 (%)",
            out_file=OUT_DIR / f"O3_CV_{dn}_{sector}.png",
            vmin=dn_vmin,
            vmax=dn_vmax,
            show_names=SHOW_NAMES,
        )

    # --------------------------------------------------------
    # Seasons
    # --------------------------------------------------------

    season_values = {}

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        vals = extract_station_values(
            ds=ds,
            station_info=station_info,
            var_name="cv_pct_season",
            sector=sector,
            season=season,
            level_col=f"lev_{season}",
        )

        season_values[season] = vals
        csv_season = station_info.copy()
        csv_season["season"] = season
        csv_season["cv_pct"] = vals

        csv_season.to_csv(
        OUT_DIR / f"station_cv_{season}_{sector}.csv",
        index=False,
        )
    if USE_SAME_COLORBAR_SEASONS:
        all_vals = np.concatenate(list(season_values.values()))
        season_vmin = np.nanmin(all_vals)
        season_vmax = np.nanmax(all_vals)
    else:
        season_vmin = None
        season_vmax = None

    for season, vals in season_values.items():
        plot_station_map(
            station_info,
            vals,
            title=f"Coefficient of Variation {season} {sector} Approach 3 (%)",
            out_file=OUT_DIR / f"O3_CV_{season}_{sector}.png",
            vmin=season_vmin,
            vmax=season_vmax,
            show_names=SHOW_NAMES,
        )
'''        

def plot_all_sectors_together(ds, station_info):
    sectors = SECTORS_ALL

    # --------------------------------------------------------
    # Whole period
    # --------------------------------------------------------

    all_values = {}

    for sector in sectors:
        vals = extract_station_values(
            ds=ds,
            station_info=station_info,
            var_name="cv_pct_all",
            sector=sector,
            level_col="lev_all",
        )
        all_values[sector] = vals

    all_concat = np.concatenate(list(all_values.values()))
    vmin = np.nanmin(all_concat)
    vmax = np.nanmax(all_concat)

    plot_all_sectors_subplot(
        station_info=station_info,
        sector_values_dict=all_values,
        title="Climatological CV (%) | all sectors | Approach 3",
        out_file=OUT_DIR / "O3_CV_all_all_sectors.png",
        vmin=vmin,
        vmax=vmax,
        show_names=SHOW_NAMES,
    )
    '''
    # --------------------------------------------------------
    # Day / night
    # --------------------------------------------------------

    daynight_all = {}

    for dn in ["day", "night"]:
        daynight_all[dn] = {}

        for sector in sectors:
            vals = extract_station_values(
                ds=ds,
                station_info=station_info,
                var_name="cv_pct_daynight",
                sector=sector,
                day_night=dn,
                level_col=f"lev_{dn}",
            )
            daynight_all[dn][sector] = vals

    if USE_SAME_COLORBAR_DAYNIGHT:
        concat = []
        for dn in ["day", "night"]:
            concat.extend(daynight_all[dn].values())

        all_concat = np.concatenate(concat)
        dn_vmin = np.nanmin(all_concat)
        dn_vmax = np.nanmax(all_concat)
    else:
        dn_vmin = None
        dn_vmax = None

    for dn in ["day", "night"]:
        plot_all_sectors_subplot(
            station_info=station_info,
            sector_values_dict=daynight_all[dn],
            title=f"O3 CV - {dn} - all sectors",
            out_file=OUT_DIR / f"O3_CV_{dn}_all_sectors.png",
            vmin=dn_vmin,
            vmax=dn_vmax,
            show_names=SHOW_NAMES,
        )

    # --------------------------------------------------------
    # Seasons
    # --------------------------------------------------------

    season_all = {}

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        season_all[season] = {}

        for sector in sectors:
            vals = extract_station_values(
                ds=ds,
                station_info=station_info,
                var_name="cv_pct_season",
                sector=sector,
                season=season,
                level_col=f"lev_{season}",
            )
            season_all[season][sector] = vals

    if USE_SAME_COLORBAR_SEASONS:
        concat = []
        for season in ["Winter", "Spring", "Summer", "Autumn"]:
            concat.extend(season_all[season].values())

        all_concat = np.concatenate(concat)
        season_vmin = np.nanmin(all_concat)
        season_vmax = np.nanmax(all_concat)
    else:
        season_vmin = None
        season_vmax = None

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        plot_all_sectors_subplot(
            station_info=station_info,
            sector_values_dict=season_all[season],
            title=f"O3 CV - {season} - all sectors",
            out_file=OUT_DIR / f"O3_CV_{season}_all_sectors.png",
            vmin=season_vmin,
            vmax=season_vmax,
            show_names=SHOW_NAMES,
        )        
'''        
def plot_difference_for_sectors(ds, station_info, sector_a, sector_b):
    diff_name = f"{sector_a}_minus_{sector_b}"

    # --------------------------------------------------------
    # Whole period
    # --------------------------------------------------------

    vals_all = extract_station_difference(
        ds=ds,
        station_info=station_info,
        var_name="cv_pct_all",
        sector_a=sector_a,
        sector_b=sector_b,
        level_col="lev_all",
    )

    lim = np.nanmax(np.abs(vals_all))

    csv_all = station_info.copy()
    csv_all["sector_a"] = sector_a
    csv_all["sector_b"] = sector_b
    csv_all["cv_diff_pct"] = vals_all
    csv_all.to_csv(
        OUT_DIR / f"station_cv_diff_all_{diff_name}.csv",
        index=False,
    )

    plot_station_map(
        station_info,
        vals_all,
        title=f"O3 CV difference - Whole period - {sector_a} - {sector_b}",
        out_file=OUT_DIR / f"O3_CV_diff_all_{diff_name}.png",
        vmin=-lim,
        vmax=lim,
        show_names=SHOW_NAMES,
        cmap=DIFF_CMAP,
        cbar_label="ΔCV (%)",
    )
    '''
    # --------------------------------------------------------
    # Day / night
    # --------------------------------------------------------

    daynight_values = {}

    for dn in ["day", "night"]:
        vals = extract_station_difference(
            ds=ds,
            station_info=station_info,
            var_name="cv_pct_daynight",
            sector_a=sector_a,
            sector_b=sector_b,
            day_night=dn,
            level_col=f"lev_{dn}",
        )

        daynight_values[dn] = vals

        csv_dn = station_info.copy()
        csv_dn["day_night"] = dn
        csv_dn["sector_a"] = sector_a
        csv_dn["sector_b"] = sector_b
        csv_dn["cv_diff_pct"] = vals
        csv_dn.to_csv(
            OUT_DIR / f"station_cv_diff_{dn}_{diff_name}.csv",
            index=False,
        )

    if USE_SAME_COLORBAR_DAYNIGHT:
        all_vals = np.concatenate(list(daynight_values.values()))
        lim = np.nanmax(np.abs(all_vals))
        dn_vmin, dn_vmax = -lim, lim
    else:
        dn_vmin, dn_vmax = None, None

    for dn, vals in daynight_values.items():
        plot_station_map(
            station_info,
            vals,
            title=f"O3 CV difference - {dn} - {sector_a} - {sector_b}",
            out_file=OUT_DIR / f"O3_CV_diff_{dn}_{diff_name}.png",
            vmin=dn_vmin,
            vmax=dn_vmax,
            show_names=SHOW_NAMES,
            cmap=DIFF_CMAP,
            cbar_label="ΔCV (%)",
        )

    # --------------------------------------------------------
    # Seasons
    # --------------------------------------------------------

    season_values = {}

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        vals = extract_station_difference(
            ds=ds,
            station_info=station_info,
            var_name="cv_pct_season",
            sector_a=sector_a,
            sector_b=sector_b,
            season=season,
            level_col=f"lev_{season}",
        )

        season_values[season] = vals

        csv_season = station_info.copy()
        csv_season["season"] = season
        csv_season["sector_a"] = sector_a
        csv_season["sector_b"] = sector_b
        csv_season["cv_diff_pct"] = vals
        csv_season.to_csv(
            OUT_DIR / f"station_cv_diff_{season}_{diff_name}.csv",
            index=False,
        )

    if USE_SAME_COLORBAR_SEASONS:
        all_vals = np.concatenate(list(season_values.values()))
        lim = np.nanmax(np.abs(all_vals))
        season_vmin, season_vmax = -lim, lim
    else:
        season_vmin, season_vmax = None, None

    for season, vals in season_values.items():
        plot_station_map(
            station_info,
            vals,
            title=f"O3 CV difference - {season} - {sector_a} - {sector_b}",
            out_file=OUT_DIR / f"O3_CV_diff_{season}_{diff_name}.png",
            vmin=season_vmin,
            vmax=season_vmax,
            show_names=SHOW_NAMES,
            cmap=DIFF_CMAP,
            cbar_label="ΔCV (%)",
        )
'''
# ============================================================
# MAIN
# ============================================================

def main():
    ds = xr.open_dataset(NC_FILE)

    const_df = load_const_k_csv()
    ts_df = load_big_timeseries()

    station_info = build_station_level_table(ts_df, const_df)

    print(station_info.head())
    print()
    print(f"Total stations: {len(station_info)}")
    print(station_info["source"].value_counts())
    print()
    print("Stations with variable k:")
    print((station_info["n_unique_k"] > 1).sum())

    station_info.to_csv(
        OUT_DIR / "station_selected_levels_for_plots.csv",
        index=False,
    )

    if PLOT_SECTORS_TOGETHER:
        print("Plotting all sectors together")
        plot_all_sectors_together(ds, station_info)
    else:
        if PLOT_ALL_SECTORS:
         sectors = [str(s) for s in ds["sector"].values]
        else:
         sectors = [SECTOR]
    
        for sector in sectors:
           print(f"Plotting sector {sector}")
           plot_for_sector(ds, station_info, sector)    

    if PLOT_DIFFERENCE:
        print(f"Plotting difference: {DIFF_SECTOR_A} - {DIFF_SECTOR_B}")
        plot_difference_for_sectors(
        ds,
        station_info,
        sector_a=DIFF_SECTOR_A,
        sector_b=DIFF_SECTOR_B,
        )
    

    print('END')
if __name__ == "__main__":
    main()

# %%
