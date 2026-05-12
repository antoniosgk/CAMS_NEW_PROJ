#%%
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ============================================================
# SETTINGS
# ============================================================

NC_FILE = Path("/mnt/store01/agkiokas/CAMS/O3_temporal_mean.nc")
STATIONS_TXT = Path("/home/agkiokas/CAMS/CHINESE_STATIONS_INFO_2015_2023.txt")

OUT_FILE = Path("/mnt/store01/agkiokas/CAMS/O3_temporal_mean_level22_stations.png")

VAR_NAME = "O3_temporal_mean"
O3_MW = 48.0
AIR_MW = 28.9647

CONVERT_KGKG_TO_PPB = True
LEVEL_TO_PLOT = 72          # last level
PLOT_O3 = True
PLOT_STATIONS = True

HIGHLIGHT_STATIONS = []   # [] for none

STATION_NAME_COL = "Station_Name"
STATION_LAT_COL = "Latitude"
STATION_LON_COL = "Longitude"

FIGSIZE = (11, 8)
CMAP = "viridis"
MARKER_SIZE = 25
HIGHLIGHT_SIZE = 90

MAP_PADDING_DEG = 0


# ============================================================
# LOAD DATA
# ============================================================

def load_o3_level():
    ds = xr.open_dataset(NC_FILE)

    da = ds[VAR_NAME].sel(lev=LEVEL_TO_PLOT)

    if CONVERT_KGKG_TO_PPB:
        # kg/kg -> mol/mol -> ppb
        da = da * (AIR_MW / O3_MW) * 1e9
        da.attrs["units"] = "ppb"

    return ds, da


def load_stations_txt():
    df = pd.read_csv(
        STATIONS_TXT,
        sep=None,              # auto-detect separator
        engine="python",
    )

    needed = [STATION_NAME_COL, STATION_LAT_COL, STATION_LON_COL]
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise ValueError(
            f"Station txt missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[needed].copy()

    df[STATION_NAME_COL] = df[STATION_NAME_COL].astype(str)
    df[STATION_LAT_COL] = pd.to_numeric(df[STATION_LAT_COL], errors="coerce")
    df[STATION_LON_COL] = pd.to_numeric(df[STATION_LON_COL], errors="coerce")

    df = df.dropna(subset=[STATION_LAT_COL, STATION_LON_COL])

    return df


# ============================================================
# PLOT
# ============================================================

def plot_map():
    ds, da = load_o3_level()
    stations = load_stations_txt()

    lon = ds["lon"].values if "lon" in ds else ds["longitude"].values
    lat = ds["lat"].values if "lat" in ds else ds["latitude"].values

    fig = plt.figure(figsize=FIGSIZE)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # --------------------------------------------------------
    # Extent
    # --------------------------------------------------------

    if PLOT_O3:
        lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
        lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    else:
        lon_min = float(stations[STATION_LON_COL].min())
        lon_max = float(stations[STATION_LON_COL].max())
        lat_min = float(stations[STATION_LAT_COL].min())
        lat_max = float(stations[STATION_LAT_COL].max())

    ax.set_extent(
        [
            lon_min - MAP_PADDING_DEG,
            lon_max + MAP_PADDING_DEG,
            lat_min - MAP_PADDING_DEG,
            lat_max + MAP_PADDING_DEG,
        ],
        crs=ccrs.PlateCarree(),
    )

    # --------------------------------------------------------
    # Cartopy features
    # --------------------------------------------------------

    ax.coastlines(resolution="10m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    # --------------------------------------------------------
    # O3 field
    # --------------------------------------------------------

    if PLOT_O3:
        if lat.ndim == 1 and lon.ndim == 1:
            mesh = ax.pcolormesh(
                lon,
                lat,
                da.values,
                cmap=CMAP,
                shading="auto",
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
        else:
            mesh = ax.pcolormesh(
                lon,
                lat,
                da.values,
                cmap=CMAP,
                shading="auto",
                transform=ccrs.PlateCarree(),
                zorder=1,
            )

        cbar = plt.colorbar(mesh, ax=ax, shrink=0.8, pad=0.03)
        units = da.attrs.get("units", "")
        cbar.set_label(f"ozone temporal mean ({units})")

    # --------------------------------------------------------
    # Station markers
    # --------------------------------------------------------

    if PLOT_STATIONS:
        ax.scatter(
            stations[STATION_LON_COL],
            stations[STATION_LAT_COL],
            s=MARKER_SIZE,
            facecolor="black",
            edgecolor="black",
            linewidth=0.7,
            marker="o",
            transform=ccrs.PlateCarree(),
            zorder=5,
            label="Stations",
        )

    # --------------------------------------------------------
    # Highlight selected stations
    # --------------------------------------------------------

    if HIGHLIGHT_STATIONS:
        highlight = stations[
            stations[STATION_NAME_COL].isin([str(s) for s in HIGHLIGHT_STATIONS])
        ]

        ax.scatter(
            highlight[STATION_LON_COL],
            highlight[STATION_LAT_COL],
            s=HIGHLIGHT_SIZE,
            facecolor="none",
            edgecolor="red",
            linewidth=2.0,
            marker="o",
            transform=ccrs.PlateCarree(),
            zorder=7,
            label="Highlighted stations",
        )

        for _, row in highlight.iterrows():
            ax.text(
                row[STATION_LON_COL],
                row[STATION_LAT_COL],
                row[STATION_NAME_COL],
                fontsize=8,
                fontweight="bold",
                transform=ccrs.PlateCarree(),
                zorder=8,
            )

    if PLOT_STATIONS or HIGHLIGHT_STATIONS:
        ax.legend(loc="lower left")

    ax.set_title(f"Ozone temporal mean - ground level")
    #ax.set_title('Stations positions')
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_map()
# %%
