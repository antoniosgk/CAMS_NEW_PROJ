#%%
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

# ============================================================
# SETTINGS
# ============================================================

NC_FILE = Path("/mnt/store01/agkiokas/CAMS/O3_temporal_mean_std_season_daynight.nc")
STATIONS_TXT = Path("/home/agkiokas/CAMS/CHINESE_STATIONS_INFO_2015_2023.txt")

OUT_FILE = Path("/mnt/store01/agkiokas/CAMS/O3_temporal_mean_level22_stations.png")

VAR_NAME = "O3_temporal_std" #O3_temporal_std,O3_seasonal_std
O3_MW = 48.0
AIR_MW = 28.9647

CONVERT_KGKG_TO_PPB = True
LEVEL_TO_PLOT = 72          # last level
PLOT_O3 = True
PLOT_STATIONS = False

HIGHLIGHT_STATIONS = []   # [] for none

STATION_NAME_COL = "Station_Name"
STATION_LAT_COL = "Latitude"
STATION_LON_COL = "Longitude"

FIGSIZE = (11, 8)
CMAP = "viridis"
MARKER_SIZE = 25
HIGHLIGHT_SIZE = 90

MAP_PADDING_DEG = 0

DRAW_SECTOR_BOX = False
BOX_STATION = "1002A"      # station around which to draw the box
BOX_SECTOR = "C10"         # C1 ... C10

BOX_EDGE_COLOR = "red"
BOX_LINEWIDTH = 1.0
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
def get_sector_box_bounds(st_lon, st_lat, lon, lat, sector):
    """
    C1 = 3x3 grid cells
    C2 = 5x5 grid cells
    ...
    C10 = 21x21 grid cells
    """

    sector_num = int(str(sector).replace("C", ""))
    half_cells = sector_num

    lon_1d = lon if lon.ndim == 1 else lon[0, :]
    lat_1d = lat if lat.ndim == 1 else lat[:, 0]

    i_lon = int(np.argmin(np.abs(lon_1d - st_lon)))
    i_lat = int(np.argmin(np.abs(lat_1d - st_lat)))

    i_lon_min = max(i_lon - half_cells, 0)
    i_lon_max = min(i_lon + half_cells, len(lon_1d) - 1)

    i_lat_min = max(i_lat - half_cells, 0)
    i_lat_max = min(i_lat + half_cells, len(lat_1d) - 1)

    lon_min = lon_1d[i_lon_min]
    lon_max = lon_1d[i_lon_max]
    lat_min = lat_1d[i_lat_min]
    lat_max = lat_1d[i_lat_max]

    return lon_min, lon_max, lat_min, lat_max

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
        cbar.set_label(f"ozone temporal std ({units})")

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
        '''
        ax.set_extent(
        [
            105 ,
            130 + MAP_PADDING_DEG,
            30 - MAP_PADDING_DEG,
            50 + MAP_PADDING_DEG,
        ],
        crs=ccrs.PlateCarree(),
    )
 '''
        
    if DRAW_SECTOR_BOX:
        box_station = stations[stations[STATION_NAME_COL].astype(str) == str(BOX_STATION)]

        if box_station.empty:
            raise ValueError(f"BOX_STATION={BOX_STATION} not found in stations file.")

        st_lon = float(box_station.iloc[0][STATION_LON_COL])
        st_lat = float(box_station.iloc[0][STATION_LAT_COL])

        lon_min_box, lon_max_box, lat_min_box, lat_max_box = get_sector_box_bounds(
            st_lon=st_lon,
            st_lat=st_lat,
            lon=lon,
            lat=lat,
            sector=BOX_SECTOR
        )

        rect = Rectangle(
            (lon_min_box, lat_min_box),
            lon_max_box - lon_min_box,
            lat_max_box - lat_min_box,
            linewidth=BOX_LINEWIDTH,
            edgecolor=BOX_EDGE_COLOR,
            facecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=8,
            label=f"{BOX_SECTOR} sector"
        )

        ax.add_patch(rect)



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

    ax.set_title(f"Ozone temporal std - ground level")
    #ax.set_title('Stations positions with C10 sector')
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_map()
# %%
