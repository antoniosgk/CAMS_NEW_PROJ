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

#VAR_NAME = "O3_seasonal_mean" #O3_temporal_std,O3_seasonal_std
O3_MW = 48.0
AIR_MW = 28.9647
# ============================================================
# VARIABLE SETTINGS
# ============================================================

# Examples:
# "O3_temporal_mean"
# "O3_temporal_std"
# "O3_valid_count"
# "O3_seasonal_mean"
# "O3_seasonal_std"
USE_COMMON_COLORBAR_FOR_GROUPS = True

SEASONS_TO_PLOT = ["DJF", "MAM", "JJA", "SON"]
DAYNIGHT_TO_PLOT_LIST = ["day", "night"]

VAR_NAME = "O3_seasonal_mean"

# Only used if variable has season dimension
SEASON_TO_PLOT = "JJA"

# Only used if variable has daynight dimension
DAYNIGHT_TO_PLOT = "day"

# Automatically detect dimensions
AUTO_SELECT_COORDS = True

CONVERT_KGKG_TO_PPB = True
LEVEL_TO_PLOT = 72          # last level
PLOT_O3 = True
PLOT_STATIONS = False

#HIGHLIGHT_STATIONS = []   # [] for none

STATION_NAME_COL = "Station_Name"
STATION_LAT_COL = "Latitude"
STATION_LON_COL = "Longitude"

FIGSIZE = (11, 8)
CMAP = "viridis"
MARKER_SIZE = 25
HIGHLIGHT_SIZE = 90

MAP_PADDING_DEG = 0

# BOX / SECTOR SETTINGS
# ============================================================
CENTER_BOXES_ON_GRIDCELL = True
SHOW_STATION_LABELS = False
HIGHLIGHT_STATIONS = False #["1002A"]   # one or more stations to highlight

DRAW_SECTOR_BOXES = True
BOX_STATION = "1002A"

# Plot one or many sectors
BOX_SECTORS = ["C1", "C2","C3","C4", "C5", "C10"]
# BOX_SECTORS = [f"C{i}" for i in range(1, 11)]

DRAW_CAMS_BOX = True
CAMS_BOX_DEG = 0.4        # 0.4 degree box
CAMS_BOX_COLOR = "gray"
CAMS_BOX_LINEWIDTH = 2.0
CAMS_BOX_LABEL = "0.4° simulation box"

SECTOR_LINEWIDTH = 1.2

# pink -> dark red gradient for C1...C10
SECTOR_COLORS = {
    f"C{i}": plt.cm.Reds(0.25 + 0.70 * (i - 1) / 9)
    for i in range(1, 11)
}
# ============================================================
# LOAD DATA
# ============================================================

def load_o3_level():

    ds = xr.open_dataset(NC_FILE)

    if VAR_NAME not in ds.variables:
        raise ValueError(
            f"{VAR_NAME} not found.\n"
            f"Available variables:\n{list(ds.data_vars)}"
        )

    da = ds[VAR_NAME]

    print("\nSelected variable:")
    print(da)

    # --------------------------------------------------------
    # Automatic selections depending on dimensions
    # --------------------------------------------------------

    if "season" in da.dims:
        if SEASON_TO_PLOT is not None:
            da = da.sel(season=SEASON_TO_PLOT)

    if "daynight" in da.dims:
        if DAYNIGHT_TO_PLOT is not None:
           da = da.sel(daynight=DAYNIGHT_TO_PLOT)

    if "lev" in da.dims:
        da = da.sel(lev=LEVEL_TO_PLOT)

    # --------------------------------------------------------
    # Unit conversion only for concentration variables
    # --------------------------------------------------------

    if (
        CONVERT_KGKG_TO_PPB
        and ("mean" in VAR_NAME or "std" in VAR_NAME)
    ):
        da = da * (AIR_MW / O3_MW) * 1e9
        da.attrs["units"] = "ppb"

    return ds, da

def get_common_vmin_vmax(ds):
    da_all = ds[VAR_NAME]

    if "lev" in da_all.dims:
        da_all = da_all.sel(lev=LEVEL_TO_PLOT)

    if CONVERT_KGKG_TO_PPB and ("mean" in VAR_NAME or "std" in VAR_NAME):
        da_all = da_all * (AIR_MW / O3_MW) * 1e9

    if "season" in da_all.dims:
        da_all = da_all.sel(season=SEASONS_TO_PLOT)

    if "daynight" in da_all.dims:
        da_all = da_all.sel(daynight=DAYNIGHT_TO_PLOT_LIST)

    vmin = float(da_all.min(skipna=True).values)
    vmax = float(da_all.max(skipna=True).values)

    return vmin, vmax

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
    Sector size:
    C1  = 3x3 cells
    C2  = 5x5 cells
    ...
    C10 = 21x21 cells

    With 0.0625 deg resolution:
    C1  width ≈ 3  * 0.0625 = 0.1875 deg
    C10 width ≈ 21 * 0.0625 = 1.3125 deg
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

    lon_min = float(lon_1d[i_lon_min])
    lon_max = float(lon_1d[i_lon_max])
    lat_min = float(lat_1d[i_lat_min])
    lat_max = float(lat_1d[i_lat_max])

    return lon_min, lon_max, lat_min, lat_max

def get_degree_box_bounds(st_lon, st_lat, box_deg):
    half = box_deg / 2.0

    lon_min = st_lon - half
    lon_max = st_lon + half
    lat_min = st_lat - half
    lat_max = st_lat + half

    return lon_min, lon_max, lat_min, lat_max
def get_nearest_gridcell_center(st_lon, st_lat, lon, lat):
    """
    Returns nearest grid-cell center to station position.
    Falls back to station lon/lat if something fails.
    """

    try:
        lon_1d = lon if lon.ndim == 1 else lon[0, :]
        lat_1d = lat if lat.ndim == 1 else lat[:, 0]

        i_lon = int(np.argmin(np.abs(lon_1d - st_lon)))
        i_lat = int(np.argmin(np.abs(lat_1d - st_lat)))

        center_lon = float(lon_1d[i_lon])
        center_lat = float(lat_1d[i_lat])

        return center_lon, center_lat, i_lon, i_lat

    except Exception:
        return float(st_lon), float(st_lat), None, None


def get_sector_box_bounds_from_center(center_lon, center_lat, lon, lat, sector):
    """
    Sector boxes centered on nearest grid-cell center.

    C1  = 3x3 cells
    C2  = 5x5 cells
    ...
    C10 = 21x21 cells
    """

    sector_num = int(str(sector).replace("C", ""))
    half_cells = sector_num

    lon_1d = lon if lon.ndim == 1 else lon[0, :]
    lat_1d = lat if lat.ndim == 1 else lat[:, 0]

    i_lon = int(np.argmin(np.abs(lon_1d - center_lon)))
    i_lat = int(np.argmin(np.abs(lat_1d - center_lat)))

    i_lon_min = max(i_lon - half_cells, 0)
    i_lon_max = min(i_lon + half_cells, len(lon_1d) - 1)

    i_lat_min = max(i_lat - half_cells, 0)
    i_lat_max = min(i_lat + half_cells, len(lat_1d) - 1)

    lon_min = float(lon_1d[i_lon_min])
    lon_max = float(lon_1d[i_lon_max])
    lat_min = float(lat_1d[i_lat_min])
    lat_max = float(lat_1d[i_lat_max])

    return lon_min, lon_max, lat_min, lat_max


def get_degree_box_bounds(center_lon, center_lat, box_deg):
    half = box_deg / 2.0

    return (
        center_lon - half,
        center_lon + half,
        center_lat - half,
        center_lat + half,
    )

def plot_map():
    ds, da = load_o3_level()
    vmin, vmax = None, None

    if USE_COMMON_COLORBAR_FOR_GROUPS:
        if "season" in ds[VAR_NAME].dims or "daynight" in ds[VAR_NAME].dims:
            vmin, vmax = get_common_vmin_vmax(ds)
            print(f"Common colorbar: vmin={vmin}, vmax={vmax}")
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
                zorder=1,vmin=vmin,vmax=vmax,
            )
        else:
            mesh = ax.pcolormesh(
                lon,
                lat,
                da.values,
                cmap=CMAP,
                shading="auto",
                transform=ccrs.PlateCarree(),
                zorder=1,vmin=vmin,vmax=vmax,
            )
        
        cbar = plt.colorbar(mesh, ax=ax, shrink=0.8, pad=0.03)

        units = da.attrs.get("units", "")

        if units:
           cbar.set_label(f"{VAR_NAME} ({units})")
        else:
           cbar.set_label(VAR_NAME)

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
        
        # --------------------------------------------------------
    # Sector boxes and CAMS 0.4 degree box
    # --------------------------------------------------------

    if DRAW_SECTOR_BOXES or DRAW_CAMS_BOX:

        box_station = stations[
            stations[STATION_NAME_COL].astype(str) == str(BOX_STATION)
        ]

        if box_station.empty:
            raise ValueError(f"BOX_STATION={BOX_STATION} not found in stations file.")

        st_lon = float(box_station.iloc[0][STATION_LON_COL])
        st_lat = float(box_station.iloc[0][STATION_LAT_COL])

        if CENTER_BOXES_ON_GRIDCELL:
            center_lon, center_lat, i_lon, i_lat = get_nearest_gridcell_center(
                st_lon=st_lon,
                st_lat=st_lat,
                lon=lon,
                lat=lat
            )
            center_label = "nearest grid-cell center"
        else:
            center_lon, center_lat = st_lon, st_lat
            center_label = "station position"
        # -----------------------------
        # CAMS 0.4 degree box
        # -----------------------------
        if DRAW_CAMS_BOX:

            lon_min_box, lon_max_box, lat_min_box, lat_max_box = get_degree_box_bounds(
                center_lon=center_lon,
                center_lat=center_lat,
                box_deg=CAMS_BOX_DEG
            )

            rect = Rectangle(
                (lon_min_box, lat_min_box),
                lon_max_box - lon_min_box,
                lat_max_box - lat_min_box,
                linewidth=CAMS_BOX_LINEWIDTH,
                edgecolor=CAMS_BOX_COLOR,
                facecolor="none",
                linestyle="--",
                transform=ccrs.PlateCarree(),
                zorder=9,
                label=CAMS_BOX_LABEL,
            )

            ax.add_patch(rect)

        # -----------------------------
        # C1...C10 sector boxes
        # -----------------------------
        if DRAW_SECTOR_BOXES:

            for sector in BOX_SECTORS:

                lon_min_box, lon_max_box, lat_min_box, lat_max_box = get_sector_box_bounds_from_center(
                 center_lon=center_lon, center_lat=center_lat,
             lon=lon,lat=lat,sector=sector
            )

                rect = Rectangle(
                    (lon_min_box, lat_min_box),
                    lon_max_box - lon_min_box,
                    lat_max_box - lat_min_box,
                    linewidth=SECTOR_LINEWIDTH,
                    edgecolor=SECTOR_COLORS.get(sector, "red"),
                    facecolor="none",
                    transform=ccrs.PlateCarree(),
                    zorder=10,
                    label=f"{sector}: {2 * int(sector.replace('C', '')) + 1}x{2 * int(sector.replace('C', '')) + 1}",
                )

                ax.add_patch(rect)

               # Actual station position
        '''       
        ax.scatter(
            st_lon,
            st_lat,
            s=HIGHLIGHT_SIZE,
            facecolor="yellow",
            edgecolor="black",
            linewidth=1.5,
            marker="*",
            transform=ccrs.PlateCarree(),
            zorder=12,
            label=f"Station {BOX_STATION}",
        )
'''
        # Grid-cell center used for boxes
        if CENTER_BOXES_ON_GRIDCELL:
            ax.scatter(
                center_lon,
                center_lat,
                s=60,
                facecolor="none",
                edgecolor="black",
                linewidth=1.5,
                marker="o",
                transform=ccrs.PlateCarree(),
                zorder=12,
                label="Box center grid cell",
            )

        if SHOW_STATION_LABELS:
            ax.text(
                st_lon,
                st_lat,
                f" {BOX_STATION}",
                fontsize=9,
                fontweight="bold",
                transform=ccrs.PlateCarree(),
                zorder=13,
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

    if PLOT_STATIONS or HIGHLIGHT_STATIONS or DRAW_SECTOR_BOXES or DRAW_CAMS_BOX:
        ax.legend(loc="lower left", fontsize=8)

    ax.set_title(f"Ozone temporal std - ground level")
    #ax.set_title('Stations positions with C10 sector')
    title = VAR_NAME

    if "season" in da.coords:
        try:
            title += f" | season={str(da['season'].values)}"
        except Exception:
            pass

    if "daynight" in da.coords:
        try:
            title += f" | daynight={str(da['daynight'].values)}"
        except Exception:
            pass

    if "lev" in da.coords:
        try:
            title += f" | lev={float(da['lev'].values)}"
        except Exception:
            pass

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_map()
# %%
