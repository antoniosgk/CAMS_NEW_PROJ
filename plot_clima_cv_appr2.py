#%%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
plt.rcParams['font.weight'] = 'bold'
# ============================================================
# USER SETTINGS
# ============================================================
seadn = "all"

PLOT_SECTORS_TOGETHER = True
SUBPLOT_LAYOUT = "5x2"   # "2x5" or "5x2"

SECTOR = "C3"

SECTORS_ALL = [f"C{i}" for i in range(1, 11)]


CSV_TEMPLATE = "/home/agkiokas/CAMS/climatological_plots/station_climatology_summary_{seadn}_{sector}.csv"
OUT_FILE = Path(f"/home/agkiokas/CAMS/climatological_plots/stations_cv_map_all_sectors_{seadn}_appr2.png")
# Column names
LAT_COL = "lat"
LON_COL = "lon"
VALUE_COL = "approach5_mean_cvw_pct"
STATION_COL = "station"
plt.rcParams['font.weight'] = 'bold'
# ============================================================
# MAP SETTINGS
# ============================================================

FIGSIZE = (20, 10)
DPI = 400

CMAP = "jet"

DOT_SIZE = 10
DOT_EDGE_COLOR = "black"
DOT_EDGE_WIDTH = 0.4

# Optional manual limits
VMIN = 0
VMAX = 3

# ============================================================
# TITLE SETTINGS
# ============================================================

TITLE = "Coefficient of Variation C10 Approach 2 (%)"

TITLE_SIZE = 14
TITLE_WEIGHT = "bold"

LABEL_SIZE = 13
LABEL_WEIGHT = "bold"

TICK_SIZE = 13

# ============================================================
# COLORBAR SETTINGS
# ============================================================

CBAR_LABEL = "CV (%)"

CBAR_LABEL_SIZE = 13
CBAR_LABEL_WEIGHT = "bold"

CBAR_TICK_SIZE = 13

# ============================================================
# STATION LABELS
# ============================================================

SHOW_STATION_NAMES = False

STATION_LABEL_SIZE = 7
STATION_LABEL_WEIGHT = "bold"

# ============================================================
# MAP EXTENT
# ============================================================

# None -> automatic
#MAP_EXTENT = None

# Example:
MAP_EXTENT = [70, 135, 15, 52]
def load_sector_csv(seadn, sector):
    csv_file = Path(CSV_TEMPLATE.format(seadn=seadn, sector=sector))

    if not csv_file.exists():
        print(f"Missing file: {csv_file}")
        return None

    df = pd.read_csv(csv_file)

    required_cols = [LAT_COL, LON_COL, VALUE_COL]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {csv_file}")

    df = df.dropna(subset=required_cols).copy()
    df["sector"] = sector

    return df
def plot_all_sectors_subplot(sector_data, out_file):
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

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=False,
    )

    axes = axes.ravel()

    all_vals = pd.concat(
        [df[VALUE_COL] for df in sector_data.values()],
        ignore_index=True
    )

    vmin = VMIN if VMIN is not None else all_vals.min()
    vmax = VMAX if VMAX is not None else all_vals.max()

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    last_sc = None

    for ax, sector in zip(axes, SECTORS_ALL):
        df = sector_data.get(sector)

        if df is None or df.empty:
            ax.set_title(
                f"{sector} - no data",
                fontsize=TITLE_SIZE,
                fontweight=TITLE_WEIGHT
            )
            continue

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="--"
        )

        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        gl.xlocator = mticker.MaxNLocator(4)
        gl.ylocator = mticker.MaxNLocator(4)

        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.2)
        ax.add_feature(cfeature.OCEAN, alpha=0.1)

        if MAP_EXTENT is not None:
            ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        else:
            lon_pad = 1
            lat_pad = 1
            ax.set_extent([
                df[LON_COL].min() - lon_pad,
                df[LON_COL].max() + lon_pad,
                df[LAT_COL].min() - lat_pad,
                df[LAT_COL].max() + lat_pad,
            ])

        last_sc = ax.scatter(
            df[LON_COL],
            df[LAT_COL],
            c=df[VALUE_COL],
            cmap=cmap,
            norm=norm,
            s=DOT_SIZE,
            edgecolor=DOT_EDGE_COLOR,
            linewidth=DOT_EDGE_WIDTH,
            transform=ccrs.PlateCarree(),
            zorder=5
        )

        if SHOW_STATION_NAMES:
            for _, row in df.iterrows():
                ax.text(
                    row[LON_COL],
                    row[LAT_COL],
                    str(row[STATION_COL]),
                    fontsize=STATION_LABEL_SIZE,
                    fontweight=STATION_LABEL_WEIGHT,
                    transform=ccrs.PlateCarree(),
                    zorder=6
                )

        ax.set_title(
            sector,
            fontsize=TITLE_SIZE,
            fontweight=TITLE_WEIGHT
        )

    fig.suptitle(
        f"Climatological CV (%) | all sectors | Approach 2 ",
        fontsize=TITLE_SIZE + 4,
        fontweight=TITLE_WEIGHT
    )

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
        extend="max"
    )

    cbar.set_label(
        CBAR_LABEL,
        fontsize=CBAR_LABEL_SIZE,
        fontweight=CBAR_LABEL_WEIGHT
    )

    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        out_file,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Saved: {out_file}")
# ============================================================
# MAIN
# ============================================================

if not PLOT_SECTORS_TOGETHER:

    # ============================================================
    # SINGLE SECTOR (current behaviour)
    # ============================================================

    df = pd.read_csv(CSV_FILE)

    required_cols = [LAT_COL, LON_COL, VALUE_COL]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")

    df = df.dropna(subset=required_cols).copy()

    #
    # KEEP YOUR CURRENT SINGLE-SECTOR PLOTTING CODE
    #
    # (everything exactly as it is now)
    #

else:

    # ============================================================
    # ALL SECTORS TOGETHER
    # ============================================================

    sector_data = {}

    for sector in SECTORS_ALL:

        df_sector = load_sector_csv(
            seadn=seadn,
            sector=sector
        )

        if df_sector is not None:
            sector_data[sector] = df_sector

    if len(sector_data) == 0:
        raise RuntimeError("No sector files found.")

    OUT_FILE_ALL = Path(
        f"/home/agkiokas/CAMS/climatological_plots/"
        f"stations_cv_map_all_sectors_{seadn}_appr2.png"
    )

    plot_all_sectors_subplot(
        sector_data=sector_data,
        out_file=OUT_FILE_ALL
    )
# %%
