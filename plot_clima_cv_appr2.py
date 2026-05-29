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
seadn='DJF' #all,day,night,MAM,SON,JJA,DJF
SECTOR="C3"
CSV_FILE = Path(f"/home/agkiokas/CAMS/climatological_plots/station_climatology_summary_{seadn}_{SECTOR}.csv")
OUT_FILE = Path(f"/home/agkiokas/CAMS/climatological_plots/stations_cv_map_{SECTOR}_{seadn}(appr2).png")

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

CMAP = "viridis"

DOT_SIZE = 100
DOT_EDGE_COLOR = "black"
DOT_EDGE_WIDTH = 0.4

# Optional manual limits
VMIN = 1
VMAX = 4

# ============================================================
# TITLE SETTINGS
# ============================================================

TITLE = "Coefficient of Variation C10 Approach 2 (%)"

TITLE_SIZE = 16
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

# ============================================================
# MAIN
# ============================================================

df = pd.read_csv(CSV_FILE)

required_cols = [LAT_COL, LON_COL, VALUE_COL]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in CSV.")

df = df.dropna(subset=required_cols).copy()

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

# ============================================================
# MAP FEATURES
# ============================================================

ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, alpha=0.2)
ax.add_feature(cfeature.OCEAN, alpha=0.1)

# ============================================================
# MAP EXTENT
# ============================================================

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

# ============================================================
# STATION SCATTER
# ============================================================

norm = mpl.colors.Normalize(
    vmin=VMIN if VMIN is not None else df[VALUE_COL].min(),
    vmax=VMAX if VMAX is not None else df[VALUE_COL].max()
)

sc = ax.scatter(
    df[LON_COL],
    df[LAT_COL],
    c=df[VALUE_COL],
    cmap=CMAP,
    norm=norm,
    s=DOT_SIZE,
    edgecolor=DOT_EDGE_COLOR,
    linewidth=DOT_EDGE_WIDTH,
    transform=ccrs.PlateCarree(),
    zorder=5
)

# ============================================================
# OPTIONAL STATION LABELS
# ============================================================

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

# ============================================================
# COLORBAR
# ============================================================

cbar = plt.colorbar(
    sc,
    ax=ax,
    shrink=0.95,
    pad=0.03,extend='both'
)

cbar.set_label(
    CBAR_LABEL,
    fontsize=CBAR_LABEL_SIZE,
    fontweight=CBAR_LABEL_WEIGHT
)

cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)

# ============================================================
# TITLE
# ============================================================

ax.set_title(
    TITLE,
    fontsize=TITLE_SIZE,
    fontweight=TITLE_WEIGHT
)

# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUT_FILE,
    dpi=DPI,
    bbox_inches="tight"
)

plt.show()

print(f"Saved: {OUT_FILE}")
# %%
