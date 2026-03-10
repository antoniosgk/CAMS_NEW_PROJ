# plots.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from calculation import haversine_km
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import os
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import pandas as pd

def _gradual_reds(n, cmap_name="Reds", start=0.85, end=0.25):
    """
    Returns n colors sampled from a red colormap from dark->light.
    start/end are positions in [0,1] within the colormap.
    """
    if n <= 0:
        return []
    cmap = mpl.colormaps.get_cmap(cmap_name)
    vals = np.linspace(start, end, n)
    return [cmap(v) for v in vals]

def get_sector_colormap(n_sectors, cmap_name="viridis"):
    """
    Returns a consistent list of colors for sectors.
    Used everywhere (rectangles, sector CV plots, ratio plots).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    cmap = plt.get_cmap(cmap_name)

    colors = [cmap(i) for i in np.linspace(0, 1, n_sectors)]

    return colors, cmap

def sector_color_mapping(labels=None, n=None, cmap_name="Reds",
                         start=0.85, end=0.25, base0_color="black"):
    """
    Builds a deterministic color mapping for labels OR returns a color list.

    - If labels include C0 or d0 (case-insensitive), those get base0_color.
    - The remaining labels get a gradual red palette (dark->light).

    Returns:
      - If labels is provided: dict {label: color}
      - Else: list of colors length n
    """
    if labels is None:
        if n is None:
            raise ValueError("sector_color_mapping: provide either labels or n.")
        return _gradual_reds(n, cmap_name=cmap_name, start=start, end=end)

    labels = list(labels)
    lower = [str(x).lower() for x in labels]

    # Detect "center" bin (C0 or d0)
    is_base0 = [("c0" == s) or ("d0" == s) for s in lower]

    # Build palette for non-base0 items
    n_non0 = sum(not b for b in is_base0)
    reds = _gradual_reds(n_non0, cmap_name=cmap_name, start=start, end=end)

    mapping = {}
    ridx = 0
    for lab, b0 in zip(labels, is_base0):
        if b0:
            mapping[lab] = base0_color
        else:
            mapping[lab] = reds[ridx]
            ridx += 1
    return mapping

def _sanitize_filename(s: str) -> str:
    # keep it filesystem-safe
    s = re.sub(r"\s+", "_", str(s).strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "", s)
    return s


def save_figure(fig, out_dir, filename_base, dpi=200):
    """
    Save a matplotlib figure as PNG into out_dir with a safe filename.
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = _sanitize_filename(filename_base) + ".png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path

def build_meta_title(meta,kind=""):
    """
    kind: optional short label like 'Map', 'Profile P–T', etc.
    """
    if meta is None:
        return kind

    header = f"{meta['species']} ({meta['units']}) at {meta['time_str']}"
    if kind:
        header = f"{kind} | " + header

    line2 = (
        f"Station {meta['station_name']}: "
        f"({meta['station_lat']:.2f}, {meta['station_lon']:.2f}), alt={meta['station_alt']:.1f} m"
    )
    line3 = (
        f"Model: ({meta['model_lat']:.2f}, {meta['model_lon']:.2f}), "
        f"lev={meta['k_star_center']+1}, alt={meta['z_star_m']:.1f} m"
    )
    mode=meta.get("mode","").upper()
    if mode=="A":
        line4=(f'Same model level for the whole area')
    elif mode=="HEIGHT":
        line4=(f'Same altitude ASL for the whole area')    
    else:
        line4=""    
    
    return header + "\n" + line2 + "\n" + line3 + "\n" + line4

def plot_variable_on_map(
    lats_small,
    lons_small,
    data_arr,
    # station
    lon_s,
    lat_s,
    # labels/meta
    units="",
    species_name="var",
    time_str=None,
    meta=None,
    # map control
    d=0.4,
    proj=None,
    ax=None,
    # terrain background on a DIFFERENT grid (always 1D)
    lats_terrain=None,
    lons_terrain=None,
    z_orog_m=None,
    terrain_alpha=0.5,
    field_alpha=0.8,
    add_orog_contours=True,cmap="viridis",vmin=None,vmax=None,
    plot_species=True,plot_orography=False
    ):
    

    # --- Projection / axes (must be GeoAxes) ---
    if proj is None:
        proj = ccrs.PlateCarree()

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": proj})
    else:
        fig = ax.figure

    # --- extent around station (lon/lat degrees => PlateCarree) ---
    lon_min, lon_max = lon_s - d, lon_s + d
    lat_min, lat_max = lat_s - d, lat_s + d
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- ocean background first (so masked sea shows this color) ---
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=-2)

    # --- TERRAIN underlay (optional) ---
    terrain_im = None
    if plot_orography and (z_orog_m is not None) and (lats_terrain is not None) and (lons_terrain is not None):
        lats_terrain = np.asarray(lats_terrain, dtype=float)
        lons_terrain = np.asarray(lons_terrain, dtype=float)
        z_orog_m = np.asarray(z_orog_m, dtype=float)

        LON_T, LAT_T = np.meshgrid(lons_terrain, lats_terrain)

        # Mask sea so it doesn't use the terrain colormap
        z_plot = np.ma.masked_where(~np.isfinite(z_orog_m) | (z_orog_m <= 0.0), z_orog_m)

        # Use land-only min/max for better contrast (avoid sea pulling vmin down)
        if np.ma.is_masked(z_plot):
            zmin = float(z_plot.min())
            zmax = float(z_plot.max())
        else:
            zmin = float(np.nanmin(z_orog_m))
            zmax = float(np.nanmax(z_orog_m))

        terrain_im = ax.pcolormesh(
            LON_T, LAT_T, z_plot,
            cmap="terrain",
            shading="auto",
            vmin=zmin, vmax=zmax,
            alpha=terrain_alpha,
            transform=ccrs.PlateCarree(),
            zorder=-1,
        )

        if add_orog_contours:
            step_m = 200.0
            levels = np.arange(np.floor(zmin / step_m) * step_m,
                               np.ceil(zmax / step_m) * step_m + step_m,
                               step_m)
            ax.contour(
                LON_T, LAT_T, z_orog_m,
                levels=levels,
                colors="k",
                linewidths=0.4,
                alpha=0.30,
                transform=ccrs.PlateCarree(),
                zorder=0,
            )

    # --- Species overlay (small domain) ---
    lats_small = np.asarray(lats_small, dtype=float)
    lons_small = np.asarray(lons_small, dtype=float)
    data_arr = np.asarray(data_arr, dtype=float)

    
    # ✅ Use external vmin/vmax if provided; otherwise compute from this data_arr
    if vmin is None:
        vmin = float(np.nanmin(data_arr))
    if vmax is None:
        vmax = float(np.nanmax(data_arr))

    norm = Normalize(vmin=vmin, vmax=vmax)

    im = None
    if plot_species:
      LON_S, LAT_S = np.meshgrid(lons_small, lats_small)
      cmap_obj = plt.get_cmap(cmap).copy()
      cmap_obj.set_bad("white")   #  HEIGHT-mode NaNs (below-ground) become white
      im = ax.pcolormesh(
        LON_S, LAT_S, data_arr,
        cmap=cmap_obj,
        shading="auto",
        norm=norm,
        transform=ccrs.PlateCarree(),
        alpha=field_alpha,
        zorder=2,
    )


    # station marker
    ax.plot(
        lon_s, lat_s, "kx",
        markersize=12,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    # coast/borders on top
    ax.coastlines(resolution="10m", linewidth=0.8)
    #ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=5)

    # --- Two colorbars (terrain left, species right) ---
    cb_w=0.02
    cb_h=0.60
    cb_y=0.18
    #left colorbar(terrain)
    if terrain_im is not None:
        cax_terr=fig.add_axes([0.01,cb_y,cb_w,cb_h])
        cb_terr=fig.colorbar(terrain_im,cax=cax_terr)
        cb_terr.set_label("Elevation (m)")

    #right colorbar (species)
    if plot_species:
      cax_sp=fig.add_axes([0.8,cb_y,cb_w,cb_h])
      cb_sp=fig.colorbar(im,cax=cax_sp)
      cb_sp.set_label(units)    
    
    if not plot_species and terrain_im is not None:
      ax.set_title(
        f"Station {meta['station_name']} "
        f"({meta['station_lat']:.3f}°, {meta['station_lon']:.3f}°), "
        f"{meta['station_alt']:.0f} m ASL\n"
        "Topography map",
        pad=18,
    )


    # -----------------------------
    # Gridlines with DMS labels
    # -----------------------------
    lon_span = float(abs(lon_max - lon_min))
    lat_span = float(abs(lat_max - lat_min))
    span = max(lon_span, lat_span)
    # target at least 4 tick marks (3 intervals) across the span
# step_raw is "span/3", then snap to a "nice" step
    target_intervals = 3.0
    step_raw = span / target_intervals if span > 0 else 0.1

# snap to "nice" steps (degrees)
    nice_steps = np.array([0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
    step = float(nice_steps[np.argmin(np.abs(nice_steps - step_raw))])

# safeguard: if extremely zoomed in, keep it fine
    step = max(step, 0.02)
    lon0 = np.floor(lon_min / step) * step
    lon1 = np.ceil(lon_max / step) * step
    lat0 = np.floor(lat_min / step) * step
    lat1 = np.ceil(lat_max / step) * step

    xticks = np.round(np.arange(lon0, lon1 + 0.5 * step, step), 6)
    yticks = np.round(np.arange(lat0, lat1 + 0.5 * step, step), 6)
    for _ in range(2):
        if xticks.size < 3 or yticks.size < 3:
         step = step / 2.0
         lon0 = np.floor(lon_min / step) * step
         lon1 = np.ceil(lon_max / step) * step
         lat0 = np.floor(lat_min / step) * step
         lat1 = np.ceil(lat_max / step) * step
         xticks = np.round(np.arange(lon0, lon1 + 0.5 * step, step), 6)
         yticks = np.round(np.arange(lat0, lat1 + 0.5 * step, step), 6)
        else:
         break
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.6,
        alpha=0.35,
        linestyle="--",
        zorder=5,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = ticker.FixedLocator(xticks)
    gl.ylocator = ticker.FixedLocator(yticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    ax.set_title(build_meta_title(meta, kind="Map with Station"), pad=18)
    fig.subplots_adjust(left=0.1,right=0.9,top=0.82,bottom=0.1)  # keep title readable

    return fig, ax, im

def plot_rectangles(
    ax,
    lats_small,
    lons_small,
    ii,
    jj,
    im,
    units="",
    species_name="var",
    time_str=None,
    meta=None,
    radii=None,
    cmap_name="Reds",
    start=0.85,
    end=0.25,
    base_center_color="black"
):
    import numpy as np
    import cartopy.crs as ccrs
    from matplotlib.patches import Rectangle

    # ---------- SAFETY ----------
    if radii is None:
        raise ValueError("plot_rectangles: 'radii' must be provided (e.g. [1,2,3]).")

    lats_small = np.asarray(lats_small, dtype=float)
    lons_small = np.asarray(lons_small, dtype=float)

    if not (0 <= ii < len(lats_small) and 0 <= jj < len(lons_small)):
        raise IndexError(f"(ii,jj)=({ii},{jj}) out of bounds.")

    # ---------- GRID SPACING ----------
    dlon = float(np.abs(lons_small[1] - lons_small[0]))
    dlat = float(np.abs(lats_small[1] - lats_small[0]))

    cx = float(lons_small[jj])
    cy = float(lats_small[ii])

   # Optional: draw the center cell outline in black (matches C0/d0 behavior)
    center_w = dlon
    center_h = dlat
    ax.add_patch(
        Rectangle(
            (cx - center_w / 2, cy - center_h / 2),
            center_w,
            center_h,
            facecolor="none",
            edgecolor=base_center_color,
            linewidth=2.5,
            transform=ccrs.PlateCarree(),
            zorder=20,
        )
    )

    # Colors for C1..Ck (k=len(radii))
    rect_colors = sector_color_mapping(n=len(radii), cmap_name=cmap_name, start=start, end=end)

    # ---------- DRAW RECTANGLES ----------
    for k, r in enumerate(radii, start=1):
        color = rect_colors[(k - 1)]

        width = (2 * r + 1) * dlon
        height = (2 * r + 1) * dlat
        left = cx - width / 2
        bottom = cy - height / 2

        rect = Rectangle(
            (left, bottom),
            width,
            height,
            facecolor="none",
            edgecolor=color,       
            linewidth=2.5,         
            transform=ccrs.PlateCarree(),
            zorder=20              # always above fields & terrain
        )
        ax.add_patch(rect)

    # ---------- TITLE ----------
    if meta is not None:
        ax.set_title(build_meta_title(meta, kind="Map with Sectors"), pad=22)
        ax.figure.subplots_adjust(top=0.84)

    return ax, im



def _sort_by_pressure_with_index(p_hPa, idx_level, *arrays):
    """
    Sort profiles from surface (max p) to top (min p),
    and return the new index of the selected level.

    Returns
    -------
    p_sorted, arrays_sorted..., idx_sorted
    """
    p_hPa = np.asarray(p_hPa)
    order = np.argsort(p_hPa)[::-1]  # descending: surface → top

    p_sorted = p_hPa[order]
    sorted_arrays = []
    for arr in arrays:
        if arr is None:
            sorted_arrays.append(None)
        else:
            arr = np.asarray(arr)
            sorted_arrays.append(arr[order])

    # find where the original idx_level moved to
    idx_sorted = int(np.where(order == idx_level)[0][0])

    return (p_sorted, *sorted_arrays, idx_sorted)
def plot_cv_cumulative_sectors(stats_unw, stats_w, title=None, ax=None):
    """
    Line plot of CV (unweighted vs area-weighted)
    for cumulative sectors C1, C2, ...
    """

    cv_unw = [d["cv"] for d in stats_unw]
    cv_w   = [d["cv_w"] for d in stats_w]
    x = np.arange(1, len(cv_unw) + 1)
    cv_unw=pd.to_numeric(cv_unw,errors="coerce")
    cv_w=pd.to_numeric(cv_w,errors="coerce")
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(x, cv_unw * 100, marker="o", linewidth=2,
            label="CV (unweighted)")
    ax.plot(x, cv_w * 100 , marker="s", linewidth=2,
            label="CV (area-weighted)")

    ax.set_xlabel("Sector")
    ax.set_ylabel("Coefficient of Variation %")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}" for k in x])
    ax.grid(True, linestyle="--", alpha=0.4)
    #ax.set_ylim(0.0,20)
    ax.legend()

    if title:
        ax.set_title(title)

    return fig, ax
def plot_cv_vs_distance(df_unw, df_w=None, ax=None, title=None):
    """
    Line plot of CV vs distance.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    df_unw['cv']=pd.to_numeric(df_unw['cv'],errors="coerce")
    df_w['cv_w']=pd.to_numeric(df_w['cv_w'],errors="coerce")
    ax.plot(df_unw["dmax_km"], df_unw["cv"]*100, marker="o", label="Unweighted")
    
    if df_w is not None:
        ax.plot(df_w["dmax_km"], df_w["cv_w"]*100, marker="s", label="Area-weighted")

    ax.set_xlabel("Distance from station (km)")
    ax.set_ylabel("Coefficient of Variation %")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    if title:
        ax.set_title(title)

    return fig, ax

def plot_profile_P_T(p_prof_Pa, T_prof_K, idx_level,
                     time_str=None, ax=None,meta=None):
    """
    Plot vertical profile: Pressure (hPa) vs Temperature (°C)
    for a single grid cell, with a red dot at idx_level.

    Parameters
    ----------
    p_prof_Pa : 1D array, pressure in Pa
    T_prof_K  : 1D array, temperature in K
    idx_level : int, selected model level index (0-based)
    time_str  : optional, string to show in title (e.g. '2025-12-15 00:00 UTC')
    ax        : optional matplotlib Axes

    Returns (fig, ax)
    """
    p_hPa = np.asarray(p_prof_Pa) / 100.0
    T_C = np.asarray(T_prof_K) - 273.15

    # Sort from surface (max p) to top (min p), track level index
    p_sorted, T_sorted, idx_sorted = _sort_by_pressure_with_index(
        p_hPa, idx_level, T_C
    )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # main profile
    ax.plot(T_sorted, p_sorted, "-o")

    # red dot at selected level
    ax.scatter(T_sorted[idx_sorted], p_sorted[idx_sorted],
               color="red", zorder=3, label="Selected level")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title(build_meta_title(meta, kind="Profile T-P"))
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    return fig, ax

def plot_profile_T_Z(T_prof_K, z_prof_m, idx_level,
                     time_str=None, z_units="km", ax=None,meta=None):
    """
    Plot vertical profile: Temperature (°C) vs Height (Z),
    with red dot at idx_level.

    Parameters
    ----------
    T_prof_K  : 1D array, temperature in K
    z_prof_m  : 1D array, height in m (ASL)
    idx_level : int, selected model level index
    time_str  : optional, string for title
    z_units   : 'km' or 'm'
    ax        : optional Axes

    Returns (fig, ax)
    """
    T_C = np.asarray(T_prof_K) - 273.15
    z_m = np.asarray(z_prof_m)

    # sort by height ascending (surface → top)
    order = np.argsort(z_m)
    z_sorted = z_m[order]
    T_sorted = T_C[order]
    idx_sorted = int(np.where(order == idx_level)[0][0])

    if z_units == "km":
        z_vals = z_sorted / 1000.0
        ylabel = "Height (km)"
    else:
        z_vals = z_sorted
        ylabel = "Height (m)"

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(T_sorted, z_vals, "-o")
    ax.scatter(T_sorted[idx_sorted], z_vals[idx_sorted],
               color="red", zorder=3, label="Selected level")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(ylabel)
    ax.set_title(build_meta_title(meta, kind="Profile T–Z"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    return fig, ax
def plot_profile_species_Z(z_prof_m, species_prof, idx_level,
                           species_name="species", species_units="",
                           time_str=None, z_units="km", ax=None,meta=None):
    """
    Plot vertical profile: species vs Height (Z),
    with red dot at idx_level.

    Parameters
    ----------
    z_prof_m     : 1D array, height in m (ASL)
    species_prof : 1D array, species values
    idx_level    : int, selected model level index
    species_name : name of species
    species_units: units of species
    time_str     : optional string for title
    z_units      : 'km' or 'm'
    ax           : optional Axes

    Returns (fig, ax)
    """
    z_m = np.asarray(z_prof_m)
    sp = np.asarray(species_prof)

    # sort by height ascending
    order = np.argsort(z_m)
    z_sorted = z_m[order]
    sp_sorted = sp[order]
    idx_sorted = int(np.where(order == idx_level)[0][0])

    if z_units == "km":
        z_vals = z_sorted / 1000.0
        ylabel = "Height (km)"
    else:
        z_vals = z_sorted
        ylabel = "Height (m)"

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(sp_sorted, z_vals, "-o")
    ax.scatter(sp_sorted[idx_sorted], z_vals[idx_sorted],
               color="red", zorder=3, label="Selected level")

    xlabel = species_name
    if species_units:
        xlabel += f" ({species_units})"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title(build_meta_title(meta, kind="Profile concentration-Height"))


    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    return fig, ax

def plot_cv_bars_distance_both(df_cv_unw, df_cv_w, ax=None, title=None, reverse=True):
    """
    Grouped bar plot of CV vs distance bins:
    - Unweighted: 'cv'
    - Area-weighted: 'cv_w'

    Both DataFrames must refer to the same bins and contain:
      - either 'bin_label' OR 'dmax_km'
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    du = df_cv_unw.copy()
    dw = df_cv_w.copy()

    if "cv" not in du.columns:
        raise KeyError("df_cv_unw must contain column 'cv'.")
    if "cv_w" not in dw.columns:
        raise KeyError("df_cv_w must contain column 'cv_w'.")

    # Build labels consistently from dmax_km (preferred) or bin_label
    def _make_labels(df):
        if "bin_label" in df.columns:
            return df["bin_label"].astype(str)
        if "dmax_km" in df.columns:
            return df["dmax_km"].apply(lambda x: f"≤ {float(x):g} km")
        raise KeyError("DataFrame must contain either 'bin_label' or 'dmax_km'.")

    du["label"] = _make_labels(du)
    dw["label"] = _make_labels(dw)

    du["cv"] = pd.to_numeric(du["cv"], errors="coerce")
    dw["cv_w"] = pd.to_numeric(dw["cv_w"], errors="coerce")

    du = du[np.isfinite(du["cv"])].dropna(subset=["label"])
    dw = dw[np.isfinite(dw["cv_w"])].dropna(subset=["label"])

    # Merge on labels to ensure same order/bins
    m = du[["label", "cv"]].merge(dw[["label", "cv_w"]], on="label", how="inner")

    # If dmax_km exists in both, sort by numeric bin edge for correct order
    if ("dmax_km" in du.columns) and ("dmax_km" in dw.columns):
        # rebuild a numeric key for sorting using du's dmax_km (assumes same bins)
        du_key = du[["label", "dmax_km"]].drop_duplicates()
        m = m.merge(du_key, on="label", how="left")
        m = m.sort_values("dmax_km")
        m = m.drop(columns=["dmax_km"])
    # else keep merge order

    if reverse:
        m = m.iloc[::-1].reset_index(drop=True)

    labels = m["label"].tolist()
    y_unw = m["cv"].to_numpy(dtype=float)*100
    y_w = m["cv_w"].to_numpy(dtype=float) *100

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure
    
    x = np.arange(len(labels))
    width = 0.38

    ax.bar(x - width/2, y_unw, width=width, label="Unweighted")
    ax.bar(x + width/2, y_w,   width=width, label="Area-weighted")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Coefficient of Variation %")
    ax.grid(True, linestyle="--", alpha=0.35)
    if title:
        ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig, ax

def plot_cv_bars_sector_both(stats_unw, stats_w, title=None, ax=None,cmap_name="Reds",
                           start=0.85, end=0.25):
    """
    Bar plot of CV (unweighted vs area-weighted) for cumulative sectors C1, C2, ...
    X-axis ticks: 1..N (Sector), matching the style in your image.
    """
    cv_unw = [d.get("cv", np.nan) for d in stats_unw]
    cv_w   = [d.get("cv_w", np.nan) for d in stats_w]
    n = len(cv_unw)
    x = np.arange(1,n+1)
    cv_unw = pd.to_numeric(cv_unw, errors="coerce") * 100.0
    cv_w   = pd.to_numeric(cv_w,   errors="coerce") * 100.0

    labels = [f"{i}" for i in range(1, n + 1)]
    # Sector colors (C1..Cn)
    colors = sector_color_mapping(n=n, cmap_name=cmap_name, start=start, end=end)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    width = 0.38
    ax.bar(
        x - width / 2,
        cv_unw,
        width=width,
        label="Unweighted",
        color=colors,
        alpha=0.55,
        edgecolor="black",
        linewidth=0.6,
        hatch="///",
    )
    ax.bar(
        x + width / 2,
        cv_w,
        width=width,
        label="Area-weighted",
        color=colors,
        alpha=0.95,
        edgecolor="black",
        linewidth=0.6,
    )
    # --- match x-axis look like your screenshot ---
    ax.set_xlim(0.5, n + 0.5)   # nice margins like the line plot
    # --------------------------------------------
    ax.set_xlabel("Sector")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel("Coefficient of Variation %")
    ax.grid(True, linestyle="--", alpha=0.35)

    if title:
        ax.set_title(title)

    ax.legend()
    return fig, ax

def plot_ratio_bars(df_ratio, ax=None, title=None, ylabel="Mean / center value",
                    xlabel='Distance',cmap_name="Reds", start=0.85, end=0.25):
    """
    df_ratio: DataFrame with columns ["label", "ratio"]
     Bars are colored consistently with sectors:
    - if label contains C0 or d0 -> black
    - remaining -> gradual reds (dark->light)
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    labels = list(df_ratio["label"].astype(str))
    mapping = sector_color_mapping(labels=labels, cmap_name=cmap_name, start=start, end=end, base0_color="black")
    bar_colors = [mapping[l] for l in labels]    

    ax.bar(labels, df_ratio["ratio"], color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, linestyle="--", linewidth=1)  # reference line at 1
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    rmin = np.nanmin(df_ratio["ratio"].values.astype(float))
    rmax = np.nanmax(df_ratio["ratio"].values.astype(float))
    if np.isfinite(rmin) and np.isfinite(rmax):
        # Expand slightly around observed range but keep close-to-1 feel
        pad = max(0.001, 0.05 * (rmax - rmin))
        lo = min(0.98, rmin - pad)
        hi = max(1.01, rmax + pad)
        ax.set_ylim(lo, hi)
    
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    if title:
        ax.set_title(title)

    return fig, ax



def _ensure_datetime(df, date_col="date", time_col="time"):
    d = df[date_col].astype(str).str.strip()
    t = df[time_col].astype(str).str.strip().str.zfill(4)  # ✅ ensures HHMM
    return pd.to_datetime(d + t, format="%Y%m%d%H%M", errors="coerce")


def plot_cum_sector_ratio_timeseries(
    df_per_timestep,cmap_name="Reds",start=0.85,end=0.25,
    ax=None,
    title=None,
    ylabel="Cumulative sector mean / center",xlim=None
):
    """
    Uses rows where sector_type == 'CUM'.
    Expects columns: date, time, sector, mean, center_ppb.
    Produces one line per cumulative sector (C1, C2, ...).
    """
    df = df_per_timestep.copy()
    if "center_ppb" not in df.columns:
        raise ValueError("Missing 'center_ppb' in df_per_timestep. Add it in the period runner.")

    df = df[df["sector_type"] == "CUM"].copy()
    if df.empty:
        raise ValueError("No 'CUM' rows found in df_per_timestep.")

    df["datetime"] = _ensure_datetime(df)
    df = df[df["datetime"].notna()].copy()
    '''
    df["mean"] = pd.to_numeric(df["mean_w"], errors="coerce")
    df["center_ppb"] = pd.to_numeric(df["center_ppb"], errors="coerce")

    df = df[np.isfinite(df["mean"]) & np.isfinite(df["center_ppb"]) & (df["center_ppb"] != 0)].copy()
    df["ratio"] = df["mean"] / df["center_ppb"]

    wide = df.pivot_table(index="datetime", columns="sector", values="ratio", aggfunc="mean").sort_index()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for col in wide.columns:
        ax.plot(wide.index, wide[col].values, label=str(col))
    '''
    mean_col = "mean_w" if "mean_w" in df.columns else "mean"
    df["mean_val"] = pd.to_numeric(df[mean_col], errors="coerce")
    df["center_ppb"] = pd.to_numeric(df["center_ppb"], errors="coerce")

    df = df[np.isfinite(df["mean_val"]) & np.isfinite(df["center_ppb"]) & (df["center_ppb"] != 0)].copy()
    df["ratio"] = df["mean_val"] / df["center_ppb"]

    wide = df.pivot_table(index="datetime", columns="sector", values="ratio", aggfunc="mean").sort_index()

    def _cnum(s):
        try:
            return int(str(s).replace("C", ""))
        except Exception:
            return 10**9

    cols = sorted(list(wide.columns), key=_cnum)
    wide = wide.reindex(cols, axis=1)

    mapping = sector_color_mapping(labels=cols, cmap_name=cmap_name, start=start, end=end, base0_color="black")
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    for col in wide.columns:
        ax.plot(wide.index, wide[col].values, label=str(col), color=mapping[col], linewidth=2.0,marker="o", markersize=3)

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    plt.xticks(rotation=45)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Cumulative sector", ncol=2)
    if title:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    return fig, ax


def plot_cum_distance_ratio_timeseries(
    df_per_timestep,
    dist_bins_km,xlim=None,
    ax=None,
    title=None,cmap="Reds",start=0.85,end=0.25,
    ylabel="Cumulative distance mean / center",
):
    """
    Builds cumulative-distance mean ratios from NON-cumulative BIN rows.

    Expects BIN rows with:
      - sector_type == 'DISTCUM'
      - radius == upper edge of bin (e.g. 10,20,...)
      - mean == mean in that bin
      - n == number of cells in that bin
      - center_ppb == denominator for that timestamp

    Computes for each timestamp and each threshold D<=dmax:
      cum_mean(dmax) = sum_{bins<=dmax}(mean_bin * n_bin) / sum_{bins<=dmax}(n_bin)
      ratio(dmax) = cum_mean(dmax) / center_ppb
    Produces one line per D<=dmax.
    """
    df = df_per_timestep.copy()
    print(df)
    if "center_ppb" not in df.columns:
        raise ValueError("Missing 'center_ppb' in df_per_timestep. Add it in the period runner.")

    df = df[df["sector_type"] == "DISTCUM"].copy()
    if df.empty:
        raise ValueError("No 'DISTCUM' rows found in df_per_timestep.")

    df["datetime"] = _ensure_datetime(df)
    df = df[df["datetime"].notna()].copy()

    # numeric
    df["radius"] = pd.to_numeric(df["radius"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean_w"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["center_ppb"] = pd.to_numeric(df["center_ppb"], errors="coerce")

    df = df[
        np.isfinite(df["radius"]) &
        np.isfinite(df["mean"]) &
        np.isfinite(df["n"]) & (df["n"] > 0) &
        np.isfinite(df["center_ppb"]) & (df["center_ppb"] != 0)
    ].copy()

    # keep only bins that match your radii_km edges
    dist_bins_km = np.asarray(dist_bins_km, dtype=float)
    df = df[df["radius"].isin(dist_bins_km)].copy()

    out_rows = []
    for ts, g in df.groupby("datetime"):
        g = g.sort_values("radius")

        center_val = float(g["center_ppb"].iloc[0])

        for dmax, mean_val in zip(g["radius"].values, g["mean"].values):
            out_rows.append({
            "datetime": ts,
            "label": f"D≤{int(dmax)}km",
            "ratio": float(mean_val) / center_val,
        })


    df_line = pd.DataFrame(out_rows)
    if df_line.empty:
        raise ValueError("No cumulative distance ratios computed (check bins and data).")

    wide = df_line.pivot_table(index="datetime", columns="label", values="ratio", aggfunc="mean").sort_index()

    # order columns by increasing dmax
    def _dmax(s):
        try:
            return float(s.replace("D≤", "").replace("km", ""))
        except Exception:
            return np.inf

    wide = wide.reindex(sorted(wide.columns, key=_dmax), axis=1)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for col in wide.columns:
        ax.plot(wide.index, wide[col].values, label=str(col))

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    plt.xticks(rotation=45)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Cumulative distance", ncol=2)
    if title:
        ax.set_title(title)

    return fig, ax
def plot_cum_sector_cv_timeseries(
    df_per_timestep,
    ax=None,
    title=None,
    ylabel="CV (%)",
    weighted=True,xlim=None,
    cmap_name="Reds",
    start=0.85,
    end=0.25,marker="o", markersize=3
):
    """
    Time-series line plot:
      - one line per cumulative sector (C1, C2, ...)
      - x axis: time
      - y axis: CV (%) (cv_w if weighted else cv)
    Colors are the SAME Reds palette used by plot_rectangles/plot_ratio_bars.
    """


    df = df_per_timestep.copy()

    # keep only cumulative sectors
    df = df[df["sector_type"] == "CUM"].copy()
    if df.empty:
        raise ValueError("No 'CUM' rows found in df_per_timestep.")

    # datetime
    df["datetime"] = _ensure_datetime(df)
    df = df[df["datetime"].notna()].copy()

    # choose CV column
    cv_col = "cv_w" if weighted else "cv"
    if cv_col not in df.columns:
        raise ValueError(f"Missing '{cv_col}' in df_per_timestep.")

    df[cv_col] = pd.to_numeric(df[cv_col], errors="coerce")
    df = df[np.isfinite(df[cv_col])].copy()

    # pivot: time x sector
    wide = df.pivot_table(index="datetime", columns="sector", values=cv_col, aggfunc="mean").sort_index()

    # convert to percent
    wide = wide.astype(float) * 100.0

    # enforce sector order C1..Ck
    def _cnum(s):
        try:
            return int(str(s).replace("C", ""))
        except Exception:
            return 10**9

    cols = sorted(list(wide.columns), key=_cnum)
    wide = wide.reindex(cols, axis=1)

    # colors: same Reds scheme as rectangles/bars
    mapping = sector_color_mapping(
        labels=cols,
        cmap_name=cmap_name,
        start=start,
        end=end,
        base0_color="black",
    )

    # plot
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    print("NaT count:", df["datetime"].isna().sum(), "rows:", len(df))
    print(df[["date","time","datetime"]].head(10))
    for col in wide.columns:
        ax.plot(
            wide.index,
            wide[col].values,
            label=str(col),
            color=mapping[col],
            linewidth=2.0,
        )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    if xlim is not None:
        ax.set_xlim(xlim)
    #ax.set_xlim()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Cumulative sector", ncol=2)
    plt.xticks(rotation=45)

    if title:
        ax.set_title(title)

    return fig, ax