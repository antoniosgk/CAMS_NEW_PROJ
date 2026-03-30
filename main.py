# main.py (compact: two big functions: single timestep + time interval)
#%%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import time
import warnings
warnings.filterwarnings('ignore')

from file_utils import (
    base_path, product, species, stations_path,
    species_file, T_file, pl_file, RH_file, orog_file,
    build_paths, iter_timestamps
)
from stations_utils import load_stations, select_station
from horizontal_indexing import nearest_grid_index, add_distance_bins

from calculation import (
    compute_w_area_small,
    compute_sector_tables_generic,
    compute_cumulative_sector_tables,
    sector_stats_unweighted,
    sector_stats_weighted,
    build_distance_dataframe,
    stats_by_distance_bins,
    cumulative_mean_ratio_to_center,
    distance_cumulative_mean_ratio_to_center,
    run_period_cumulative_sector_timeseries,
    to_ppb_mmr,
)

from vertical_indexing import (
    extract_smallbox_ppb_optionA_fixed_k,
    extract_smallbox_ppb_optionHeight_fixed_z,extract_smallbox_ppb_optionA_given_k
)

from plots import (
    plot_cv_cumulative_sectors,
    plot_cv_bars_sector_both,
    plot_cv_vs_distance,
    plot_cv_bars_distance_both,
    plot_ratio_bars,
    plot_cum_sector_ratio_timeseries,
    plot_cum_distance_ratio_timeseries,
    plot_variable_on_map,plot_rectangles,
    plot_cum_sector_cv_timeseries
)

from io_netcdf import df30min_to_netcdf_station_species
from station_level_lookup import build_and_save_station_level_lookup

#%%
# -----------------------
# USER SETTINGS
# -----------------------
RUN_PERIOD = True
BUILD_K_LOOKUP = False

START_DT = datetime.datetime(2005, 6, 16, 0, 0)
END_DT   = datetime.datetime(2005, 6, 17, 0, 0)

MODE = "A"          # "A" or "HEIGHT"

# station selection mode:
#   "single" -> use STATION_IDX
#   "list"   -> use STATION_IDXS
#   "all"    -> all valid stations
STATION_SELECTION = "single"

STATION_IDX = 17
STATION_IDXS = [1, 5, 8]

cell_nums = 10
dist_bins_km = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

out_dir = "/home/agkiokas/CAMS/plots/"
lookup_dir = "/home/agkiokas/CAMS/lookups/"

LEVEL_LOOKUP_PATH = f"{lookup_dir}/station_level_timeseries.parquet"
HORIZONTAL_LOOKUP_PATH = f"{lookup_dir}/station_horizontal_lookup.parquet"

#%%
# -----------------------
# INTERNAL HELPERS
# -----------------------
def _attach_w_area_to_sector_dfs(dfs, w_area_small, w_col="w_area"):
    """
    Attach area weights to sector/cumulative-sector DataFrames.
    Assumes each df has columns: lat_idx, lon_idx (indices inside the small box).
    """
    out = []
    for df in dfs:
        df2 = df.copy()
        iy = pd.to_numeric(df2["lat_idx"], errors="coerce").to_numpy(dtype=int)
        ix = pd.to_numeric(df2["lon_idx"], errors="coerce").to_numpy(dtype=int)
        df2[w_col] = np.asarray(w_area_small)[iy, ix]
        out.append(df2)
    return out


def _time_str_from_species_ds(ds_species, species_name):
    time_val = ds_species[species_name]["time"].values[0]
    return pd.to_datetime(time_val).strftime("%Y-%m-%d %H:%M")


def _single_timestep_small_box_indices(i, j, Ny, Nx, cell_nums):
    i1_s, i2_s = max(0, i - cell_nums), min(Ny - 1, i + cell_nums)
    j1_s, j2_s = max(0, j - cell_nums), min(Nx - 1, j + cell_nums)
    ii = i - i1_s
    jj = j - j1_s
    return i1_s, i2_s, j1_s, j2_s, ii, jj
def _get_time_str(ds_species):
    tval = ds_species[species]["time"].values[0]
    return pd.to_datetime(tval).strftime("%Y-%m-%d %H:%M")
def get_target_stations(stations_df, selection="single", idx=None, idxs=None):
    """
    Return a DataFrame of target stations.

    selection:
      - "single": one station from idx
      - "list": multiple stations from idxs
      - "all": all valid stations
    """
    valid = stations_df[stations_df["is_valid"]].copy()

    if selection == "single":
        if idx is None:
            raise ValueError("For selection='single', idx must be provided.")
        row = select_station(stations_df, idx=idx)
        return pd.DataFrame([row])

    if selection == "list":
        if idxs is None or len(idxs) == 0:
            raise ValueError("For selection='list', idxs must be a non-empty list.")
        rows = []
        for one_idx in idxs:
            rows.append(select_station(stations_df, idx=one_idx))
        return pd.DataFrame(rows)

    if selection == "all":
        return valid.reset_index(drop=True)

    raise ValueError("selection must be one of: 'single', 'list', 'all'")
#start_time = time.time()
# -----------------------
# 1) SINGLE TIMESTEP
# -----------------------
def run_single_timestep(mode="A", weighted=True):
    """
    Single timestep analysis using the SAME vertical strategies as the period runner.

    mode="A":      same k* everywhere (k* from central column)
    mode="HEIGHT": same target height everywhere, k varies per cell

    Uses the fixed paths imported from file_utils:
      species_file, T_file, pl_file, RH_file, orog_file
    """
    # ---- local plotting settings (same as your old main) ----
    d_zoom_species = 0.8
    d_zoom_topo =45 #fig 3 zoom and terrain
    zoom_map = 10 #for fig4
    fig4_with_topo = True # set True if you want station map on topo background

    stations = load_stations(stations_path)
    station = select_station(stations, STATION_IDX)

    name = station["Station_Name"]
    lat_s = float(station["Latitude"])
    lon_s = float(station["Longitude"])
    alt_s = float(station["Altitude"])

    print(f"\nSelected station: {name} (lat={lat_s}, lon={lon_s}, alt={alt_s} m)")
    print(f"Mode: {mode} | Weighted: {weighted}")

    ds_species = xr.open_dataset(species_file)
    ds_T = xr.open_dataset(T_file)
    ds_PL = xr.open_dataset(pl_file)
    ds_RH = xr.open_dataset(RH_file)
    ds_orog = xr.open_dataset(orog_file)

    try:
        lats = ds_species["lat"].values
        lons = ds_species["lon"].values
        i, j = nearest_grid_index(lat_s, lon_s, lats, lons)

        Ny, Nx = (lats.shape[0], lons.shape[0]) if np.ndim(lats) == 1 else lats.shape
        i1_s, i2_s, j1_s, j2_s, ii, jj = _single_timestep_small_box_indices(i, j, Ny, Nx, cell_nums)
        
        lats_small = lats[i1_s:i2_s + 1]
        lons_small = lons[j1_s:j2_s + 1]
        radii = list(range(1, cell_nums + 1))

        # weights aligned to small box
        w_area_small = compute_w_area_small(lats_small, lons_small) if weighted else None
        # ---- time string ----
        time_str = _get_time_str(ds_species)

        # ---- Topography background domain (big box for topo plot) ----
        dlat_deg = float(np.abs(lats[1] - lats[0]))
        dlon_deg = float(np.abs(lons[1] - lons[0]))

        cell_nums_lat = int(np.ceil(d_zoom_topo / dlat_deg))
        cell_nums_lon = int(np.ceil(d_zoom_topo / dlon_deg))
        cell_nums_bg = max(cell_nums_lat, cell_nums_lon)

        i1_bg = max(0, i - cell_nums_bg)
        i2_bg = min(Ny - 1, i + cell_nums_bg)
        j1_bg = max(0, j - cell_nums_bg)
        j2_bg = min(Nx - 1, j + cell_nums_bg)

        # NOTE: if ds_orog has no time dim, drop isel(time=0)
        PHIS_bg = ds_orog["PHIS"].isel(time=0, lat=slice(i1_bg, i2_bg + 1), lon=slice(j1_bg, j2_bg + 1)).values
        z_orog_bg = PHIS_bg / 9.80665

        lats_bg = lats[i1_bg:i2_bg + 1]
        lons_bg = lons[j1_bg:j2_bg + 1]
        grid_ppb_ref, meta_ref = extract_smallbox_ppb_optionA_fixed_k(
         ds_species, ds_T, ds_PL, ds_RH, ds_orog,species, alt_s,
         i, j, i1_s, i2_s, j1_s, j2_s,to_ppb_fn=to_ppb_mmr)

       # Define our global plot limits
        vmin_plot = meta_ref['min']
        vmax_plot = meta_ref['max']
        # build ppb field (small box) using chosen vertical mode
        if str(mode).upper() == "A":
            grid_ppb, meta_v = extract_smallbox_ppb_optionA_fixed_k(
                ds_species, ds_T, ds_PL, ds_RH, ds_orog,
                species, alt_s,
                i, j, i1_s, i2_s, j1_s, j2_s,
                to_ppb_fn=to_ppb_mmr
            )
        elif str(mode).upper() == "HEIGHT":
            grid_ppb, meta_v = extract_smallbox_ppb_optionHeight_fixed_z(
                ds_species, ds_T, ds_PL, ds_RH, ds_orog,
                species, alt_s,
                i, j, i1_s, i2_s, j1_s, j2_s,
                to_ppb_fn=to_ppb_mmr
            )
            
            print(np.unique(meta_v["k_grid"]).size, meta_v["k_grid"].min(), meta_v["k_grid"].max())
        else:
            raise ValueError("mode must be 'A' or 'HEIGHT'")
            
        time_str = _time_str_from_species_ds(ds_species, species)
        center_value = float(grid_ppb[ii, jj])
        meta = {
            "station_name": name,
            "station_lat": lat_s,
            "station_lon": lon_s,
            "station_alt": alt_s,
            "model_lat": float(lats[i]) if np.ndim(lats) == 1 else float(lats[i, j]),
            "model_lon": float(lons[j]) if np.ndim(lons) == 1 else float(lons[i, j]),
            "time_str": time_str,
            "species": species,
            "units": "ppb",
            "mode": mode,
            **meta_v,
        }
        # sectors + cumulative sectors
        sector_dfs, sector_masks = compute_sector_tables_generic(
            ii, jj, lats_small, lons_small, grid_ppb, species, radii=radii
        )
        cum_dfs, _ = compute_cumulative_sector_tables(
            sector_masks, lats_small, lons_small, grid_ppb, species
        )

        # attach area weights to dfs (proper area weights, not only cos(lat))
        if weighted:
            sector_dfs = _attach_w_area_to_sector_dfs(sector_dfs, w_area_small, w_col="w_area")
            cum_dfs = _attach_w_area_to_sector_dfs(cum_dfs, w_area_small, w_col="w_area")

        # cumulative sector stats for plotting
        cum_stats_unw = [sector_stats_unweighted(df, species) for df in cum_dfs]
        cum_stats_w = [sector_stats_weighted(df, species, w_col="w_area") for df in cum_dfs] if weighted else None

        # distance dataframe + distance stats
        df_dist = build_distance_dataframe(
            lats_small, lons_small, grid_ppb, lat_s, lon_s, species,
            w_area=w_area_small if weighted else None
        )
        df_cv_unw = stats_by_distance_bins(df_dist, species, dist_bins_km)
        df_cv_w = stats_by_distance_bins(df_dist, species, dist_bins_km, w_col="w_area") if weighted else None

        # plots: CV cumulative sectors (line + bars)
        fig_cv_cum, ax_cv_cum = plot_cv_cumulative_sectors(
            cum_stats_unw,
            cum_stats_w if weighted else cum_stats_unw,
            title=f"{species} CV — cumulative sectors at {time_str} UTC"
        )
        fig_cv_cum_b, ax_cv_cum_b = plot_cv_bars_sector_both(
            cum_stats_unw,
            cum_stats_w if weighted else cum_stats_unw,
            title=f"{species} CV: Sectors at {time_str} UTC for same model level" if mode=='A' else f"{species} CV: Sectors at {time_str} UTC for same altitude ASL"
            )
        
        # plots: CV vs distance (line + bars)
        fig_cv, ax_cv = plot_cv_vs_distance(
            df_cv_unw,
            df_cv_w if weighted else df_cv_unw,
            title=f"{species} CV vs distance ({name}) at {time_str}"
        )
        fig_cv_b, ax_cv_b = plot_cv_bars_distance_both(
            df_cv_unw,
            df_cv_w if weighted else df_cv_unw,
            title=f"{species} CV vs distance ({name}) at {time_str}",
            reverse=False
        )

        # ratios to center (bar plots)
        df_dist_binned = add_distance_bins(df_dist, dist_bins_km)

        df_ratio_cum = cumulative_mean_ratio_to_center(
            cum_dfs=cum_dfs,
            var_name=species,
            center_value=center_value,
            labels=[f"{k}" for k in range(1, len(cum_dfs) + 1)],
            w_col="w_area" if weighted else None
        )
        
        fig_r1, ax_r1 = plot_ratio_bars(
            df_ratio_cum,
            xlabel="Sectors",
            title=f"{species} {time_str} UTC: sector mean / center: Same Model Level" if mode=='A'  else f"{species} {time_str} UTC: sector mean / center: Same altitude ASL"
             )

        df_ratio_dist = distance_cumulative_mean_ratio_to_center(
            df_dist=df_dist_binned,
            var_name=species,
            center_value=center_value,
            d_bins_km=dist_bins_km,
            w_col="w_area" if weighted else None
        )
        fig_r2, ax_r2 = plot_ratio_bars(
            df_ratio_dist,
            xlabel="Distance < km",
            title=f"{species} {time_str} UTC: Cumulative distance mean / center"
        )

        vmin = np.nanmin(vmin_plot)
        vmax = np.nanmax(vmax_plot)
        
        fig1, ax1, im1 = plot_variable_on_map(
        lats_small, lons_small, grid_ppb,
         lon_s, lat_s,
        units="ppb",
        species_name=species,vmin=vmin,vmax=vmax,
        d=d_zoom_species,
        time_str=time_str,
        meta=meta, z_orog_m=z_orog_bg,          
        lats_terrain=lats_bg, lons_terrain=lons_bg,
        add_orog_contours=True, plot_orography=False
                      )

    

        fig2, ax2, im2 = plot_variable_on_map(
        lats_small, lons_small, grid_ppb,
        lon_s, lat_s,
        units='ppb',
        species_name=species,
        d=d_zoom_species,
        time_str=time_str,
         meta=meta,
        z_orog_m=z_orog_bg,vmin=vmin,vmax=vmax,
        lats_terrain=lats_bg,
        lons_terrain=lons_bg,plot_orography=False)

        plot_rectangles(ax2, lats_small, lons_small, ii, jj, im2, meta=meta,radii=radii)
        plt.show()
         # FIG 3 — Topography-only map
        # ============================================================
        fig3, ax3, _ = plot_variable_on_map(
            lats_small,
            lons_small,
            data_arr=None,
            lon_s=lon_s,
            lat_s=lat_s,
            units="",
            species_name="",
            d=d_zoom_topo,
            meta=meta,
            lats_terrain=lats_bg,
            lons_terrain=lons_bg,
            plot_orography=True,
            z_orog_m=z_orog_bg,
            terrain_alpha=0.6,
            add_orog_contours=True,
            plot_species=False,
        )
        plt.show()

        # ============================================================
        # FIG 4 — Stations context map (optionally with topography)
        # ============================================================
        proj = ccrs.PlateCarree()

        if fig4_with_topo:
            fig4, ax4, _ = plot_variable_on_map(
                lats_small,
                lons_small,
                data_arr=None,
                lon_s=lon_s,
                lat_s=lat_s,
                units="",
                species_name="",
                d=d_zoom_topo,
                meta=meta,
                lats_terrain=lats_bg,
                lons_terrain=lons_bg,
                plot_orography=True,
                z_orog_m=z_orog_bg,
                add_orog_contours=True,
                plot_species=False,
            )
        else:
            fig4, ax4 = plt.subplots(subplot_kw={"projection": proj})
            lon_min, lon_max = lon_s - zoom_map, lon_s + zoom_map
            lat_min, lat_max = lat_s - zoom_map, lat_s + zoom_map
            ax4.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

            ax4.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
            ax4.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
            ax4.coastlines(resolution="10m", linewidth=0.8)
            ax4.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=1)

            gl = ax4.gridlines(
                crs=proj,
                draw_labels=True,
                linewidth=0.6,
                alpha=0.95,
                linestyle="--",
                zorder=1
            )

        st = stations.copy()
        st["Latitude"] = pd.to_numeric(st["Latitude"], errors="coerce")
        st["Longitude"] = pd.to_numeric(st["Longitude"], errors="coerce")
        st = st.dropna(subset=["Latitude", "Longitude"])

        lon_min, lon_max = lon_s - zoom_map, lon_s + zoom_map
        lat_min, lat_max = lat_s - zoom_map, lat_s + zoom_map
        ax4.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

        st = st[st["Longitude"].between(lon_min, lon_max) & st["Latitude"].between(lat_min, lat_max)]
        ax4.scatter(
            st["Longitude"].values,
            st["Latitude"].values,
            s=6,
            c="k",
            alpha=0.7,
            transform=proj,
            zorder=6,
            label="Stations",
        )

        ax4.scatter(
            [lon_s],
            [lat_s],
            s=60,
            c="blue",
            edgecolors="k",
            linewidths=0.6,
            transform=proj,
            zorder=7,
            label=f"Selected: {name}",
        )
        ax4.legend(loc="upper right")
        ax4.set_title("Stations map", pad=18)
        plt.show()
        #HERE PUT 2 PROFILE PLOTS
    
        print("Vertical meta:", meta_v)

    finally:
        for ds in (ds_species, ds_T, ds_PL, ds_RH, ds_orog):
            try:
                ds.close()
            except Exception:
                pass


# -----------------------
# 2) TIME INTERVAL
# -----------------------
def run_time_interval(
    mode="A",
    weighted=True,
    start_dt=None,
    end_dt=None,
    step_minutes=30,
    level_lookup_df=None,
    level_lookup_path=None,
    station_selection="single",
    station_idx=None,
    station_idxs=None,
    make_plots=True,
):
    """
    Period analysis for one, many, or all stations.

    It:
      - loads the precomputed station/timestep level lookup
      - selects target stations
      - runs run_period_cumulative_sector_timeseries for each station
      - saves one CSV + one summary CSV per station
      - optionally saves plots per station

    Returns
    -------
    results : dict
        Dictionary keyed by station idx:
        results[idx] = {
            "station_name": ...,
            "df_30min": ...,
            "df_summary": ...
        }
    """
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if start_dt is None:
        start_dt = START_DT
    if end_dt is None:
        end_dt = END_DT

    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------
    # load stations and select targets
    # ----------------------------
    stations = load_stations(stations_path)

    target_stations = get_target_stations(
        stations_df=stations,
        selection=station_selection,
        idx=station_idx,
        idxs=station_idxs,
    )

    if target_stations.empty:
        raise ValueError("No target stations selected.")

    # ----------------------------
    # load lookup
    # ----------------------------
    if level_lookup_df is None:
        if level_lookup_path is None:
            raise ValueError("Provide either level_lookup_df or level_lookup_path.")

        level_lookup_path = Path(level_lookup_path)
        if not level_lookup_path.exists():
            raise FileNotFoundError(f"Lookup file not found: {level_lookup_path}")

        if level_lookup_path.suffix.lower() == ".parquet":
            level_lookup_df = pd.read_parquet(level_lookup_path)
        elif level_lookup_path.suffix.lower() == ".csv":
            level_lookup_df = pd.read_csv(level_lookup_path)
        else:
            raise ValueError(f"Unsupported lookup format: {level_lookup_path.suffix}")

    level_lookup_df = level_lookup_df.copy()
    if "time" not in level_lookup_df.columns:
        raise ValueError("level_lookup_df must contain a 'time' column.")
    level_lookup_df["time"] = pd.to_datetime(level_lookup_df["time"])

    required_cols = {"idx", "time", "level_idx"}
    missing = required_cols - set(level_lookup_df.columns)
    if missing:
        raise ValueError(f"level_lookup_df is missing required columns: {sorted(missing)}")

    # ----------------------------
    # run each station
    # ----------------------------
    results = {}

    for _, station in target_stations.iterrows():
        name = station["Station_Name"]
        st_idx = int(station["idx"])

        print(f"\n=== Running station: {name} (idx={st_idx}) ===")

        station_lookup = level_lookup_df[level_lookup_df["idx"] == st_idx].copy()
        station_lookup = station_lookup[
            (station_lookup["time"] >= pd.Timestamp(start_dt)) &
            (station_lookup["time"] <= pd.Timestamp(end_dt))
        ].copy()

        if station_lookup.empty:
            print(f"[skip] No lookup rows found for station {name} (idx={st_idx}) in requested period.")
            continue

        df_30min, df_summary = run_period_cumulative_sector_timeseries(
            base_path=base_path,
            product=product,
            species=species,
            station=station,
            start_dt=start_dt,
            end_dt=end_dt,
            cell_nums=cell_nums,
            radii_km=dist_bins_km,
            mode=mode,
            step_minutes=step_minutes,
            weighted=weighted,
            level_lookup_df=station_lookup,
        )

        if df_30min.empty:
            print(f"[skip] No output rows produced for station {name} (idx={st_idx}).")
            continue

        # save CSVs automatically
        out_csv = f"{out_dir}/{name}_{species}_{mode}_30min.csv"
        out_sum = f"{out_dir}/{name}_{species}_{mode}_summary.csv"

        df_30min.to_csv(out_csv, index=False)
        df_summary.to_csv(out_sum, index=False)

        print("Saved:", out_csv)
        print("Saved:", out_sum)

        # plots
        if make_plots:
            fig1, ax1 = plot_cum_sector_ratio_timeseries(
                df_30min,
                xlim=(start_dt, end_dt),
                title=f"{species}: CUM sector mean / center ({name} {mode})"
            )
            fig1.savefig(f"{out_dir}/{name}_{species}_{mode}_ts_ratio_CUM.png", dpi=200)
            plt.close(fig1)

            fig_cv_ts, ax_cv_ts = plot_cum_sector_cv_timeseries(
                df_30min,
                weighted=weighted,
                title=f"{species}: CV over time by cumulative sector ({name} {mode})",
                cmap_name="Reds",
                start=0.25,
                end=0.85,
                xlim=(start_dt, end_dt)
            )
            fig_cv_ts.savefig(f"{out_dir}/{name}_{species}_{mode}_ts_CV_CUM.png", dpi=200)
            plt.close(fig_cv_ts)

        results[st_idx] = {
            "station_name": name,
            "df_30min": df_30min,
            "df_summary": df_summary,
        }

    if not results:
        print("No station produced output.")
    else:
        print(f"\nFinished. Stations with output: {len(results)}")

    return results
    '''
    ds_out = df30min_to_netcdf_station_species(
        df_30min=df_30min,
        station_dict=station,
        model_lat=model_lat,
        model_lon=model_lon,
        species=species,
        out_nc_path=out_nc,
        mode=mode,
        weighted=weighted
    )
    
    print("Saved:", out_csv)
    print("Saved:", out_sum)
    #print("Wrote:", out_nc)
'''

def main():
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(lookup_dir, exist_ok=True)
    if BUILD_K_LOOKUP:
        start_time=time.time()
        print(datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'))
        print(start_time)
        hlookup, lookup_df = build_and_save_station_level_lookup(
            stations_path=stations_path,
            base_path=base_path,
            product=product,
            species_for_grid=species,
            start_dt=START_DT,
            end_dt=END_DT,
            step_minutes=30,
            out_horizontal_path=HORIZONTAL_LOOKUP_PATH,
            out_lookup_path=LEVEL_LOOKUP_PATH,
        )
        print("Saved horizontal lookup:", HORIZONTAL_LOOKUP_PATH)
        print("Saved level lookup:", LEVEL_LOOKUP_PATH)
        end_time = time.time()
        print(datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'))
        execution_time = (end_time - start_time)/60
        print(f"Execution time: {execution_time:.2f} minutes")
    if RUN_PERIOD:
        start_time = time.time()
        print(datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'))
        print(start_time)
        results=run_time_interval(mode=MODE,
            weighted=True,
            start_dt=START_DT,
            end_dt=END_DT,
            step_minutes=30,
            level_lookup_path=LEVEL_LOOKUP_PATH,
            station_selection=STATION_SELECTION,
            station_idx=STATION_IDX,
            station_idxs=STATION_IDXS,
            make_plots=True,)
        end_time = time.time()
        print(datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'))
        execution_time = (end_time - start_time)/60
        print(f"Execution time: {execution_time:.2f} minutes")
    '''    
    else:
        start_time = time.time()
        run_single_timestep(mode=MODE, weighted=True)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
'''

if __name__ == "__main__":
    main()
# %%
