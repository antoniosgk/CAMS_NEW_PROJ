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
    extract_smallbox_ppb_optionHeight_fixed_z,
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

#%%
# -----------------------
# USER SETTINGS
# -----------------------
RUN_PERIOD = False

# Period settings (only used when RUN_PERIOD=True)
START_DT = datetime.datetime(2005, 5, 20, 0, 0)  #yyyy,m,d,hr,min
END_DT   = datetime.datetime(2005, 5, 21, 0, 00)
print("START_DT:", START_DT)
print("END_DT:", END_DT)
print("Generated timestamps:", list(iter_timestamps(START_DT, END_DT, 30)))
# Mode works for BOTH single timestep and period
MODE = "A"          # "A" or "HEIGHT"
idx = 1467
cell_nums = 10
dist_bins_km = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
out_dir = "/home/agkiokas/CAMS/plots/"

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
    d_zoom_topo = 20.0
    zoom_map = 45.0
    fig4_with_topo = False  # set True if you want station map on topo background

    stations = load_stations(stations_path)
    station = select_station(stations, idx)

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
            s=45,
            c="red",
            edgecolors="k",
            linewidths=0.6,
            transform=proj,
            zorder=7,
            label=f"Selected: {name}",
        )

        ax4.legend(loc="upper right")
        ax4.set_title("Stations context map", pad=18)
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
def run_time_interval(mode="A", weighted=True,start_dt=None,end_dt=None,step_minutes=30):
    """
    Period analysis:
      - runs run_period_cumulative_sector_timeseries
      - creates ratio time-series plots
      - saves CSV + summary + NetCDF
    """
    if start_dt is None:
        start_dt = START_DT
    if end_dt is None:
        end_dt = END_DT
    stations = load_stations(stations_path)
    station = select_station(stations, idx)

    name = station["Station_Name"]
    lat_s = float(station["Latitude"])
    lon_s = float(station["Longitude"])

    # infer model cell lat/lon from first existing species file in period
    model_lat = model_lon = None
    found = False
    for d0, t0 in iter_timestamps(start_dt, end_dt, step_minutes):
        sp0, _, _, _, _ = build_paths(base_path, product, species, d0, t0)
        try:
            ds0 = xr.open_dataset(sp0)
            lats = ds0["lat"].values
            lons = ds0["lon"].values
            ds0.close()
            i, j = nearest_grid_index(lat_s, lon_s, lats, lons)
            model_lat = float(lats[i]) if np.ndim(lats) == 1 else float(lats[i, j])
            model_lon = float(lons[j]) if np.ndim(lons) == 1 else float(lons[i, j])
            found = True
            break
        except FileNotFoundError:
            continue

    if not found:
        raise FileNotFoundError("No species files found in the requested period.")
    ts = list(iter_timestamps(START_DT, END_DT, 30))
    print("Generated timestamps:", ts)
    '''
    for (d0, t0) in ts:
        spf, Tf, PLf, RHf, orogf = build_paths(base_path, product, species, d0, t0)
        print(d0, t0, "->", str(spf), "exists:", os.path.exists(spf))
        print(f"\n{d0} {t0}")
        print("  sp  exists:", os.path.exists(spf),  spf)
        print("  T   exists:", os.path.exists(Tf),   Tf)
        print("  PL  exists:", os.path.exists(PLf),  PLf)
        print("  RH  exists:", os.path.exists(RHf),  RHf)
        print("  orog exists:", os.path.exists(orogf),orogf)
        '''
    df_30min, df_summary = run_period_cumulative_sector_timeseries(
        base_path=base_path,
        product=product,
        species=species,
        station=station,
        start_dt=START_DT,
        end_dt=END_DT,
        cell_nums=cell_nums,
        radii_km=dist_bins_km,
        mode=mode,
        step_minutes=step_minutes,
        weighted=weighted
    )

    # time-series ratio plots
    fig1, ax1 = plot_cum_sector_ratio_timeseries(
        df_30min,xlim=(start_dt,end_dt),
        title=f"{species}: CUM sector mean / center ({name} {mode})"
    )
    fig1.savefig(f"{out_dir}/{name}_{species}_{mode}_ts_ratio_CUM.png", dpi=200)
    '''
    fig2, ax2 = plot_cum_distance_ratio_timeseries(
        df_30min,
        dist_bins_km=dist_bins_km,xlim=(start_dt,end_dt),
        title=f"{species}: CUM distance mean / center ({name} {mode})"
    )
    fig2.savefig(f"{out_dir}/{name}_{species}_{mode}_ts_ratio_DISTCUM.png", dpi=200)
    '''
    # CV time-series per sector (colors match Reds palette used in rectangles/bars)
    fig_cv_ts, ax_cv_ts = plot_cum_sector_cv_timeseries(
    df_30min,
    weighted=weighted,
    title=f"{species}: CV over time by cumulative sector ({name} {mode})",
    cmap_name="Reds",
    start=0.85,
    end=0.25,xlim=(start_dt,end_dt)
)
    fig_cv_ts.savefig(f"{out_dir}/{name}_{species}_{mode}_ts_CV_CUM.png", dpi=200)
    plt.show()
    
    # save outputs
    out_csv = f"{out_dir}/{name}_{species}_{mode}_30min.csv"
    out_sum = f"{out_dir}/{name}_{species}_{mode}_summary.csv"
    out_nc = f"{out_dir}/{name}_{species}_{mode}.nc"

    df_30min.to_csv(out_csv, index=False)
    df_summary.to_csv(out_sum, index=False)
    print(df_30min)
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
    '''
    print("Saved:", out_csv)
    print("Saved:", out_sum)
    #print("Wrote:", out_nc)


def main():
    if RUN_PERIOD:
        start_time = time.time()
        run_time_interval(mode=MODE, weighted=True,start_dt=START_DT,end_dt=END_DT,step_minutes=30)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
    else:
        start_time = time.time()
        run_single_timestep(mode=MODE, weighted=True)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")


if __name__ == "__main__":
    main()
# %%
