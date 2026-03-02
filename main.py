# main.py (compact: two big functions: single timestep + time interval)
#%%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime

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
)

from io_netcdf import df30min_to_netcdf_station_species

#%%
# -----------------------
# USER SETTINGS
# -----------------------
RUN_PERIOD = False

# Period settings (only used when RUN_PERIOD=True)
START_DT = datetime.datetime(2005, 5, 20, 2, 30)
END_DT   = datetime.datetime(2005, 5, 20, 3, 0)

# Mode works for BOTH single timestep and period
MODE = "A"          # "A" or "HEIGHT"
idx = 5
cell_nums = 20
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
        else:
            raise ValueError("mode must be 'A' or 'HEIGHT'")

        time_str = _time_str_from_species_ds(ds_species, species)
        center_value = float(grid_ppb[ii, jj])

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
            title=f"{species} CV — cumulative sectors at {time_str}"
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
        """
        fig_r1, ax_r1 = plot_ratio_bars(
            df_ratio_cum,
            xlabel="Sectors",
            title=f"{species} {time_str}: Cumulative sector mean / center"
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
            title=f"{species} {time_str}: Cumulative distance mean / center"
        )
    """
        plt.show()
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
def run_time_interval(mode="A", weighted=True):
    """
    Period analysis:
      - runs run_period_cumulative_sector_timeseries
      - creates ratio time-series plots
      - saves CSV + summary + NetCDF
    """
    stations = load_stations(stations_path)
    station = select_station(stations, idx)

    name = station["Station_Name"]
    lat_s = float(station["Latitude"])
    lon_s = float(station["Longitude"])

    # infer model cell lat/lon from first existing species file in period
    model_lat = model_lon = None
    found = False
    for d0, t0 in iter_timestamps(START_DT, END_DT, 30):
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
        step_minutes=30,
        weighted=weighted
    )

    # time-series ratio plots
    fig1, ax1 = plot_cum_sector_ratio_timeseries(
        df_30min,
        title=f"{species}: CUM sector mean / center ({name} {mode})"
    )
    fig1.savefig(f"{out_dir}/{name}_{species}_{mode}_ts_ratio_CUM.png", dpi=200)

    fig2, ax2 = plot_cum_distance_ratio_timeseries(
        df_30min,
        dist_bins_km=dist_bins_km,
        title=f"{species}: CUM distance mean / center ({name} {mode})"
    )
    fig2.savefig(f"{out_dir}/{name}_{species}_{mode}_ts_ratio_DISTCUM.png", dpi=200)

    plt.show()

    # save outputs
    out_csv = f"{out_dir}/{name}_{species}_{mode}_30min.csv"
    out_sum = f"{out_dir}/{name}_{species}_{mode}_summary.csv"
    out_nc = f"{out_dir}/{name}_{species}_{mode}.nc"

    df_30min.to_csv(out_csv, index=False)
    df_summary.to_csv(out_sum, index=False)

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
    print("Wrote:", out_nc)


def main():
    if RUN_PERIOD:
        run_time_interval(mode=MODE, weighted=True)
    else:
        run_single_timestep(mode=MODE, weighted=True)


if __name__ == "__main__":
    main()
# %%
