import os
import numpy as np
import pandas as pd
import xarray as xr

from metpy.units import units as mp_units
from metpy.constants import g
from horizontal_indexing import nearest_grid_index
from calculation import iter_timestamps
from vertical_indexing import metpy_compute_heights
from file_utils import build_paths
from stations_utils import load_stations

# expects these to already exist in your project and be imported where needed:
# - iter_timestamps
# - build_paths
# - nearest_grid_index
# - metpy_compute_heights
# - metpy_find_level_index
# - load_stations


def build_station_horizontal_lookup(stations_df, sample_species_file):
    """
    Compute nearest model grid cell once for each valid station.
    """
    ds = xr.open_dataset(sample_species_file, decode_times=False)
    lats = ds["lat"].values
    lons = ds["lon"].values
    ds.close()

    rows = []
    valid_stations = stations_df[stations_df["is_valid"]].copy()

    for _, st in valid_stations.iterrows():
        i, j = nearest_grid_index(
            float(st["Latitude"]),
            float(st["Longitude"]),
            lats,
            lons
        )

        model_lat = float(lats[i]) if np.ndim(lats) == 1 else float(lats[i, j])
        model_lon = float(lons[j]) if np.ndim(lons) == 1 else float(lons[i, j])

        rows.append({
            "idx": int(st["idx"]),
            "Station_Name": st["Station_Name"],
            "Latitude": float(st["Latitude"]),
            "Longitude": float(st["Longitude"]),
            "Altitude": float(st["Altitude"]),
            "i": int(i),
            "j": int(j),
            "model_lat": model_lat,
            "model_lon": model_lon,
        })

    return pd.DataFrame(rows)


def find_first_existing_species_file(base_path, product, species, start_dt, end_dt, step_minutes):
    """
    Find first existing species file in the requested period.
    """
    for d0, t0 in iter_timestamps(start_dt, end_dt, step_minutes):
        d0 = str(d0).zfill(8)
        t0 = str(t0).zfill(4)
        spf, _, _, _, _ = build_paths(base_path, product, species, d0, t0)
        if os.path.exists(spf):
            return spf
    raise FileNotFoundError("No species files found in the requested period.")


def precompute_station_level_timeseries_vectorized(
    stations_lookup_df,
    base_path,
    product,
    species_for_grid,
    start_dt,
    end_dt,
    step_minutes=30,
):
    """
    Precompute model level for each station and timestamp.
    Returns a DataFrame with one row per station per timestep.
    """
    rows = []

    ii = stations_lookup_df["i"].astype(int).to_numpy()
    jj = stations_lookup_df["j"].astype(int).to_numpy()
    station_alts = stations_lookup_df["Altitude"].astype(float).to_numpy()

    for d0, t0 in iter_timestamps(start_dt, end_dt, step_minutes):
        d0 = str(d0).zfill(8)
        t0 = str(t0).zfill(4)

        _, Tf, PLf, RHf, orogf = build_paths(base_path, product, species_for_grid, d0, t0)

        if not (os.path.exists(Tf) and os.path.exists(PLf) and os.path.exists(RHf) and os.path.exists(orogf)):
            print(f"[skip] Missing met files for {d0} {t0}")
            continue

        dsT = dsPL = dsRH = dsO = None
        try:
            dsT = xr.open_dataset(Tf, decode_times=False)
            dsPL = xr.open_dataset(PLf, decode_times=False)
            dsRH = xr.open_dataset(RHf, decode_times=False)
            dsO = xr.open_dataset(orogf, decode_times=False)

            T4 = dsT["T"].isel(time=0).values      # (lev, lat, lon)
            PL4 = dsPL["PL"].isel(time=0).values
            RH4 = dsRH["RH"].isel(time=0).values
            PHIS2 = dsO["PHIS"].isel(time=0).values

            # (nlev, nstations)
            T_prof = T4[:, ii, jj]
            PL_prof = PL4[:, ii, jj]
            RH_prof = RH4[:, ii, jj]
            z0 = (PHIS2[ii, jj] * mp_units("m^2/s^2") / g).to("meter").magnitude

            z_prof = metpy_compute_heights(
                p_prof_Pa=PL_prof,
                T_prof_K=T_prof,
                RH=RH_prof,
                z0=z0
            )

            diff = np.abs(z_prof - station_alts[None, :])
            diff[~np.isfinite(diff)] = np.inf
            lev_idx = np.argmin(diff, axis=0)

            p_hPa = PL_prof[lev_idx, np.arange(len(ii))] / 100.0
            z_m = z_prof[lev_idx, np.arange(len(ii))]

            timestamp = pd.Timestamp(f"{d0[:4]}-{d0[4:6]}-{d0[6:8]} {t0[:2]}:{t0[2:]}:00")

            for n, (_, st) in enumerate(stations_lookup_df.iterrows()):
                rows.append({
                    "time": timestamp,
                    "idx": int(st["idx"]),
                    "Station_Name": st["Station_Name"],
                    "Latitude": float(st["Latitude"]),
                    "Longitude": float(st["Longitude"]),
                    "Altitude": float(st["Altitude"]),
                    "i": int(st["i"]),
                    "j": int(st["j"]),
                    "model_lat": float(st["model_lat"]),
                    "model_lon": float(st["model_lon"]),
                    "z_surf_model_m": float(z0[n]),
                    "level_idx": int(lev_idx[n]),
                    "level_pressure_hPa": float(p_hPa[n]),
                    "level_height_m": float(z_m[n]),
                    "abs_height_diff_m": float(abs(z_m[n] - float(st["Altitude"]))),
                })

        except Exception as e:
            print(f"[error] {d0} {t0}: {e}")

        finally:
            for ds in (dsT, dsPL, dsRH, dsO):
                if ds is not None:
                    try:
                        ds.close()
                    except Exception:
                        pass

    return pd.DataFrame(rows)


def build_and_save_station_level_lookup(
    stations_path,
    base_path,
    product,
    species_for_grid,
    start_dt,
    end_dt,
    step_minutes=30,
    out_horizontal_path=None,
    out_lookup_path=None,
):
    """
    Full pipeline:
      1) load stations
      2) compute horizontal lookup
      3) compute timestep-wise vertical levels
      4) save outputs
    """
    stations = load_stations(stations_path)

    sample_species_file = find_first_existing_species_file(
        base_path=base_path,
        product=product,
        species=species_for_grid,
        start_dt=start_dt,
        end_dt=end_dt,
        step_minutes=step_minutes,
    )

    hlookup = build_station_horizontal_lookup(stations, sample_species_file)

    lookup_df = precompute_station_level_timeseries_vectorized(
        stations_lookup_df=hlookup,
        base_path=base_path,
        product=product,
        species_for_grid=species_for_grid,
        start_dt=start_dt,
        end_dt=end_dt,
        step_minutes=step_minutes,
    )

    if out_horizontal_path is not None:
        if str(out_horizontal_path).lower().endswith(".parquet"):
            hlookup.to_parquet(out_horizontal_path, index=False)
        else:
            hlookup.to_csv(out_horizontal_path, index=False)

    if out_lookup_path is not None:
        if str(out_lookup_path).lower().endswith(".parquet"):
            lookup_df.to_parquet(out_lookup_path, index=False)
        else:
            lookup_df.to_csv(out_lookup_path, index=False)

    return hlookup, lookup_df