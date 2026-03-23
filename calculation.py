# calculation.py
import numpy as np
import pandas as pd
import xarray as xr

from file_utils import build_paths, iter_timestamps, OrogCache
from horizontal_indexing import nearest_grid_index, make_small_box_indices, add_distance_bins
from vertical_indexing import (
    extract_smallbox_ppb_optionA_fixed_k,
    extract_smallbox_ppb_optionHeight_fixed_z,
)


# Keep your project constant
EARTH_RADIUS_KM = 6371.0
from pathlib import Path
def all_inputs_exist(spf, Tf, PLf, RHf, orogf):
    """Return True only if all required files exist."""
    return (Path(spf).exists()
            and Path(Tf).exists()
            and Path(PLf).exists()
            and Path(RHf).exists()
            and Path(orogf).exists())

# -----------------------------
# Weights
# -----------------------------
def compute_w_area_small(lats_small, lons_small, earth_radius_km=EARTH_RADIUS_KM):
    """
    Area-like weights for a regular lat/lon grid (small box).
    Returns w_area_small with shape (Ny_s, Nx_s).

    Formula:
      w ~ R^2 * dlat_rad * dlon_rad * cos(lat)

    Notes:
      - Uses mean spacing from lats_small/lons_small.
      - Requires at least 2 lat and 2 lon points.
    """
    lats_small = np.asarray(lats_small)
    lons_small = np.asarray(lons_small)

    if lats_small.size < 2 or lons_small.size < 2:
        raise ValueError("Need at least 2 lat and 2 lon points to compute grid spacing.")

    dlat_deg = float(np.mean(np.abs(np.diff(lats_small))))
    dlon_deg = float(np.mean(np.abs(np.diff(lons_small))))

    dlat_rad = np.deg2rad(dlat_deg)
    dlon_rad = np.deg2rad(dlon_deg)

    Rm = float(earth_radius_km) * 1000.0

    coslat = np.cos(np.deg2rad(lats_small))  # (Ny_s,)
    w_area_small = (Rm**2) * dlat_rad * dlon_rad * coslat[:, None] * np.ones((1, len(lons_small)))

    return np.clip(w_area_small, 0.0, None)

import datetime as dt
import numpy as np

def season_from_datetime(ts: dt.datetime):
    """Meteorological seasons."""
    m = ts.month
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Autumn"


def day_night_label(lat, lon, ts: dt.datetime, tz="UTC"):
    """
    Returns 'day' or 'night'.
    Tries Astral if installed; otherwise uses a simple solar-elevation approximation.
    Assumes ts is in UTC unless you pass a tz-aware datetime.
    """
    # --- Try Astral (best if available) ---
    try:
        from astral import LocationInfo
        from astral.sun import sun
        import pytz

        tzinfo = pytz.timezone(tz)
        if ts.tzinfo is None:
            ts_loc = tzinfo.localize(ts)
        else:
            ts_loc = ts.astimezone(tzinfo)

        loc = LocationInfo(name="x", region="x", timezone=tz, latitude=float(lat), longitude=float(lon))
        s = sun(loc.observer, date=ts_loc.date(), tzinfo=tzinfo)
        return "day" if (s["sunrise"] <= ts_loc <= s["sunset"]) else "night"

    except Exception:
        # --- Fallback: solar elevation approximation (UTC) ---
        # Accuracy is approximate; good enough for day/night split.
        lat = np.deg2rad(float(lat))
        lon = float(lon)

        # fractional year
        day_of_year = ts.timetuple().tm_yday
        hour = ts.hour + ts.minute / 60.0
        gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1 + (hour - 12) / 24.0)

        # declination (rad) and equation of time (minutes)
        decl = (
            0.006918
            - 0.399912 * np.cos(gamma)
            + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma)
            + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma)
            + 0.001480 * np.sin(3 * gamma)
        )
        eqtime = 229.18 * (
            0.000075
            + 0.001868 * np.cos(gamma)
            - 0.032077 * np.sin(gamma)
            - 0.014615 * np.cos(2 * gamma)
            - 0.040849 * np.sin(2 * gamma)
        )

        # true solar time (minutes)
        tst = (hour * 60.0 + eqtime + 4.0 * lon) % 1440.0
        ha = np.deg2rad((tst / 4.0) - 180.0)  # hour angle

        cos_zen = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
        cos_zen = np.clip(cos_zen, -1.0, 1.0)
        zen = np.arccos(cos_zen)
        elev = np.rad2deg(np.pi / 2.0 - zen)

        return "day" if elev > 0.0 else "night"
# -----------------------------
# Units conversion
# -----------------------------
def to_ppb_mmr(data_arr, species):
    """
    Convert mass mixing ratio (kg/kg) to ppb using MW_air/MW_species * 1e9.
    Extend MW_map if you add species.
    """
    MW_air = 28.9647
    MW_map = {"O3": 48.0}
    if species not in MW_map:
        raise ValueError(f"No MW defined for species={species}. Add it to MW_map.")
    return np.asarray(data_arr, dtype=float) * (MW_air / MW_map[species]) * 1e9


# -----------------------------
# Sector masks + tables
# -----------------------------
def safe_slice(low, high, maxN):
    return slice(max(low, 0), min(high, maxN))


def compute_ring_sector_masks(ii, jj, Ny, Nx, radii):
    """
    Create disjoint ring masks around (ii,jj):
      S1 = box(r1)
      S2 = box(r2) - box(r1)
      ...
    """
    masks = []
    prev = np.zeros((Ny, Nx), dtype=bool)

    for r in radii:
        box = np.zeros((Ny, Nx), dtype=bool)
        box[
            safe_slice(ii - r, ii + r + 1, Ny),
            safe_slice(jj - r, jj + r + 1, Nx)
        ] = True

        ring = box & (~prev)
        masks.append(ring)
        prev = box

    return masks


def sector_table(mask, lats_small, lons_small, data_arr, var_name, invalid_mask=None, w_area=None):
    """
    Build a DataFrame for a given sector mask.

    New:
      - invalid_mask: 2D bool same shape as data_arr; True means invalid (below-ground)
      - w_area: 2D weights same shape as data_arr (optional)
      - adds columns:
          * is_valid (True/False)
          * w_area (if provided)
    """
    iy, ix = np.where(mask)

    # 1D vs 2D lat/lon
    if lats_small.ndim == 1 and lons_small.ndim == 1:
        lat_vals = lats_small[iy]
        lon_vals = lons_small[ix]
    else:
        lat_vals = lats_small[iy, ix]
        lon_vals = lons_small[iy, ix]

    vals = np.asarray(data_arr)[iy, ix]

    df = pd.DataFrame({
        "lat_idx": iy,
        "lon_idx": ix,
        "lat": lat_vals,
        "lon": lon_vals,
        var_name: vals,
    })

    if invalid_mask is not None:
        inv = np.asarray(invalid_mask, dtype=bool)[iy, ix]
        df["is_valid"] = ~inv
    else:
        df["is_valid"] = np.isfinite(pd.to_numeric(df[var_name], errors="coerce"))

    if w_area is not None:
        df["w_area"] = np.asarray(w_area)[iy, ix]

    return df


def compute_sector_tables_generic(ii, jj, lats_small, lons_small, data_arr, var_name, radii,
                                  invalid_mask=None, w_area=None):
    Ny, Nx = data_arr.shape
    masks = compute_ring_sector_masks(ii, jj, Ny, Nx, radii)

    dfs = [
        sector_table(m, lats_small, lons_small, data_arr, var_name,
                    invalid_mask=invalid_mask, w_area=w_area)
        for m in masks
    ]
    return dfs, masks


def cumulative_sector_masks(sector_masks):
    running = np.zeros_like(sector_masks[0], dtype=bool)
    out = []
    for S in sector_masks:
        running = running | S
        out.append(running.copy())
    return out


def compute_cumulative_sector_tables(sector_masks, lats_small, lons_small, data_arr, var_name,
                                     invalid_mask=None, w_area=None):
    cum_masks = cumulative_sector_masks(sector_masks)
    dfs = [
        sector_table(m, lats_small, lons_small, data_arr, var_name,
                    invalid_mask=invalid_mask, w_area=w_area)
        for m in cum_masks
    ]
    return dfs, cum_masks

def apply_mask_to_data_arr(data_arr, invalid_mask, fill_value=np.nan):
    """
    Returns a copy of data_arr where invalid_mask==True is replaced by fill_value.
    data_arr: 2D array
    invalid_mask: 2D bool array same shape
    """
    out = np.array(data_arr, dtype=float, copy=True)
    out[np.asarray(invalid_mask, dtype=bool)] = fill_value
    return out
# -----------------------------
# Weighted/unweighted stats
# -----------------------------
def weighted_quantile(x, w, q):
    """
    Weighted quantile for 1D arrays. q in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[m]
    w = w[m]
    if x.size == 0:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    w = w[order]

    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return float(np.interp(q, cw, x))


def sector_stats_unweighted(df, var_col):
    """
    Return: n, mean, std, cv, median, q25, q75, iqr
    """
    x = pd.to_numeric(df[var_col], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "cv": np.nan,
                "median": np.nan, "q25": np.nan, "q75": np.nan, "iqr": np.nan}

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    median = float(np.median(x))
    q25 = float(np.quantile(x, 0.25))
    q75 = float(np.quantile(x, 0.75))
    iqr = float(q75 - q25)
    cv = float(std / mean) if np.isfinite(mean) and mean != 0 else np.nan

    return {"n": int(x.size), "mean": mean, "std": std, "cv": cv,
            "median": median, "q25": q25, "q75": q75, "iqr": iqr}


def sector_stats_weighted(df, var_name, w_col="w_area"):
    """
    Weighted stats. Requires df[w_col] to exist (added by sector_table/build_distance_dataframe).
    Returns keys with _w suffix.
    """
    vals = pd.to_numeric(df[var_name], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df[w_col], errors="coerce").to_numpy(dtype=float)

    m = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    vals = vals[m]
    w = w[m]

    if vals.size == 0:
        return {"n": 0,
                "mean_w": np.nan, "std_w": np.nan, "cv_w": np.nan,
                "median_w": np.nan, "q1_w": np.nan, "q3_w": np.nan, "iqr_w": np.nan}

    wsum = float(np.sum(w))
    mean_w = float(np.sum(w * vals) / wsum)

    var_w = float(np.sum(w * (vals - mean_w) ** 2) / wsum)
    std_w = float(np.sqrt(var_w))
    cv_w = float(std_w / mean_w) if mean_w != 0 else np.nan

    q1_w = float(weighted_quantile(vals, w, 0.25))
    med_w = float(weighted_quantile(vals, w, 0.50))
    q3_w = float(weighted_quantile(vals, w, 0.75))
    iqr_w = float(q3_w - q1_w)

    return {"n": int(vals.size),
            "mean_w": mean_w, "std_w": std_w, "cv_w": cv_w,
            "median_w": med_w, "q1_w": q1_w, "q3_w": q3_w, "iqr_w": iqr_w}

def stats_unweighted_arr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "cv": np.nan,
                "median": np.nan, "q25": np.nan, "q75": np.nan, "iqr": np.nan}

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    cv = float(std / mean) if np.isfinite(mean) and mean != 0 else np.nan
    q25 = float(np.quantile(x, 0.25))
    med = float(np.quantile(x, 0.50))
    q75 = float(np.quantile(x, 0.75))
    return {"n": int(x.size), "mean": mean, "std": std, "cv": cv,
            "median": med, "q25": q25, "q75": q75, "iqr": float(q75 - q25)}


def stats_weighted_arr(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[m]; w = w[m]
    if x.size == 0:
        return {"n": 0, "mean_w": np.nan, "std_w": np.nan, "cv_w": np.nan,
                "median_w": np.nan, "q1_w": np.nan, "q3_w": np.nan, "iqr_w": np.nan}

    wsum = float(np.sum(w))
    mean = float(np.sum(w * x) / wsum)
    var = float(np.sum(w * (x - mean) ** 2) / wsum)
    std = float(np.sqrt(var))
    cv = float(std / mean) if mean != 0 else np.nan

    q1 = float(weighted_quantile(x, w, 0.25))
    med = float(weighted_quantile(x, w, 0.50))
    q3 = float(weighted_quantile(x, w, 0.75))

    return {"n": int(x.size),
            "mean_w": mean, "std_w": std, "cv_w": cv,
            "median_w": med, "q1_w": q1, "q3_w": q3, "iqr_w": float(q3 - q1)}
# -----------------------------
# Distance dataframe
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance (km). lat/lon in degrees.
    """
    lat1 = np.deg2rad(lat1); lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2); lon2 = np.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def build_distance_dataframe(lats_small, lons_small, data_arr, lat_s, lon_s, var_name, w_area=None):
    """
    Create DataFrame with distance_km from station for each small-box grid cell.
    If w_area is provided, adds 'w_area' column.
    """
    if np.ndim(lats_small) == 1 and np.ndim(lons_small) == 1:
        LON2D, LAT2D = np.meshgrid(lons_small, lats_small)
    else:
        LAT2D = np.asarray(lats_small)
        LON2D = np.asarray(lons_small)

    dist_km = haversine_km(lat_s, lon_s, LAT2D, LON2D)

    df = pd.DataFrame({
        "lat": LAT2D.ravel(),
        "lon": LON2D.ravel(),
        "distance_km": dist_km.ravel(),
        var_name: np.asarray(data_arr).ravel(),
    })

    if w_area is not None:
        df["w_area"] = np.asarray(w_area).ravel()

    return df


# -----------------------------
# Period runner (cumulative sectors + cumulative distance)
# -----------------------------
def run_period_cumulative_sector_timeseries(
    base_path, product, species,
    station,
    start_dt, end_dt,
    cell_nums,
    radii_km,
    mode="A",
    step_minutes=30,
    weighted=True,
    tz_for_daynight="UTC",
):
    import numpy as np
    import pandas as pd
    import xarray as xr
    import datetime as dt
    from pathlib import Path

    # ---- always return a df with columns (even if empty) ----
    expected_cols = [
        "station","station_lat","station_lon","station_alt",
        "model_lat","model_lon",
        "date","time","timestamp","datetime",
        "season","day_night",
        "mode","sector_type","sector","radius",
        "k_star_center","z_target_m","center_ppb",
        "n_total","n_valid","n_excluded","frac_excluded",
        # stats (both sets; unused will be NaN)
        "n","mean","std","cv","median","q25","q75","iqr",
        "mean_w","std_w","cv_w","median_w","q1_w","q3_w","iqr_w",
    ]

    def empty_outputs():
        return pd.DataFrame(columns=expected_cols), pd.DataFrame()

    lat_s = float(station["Latitude"])
    lon_s = float(station["Longitude"])
    alt_s = float(station["Altitude"])
    station_name = station.get("Station_Name", "station")

    # --- Find first existing species file to read grid ---
    lats = lons = None
    first_found = None
    for d0, t0 in iter_timestamps(start_dt, end_dt, step_minutes):
        d0 = str(d0).zfill(8); t0 = str(t0).zfill(4)
        sp0, _, _, _, _ = build_paths(base_path, product, species, d0, t0)
        if not Path(sp0).exists():
            continue
        try:
            ds0 = xr.open_dataset(sp0, decode_times=False)
            lats = ds0["lat"].values
            lons = ds0["lon"].values
            ds0.close()
            first_found = (d0, t0)
            break
        except Exception:
            continue

    if first_found is None:
        return empty_outputs()

    i, j = nearest_grid_index(lat_s, lon_s, lats, lons)
    model_lat = float(lats[i]) if np.ndim(lats) == 1 else float(lats[i, j])
    model_lon = float(lons[j]) if np.ndim(lons) == 1 else float(lons[i, j])

    Ny, Nx = (lats.shape[0], lons.shape[0]) if np.ndim(lats) == 1 else lats.shape
    i1_s, i2_s, j1_s, j2_s, ii, jj = make_small_box_indices(i, j, Ny, Nx, cell_nums)

    lats_small = lats[i1_s:i2_s + 1]
    lons_small = lons[j1_s:j2_s + 1]
    Ny_s = len(lats_small); Nx_s = len(lons_small)

    radii = list(range(1, cell_nums + 1))

    # weights once
    w_area_small = compute_w_area_small(lats_small, lons_small, earth_radius_km=EARTH_RADIUS_KM) if weighted else None

    # sanitize radii_km thresholds
    radii_km = np.asarray(radii_km, dtype=float)
    radii_km = radii_km[np.isfinite(radii_km) & (radii_km > 0)]
    radii_km = np.unique(radii_km)
    radii_km.sort()

    # ---- PRECOMPUTE MASKS ONCE (huge speed win) ----
    ring_masks = compute_ring_sector_masks(ii, jj, Ny_s, Nx_s, radii)       # list of (Ny_s,Nx_s)
    cum_masks = cumulative_sector_masks(ring_masks)                         # list of (Ny_s,Nx_s)

    # distance grid once
    if lats_small.ndim == 1 and lons_small.ndim == 1:
        LON2D, LAT2D = np.meshgrid(lons_small, lats_small)
    else:
        LAT2D = lats_small
        LON2D = lons_small
    dist_km_grid = haversine_km(lat_s, lon_s, LAT2D, LON2D)
    dist_cum_masks = [dist_km_grid <= float(dmax) for dmax in radii_km]

    orog_cache = OrogCache()
    rows = []

    # helper to create row with all columns present
    def make_row():
        return {c: np.nan for c in expected_cols}

    for date, time in iter_timestamps(start_dt, end_dt, step_minutes):
        date = str(date).zfill(8)
        time = str(time).zfill(4)

        spf, Tf, PLf, RHf, orogf = build_paths(base_path, product, species, date, time)

        # skip missing timestep inputs
        if not all_inputs_exist(spf, Tf, PLf, RHf, orogf):
            continue

        ds_species = ds_T = ds_PL = ds_RH = None
        try:
            ds_species = xr.open_dataset(spf, decode_times=False)
            ds_T       = xr.open_dataset(Tf,  decode_times=False)
            ds_PL      = xr.open_dataset(PLf, decode_times=False)
            ds_RH      = xr.open_dataset(RHf, decode_times=False)
            ds_orog    = orog_cache.get(orogf)
        except Exception:
            for ds in (ds_species, ds_T, ds_PL, ds_RH):
                if ds is not None:
                    try: ds.close()
                    except Exception: pass
            continue

        # timestamp datetime
        ts = dt.datetime.strptime(date + time, "%Y%m%d%H%M")
        season = season_from_datetime(ts)
        dn = day_night_label(lat_s, lon_s, ts, tz=tz_for_daynight)

        # compute grid_ppb
        if mode.upper() == "A":
            grid_ppb, meta_v = extract_smallbox_ppb_optionA_fixed_k(
                ds_species, ds_T, ds_PL, ds_RH, ds_orog,
                species, alt_s,
                i, j, i1_s, i2_s, j1_s, j2_s,
                to_ppb_fn=to_ppb_mmr,
            )
        elif mode.upper() == "HEIGHT":
            grid_ppb, meta_v = extract_smallbox_ppb_optionHeight_fixed_z(
                ds_species, ds_T, ds_PL, ds_RH, ds_orog,
                species, alt_s,
                i, j, i1_s, i2_s, j1_s, j2_s,
                to_ppb_fn=to_ppb_mmr,
            )
        else:
            for ds in (ds_species, ds_T, ds_PL, ds_RH):
                ds.close()
            raise ValueError("mode must be 'A' or 'HEIGHT'")

        k_center = meta_v.get("k_star_center", meta_v.get("k_star", -999))
        z_target = meta_v.get("z_target_m", meta_v.get("z_star_m", np.nan))
        invalid_mask = meta_v.get("below_ground_mask", None)

        center_ppb = float(grid_ppb[ii, jj])

        # global valid mask for stats
        valid = np.isfinite(grid_ppb)
        if invalid_mask is not None:
            valid = valid & (~np.asarray(invalid_mask, dtype=bool))

        # ---- CUM SECTORS (fast) ----
        for k, mask_total in enumerate(cum_masks, start=1):
            mask_total = np.asarray(mask_total, dtype=bool)
            mask_valid = mask_total & valid

            n_total = int(mask_total.sum())
            n_valid = int(mask_valid.sum())
            n_excl = int(n_total - n_valid)
            frac_excl = float(n_excl / n_total) if n_total > 0 else np.nan

            vals = grid_ppb[mask_valid]

            row = make_row()
            row.update({
                "station": station_name,
                "station_lat": lat_s,
                "station_lon": lon_s,
                "station_alt": alt_s,
                "model_lat": model_lat,
                "model_lon": model_lon,
                "date": date,
                "time": time,
                "timestamp": f"{date} {time}",
                "datetime": ts,
                "season": season,
                "day_night": dn,
                "mode": mode.upper(),
                "sector_type": "CUM",
                "sector": f"C{k}",
                "radius": radii[k-1],
                "k_star_center": int(k_center),
                "z_target_m": float(z_target) if np.isfinite(z_target) else np.nan,
                "center_ppb": center_ppb,
                "n_total": n_total,
                "n_valid": n_valid,
                "n_excluded": n_excl,
                "frac_excluded": frac_excl,
            })

            if weighted and (w_area_small is not None):
                w = w_area_small[mask_valid]
                row.update(stats_weighted_arr(vals, w))
            else:
                row.update(stats_unweighted_arr(vals))

            rows.append(row)
        '''
        # ---- DIST CUM (fast) ----
        for dmax, mask_total in zip(radii_km, dist_cum_masks):
            mask_total = np.asarray(mask_total, dtype=bool)
            mask_valid = mask_total & valid

            n_total = int(mask_total.sum())
            n_valid = int(mask_valid.sum())
            n_excl = int(n_total - n_valid)
            frac_excl = float(n_excl / n_total) if n_total > 0 else np.nan

            vals = grid_ppb[mask_valid]

            row = make_row()
            row.update({
                "station": station_name,
                "station_lat": lat_s,
                "station_lon": lon_s,
                "station_alt": alt_s,
                "model_lat": model_lat,
                "model_lon": model_lon,
                "date": date,
                "time": time,
                "timestamp": f"{date} {time}",
                "datetime": ts,
                "season": season,
                "day_night": dn,
                "mode": mode.upper(),
                "sector_type": "DISTCUM",
                "sector": f"D≤{int(dmax)}",
                "radius": float(dmax),
                "k_star_center": int(k_center),
                "z_target_m": float(z_target) if np.isfinite(z_target) else np.nan,
                "center_ppb": center_ppb,
                "n_total": n_total,
                "n_valid": n_valid,
                "n_excluded": n_excl,
                "frac_excluded": frac_excl,
            })
             
            if weighted and (w_area_small is not None):
                w = w_area_small[mask_valid]
                row.update(stats_weighted_arr(vals, w))
            else:
                row.update(stats_unweighted_arr(vals))

            rows.append(row)
'''
        for ds in (ds_species, ds_T, ds_PL, ds_RH):
            ds.close()

    orog_cache.close_all()

    df_per_timestep = pd.DataFrame.from_records(rows, columns=expected_cols)

    if df_per_timestep.empty:
        return df_per_timestep, pd.DataFrame()

    # summary (same as before)
    if weighted:
        stat_cols = [c for c in ["n","mean_w","std_w","cv_w","median_w","iqr_w","q1_w","q3_w"] if c in df_per_timestep.columns]
    else:
        stat_cols = [c for c in ["n","mean","std","cv","median","iqr","q25","q75"] if c in df_per_timestep.columns]

    summary = (
        df_per_timestep
        .groupby(["station", "mode", "sector_type", "sector"], as_index=False)[stat_cols]
        .agg(["mean", "std", "median"])
    )
    summary.columns = [f"{a}_{b}" if b else a for (a, b) in summary.columns.to_flat_index()]

    return df_per_timestep, summary

# -----------------------------
# Utilities you already had (kept)
# -----------------------------
def stats_by_distance_bins(df, var_name, dist_bins_km, w_col=None):
    """
    NOTE: This is actually cumulative (<= dmax), not disjoint bins.
    Kept as-is.
    """
    records = []
    for dmax in dist_bins_km:
        sub = df[df["distance_km"] <= dmax]
        stats = sector_stats_unweighted(sub, var_name) if w_col is None else sector_stats_weighted(sub, var_name, w_col=w_col)
        stats["dmax_km"] = dmax
        records.append(stats)
    return pd.DataFrame(records)


def cumulative_mean_ratio_to_center(cum_dfs, var_name, center_value, labels=None, w_col=None):
    if labels is None:
        labels = [f"C{k}" for k in range(1, len(cum_dfs) + 1)]

    rows = [{"label": "C0", "ratio": 1.0}]

    for lab, df in zip(labels, cum_dfs):
        vals = pd.to_numeric(df[var_name], errors="coerce").to_numpy(dtype=float)

        if w_col is None:
            vals = vals[np.isfinite(vals)]
            mean_val = np.nanmean(vals) if vals.size else np.nan
        else:
            w = pd.to_numeric(df[w_col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(vals) & np.isfinite(w) & (w > 0)
            vals = vals[m]; w = w[m]
            mean_val = (np.sum(w * vals) / np.sum(w)) if vals.size else np.nan

        ratio = mean_val / center_value if (np.isfinite(mean_val) and center_value != 0) else np.nan
        rows.append({"label": lab, "ratio": float(ratio) if np.isfinite(ratio) else np.nan})

    return pd.DataFrame(rows)


def distance_cumulative_mean_ratio_to_center(df_dist, var_name, center_value, d_bins_km, w_col=None):
    rows = [{"label": "D0", "ratio": 1.0}]

    for dmax in d_bins_km:
        df_in = df_dist[df_dist["distance_km"] <= dmax]
        vals = pd.to_numeric(df_in[var_name], errors="coerce").to_numpy(dtype=float)

        if w_col is None:
            vals = vals[np.isfinite(vals)]
            mean_val = np.nanmean(vals) if vals.size else np.nan
        else:
            w = pd.to_numeric(df_in[w_col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(vals) & np.isfinite(w) & (w > 0)
            vals = vals[m]; w = w[m]
            mean_val = (np.sum(w * vals) / np.sum(w)) if vals.size else np.nan

        ratio = mean_val / center_value if (np.isfinite(mean_val) and center_value != 0) else np.nan
        rows.append({"label": f" {dmax}", "ratio": float(ratio) if np.isfinite(ratio) else np.nan})

    return pd.DataFrame(rows)