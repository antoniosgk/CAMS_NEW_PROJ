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


def sector_table(mask, lats_small, lons_small, data_arr, var_name, w_area=None):
    """
    Convert a mask + 2D field into a DataFrame of selected grid cells.
    If w_area is provided, adds 'w_area' column (required for weighted stats).
    """
    iy, ix = np.where(mask)

    if np.ndim(lats_small) == 1 and np.ndim(lons_small) == 1:
        lat_vals = np.asarray(lats_small)[iy]
        lon_vals = np.asarray(lons_small)[ix]
    else:
        lat_vals = np.asarray(lats_small)[iy, ix]
        lon_vals = np.asarray(lons_small)[iy, ix]

    vals = np.asarray(data_arr)[iy, ix]

    out = pd.DataFrame({
        "lat_idx": iy,
        "lon_idx": ix,
        "lat": lat_vals,
        "lon": lon_vals,
        var_name: vals,
    })

    if w_area is not None:
        out["w_area"] = np.asarray(w_area)[iy, ix]

    return out


def compute_sector_tables_generic(ii, jj, lats_small, lons_small, data_arr, var_name, radii, w_area=None):
    Ny, Nx = np.asarray(data_arr).shape
    masks = compute_ring_sector_masks(ii, jj, Ny, Nx, radii)
    dfs = [sector_table(m, lats_small, lons_small, data_arr, var_name, w_area=w_area) for m in masks]
    return dfs, masks


def cumulative_sector_masks(sector_masks):
    running = np.zeros_like(sector_masks[0], dtype=bool)
    out = []
    for S in sector_masks:
        running = running | S
        out.append(running.copy())
    return out


def compute_cumulative_sector_tables(sector_masks, lats_small, lons_small, data_arr, var_name, w_area=None):
    cum_masks = cumulative_sector_masks(sector_masks)
    dfs = [sector_table(m, lats_small, lons_small, data_arr, var_name, w_area=w_area) for m in cum_masks]
    return dfs, cum_masks


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
    radii_km,             # cumulative distance thresholds, e.g. [10,20,50,100]
    mode="A",             # "A" or "HEIGHT"
    step_minutes=30,
    weighted=True
):
    """
    Output per timestep:
      - sector_type == "CUM": sectors C1..Ck (cumulative rings)
      - sector_type == "DISTCUM": sectors D≤10, D≤20, ... (cumulative distance)

    Weighted=True:
      - adds w_area_small once (from lats_small/lons_small)
      - uses sector_stats_weighted -> mean_w/std_w/cv_w/...
    """
    import xarray as xr  # local import ok

    lat_s = float(station["Latitude"])
    lon_s = float(station["Longitude"])
    alt_s = float(station["Altitude"])
    station_name = station.get("Station_Name", "station")

    # --- Find first existing timestamp for reading grid ---
    lats = lons = None
    first_found = None
    for d0, t0 in iter_timestamps(start_dt, end_dt, step_minutes):
        sp0, _, _, _, _ = build_paths(base_path, product, species, d0, t0)
        try:
            ds0 = xr.open_dataset(sp0)
            lats = ds0["lat"].values
            lons = ds0["lon"].values
            ds0.close()
            first_found = (d0, t0)
            break
        except FileNotFoundError:
            continue

    if first_found is None:
        raise FileNotFoundError("No species files found in the requested period.")

    i, j = nearest_grid_index(lat_s, lon_s, lats, lons)
    Ny, Nx = (lats.shape[0], lons.shape[0]) if np.ndim(lats) == 1 else lats.shape
    i1_s, i2_s, j1_s, j2_s, ii, jj = make_small_box_indices(i, j, Ny, Nx, cell_nums)

    # small box coords
    lats_small = lats[i1_s:i2_s + 1]
    lons_small = lons[j1_s:j2_s + 1]

    radii = list(range(1, cell_nums + 1))

    # ✅ IMPORTANT: compute weights only AFTER lats_small/lons_small exist
    w_area_small = compute_w_area_small(lats_small, lons_small, earth_radius_km=EARTH_RADIUS_KM) if weighted else None

    # sanitize radii_km thresholds
    radii_km = np.asarray(radii_km, dtype=float)
    radii_km = radii_km[np.isfinite(radii_km) & (radii_km > 0)]
    radii_km = np.unique(radii_km)
    radii_km.sort()

    orog_cache = OrogCache()
    rows = []

    for date, time in iter_timestamps(start_dt, end_dt, step_minutes):
        spf, Tf, PLf, RHf, orogf = build_paths(base_path, product, species, date, time)

        ds_species = ds_T = ds_PL = ds_RH = None
        try:
            ds_species = xr.open_dataset(spf)
            ds_T = xr.open_dataset(Tf)
            ds_PL = xr.open_dataset(PLf)
            ds_RH = xr.open_dataset(RHf)
        except FileNotFoundError:
            for ds in (ds_species, ds_T, ds_PL, ds_RH):
                if ds is not None:
                    try:
                        ds.close()
                    except Exception:
                        pass
            continue

        ds_orog = orog_cache.get(orogf)

        # --- vertical selection: build 2D ppb grid in the small box ---
        if mode.upper() == "A":
            grid_ppb, meta_v = extract_smallbox_ppb_optionA_fixed_k(
                ds_species, ds_T, ds_PL, ds_RH, ds_orog,
                species, alt_s,
                i, j, i1_s, i2_s, j1_s, j2_s,
                to_ppb_fn=to_ppb_mmr,
            )
            z_target = meta_v["z_star_m"]
            k_center = meta_v["k_star"]
        elif mode.upper() == "HEIGHT":
            grid_ppb, meta_v = extract_smallbox_ppb_optionHeight_fixed_z(
                ds_species, ds_T, ds_PL, ds_RH, ds_orog,
                species, alt_s,
                i, j, i1_s, i2_s, j1_s, j2_s,
                to_ppb_fn=to_ppb_mmr,
            )
            z_target = meta_v["z_target_m"]
            k_center = meta_v["k_star_center"]
        else:
            for ds in (ds_species, ds_T, ds_PL, ds_RH):
                ds.close()
            raise ValueError("mode must be 'A' or 'HEIGHT'")

        center_ppb = float(grid_ppb[ii, jj])

        # --- CUMULATIVE SECTORS (C1..Ck) ---
        _, sector_masks = compute_sector_tables_generic(
            ii, jj, lats_small, lons_small, grid_ppb, species, radii=radii,
            w_area=w_area_small
        )
        cum_dfs, _ = compute_cumulative_sector_tables(
            sector_masks, lats_small, lons_small, grid_ppb, species,
            w_area=w_area_small
        )

        for k, df_c in enumerate(cum_dfs, start=1):
            st = sector_stats_weighted(df_c, species, w_col="w_area") if weighted else sector_stats_unweighted(df_c, species)
            rows.append({
                "station": station_name,
                "date": date,
                "time": time,
                "timestamp": f"{date} {time}",
                "mode": mode.upper(),
                "sector_type": "CUM",
                "sector": f"C{k}",
                "radius": radii[k - 1],
                "k_star_center": int(k_center),
                "z_target_m": float(z_target),
                "center_ppb": center_ppb,
                **st,
            })

        # --- DISTANCE CUMULATIVE ONLY (D≤...) ---
        df_dist = build_distance_dataframe(
            lats_small, lons_small, grid_ppb, lat_s, lon_s,
            var_name=species, w_area=w_area_small
        )

        for dmax in radii_km:
            df_cum = df_dist[df_dist["distance_km"] <= float(dmax)]
            if df_cum.empty:
                continue
            st = sector_stats_weighted(df_cum, species, w_col="w_area") if weighted else sector_stats_unweighted(df_cum, species)
            rows.append({
                "station": station_name,
                "date": date,
                "time": time,
                "timestamp": f"{date} {time}",
                "mode": mode.upper(),
                "sector_type": "DISTCUM",
                "sector": f"D≤{int(dmax)}",
                "radius": float(dmax),
                "k_star_center": int(k_center),
                "z_target_m": float(z_target),
                "center_ppb": center_ppb,
                **st,
            })

        for ds in (ds_species, ds_T, ds_PL, ds_RH):
            ds.close()

    orog_cache.close_all()

    df_per_timestep = pd.DataFrame(rows)
    if df_per_timestep.empty:
        return df_per_timestep, df_per_timestep

    # --- temporal summary over time for each sector ---
    if weighted:
        stat_cols = [c for c in ["n", "mean_w", "std_w", "cv_w", "median_w", "iqr_w", "q1_w", "q3_w"] if c in df_per_timestep.columns]
    else:
        stat_cols = [c for c in ["n", "mean", "std", "cv", "median", "iqr", "q25", "q75"] if c in df_per_timestep.columns]

    summary = (
        df_per_timestep
        .groupby(["station", "mode", "sector_type", "sector"], as_index=False)[stat_cols]
        .agg(["mean", "std", "median"])
    )
    summary.columns = [f"{a}_{b}" if b else a for (a, b) in summary.columns.to_flat_index()]
    df_temporal_summary = summary

    return df_per_timestep, df_temporal_summary


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