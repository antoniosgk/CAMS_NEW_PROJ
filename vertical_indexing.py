# vertical_indexing.py
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from metpy.constants import Rd, g
import warnings

from metpy.units import units as mp_units

import numpy as np

# constants (float)
RD = 287.05       # J/(kg*K)
G0 = 9.80665      # m/s^2
EPS = 0.622       # Rd/Rv approx

def _rh_to_virtual_temperature_numpy(p_pa, T_k, RH):
    """
    Vectorized virtual temperature using RH.
    p_pa, T_k, RH are numpy arrays of same shape.
    RH can be 0-1 or 0-100.
    Returns Tv [K] float array.
    """
    p = np.asarray(p_pa, dtype=float)
    T = np.asarray(T_k, dtype=float)
    rh = np.asarray(RH, dtype=float)

    # sanitize
    T = np.where(np.isfinite(T), T, np.nan)
    p = np.where(np.isfinite(p) & (p > 0), p, np.nan)

    # RH scaling (0-100 -> 0-1)
    rh_max = np.nanmax(rh)
    rhf = rh / 100.0 if (np.isfinite(rh_max) and rh_max > 1.5) else rh
    rhf = np.clip(rhf, 0.0, 1.0)

    # saturation vapor pressure over water (Bolton-style), in Pa
    Tc = T - 273.15
    es_hPa = 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5))
    es = es_hPa * 100.0

    # actual vapor pressure
    e = rhf * es

    # avoid e >= p
    e = np.minimum(e, 0.99 * p)

    # mixing ratio r = eps * e/(p-e) [kg/kg]
    r = EPS * e / (p - e)

    # specific humidity q
    q = r / (1.0 + r)

    # virtual temperature (approx)
    Tv = T * (1.0 + 0.61 * q)
    return Tv


def compute_heights_fast(p_prof_Pa, T_prof_K, RH=None, z0=0.0):
    """
    Fast hypsometric height integration (NumPy only).

    Supports:
      - 1D: (nlev,)
      - 2D: (nlev,ncol)

    Assumes lev order is consistent within a column. Uses:
      dz[k] = (RD/G0) * Tv_layer[k] * ln(p[k+1]/p[k])   (k=0..nlev-2)
    Anchors height at the max-pressure index (k_surf) to z0.

    Fast path: if k_surf is constant and equals the last level, uses pure cumsum (no Python loops).
    """
    p = np.asarray(p_prof_Pa, dtype=float)
    T = np.asarray(T_prof_K, dtype=float)
    if p.shape != T.shape:
        raise ValueError("p and T must have same shape")

    # floor pressure
    p_floor = 1e-6
    p = np.where(np.isfinite(p) & (p > 0), np.maximum(p, p_floor), np.nan)
    T = np.where(np.isfinite(T), T, np.nan)

    # virtual temperature
    if RH is None:
        Tv = T
    else:
        Tv = _rh_to_virtual_temperature_numpy(p, T, RH)

    if p.ndim == 1:
        nlev = p.shape[0]
        k_surf = int(np.nanargmax(p))

        # layer thicknesses dz for k=0..nlev-2
        Tv_layer = 0.5 * (Tv[1:] + Tv[:-1])
        log_ratio = np.log(p[1:] / p[:-1])
        dz = (RD / G0) * Tv_layer * log_ratio  # (nlev-1,)

        z = np.full(nlev, np.nan, dtype=float)
        z[k_surf] = float(z0)

        # if surface is last level -> fast vectorized upward integration
        if k_surf == nlev - 1:
            # z[k] = z0 + sum_{m=k}^{nlev-2} dz[m]
            z_up = float(z0) + np.cumsum(dz[::-1])[::-1]  # (nlev-1,)
            z[:-1] = z_up
            return z

        # otherwise fallback (still O(nlev) but cheap)
        for k in range(k_surf - 1, -1, -1):
            z[k] = z[k + 1] + dz[k]
        for k in range(k_surf + 1, nlev):
            z[k] = z[k - 1] - dz[k - 1]
        return z

    if p.ndim == 2:
        nlev, ncol = p.shape
        k_surf = np.nanargmax(p, axis=0)  # (ncol,)

        z0_arr = np.asarray(z0, dtype=float)
        if z0_arr.ndim == 0:
            z0_col = np.full(ncol, float(z0_arr), dtype=float)
        elif z0_arr.shape == (ncol,):
            z0_col = z0_arr.astype(float)
        else:
            raise ValueError("For 2D, z0 must be scalar or shape (ncol,)")

        Tv_layer = 0.5 * (Tv[1:, :] + Tv[:-1, :])          # (nlev-1,ncol)
        log_ratio = np.log(p[1:, :] / p[:-1, :])           # (nlev-1,ncol)
        dz = (RD / G0) * Tv_layer * log_ratio              # (nlev-1,ncol)

        z = np.full((nlev, ncol), np.nan, dtype=float)

        # FAST PATH: all columns have same surface index AND it's last level
        if np.all(k_surf == (nlev - 1)):
            z[-1, :] = z0_col
            z_up = z0_col[None, :] + np.cumsum(dz[::-1, :], axis=0)[::-1, :]  # (nlev-1,ncol)
            z[:-1, :] = z_up
            return z

        # General fallback: loop over levels (still much faster than units/metpy)
        # anchor
        cols = np.arange(ncol)
        z[k_surf, cols] = z0_col

        # integrate upward (k decreasing)
        for k in range(nlev - 2, -1, -1):
            mask = (k < k_surf)
            if np.any(mask):
                z[k, mask] = z[k + 1, mask] + dz[k, mask]

        # integrate downward (k increasing)
        for k in range(1, nlev):
            mask = (k > k_surf)
            if np.any(mask):
                z[k, mask] = z[k - 1, mask] - dz[k - 1, mask]

        return z

    raise ValueError("compute_heights_fast supports only 1D or 2D arrays")
#-----------------------------------------
def metpy_compute_heights(p_prof_Pa, T_prof_K, RH=None, z0=0.0):
    """
    Compute ASL geometric heights from pressure + temperature using the hypsometric equation.
    Supports:
      - 1D: (nlev,)
      - 2D: (nlev, ncol)

    If RH is given, compute virtual temperature using mixing ratio from RH.
    Anchor height at the max-pressure index to z0.
    """
    p_arr = np.asarray(p_prof_Pa, dtype=float)
    T_arr = np.asarray(T_prof_K, dtype=float)

    if p_arr.shape != T_arr.shape:
        raise ValueError("p_prof_Pa and T_prof_K must have the same shape.")

    RH_arr = None
    if RH is not None:
        RH_arr = np.asarray(RH, dtype=float)
        if RH_arr.shape != p_arr.shape:
            raise ValueError("RH must have the same shape as p_prof_Pa and T_prof_K.")

    # floor pressure
    p_floor = 1.0e-6
    p_arr = np.where(np.isfinite(p_arr) & (p_arr > 0.0), p_arr, np.nan)
    p_arr = np.where(np.isfinite(p_arr), np.maximum(p_arr, p_floor), np.nan)
    T_arr = np.where(np.isfinite(T_arr), T_arr, np.nan)

    p = p_arr * units.pascal
    T = T_arr * units.kelvin

    # virtual temperature
    if RH_arr is None:
        Tv = T
    else:
        rh_max = np.nanmax(RH_arr)
        RH_frac = RH_arr / 100.0 if (np.isfinite(rh_max) and rh_max > 1.5) else RH_arr
        RH_frac = np.clip(RH_frac, 0.0, 1.0)
        RH_q = RH_frac * units.dimensionless

        e_s = mpcalc.saturation_vapor_pressure(T)
        valid = np.isfinite(p.magnitude) & np.isfinite(T.magnitude) & (p > e_s)

        mr = np.zeros_like(p_arr, dtype=float)
        if np.any(valid):
            p_valid = p[valid]; T_valid = T[valid]; RH_valid = RH_q[valid]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Saturation mixing ratio is undefined*", category=UserWarning)
                mr_valid = mpcalc.mixing_ratio_from_relative_humidity(p_valid, T_valid, RH_valid)
            mr_valid = mr_valid.to("kg/kg").magnitude
            mr_valid = np.where(np.isfinite(mr_valid) & (mr_valid >= 0.0), mr_valid, 0.0)
            mr[valid] = mr_valid

        mixing_ratio = mr * units("kg/kg")
        Tv = mpcalc.virtual_temperature(T, mixing_ratio)

    # sanitize Tv
    Tv_mag = Tv.to("kelvin").magnitude
    T_mag = T.to("kelvin").magnitude
    Tv_mag = np.where(np.isfinite(Tv_mag), Tv_mag, T_mag)
    if np.any(~np.isfinite(Tv_mag)):
        Tv_mag = np.where(np.isfinite(Tv_mag), Tv_mag, 250.0)
    Tv = Tv_mag * units.kelvin

    # integrate
    if p.ndim == 1:
        nlev = p.shape[0]
        z = np.empty(nlev) * units.meter
        k_surf = int(np.nanargmax(p.magnitude))
        z[k_surf] = float(z0) * units.meter

        for k in range(k_surf - 1, -1, -1):
            Tv_layer = 0.5 * (Tv[k + 1] + Tv[k])
            ratio = p[k + 1] / p[k]
            dz = (Rd * Tv_layer / g) * np.log(ratio)
            z[k] = z[k + 1] + dz

        for k in range(k_surf + 1, nlev):
            Tv_layer = 0.5 * (Tv[k - 1] + Tv[k])
            ratio = p[k - 1] / p[k]
            dz = (Rd * Tv_layer / g) * np.log(ratio)
            z[k] = z[k - 1] + dz

        return z.magnitude

    if p.ndim == 2:
        nlev, ncol = p.shape
        z = np.empty_like(p.magnitude) * units.meter
        k_surf = np.nanargmax(p.magnitude, axis=0)

        z0_arr = np.asarray(z0, dtype=float)
        if z0_arr.ndim == 0:
            z_surf = np.full(ncol, z0_arr) * units.meter
        elif z0_arr.shape == (ncol,):
            z_surf = z0_arr * units.meter
        else:
            raise ValueError("For 2D p/T, z0 must be scalar or shape (ncol,)")

        for j in range(ncol):
            z[k_surf[j], j] = z_surf[j]

        for k in range(nlev - 2, -1, -1):
            mask = k < k_surf
            if not np.any(mask):
                continue
            Tv_layer = 0.5 * (Tv[k + 1, mask] + Tv[k, mask])
            ratio = p[k + 1, mask] / p[k, mask]
            dz = (Rd * Tv_layer / g) * np.log(ratio)
            z[k, mask] = z[k + 1, mask] + dz

        for k in range(1, nlev):
            mask = k > k_surf
            if not np.any(mask):
                continue
            Tv_layer = 0.5 * (Tv[k - 1, mask] + Tv[k, mask])
            ratio = p[k - 1, mask] / p[k, mask]
            dz = (Rd * Tv_layer / g) * np.log(ratio)
            z[k, mask] = z[k - 1, mask] + dz

        return z.magnitude

    raise ValueError("metpy_compute_heights supports only 1D or 2D p/T.")


def metpy_find_level_index(p_prof_Pa, T_prof_K, station_alt_m, RH=None, z_surf_model=0.0):
    """
    Compute z(k) profile (ASL) and return the level index k* closest to station altitude.

    Returns:
      idx_level (int)
      p_level_hPa (float)
      z_level_m (float)
    """
    p_arr = np.asarray(p_prof_Pa)
    T_arr = np.asarray(T_prof_K)
    RH_arr = np.asarray(RH) if RH is not None else None

    if p_arr.shape != T_arr.shape:
        raise ValueError("p_prof_Pa and T_prof_K must have the same shape.")

    z_prof = metpy_compute_heights(p_arr, T_arr, RH=RH_arr, z0=z_surf_model)
    diff = np.abs(z_prof - float(station_alt_m))
    diff[~np.isfinite(diff)] = np.inf
    idx = int(np.argmin(diff))

    p_hPa = float(p_arr[idx] / 100.0)
    z_m = float(z_prof[idx])
    return idx, p_hPa, z_m


def surface_height_grid_m(ds_orog, i1, i2, j1, j2):
    """
    Convert PHIS (m^2/s^2) to surface height z0 (m) for a slice.
    """
    PHIS = ds_orog["PHIS"].isel(time=0, lat=slice(i1, i2 + 1), lon=slice(j1, j2 + 1)).values
    return (PHIS * mp_units("m^2/s^2") / g).to("meter").magnitude


def extract_smallbox_ppb_optionA_fixed_k(
    ds_species, ds_T, ds_PL, ds_RH, ds_orog,
    species, station_alt_m,
    i, j, i1_s, i2_s, j1_s, j2_s,
    to_ppb_fn,
):
    """
    MODE = "A"
    - Compute k* at the central (i,j) column using station altitude.
    - Extract species at this SAME k* for the entire small box.
    """
    PHIS_c = ds_orog["PHIS"].isel(time=0, lat=i, lon=j).item()
    z0_c = (PHIS_c * mp_units("m^2/s^2") / g).to("meter").magnitude

    T_c  = ds_T["T"].values[0, :, i, j]
    p_c  = ds_PL["PL"].values[0, :, i, j]
    RH_c = ds_RH["RH"].values[0, :, i, j]

    k_star, p_hPa, z_star = metpy_find_level_index(
        p_prof_Pa=p_c,
        T_prof_K=T_c,
        RH=RH_c,
        station_alt_m=station_alt_m,
        z_surf_model=z0_c,
    )

    field = ds_species[species].values[0, k_star, i1_s:i2_s + 1, j1_s:j2_s + 1]
    grid_ppb = to_ppb_fn(field, species)
    min=np.nanmin(grid_ppb)
    max=np.nanmax(grid_ppb)
    meta = {"k_star_center": int(k_star), "p_hPa": float(p_hPa), "z_star_m": float(z_star),"min": np.nanmin(grid_ppb),"max":np.nanmax(grid_ppb)}
    return grid_ppb, meta


def extract_smallbox_ppb_optionHeight_fixed_z(
    ds_species, ds_T, ds_PL, ds_RH, ds_orog,
    species, station_alt_m,
    i, j, i1_s, i2_s, j1_s, j2_s,
    to_ppb_fn,
    below_ground_margin_m=0.0,   # <-- ADD (optional)
):
    """
    MODE = "HEIGHT"
    - Find k* at central column using station altitude.
    - Take the HEIGHT z* of that central level.
    - For each small-box cell, choose k(cell) whose z(k,cell) is closest to z*.
      (so k varies spatially but targets the same height surface)

    New behavior:
    - Cells where z_target is below ground (z0_cell > z_target - margin) are masked (NaN).
    """

    PHIS_c = ds_orog["PHIS"].isel(time=0, lat=i, lon=j).item()
    z0_c = (PHIS_c * mp_units("m^2/s^2") / g).to("meter").magnitude

    T_c  = ds_T["T"].values[0, :, i, j]
    p_c  = ds_PL["PL"].values[0, :, i, j]
    RH_c = ds_RH["RH"].values[0, :, i, j]

    k_star_c, p_hPa_c, z_star = metpy_find_level_index(
        p_prof_Pa=p_c,
        T_prof_K=T_c,
        RH=RH_c,
        station_alt_m=station_alt_m,
        z_surf_model=z0_c,
    )
    z_target = float(z_star)

    # small box profiles
    T_box  = ds_T["T"].values[0, :, i1_s:i2_s + 1, j1_s:j2_s + 1]
    p_box  = ds_PL["PL"].values[0, :, i1_s:i2_s + 1, j1_s:j2_s + 1]
    RH_box = ds_RH["RH"].values[0, :, i1_s:i2_s + 1, j1_s:j2_s + 1]

    nlev, Ny_s, Nx_s = T_box.shape
    ncol = Ny_s * Nx_s

    T_2d  = T_box.reshape(nlev, ncol)
    p_2d  = p_box.reshape(nlev, ncol)
    RH_2d = RH_box.reshape(nlev, ncol)

    z0_grid_2d = surface_height_grid_m(ds_orog, i1_s, i2_s, j1_s, j2_s)  # (Ny_s, Nx_s)
    z0_grid_1d = z0_grid_2d.reshape(ncol)

    # --- BELOW-GROUND MASK (key fix) ---
    # invalid if target is below ground (optionally allow small margin)
    below_ground_1d = z0_grid_1d > (z_target - float(below_ground_margin_m))

    # compute z profiles
    z_2d = compute_heights_fast(p_2d, T_2d, RH=RH_2d, z0=z0_grid_1d) #or metpy_compute_heights()

    diff = np.abs(z_2d - z_target)
    diff[~np.isfinite(diff)] = np.inf

    # enforce invalid columns: never select any k there
    diff[:, below_ground_1d] = np.inf

    k_grid_1d = np.argmin(diff, axis=0)  # (ncol,)

    # extract species at k(cell)
    sp_box = ds_species[species].values[0, :, i1_s:i2_s + 1, j1_s:j2_s + 1]  # (lev,Ny,Nx)
    sp_2d = sp_box.reshape(nlev, ncol)
    cols = np.arange(ncol)

    sel_1d = sp_2d[k_grid_1d, cols]  # (ncol,)
    sel_1d = sel_1d.astype(float)

    # set below-ground columns to NaN (plots white; stats ignore)
    sel_1d[below_ground_1d] = np.nan

    sel = sel_1d.reshape(Ny_s, Nx_s)

    grid_ppb = to_ppb_fn(sel, species)

    meta = {
        "k_star_center": int(k_star_c),
        "p_hPa_center": float(p_hPa_c),
        "z_star_m": float(z_target),
        "k_grid": k_grid_1d.reshape(Ny_s, Nx_s),
        "below_ground_mask": below_ground_1d.reshape(Ny_s, Nx_s),
        "below_ground_n": int(np.sum(below_ground_1d))
    }
    print(meta)
    return grid_ppb, meta