#%%
"""
Area-based multi-model fit backfill for ONE station (extensible to all).

Physical rationale
------------------
The earlier fits used sector index x = 1..10, which treats sectors as equally
spaced. Physically the relevant abscissa is the cumulative SURFACE AREA of each
nested-square sector, because the metric (ratio, CV) is a running statistic over
a growing area. Area is computed in km^2 on the sphere, so it is consistent
across stations regardless of latitude.

Sector geometry
---------------
Nested squares centred on the station's model grid cell (i_center, j_center):
    C0 = 1x1   = 1   cell   (central pixel; not fitted, baseline only)
    C1 = 3x3   = 9   cells
    C2 = 5x5   = 25  cells
    ...
    Ck = (2k+1)^2 cells
    ...
    C10 = 21x21 = 441 cells
Grid resolution: GRID_DEG = 0.0625 deg in both lat and lon.

Cumulative area of sector Ck = sum of the true areas of all cells in the
(2k+1) x (2k+1) block. Each cell at latitude band [phi1, phi2] spanning
dlon in longitude has area:
    A_cell = R^2 * (sin phi2 - sin phi1) * dlon_radians
with R the Earth radius. Cells at higher latitude are smaller (cos-lat effect),
handled exactly by the sin difference.

Models fitted per timestep (x = cumulative area in km^2, y = ratio or cv_w):
    linear        y = a + b x
    quadratic     y = a + b x + c x^2
    cubic         y = a + b x + c x^2 + d x^3
    logarithmic   y = a + b ln(x)
    exponential   y = a + b exp(c x)
    power         y = a x^b
    saturating    y = a + b (1 - exp(-x / lam))     [lam stored as 'c']

Linear-family models (linear, quadratic, cubic, logarithmic) are fitted by
closed-form least squares (vectorised, no iteration). The three genuinely
nonlinear models (exponential, power, saturating) use scipy.curve_fit with
bounds and graceful NaN fallback.

For every (variable, model, timestep) the script stores:
    a, b, c, d (as applicable)
    slope        = average slope of the fitted curve across the area range,
                   (y(x_max) - y(x_min)) / (x_max - x_min)        [Option A]
    r2, adj_r2, aic, aicc, bic, n, rss
Per variable it also stores:
    {var}_best_model_aic / _aicc / _bic

Output: one CSV per station, suffix configurable.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import time
import datetime as dt
from scipy.optimize import curve_fit

# ============================================================
# SETTINGS
# ============================================================

IN_DIR  = Path("/mnt/store01/agkiokas/CAMS/derived_station_metrics/")            # where the source CSV lives
OUT_DIR = Path("/mnt/store01/agkiokas/CAMS/fit_outputs")            # where to write results
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = "O3"

# One-station mode: name the station id. For the network run later, this can be
# turned into a loop / CLI argument over many station files.
STATION_ID = "1002A"

# Source filename pattern (the file you already have).
IN_FILENAME = f"{STATION_ID}_{SPECIES}_sector_ratio_cv_fits_with_aic.csv"

# Output filename.
OUT_FILENAME = f"{STATION_ID}_{SPECIES}_area_fits.csv"

# Variables to fit (each must have {var}_C1..{var}_C10 columns).
VARIABLES = ["ratio", "cv_w","mean_w"]

# Grid resolution (degrees), same in lat and lon.
GRID_DEG = 0.0625

# Earth radius (km). Mean radius.
EARTH_RADIUS_KM = 6371.0088

# Sectors fitted: C1..C10 (C0 is the central pixel, baseline only, not fitted).
SECTORS = list(range(1, 11))

# Minimum valid points required to attempt a fit of given complexity.
# (Used to avoid degenerate fits; with 10 points this is rarely triggered.)
MIN_POINTS = {
    "linear": 2, "quadratic": 3, "cubic": 4,
    "logarithmic": 2, "exponential": 3, "power": 2, "saturating": 3,
}

# Free-parameter count k per model (for AIC/BIC; +1 for variance added inside).
MODEL_K = {
    "linear": 2, "quadratic": 3, "cubic": 4,
    "logarithmic": 2, "exponential": 3, "power": 2, "saturating": 3,
}

MODELS = ["linear", "quadratic", "cubic", "logarithmic",
          "exponential", "power", "saturating"]

# Time-key detection candidates (first that exists wins).
TIME_CANDIDATES = ["datetime", "timestamp", "time_key"]


# ============================================================
# SECTOR AREA (TRUE SPHERICAL, PER STATION)
# ============================================================
 
def cell_area_km2(lat_south_deg, lat_north_deg, dlon_deg):
    """
    Area of a single grid cell spanning [lat_south, lat_north] in latitude and
    dlon_deg in longitude, on a sphere of radius EARTH_RADIUS_KM.
 
    A = R^2 * (sin(phi_north) - sin(phi_south)) * dlon_radians
    """
    phi_s = np.radians(lat_south_deg)
    phi_n = np.radians(lat_north_deg)
    dlon  = np.radians(dlon_deg)
    return (EARTH_RADIUS_KM ** 2) * (np.sin(phi_n) - np.sin(phi_s)) * dlon
 
 
def cumulative_sector_areas_km2(center_lat_deg):
    """
    Compute cumulative area (km^2) for sectors C1..C10 as nested squares
    centred on a cell at latitude center_lat_deg.
 
    Ck is a (2k+1) x (2k+1) block of cells. Because area depends only on
    latitude (not longitude), we sum cell areas row by row: each latitude row
    of the block contributes (2k+1) identical cells (same lat band, each
    dlon = GRID_DEG wide).
 
    Returns
    -------
    np.ndarray of length 10: cumulative km^2 for C1..C10.
    """
    half_deg = GRID_DEG / 2.0
 
    # Latitude edges of each cell row, indexed by offset r from the centre row.
    # The central cell row spans [center_lat - half, center_lat + half].
    # Row offset r (can be negative) spans center +/- r*GRID_DEG.
    def row_area_sum(k):
        """Sum of areas of one full (2k+1)-wide row block, for all rows in Ck."""
        total = 0.0
        n_side = 2 * k + 1                      # cells per row in Ck
        for r in range(-k, k + 1):             # latitude rows -k..k
            row_center = center_lat_deg + r * GRID_DEG
            lat_s = row_center - half_deg
            lat_n = row_center + half_deg
            a_cell = cell_area_km2(lat_s, lat_n, GRID_DEG)
            total += a_cell * n_side           # n_side identical cells in this row
        return total
 
    areas = np.array([row_area_sum(k) for k in SECTORS], dtype=float)
    return areas
 
 
# ============================================================
# FIT QUALITY METRICS
# ============================================================
 
def fit_quality_metrics(y_true, y_pred, k):
    """
    n, rss, r2, adj_r2, aic, aicc, bic for a model with k free params.
    AIC/AICc/BIC use Gaussian ML variance sigma^2 = RSS/n; +1 param for variance.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid  = np.isfinite(y_true) & np.isfinite(y_pred)
    n      = int(valid.sum())
 
    out = {"n": n, "rss": np.nan, "r2": np.nan, "adj_r2": np.nan,
           "aic": np.nan, "aicc": np.nan, "bic": np.nan}
    if n < 2:
        return out
 
    yt = y_true[valid]
    yp = y_pred[valid]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    out["rss"] = ss_res
 
    if ss_tot > 0:
        out["r2"] = 1.0 - ss_res / ss_tot
 
    denom = n - k - 1
    if denom > 0 and np.isfinite(out["r2"]):
        out["adj_r2"] = 1.0 - (1.0 - out["r2"]) * (n - 1) / denom
 
    sigma2   = ss_res / n
    n_params = k + 1
    if sigma2 > 0:
        ll = n * np.log(sigma2)
        out["aic"] = ll + 2.0 * n_params
        out["bic"] = ll + np.log(n) * n_params
        aicc_denom = n - n_params - 1
        out["aicc"] = (out["aic"] + 2.0 * n_params * (n_params + 1) / aicc_denom
                       if aicc_denom > 0 else np.inf)
    else:
        out["aic"] = out["aicc"] = out["bic"] = -np.inf
    return out
 
 
def average_slope(x, y_pred):
    """
    Option A slope: chord across the fitted curve over the area range.
        (y(x_max) - y(x_min)) / (x_max - x_min)
    For a linear fit this equals b exactly.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    i_lo = int(np.argmin(x))
    i_hi = int(np.argmax(x))
    dx = x[i_hi] - x[i_lo]
    if dx == 0:
        return np.nan
    return (y_pred[i_hi] - y_pred[i_lo]) / dx
 
 
# ============================================================
# MODELS
# ============================================================
# Each fit_* returns a dict with keys a,b,c,d (NaN where unused), plus the
# predicted y vector for metric/slope computation.
 
def _nan_params():
    return {"a": np.nan, "b": np.nan, "c": np.nan, "d": np.nan}
 
 
# ---- linear family: closed-form least squares on a design matrix ----
 
def _lstsq_fit(x, y, design_fn, param_names):
    """
    Generic closed-form least-squares fit.
    design_fn(x) -> design matrix columns (n, p).
    param_names  -> names mapped onto the solved coefficients in order.
    Returns (params_dict, y_pred) or (nan_params, None) if singular.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < len(param_names):
        return _nan_params(), None
    xv, yv = x[valid], y[valid]
    A = design_fn(xv)
    try:
        coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    except np.linalg.LinAlgError:
        return _nan_params(), None
    params = _nan_params()
    for name, value in zip(param_names, coef):
        params[name] = float(value)
    y_pred_full = np.full_like(x, np.nan, dtype=float)
    y_pred_full[valid] = A @ coef
    return params, y_pred_full
 
 
def fit_linear(x, y):
    # y = a + b x   -> design [1, x], params [a, b]
    return _lstsq_fit(x, y, lambda xv: np.column_stack([np.ones_like(xv), xv]),
                      ["a", "b"])
 
 
def fit_quadratic(x, y):
    # y = a + b x + c x^2
    return _lstsq_fit(x, y,
                      lambda xv: np.column_stack([np.ones_like(xv), xv, xv**2]),
                      ["a", "b", "c"])
 
 
def fit_cubic(x, y):
    # y = a + b x + c x^2 + d x^3
    return _lstsq_fit(x, y,
                      lambda xv: np.column_stack([np.ones_like(xv), xv, xv**2, xv**3]),
                      ["a", "b", "c", "d"])
 
 
def fit_logarithmic(x, y):
    # y = a + b ln(x)   (x > 0 guaranteed: area is positive)
    return _lstsq_fit(x, y,
                      lambda xv: np.column_stack([np.ones_like(xv), np.log(xv)]),
                      ["a", "b"])
 
 
# ---- nonlinear models: scipy.curve_fit with bounds and fallback ----
 
from scipy.optimize import curve_fit, least_squares
 
 
# ---- nonlinear models: least_squares with analytic jacobian (fast) ----
# x is scaled to [0,1] internally for numerical stability and fast convergence;
# fitted parameters are converted back to the original km^2 x-scale before
# returning, so all stored parameters are in physical (per-km^2) units.
 
def fit_exponential(x, y):
    """
    y = a + b exp(c x).  Internally fits on xs = x / x_max, then rescales c.
    On xs: y = a + b exp(c' xs) with c' = c * x_max  =>  c = c' / x_max.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < MIN_POINTS["exponential"]:
        return _nan_params(), None
    xv, yv = x[valid], y[valid]
    if np.nanstd(yv) < 1e-12:
        return _nan_params(), None
    xmax = float(np.nanmax(xv))
    if xmax <= 0:
        return _nan_params(), None
    xs = xv / xmax
    ym = float(np.mean(yv))
    yr = float(np.nanmax(yv) - np.nanmin(yv))
    if yr == 0:
        return _nan_params(), None
 
    def resid(p):
        a, b, c = p
        return a + b * np.exp(np.clip(c * xs, -50, 50)) - yv
 
    def jac(p):
        a, b, c = p
        e = np.exp(np.clip(c * xs, -50, 50))
        return np.column_stack([np.ones_like(xs), e, b * xs * e])
 
    try:
        r = least_squares(
            resid, [ym, yv[0] - ym, 0.0], jac=jac,
            bounds=([np.nanmin(yv) - 10*yr, -10*yr, -50.0],
                    [np.nanmax(yv) + 10*yr,  10*yr,  50.0]),
            max_nfev=500, method="trf",
        )
        a, b, c_scaled = r.x
        c = c_scaled / xmax                      # back to per-km^2 units
        params = _nan_params(); params.update(a=a, b=b, c=c)
        y_pred = np.full_like(x, np.nan, dtype=float)
        y_pred[valid] = a + b * np.exp(np.clip(c * xv, -50, 50))
        return params, y_pred
    except Exception:
        return _nan_params(), None
 
 
def fit_power(x, y):
    """y = a x^b. Init from log-log linear regression where possible."""
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if valid.sum() < MIN_POINTS["power"]:
        return _nan_params(), None
    xv, yv = x[valid], y[valid]
    try:
        pos = yv > 0
        if pos.sum() >= 2:
            bb, la = np.polyfit(np.log(xv[pos]), np.log(yv[pos]), 1)
            a0, b0 = np.exp(la), bb
        else:
            a0, b0 = (float(np.nanmean(np.abs(yv))) or 1.0), 0.0
        if not np.isfinite(a0) or a0 == 0:
            a0 = 1.0
 
        def resid(p):
            a, b = p
            return a * np.power(xv, b) - yv
 
        # Bound the exponent to a physically sane range; an unbounded power
        # exponent can diverge on noisy data and produce absurd R^2.
        r = least_squares(resid, [a0, b0],
                          bounds=([-np.inf, -10.0], [np.inf, 10.0]),
                          max_nfev=500, method="trf")
        a, b = r.x
        y_pred = np.full_like(x, np.nan, dtype=float)
        y_pred[valid] = a * np.power(xv, b)
        # reject pathological fits (overflow / non-finite predictions)
        if not np.all(np.isfinite(y_pred[valid])):
            return _nan_params(), None
        # reject fits far worse than a flat mean line (R^2 << 0): a power law
        # that fits worse than the mean carries no useful information here.
        ss_res = float(np.sum((yv - y_pred[valid]) ** 2))
        ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
        if ss_tot > 0 and (1.0 - ss_res / ss_tot) < -1.0:
            return _nan_params(), None
        params = _nan_params(); params.update(a=a, b=b)
        return params, y_pred
    except Exception:
        return _nan_params(), None
 
 
def fit_saturating(x, y):
    """
    y = a + b (1 - exp(-x/lam)); lam stored in 'c'.
    Internally fits on xs = x / x_max, then rescales lam: lam = lam' * x_max.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < MIN_POINTS["saturating"]:
        return _nan_params(), None
    xv, yv = x[valid], y[valid]
    if np.nanstd(yv) < 1e-12:
        return _nan_params(), None
    xmax = float(np.nanmax(xv))
    if xmax <= 0:
        return _nan_params(), None
    xs = xv / xmax
    y0 = float(yv[np.argmin(xv)])
    ye = float(yv[np.argmax(xv)])
 
    def resid(p):
        a, b, lam = p
        return a + b * (1.0 - np.exp(-np.clip(xs / lam, -50, 50))) - yv
 
    try:
        r = least_squares(
            resid, [y0, ye - y0, 0.33],
            bounds=([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, np.inf]),
            max_nfev=500, method="trf",
        )
        a, b, lam_scaled = r.x
        lam = lam_scaled * xmax                   # back to km^2 units
        params = _nan_params(); params.update(a=a, b=b, c=lam)
        y_pred = np.full_like(x, np.nan, dtype=float)
        y_pred[valid] = a + b * (1.0 - np.exp(-np.clip(xv / lam, -50, 50)))
        return params, y_pred
    except Exception:
        return _nan_params(), None
 
 
FIT_FUNCS = {
    "linear":      fit_linear,
    "quadratic":   fit_quadratic,
    "cubic":       fit_cubic,
    "logarithmic": fit_logarithmic,
    "exponential": fit_exponential,
    "power":       fit_power,
    "saturating":  fit_saturating,
}
 
 
def fit_all_models_for_variable(x, y, prefix):
    """
    Fit all 7 models to (x, y). Returns a flat dict of columns:
        {prefix}_{model}_a/_b/_c/_d
        {prefix}_{model}_slope
        {prefix}_{model}_r2/_adj_r2/_aic/_aicc/_bic/_n/_rss
    """
    out = {}
    for model in MODELS:
        params, y_pred = FIT_FUNCS[model](x, y)
        k = MODEL_K[model]
 
        out[f"{prefix}_{model}_a"] = params["a"]
        out[f"{prefix}_{model}_b"] = params["b"]
        out[f"{prefix}_{model}_c"] = params["c"]
        out[f"{prefix}_{model}_d"] = params["d"]
 
        if y_pred is None:
            out[f"{prefix}_{model}_slope"]  = np.nan
            for key in ("n", "rss", "r2", "adj_r2", "aic", "aicc", "bic"):
                out[f"{prefix}_{model}_{key}"] = np.nan
            continue
 
        out[f"{prefix}_{model}_slope"] = average_slope(x, y_pred)
        m = fit_quality_metrics(y, y_pred, k=k)
        for key in ("n", "rss", "r2", "adj_r2", "aic", "aicc", "bic"):
            out[f"{prefix}_{model}_{key}"] = m[key]
 
    # Best model per criterion (lowest value wins; inf excluded)
    # Also store delta_{crit} = {crit} - min({crit} across the 7 models) for
    # every model, so the "how much better/worse" margin is available
    # without a separate backfill step.
    for crit in ("aic", "aicc", "bic"):
        vals = {m: out[f"{prefix}_{m}_{crit}"] for m in MODELS}
        finite = {m: v for m, v in vals.items()
                  if v is not None and np.isfinite(v)}
        out[f"{prefix}_best_model_{crit}"] = (
            min(finite, key=finite.get) if finite else np.nan
        )
 
        if finite:
            best_val = min(finite.values())
            for m in MODELS:
                v = vals[m]
                out[f"{prefix}_{m}_delta_{crit}"] = (
                    v - best_val if (v is not None and np.isfinite(v)) else np.nan
                )
        else:
            for m in MODELS:
                out[f"{prefix}_{m}_delta_{crit}"] = np.nan
 
    return out
 
 
# ============================================================
# TIME KEY
# ============================================================
 
def detect_time_key(df):
    for name in TIME_CANDIDATES:
        if name in df.columns:
            return name
    if "date" in df.columns and "time" in df.columns:
        df["time_key"] = (df["date"].astype(str) + "_"
                          + df["time"].astype(str).str.zfill(4))
        return "time_key"
    raise ValueError("No usable time key found.")
 
 
# ============================================================
# METADATA TO CARRY THROUGH
# ============================================================
 
META_CANDIDATES = [
    "datetime", "station", "station_idx", "station_lat", "station_lon",
    "station_alt", "model_lat", "model_lon", "i_center", "j_center",
    "date", "time", "timestamp", "season", "day_night", "mode",
    "sector_type", "z_target_m", "center_ppb",
]
 
 
# ============================================================
# MAIN PROCESSING
# ============================================================
 
def process_station(in_path, out_path):
    print(f"Reading {in_path.name} ...")
    df = pd.read_csv(in_path)
    n_rows = len(df)
    print(f"  rows: {n_rows:,}")
 
    time_key = detect_time_key(df)
 
    # --- sector areas: depend only on the station's model latitude ---
    # one station => one model_lat => compute the area vector once.
    center_lat = float(df["model_lat"].iloc[0])
    area_km2 = cumulative_sector_areas_km2(center_lat)   # length-10, C1..C10
    print(f"  model_lat={center_lat:.4f}  "
          f"area C1={area_km2[0]:.1f} km^2  C10={area_km2[-1]:.1f} km^2")
 
    # sector value column names per variable
    sec_cols = {var: [f"{var}_C{s}" for s in SECTORS] for var in VARIABLES}
    for var in VARIABLES:
        missing = [c for c in sec_cols[var] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing sector columns for {var}: {missing}")
 
    meta_cols = [c for c in META_CANDIDATES if c in df.columns]
 
    # Pull sector value matrices once (vectorised access), shape (rows, 10)
    y_mats = {var: df[sec_cols[var]].to_numpy(dtype=float) for var in VARIABLES}
 
    x = area_km2   # same x for every timestep (area is time-invariant)
 
    rows_out = []
    t0 = time.time()
    for i in range(n_rows):
        row = {time_key: df[time_key].iloc[i]}
        for c in meta_cols:
            row[c] = df[c].iloc[i]
        # store the area vector once per row for reference / plotting
        for s_idx, s in enumerate(SECTORS):
            row[f"area_km2_C{s}"] = x[s_idx]
 
        for var in VARIABLES:
            y = y_mats[var][i]
            row.update(fit_all_models_for_variable(x, y, var))
 
        rows_out.append(row)
 
        if (i + 1) % 5000 == 0:
            el = time.time() - t0
            rate = (i + 1) / el
            eta = (n_rows - i - 1) / rate
            print(f"  {i+1:,}/{n_rows:,}  ({rate:.0f} rows/s, ETA {eta/60:.1f} min)")
 
    out = pd.DataFrame(rows_out)
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}  ({len(out):,} rows, {len(out.columns)} cols)")
    return out_path
 
 
def main():
    start = time.time()
    print("Start:", dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"))
 
    in_path  = IN_DIR / IN_FILENAME
    out_path = OUT_DIR / OUT_FILENAME
 
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")
 
    process_station(in_path, out_path)
 
    end = time.time()
    print("End:", dt.datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Execution time: {(end - start)/60:.2f} minutes")
 
 
if __name__ == "__main__":
    main()
# %%
 