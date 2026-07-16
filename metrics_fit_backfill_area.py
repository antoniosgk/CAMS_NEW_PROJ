"""
Area-based multi-model fit for station(s).

Supports three selection modes:
    SELECTION_MODE = "single"     -> one station via STATION_ID
    SELECTION_MODE = "name_list"  -> multiple stations via STATION_IDS list
    SELECTION_MODE = "all"        -> all *_sector_ratio_cv_fits_with_aic.csv in IN_DIR

Output: one CSV per station in OUT_DIR:  {station}_{SPECIES}_area_fits.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import time
import datetime as dt
from scipy.optimize import least_squares

# ============================================================
# SETTINGS
# ============================================================

IN_DIR  = Path("/mnt/store01/agkiokas/CAMS/derived_station_metrics/")
OUT_DIR = Path("/mnt/store01/agkiokas/CAMS/fit_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = "O3"

# ------------------------------------------------------------
# Station selection
# ------------------------------------------------------------
#   "single"     -> use STATION_ID
#   "name_list"  -> use STATION_IDS
#   "all"        -> process every *_sector_ratio_cv_fits_with_aic.csv in IN_DIR
#SELECTION_MODE = "single"       # one station via STATION_ID
#SELECTION_MODE = "name_list"    # explicit names via STATION_IDS
SELECTION_MODE = "idx_range"    # numeric index range via IDX_MIN / IDX_MAX
#SELECTION_MODE = "all"          # every *_with_aic.csv in IN_DIR

IDX_MIN = 501      # inclusive, used when SELECTION_MODE == "idx_range"
IDX_MAX = 600    # inclusive

STATION_ID  = "1003A"
STATION_IDS = ["1001A", "1002A", "1003A", "1004A", "1006A"]

# If True, skip stations whose output file already exists
SKIP_EXISTING = True

# Variables to fit
VARIABLES = ["ratio", "cv_w", "mean_w"]

# Grid resolution (degrees)
GRID_DEG = 0.0625

# Hypothetical coarse-model grid box (deg). Its spherical area (km^2) is
# stored per row as area_km2_box04, used by the plotting script as a
# physically meaningful reference scale for slope evaluation.
BOX_DEG = 0.4

# Earth radius (km)
EARTH_RADIUS_KM = 6371.0088

# Sectors fitted: C1..C10
SECTORS = list(range(1, 11))

# Minimum valid points required to attempt a fit
MIN_POINTS = {
    "linear": 2, "quadratic": 3, "cubic": 4,
    "logarithmic": 2, "exponential": 3, "power": 2, "saturating": 3,
}

# Free-parameter count k per model
MODEL_K = {
    "linear": 2, "quadratic": 3, "cubic": 4,
    "logarithmic": 2, "exponential": 3, "power": 2, "saturating": 3,
}

MODELS = ["linear", "quadratic", "cubic", "logarithmic",
          "exponential", "power", "saturating"]

TIME_CANDIDATES = ["datetime", "timestamp", "time_key"]


# ============================================================
# STATION FILE DISCOVERY
# ============================================================

def get_station_files():
    """
    Return list of (station_id, input_path, output_path) tuples to process.
    """
    suffix_in  = f"_{SPECIES}_sector_ratio_cv_fits_with_aic.csv"
    suffix_out = f"_{SPECIES}_area_fits.csv"

    if SELECTION_MODE == "single":
        ids = [STATION_ID]
    elif SELECTION_MODE == "name_list":
        ids = list(STATION_IDS)
    elif SELECTION_MODE == "idx_range":
        # Read only the station_idx column from each candidate file to check
        # whether its index falls within [IDX_MIN, IDX_MAX].
        all_files = sorted(IN_DIR.glob(f"*{suffix_in}"))
        ids = []
        for f in all_files:
            try:
                tmp = pd.read_csv(f, usecols=["station_idx"], nrows=1)
                if tmp.empty:
                    continue
                idx_val = int(tmp["station_idx"].iloc[0])
                if IDX_MIN <= idx_val <= IDX_MAX:
                    ids.append(f.name.replace(suffix_in, ""))
            except Exception as e:
                print(f"[SKIP] Could not read station_idx from {f.name}: {e}")    
    elif SELECTION_MODE == "all":
        files = sorted(IN_DIR.glob(f"*{suffix_in}"))
        ids = [f.name.replace(suffix_in, "") for f in files]
    else:
        raise ValueError(f"Invalid SELECTION_MODE: {SELECTION_MODE!r}. "
                         f"Use 'single', 'name_list', or 'all'.")

    result = []
    for sid in ids:
        in_path  = IN_DIR / f"{sid}{suffix_in}"
        out_path = OUT_DIR / f"{sid}{suffix_out}"

        if not in_path.exists():
            print(f"[SKIP] Input not found for {sid}: {in_path}")
            continue

        if SKIP_EXISTING and out_path.exists():
            print(f"[SKIP] Output already exists for {sid}: {out_path}")
            continue

        result.append((sid, in_path, out_path))

    print(f"Selection mode: {SELECTION_MODE}")
    print(f"Stations to process: {len(result)}")
    return result


# ============================================================
# SECTOR AREA (TRUE SPHERICAL, PER STATION)
# ============================================================

def cell_area_km2(lat_south_deg, lat_north_deg, dlon_deg):
    phi_s = np.radians(lat_south_deg)
    phi_n = np.radians(lat_north_deg)
    dlon  = np.radians(dlon_deg)
    return (EARTH_RADIUS_KM ** 2) * (np.sin(phi_n) - np.sin(phi_s)) * dlon


def cumulative_sector_areas_km2(center_lat_deg):
    half_deg = GRID_DEG / 2.0

    def row_area_sum(k):
        total = 0.0
        n_side = 2 * k + 1
        for r in range(-k, k + 1):
            row_center = center_lat_deg + r * GRID_DEG
            lat_s = row_center - half_deg
            lat_n = row_center + half_deg
            a_cell = cell_area_km2(lat_s, lat_n, GRID_DEG)
            total += a_cell * n_side
        return total

    return np.array([row_area_sum(k) for k in SECTORS], dtype=float)


# ============================================================
# FIT QUALITY METRICS
# ============================================================

def fit_quality_metrics(y_true, y_pred, k):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid  = np.isfinite(y_true) & np.isfinite(y_pred)
    n      = int(valid.sum())

    out = {"n": n, "rss": np.nan, "r2": np.nan, "adj_r2": np.nan,
           "aic": np.nan, "aicc": np.nan, "bic": np.nan}
    if n < 2:
        return out

    yt, yp = y_true[valid], y_pred[valid]
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
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    i_lo, i_hi = int(np.argmin(x)), int(np.argmax(x))
    dx = x[i_hi] - x[i_lo]
    return (y_pred[i_hi] - y_pred[i_lo]) / dx if dx != 0 else np.nan


# ============================================================
# MODELS
# ============================================================

def _nan_params():
    return {"a": np.nan, "b": np.nan, "c": np.nan, "d": np.nan}


def _lstsq_fit(x, y, design_fn, param_names):
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
    return _lstsq_fit(x, y, lambda xv: np.column_stack([np.ones_like(xv), xv]),
                      ["a", "b"])

def fit_quadratic(x, y):
    return _lstsq_fit(x, y,
                      lambda xv: np.column_stack([np.ones_like(xv), xv, xv**2]),
                      ["a", "b", "c"])

def fit_cubic(x, y):
    return _lstsq_fit(x, y,
                      lambda xv: np.column_stack([np.ones_like(xv), xv, xv**2, xv**3]),
                      ["a", "b", "c", "d"])

def fit_logarithmic(x, y):
    return _lstsq_fit(x, y,
                      lambda xv: np.column_stack([np.ones_like(xv), np.log(xv)]),
                      ["a", "b"])


def fit_exponential(x, y):
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
        r = least_squares(resid, [ym, yv[0] - ym, 0.0], jac=jac,
                          bounds=([np.nanmin(yv)-10*yr, -10*yr, -50.0],
                                  [np.nanmax(yv)+10*yr,  10*yr,  50.0]),
                          max_nfev=500, method="trf")
        a, b, c_scaled = r.x
        c = c_scaled / xmax
        params = _nan_params(); params.update(a=a, b=b, c=c)
        y_pred = np.full_like(x, np.nan, dtype=float)
        y_pred[valid] = a + b * np.exp(np.clip(c * xv, -50, 50))
        return params, y_pred
    except Exception:
        return _nan_params(), None


def fit_power(x, y):
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

        r = least_squares(resid, [a0, b0],
                          bounds=([-np.inf, -10.0], [np.inf, 10.0]),
                          max_nfev=500, method="trf")
        a, b = r.x
        y_pred = np.full_like(x, np.nan, dtype=float)
        y_pred[valid] = a * np.power(xv, b)
        if not np.all(np.isfinite(y_pred[valid])):
            return _nan_params(), None
        ss_res = float(np.sum((yv - y_pred[valid]) ** 2))
        ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
        if ss_tot > 0 and (1.0 - ss_res / ss_tot) < -1.0:
            return _nan_params(), None
        params = _nan_params(); params.update(a=a, b=b)
        return params, y_pred
    except Exception:
        return _nan_params(), None


def fit_saturating(x, y):
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
        r = least_squares(resid, [y0, ye - y0, 0.33],
                          bounds=([-np.inf, -np.inf, 1e-6],
                                  [np.inf,  np.inf,  np.inf]),
                          max_nfev=500, method="trf")
        a, b, lam_scaled = r.x
        lam = lam_scaled * xmax
        params = _nan_params(); params.update(a=a, b=b, c=lam)
        y_pred = np.full_like(x, np.nan, dtype=float)
        y_pred[valid] = a + b * (1.0 - np.exp(-np.clip(xv / lam, -50, 50)))
        return params, y_pred
    except Exception:
        return _nan_params(), None


FIT_FUNCS = {
    "linear": fit_linear, "quadratic": fit_quadratic, "cubic": fit_cubic,
    "logarithmic": fit_logarithmic, "exponential": fit_exponential,
    "power": fit_power, "saturating": fit_saturating,
}


def fit_all_models_for_variable(x, y, prefix):
    out = {}
    for model in MODELS:
        params, y_pred = FIT_FUNCS[model](x, y)
        k = MODEL_K[model]

        out[f"{prefix}_{model}_a"] = params["a"]
        out[f"{prefix}_{model}_b"] = params["b"]
        out[f"{prefix}_{model}_c"] = params["c"]
        out[f"{prefix}_{model}_d"] = params["d"]

        if y_pred is None:
            out[f"{prefix}_{model}_slope"] = np.nan
            for key in ("n", "rss", "r2", "adj_r2", "aic", "aicc", "bic"):
                out[f"{prefix}_{model}_{key}"] = np.nan
            continue

        out[f"{prefix}_{model}_slope"] = average_slope(x, y_pred)
        m = fit_quality_metrics(y, y_pred, k=k)
        for key in ("n", "rss", "r2", "adj_r2", "aic", "aicc", "bic"):
            out[f"{prefix}_{model}_{key}"] = m[key]

    for crit in ("aic", "aicc", "bic"):
        vals = {m: out[f"{prefix}_{m}_{crit}"] for m in MODELS}
        finite = {m: v for m, v in vals.items()
                  if v is not None and np.isfinite(v)}
        out[f"{prefix}_best_model_{crit}"] = (
            min(finite, key=finite.get) if finite else np.nan)
        if finite:
            best_val = min(finite.values())
            for m in MODELS:
                v = vals[m]
                out[f"{prefix}_{m}_delta_{crit}"] = (
                    v - best_val if (v is not None and np.isfinite(v)) else np.nan)
        else:
            for m in MODELS:
                out[f"{prefix}_{m}_delta_{crit}"] = np.nan

    return out


# ============================================================
# TIME KEY / METADATA
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


META_CANDIDATES = [
    "datetime", "station", "station_idx", "station_lat", "station_lon",
    "station_alt", "model_lat", "model_lon", "i_center", "j_center",
    "date", "time", "timestamp", "season", "day_night", "mode",
    "sector_type", "k_star_cen", "z_target_m", "center_ppb",
]


# ============================================================
# PROCESS ONE STATION
# ============================================================

def process_station(station_id, in_path, out_path):
    print(f"\n{'='*70}")
    print(f"Station: {station_id}")
    print(f"  Input:  {in_path}")
    print(f"  Output: {out_path}")

    df = pd.read_csv(in_path)
    n_rows = len(df)
    print(f"  rows: {n_rows:,}")

    time_key = detect_time_key(df)

    center_lat = float(df["model_lat"].iloc[0])
    area_km2 = cumulative_sector_areas_km2(center_lat)
    print(f"  model_lat={center_lat:.4f}  "
          f"area C1={area_km2[0]:.1f} km^2  C10={area_km2[-1]:.1f} km^2")

    box_area_km2 = cell_area_km2(center_lat - BOX_DEG / 2.0,
                                 center_lat + BOX_DEG / 2.0, BOX_DEG)
    print(f"  {BOX_DEG} deg box area: {box_area_km2:.1f} km^2")

    sec_cols = {var: [f"{var}_C{s}" for s in SECTORS] for var in VARIABLES}
    for var in VARIABLES:
        missing = [c for c in sec_cols[var] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing sector columns for {var}: {missing}")

    meta_cols = [c for c in META_CANDIDATES if c in df.columns]
    y_mats = {var: df[sec_cols[var]].to_numpy(dtype=float) for var in VARIABLES}
    x = area_km2

    rows_out = []
    t0 = time.time()
    for i in range(n_rows):
        row = {time_key: df[time_key].iloc[i]}
        for c in meta_cols:
            row[c] = df[c].iloc[i]

        for s_idx, s in enumerate(SECTORS):
            row[f"area_km2_C{s}"] = x[s_idx]
        row["area_km2_box04"] = box_area_km2

        for var in VARIABLES:
            y = y_mats[var][i]
            for s_idx, s_num in enumerate(SECTORS):
                row[f"{var}_C{s_num}"] = y[s_idx]
            row.update(fit_all_models_for_variable(x, y, var))

        rows_out.append(row)

        if (i + 1) % 5000 == 0:
            el = time.time() - t0
            rate = (i + 1) / el
            eta = (n_rows - i - 1) / rate
            print(f"  {i+1:,}/{n_rows:,}  ({rate:.0f} rows/s, ETA {eta/60:.1f} min)")

    out = pd.DataFrame(rows_out)
    out.to_csv(out_path, index=False)
    el = time.time() - t0
    print(f"  Saved: {len(out):,} rows, {len(out.columns)} cols, {el/60:.1f} min")
    return out_path


# ============================================================
# MAIN
# ============================================================

def main():
    start = time.time()
    print("Start:", dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"))

    station_files = get_station_files()

    created = []
    for idx, (sid, in_path, out_path) in enumerate(station_files, 1):
        print(f"\n[{idx}/{len(station_files)}]")
        try:
            result = process_station(sid, in_path, out_path)
            created.append(result)
        except Exception as e:
            print(f"  [ERROR] {sid}: {e}")

    end = time.time()
    print(f"\n{'='*70}")
    print(f"Completed: {len(created)}/{len(station_files)} stations")
    print("End:", dt.datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Total time: {(end - start)/60:.2f} minutes")


if __name__ == "__main__":
    main()
# %%