"""
aggregate_slope_stats.py
========================
For every station CSV in IN_DIR, compute the analytic derivative (dy/dx)
of each of the 7 fitted models at the 0.4° box area for both ratio and cv_w.

Statistics computed per station:
  - mean, median, std of the derivative
  - grouped by: whole period, 4 seasons (DJF, MAM, JJA, SON),
                day/night

Output: one CSV with one row per station, columns:
  station_id, station_lat, station_lon, station_alt, area_km2_box04,
  {var}_{model}_{group}_{stat}
  e.g. ratio_linear_all_mean, cv_w_power_JJA_median, ...
"""

from pathlib import Path
import numpy as np
import pandas as pd
import time
import datetime as dt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# USER SETTINGS
# ============================================================

IN_DIR   = Path("/mnt/store01/agkiokas/CAMS/fit_outputs")
OUT_DIR  = Path("/mnt/store01/agkiokas/CAMS/fit_outputs/summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES    = "O3"
CSV_SUFFIX = f"_{SPECIES}_area_fits_new.csv"

# Which variables and models to process
VARIABLES = ["ratio", "cv_w"]
MODELS    = ["linear", "quadratic", "cubic", "logarithmic",
             "exponential", "power", "saturating"]

# Minimum number of sectors for a fit to be considered valid
MIN_N_SECTORS = 8

# Box size for the derivative evaluation point
BOX_DEG = 0.4

# Season mapping (month -> season abbreviation)
SEASON_MAP = {12: "DJF", 1: "DJF", 2: "DJF",
               3: "MAM", 4: "MAM", 5: "MAM",
               6: "JJA", 7: "JJA", 8: "JJA",
               9: "SON", 10: "SON", 11: "SON"}
SEASONS = ["DJF", "MAM", "JJA", "SON"]

# Groupings to aggregate over
# Each entry: (group_label, filter_function)
# The filter function receives the dataframe and returns a boolean mask
GROUPS = (
    [("all", lambda df: pd.Series(True, index=df.index))] +
    [(s, lambda df, s=s: df["_season"] == s) for s in SEASONS] +
    [("day",   lambda df: df["_daynight"].str.lower() == "day"),
     ("night", lambda df: df["_daynight"].str.lower() == "night")]
)


# ============================================================
# ANALYTIC DERIVATIVE  (same formulas as plot_area_fit_results.py)
# ============================================================

def clean(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def analytic_derivative(df, variable, model, x_star):
    """dy/dx at x = x_star (km²), vectorised over all rows."""
    a = clean(df.get(f"{variable}_{model}_a"))
    b = clean(df.get(f"{variable}_{model}_b"))
    c = clean(df.get(f"{variable}_{model}_c"))
    d = clean(df.get(f"{variable}_{model}_d"))

    if model == "linear":
        return b
    if model == "quadratic":
        return b + 2.0 * c * x_star
    if model == "cubic":
        return b + 2.0 * c * x_star + 3.0 * d * x_star ** 2
    if model == "logarithmic":
        return b / x_star
    if model == "exponential":
        return b * c * np.exp(np.clip(c * x_star, -50, 50))
    if model == "power":
        return a * b * np.power(x_star, b - 1.0)
    if model == "saturating":
        lam = c
        return (b / lam) * np.exp(-np.clip(x_star / lam, -50, 50))
    raise ValueError(f"Unknown model: {model}")


def filter_by_n(df, variable, model):
    n_col = f"{variable}_{model}_n"
    if n_col in df.columns:
        return df[n_col] >= MIN_N_SECTORS
    return pd.Series(True, index=df.index)


# ============================================================
# DETECT DATETIME
# ============================================================

def detect_datetime(df):
    if "datetime" in df.columns:
        df["_dt"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "timestamp" in df.columns:
        df["_dt"] = pd.to_datetime(df["timestamp"].astype(str),
                                   format="%Y%m%d %H%M", errors="coerce")
        if df["_dt"].isna().all():
            df["_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "time_key" in df.columns:
        df["_dt"] = pd.to_datetime(df["time_key"].astype(str),
                                   format="%Y%m%d_%H%M", errors="coerce")
    else:
        df["_dt"] = pd.NaT
    return df


# ============================================================
# PROCESS ONE STATION
# ============================================================

def process_station(csv_path):
    """
    Returns a dict with one row of statistics for this station,
    or None if the file is unusable.
    """
    station_id = csv_path.stem.replace(CSV_SUFFIX.replace(".csv", ""), "")
    # more robust: take everything before the first underscore pattern
    station_id = csv_path.stem.split(f"_{SPECIES}")[0]

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"  [ERROR] {csv_path.name}: {e}")
        return None

    if len(df) == 0:
        return None

    df = detect_datetime(df)

    # --- resolve x_star (0.4° box area) ---
    if "area_km2_box04" in df.columns:
        x_star = float(df["area_km2_box04"].iloc[0])
    else:
        # recompute from model_lat
        if "model_lat" not in df.columns:
            print(f"  [SKIP] {station_id}: no area_km2_box04 or model_lat")
            return None
        lat = float(df["model_lat"].iloc[0])
        phi_s = np.radians(lat - BOX_DEG / 2.0)
        phi_n = np.radians(lat + BOX_DEG / 2.0)
        R = 6371.0088
        x_star = R**2 * (np.sin(phi_n) - np.sin(phi_s)) * np.radians(BOX_DEG)

    # --- build season / day-night columns ---
    if "season" in df.columns:
        df["_season"] = df["season"].astype(str).str.upper().str.strip()
    elif df["_dt"].notna().any():
        df["_season"] = df["_dt"].dt.month.map(SEASON_MAP)
    else:
        df["_season"] = np.nan

    if "day_night" in df.columns:
        df["_daynight"] = df["day_night"].astype(str).str.strip()
    else:
        df["_daynight"] = "unknown"

    # --- station metadata ---
    row = {"station_id": station_id}
    for col in ["station_lat", "station_lon", "station_alt",
                "model_lat", "model_lon"]:
        if col in df.columns:
            row[col] = df[col].iloc[0]
    row["area_km2_box04"] = x_star
    row["n_timesteps"] = len(df)

    # --- compute derivatives for all models × variables ---
    derivs = {}
    for var in VARIABLES:
        for model in MODELS:
            mask = filter_by_n(df, var, model)
            vals = analytic_derivative(df, var, model, x_star)
            vals = vals.where(mask, np.nan)
            derivs[(var, model)] = vals

    # --- aggregate by group ---
    for group_label, group_filter in GROUPS:
        try:
            gmask = group_filter(df)
        except Exception:
            gmask = pd.Series(False, index=df.index)

        for var in VARIABLES:
            for model in MODELS:
                vals = derivs[(var, model)]
                sub = vals[gmask].dropna()
                prefix = f"{var}_{model}_{group_label}"
                row[f"{prefix}_mean"]   = float(sub.mean())   if len(sub) > 0 else np.nan
                row[f"{prefix}_median"] = float(sub.median()) if len(sub) > 0 else np.nan
                row[f"{prefix}_std"]    = float(sub.std())    if len(sub) > 0 else np.nan
                row[f"{prefix}_count"]  = int(len(sub))

    return row


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("aggregate_slope_stats.py")
    print("=" * 60)
    print(f"Start: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"IN_DIR:  {IN_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")

    csv_files = sorted(IN_DIR.glob(f"*{CSV_SUFFIX}"))
    print(f"Found {len(csv_files)} station CSV files")

    if not csv_files:
        print("No CSV files found. Check IN_DIR and CSV_SUFFIX.")
        return

    rows = []
    for i, csv_path in enumerate(csv_files):
        station_id = csv_path.stem.split(f"_{SPECIES}")[0]
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(csv_files)}]  {station_id}")
        result = process_station(csv_path)
        if result is not None:
            rows.append(result)

    df_out = pd.DataFrame(rows)

    # Sort columns for readability
    meta_cols = ["station_id", "station_lat", "station_lon", "station_alt",
                 "model_lat", "model_lon", "area_km2_box04", "n_timesteps"]
    meta_cols = [c for c in meta_cols if c in df_out.columns]
    stat_cols = sorted([c for c in df_out.columns if c not in meta_cols])
    df_out = df_out[meta_cols + stat_cols]

    out_path = OUT_DIR / "slope_stats_box04.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  {len(df_out)} stations × {len(df_out.columns)} columns")

    elapsed = time.time() - t0
    print(f"Done in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()