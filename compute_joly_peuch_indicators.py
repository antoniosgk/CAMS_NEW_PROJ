"""
compute_joly_peuch_indicators.py
=================================
Compute the 8 Joly & Peuch (2012) temporal indicators from per-station
parquet files, following the methodology described in:

  Joly, M., Peuch, V.-H. (2012). Objective classification of air quality
  monitoring sites over Europe. Atmospheric Environment, 47, 111-123.

The 8 indicators, all derived from the hourly concentration time series:
  DIURNAL CYCLE (from 31-day sliding-window average):
    1. DC_MAX         — annual mean of the diurnal-cycle daily maximum
    2. DC_MAX_WS      — winter/summer ratio of DC_MAX
    3. DC_AMP         — annual mean of the diurnal-cycle amplitude (max - min)
    4. DC_AMP_WS      — winter/summer ratio of DC_AMP

  WEEKEND EFFECT (Sunday/Monday ratio, from Sun-Mon pairs with 48h data):
    5. WE_MEAN        — weekend effect on daily mean
    6. WE_MAX         — weekend effect on daily maximum
    7. WE_STD         — weekend effect on daily standard deviation

  HIGH-FREQUENCY VARIABILITY:
    8. HF_STD         — standard deviation of the high-frequency signal
                        (periods < 3 days, after FIR bandpass filtering)

Adaptations from the original paper:
  - J&P used observed hourly data from AirBase (2002-2009).
    We use GEOS-5 Nature Run concentrations (center_ppb) at the station
    grid cell — gap-free, globally consistent hourly fields.
  - The center_ppb value is the GEOS-5 NR concentration at the station's
    nearest grid point, representing the "true" concentration the station
    would measure in the model world.
  - Since GEOS-5 NR is every 30 min, we resample to hourly by averaging
    each hour's two 30-min steps.
  - We only extract C0 sector (center pixel) since the classification
    uses only the temporal fingerprint at the station location.

Input:
  Per-station parquet files: {station_id}_{SPECIES}_A_30min.parquet
  Each in long format (one row per timestep × sector), with columns:
    datetime, station, station_lat, station_lon, model_lat, model_lon,
    season, day_night, sector (C0..C10), center_ppb, mean_w, cv_w, ...

Output:
  joly_peuch_indicators_{SPECIES}.csv
  One row per station with: station_id, lat, lon, + 8 indicators

Dependencies:
  numpy, pandas, scipy (for FIR filter)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import time
import datetime as dt
from scipy.signal import firwin, filtfilt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# USER SETTINGS
# ============================================================
SPECIES = "SO2"
IN_DIR   = Path(f"/mnt/store01/agkiokas/CAMS/stations_parquet/{SPECIES}")
OUT_DIR  = Path(f"/mnt/store01/agkiokas/CAMS/fit_outputs/summary/{SPECIES}")
OUT_DIR.mkdir(parents=True, exist_ok=True)



# Minimum requirements (following J&P: discard if < 1 year of hourly data)
MIN_HOURS = 8760   # 1 full year of hourly data

# FIR filter parameters for high-frequency variability (Section 2.3)
# Cut-off frequency: 3 days = 72 hours  →  f_cutoff = 1/72 cycles/hour
# Kaiser window with 2-day transition width
FIR_NUMTAPS = 97   # ~ 2-day transition (48 hours × 2 + 1, must be odd)

# Minimum fraction of valid hours for a 31-day sliding window diurnal cycle
MIN_VALID_FRAC = 0.20

# Minimum Sunday-Monday pairs for weekend effect
MIN_SUN_MON_PAIRS = 20

# Season definitions (months)
SUMMER_MONTHS = [4, 5, 6, 7, 8, 9]    # April–September
WINTER_MONTHS = [10, 11, 12, 1, 2, 3]  # October–March


# ============================================================
# FIR FILTER DESIGN
# ============================================================

def design_hf_filter(numtaps=FIR_NUMTAPS):
    """
    Design a high-pass FIR filter to extract variability at periods < 3 days.
    Following J&P Section 2.3: FIR with cut-off at 3 days, Kaiser window
    with 2-day transition width.

    Sampling rate: 1 sample/hour → Nyquist = 0.5 cycles/hour
    Cut-off: 1/(3×24) = 1/72 cycles/hour ≈ 0.0139 cycles/hour
    """
    fs = 1.0       # 1 sample per hour
    f_cutoff = 1.0 / 72.0   # 3-day period → cycles/hour

    # Design a LOW-pass filter, then subtract to get HIGH-pass
    # (the high-frequency residual is s - c - l in J&P notation)
    lp_coeffs = firwin(numtaps, f_cutoff, fs=fs, window=("kaiser", 5.0))
    return lp_coeffs


# ============================================================
# COMPUTE THE 8 INDICATORS FOR ONE STATION
# ============================================================

def compute_indicators(df_hourly):
    """
    Given an hourly time series DataFrame with columns:
      datetime (DatetimeIndex), concentration (float)
    compute the 8 Joly & Peuch indicators.

    Returns dict with the 8 indicator values, or None if insufficient data.
    """
    ts = df_hourly.copy()
    ts = ts.sort_index()
    n_hours = ts["concentration"].notna().sum()

    if n_hours < MIN_HOURS:
        return None

    conc = ts["concentration"].values.astype(float)
    dt_idx = ts.index

    # ================================================================
    # PART A: DIURNAL CYCLE (Section 2.2)
    # ================================================================
    # For each day, compute the diurnal cycle as a 31-day sliding window
    # average of the hourly concentrations.
    #
    # For each day d, the sliding diurnal cycle c(h, d) for hour h is:
    #   c(h, d) = mean of conc at hour h over days [d-15 .. d+15]
    #
    # From each daily diurnal cycle we extract:
    #   daily_max = max_h c(h, d)
    #   daily_amp = max_h c(h, d) - min_h c(h, d)

    # Build a (n_days × 24) matrix of hourly concentrations
    dates = pd.Series(dt_idx.date).unique()
    date_to_idx = {d: i for i, d in enumerate(dates)}
    n_days = len(dates)

    hourly_matrix = np.full((n_days, 24), np.nan)
    for i, row in enumerate(ts.itertuples()):
        d = row.Index.date()
        h = row.Index.hour
        if d in date_to_idx:
            hourly_matrix[date_to_idx[d], h] = row.concentration

    # Sliding 31-day window diurnal cycle
    half_win = 15
    daily_max = np.full(n_days, np.nan)
    daily_amp = np.full(n_days, np.nan)
    daily_months = np.array([d.month for d in dates])

    for d_idx in range(n_days):
        lo = max(0, d_idx - half_win)
        hi = min(n_days, d_idx + half_win + 1)
        window = hourly_matrix[lo:hi, :]   # shape: (window_size, 24)

        # For each hour, compute the mean over the window days
        # J&P: require ≥ 20% valid hours for the hourly average
        valid_counts = np.sum(~np.isnan(window), axis=0)
        window_size = hi - lo
        diurnal = np.where(valid_counts >= window_size * MIN_VALID_FRAC,
                           np.nanmean(window, axis=0), np.nan)

        if np.sum(~np.isnan(diurnal)) >= 12:  # at least 12 hours with values
            daily_max[d_idx] = np.nanmax(diurnal)
            daily_amp[d_idx] = np.nanmax(diurnal) - np.nanmin(diurnal)

    # Monthly averages of daily_max and daily_amp
    # Then annual means and W/S ratios
    monthly_max = np.full(12, np.nan)
    monthly_amp = np.full(12, np.nan)

    for m in range(1, 13):
        mask = daily_months == m
        vals_max = daily_max[mask]
        vals_amp = daily_amp[mask]
        valid_max = vals_max[~np.isnan(vals_max)]
        valid_amp = vals_amp[~np.isnan(vals_amp)]
        if len(valid_max) >= 20:  # J&P: ≥ 20 days for monthly average
            monthly_max[m - 1] = np.mean(valid_max)
        if len(valid_amp) >= 20:
            monthly_amp[m - 1] = np.mean(valid_amp)

    # Check that all 12 months have values (J&P: discard if not)
    if np.any(np.isnan(monthly_max)) or np.any(np.isnan(monthly_amp)):
        # Relax: require at least 10 out of 12 months
        if np.sum(~np.isnan(monthly_max)) < 10:
            return None

    # Indicator 1: DC_MAX — annual mean of diurnal-cycle maximum
    dc_max = np.nanmean(monthly_max)

    # Indicator 3: DC_AMP — annual mean of diurnal-cycle amplitude
    dc_amp = np.nanmean(monthly_amp)

    # Indicators 2 & 4: Winter/Summer ratios
    summer_mask = np.array([(m + 1) in SUMMER_MONTHS for m in range(12)])
    winter_mask = np.array([(m + 1) in WINTER_MONTHS for m in range(12)])

    summer_max = np.nanmean(monthly_max[summer_mask])
    winter_max = np.nanmean(monthly_max[winter_mask])
    dc_max_ws = winter_max / summer_max if summer_max != 0 else np.nan

    summer_amp = np.nanmean(monthly_amp[summer_mask])
    winter_amp = np.nanmean(monthly_amp[winter_mask])
    dc_amp_ws = winter_amp / summer_amp if summer_amp != 0 else np.nan

    # ================================================================
    # PART B: WEEKEND EFFECT (Section 2.4)
    # ================================================================
    # Find all Sunday-Monday pairs where both days have 24 valid hours
    # (or at least 48 total hourly values across the pair).

    # Compute daily stats: mean, max, std for each day
    daily_mean_arr = np.full(n_days, np.nan)
    daily_max_raw  = np.full(n_days, np.nan)
    daily_std_arr  = np.full(n_days, np.nan)
    daily_dow      = np.array([d.weekday() for d in dates])  # 0=Mon, 6=Sun

    for d_idx in range(n_days):
        day_vals = hourly_matrix[d_idx, :]
        valid = day_vals[~np.isnan(day_vals)]
        if len(valid) >= 18:  # at least 18 out of 24 hours
            daily_mean_arr[d_idx] = np.mean(valid)
            daily_max_raw[d_idx]  = np.max(valid)
            daily_std_arr[d_idx]  = np.std(valid, ddof=1) if len(valid) > 1 else 0.0

    # Find Sunday-Monday pairs (Sunday = dow 6, next day Monday = dow 0)
    sun_means, mon_means = [], []
    sun_maxs,  mon_maxs  = [], []
    sun_stds,  mon_stds  = [], []

    for d_idx in range(n_days - 1):
        if daily_dow[d_idx] == 6 and daily_dow[d_idx + 1] == 0:
            # Sunday-Monday pair
            if (not np.isnan(daily_mean_arr[d_idx]) and
                not np.isnan(daily_mean_arr[d_idx + 1])):
                sun_means.append(daily_mean_arr[d_idx])
                mon_means.append(daily_mean_arr[d_idx + 1])
                sun_maxs.append(daily_max_raw[d_idx])
                mon_maxs.append(daily_max_raw[d_idx + 1])
                sun_stds.append(daily_std_arr[d_idx])
                mon_stds.append(daily_std_arr[d_idx + 1])

    if len(sun_means) >= MIN_SUN_MON_PAIRS:
        # Indicator 5: WE_MEAN — ratio of avg Sunday values / avg Monday values
        we_mean = np.mean(sun_means) / np.mean(mon_means) if np.mean(mon_means) != 0 else np.nan
        # Indicator 6: WE_MAX
        we_max  = np.mean(sun_maxs) / np.mean(mon_maxs) if np.mean(mon_maxs) != 0 else np.nan
        # Indicator 7: WE_STD
        we_std  = np.mean(sun_stds) / np.mean(mon_stds) if np.mean(mon_stds) != 0 else np.nan
    else:
        we_mean = np.nan
        we_max  = np.nan
        we_std  = np.nan

    # ================================================================
    # PART C: HIGH-FREQUENCY VARIABILITY (Section 2.3)
    # ================================================================
    # Apply FIR low-pass filter to get the low-frequency signal (l).
    # The diurnal cycle (c) has already been computed above.
    # High-frequency = s - c - l  (where s is the raw signal).
    # But the simpler approach (following J&P more directly):
    #   1. Subtract the sliding diurnal cycle from the raw series → s - c
    #   2. Apply low-pass FIR to (s - c) → this gives the low-freq part (l)
    #   3. High-frequency residual = (s - c) - l = s - c - l
    #   4. HF_STD = std of this residual
    #
    # We handle missing values by interpolating short gaps before filtering.

    # Build the diurnal cycle for every hour of the time series
    # (using the 31-day sliding window already computed)
    conc_series = ts["concentration"].copy()
    diurnal_series = pd.Series(np.nan, index=ts.index)

    for d_idx, date in enumerate(dates):
        lo = max(0, d_idx - half_win)
        hi = min(n_days, d_idx + half_win + 1)
        window = hourly_matrix[lo:hi, :]
        valid_counts = np.sum(~np.isnan(window), axis=0)
        window_size = hi - lo
        diurnal = np.where(valid_counts >= window_size * MIN_VALID_FRAC,
                           np.nanmean(window, axis=0), np.nan)
        for h in range(24):
            t = pd.Timestamp(date) + pd.Timedelta(hours=h)
            if t in diurnal_series.index and not np.isnan(diurnal[h]):
                diurnal_series.loc[t] = diurnal[h]

    # s - c (raw minus diurnal cycle)
    s_minus_c = conc_series - diurnal_series
    s_minus_c = s_minus_c.dropna()

    if len(s_minus_c) < FIR_NUMTAPS * 3:
        hf_std = np.nan
    else:
        # Interpolate short gaps (≤ 6 hours) for the filter
        s_mc_interp = s_minus_c.interpolate(method="linear", limit=6)
        s_mc_vals = s_mc_interp.values

        # Design and apply FIR low-pass filter
        lp_coeffs = design_hf_filter()
        try:
            low_freq = filtfilt(lp_coeffs, 1.0, s_mc_vals, padtype="even")
            hf_signal = s_mc_vals - low_freq
            # Indicator 8: HF_STD — std of the high-frequency residual
            hf_std = np.nanstd(hf_signal)
        except Exception:
            hf_std = np.nan

    # ================================================================
    # Collect all 8 indicators
    # ================================================================
    return {
        "DC_MAX":     dc_max,
        "DC_MAX_WS":  dc_max_ws,
        "DC_AMP":     dc_amp,
        "DC_AMP_WS":  dc_amp_ws,
        "WE_MEAN":    we_mean,
        "WE_MAX":     we_max,
        "WE_STD":     we_std,
        "HF_STD":     hf_std,
    }


# ============================================================
# PROCESS ONE STATION PARQUET FILE
# ============================================================

def process_station(pq_path, station_id):
    """
    Read a station parquet file, extract center_ppb time series,
    resample to hourly, compute the 8 J&P indicators.
    Returns a dict (one row) or None.
    """
    try:
        # Read only the columns we need (sector C0 = center pixel)
        # We use center_ppb which is the concentration at the station grid cell
        cols_needed = ["datetime", "station", "station_lat", "station_lon",
                       "model_lat", "model_lon", "sector", "center_ppb"]
        # Also try timestamp if datetime doesn't exist
        try:
            df = pd.read_parquet(pq_path, columns=cols_needed)
        except Exception:
            cols_alt = ["timestamp", "station", "station_lat", "station_lon",
                        "model_lat", "model_lon", "sector", "center_ppb"]
            df = pd.read_parquet(pq_path, columns=cols_alt)
            df["datetime"] = pd.to_datetime(df["timestamp"].astype(str),
                                            format="%Y%m%d %H%M",
                                            errors="coerce")

        if len(df) == 0:
            return None

        # Deduplicate to one row per timestep.
        # center_ppb is the same across all sectors for a given timestep,
        # so we just need one row per datetime. Try filtering to C0 first;
        # if no C0 sector exists, take C1; otherwise just drop duplicates.
        if "sector" in df.columns and "C0" in df["sector"].values:
            df = df[df["sector"] == "C0"].copy()
        elif "sector" in df.columns and "C1" in df["sector"].values:
            df = df[df["sector"] == "C1"].copy()
        else:
            # No sector column or unknown labels — deduplicate on datetime
            df = df.drop_duplicates(subset=["datetime"]).copy()

        if len(df) == 0:
            return None

        # Parse datetime
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])

        # Station metadata (from first row)
        meta = {
            "station_id": station_id,
            "station_lat": df["station_lat"].iloc[0] if "station_lat" in df.columns else np.nan,
            "station_lon": df["station_lon"].iloc[0] if "station_lon" in df.columns else np.nan,
            "model_lat": df["model_lat"].iloc[0] if "model_lat" in df.columns else np.nan,
            "model_lon": df["model_lon"].iloc[0] if "model_lon" in df.columns else np.nan,
        }

        # Build hourly time series of center_ppb
        # GEOS-5 NR is every 30 min → resample to hourly by averaging
        ts = df[["datetime", "center_ppb"]].copy()
        ts = ts.set_index("datetime").sort_index()
        ts = ts.rename(columns={"center_ppb": "concentration"})

        # Resample to hourly (averaging the two 30-min values per hour)
        ts_hourly = ts.resample("1h").mean()

        # Drop all-NaN hours
        ts_hourly = ts_hourly.dropna(subset=["concentration"])

        meta["n_hours"] = len(ts_hourly)

        if len(ts_hourly) < MIN_HOURS:
            return None

        # Compute the 8 indicators
        indicators = compute_indicators(ts_hourly)

        if indicators is None:
            return None

        # Merge metadata + indicators
        row = {**meta, **indicators}
        return row

    except Exception as e:
        print(f"  [ERROR] {station_id}: {e}")
        return None


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("compute_joly_peuch_indicators.py")
    print("=" * 60)
    print(f"Start: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"IN_DIR:  {IN_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"SPECIES: {SPECIES}")
    print()

    suffix = f"_{SPECIES}_A_30min.parquet"
    pq_files = sorted(IN_DIR.glob(f"*{suffix}"))
    print(f"Found {len(pq_files)} station parquet files")

    if not pq_files:
        print("No parquet files found. Check IN_DIR and SPECIES.")
        return

    rows = []
    n_skip = 0
    for i, pq_path in enumerate(pq_files):
        station_id = pq_path.name.replace(suffix, "")
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(pq_files)}]  {station_id}")

        result = process_station(pq_path, station_id)
        if result is not None:
            rows.append(result)
        else:
            n_skip += 1

    df_out = pd.DataFrame(rows)

    # Order columns
    meta_cols = ["station_id", "station_lat", "station_lon",
                 "model_lat", "model_lon", "n_hours"]
    indicator_cols = ["DC_MAX", "DC_MAX_WS", "DC_AMP", "DC_AMP_WS",
                      "WE_MEAN", "WE_MAX", "WE_STD", "HF_STD"]
    all_cols = [c for c in meta_cols if c in df_out.columns] + \
               [c for c in indicator_cols if c in df_out.columns]
    # Add any remaining columns
    remaining = [c for c in df_out.columns if c not in all_cols]
    df_out = df_out[all_cols + remaining]

    out_path = OUT_DIR / f"joly_peuch_indicators_{SPECIES}.csv"
    df_out.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print()
    print(f"Processed: {len(rows)} stations  |  Skipped: {n_skip}")
    print(f"Saved:     {out_path}")
    print(f"Columns:   {list(df_out.columns)}")
    print(f"Done in    {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()