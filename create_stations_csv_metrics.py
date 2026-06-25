#%%
from pathlib import Path
import numpy as np
import pandas as pd
import datetime as dt
import time
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# ============================================================
# STATION SELECTION SETTINGS
# ============================================================

PARQUET_DIR = Path("/mnt/store01/agkiokas/CAMS/stations_parquet/")
PARQUET_GLOB = "*_O3_A_30min.parquet"
OUT_DIR = Path("/mnt/store01/agkiokas/CAMS/derived_station_metrics/")
OUT_DIR.mkdir(parents=True, exist_ok=True)
species="O3"
# Selection mode:
#   "name_list"  -> use STATIONS list, e.g. ["1001A", "1002A"]
#   "idx_range"  -> use IDX_MIN and IDX_MAX
#   "all"        -> process all parquet files
SELECTION_MODE = "idx_range"

# Used only if SELECTION_MODE = "name_list"
STATIONS = [
    "1001A",
    "1002A",
    "1003A",
]

# Used only if SELECTION_MODE = "idx_range"
IDX_COL = "station_idx"
IDX_MIN = 1451
IDX_MAX = 1500   # inclusive

# If True, existing CSV files will be overwritten
OVERWRITE = True

# Keep only cumulative sectors
SECTOR_TYPE_KEEP = "CUM"

# Expected parquet columns
CENTER_COL = "center_ppb"
MEAN_COL = "mean_w"
STD_COL = "std_w"
CV_COL = "cv_w"

# Optional median column
# The script will use it only if it exists in the parquet file.
MEDIAN_COL = "median_w"

SECTOR_COL = "sector"
RADIUS_COL = "radius"

# Use "radius" if available; otherwise sector number C1=1, ..., C10=10
FIT_X_COL = "radius"

REQUIRED_SECTORS = set(range(1, 11))


# ============================================================
# FITTING FUNCTIONS
# ============================================================

def sector_to_number(sector_value):
    """
    Convert sector labels such as C1, C2, ..., C10 to numbers 1, 2, ..., 10.
    """
    return int(str(sector_value).replace("C", ""))


def safe_r2(y_true, y_pred):
    """
    Compute R² safely.
    Returns NaN if the calculation is not meaningful.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    if valid.sum() < 2:
        return np.nan

    if np.nanstd(y_true[valid]) == 0:
        return np.nan

    return r2_score(y_true[valid], y_pred[valid])


def fit_linear(x, y):
    """
    Fit y = a + b*x.
    Returns a, b, r2.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 2:
        return {"a": np.nan, "b": np.nan, "r2": np.nan}

    coeffs = np.polyfit(x[valid], y[valid], deg=1)

    b = coeffs[0]
    a = coeffs[1]

    y_pred = a + b * x[valid]

    return {
        "a": a,
        "b": b,
        "r2": safe_r2(y[valid], y_pred),
    }


def fit_quadratic(x, y):
    """
    Fit y = a + b*x + c*x².
    Returns a, b, c, r2.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 3:
        return {"a": np.nan, "b": np.nan, "c": np.nan, "r2": np.nan}

    coeffs = np.polyfit(x[valid], y[valid], deg=2)

    c = coeffs[0]
    b = coeffs[1]
    a = coeffs[2]

    y_pred = a + b * x[valid] + c * x[valid] ** 2

    return {
        "a": a,
        "b": b,
        "c": c,
        "r2": safe_r2(y[valid], y_pred),
    }


def exponential_model(x, a, b, c):
    """
    Safer exponential model:

        y = a + b * exp(c*x)

    np.clip prevents overflow during curve fitting.
    """
    z = np.clip(c * x, -50, 50)
    return a + b * np.exp(z)


def fit_exponential(x, y):
    """
    Fit y = a + b*exp(c*x).

    Safer version:
    - clips the exponential internally,
    - bounds c to avoid extreme growth/decay,
    - returns NaN if the fit is unstable.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 3:
        return {"a": np.nan, "b": np.nan, "c": np.nan, "r2": np.nan}

    x_fit = x[valid]
    y_fit = y[valid]

    # If y is almost constant, exponential fitting is not meaningful
    if np.nanstd(y_fit) < 1e-12:
        return {"a": np.nan, "b": np.nan, "c": np.nan, "r2": np.nan}

    try:
        y_mean = np.nanmean(y_fit)
        y_range = np.nanmax(y_fit) - np.nanmin(y_fit)

        if not np.isfinite(y_range) or y_range == 0:
            return {"a": np.nan, "b": np.nan, "c": np.nan, "r2": np.nan}

        # Initial guesses
        a0 = y_mean
        b0 = y_fit[0] - y_mean
        c0 = 0.0

        # Bounds to prevent extreme exponentials
        lower_bounds = [
            np.nanmin(y_fit) - 10 * y_range,   # a lower
            -10 * y_range,                     # b lower
            -1.0,                              # c lower
        ]

        upper_bounds = [
            np.nanmax(y_fit) + 10 * y_range,   # a upper
            10 * y_range,                      # b upper
            1.0,                               # c upper
        ]

        popt, _ = curve_fit(
            exponential_model,
            x_fit,
            y_fit,
            p0=[a0, b0, c0],
            bounds=(lower_bounds, upper_bounds),
            maxfev=3000,
        )

        a, b, c = popt

        y_pred = exponential_model(x_fit, a, b, c)

        return {
            "a": a,
            "b": b,
            "c": c,
            "r2": safe_r2(y_fit, y_pred),
        }

    except Exception:
        return {"a": np.nan, "b": np.nan, "c": np.nan, "r2": np.nan}


def fit_all_models_for_variable(x, y, prefix):
    """
    Fit linear, quadratic and exponential models for one variable.
    """
    out = {}

    lin = fit_linear(x, y)
    out[f"{prefix}_linear_a"] = lin["a"]
    out[f"{prefix}_linear_b"] = lin["b"]
    out[f"{prefix}_linear_r2"] = lin["r2"]

    quad = fit_quadratic(x, y)
    out[f"{prefix}_quadratic_a"] = quad["a"]
    out[f"{prefix}_quadratic_b"] = quad["b"]
    out[f"{prefix}_quadratic_c"] = quad["c"]
    out[f"{prefix}_quadratic_r2"] = quad["r2"]

    exp = fit_exponential(x, y)
    out[f"{prefix}_exponential_a"] = exp["a"]
    out[f"{prefix}_exponential_b"] = exp["b"]
    out[f"{prefix}_exponential_c"] = exp["c"]
    out[f"{prefix}_exponential_r2"] = exp["r2"]

    return out


# ============================================================
# DATA PREPARATION FUNCTIONS
# ============================================================

def get_station_files():
    """
    Return a list of parquet files to process.

    Selection options:
        SELECTION_MODE = "name_list"
        SELECTION_MODE = "idx_range"
        SELECTION_MODE = "all"

    Only files matching PARQUET_GLOB are considered.
    This excludes summary files such as:
        1006A_O3_summary.parquet
    """

    all_files = sorted(PARQUET_DIR.glob(PARQUET_GLOB))

    selected_files = []

    if SELECTION_MODE == "all":
        selected_files = all_files

    elif SELECTION_MODE == "name_list":
        wanted = set(str(s).strip() for s in STATIONS)

        for f in all_files:
            file_stem = f.stem.strip()

            for station_name in wanted:
                if file_stem == station_name or file_stem.startswith(station_name + "_"):
                    selected_files.append(f)
                    break

    elif SELECTION_MODE == "idx_range":
        for f in all_files:
            try:
                # Read only idx column
                tmp = pd.read_parquet(f, columns=[IDX_COL])
                idx_values = tmp[IDX_COL].dropna().unique()

                if len(idx_values) == 0:
                    print(f"[SKIP] No idx found in {f.name}")
                    continue

                if len(idx_values) > 1:
                    print(f"[WARNING] Multiple idx values found in {f.name}: {idx_values}")

                station_idx = int(idx_values[0])

                if IDX_MIN <= station_idx <= IDX_MAX:
                    selected_files.append(f)

            except Exception as e:
                print(f"[SKIP] Could not read idx from {f.name}: {e}")

    else:
        raise ValueError(
            "Invalid SELECTION_MODE. Use 'name_list', 'idx_range', or 'all'."
        )

    print(f"Total full parquet files found: {len(all_files)}")
    print(f"Selected files: {len(selected_files)}")

    return selected_files


def get_station_id_from_file(parquet_file):
    """
    Extract station ID from filenames like:
        1001A_O3_A_30min.parquet

    Returns:
        1001A
    """
    return parquet_file.stem.split("_")[0]


def get_output_file(station_id):
    """
    Get output CSV path for one station.
    """
    return OUT_DIR / f"{station_id}_{species}_sector_ratio_cv_fits.csv"


def detect_time_key(df):
    """
    Create or detect a timestep key.
    """
    if "datetime" in df.columns:
        return "datetime"

    if "timestamp" in df.columns:
        return "timestamp"

    if "date" in df.columns and "time" in df.columns:
        time_key = "time_key"
        df[time_key] = df["date"].astype(str) + "_" + df["time"].astype(str).str.zfill(4)
        return time_key

    raise ValueError("Could not detect time key. Expected datetime, timestamp, or date + time.")


def prepare_station_dataframe(df, station_id):
    """
    Filter and prepare one station dataframe.

    Important:
    Since each parquet file already corresponds to one station,
    we do NOT filter by station name by default.
    This avoids accidentally deleting all rows when the filename
    does not exactly match the station column.
    """

    df = df.copy()

    print(f"Initial rows: {len(df):,}")

    # ------------------------------------------------------------
    # Optional station-name filtering
    # ------------------------------------------------------------
    # Usually not needed because one parquet = one station.
    # Keep this disabled unless you are sure the station column
    # exactly matches the filename.
    FILTER_BY_STATION_NAME = False

    if FILTER_BY_STATION_NAME and "station" in df.columns:
        before = len(df)
        df = df[df["station"].astype(str) == str(station_id)].copy()
        print(f"After station filter: {len(df):,} rows removed={before-len(df):,}")

    # ------------------------------------------------------------
    # Optional sector_type filtering
    # ------------------------------------------------------------
    if "sector_type" in df.columns:
        print("Available sector_type values:", df["sector_type"].dropna().unique()[:20])

        before = len(df)
        df = df[df["sector_type"].astype(str) == SECTOR_TYPE_KEEP].copy()
        print(f"After sector_type={SECTOR_TYPE_KEEP} filter: {len(df):,} rows removed={before-len(df):,}")

        # If this removed everything, stop early with useful message
        if len(df) == 0:
            raise ValueError(
                f"All rows removed by sector_type filter. "
                f"Check SECTOR_TYPE_KEEP='{SECTOR_TYPE_KEEP}'."
            )

    # ------------------------------------------------------------
    # Sector filtering
    # ------------------------------------------------------------
    print("Available sector values before sector filter:")
    print(df[SECTOR_COL].dropna().unique()[:30])

    before = len(df)
    df = df[df[SECTOR_COL].astype(str).str.match(r"^C\d+$")].copy()
    print(f"After C-number sector filter: {len(df):,} rows removed={before-len(df):,}")

    if len(df) == 0:
        raise ValueError(
            "All rows removed by sector filter. "
            "The sector column may not contain labels like C1, C2, ..., C10."
        )

    df["sector_num"] = df[SECTOR_COL].apply(sector_to_number)

    before = len(df)
    df = df[df["sector_num"].between(1, 10)].copy()
    print(f"After sector_num 1-10 filter: {len(df):,} rows removed={before-len(df):,}")

    # ------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------
    required_base_cols = [
        CENTER_COL,
        MEAN_COL,
        STD_COL,
        CV_COL,
        SECTOR_COL,
    ]

    missing = [c for c in required_base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in station {station_id}: {missing}")

    # ------------------------------------------------------------
    # Ratio
    # ------------------------------------------------------------
    df["ratio_mean_w_center"] = df[MEAN_COL] / df[CENTER_COL]

    # ------------------------------------------------------------
    # Fitting x-axis
    # ------------------------------------------------------------
    if FIT_X_COL in df.columns:
        df["fit_x"] = df[FIT_X_COL].astype(float)
    else:
        df["fit_x"] = df["sector_num"].astype(float)

    # ------------------------------------------------------------
    # Time key
    # ------------------------------------------------------------
    time_key = detect_time_key(df)

    df = df.sort_values([time_key, "sector_num"]).copy()

    print(f"Prepared rows: {len(df):,}")
    print(f"Number of timesteps: {df[time_key].nunique():,}")
    print("Sector counts:")
    print(df["sector_num"].value_counts().sort_index())

    print("Number of sectors per timestep:")
    print(df.groupby(time_key)["sector_num"].nunique().value_counts().sort_index())

    return df, time_key


def get_metadata_columns(df):
    """
    Metadata columns to keep once per timestep.
    """
    candidate_meta_cols = [
        "station",
        "station_idx",
        "station_lat",
        "station_lon",
        "station_alt",
        "model_lat",
        "model_lon",
        "i_center",
        "j_center",
        "date",
        "time",
        "timestamp",
        "datetime",
        "season",
        "day_night",
        "mode",
        "sector_type",
        "k_star_cen",
        "z_target_m",
        CENTER_COL,
    ]

    meta_cols = [c for c in candidate_meta_cols if c in df.columns]
    meta_cols = list(dict.fromkeys(meta_cols))

    return meta_cols


# ============================================================
# MAIN STATION PROCESSING FUNCTION
# ============================================================
def process_one_station(parquet_file):
    """
    Create one derived CSV for one station.
    """

    station_id = get_station_id_from_file(parquet_file)
    output_file = get_output_file(station_id)

    if not parquet_file.exists():
        print(f"[SKIP] Missing parquet: {parquet_file}")
        return None

    if output_file.exists() and not OVERWRITE:
        print(f"[SKIP] Output already exists for station {station_id}: {output_file}")
        return output_file

    print("\n" + "=" * 80)
    print(f"Processing station: {station_id}")
    print(f"Input: {parquet_file}")

    df = pd.read_parquet(parquet_file)

    df, time_key = prepare_station_dataframe(df, station_id)

    if IDX_COL in df.columns:
        idx_values = df[IDX_COL].dropna().unique()
        if len(idx_values) > 0:
            print(f"Station idx: {idx_values[0]}")

    has_median = MEDIAN_COL in df.columns

    if has_median:
        print(f"Median column found: {MEDIAN_COL}")
    else:
        print(f"Median column not found: {MEDIAN_COL}. Median columns and fits will be skipped.")

    meta_cols = get_metadata_columns(df)

    rows = []

    skipped_incomplete = 0

    for timestep, g in df.groupby(time_key):
        g = g.sort_values("sector_num").copy()

        existing_sectors = set(g["sector_num"].tolist())

        # Need at least 3 sectors for quadratic/exponential fits.
        # Linear would need only 2, but 3 is safer because you are fitting all models.
        if len(existing_sectors) < 3:
            skipped_incomplete += 1
            continue

        row = {}

        row[time_key] = timestep

        first = g.iloc[0]

        for col in meta_cols:
            row[col] = first[col]

        # ------------------------------------------------------------
        # Store sector-wise values as wide columns
        # This allows missing sectors instead of skipping the timestep.
        # ------------------------------------------------------------
        for s in range(1, 11):
            gs = g[g["sector_num"] == s]

            if len(gs) == 0:
                row[f"ratio_C{s}"] = np.nan
                row[f"mean_w_C{s}"] = np.nan
                row[f"std_w_C{s}"] = np.nan
                row[f"cv_w_C{s}"] = np.nan

                if has_median:
                    row[f"median_w_C{s}"] = np.nan

                continue

            r = gs.iloc[0]

            row[f"ratio_C{s}"] = r["ratio_mean_w_center"]
            row[f"mean_w_C{s}"] = r[MEAN_COL]
            row[f"std_w_C{s}"] = r[STD_COL]
            row[f"cv_w_C{s}"] = r[CV_COL]

            if has_median:
                row[f"median_w_C{s}"] = r[MEDIAN_COL]

        # ------------------------------------------------------------
        # Main difference metrics
        # These become NaN if C1 or C10 is missing.
        # ------------------------------------------------------------
        row["R_C10_minus_1"] = row["ratio_C10"] - 1.0
        row["R_C10_minus_R_C1"] = row["ratio_C10"] - row["ratio_C1"]
        row["CV_C10_minus_CV_C1"] = row["cv_w_C10"] - row["cv_w_C1"]

        if has_median:
            row["median_w_C10_minus_median_w_C1"] = (
                row["median_w_C10"] - row["median_w_C1"]
            )

        # ------------------------------------------------------------
        # Arrays for fitting
        # Uses only available sectors.
        # ------------------------------------------------------------
        x = g["fit_x"].to_numpy(dtype=float)

        ratio_y = g["ratio_mean_w_center"].to_numpy(dtype=float)
        cv_y = g[CV_COL].to_numpy(dtype=float)
        mean_y = g[MEAN_COL].to_numpy(dtype=float)
        std_y = g[STD_COL].to_numpy(dtype=float)

        row.update(fit_all_models_for_variable(x, ratio_y, "ratio"))
        row.update(fit_all_models_for_variable(x, cv_y, "cv_w"))
        row.update(fit_all_models_for_variable(x, mean_y, "mean_w"))
        row.update(fit_all_models_for_variable(x, std_y, "std_w"))

        if has_median:
            median_y = g[MEDIAN_COL].to_numpy(dtype=float)
            row.update(fit_all_models_for_variable(x, median_y, "median_w"))

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) == 0:
        print(f"[WARNING] Output dataframe is empty for station {station_id}.")
        print("This usually means no timestep had at least 3 valid sectors after filtering.")
    else:
        print("Output preview:")
        print(out.head())

    out.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Output rows: {len(out):,}")
    print(f"Output columns: {len(out.columns):,}")
    print(f"Skipped timesteps with fewer than 3 sectors: {skipped_incomplete:,}")

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():
    start = time.time()
    print("\n===== PRODUCTION RUN =====")
    print("Start:", dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"))

    station_files = get_station_files()

    print(f"Selection mode: {SELECTION_MODE}")

    if SELECTION_MODE == "idx_range":
        print(f"Selected idx range: {IDX_MIN} to {IDX_MAX}")

    print(f"Number of station files to process: {len(station_files)}")
    print(f"Output directory: {OUT_DIR}")

    created_files = []

    for idx, parquet_file in enumerate(station_files, start=1):
        print(f"\n[{idx}/{len(station_files)}] File: {parquet_file.name}")

        try:
            output_file = process_one_station(parquet_file)

            if output_file is not None:
                created_files.append(output_file)

        except Exception as e:
            print(f"[ERROR] File {parquet_file.name} failed.")
            print(f"Reason: {e}")

    print("\n" + "=" * 80)
    print("Processing completed.")
    print(f"Created/updated CSV files: {len(created_files)}")

    if len(created_files) > 0:
        created_list_file = OUT_DIR / (
            f"created_station_metric_files_{SELECTION_MODE}"
        )

        if SELECTION_MODE == "idx_range":
            created_list_file = OUT_DIR / (
                f"created_station_metric_files_idx_{IDX_MIN}_{IDX_MAX}.txt"
            )
        else:
            created_list_file = OUT_DIR / (
                f"created_station_metric_files_{SELECTION_MODE}.txt"
            )

        with open(created_list_file, "w") as f:
            for path in created_files:
                f.write(str(path) + "\n")

        print(f"Saved list of created files: {created_list_file}")
    end = time.time()
    print("End:", dt.datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Execution time: {(end - start) / 60:.2f} minutes")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
# %%
