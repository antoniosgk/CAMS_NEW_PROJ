#%%
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# ============================================================
# USER SETTINGS
# ============================================================
STATION_ID = "1006A"
INPUT_PARQUET = Path(f"/mnt/store01/agkiokas/CAMS/stations_parquet/{STATION_ID}_O3_A_30min.parquet")

OUT_DIR = Path("/home/agkiokas/CAMS/derived_station_metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / f"{STATION_ID}_sector_ratio_cv_fits.csv"

# Use only cumulative sectors C1-C10
SECTOR_TYPE_KEEP = "CUM"

# Columns expected in the parquet file
CENTER_COL = "center_ppb"
MEAN_COL = "mean_w"
STD_COL = "std_w"
CV_COL = "cv_w"

SECTOR_COL = "sector"
RADIUS_COL = "radius"

# Use "radius" if available. Otherwise use sector number C1=1, ..., C10=10.
FIT_X_COL = "radius"


# ============================================================
# HELPER FUNCTIONS
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
    Fit y = a + b*x
    Returns intercept, slope, r2.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 2:
        return {
            "a": np.nan,
            "b": np.nan,
            "r2": np.nan,
        }

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
    Fit y = a + b*x + c*x²
    Returns intercept, linear coefficient, quadratic coefficient, r2.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 3:
        return {
            "a": np.nan,
            "b": np.nan,
            "c": np.nan,
            "r2": np.nan,
        }

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
    Exponential relationship:

        y = a + b * exp(c*x)

    This form includes an offset a.
    """
    return a + b * np.exp(c * x)


def fit_exponential(x, y):
    """
    Fit y = a + b*exp(c*x)
    Returns a, b, c, r2.

    This may fail for some timesteps, especially when the sector profile is almost flat.
    In those cases, NaN values are returned.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 3:
        return {
            "a": np.nan,
            "b": np.nan,
            "c": np.nan,
            "r2": np.nan,
        }

    x_fit = x[valid]
    y_fit = y[valid]

    try:
        # Initial guesses
        a0 = np.nanmean(y_fit)
        b0 = y_fit[0] - a0
        c0 = 0.01

        popt, _ = curve_fit(
            exponential_model,
            x_fit,
            y_fit,
            p0=[a0, b0, c0],
            maxfev=10000,
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
        return {
            "a": np.nan,
            "b": np.nan,
            "c": np.nan,
            "r2": np.nan,
        }


def fit_all_models_for_variable(x, y, prefix):
    """
    Fit linear, quadratic and exponential models for one variable.
    Prefix can be, for example:
        ratio
        cv_w
        mean_w
        std_w
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
# LOAD DATA
# ============================================================

df = pd.read_parquet(INPUT_PARQUET)

print(f"Loaded: {INPUT_PARQUET}")
print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# BASIC FILTERING AND PREPARATION
# ============================================================

# Keep one station only, in case the file contains more than one
if "station" in df.columns:
    df = df[df["station"].astype(str) == STATION_ID].copy()

# Keep only cumulative sectors if the column exists
if "sector_type" in df.columns:
    df = df[df["sector_type"].astype(str) == SECTOR_TYPE_KEEP].copy()

# Keep only C1-C10
df = df[df[SECTOR_COL].astype(str).str.match(r"^C\d+$")].copy()

df["sector_num"] = df[SECTOR_COL].apply(sector_to_number)

df = df[df["sector_num"].between(1, 10)].copy()

# Create ratio as a stored variable
df["ratio_mean_w_center"] = df[MEAN_COL] / df[CENTER_COL]

# Select x-axis for fitting
if FIT_X_COL in df.columns:
    df["fit_x"] = df[FIT_X_COL].astype(float)
else:
    df["fit_x"] = df["sector_num"].astype(float)

# Create a robust timestep key
# Prefer timestamp/datetime if available, otherwise use date + time.
if "datetime" in df.columns:
    TIME_KEY = "datetime"
elif "timestamp" in df.columns:
    TIME_KEY = "timestamp"
else:
    TIME_KEY = "time_key"
    df[TIME_KEY] = df["date"].astype(str) + "_" + df["time"].astype(str).str.zfill(4)

# Sort for consistency
df = df.sort_values([TIME_KEY, "sector_num"]).copy()


# ============================================================
# METADATA COLUMNS TO KEEP ONCE PER TIMESTEP
# ============================================================

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

# Remove duplicated possible time key from metadata if needed
meta_cols = list(dict.fromkeys(meta_cols))


# ============================================================
# CREATE ONE ROW PER TIMESTEP
# ============================================================

rows = []

for timestep, g in df.groupby(TIME_KEY):
    g = g.sort_values("sector_num").copy()

    # Require all sectors C1-C10 to exist
    existing_sectors = set(g["sector_num"].tolist())
    required_sectors = set(range(1, 11))

    if not required_sectors.issubset(existing_sectors):
        continue

    # Start with metadata from first row
    row = {}

    row[TIME_KEY] = timestep

    first = g.iloc[0]
    for col in meta_cols:
        row[col] = first[col]

    # Store sector-wise values as columns
    for _, r in g.iterrows():
        s = int(r["sector_num"])

        row[f"ratio_C{s}"] = r["ratio_mean_w_center"]
        row[f"mean_w_C{s}"] = r[MEAN_COL]
        row[f"std_w_C{s}"] = r[STD_COL]
        row[f"cv_w_C{s}"] = r[CV_COL]

    # Differences requested
    row["R_C10_minus_1"] = row["ratio_C10"] - 1.0
    row["R_C10_minus_R_C1"] = row["ratio_C10"] - row["ratio_C1"]
    row["CV_C10_minus_CV_C1"] = row["cv_w_C10"] - row["cv_w_C1"]

    # Arrays for fitting
    x = g["fit_x"].to_numpy(dtype=float)

    ratio_y = g["ratio_mean_w_center"].to_numpy(dtype=float)
    cv_y = g[CV_COL].to_numpy(dtype=float)
    mean_y = g[MEAN_COL].to_numpy(dtype=float)
    std_y = g[STD_COL].to_numpy(dtype=float)

    # Fit models across sectors
    row.update(fit_all_models_for_variable(x, ratio_y, "ratio"))
    row.update(fit_all_models_for_variable(x, cv_y, "cv_w"))
    row.update(fit_all_models_for_variable(x, mean_y, "mean_w"))
    row.update(fit_all_models_for_variable(x, std_y, "std_w"))

    rows.append(row)


out = pd.DataFrame(rows)


# ============================================================
# SAVE OUTPUT
# ============================================================

out.to_csv(OUTPUT_CSV, index=False)

print(f"Saved output CSV:")
print(OUTPUT_CSV)
print(f"Output rows: {len(out):,}")
print(f"Output columns: {len(out.columns):,}")

print("\nExample columns:")
print(out.columns.tolist()[:40])
# %%
