#%%
"""
Backfill goodness-of-fit metrics into ALREADY-EXISTING station CSVs.

The CSVs produced by create_stations_csv_metrics.py contain, per timestep:
    - the wide sector values  {var}_C1 ... {var}_C10
    - the fitted coefficients  {var}_{model}_a / _b / _c
    - the ordinary R2          {var}_{model}_r2

This script does NOT re-fit anything. It reconstructs each model's
predictions at x = 1..10 from the stored coefficients, compares them to
the stored sector values, and derives:
    {var}_{model}_n        number of sectors used
    {var}_{model}_rss      residual sum of squares
    {var}_{model}_adj_r2   adjusted R2
    {var}_{model}_aic      Akaike Information Criterion
    {var}_{model}_bic      Bayesian Information Criterion

Because x == sector index == radius (1..10), reconstruction is exact and
the result is numerically identical to having computed these in the
original producer. It also recomputes _r2 and writes it as
{var}_{model}_r2_check so you can verify agreement with the stored _r2.

Idempotent: re-running overwrites the derived columns, so it is safe.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

CSV_DIR = Path("/mnt/store01/agkiokas/CAMS/derived_station_metrics/")
CSV_GLOB = "*_sector_ratio_cv_fits.csv"

# Variables whose fits were stored. std_w/median_w included if present.
VARIABLES = ["ratio", "cv_w", "mean_w", "std_w", "median_w"]

# Free-parameter counts per model (k). +1 for variance is added inside.
MODEL_K = {"linear": 2, "quadratic": 3, "exponential": 3}

# Sectors and their x-values. x == sector number == radius.
SECTORS = list(range(1, 11))
X = np.array(SECTORS, dtype=float)

# If True, write a *_r2_check column to validate against stored _r2.
WRITE_R2_CHECK = True

# Write to a new file instead of overwriting the original.
# If None, the original CSV is overwritten in place.
OUT_SUFFIX = "_with_aic"   # e.g. 1006A_O3_sector_ratio_cv_fits_with_aic.csv
# OUT_SUFFIX = None         # overwrite in place


# ============================================================
# PREDICTION RECONSTRUCTION
# ============================================================

def predict_linear(a, b):
    return a + b * X


def predict_quadratic(a, b, c):
    return a + b * X + c * X ** 2


def predict_exponential(a, b, c):
    z = np.clip(c * X, -50, 50)   # same clipping as the producer
    return a + b * np.exp(z)


PREDICTORS = {
    "linear": (predict_linear, ["a", "b"]),
    "quadratic": (predict_quadratic, ["a", "b", "c"]),
    "exponential": (predict_exponential, ["a", "b", "c"]),
}


def metrics_from_residuals(y_obs, y_pred, k):
    """
    Compute n, rss, r2, adj_r2, aic, bic from observed/predicted vectors.

    y_obs, y_pred : length-10 arrays (sectors C1..C10); NaNs allowed.
    k             : free parameters of the model.
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_obs) & np.isfinite(y_pred)
    n = int(valid.sum())

    out = {"n": n, "rss": np.nan, "r2": np.nan,
           "adj_r2": np.nan, "aic": np.nan, "bic": np.nan}

    if n < 2:
        return out

    yo = y_obs[valid]
    yp = y_pred[valid]

    ss_res = float(np.sum((yo - yp) ** 2))
    ss_tot = float(np.sum((yo - np.mean(yo)) ** 2))

    out["rss"] = ss_res

    if ss_tot > 0:
        out["r2"] = 1.0 - ss_res / ss_tot

    denom = n - k - 1
    if denom > 0 and np.isfinite(out["r2"]):
        out["adj_r2"] = 1.0 - (1.0 - out["r2"]) * (n - 1) / denom

    sigma2 = ss_res / n
    n_params = k + 1
    if sigma2 > 0:
        ll_term = n * np.log(sigma2)
        out["aic"] = ll_term + 2.0 * n_params
        out["bic"] = ll_term + np.log(n) * n_params
    else:
        out["aic"] = -np.inf
        out["bic"] = -np.inf

    return out


def sector_columns_present(df, var):
    """Return the list of {var}_C{s} columns that exist in df."""
    return [f"{var}_C{s}" for s in SECTORS if f"{var}_C{s}" in df.columns]


def process_variable_model(df, var, model):
    """
    Add derived metric columns for one (variable, model) pair, vectorized
    where cheap and row-wise where the reconstruction needs it.
    """
    predictor, param_names = PREDICTORS[model]
    k = MODEL_K[model]

    # Coefficient columns, e.g. ratio_linear_a, ratio_linear_b, ...
    coef_cols = [f"{var}_{model}_{p}" for p in param_names]
    sec_cols = [f"{var}_C{s}" for s in SECTORS]

    have_coefs = all(c in df.columns for c in coef_cols)
    have_sectors = all(c in df.columns for c in sec_cols)

    if not (have_coefs and have_sectors):
        # Nothing to do for this pair (variable or model not present).
        return 0

    n_vals = np.full(len(df), np.nan)
    rss_vals = np.full(len(df), np.nan)
    r2_vals = np.full(len(df), np.nan)
    adj_vals = np.full(len(df), np.nan)
    aic_vals = np.full(len(df), np.nan)
    bic_vals = np.full(len(df), np.nan)

    coefs = df[coef_cols].to_numpy(dtype=float)
    y_obs_all = df[sec_cols].to_numpy(dtype=float)  # shape (rows, 10)

    for i in range(len(df)):
        params = coefs[i]
        if not np.all(np.isfinite(params)):
            # Fit failed for this row/model -> metrics stay NaN.
            continue

        y_pred = predictor(*params)
        m = metrics_from_residuals(y_obs_all[i], y_pred, k=k)

        n_vals[i] = m["n"]
        rss_vals[i] = m["rss"]
        r2_vals[i] = m["r2"]
        adj_vals[i] = m["adj_r2"]
        aic_vals[i] = m["aic"]
        bic_vals[i] = m["bic"]

    df[f"{var}_{model}_n"] = n_vals
    df[f"{var}_{model}_rss"] = rss_vals
    df[f"{var}_{model}_adj_r2"] = adj_vals
    df[f"{var}_{model}_aic"] = aic_vals
    df[f"{var}_{model}_bic"] = bic_vals

    if WRITE_R2_CHECK:
        df[f"{var}_{model}_r2_check"] = r2_vals

    return 1


def add_delta_aic(df, var):
    """
    Add {var}_best_model and {var}_{model}_delta_aic across the three models
    for a given variable, so model selection is one lookup downstream.
    """
    aic_cols = {m: f"{var}_{m}_aic" for m in MODEL_K
                if f"{var}_{m}_aic" in df.columns}

    if len(aic_cols) < 2:
        return

    aic_df = df[list(aic_cols.values())].copy()
    aic_df.columns = list(aic_cols.keys())

    best_aic = aic_df.min(axis=1)

    for m in aic_cols:
        df[f"{var}_{m}_delta_aic"] = df[f"{var}_{m}_aic"] - best_aic

    # Name of the lowest-AIC model per row (NaN-safe).
    valid_any = aic_df.notna().any(axis=1)
    best_model = pd.Series(index=df.index, dtype=object)
    best_model[valid_any] = aic_df.loc[valid_any].idxmin(axis=1)
    df[f"{var}_best_model_aic"] = best_model


def process_csv(csv_path):
    df = pd.read_csv(csv_path)

    touched = 0
    vars_present = []

    for var in VARIABLES:
        if len(sector_columns_present(df, var)) == 0:
            continue
        vars_present.append(var)
        for model in MODEL_K:
            touched += process_variable_model(df, var, model)
        add_delta_aic(df, var)

    if OUT_SUFFIX is None:
        out_path = csv_path
    else:
        out_path = csv_path.with_name(csv_path.stem + OUT_SUFFIX + csv_path.suffix)

    df.to_csv(out_path, index=False)

    # Validation summary: max abs difference between recomputed and stored r2.
    check_msg = ""
    if WRITE_R2_CHECK:
        diffs = []
        for var in vars_present:
            for model in MODEL_K:
                stored = f"{var}_{model}_r2"
                check = f"{var}_{model}_r2_check"
                if stored in df.columns and check in df.columns:
                    d = (df[stored] - df[check]).abs()
                    if d.notna().any():
                        diffs.append(d.max())
        if diffs:
            check_msg = f" | max |r2 - r2_check| = {max(diffs):.2e}"

    print(f"[OK] {csv_path.name}: {len(df):,} rows, "
          f"vars={vars_present}, pairs updated={touched} -> {out_path.name}{check_msg}")


def main():
    csv_files = sorted(CSV_DIR.glob(CSV_GLOB))
    # Avoid re-processing our own outputs.
    if OUT_SUFFIX:
        csv_files = [f for f in csv_files if OUT_SUFFIX not in f.stem]

    print(f"Found {len(csv_files)} CSV files in {CSV_DIR}")

    for i, f in enumerate(csv_files, 1):
        try:
            process_csv(f)
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
# %%