#%%
"""
aggregate_station_data.py
=========================
Incremental pass over all *_O3_area_fits.csv files.
On the first run it builds both summary files from scratch.
On subsequent runs it skips already-processed stations and merges
only the new ones into the existing files.

To force reprocessing of specific stations (e.g. after re-fitting),
add their IDs to FORCE_REPROCESS. Their old rows and count
contributions are subtracted before reprocessing.

Outputs (written to OUT_DIR)
----------------------------
station_summary.csv
    One row per station.
    station_id | lat | lon
    | {var}_{crit}_winner   (modal winning model name)
    | {var}_{crit}_win_pct  (% of valid timesteps won by modal winner)
    for every var in [ratio, cv_w] x crit in [r2, adj_r2, aic, aicc, bic]

global_counts.csv
    One row per (var, criterion, model) -- always exactly 70 rows.
    var | criterion | model | count
    count = cumulative total (station x timestep) pairs where that model won,
    summed across ALL stations processed so far.
"""

from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

CSV_DIR    = Path("/mnt/store01/agkiokas/CAMS/fit_outputs")
OUT_DIR    = CSV_DIR / "summary"
CSV_SUFFIX = "_O3_area_fits.csv"

VARIABLES = ["ratio", "cv_w"]
MODELS    = ["linear", "quadratic", "cubic", "logarithmic",
             "exponential", "power", "saturating"]
CRITERIA  = ["r2", "adj_r2", "aic", "aicc", "bic"]

# Preferred lat/lon columns — first found in the file is used
LAT_CANDIDATES = ["station_lat", "model_lat"]
LON_CANDIDATES = ["station_lon", "model_lon"]

# Station IDs to force-reprocess even if already in summary
# (e.g. after re-running metrics_fit_backfill_area.py on them)
# Example: FORCE_REPROCESS = ["1002A", "1003A"]
FORCE_REPROCESS: list[str] = []

# Progress print every N stations
PROGRESS_EVERY = 100

# ============================================================
# COLUMN HELPERS
# ============================================================

def _build_needed_columns() -> set:
    """Minimal column set to load from each CSV."""
    cols = set(LAT_CANDIDATES + LON_CANDIDATES)
    for var in VARIABLES:
        for model in MODELS:
            cols.add(f"{var}_{model}_r2")
            cols.add(f"{var}_{model}_adj_r2")
        for crit in ("aic", "aicc", "bic"):
            cols.add(f"{var}_best_model_{crit}")
    return cols


def _first_valid(df: pd.DataFrame, candidates: list) -> float:
    for c in candidates:
        if c in df.columns:
            val = df[c].iloc[0]
            if pd.notna(val):
                return float(val)
    return np.nan


def _winner_series(df: pd.DataFrame, var: str, crit: str) -> pd.Series:
    """
    Per-row winning model name.
    r2 / adj_r2 : argmax across 7 model columns (higher = better).
    aic/aicc/bic: pre-computed best_model column   (lower = better).
    """
    if crit in ("r2", "adj_r2"):
        cols_wanted  = [f"{var}_{m}_{crit}" for m in MODELS]
        cols_present = [c for c in cols_wanted if c in df.columns]
        if not cols_present:
            return pd.Series([np.nan] * len(df), dtype=object)
        sub     = df[cols_present].apply(pd.to_numeric, errors="coerce")
        all_nan = sub.isna().all(axis=1)
        prefix  = f"{var}_"
        suffix  = f"_{crit}"
        winner  = sub.idxmax(axis=1).str[len(prefix):-len(suffix)]
        winner  = winner.astype(object)
        winner[all_nan] = np.nan
        return winner
    else:
        bcol = f"{var}_best_model_{crit}"
        if bcol not in df.columns:
            return pd.Series([np.nan] * len(df), dtype=object)
        return df[bcol].copy().astype(object)


def _per_station_stats(ws: pd.Series) -> tuple:
    """Return (modal_winner, win_pct) for one station's winner series."""
    valid = ws.dropna()
    valid = valid[valid.isin(MODELS)]
    if valid.empty:
        return np.nan, np.nan
    vc           = valid.value_counts()
    modal_winner = vc.index[0]
    win_pct      = round(100.0 * vc.iloc[0] / len(valid), 4)
    return modal_winner, win_pct


def _winner_counts_from_series(ws: pd.Series) -> dict:
    """
    Return {var: {crit: {model: count}}} contribution of one station's
    winner series.  ws must already be for a single (var, crit).
    """
    valid = ws.dropna()
    valid = valid[valid.isin(MODELS)]
    return valid.value_counts().to_dict()   # {model: count}

# ============================================================
# INCREMENTAL STATE MANAGEMENT
# ============================================================

STATIONS_PATH = None   # set in aggregate()
COUNTS_PATH   = None

def _load_existing_summary() -> pd.DataFrame:
    """Load station_summary.csv if it exists, else return empty DataFrame."""
    if STATIONS_PATH.exists():
        df = pd.read_csv(STATIONS_PATH)
        print(f"  Existing summary: {len(df)} station(s) already processed.")
        return df
    print("  No existing summary found — starting from scratch.")
    return pd.DataFrame()


def _load_existing_counts() -> dict:
    """
    Load global_counts.csv into nested dict {var}{crit}{model: count}.
    Returns zeroed dict if file does not exist.
    """
    zeroed = {
        v: {c: {m: 0 for m in MODELS} for c in CRITERIA}
        for v in VARIABLES
    }
    if not COUNTS_PATH.exists():
        return zeroed
    df = pd.read_csv(COUNTS_PATH)
    for _, row in df.iterrows():
        v, c, m = row["var"], row["criterion"], row["model"]
        if v in zeroed and c in zeroed[v] and m in zeroed[v][c]:
            zeroed[v][c][m] = int(row["count"])
    return zeroed


def _subtract_station_counts(
    global_counts: dict,
    station_row: pd.Series,
    df_station_csv: pd.DataFrame,
    needed: set,
) -> None:
    """
    Subtract one station's timestep-level winner counts from global_counts.
    Used when force-reprocessing a station to avoid double-counting.
    We reload its CSV (already in memory as df_station_csv) and subtract.
    """
    for var in VARIABLES:
        for crit in CRITERIA:
            ws    = _winner_series(df_station_csv, var, crit)
            valid = ws.dropna()
            valid = valid[valid.isin(MODELS)]
            for model, cnt in valid.value_counts().items():
                global_counts[var][crit][model] = max(
                    0, global_counts[var][crit][model] - int(cnt)
                )


def _save_outputs(
    df_summary: pd.DataFrame,
    global_counts: dict,
) -> None:
    df_summary = df_summary.sort_values("station_id").reset_index(drop=True)
    df_summary.to_csv(STATIONS_PATH, index=False)
    print(f"\nSaved: {STATIONS_PATH}  ({len(df_summary)} stations total)")

    count_rows = [
        {"var": v, "criterion": c, "model": m,
         "count": global_counts[v][c][m]}
        for v in VARIABLES
        for c in CRITERIA
        for m in MODELS
    ]
    pd.DataFrame(count_rows).to_csv(COUNTS_PATH, index=False)
    print(f"Saved: {COUNTS_PATH}  (70 rows, cumulative counts)")


# ============================================================
# MAIN AGGREGATION
# ============================================================

def aggregate():
    global STATIONS_PATH, COUNTS_PATH

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATIONS_PATH = OUT_DIR / "station_summary.csv"
    COUNTS_PATH   = OUT_DIR / "global_counts.csv"

    needed    = _build_needed_columns()
    csv_files = sorted(CSV_DIR.glob(f"*{CSV_SUFFIX}"))

    if not csv_files:
        raise FileNotFoundError(f"No files matching *{CSV_SUFFIX} in {CSV_DIR}")

    print(f"Found {len(csv_files)} CSV file(s) in {CSV_DIR}")

    # --- Load existing state ---
    df_summary    = _load_existing_summary()
    global_counts = _load_existing_counts()

    already_done  = set(df_summary["station_id"].tolist()) \
                    if not df_summary.empty else set()
    force_set     = set(FORCE_REPROCESS)

    # Stations to skip (done and not forced)
    skip_set      = already_done - force_set

    to_process = [f for f in csv_files
                  if f.name[: -len(CSV_SUFFIX)] not in skip_set]

    n_new    = sum(1 for f in to_process
                   if f.name[: -len(CSV_SUFFIX)] not in already_done)
    n_forced = sum(1 for f in to_process
                   if f.name[: -len(CSV_SUFFIX)] in force_set)

    print(f"  To process : {len(to_process)} "
          f"({n_new} new, {n_forced} forced reprocess, "
          f"{len(skip_set)} skipped)\n")

    if not to_process:
        print("Nothing to do.")
        return

    # Convert existing summary to list of dicts for easy update
    summary_rows = df_summary.to_dict("records") if not df_summary.empty else []
    # Index by station_id for fast lookup when replacing forced rows
    summary_index = {r["station_id"]: i for i, r in enumerate(summary_rows)}

    t0 = time.time()
    for file_idx, fpath in enumerate(to_process, 1):
        sid = fpath.name[: -len(CSV_SUFFIX)]

        try:
            df = pd.read_csv(
                fpath,
                usecols=lambda c: c in needed,
                low_memory=False,
            )
        except Exception as exc:
            warnings.warn(f"[SKIP] {fpath.name}: {exc}")
            continue

        if df.empty:
            warnings.warn(f"[SKIP] {sid}: empty file")
            continue

        # If force-reprocessing: subtract old counts first, then remove old row
        if sid in force_set and sid in summary_index:
            _subtract_station_counts(global_counts, None, df, needed)
            idx = summary_index.pop(sid)
            summary_rows.pop(idx)
            # Re-index after removal
            summary_index = {r["station_id"]: i
                             for i, r in enumerate(summary_rows)}

        lat = _first_valid(df, LAT_CANDIDATES)
        lon = _first_valid(df, LON_CANDIDATES)
        row = {"station_id": sid, "lat": lat, "lon": lon}

        for var in VARIABLES:
            for crit in CRITERIA:
                ws = _winner_series(df, var, crit)

                # Add to global counts
                valid_ws = ws.dropna()
                valid_ws = valid_ws[valid_ws.isin(MODELS)]
                for model, cnt in valid_ws.value_counts().items():
                    global_counts[var][crit][model] += int(cnt)

                # Per-station stats
                winner, win_pct = _per_station_stats(ws)
                row[f"{var}_{crit}_winner"]  = winner
                row[f"{var}_{crit}_win_pct"] = win_pct

        summary_rows.append(row)
        summary_index[sid] = len(summary_rows) - 1

        if file_idx % PROGRESS_EVERY == 0 or file_idx == len(to_process):
            elapsed = time.time() - t0
            rate    = file_idx / elapsed
            eta     = (len(to_process) - file_idx) / rate if rate > 0 else 0
            print(f"  {file_idx:>4}/{len(to_process)}  "
                  f"{elapsed/60:.1f} min elapsed  ETA {eta/60:.1f} min")

    _save_outputs(pd.DataFrame(summary_rows), global_counts)
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    aggregate()
# %%
