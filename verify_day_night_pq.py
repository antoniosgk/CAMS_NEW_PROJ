#%%
from pathlib import Path
import pandas as pd
import numpy as np
import datetime as dt

# ============================================================
# SETTINGS
# ============================================================

PARQUET_DIR = Path("/mnt/store01/agkiokas/CAMS/stations_parquet/")

STATIONS = ["1001A", "3401A"]
# STATIONS = "all"

TIME_OFFSET_HOURS = 0
THRESHOLD_DEG = -0.833

FIX_LABELS = False

# ============================================================
# SOLAR FUNCTION
# ============================================================

def day_night_label(lat, lon, ts, threshold_deg=-0.833, return_elev=False):
    lon = float(lon)
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat_rad = np.deg2rad(float(lat))

    if ts.tzinfo is not None:
        ts_utc = ts.astimezone(dt.timezone.utc)
    else:
        ts_utc = ts

    doy = ts_utc.timetuple().tm_yday

    hour = (
        ts_utc.hour
        + ts_utc.minute / 60.0
        + ts_utc.second / 3600.0
    )

    gamma = 2.0 * np.pi / 365.0 * (
        doy - 1 + (hour - 12.0) / 24.0
    )

    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.001480 * np.sin(3 * gamma)
    )

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    tst = (hour * 60.0 + eqtime + 4.0 * lon) % 1440.0

    ha = np.deg2rad((tst / 4.0) - 180.0)

    cosz = (
        np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.cos(ha)
    )

    cosz = np.clip(cosz, -1.0, 1.0)

    zen = np.arccos(cosz)

    elev_deg = np.rad2deg(np.pi / 2.0 - zen)

    label = "day" if elev_deg > threshold_deg else "night"

    return (label, elev_deg) if return_elev else label


# ============================================================
# HELPERS
# ============================================================

def build_datetime(df):
    date_str = df["timestamp"].astype(str).str[:8]
    time_str = df["time"].astype(str).str.zfill(4)

    df["datetime"] = pd.to_datetime(
        date_str + time_str,
        format="%Y%m%d%H%M",
        errors="coerce"
    )

    df["datetime"] = (
        df["datetime"]
        + pd.Timedelta(hours=TIME_OFFSET_HOURS)
    )

    return df


def get_station_files():
    # exclude summary parquet files
    all_files = [
        fp for fp in PARQUET_DIR.glob("*.parquet")
        if "summary" not in fp.name.lower()
    ]

    if STATIONS == "all":
        return sorted(all_files)

    out = []
    for st in STATIONS:
        matches = [
            fp for fp in all_files
            if st in fp.name
        ]

        if not matches:
            print(f"Warning: no parquet file found for station {st}")

        out.extend(matches)

    return sorted(out)


# ============================================================
# MAIN
# ============================================================

files = get_station_files()

for fp in files:
    print("\n================================================")
    print(fp.name)
    print("================================================")

    df = pd.read_parquet(fp)
    df = build_datetime(df)
    
    # Compute once per unique timestamp to optimize performance
    unique_times = df[["datetime"]].drop_duplicates().copy()

    lat = float(df["station_lat"].iloc[0])
    lon = float(df["station_lon"].iloc[0])

    print(f"Station lat/lon: {lat:.3f}, {lon:.3f}")

    labels = []
    elevations = []

    # FIXED: Properly indented the append steps inside the loop
    for ts in unique_times["datetime"]:
        lbl, elev = day_night_label(
            lat,
            lon,
            ts,
            threshold_deg=THRESHOLD_DEG,
            return_elev=True
        )
        labels.append(lbl)
        elevations.append(elev)

    unique_times["day_night_new"] = labels
    unique_times["solar_elev_deg"] = elevations

    # Merge back to all rows (preserving all 10 sector rows)
    df = df.merge(unique_times, on="datetime", how="left")

    # FIXED: Completely removed the destructive duplicate drop and the redundant 2nd loop here

    mismatches = (
        df["day_night"].astype(str).str.lower()
        != df["day_night_new"].astype(str).str.lower()
    )

    print(f"Total rows checked: {len(df)}")
    print(f"Mismatch rows: {mismatches.sum()}")
    print(f"Mismatch %: {100 * mismatches.mean():.4f}")

    # ========================================================
    # FIND TRANSITIONS
    # ========================================================
    
    # We drop duplicates safely here only to cleanly spot time transitions
    check_df = df.drop_duplicates(subset=["datetime"]).copy()

    check_df["transition"] = (
        check_df["day_night_new"]
        != check_df["day_night_new"].shift(1)
    )

    transitions = check_df[check_df["transition"]]

    print("\nTransitions:")
    print(
        transitions[
            ["datetime", "day_night_new", "solar_elev_deg"]
        ]tail(20)
    )

    # ========================================================
    # OPTIONAL FIX
    # ========================================================

    if FIX_LABELS:
        df["day_night"] = df["day_night_new"]
        
        # Clean up temporary processing columns before saving
        df = df.drop(columns=["day_night_new", "solar_elev_deg"])

        # Saves the complete dataset back safely
        df.to_parquet(fp, index=False)
        print("Saved corrected parquet.")
# %%
