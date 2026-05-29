#%%
from pathlib import Path
import pandas as pd
import numpy as np
from astral import Observer
from astral.sun import elevation
import datetime as dt

# ============================================================
# SETTINGS
# ============================================================

PARQUET_DIR = Path("/mnt/store01/agkiokas/CAMS/stations_parquet/")
OUT_DIR = Path("/mnt/store01/agkiokas/CAMS/daynight_astral_check/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

#STATIONS = ["1001A", "3401A"]
STATIONS = "all"

THRESHOLD_DEG = -0.833

# IMPORTANT:
# If timestamp/time columns represent UTC, keep True.
# If they represent China local time UTC+8, set False.
TIMESTAMPS_ARE_UTC = True
LOCAL_TIME_OFFSET_HOURS = 8

FIX_LABELS = False

SUMMARY_CSV = OUT_DIR / "daynight_astral_check_summary.csv"


# ============================================================
# HELPERS
# ============================================================

def build_datetime(df):
    date_str = df["timestamp"].astype(str).str[:8]
    time_str = df["time"].astype(str).str.zfill(4)

    naive_dt = pd.to_datetime(
        date_str + time_str,
        format="%Y%m%d%H%M",
        errors="coerce"
    )

    if TIMESTAMPS_ARE_UTC:
        df["datetime_utc"] = naive_dt.dt.tz_localize("UTC")
    else:
        df["datetime_utc"] = (
            naive_dt - pd.Timedelta(hours=LOCAL_TIME_OFFSET_HOURS)
        ).dt.tz_localize("UTC")

    return df


def get_station_files():
    all_files = [
        fp for fp in PARQUET_DIR.glob("*.parquet")
        if "summary" not in fp.name.lower()
    ]

    if STATIONS == "all":
        return sorted(all_files)

    out = []
    for st in STATIONS:
        matches = [fp for fp in all_files if st in fp.name]
        if not matches:
            print(f"Warning: no parquet file found for station {st}")
        out.extend(matches)

    return sorted(out)


def astral_day_night_label(lat, lon, timestamp_utc):
    obs = Observer(
        latitude=float(lat),
        longitude=float(lon),
        elevation=0
    )

    elev = elevation(obs, timestamp_utc.to_pydatetime())

    label = "day" if elev > THRESHOLD_DEG else "night"

    return label, elev


# ============================================================
# MAIN
# ============================================================

summary_rows = []

files = get_station_files()

for fp in files:

    print("\n================================================")
    print(fp.name)
    print("================================================")

    df = pd.read_parquet(fp)

    required = ["timestamp", "time", "station_lat", "station_lon", "day_night"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"Skipping {fp.name}: missing columns {missing}")
        continue

    df = build_datetime(df)
    df = df.dropna(subset=["datetime_utc"]).copy()

    if df.empty:
        print(f"Skipping {fp.name}: no valid datetimes")
        continue

    station = str(df["station"].iloc[0]) if "station" in df.columns else fp.stem
    lat = float(df["station_lat"].iloc[0])
    lon = float(df["station_lon"].iloc[0])

    print(f"Station: {station}")
    print(f"Lat/Lon: {lat:.4f}, {lon:.4f}")

    # one calculation per timestep, then merge back to all 10 sector rows
    unique_times = df[["datetime_utc"]].drop_duplicates().copy()

    labels = []
    elevations = []

    for ts in unique_times["datetime_utc"]:
        lbl, elev = astral_day_night_label(lat, lon, ts)
        labels.append(lbl)
        elevations.append(elev)

    unique_times["day_night_astral"] = labels
    unique_times["solar_elev_astral_deg"] = elevations

    df = df.merge(unique_times, on="datetime_utc", how="left")

    old = df["day_night"].astype(str).str.lower()
    new = df["day_night_astral"].astype(str).str.lower()

    mismatches = old != new

    n_rows = len(df)
    n_bad = int(mismatches.sum())
    pct_bad = 100 * n_bad / n_rows if n_rows > 0 else np.nan

    print(f"Total rows checked : {n_rows}")
    print(f"Mismatch rows      : {n_bad}")
    print(f"Mismatch %         : {pct_bad:.4f}")

    # transition diagnostics: one row per timestep
    check_df = df.drop_duplicates(subset=["datetime_utc"]).copy()
    check_df = check_df.sort_values("datetime_utc")

    check_df["transition_astral"] = (
        check_df["day_night_astral"]
        != check_df["day_night_astral"].shift(1)
    )

    transitions = check_df[check_df["transition_astral"]]

    print("\nLast 20 Astral transitions:")
    print(
        transitions[
            ["datetime_utc", "day_night_astral", "solar_elev_astral_deg"]
        ].tail(20).to_string(index=False)
    )

    # save mismatch rows for inspection
    mismatch_out = df.loc[
        mismatches,
        [
            "station",
            "datetime_utc",
            "station_lat",
            "station_lon",
            "day_night",
            "day_night_astral",
            "solar_elev_astral_deg",
        ]
    ].copy() if "station" in df.columns else df.loc[
        mismatches,
        [
            "datetime_utc",
            "station_lat",
            "station_lon",
            "day_night",
            "day_night_astral",
            "solar_elev_astral_deg",
        ]
    ].copy()

    mismatch_csv = OUT_DIR / f"{station}_daynight_mismatches_astral.csv"
    mismatch_out.to_csv(mismatch_csv, index=False)

    print(f"Saved mismatch details: {mismatch_csv}")

    summary_rows.append({
        "station": station,
        "file": fp.name,
        "lat": lat,
        "lon": lon,
        "rows_checked": n_rows,
        "mismatch_rows": n_bad,
        "mismatch_pct": pct_bad,
    })

    if FIX_LABELS:
        df["day_night"] = df["day_night_astral"]

        df = df.drop(
            columns=[
                "day_night_astral",
                "solar_elev_astral_deg"
            ],
            errors="ignore"
        )

        df.to_parquet(fp, index=False)
        print("Saved corrected parquet.")


summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_CSV, index=False)

print("\n================================================")
print("FINAL SUMMARY")
print("================================================")
print(summary.to_string(index=False))
print(f"\nSaved summary: {SUMMARY_CSV}")
# %%
