#%%
from pathlib import Path
import pandas as pd

# ============================================================
# USER SETTINGS
# ============================================================

PARQUET_DIR = Path("/mnt/store01/agkiokas/CAMS/stations_parquet/")
OUT_DIR = Path("/mnt/store01/agkiokas/CAMS/daily_ranges/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# One station:
#STATIONS = ["1006A"]

# Multiple stations:
# STATIONS = ["1001A", "1002A"]

# All stations:
STATIONS = "all"

COLUMN = "center_ppb"

SAVE_BIG_CSV = True
OUT_BIG_CSV = OUT_DIR / f"all_stations_{COLUMN}_daily_min_max_range.csv"


# ============================================================
# HELPERS
# ============================================================

def get_station_files(parquet_dir, stations):
    all_files = [
        fp for fp in parquet_dir.glob("*.parquet")
        if "summary" not in fp.name.lower()
    ]

    if stations == "all":
        return sorted(all_files)

    files = []
    for st in stations:
        matches = [
            fp for fp in all_files
            if st in fp.name
        ]

        if not matches:
            print(f"Warning: no parquet file found for station {st}")

        files.extend(matches)

    return sorted(files)


def build_datetime(df):
    date_str = df["timestamp"].astype(str).str[:8]
    time_str = df["time"].astype(str).str.zfill(4)

    df["datetime"] = pd.to_datetime(
        date_str + time_str,
        format="%Y%m%d%H%M",
        errors="coerce"
    )

    return df


def process_station_file(fp, column):
    df = pd.read_parquet(fp)

    if column not in df.columns:
        print(f"Skipping {fp.name}: column {column} not found")
        return None, None

    df = build_datetime(df)
    df = df.dropna(subset=["datetime", column])

    if df.empty:
        print(f"Skipping {fp.name}: no valid data")
        return None, None

    # center_ppb is repeated over 10 sectors
    df = df.drop_duplicates(subset=["datetime"]).copy()

    station = str(df["station"].iloc[0]) if "station" in df.columns else fp.stem.split("_")[0]

    df["date_only"] = df["datetime"].dt.date

    idx_min = df.groupby("date_only")[column].idxmin()
    idx_max = df.groupby("date_only")[column].idxmax()

    min_df = (
        df.loc[idx_min, ["date_only", "datetime", column]]
        .rename(columns={
            "datetime": "time_of_min",
            column: "min_value"
        })
    )

    max_df = (
        df.loc[idx_max, ["date_only", "datetime", column]]
        .rename(columns={
            "datetime": "time_of_max",
            column: "max_value"
        })
    )

    daily_stats = min_df.merge(max_df, on="date_only", how="inner")

    daily_stats["difference"] = (
        daily_stats["max_value"] - daily_stats["min_value"]
    )

    avg_difference = daily_stats["difference"].mean()

    daily_stats.insert(0, "station", station)
    daily_stats["average_difference_for_station"] = avg_difference

    return station, daily_stats


# ============================================================
# MAIN
# ============================================================

files = get_station_files(PARQUET_DIR, STATIONS)

if not files:
    raise FileNotFoundError("No parquet files found.")

all_results = []

for fp in files:
    print(f"\nProcessing: {fp.name}")

    station, daily_stats = process_station_file(fp, COLUMN)

    if daily_stats is None:
        continue

    all_results.append(daily_stats)

    print(daily_stats.to_string(index=False))

    print("\nSummary:")
    print(f"Station: {station}")
    print(f"Days: {len(daily_stats)}")
    print(f"Average daily difference: {daily_stats['difference'].mean():.3f}")

if all_results:
    big_df = pd.concat(all_results, ignore_index=True)

    print("\n====================================================")
    print("FINAL SUMMARY")
    print("====================================================")
    print(f"Stations processed: {big_df['station'].nunique()}")
    print(f"Total station-days: {len(big_df)}")
    print(f"Overall average daily difference: {big_df['difference'].mean():.3f}")

    if SAVE_BIG_CSV:
        big_df.to_csv(OUT_BIG_CSV, index=False)
        print(f"\nSaved big CSV: {OUT_BIG_CSV}")
else:
    print("No results produced.")
# %%
