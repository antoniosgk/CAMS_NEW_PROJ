#%%
import pandas as pd
import numpy as np
from file_utils import base_path,stations_path
from vertical_indexing import metpy_find_level_index
from horizontal_indexing import nearest_grid_index
#%%
def load_stations(stations_path):
    """Load station table. Keeps all rows; marks invalid rows (NaNs) for later filtering."""
    df = pd.read_csv(stations_path, sep="\t")
    df = df.reset_index().rename(columns={"index": "idx"})

    # normalize column names
    for col in df.columns:
        if col.lower().startswith("station"):
            df = df.rename(columns={col: "Station_Name"})
        if col.lower().startswith("lat"):
            df = df.rename(columns={col: "Latitude"})
        if col.lower().startswith("lon"):
            df = df.rename(columns={col: "Longitude"})
        if col.lower().startswith("alt"):
            df = df.rename(columns={col: "Altitude"})

    expected = ["idx", "Station_Name", "Latitude", "Longitude", "Altitude"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Stations file missing expected columns: {missing}")

    # coerce numeric; DO NOT drop rows here
    for col in ["Latitude", "Longitude", "Altitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # mark validity (so bulk loops can filter)
    df["is_valid"] = df[["Latitude", "Longitude", "Altitude"]].notna().all(axis=1)

    return df[expected + ["is_valid"]]



def select_station(df, idx=None, name=None):
    """Return station row by idx or name. If station exists but has NaNs, raise a clear error."""
    if idx is None and name is None:
        print(df[["idx", "Station_Name", "is_valid"]])
        s = input("Select station by index or name: ").strip()
    else:
        s = str(idx if idx is not None else name)

    row = None

    if s.isdigit():
        r = df[df["idx"] == int(s)]
        if not r.empty:
            row = r.iloc[0]
    else:
        r = df[df["Station_Name"] == s]
        if not r.empty:
            row = r.iloc[0]

    if row is None:
        raise ValueError(f"Station '{s}' not found.")

    if not bool(row.get("is_valid", True)):
        raise ValueError(
            f"Station '{row['Station_Name']}' (idx={row['idx']}) has missing "
            f"Latitude/Longitude/Altitude in the stations file; cannot run calculations."
        )

    return row