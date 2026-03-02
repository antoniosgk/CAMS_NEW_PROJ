# file_utils.py
from pathlib import Path
from datetime import datetime, timedelta
import xarray as xr

# ---- Your existing config constants (keep if you want) ----
stations_path = "/home/agkiokas/CAMS/CHINESE_STATIONS_INFO_2015_2023.txt"
base_path = "/mnt/store01/agkiokas/CAMS/inst/subsets"

product = "inst3d"
species = "O3"
date = "20050520"
time = "0300"

species_file = Path(f"{base_path}/{species}/{product}_{species}_{date}_{time}.nc4")
pl_file      = Path(f"{base_path}/PL/{product}_PL_{date}_{time}.nc4")
T_file       = Path(f"{base_path}/T/{product}_T_{date}_{time}.nc4")
orog_file    = Path(f"{base_path}/const/const_2d_asm_Nx_{date}.nc4")
RH_file      = Path(f"{base_path}/RH/{product}_RH_{date}_{time}.nc4")


def build_paths(base_path: str, product: str, species: str, date: str, time: str):
    """
    Build paths for a single timestamp:
      - species file
      - temperature file
      - pressure-level file
      - RH file
      - daily orography file (uses date only)
    """
    sp = Path(f"{base_path}/{species}/{product}_{species}_{date}_{time}.nc4")
    T  = Path(f"{base_path}/T/{product}_T_{date}_{time}.nc4")
    PL = Path(f"{base_path}/PL/{product}_PL_{date}_{time}.nc4")
    RH = Path(f"{base_path}/RH/{product}_RH_{date}_{time}.nc4")
    orog = Path(f"{base_path}/const/const_2d_asm_Nx_{date}.nc4")
    return sp, T, PL, RH, orog


def iter_timestamps(start_dt: datetime, end_dt: datetime, step_minutes: int = 30):
    """
    Yield (date_str, time_str) every step_minutes from start_dt to end_dt inclusive.
    date_str: YYYYMMDD
    time_str: HHMM
    """
    dt = start_dt
    while dt <= end_dt:
        yield dt.strftime("%Y%m%d"), dt.strftime("%H%M")
        dt += timedelta(minutes=step_minutes)


class OrogCache:
    """
    Cache daily orography datasets so we don't reopen them for every 30-min timestep.
    """
    def __init__(self):
        self._cache = {}

    def get(self, orog_path: Path):
        key = str(orog_path)
        if key not in self._cache:
            self._cache[key] = xr.open_dataset(orog_path)
        return self._cache[key]

    def close_all(self):
        for ds in self._cache.values():
            try:
                ds.close()
            except Exception:
                pass
        self._cache.clear()