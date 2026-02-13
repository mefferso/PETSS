from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import requests


COOPS_DATA_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
COOPS_MDAPI = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"


@dataclass
class FloodLevelsFt:
    minor: Optional[float]
    moderate: Optional[float]
    major: Optional[float]


def fetch_flood_levels(station_id: str, timeout: int = 30) -> FloodLevelsFt:
    url = f"{COOPS_MDAPI}/stations/{station_id}/floodlevels.json"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        j = r.json()
    except Exception:
        return FloodLevelsFt(None, None, None)

    node = j.get("floodLevel", j)

    def _to_float(x):
        try:
            return float(x) if x is not None else None
        except Exception:
            return None

    return FloodLevelsFt(
        minor=_to_float(node.get("nos_minor")),
        moderate=_to_float(node.get("nos_moderate")),
        major=_to_float(node.get("nos_major")),
    )


def fetch_recent_water_levels(
    station_id: str,
    begin_date_yyyymmdd: str,
    end_date_yyyymmdd: str,
    datum: str = "MLLW",
    time_zone: str = "gmt",
    units: str = "english",
    interval: str = "6",
    timeout: int = 30,
) -> Dict:
    params = {
        "product": "water_level",
        "application": "coastal-flood-intel",
        "format": "json",
        "station": station_id,
        "begin_date": begin_date_yyyymmdd,
        "end_date": end_date_yyyymmdd,
        "datum": datum,
        "time_zone": time_zone,
        "units": units,
        "interval": interval,
    }
    r = requests.get(COOPS_DATA_API, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()
