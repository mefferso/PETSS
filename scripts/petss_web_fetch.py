from __future__ import annotations

import requests


PETSS_DATA_URL = "https://slosh.nws.noaa.gov/petss/fixed/php/getData.php"


def fetch_petss_table_text(station_id: str, timeout: int = 45) -> str:
    """
    Directly fetch PETSS hydrograph table text using the internal data endpoint.
    Returns raw comma-separated text block.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://slosh.nws.noaa.gov/petss/index.php?stid={station_id}"
    }

    r = requests.post(
        PETSS_DATA_URL,
        data={"st": station_id},
        headers=headers,
        timeout=timeout
    )

    r.raise_for_status()

    text = r.text.strip()

    if "Date(GMT)" not in text:
        raise RuntimeError(
            f"Unexpected PETSS response. First 300 chars:\n{text[:300]}"
        )

    return text


def parse_petss_csv_text(raw: str):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    header = lines[0].split(",")
    header = [h.strip().lower().replace("%", "").replace(" ", "") for h in header]

    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))

    return rows
