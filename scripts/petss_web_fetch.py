from __future__ import annotations

import re
from typing import List, Dict, Optional
import requests


def build_petss_url(stid: str, datum: str, show: str) -> str:
    # stid, datum, show are exactly what you see in your URL
    return f"https://slosh.nws.noaa.gov/petss/index.php?stid={stid}&datum={datum}&show={show}"


def fetch_petss_table_text(url: str, timeout: int = 45) -> str:
    """
    Downloads the PETSS HTML page and extracts the embedded text table
    (the same one you see under the plot).

    Returns raw table text including header and comma-separated rows.
    """
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    html = r.text

    # The data table is typically inside a <textarea> element.
    # We'll extract the first large-ish textarea that contains "Date(GMT)"
    # and commas.
    textareas = re.findall(r"<textarea[^>]*>(.*?)</textarea>", html, flags=re.DOTALL | re.IGNORECASE)
    if not textareas:
        raise RuntimeError("No <textarea> found in PETSS page HTML.")

    # Unescape a few common HTML entities
    def _clean(s: str) -> str:
        return (
            s.replace("&gt;", ">")
             .replace("&lt;", "<")
             .replace("&amp;", "&")
             .strip()
        )

    candidates = [_clean(t) for t in textareas]
    for t in candidates:
        if "Date(GMT)" in t and "," in t:
            return t

    # Fallback: return the largest textarea
    biggest = max(candidates, key=len)
    return biggest


def parse_petss_csv_text(raw: str) -> List[Dict]:
    """
    Parses the PETSS embedded CSV-ish text block into a list of dict rows.

    Expected header like:
      Date(GMT), Surge, Tide, Obs, Fcst, Anom, Fcst90%, Fcst10%

    Returns list of dicts with keys normalized to:
      date_gmt, surge, tide, obs, fcst, anom, fcst90, fcst10
    """
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    # Find header line containing Date(GMT)
    header_idx = None
    for i, ln in enumerate(lines):
        if "Date(GMT)" in ln:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not find header line containing 'Date(GMT)' in extracted text.")

    header = [h.strip() for h in lines[header_idx].split(",")]
    # Normalize header names
    def norm(h: str) -> str:
        h = h.strip().lower()
        h = h.replace("date(gmt)", "date_gmt")
        h = h.replace("%", "")
        h = h.replace("fcst90", "fcst90")
        h = h.replace("fcst10", "fcst10")
        h = h.replace(" ", "")
        return h

    nheader = [norm(h) for h in header]

    rows = []
    for ln in lines[header_idx + 1:]:
        # stop if we hit something that isn't CSV-ish
        if "," not in ln:
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(nheader):
            # some lines can be malformed; skip them
            continue
        row = dict(zip(nheader, parts))
        rows.append(row)

    if not rows:
        raise RuntimeError("Parsed 0 data rows from PETSS extracted text.")
    return rows
