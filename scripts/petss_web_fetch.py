from __future__ import annotations

import re
from typing import List, Dict
import requests


def build_petss_url(stid: str, datum: str, show: str) -> str:
    return f"https://slosh.nws.noaa.gov/petss/index.php?stid={stid}&datum={datum}&show={show}"


def _strip_html_to_text(html: str) -> str:
    """
    Very lightweight HTML-to-text:
    - convert <br> and </tr> to newlines
    - remove tags
    - unescape a few entities
    """
    s = html

    # normalize newlines from common tags
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</tr\s*>", "\n", s)
    s = re.sub(r"(?i)</p\s*>", "\n", s)
    s = re.sub(r"(?i)</div\s*>", "\n", s)

    # remove all tags
    s = re.sub(r"<[^>]+>", "", s)

    # unescape minimal entities
    s = (
        s.replace("&nbsp;", " ")
         .replace("&gt;", ">")
         .replace("&lt;", "<")
         .replace("&amp;", "&")
    )

    # collapse ugly whitespace a bit
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def fetch_petss_table_text(url: str, timeout: int = 45) -> str:
    """
    Downloads the PETSS HTML page and extracts the embedded comma-separated data table
    (the same one shown under the plot).

    Strategy:
    1) Use browser-like headers (User-Agent) so server returns normal HTML.
    2) Try to extract from <textarea> (if present).
    3) Otherwise, strip HTML -> text and locate the block starting with 'Date(GMT)'.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    html = r.text

    # 1) Try textarea path (some versions use it)
    textareas = re.findall(r"(?is)<textarea[^>]*>(.*?)</textarea>", html)
    if textareas:
        for t in textareas:
            if "Date(GMT)" in t and "," in t:
                return t.strip()

    # 2) Strip HTML and search for a CSV-ish block beginning at Date(GMT)
    text = _strip_html_to_text(html)

    # Find the first occurrence of header
    idx = text.find("Date(GMT)")
    if idx == -1:
        # last resort: sometimes it shows as Date(GMT),Surge,... with no space
        m = re.search(r"Date\(GMT\)[^\n]*", text)
        if not m:
            # Helpful debug output in logs (shortened)
            raise RuntimeError(
                "Could not find 'Date(GMT)' in page text. "
                "Page might be JS-rendered or blocked. "
                f"First 400 chars of text: {text[:400]!r}"
            )
        idx = m.start()

    # Take a chunk after the header and keep only lines that look CSV-ish
    chunk = text[idx: idx + 20000]  # plenty
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]

    # Keep lines that contain commas (table rows)
    csv_lines = [ln for ln in lines if "," in ln]

    # We need at least a header + a few rows
    if len(csv_lines) < 5:
        raise RuntimeError(
            "Found 'Date(GMT)' but did not find enough comma-separated rows. "
            f"Sample lines: {csv_lines[:5]}"
        )

    # Return as newline-separated block
    return "\n".join(csv_lines)


def parse_petss_csv_text(raw: str) -> List[Dict]:
    """
    Parses the PETSS extracted comma-separated block into dict rows.

    Expected header like:
      Date(GMT), Surge, Tide, Obs, Fcst, Anom, Fcst90%, Fcst10%

    Returns list of dicts with keys:
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

    def norm(h: str) -> str:
        h = h.strip().lower()
        h = h.replace("date(gmt)", "date_gmt")
        h = h.replace("fcst90%", "fcst90")
        h = h.replace("fcst10%", "fcst10")
        h = h.replace("%", "")
        h = h.replace(" ", "")
        return h

    nheader = [norm(h) for h in header]

    rows = []
    for ln in lines[header_idx + 1:]:
        if "," not in ln:
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(nheader):
            continue
        row = dict(zip(nheader, parts))
        rows.append(row)

    if not rows:
        raise RuntimeError("Parsed 0 data rows from PETSS extracted text.")
    return rows
