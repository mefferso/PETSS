from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml

from .coops_api import fetch_flood_levels, fetch_recent_water_levels
from .petss_fetch import (
    find_latest_petss_csv_tar,
    download_csv_tarball,
    extract_csvs_from_tarball,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def decode_bytes(b: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            pass
    return b.decode("utf-8", errors="replace")


def load_station_cfg(stations_path: Path) -> Dict[str, Any]:
    if not stations_path.exists():
        raise FileNotFoundError(f"Missing {stations_path}. Expected stations.yml in repo root or config/.")

    cfg = yaml.safe_load(stations_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or "stations" not in cfg:
        raise ValueError("stations.yml must have top-level key: stations:")

    stations = cfg.get("stations") or []
    if not stations:
        raise ValueError("stations.yml has no stations entries.")

    st = stations[0]  # Waveland-only
    for k in ("id", "petss_id", "name"):
        if k not in st:
            raise ValueError(f"stations.yml missing required field: {k}")
    return st


def coops_recent_df(station_id: str, datum: str, tz: str) -> pd.DataFrame:
    end = utc_now()
    begin = end - timedelta(days=2)

    j = fetch_recent_water_levels(
        station_id=station_id,
        begin_date_yyyymmdd=yyyymmdd(begin),
        end_date_yyyymmdd=yyyymmdd(end),
        datum=datum,
        time_zone=tz,
        units="english",
        interval="6",
        timeout=30,
    )

    data = j.get("data", []) if isinstance(j, dict) else []
    df = pd.DataFrame(data)
    if df.empty:
        return df

    if "t" in df.columns:
        df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    if "v" in df.columns:
        df["v"] = pd.to_numeric(df["v"], errors="coerce")

    return df


def try_filter_petss_for_station(csv_bytes: bytes, petss_id: str) -> Optional[pd.DataFrame]:
    """
    Best-effort filter for station:
      - quick string check
      - parse CSV (assumes comma-delimited)
      - filter by likely station column if present
      - else filter rows where any cell contains petss_id
    """
    text = decode_bytes(csv_bytes)
    if petss_id not in text:
        return None

    try:
        df = pd.read_csv(pd.io.common.StringIO(text))
    except Exception:
        return None

    if df.empty:
        return None

    station_cols = [c for c in df.columns if str(c).strip().lower() in {
        "station", "station_id", "stid", "id", "gauge", "gauge_id", "petss_id", "site", "site_id"
    }]

    for c in station_cols:
        try:
            m = df[df[c].astype(str) == petss_id]
            if not m.empty:
                return m
        except Exception:
            continue

    try:
        mask = df.astype(str).apply(lambda col: col.str.contains(petss_id, na=False))
        m = df.loc[mask.any(axis=1)]
        return m if not m.empty else None
    except Exception:
        return None


def find_csv_containing_station(csv_map: Dict[str, bytes], petss_id: str) -> Optional[Tuple[str, bytes]]:
    """
    Fast deterministic: pick the first CSV in the tarball that literally contains the station ID string.
    This is what fixes your 'way tf off' issue (wrong station file selected).
    """
    needle = petss_id.encode("utf-8")
    for fname, b in csv_map.items():
        if needle in b:
            return fname, b
    return None


def main() -> None:
    repo_root = Path(".").resolve()

    stations_path = (
        (repo_root / "stations.yml")
        if (repo_root / "stations.yml").exists()
        else (repo_root / "config" / "stations.yml")
    )

    station = load_station_cfg(stations_path)
    coops_id = str(station["id"])
    petss_id = str(station["petss_id"])
    name = str(station["name"])
    datum = str(station.get("coops_datum", "MLLW"))
    tz = str(station.get("coops_time_zone", "gmt"))

    out_root = repo_root / "outputs"
    out_waveland = out_root / "waveland"
    safe_mkdir(out_waveland)

    # CO-OPS flood levels
    flood = fetch_flood_levels(coops_id)

    # CO-OPS recent obs
    df_obs = coops_recent_df(coops_id, datum=datum, tz=tz)
    obs_csv_path = out_waveland / "coops_recent.csv"
    if not df_obs.empty:
        df_obs.to_csv(obs_csv_path, index=False)
    else:
        obs_csv_path.write_text("t,v\n", encoding="utf-8")

    obs_last: Optional[float] = None
    obs_max_48h: Optional[float] = None
    obs_time_last: Optional[str] = None
    if not df_obs.empty and "v" in df_obs.columns:
        s = df_obs.dropna(subset=["v"])
        if not s.empty:
            obs_last = float(s.iloc[-1]["v"])
            obs_max_48h = float(s["v"].max())
            if "t" in s.columns and pd.notna(s.iloc[-1]["t"]):
                obs_time_last = str(pd.to_datetime(s.iloc[-1]["t"], utc=True))

    # PETSS latest tarball
    runref = find_latest_petss_csv_tar()
    tar_bytes = download_csv_tarball(runref)
    csv_map = extract_csvs_from_tarball(tar_bytes)

    # Write raw PETSS CSVs
    raw_dir = out_root / "petss_raw" / runref.date_dir.strip("/").replace(".", "_") / f"t{runref.cycle}z"
    safe_mkdir(raw_dir)

    raw_files = []
    for fname, b in csv_map.items():
        safe_name = fname.replace("\\", "/").split("/")[-1]
        p = raw_dir / safe_name
        p.write_bytes(b)
        raw_files.append(str(p.relative_to(repo_root)))

    # ✅ NEW: write the actual station CSV used for display (contains WVLM6)
    station_csv_path = out_waveland / "petss_station.csv"
    station_source_file = None
    hit = find_csv_containing_station(csv_map, petss_id=petss_id)
    if hit is not None:
        station_source_file, station_bytes = hit
        station_csv_path.write_bytes(station_bytes)
    else:
        # still create the file so downstream doesn't blow up
        station_csv_path.write_text("", encoding="utf-8")

    # Optional: keep the "matches" file too (handy for debugging schema)
    matches = []
    for fname, b in csv_map.items():
        m = try_filter_petss_for_station(b, petss_id=petss_id)
        if m is not None and not m.empty:
            m = m.copy()
            m.insert(0, "_source_file", fname)
            matches.append(m)

    match_csv_path = out_waveland / "petss_matches.csv"
    match_row_count = 0
    match_cols = []
    if matches:
        dfm = pd.concat(matches, ignore_index=True)
        dfm.to_csv(match_csv_path, index=False)
        match_row_count = int(dfm.shape[0])
        match_cols = [str(c) for c in dfm.columns]
    else:
        match_csv_path.write_text("", encoding="utf-8")

    summary = {
        "generated_utc": utc_now().isoformat(),
        "station": {
            "name": name,
            "coops_id": coops_id,
            "petss_id": petss_id,
            "datum": datum,
            "time_zone": tz,
        },
        "coops_flood_levels_ft": asdict(flood),
        "coops_recent": {
            "last_value_ft": obs_last,
            "last_time_utc": obs_time_last,
            "max_last_48h_ft": obs_max_48h,
            "csv_path": str(obs_csv_path.relative_to(repo_root)),
        },
        "petss_latest": {
            "date_dir": runref.date_dir,
            "cycle": runref.cycle,
            "csv_tar_url": runref.csv_tar_url,
            "raw_csv_count": len(csv_map),
            "raw_files_written": raw_files[:120],
            "station_match_csv": str(match_csv_path.relative_to(repo_root)),
            "station_match_rows": match_row_count,
            "station_match_columns": match_cols[:120],
        },
        # ✅ NEW: definitive station CSV pointer
        "petss_station_csv": {
            "path": str(station_csv_path.relative_to(repo_root)),
            "source_file": station_source_file,
            "petss_id": petss_id,
        },
    }

    write_json(out_waveland / "latest.json", summary)

    print("Wrote outputs:")
    print(f" - {out_waveland / 'latest.json'}")
    print(f" - {obs_csv_path}")
    print(f" - {station_csv_path}  (source: {station_source_file})")
    print(f" - {match_csv_path}")
    print(f" - raw PETSS csvs under: {raw_dir}")


if __name__ == "__main__":
    main()
