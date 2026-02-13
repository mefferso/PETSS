from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

import pandas as pd
import yaml

from scripts.petss_fetch import (
    find_latest_petss_csv_tar,
    download_csv_tarball,
    extract_csvs_from_tarball,
)
from scripts.coops_fetch import (
    fetch_flood_levels,
    fetch_recent_water_levels,
)
from scripts.bias import load_bias, save_bias, update_bias
from scripts.compute import (
    extract_station_series,
    compute_exceedance_probs,
    pick_peak_window,
    debug_inventory,
)


def _utc_now():
    return datetime.now(timezone.utc)


def _yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def _coops_obs_to_df(j: Dict[str, Any]) -> pd.DataFrame:
    data = j.get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna(subset=["t", "v"]).sort_values("t")
    return df


def main():
    # Load station config
    with open("config/stations.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    st = cfg["stations"][0]
    station_id = str(st["id"])
    station_name = st["name"]
    datum = st.get("coops_datum", "MLLW")
    tz = st.get("coops_time_zone", "gmt")

    print(f"Running PETSS engine for station {station_id} ({station_name})")

    # --------------------------------------------------
    # 1) PETSS latest run
    # --------------------------------------------------
    runref = find_latest_petss_csv_tar()
    print(f"Found PETSS run: {runref.date_dir} cycle {runref.cycle}z")

    tar_bytes = download_csv_tarball(runref)
    csvs = extract_csvs_from_tarball(tar_bytes)

    print("DEBUG INVENTORY JSON START")
    print(json.dumps(debug_inventory(csvs, max_files=10), indent=2))
    print("DEBUG INVENTORY JSON END")

    # --------------------------------------------------
    # 2) Extract station series (this is where it fails now)
    # --------------------------------------------------
    petss_df = extract_station_series(csvs, station_id)

    # --------------------------------------------------
    # 3) Flood thresholds
    # --------------------------------------------------
    fl = fetch_flood_levels(station_id)
    if fl.minor is None or fl.moderate is None or fl.major is None:
        raise RuntimeError("Flood thresholds missing from MDAPI.")

    minor_ft = fl.minor
    moderate_ft = fl.moderate
    major_ft = fl.major

    print(
        f"Flood thresholds (ft) — Minor: {minor_ft}, "
        f"Moderate: {moderate_ft}, Major: {major_ft}"
    )

    # --------------------------------------------------
    # 4) Observations
    # --------------------------------------------------
    now = _utc_now()
    begin = now - timedelta(hours=6)

    obs_json = fetch_recent_water_levels(
        station_id=station_id,
        begin_date_yyyymmdd=_yyyymmdd(begin),
        end_date_yyyymmdd=_yyyymmdd(now),
        datum=datum,
        time_zone=tz,
        units="english",
        interval="6",
    )

    obs_df = _coops_obs_to_df(obs_json)

    # --------------------------------------------------
    # 5) Bias handling
    # --------------------------------------------------
    bias_path = f"data/bias/{station_id}_bias.json"
    bias_state = load_bias(bias_path)

    if not obs_df.empty:
        mean_ts = (
            petss_df.groupby("valid_time", as_index=False)["stormtide_ft"]
            .mean()
            .rename(columns={"stormtide_ft": "petss_mean"})
        )

        obs_hourly = obs_df.copy()
        obs_hourly["valid_time"] = obs_hourly["t"].dt.floor("H")
        obs_hourly = (
            obs_hourly.groupby("valid_time", as_index=False)["v"]
            .mean()
            .rename(columns={"v": "obs_mean"})
        )

        merged = pd.merge(mean_ts, obs_hourly, on="valid_time", how="inner")
        merged = merged[merged["valid_time"] >= (now - timedelta(hours=3))]

        if not merged.empty:
            new_error = float((merged["petss_mean"] - merged["obs_mean"]).mean())
            bias_state = update_bias(bias_state, new_error_ft=new_error)

    petss_df["stormtide_ft_bc"] = (
        petss_df["stormtide_ft"] - bias_state.rolling_bias_ft
    )

    # --------------------------------------------------
    # 6) Probabilities
    # --------------------------------------------------
    prob_ts = compute_exceedance_probs(
        petss_df.rename(columns={"stormtide_ft_bc": "stormtide_ft"}),
        minor_ft=minor_ft,
        moderate_ft=moderate_ft,
        major_ft=major_ft,
    )

    w_start, w_end, peak_time = pick_peak_window(prob_ts)

    summary = {
        "station_id": station_id,
        "station_name": station_name,
        "run_date_dir": runref.date_dir.strip("/"),
        "run_cycle_z": f"{runref.cycle}z",
        "generated_utc": now.isoformat().replace("+00:00", "Z"),
        "peak_time_utc": peak_time,
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/waveland_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_bias(bias_path, bias_state)

    print("Done.")


if __name__ == "__main__":
    main()
