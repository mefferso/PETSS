from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

import pandas as pd
import yaml

from scripts.petss_fetch import find_latest_petss_csv_tar, download_csv_tarball, extract_csvs_from_tarball
from scripts.coops_fetch import fetch_flood_levels, fetch_recent_water_levels
from scripts.bias import load_bias, save_bias, update_bias
from scripts.compute import extract_station_series, compute_exceedance_probs, pick_peak_window


def _utc_now():
    return datetime.now(timezone.utc)


def _yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def _coops_obs_to_df(j: Dict[str, Any]) -> pd.DataFrame:
    # CO-OPS JSON water_level returns "data": [{"t":"...", "v":"..."}]
    data = j.get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna(subset=["t", "v"]).sort_values("t")
    return df


def main():
    # load config
    with open("config/stations.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    st = cfg["stations"][0]  # Waveland only for now
    station_id = str(st["id"])
    station_name = st["name"]
    datum = st.get("coops_datum", "MLLW")
    tz = st.get("coops_time_zone", "gmt")

    # 1) PETSS latest run
    runref = find_latest_petss_csv_tar()
    tar_bytes = download_csv_tarball(runref)
    csvs = extract_csvs_from_tarball(tar_bytes)

    # 2) extract station ensemble series
    petss_df = extract_station_series(csvs, station_id)

    # 3) thresholds
    override = st.get("thresholds_override_ft")
    if override:
        minor_ft = float(override["minor"])
        moderate_ft = float(override["moderate"])
        major_ft = float(override["major"])
        thresholds_source = "override"
    else:
        fl = fetch_flood_levels(station_id)
        if fl.minor is None or fl.moderate is None or fl.major is None:
            raise RuntimeError("MDAPI flood levels missing for this station; set thresholds_override_ft.")
        minor_ft, moderate_ft, major_ft = fl.minor, fl.moderate, fl.major
        thresholds_source = "mdapi"

    # 4) observations (last 6 hours)
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

    # 5) Bias state (rolling mean error)
    bias_path = f"data/bias/{station_id}_bias.json"
    bias_state = load_bias(bias_path)

    # Create verification: compare PETSS mean vs obs over last 3 hours where overlap exists
    # (This is intentionally simple and robust; can upgrade later.)
    if not obs_df.empty:
        # Build PETSS mean time series
        mean_ts = petss_df.groupby("valid_time", as_index=False)["stormtide_ft"].mean().rename(columns={"stormtide_ft": "petss_mean"})
        # match obs times to nearest hour
        obs_hourly = obs_df.copy()
        obs_hourly["valid_time"] = obs_hourly["t"].dt.floor("H")
        obs_hourly = obs_hourly.groupby("valid_time", as_index=False)["v"].mean().rename(columns={"v": "obs_mean"})

        merged = pd.merge(mean_ts, obs_hourly, on="valid_time", how="inner")
        merged = merged[merged["valid_time"] >= (now - timedelta(hours=3))]

        if not merged.empty:
            # error = forecast - observed
            new_error = float((merged["petss_mean"] - merged["obs_mean"]).mean())
            bias_state = update_bias(bias_state, new_error_ft=new_error)

    # Apply bias correction to PETSS
    petss_df["stormtide_ft_bc"] = petss_df["stormtide_ft"] - bias_state.rolling_bias_ft

    # 6) Trend blend (short-term 0–12h)
    trend_adj = 0.0
    trend_used = False
    if len(obs_df) >= 10:
        # simple linear fit on last 6 hours
        x = (obs_df["t"] - obs_df["t"].min()).dt.total_seconds().values
        y = obs_df["v"].values
        # least squares slope
        denom = (x**2).sum()
        if denom > 0:
            slope_ft_per_s = (x * (y - y.mean())).sum() / denom
            slope_ft_per_hr = slope_ft_per_s * 3600.0
            # project 6h forward (conservative)
            trend_adj = float(slope_ft_per_hr * 6.0)
            trend_used = True

    # Apply trend blend: add a fraction of trend_adj to near-term hours only
    petss_df["stormtide_ft_final"] = petss_df["stormtide_ft_bc"]
    horizon_limit = now + timedelta(hours=12)
    if trend_used and abs(trend_adj) <= 2.0:  # guardrail
        w = 0.30  # blending weight
        mask = petss_df["valid_time"] <= horizon_limit
        petss_df.loc[mask, "stormtide_ft_final"] = petss_df.loc[mask, "stormtide_ft_bc"] + w * trend_adj

    # 7) probabilities
    prob_ts = compute_exceedance_probs(
        petss_df.rename(columns={"stormtide_ft_final": "stormtide_ft"}),  # reuse function name
        minor_ft=minor_ft,
        moderate_ft=moderate_ft,
        major_ft=major_ft,
    )

    # confidence: spread-based (p90-p10) normalized by a typical spread constant
    prob_ts["spread_ft"] = prob_ts["p90_ft"] - prob_ts["p10_ft"]
    typical_spread = 1.0  # tweak later per-station
    conf = float(max(0.0, min(1.0, 1.0 - (prob_ts["spread_ft"].median() / typical_spread))))

    # peak window + recommendation
    w_start, w_end, peak_time = pick_peak_window(prob_ts, min_prob=40.0)

    # peak probs
    peak_row = prob_ts.loc[prob_ts["valid_time"] == pd.to_datetime(peak_time.replace("Z", "+00:00"))] if peak_time else prob_ts.iloc[[-1]]
    if not peak_row.empty:
        p_minor = float(peak_row["p_minor"].iloc[0])
        p_mod = float(peak_row["p_moderate"].iloc[0])
        p_maj = float(peak_row["p_major"].iloc[0])
    else:
        p_minor = p_mod = p_maj = 0.0

    if p_minor >= 70 and p_mod >= 60:
        rec = "Warning Likely"
    elif p_minor >= 50 and p_mod >= 40:
        rec = "Warning Possible"
    elif p_minor >= 60:
        rec = "Advisory Likely"
    elif p_minor >= 40:
        rec = "Advisory Possible"
    else:
        rec = "No Headline Suggested"

    summary = {
        "station_id": station_id,
        "station_name": station_name,
        "run_date_dir": runref.date_dir.strip("/"),
        "run_cycle_z": f"{runref.cycle}z",
        "generated_utc": now.isoformat().replace("+00:00", "Z"),
        "thresholds_ft": {"minor": minor_ft, "moderate": moderate_ft, "major": major_ft, "source": thresholds_source},
        "bias": {
            "rolling_bias_ft_forecast_minus_obs": bias_state.rolling_bias_ft,
            "n": bias_state.n,
            "applied": True,
        },
        "trend_blend": {
            "used": trend_used,
            "projected_adjustment_ft_6h": trend_adj if trend_used else 0.0,
            "weight": 0.30 if trend_used else 0.0,
            "applied_horizon_hours": 12 if trend_used else 0,
        },
        "peak_window": {"start_utc": w_start, "end_utc": w_end, "peak_time_utc": peak_time},
        "peak_probs_percent": {"minor": p_minor, "moderate": p_mod, "major": p_maj},
        "confidence_index_0to1": conf,
        "recommendation": rec,
    }

    timeseries = {
        "station_id": station_id,
        "station_name": station_name,
        "valid_time_utc": [t.to_pydatetime().isoformat().replace("+00:00", "Z") for t in prob_ts["valid_time"]],
        "mean_ft": prob_ts["mean_ft"].round(3).tolist(),
        "p10_ft": prob_ts["p10_ft"].round(3).tolist(),
        "p90_ft": prob_ts["p90_ft"].round(3).tolist(),
        "p_minor": prob_ts["p_minor"].round(1).tolist(),
        "p_moderate": prob_ts["p_moderate"].round(1).tolist(),
        "p_major": prob_ts["p_major"].round(1).tolist(),
        "n_members": prob_ts["n_members"].astype(int).tolist(),
        "spread_ft": prob_ts["spread_ft"].round(3).tolist(),
    }

    point_geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "station_id": station_id,
                "name": station_name,
                "recommendation": rec,
                "confidence": conf,
                "p_minor": p_minor,
                "p_moderate": p_mod,
                "p_major": p_maj,
            },
            # You can optionally pull station lat/lon from MDAPI later.
            "geometry": {"type": "Point", "coordinates": [None, None]},
        }]
    }

    # write outputs
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/waveland_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open("outputs/waveland_timeseries.json", "w", encoding="utf-8") as f:
        json.dump(timeseries, f, indent=2)
    with open("outputs/waveland_point.geojson", "w", encoding="utf-8") as f:
        json.dump(point_geojson, f, indent=2)

    # persist bias state
    save_bias(bias_path, bias_state)

    print("Done. Wrote outputs/ files.")


if __name__ == "__main__":
    main()
