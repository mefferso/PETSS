from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

import pandas as pd
import yaml

from scripts.coops_fetch import fetch_flood_levels, fetch_recent_water_levels
from scripts.bias import load_bias, save_bias, update_bias
from scripts.petss_web_fetch import fetch_petss_table_text, parse_petss_csv_text


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
    with open("config/stations.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    st = cfg["stations"][0]
    stid = str(st["id"])
    station_name = st["name"]
    datum = st.get("coops_datum", "MLLW")
    tz = st.get("coops_time_zone", "gmt")

    # IMPORTANT: this is your page parameter
    show = st.get("petss_show", "1-1-1-1-0-1-1-1")

    print(f"Fetching PETSS page: {url}")

    raw_text = fetch_petss_table_text(stid)
    rows = parse_petss_csv_text(raw_text)

    df = pd.DataFrame(rows)

    # Parse date like: 02/10 06Z, 02/10 07Z, ...
    # We'll assume CURRENT YEAR from system UTC if not provided.
    year = _utc_now().year
    # Convert to datetime (UTC)
    # Format is "MM/DD HHZ"
    df["valid_time"] = pd.to_datetime(
        df["date_gmt"].astype(str).str.replace("Z", "", regex=False).apply(lambda s: f"{year}/{s}"),
        format="%Y/%m/%d %H",
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["valid_time"])

    # Numeric columns
    for col in ["surge", "tide", "obs", "fcst", "anom", "fcst90", "fcst10"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Thresholds from MDAPI
    fl = fetch_flood_levels(stid)
    if fl.minor is None or fl.moderate is None or fl.major is None:
        raise RuntimeError("Flood thresholds missing from MDAPI (minor/moderate/major).")

    minor_ft, moderate_ft, major_ft = fl.minor, fl.moderate, fl.major
    print(f"Thresholds ft: minor={minor_ft} moderate={moderate_ft} major={major_ft}")

    # Observations (last 6 hours)
    now = _utc_now()
    begin = now - timedelta(hours=6)
    obs_json = fetch_recent_water_levels(
        station_id=stid,
        begin_date_yyyymmdd=_yyyymmdd(begin),
        end_date_yyyymmdd=_yyyymmdd(now),
        datum=datum,
        time_zone=tz,
        units="english",
        interval="6",
    )
    obs_df = _coops_obs_to_df(obs_json)

    # Bias learning using overlap (fcst - obs)
    bias_path = f"data/bias/{stid}_bias.json"
    bias_state = load_bias(bias_path)

    # Use only rows where obs exists on the PETSS table
    if "obs" in df.columns and df["obs"].notna().any():
        overlap = df[df["obs"].notna()].copy()
        # error = forecast - observed
        new_error = float((overlap["fcst"] - overlap["obs"]).mean())
        bias_state = update_bias(bias_state, new_error_ft=new_error)

    # Bias-correct forecast
    df["fcst_bc"] = df["fcst"] - bias_state.rolling_bias_ft

    # Trend blend: simple slope from CO-OPS obs last 6h
    trend_adj = 0.0
    trend_used = False
    if len(obs_df) >= 8:
        x = (obs_df["t"] - obs_df["t"].min()).dt.total_seconds().values
        y = obs_df["v"].values
        denom = (x**2).sum()
        if denom > 0:
            slope = (x * (y - y.mean())).sum() / denom  # ft per sec
            slope_hr = slope * 3600.0  # ft per hour
            trend_adj = float(slope_hr * 6.0)  # project 6h
            trend_used = True

    # Apply trend blend to near-term (next 12h) only
    df["fcst_final"] = df["fcst_bc"]
    horizon = now + timedelta(hours=12)
    if trend_used and abs(trend_adj) <= 2.0:
        w = 0.30
        mask = df["valid_time"] <= horizon
        df.loc[mask, "fcst_final"] = df.loc[mask, "fcst_bc"] + w * trend_adj

    # Probabilities (binary, since we only have mean + percentiles)
    def peak_probs(series: pd.Series, thr: float) -> float:
        return 100.0 if float(series.max()) >= thr else 0.0

    peak_fcst = float(df["fcst_final"].max())
    p_minor = 100.0 if peak_fcst >= minor_ft else 0.0
    p_mod = 100.0 if peak_fcst >= moderate_ft else 0.0
    p_maj = 100.0 if peak_fcst >= major_ft else 0.0

    # Confidence from spread (fcst90 - fcst10) if available
    conf = None
    if "fcst90" in df.columns and "fcst10" in df.columns:
        spread = (df["fcst90"] - df["fcst10"]).median()
        typical_spread = 1.0  # tweak later
        conf = float(max(0.0, min(1.0, 1.0 - (spread / typical_spread))))
    else:
        conf = 0.5

    # Peak timing
    peak_row = df.loc[df["fcst_final"].idxmax()]
    peak_time = peak_row["valid_time"].to_pydatetime().isoformat().replace("+00:00", "Z")

    # Recommendation
    if p_minor == 100.0 and p_mod == 100.0:
        rec = "Warning Possible"
    elif p_minor == 100.0:
        rec = "Advisory Likely"
    else:
        rec = "No Headline Suggested"

    summary = {
        "station_id": stid,
        "station_name": station_name,
        "source_url": url,
        "generated_utc": now.isoformat().replace("+00:00", "Z"),
        "thresholds_ft": {"minor": minor_ft, "moderate": moderate_ft, "major": major_ft},
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
        "peak_fcst_final_ft": round(peak_fcst, 3),
        "peak_time_utc": peak_time,
        "peak_flags_percent": {"minor": p_minor, "moderate": p_mod, "major": p_maj},
        "confidence_index_0to1": conf,
        "recommendation": rec,
    }

    # Timeseries output (for plotting)
    ts = {
        "station_id": stid,
        "station_name": station_name,
        "valid_time_utc": [t.to_pydatetime().isoformat().replace("+00:00", "Z") for t in df["valid_time"]],
        "surge": df["surge"].round(3).tolist() if "surge" in df.columns else [],
        "tide": df["tide"].round(3).tolist() if "tide" in df.columns else [],
        "obs": df["obs"].round(3).tolist() if "obs" in df.columns else [],
        "fcst": df["fcst"].round(3).tolist() if "fcst" in df.columns else [],
        "fcst_final": df["fcst_final"].round(3).tolist(),
        "fcst90": df["fcst90"].round(3).tolist() if "fcst90" in df.columns else [],
        "fcst10": df["fcst10"].round(3).tolist() if "fcst10" in df.columns else [],
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/waveland_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open("outputs/waveland_timeseries.json", "w", encoding="utf-8") as f:
        json.dump(ts, f, indent=2)

    save_bias(bias_path, bias_state)
    print("Done. Wrote outputs/waveland_summary.json and outputs/waveland_timeseries.json")


if __name__ == "__main__":
    main()
