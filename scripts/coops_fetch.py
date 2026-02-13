from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import pandas as pd


@dataclass
class StationOutputs:
    summary: Dict
    timeseries: Dict
    point_geojson: Dict


def _parse_any_petss_csv(csv_bytes: bytes) -> pd.DataFrame:
    """
    We don't assume exact PETSS CSV schema; we load with pandas and normalize columns.
    Must produce at least: station_id, valid_time, member, stormtide_ft
    """
    # best effort decode
    text = csv_bytes.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text))

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # find likely columns
    # station id
    st_col = None
    for cand in ["station", "station_id", "stationid", "id"]:
        if cand in df.columns:
            st_col = cand
            break
    if st_col is None:
        raise RuntimeError(f"Cannot find station column in CSV columns={df.columns.tolist()}")

    # valid time
    vt_col = None
    for cand in ["valid_time", "valid", "time", "datetime", "date_time", "date"]:
        if cand in df.columns:
            vt_col = cand
            break
    if vt_col is None:
        raise RuntimeError(f"Cannot find time column in CSV columns={df.columns.tolist()}")

    # member
    mem_col = None
    for cand in ["member", "ens", "ensemble", "gefs_member", "e"]:
        if cand in df.columns:
            mem_col = cand
            break
    if mem_col is None:
        # sometimes only a single series exists; treat as member "mean"
        df["member"] = "mean"
        mem_col = "member"

    # storm tide / water level value
    val_col = None
    for cand in ["stormtide", "storm_tide", "stormtide_ft", "water_level", "value", "wl"]:
        if cand in df.columns:
            val_col = cand
            break
    if val_col is None:
        # last resort: pick first numeric column not matching common metadata
        numeric_cols = [c for c in df.columns if c not in {st_col, vt_col, mem_col}]
        # pick one that looks numeric
        for c in numeric_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                val_col = c
                break
    if val_col is None:
        raise RuntimeError("Cannot identify storm tide value column in PETSS CSV.")

    out = df[[st_col, vt_col, mem_col, val_col]].copy()
    out = out.rename(
        columns={
            st_col: "station_id",
            vt_col: "valid_time",
            mem_col: "member",
            val_col: "stormtide_raw",
        }
    )

    # parse time
    out["valid_time"] = pd.to_datetime(out["valid_time"], errors="coerce", utc=True)
    out = out.dropna(subset=["valid_time"])

    # numeric values
    out["stormtide_raw"] = pd.to_numeric(out["stormtide_raw"], errors="coerce")
    out = out.dropna(subset=["stormtide_raw"])

    # Detect tenths-of-feet scaling: if most values are integer-ish and abs max > 20, probably tenths.
    vals = out["stormtide_raw"].values
    if len(vals) > 0:
        is_intish = (abs(vals - vals.round()) < 1e-6).mean() > 0.8
        if is_intish and (abs(vals).max() > 20.0):
            out["stormtide_ft"] = out["stormtide_raw"] / 10.0
            out["scaling"] = "tenths_ft"
        else:
            out["stormtide_ft"] = out["stormtide_raw"]
            out["scaling"] = "ft"
    else:
        out["stormtide_ft"] = out["stormtide_raw"]
        out["scaling"] = "unknown"

    return out[["station_id", "valid_time", "member", "stormtide_ft", "scaling"]]


def extract_station_series(all_csvs: Dict[str, bytes], station_id: str) -> pd.DataFrame:
    """
    Iterate through all CSVs in the tarball; concatenate rows matching station_id.
    """
    frames = []
    for name, b in all_csvs.items():
        try:
            df = _parse_any_petss_csv(b)
        except Exception:
            continue
        # Convert both to strings to ensure matching works
        sub = df[df["station_id"].astype(str) == str(station_id)].copy()
        if not sub.empty:
            sub["source_csv"] = name
            frames.append(sub)

    if not frames:
        # Fallback empty dataframe if nothing found, to prevent crash
        return pd.DataFrame(columns=["station_id", "valid_time", "member", "stormtide_ft"])

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("valid_time")
    return out


def compute_exceedance_probs(
    df: pd.DataFrame,
    minor_ft: float,
    moderate_ft: float,
    major_ft: float,
) -> pd.DataFrame:
    """
    Returns hourly time series with probabilities (0-100) for each threshold.
    """
    if df.empty:
        return pd.DataFrame()

    # group by valid_time, compute exceedance fraction by member
    def prob_exceed(group: pd.DataFrame, thr: float) -> float:
        return 100.0 * (group["stormtide_ft"] >= thr).mean()

    grouped = df.groupby("valid_time", as_index=False)

    out = grouped.apply(lambda g: pd.Series({
        "p_minor": prob_exceed(g, minor_ft),
        "p_moderate": prob_exceed(g, moderate_ft),
        "p_major": prob_exceed(g, major_ft),
        "mean_ft": g["stormtide_ft"].mean(),
        "p10_ft": g["stormtide_ft"].quantile(0.10),
        "p90_ft": g["stormtide_ft"].quantile(0.90),
        "n_members": int(g.shape[0]),
    })).reset_index(drop=True)

    return out.sort_values("valid_time")


def pick_peak_window(prob_ts: pd.DataFrame, min_prob: float = 40.0) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Peak window where p_minor >= min_prob.
    Return ISO timestamps (UTC) for start, end, and peak_time (max mean_ft within window).
    """
    if prob_ts.empty:
        return None, None, None

    mask = prob_ts["p_minor"] >= min_prob
    if not mask.any():
        # fallback: peak at max mean
        idx = prob_ts["mean_ft"].idxmax()
        peak = prob_ts.loc[idx, "valid_time"].to_pydatetime().isoformat().replace("+00:00", "Z")
        return None, None, peak

    sel = prob_ts[mask].copy()
    start = sel["valid_time"].min().to_pydatetime().isoformat().replace("+00:00", "Z")
    end = sel["valid_time"].max().to_pydatetime().isoformat().replace("+00:00", "Z")

    idx = sel["mean_ft"].idxmax()
    peak = sel.loc[idx, "valid_time"].to_pydatetime().isoformat().replace("+00:00", "Z")
    return start, end, peak
