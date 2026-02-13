import os
import json
import yaml
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd

# Internal imports
from scripts import petss_web_fetch
from scripts import coops_fetch
from scripts import coops_api
from scripts import compute

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
STATIONS_FILE = "stations.yml"
BIAS_DIR = "data/bias"
OUTPUT_DIR = "outputs"
# We default to Waveland IDs if config is missing
DEFAULT_COOPS_ID = "8747437"
DEFAULT_PETSS_ID = "WVLM6"

def main():
    # 1. Load Configuration
    if os.path.exists(STATIONS_FILE):
        with open(STATIONS_FILE, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"{STATIONS_FILE} not found. Using defaults.")
        config = {"stations": [{"id": DEFAULT_COOPS_ID, "petss_id": DEFAULT_PETSS_ID}]}

    # Find configuration for our target station
    station_cfg = next((s for s in config['stations'] if str(s['id']) == DEFAULT_COOPS_ID), None)
    if not station_cfg:
        # Fallback if the specific ID isn't in the YAML
        station_cfg = {"id": DEFAULT_COOPS_ID, "petss_id": DEFAULT_PETSS_ID}
    
    coops_id = str(station_cfg['id'])
    # Use petss_id if it exists, otherwise fallback to coops_id
    petss_id = str(station_cfg.get('petss_id', coops_id))

    logger.info(f"Starting workflow for CO-OPS: {coops_id} | PETSS: {petss_id}")

    # 2. Fetch Flood Thresholds (Uses CO-OPS Numeric ID)
    logger.info("Fetching flood levels...")
    levels = coops_api.fetch_flood_levels(coops_id)
    
    if not levels.minor: 
        levels.minor = 1.6 
    logger.info(f"Thresholds: Minor={levels.minor}, Mod={levels.moderate}, Major={levels.major}")

    # 3. Download and Parse PETSS Data (Uses NWS String ID)
    logger.info("Finding latest PETSS tarball...")
    run_ref = petss_web_fetch.find_latest_petss_csv_tar()
    logger.info(f"Downloading {run_ref.csv_tar_url}...")
    tar_bytes = petss_web_fetch.download_csv_tarball(run_ref)
    
    logger.info("Extracting CSVs...")
    all_csvs = petss_web_fetch.extract_csvs_from_tarball(tar_bytes)
    
    logger.info(f"Parsing series for PETSS ID: {petss_id}...")
    df_petss = coops_fetch.extract_station_series(all_csvs, petss_id)

    if df_petss.empty:
        logger.error(f"No data found for station {petss_id} in any downloaded CSV.")
        logger.error("Check station ID mapping or PETSS availability.")
        return # Exit gracefully
    
    logger.info(f"Found {len(df_petss)} data points.")

    # 4. Bias Correction (Uses CO-OPS Observed Data)
    bias_path = os.path.join(BIAS_DIR, f"bias_{coops_id}.json")
    bias_state = compute.load_bias(bias_path)
    
    now_utc = datetime.now(timezone.utc)
    yesterday_str = (now_utc - timedelta(days=1)).strftime("%Y%m%d")
    today_str = now_utc.strftime("%Y%m%d")
    
    try:
        logger.info(f"Fetching recent observed water levels for {coops_id}...")
        obs_data = coops_api.fetch_recent_water_levels(coops_id, yesterday_str, today_str)
        
        obs_list = obs_data.get('data', [])
        if obs_list:
            df_obs = pd.DataFrame(obs_list)
            df_obs['t'] = pd.to_datetime(df_obs['t']).dt.tz_localize('UTC') 
            df_obs['v'] = pd.to_numeric(df_obs['v'], errors='coerce')
            
            last_obs = df_obs.iloc[-1]
            last_obs_time = last_obs['t']
            last_obs_val = last_obs['v']

            df_mean = df_petss.groupby('valid_time')['stormtide_ft'].mean()
            
            idx_loc = df_mean.index.get_indexer([last_obs_time], method='nearest')
            if idx_loc[0] != -1:
                model_time = df_mean.index[idx_loc[0]]
                model_val = df_mean.iloc[idx_loc[0]]
                
                if abs((model_time - last_obs_time).total_seconds()) < 5400:
                    current_error = model_val - last_obs_val
                    logger.info(f"Updating Bias. Model: {model_val:.2f}, Obs: {last_obs_val:.2f} (diff: {current_error:.2f})")
                    bias_state = compute.update_bias(bias_state, current_error)
                    compute.save_bias(bias_path, bias_state)
                else:
                    logger.warning("Gap between model and observation too large to update bias.")
    except Exception as e:
        logger.error(f"Bias update failed (continuing without update): {e}")

    # 5. Apply Bias
    logger.info(f"Applying rolling bias: {bias_state.rolling_bias_ft:.2f} ft")
    df_petss['stormtide_ft'] = df_petss['stormtide_ft'] - bias_state.rolling_bias_ft

    # 6. Compute Stats
    t_min = levels.minor if levels.minor else 1.6
    t_mod = levels.moderate if levels.moderate else 2.5
    t_maj = levels.major if levels.major else 4.0

    stats = coops_fetch.compute_exceedance_probs(df_petss, t_min, t_mod, t_maj)
    
    if stats.empty:
         logger.warning("Forecast stats empty. Skipping output.")
         return

    # 7. Generate Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_json = os.path.join(OUTPUT_DIR, "waveland_forecast.json")
    stats['valid_time_str'] = stats['valid_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    stats.to_json(output_json, orient="records", date_format="iso", indent=2)
    
    md_path = os.path.join(OUTPUT_DIR, "README.md")
    
    start_win, end_win, peak_time = coops_fetch.pick_peak_window(stats, min_prob=30.0)
    
    if not peak_time:
         peak_idx = stats['mean_ft'].idxmax()
         peak_row = stats.loc[peak_idx]
    else:
         peak_row = stats[stats['valid_time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ') == peak_time].iloc[0]

    with open(md_path, "w") as f:
        f.write(f"# Waveland, MS Flood Forecast\n\n")
        f.write(f"**Updated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Data Source:** PETSS {run_ref.date_dir} Cycle {run_ref.cycle}Z\n")
        f.write(f"**Bias Applied:** {bias_state.rolling_bias_ft:.2f} ft (n={bias_state.n})\n\n")
        
        f.write("## Peak Forecast\n")
        f.write(f"- **Peak Time:** {peak_row['valid_time_str']}\n")
        f.write(f"- **Mean Water Level:** {peak_row['mean_ft']:.2f} ft MLLW\n")
        f.write(f"- **10% - 90% Range:** {peak_row['p10_ft']:.2f} ft to {peak_row['p90_ft']:.2f} ft\n\n")
        
        f.write("## Flood Risk Probabilities\n")
        f.write("| Threshold | Level (ft) | Probability |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| Minor | {t_min} | {peak_row['p_minor']:.1f}% |\n")
        f.write(f"| Moderate | {t_mod} | {peak_row['p_moderate']:.1f}% |\n")
        f.write(f"| Major | {t_maj} | {peak_row['p_major']:.1f}% |\n")

    logger.info("Workflow completed successfully.")

if __name__ == "__main__":
    main()
