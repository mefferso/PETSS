import os
import json
import yaml
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd

# Internal imports (assumes running as python -m scripts.run_waveland)
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
WAVELAND_ID = "8747437" # Hardcoded backup or primary target

def main():
    # 1. Load Configuration
    if os.path.exists(STATIONS_FILE):
        with open(STATIONS_FILE, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"{STATIONS_FILE} not found. Using defaults.")
        config = {"stations": [{"id": WAVELAND_ID}]}

    # Filter for Waveland (or iterate all if you expand later)
    target_station = next((s for s in config['stations'] if str(s['id']) == WAVELAND_ID), None)
    if not target_station:
        raise ValueError(f"Station {WAVELAND_ID} not found in config.")

    logger.info(f"Starting workflow for {target_station['id']}")

    # 2. Fetch Flood Thresholds
    logger.info("Fetching flood levels...")
    levels = coops_api.fetch_flood_levels(WAVELAND_ID)
    
    # Use YAML overrides if API failed or manual override exists
    if not levels.minor: 
        levels.minor = 1.6 # Approximate Waveland minor if API missing
    logger.info(f"Thresholds: Minor={levels.minor}, Mod={levels.moderate}, Major={levels.major}")

    # 3. Download and Parse PETSS Data
    logger.info("Finding latest PETSS tarball...")
    run_ref = petss_web_fetch.find_latest_petss_csv_tar()
    logger.info(f"Downloading {run_ref.csv_tar_url}...")
    tar_bytes = petss_web_fetch.download_csv_tarball(run_ref)
    
    logger.info("Extracting CSVs...")
    all_csvs = petss_web_fetch.extract_csvs_from_tarball(tar_bytes)
    
    logger.info(f"Parsing series for {WAVELAND_ID}...")
    # This returns a DataFrame with columns: [station_id, valid_time, member, stormtide_ft]
    df_petss = coops_fetch.extract_station_series(all_csvs, WAVELAND_ID)

    # 4. Bias Correction
    # Strategy: Fetch last 24h observed data. Compare most recent observation 
    # to the PETSS mean at that specific time.
    bias_path = os.path.join(BIAS_DIR, f"bias_{WAVELAND_ID}.json")
    bias_state = compute.load_bias(bias_path)
    
    now_utc = datetime.now(timezone.utc)
    yesterday_str = (now_utc - timedelta(days=1)).strftime("%Y%m%d")
    today_str = now_utc.strftime("%Y%m%d")
    
    try:
        logger.info("Fetching recent observed water levels for bias calculation...")
        obs_data = coops_api.fetch_recent_water_levels(WAVELAND_ID, yesterday_str, today_str)
        
        # Convert observations to DataFrame
        obs_list = obs_data.get('data', [])
        if obs_list:
            df_obs = pd.DataFrame(obs_list)
            df_obs['t'] = pd.to_datetime(df_obs['t']).dt.tz_localize('UTC') # API returns GMT
            df_obs['v'] = pd.to_numeric(df_obs['v'], errors='coerce')
            
            # Find the latest observation
            last_obs = df_obs.iloc[-1]
            last_obs_time = last_obs['t']
            last_obs_val = last_obs['v']

            # Find PETSS mean at that time (interpolate or nearest hour)
            # PETSS is usually hourly. Let's find the nearest hour in PETSS df
            # Calculate mean of ensemble first
            df_mean = df_petss.groupby('valid_time')['stormtide_ft'].mean()
            
            # Get model value at observation time (using nearest for simplicity)
            # Ensure indices are timezone aware and compatible
            idx_loc = df_mean.index.get_indexer([last_obs_time], method='nearest')
            if idx_loc[0] != -1:
                model_time = df_mean.index[idx_loc[0]]
                model_val = df_mean.iloc[idx_loc[0]]
                
                # Only update bias if the time difference is small (< 90 mins)
                if abs((model_time - last_obs_time).total_seconds()) < 5400:
                    # Bias = Forecast - Observed
                    # If model is 2.0 and obs is 2.5, Bias is -0.5. 
                    # Corrected = Forecast - (-0.5) = 2.5. Correct.
                    current_error = model_val - last_obs_val
                    
                    logger.info(f"Updating Bias. Model: {model_val:.2f}, Obs: {last_obs_val:.2f} (diff: {current_error:.2f})")
                    bias_state = compute.update_bias(bias_state, current_error)
                    compute.save_bias(bias_path, bias_state)
                else:
                    logger.warning("Gap between model and observation too large to update bias.")
            else:
                logger.warning("Could not align model time with observation time.")
    except Exception as e:
        logger.error(f"Failed to update bias: {e}")

    # 5. Apply Bias to Forecast
    logger.info(f"Applying rolling bias: {bias_state.rolling_bias_ft:.2f} ft")
    df_petss['stormtide_ft'] = df_petss['stormtide_ft'] - bias_state.rolling_bias_ft

    # 6. Compute Exceedance Probabilities
    # Ensure thresholds are not None
    t_min = levels.minor if levels.minor else 1.6
    t_mod = levels.moderate if levels.moderate else 2.5 # Default fallback
    t_maj = levels.major if levels.major else 4.0     # Default fallback

    stats = coops_fetch.compute_exceedance_probs(df_petss, t_min, t_mod, t_maj)

    # 7. Generate Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save Raw Data
    output_json = os.path.join(OUTPUT_DIR, "waveland_forecast.json")
    stats['valid_time_str'] = stats['valid_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    stats.to_json(output_json, orient="records", date_format="iso", indent=2)
    
    # Generate Markdown Summary
    md_path = os.path.join(OUTPUT_DIR, "README.md")
    
    start_win, end_win, peak_time = coops_fetch.pick_peak_window(stats, min_prob=30.0)
    
    peak_row = stats.loc[stats['mean_ft'].idxmax()]
    
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
