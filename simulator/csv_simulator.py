import time
import json
import argparse
import random
import requests
import pandas as pd
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Configuration ────────────────────────────────────────────────────────────
RAW_DATA_PATH = "data/raw/UNSW_NB15/UNSW_NB15_testing-set.parquet"
BACKEND_URL   = "http://localhost:8000/predict"
HEALTH_URL    = "http://localhost:8000/health"

# Pool for persistent IPs to trigger reputation logic
PERSISTENT_POOL = [
    "192.168.1.10", "192.168.1.11", "192.168.1.12", 
    "172.16.0.5", "10.0.0.99"
]

def fake_ip() -> str:
    """Generates a random plausible source IP address."""
    return f"{random.randint(1, 254)}.{random.randint(0, 254)}.{random.randint(0, 254)}.{random.randint(1, 254)}"

def run_simulator(interval: float, limit: int, shuffle: bool, attacks_only: bool, loop: bool, targeted: bool, persistent: bool):
    logging.info("Starting CSV Traffic Simulator")
    logging.info(f"Backend URL     : {BACKEND_URL}")
    logging.info(f"Interval        : {interval}s")
    logging.info(f"Max Samples     : {limit if limit > 0 else 'Unlimited'}")
    logging.info(f"Attacks Only    : {attacks_only}")
    logging.info(f"Targeted (High) : {targeted}")
    logging.info(f"Persistent IPs  : {persistent}")

    # 1. Health check
    try:
        resp = requests.get(HEALTH_URL, timeout=2)
        if resp.status_code != 200:
            logging.error("Backend health check failed. Is the server running?")
            return
    except Exception as e:
        logging.error(f"Could not connect to backend: {e}")
        return

    # 2. Load data
    logging.info(f"Loading data from {RAW_DATA_PATH}...")
    try:
        df = pd.read_parquet(RAW_DATA_PATH)
    except Exception as e:
        logging.error(f"Error loading parquet: {e}")
        return
    
    # 3. Filter and Shuffle
    # Targeted Mode: Only send High/Critical attack types
    if targeted:
        high_sev_types = ["worms", "backdoor", "dos", "exploits", "shellcode", "fuzzers"]
        df = df[df["attack_cat"].str.strip().str.lower().isin(high_sev_types)]
        logging.info(f"Targeted mode: Filtered to {len(df):,} high-severity samples.")
    elif attacks_only:
        df = df[df["attack_cat"].str.strip().str.lower() != "normal"]
        logging.info(f"Filtered to {len(df):,} attack-only samples.")
    
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
        logging.info("Data shuffled.")

    # Identify feature columns
    drop_cols = ["attack_cat", "label", "id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    iteration = 0
    count = 0
    
    try:
        while True:
            iteration += 1
            if loop:
                logging.info(f"Starting Pass {iteration} through dataset")
            
            for idx, row in df.iterrows():
                if limit > 0 and count >= limit:
                    logging.info(f"Limit of {limit} samples reached.")
                    return

                # Prepare request
                features = row[feature_cols].to_dict()
                
                # Persistence logic: Use a fixed pool to trigger reputation-based severity
                if persistent:
                    source_ip = random.choice(PERSISTENT_POOL)
                else:
                    source_ip = fake_ip()
                
                payload = {"source_ip": source_ip, "features": features}

                # POST to backend
                try:
                    t0 = time.time()
                    response = requests.post(BACKEND_URL, json=payload, timeout=5)
                    elapsed = time.time() - t0
                    
                    if response.status_code == 200:
                        res = response.json()
                        sev = res.get("severity", "Normal")
                        
                        # ANSI Color Codes
                        COLORS = {
                            "Critical": "\033[95m\033[1m", # Bold Magenta
                            "High": "\033[91m",           # Red
                            "Medium": "\033[93m",         # Yellow
                            "Low": "\033[92m",            # Gray
                            "Normal": "\033[94m"          # Blue
                        }
                        RESET = "\033[0m"
                        
                        color = COLORS.get(sev, "")
                        
                        log_msg = (f"[{count+1:04}] IP: {source_ip:<15} | "
                                   f"Type: {res['attack_type']:<15} | "
                                   f"Severity: {color}{sev:<8}{RESET} | "
                                   f"Time: {elapsed*1000:4.1f}ms")
                        
                        if sev in ["Critical", "High"]:
                            logging.warning(log_msg)
                        else:
                            logging.info(log_msg)
                    else:
                        logging.error(f"[{count+1:04}] Backend error {response.status_code}: {response.text}")
                
                except Exception as e:
                    logging.error(f"[{count+1:04}] Request failed: {e}")

                count += 1
                time.sleep(interval)
            
            if not loop:
                break
                
    except KeyboardInterrupt:
        logging.info("Simulator stopped by user.")

    logging.info(f"Simulation finished. Sent {count} total samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Netwatch IDS Traffic Simulator")
    parser.add_argument("--interval",     type=float, default=0.5, help="Seconds between requests (default: 0.5)")
    parser.add_argument("--limit",        type=int,   default=100, help="Max samples to send total (default: 100, 0 for unlimited)")
    parser.add_argument("--no-shuffle",   dest="shuffle", action="store_false", help="Don't shuffle the dataset")
    parser.add_argument("--attacks-only", action="store_true", help="Send only attack rows")
    parser.add_argument("--targeted",     action="store_true", help="Prioritize high-severity attack categories")
    parser.add_argument("--persistent-ips", action="store_true", help="Use a fixed pool of IPs to trigger reputation")
    parser.add_argument("--loop",         action="store_true", help="Loop through the dataset")
    parser.set_defaults(shuffle=True)

    args = parser.parse_args()
    run_simulator(
        interval=args.interval, 
        limit=args.limit, 
        shuffle=args.shuffle, 
        attacks_only=args.attacks_only, 
        loop=args.loop,
        targeted=args.targeted,
        persistent=args.persistent_ips
    )
