"""
TB Breath Analyzer - Baseline Collection
Collects and saves baseline sensor readings for normalization.
"""

import sys
import csv
from datetime import datetime

from config import SENSOR_NAMES, BASELINE_DATA_FILE, BASELINE_DURATION_SEC, DATA_DIR
from serial_handler import SerialHandler, calculate_baseline


def collect_baseline():
    """Collect and save baseline sensor readings."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    handler = SerialHandler()
    
    print("=" * 50)
    print("TB Breath Analyzer - Baseline Calibration")
    print("=" * 50)
    print()
    print("Keep the chamber EMPTY and environment STABLE.")
    print()
    
    if not handler.connect():
        print("Failed to connect. Check port in config.py")
        sys.exit(1)
    
    print("Connected to ESP32")
    print()
    
    baseline = calculate_baseline(handler, BASELINE_DURATION_SEC)
    if not baseline:
        print("Baseline calibration failed.")
        handler.disconnect()
        sys.exit(1)
    
    print()
    
    timestamp = datetime.now().isoformat()
    with open(BASELINE_DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(SENSOR_NAMES + ["timestamp"])
        writer.writerow(baseline + [timestamp])
    
    print("=" * 50)
    print("BASELINE SAVED")
    print("=" * 50)
    print()
    for name, value in zip(SENSOR_NAMES, baseline):
        print(f"  {name}: {value:.2f}")
    print()
    print(f"Saved to: {BASELINE_DATA_FILE}")
    
    handler.disconnect()


if __name__ == "__main__":
    collect_baseline()
