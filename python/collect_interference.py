"""
TB Breath Analyzer - Interference Sample Collection
Collects breath + confounding substances for the 'Interference' class.
"""

import sys
import csv

from config import SENSOR_NAMES, SENSOR_DATA_NEW_FILE, DATA_DIR
from serial_handler import SerialHandler


def collect_samples(label: str):
    """Collection loop for a given class label."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = SENSOR_DATA_NEW_FILE.exists()
    handler = SerialHandler()
    
    print("=" * 50)
    print(f"TB Breath Analyzer - {label} Sample Collection")
    print("=" * 50)
    print()
    print("Confounders: perfume, coffee, strong odors")
    print()
    
    if not handler.connect():
        print("Failed to connect. Check port in config.py")
        sys.exit(1)
    
    print("Connected to ESP32")
    print()
    
    with open(SENSOR_DATA_NEW_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([f"{name}_raw" for name in SENSOR_NAMES] + ["label"])
        
        print(f"Recording {label} samples. Press Ctrl+C to stop.")
        print()
        
        sample_count = 0
        try:
            while True:
                data = handler.read_sensor_data()
                if data:
                    writer.writerow(data + [label])
                    f.flush()
                    sample_count += 1
                    display = " | ".join(f"{name}: {val:4d}" for name, val in zip(SENSOR_NAMES, data))
                    print(f"\r[{sample_count:4d}] {display}", end="", flush=True)
        except KeyboardInterrupt:
            print()
            print()
            print(f"Stopped. {sample_count} samples saved to {SENSOR_DATA_NEW_FILE}")
    
    handler.disconnect()


if __name__ == "__main__":
    collect_samples("Interference")
