"""
TB Breath Analyzer - Serial Communication
Handles ESP32 communication and data parsing.
"""

import serial
import time
from typing import Optional, List
from config import SERIAL_PORT, BAUD_RATE, SERIAL_TIMEOUT, SENSOR_NAMES


class SerialHandler:
    """Handles serial communication with ESP32."""
    
    def __init__(self, port: str = SERIAL_PORT, baud_rate: int = BAUD_RATE):
        self.port = port
        self.baud_rate = baud_rate
        self.connection: Optional[serial.Serial] = None
    
    def connect(self) -> bool:
        """Establish serial connection."""
        try:
            self.connection = serial.Serial(
                port=self.port, baudrate=self.baud_rate, timeout=SERIAL_TIMEOUT
            )
            time.sleep(2)
            self._flush_buffer()
            return True
        except serial.SerialException as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection."""
        if self.connection and self.connection.is_open:
            self.connection.close()
            self.connection = None
    
    def read_sensor_data(self) -> Optional[List[int]]:
        """Read sensor values. Returns list of ints or None."""
        if not self.connection:
            return None
        try:
            line = self.connection.readline().decode('utf-8').strip()
            if not line or not line[0].isdigit():
                return None
            values = [int(x) for x in line.split(',')]
            if len(values) != len(SENSOR_NAMES):
                return None
            return values
        except (ValueError, UnicodeDecodeError):
            return None
    
    def _flush_buffer(self):
        """Clear serial buffer."""
        if self.connection:
            self.connection.reset_input_buffer()
            self.connection.reset_output_buffer()


def calculate_baseline(handler: SerialHandler, duration_sec: float) -> Optional[List[float]]:
    """Calculate baseline averages for all sensors."""
    print(f"Calculating baseline ({duration_sec}s)...")
    print("Keep chamber empty and environment stable.")
    
    samples = [[] for _ in SENSOR_NAMES]
    start_time = time.time()
    
    while time.time() - start_time < duration_sec:
        data = handler.read_sensor_data()
        if data:
            for i, value in enumerate(data):
                samples[i].append(value)
            elapsed = time.time() - start_time
            progress = int((elapsed / duration_sec) * 20)
            print(f"\r[{'=' * progress}{' ' * (20 - progress)}] {elapsed:.1f}s", end="", flush=True)
    print()
    
    baselines = []
    for i, sensor_samples in enumerate(samples):
        if not sensor_samples:
            print(f"Warning: No samples for {SENSOR_NAMES[i]}")
            return None
        avg = sum(sensor_samples) / len(sensor_samples)
        baselines.append(avg)
        print(f"  {SENSOR_NAMES[i]}: {avg:.1f} (n={len(sensor_samples)})")
    
    return baselines


def normalize_reading(values: List[int], baseline: List[float]) -> List[float]:
    """Calculate percent change from baseline."""
    return [(val - base) / base if base > 0 else 0.0 for val, base in zip(values, baseline)]
