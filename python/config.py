"""
TB Breath Analyzer - Configuration
Centralized settings for sensors, serial communication, and file paths.
"""

from pathlib import Path

SENSOR_NAMES = ["MQ3", "MQ135", "MQ2"]

CLASS_LABELS = {
    "healthy": "Healthy",
    "tb": "TB",
    "interference": "Interference"
}

SERIAL_PORT = "COM5"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 1.0

SAMPLE_INTERVAL_MS = 100
BASELINE_DURATION_SEC = 10
DASHBOARD_BASELINE_SAMPLES = 30

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

SENSOR_DATA_FILE = DATA_DIR / "sensor_data.csv"
SENSOR_DATA_NEW_FILE = DATA_DIR / "sensor_data_new.csv"
BASELINE_DATA_FILE = DATA_DIR / "baseline_data.csv"
MODEL_FILE = MODELS_DIR / "tb_model.pkl"

CHART_HISTORY_LENGTH = 100

STATUS_COLORS = {
    "Healthy": {"bg": "#1a472a", "text": "#4ade80", "label": "SCREENING NEGATIVE"},
    "TB": {"bg": "#4a1a1a", "text": "#f87171", "label": "RISK DETECTED: TB BIOMARKERS FOUND"},
    "Interference": {"bg": "#4a3f1a", "text": "#fbbf24", "label": "INCONCLUSIVE / INTERFERENCE"}
}
