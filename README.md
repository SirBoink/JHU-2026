# Breath Analyzer

A low-cost tuberculosis screening prototype using ESP32 and MOS gas sensors with machine learning classification.

> **Note:** Since real TB samples are unavailable, we use surrogates:
> - **Healthy:** Normal human breath
> - **TB:** Breath + Whiteboard Marker (Ketones) + Hand Sanitizer (alcohol) + Permanent marker fumes (Benzene + aromatic compounds)
> - **Interference:** Breath + Mint (cyclic alcohol)

---

## Hardware Requirements

| Component | Technology | Functional Role |
|-----------|------------|-----------------|
| ESP32 DevKit V1 | 32-bit Dual-Core MCU | Central processing unit for multi-channel signal acquisition and real-time inference |
| MQ-3 Sensor | SnO2 Chemiresistor | Detection of methylated metabolites and metabolic alcohol biomarkers |
| MQ-135 Sensor | SnO2 Chemiresistor | Sensitive to aromatic hydrocarbons, benzene derivatives, and ammonia |
| MQ-2 Sensor | SnO2 Chemiresistor | Broad-spectrum detection of combustible VOCs and smoke |

### Wiring

```
ESP32 DevKit V1
┌─────────────────────────┐
│                         │
│  GPIO 25 ←──── MQ-3 AO  │
│  GPIO 26 ←──── MQ-135 AO│
│  GPIO 27 ←──── MQ-2 AO  │
│                         │
│  3.3V ─────→ Sensor VCC │
│  GND ──────→ Sensor GND │
│                         │
└─────────────────────────┘
```

---

## Project Structure

```
tb_breath_analyzer/
├── firmware/
│   └── sensor_reader/
│       └── sensor_reader.ino       # ESP32 firmware
├── python/
│   ├── config.py                   # Centralized configuration
│   ├── serial_handler.py           # Serial communication module
│   ├── collect_baseline.py         # Baseline calibration
│   ├── collect_healthy.py          # Healthy sample collection
│   ├── collect_tb_surrogate.py     # TB surrogate collection
│   ├── collect_interference.py     # Interference collection
│   ├── augment.py                  # KDE-based data augmentation
│   ├── train_model.py              # Model training (baseline-normalized)
│   ├── train_augmented.py          # Model training (augmented data)
│   └── app.py                      # Streamlit dashboard
├── data/
│   ├── baseline_data.csv           # Baseline sensor readings
│   ├── sensor_data_new.csv         # Collected training data
│   ├── train_augmented.csv         # Augmented training set
│   └── test_real_hidden.csv        # Held-out test set (real data only)
├── models/
│   ├── tb_model.pkl                # Trained classifier
│   └── confusion_matrix.png        # Training visualization
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Python Environment

Requires **Python 3.10 or higher**.

```powershell
cd tb_breath_analyzer
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Serial Port

Edit `python/config.py`:

```python
SERIAL_PORT = "COM5"  # Your ESP32's COM port
```

### 3. Flash ESP32

1. Open `firmware/sensor_reader/sensor_reader.ino` in Arduino IDE
2. Select Board: ESP32 Dev Module
3. Upload and verify output at 115200 baud:
   ```
   TB Breath Analyzer Ready
   Sensors: 3
   1234,2345,1456
   ```

---

## Data Collection Workflow

Run from the `python/` directory:

```powershell
# 1. Collect baseline (empty chamber, 10 seconds)
python collect_baseline.py

# 2. Collect samples for each class
python collect_healthy.py          # Normal breath
python collect_tb_surrogate.py     # Breath + alcohol/VOCs
python collect_interference.py     # Breath + perfume/coffee
```

Press `Ctrl+C` to stop collection. Aim for 100+ samples per class.

---

## Training

### Standard Training

```powershell
python train_model.py
```

### Augmented Training

Recommended for small datasets.

```powershell
python augment.py          # Generate synthetic samples via KDE
python train_augmented.py  # Train on augmented data, test on real data
```

---

## Dashboard

```powershell
streamlit run app.py
```

Features:
- 30-sample baseline calibration on connect
- Real-time sensor visualization
- ML-based classification with confidence scores

---

## Adding a New Sensor

1. **Hardware:** Connect sensor analog output to an available GPIO pin

2. **Firmware:** Add to `sensor_reader.ino`:
   ```cpp
   const SensorConfig SENSORS[] = {
       {"MQ3",   25},
       {"MQ135", 26},
       {"MQ2",   27},
       {"MQ7",   33}  // New sensor
   };
   ```

3. **Python:** Update `config.py`:
   ```python
   SENSOR_NAMES = ["MQ3", "MQ135", "MQ2", "MQ7"]
   ```

4. Re-upload firmware and re-collect data.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Serial port not found | Check Device Manager for correct COM port |
| Permission denied | Close Arduino Serial Monitor or other serial connections |
| Model not found | Run training script first |
| Low accuracy | Collect more samples, ensure clean baseline |
| Sensors reading 0 | Check wiring, allow sensors to warm up (1-2 min) |

---

## License

MIT License

