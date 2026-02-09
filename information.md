# Python Codebase Reference

Quick reference for each Python module in the TB Breath Analyzer.

---

## `config.py`

**Purpose:** Centralized configuration for the entire project.

| Setting | Description |
|---------|-------------|
| `SENSOR_NAMES` | List of sensor identifiers (`["MQ3", "MQ135", "MQ2"]`) |
| `SERIAL_PORT` / `BAUD_RATE` | ESP32 connection settings |
| `BASELINE_DURATION_SEC` | Duration for baseline calibration (10s) |
| `DASHBOARD_BASELINE_SAMPLES` | Samples collected before dashboard is ready (30) |
| `DATA_DIR` / `MODELS_DIR` | Paths derived from project root |

**Key Behavior:** All paths use `pathlib.Path` for cross-platform compatibility.

---

## `serial_handler.py`

**Purpose:** Serial communication abstraction for ESP32.

### `SerialHandler` class

| Method | Description |
|--------|-------------|
| `connect()` | Opens serial port, waits 2s for ESP32 boot, flushes buffer |
| `disconnect()` | Closes connection safely |
| `read_sensor_data()` | Returns `List[int]` or `None` if read fails |

### Standalone Functions

| Function | Description |
|----------|-------------|
| `calculate_baseline(handler, duration)` | Collects samples over `duration` seconds, returns averages |
| `normalize_reading(values, baseline)` | Computes `(value - baseline) / baseline` per sensor |

**Key Behavior:** 
- `read_sensor_data()` silently returns `None` on parse errors (non-blocking)
- 2-second delay in `connect()` allows ESP32 to finish boot sequence

---

## `collect_baseline.py`

**Purpose:** Captures baseline sensor readings for normalization.

**Usage:** Run with empty chamber in stable environment.

**Output:** `data/baseline_data.csv` with averaged sensor values and timestamp.

**Key Behavior:** Must be run before `train_model.py` (which requires baseline for normalization).

---

## `collect_healthy.py`

**Purpose:** Collects normal breath samples labeled as `"Healthy"`.

**Output:** Appends to `data/sensor_data_new.csv`.

**Key Behavior:**
- Creates CSV header if file doesn't exist
- Continuous collection until `Ctrl+C`
- Flushes after each sample (no data loss on interrupt)

---

## `collect_tb_surrogate.py`

**Purpose:** Collects TB surrogate samples (breath + alcohol/VOCs) labeled as `"TB"`.

**Suggested Surrogates:** Hand sanitizer vapor, dry-erase marker fumes.

**Key Behavior:** Same as `collect_healthy.py`, different label.

---

## `collect_interference.py`

**Purpose:** Collects confounding samples labeled as `"Interference"`.

**Suggested Confounders:** Perfume, coffee, strong food odors.

**Key Behavior:** Same collection logic as other collectors.

---

## `augment.py`

**Purpose:** KDE-based synthetic data generation for small datasets.

### Pipeline

1. **Stratified split** (80/20) — test set is real data only
2. **Save test set** to `test_real_hidden.csv` (no synthetic contamination)
3. **Fit Gaussian KDE** per class on training data
4. **Generate synthetic samples** (2000 per class by default)
5. **Combine** real + synthetic → `train_augmented.csv`

**Key Behavior:**
- Synthetic values are clipped to `>= 0` and rounded to integers
- `is_synthetic` column marks origin (0 = real, 1 = synthetic)
- Must run before `train_augmented.py`

---

## `train_model.py`

**Purpose:** Train Random Forest with baseline normalization.

### Pipeline

1. Load baseline from `baseline_data.csv`
2. Load data from `sensor_data_new.csv`
3. Normalize: `(raw - baseline) / baseline`
4. Train/test split (80/20, stratified)
5. Train RandomForestClassifier
6. Save to `models/tb_model.pkl`

**Key Behavior:**
- Uses normalized features (`*_pct` columns)
- Generates `confusion_matrix.png`
- Requires `collect_baseline.py` to have been run first

---

## `train_augmented.py`

**Purpose:** Train on KDE-augmented data, evaluate on real data only.

### Pipeline

1. Load `train_augmented.csv` (real + synthetic)
2. Load `test_real_hidden.csv` (real only)
3. Train RandomForestClassifier on augmented set
4. Evaluate on pristine real test set
5. Save to `models/tb_model.pkl` (overwrites)

**Key Behavior:**
- Uses raw sensor values (no baseline normalization)
- Saves both `tb_model_augmented.pkl` and `tb_model.pkl`
- Requires `augment.py` to have been run first

---

## `app.py`

**Purpose:** Streamlit real-time dashboard for live classification.

### Features

| Feature | Description |
|---------|-------------|
| Connection UI | Connect/disconnect from ESP32 |
| Baseline calibration | Collects 30 samples before predictions |
| Metric cards | Live sensor values with % change from baseline |
| Status card | Classification result (Healthy/TB/Interference) |
| Real-time chart | Scrolling Altair line chart (last 200 points) |

### Session State

| Variable | Purpose |
|----------|---------|
| `serial_handler` | Active `SerialHandler` instance |
| `baseline` | Computed baseline averages |
| `calibrating` | `True` until 30 samples collected |
| `data_history` | Deques for chart history per sensor |
| `model` | Loaded sklearn classifier |

**Key Behavior:**
- Uses `@st.cache_resource` for model loading
- Auto-reruns every 50ms while connected
- Apple-inspired CSS theming embedded inline
- Predictions use raw values, not normalized (matches `train_augmented.py`)

---

## Common Patterns

### Logging

All scripts use a `log()` helper for timestamped output:
```python
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
```

### Error Handling

- Serial errors → graceful disconnect, `None` return
- Missing files → clear error message + `sys.exit(1)`

### Data Flow

```
collect_baseline.py → baseline_data.csv
                            ↓
collect_*.py → sensor_data_new.csv
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
train_model.py                          augment.py
(baseline-normalized)                   (KDE augmentation)
        ↓                                       ↓
tb_model.pkl                     train_augmented.csv + test_real_hidden.csv
                                                ↓
                                    train_augmented.py
                                                ↓
                                        tb_model.pkl
                                                ↓
                                            app.py
```
