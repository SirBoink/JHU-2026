"""
TB Breath Analyzer - Streamlit Dashboard
Real-time breath analysis with ML-based classification.
"""

import time
from collections import deque

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

from config import (
    MODEL_FILE,
    SENSOR_NAMES,
    SERIAL_PORT,
    BAUD_RATE,
    STATUS_COLORS,
    DASHBOARD_BASELINE_SAMPLES
)
from serial_handler import SerialHandler


CHART_MAX_POINTS = 200
UPDATE_INTERVAL = 0.05
Y_AXIS_MIN = 0
Y_AXIS_MAX = 2000


APPLE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background-color: #F5F5F7;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', system-ui, sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1D1D1F;
        text-align: center;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        font-weight: 400;
        color: #86868B;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 600;
        color: #1D1D1F;
        letter-spacing: -0.02em;
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #86868B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    .delta-positive { color: #34C759; }
    .delta-negative { color: #FF3B30; }
    .delta-neutral { color: #86868B; }
    
    .status-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 2rem 3rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .status-healthy { border-left: 4px solid #34C759; }
    .status-tb { border-left: 4px solid #FF3B30; }
    .status-interference { border-left: 4px solid #FF9500; }
    .status-waiting { border-left: 4px solid #86868B; }
    
    .status-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1D1D1F;
        letter-spacing: -0.01em;
    }
    
    .status-subtitle {
        font-size: 0.95rem;
        color: #86868B;
        margin-top: 0.5rem;
    }
    
    .chart-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin-top: 1.5rem;
    }
    
    .chart-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1D1D1F;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: #0071E3;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #0077ED;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 113, 227, 0.3);
    }
    
    .disconnect-btn > button { background: #FF3B30; }
    .disconnect-btn > button:hover {
        background: #FF453A;
        box-shadow: 0 4px 12px rgba(255, 59, 48, 0.3);
    }
    
    .connection-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-connected {
        background: rgba(52, 199, 89, 0.1);
        color: #34C759;
    }
    
    .status-disconnected {
        background: rgba(255, 59, 48, 0.1);
        color: #FF3B30;
    }
    
    .stProgress > div > div { background: #0071E3; }
    
    .stAlert {
        background: #FFFFFF;
        border: 1px solid #E5E5E7;
        border-radius: 8px;
    }
</style>
"""


def init_session_state():
    """Initialize session state variables."""
    if 'serial_handler' not in st.session_state:
        st.session_state.serial_handler = None
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    if 'data_history' not in st.session_state:
        st.session_state.data_history = {
            name: deque(maxlen=CHART_MAX_POINTS) for name in SENSOR_NAMES
        }
    if 'baseline' not in st.session_state:
        st.session_state.baseline = None
    if 'baseline_samples' not in st.session_state:
        st.session_state.baseline_samples = []
    if 'calibrating' not in st.session_state:
        st.session_state.calibrating = True
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'confidence' not in st.session_state:
        st.session_state.confidence = None


@st.cache_resource
def load_model():
    """Load trained ML model."""
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


def connect_sensor():
    """Establish serial connection."""
    try:
        handler = SerialHandler(port=SERIAL_PORT, baud_rate=BAUD_RATE)
        if handler.connect():
            st.session_state.serial_handler = handler
            st.session_state.connected = True
            st.session_state.calibrating = True
            st.session_state.baseline_samples = []
            return True
    except Exception as e:
        st.error(f"Connection failed: {e}")
    return False


def disconnect_sensor():
    """Close serial connection."""
    if st.session_state.serial_handler:
        st.session_state.serial_handler.disconnect()
    st.session_state.serial_handler = None
    st.session_state.connected = False
    st.session_state.calibrating = True
    st.session_state.baseline_samples = []
    st.session_state.baseline = None


def read_sensors() -> dict | None:
    """Read sensor values."""
    if not st.session_state.connected or not st.session_state.serial_handler:
        return None
    try:
        values = st.session_state.serial_handler.read_sensor_data()
        if values is None:
            return None
        return {name: values[i] for i, name in enumerate(SENSOR_NAMES)}
    except Exception:
        disconnect_sensor()
        return None


def update_baseline(data: dict):
    """Accumulate calibration samples."""
    if not st.session_state.calibrating:
        return
    st.session_state.baseline_samples.append(data)
    if len(st.session_state.baseline_samples) >= DASHBOARD_BASELINE_SAMPLES:
        baseline = {}
        for name in SENSOR_NAMES:
            values = [s[name] for s in st.session_state.baseline_samples]
            baseline[name] = np.mean(values)
        st.session_state.baseline = baseline
        st.session_state.calibrating = False


def get_delta(data: dict) -> dict:
    """Calculate percent change from baseline."""
    if not st.session_state.baseline:
        return {name: 0.0 for name in SENSOR_NAMES}
    delta = {}
    for name in SENSOR_NAMES:
        base = st.session_state.baseline[name]
        delta[name] = ((data[name] - base) / base) * 100 if base > 0 else 0.0
    return delta


def make_prediction(data: dict) -> tuple:
    """Run ML inference."""
    model = st.session_state.model
    if model is None or st.session_state.baseline is None:
        return None, None
    features = np.array([[data[name] for name in SENSOR_NAMES]])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    return prediction, max(proba)


def render_header():
    """Render page header."""
    st.markdown('<h1 class="main-header">TB Breath Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time Tuberculosis Screening System</p>', unsafe_allow_html=True)


def render_connection_controls():
    """Render connection UI."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.connected:
            st.markdown('<div class="connection-status status-connected">Connected</div>', unsafe_allow_html=True)
            st.markdown('<div class="disconnect-btn">', unsafe_allow_html=True)
            if st.button("Disconnect", use_container_width=True):
                disconnect_sensor()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="connection-status status-disconnected">Disconnected</div>', unsafe_allow_html=True)
            if st.button("Connect to Sensor", use_container_width=True):
                if connect_sensor():
                    st.rerun()


def render_calibration_progress():
    """Show calibration progress."""
    if st.session_state.calibrating and st.session_state.connected:
        progress = len(st.session_state.baseline_samples) / DASHBOARD_BASELINE_SAMPLES
        st.progress(progress)
        st.caption(f"Calibrating... {len(st.session_state.baseline_samples)}/{DASHBOARD_BASELINE_SAMPLES}")


def render_status_card(prediction: str | None, confidence: float | None):
    """Render screening result card."""
    if prediction is None:
        status_class, title, subtitle = "status-waiting", "Awaiting Reading", "Connect sensor and complete calibration"
    elif prediction == "Healthy":
        status_class, title, subtitle = "status-healthy", "Screening Negative", f"Confidence: {confidence:.1%}"
    elif prediction == "TB":
        status_class, title, subtitle = "status-tb", "Risk Detected", f"TB biomarkers found. Confidence: {confidence:.1%}"
    else:
        status_class, title, subtitle = "status-interference", "Inconclusive", f"Interference detected. Confidence: {confidence:.1%}"
    
    st.markdown(f'''
    <div class="status-card {status_class}">
        <div class="status-title">{title}</div>
        <div class="status-subtitle">{subtitle}</div>
    </div>
    ''', unsafe_allow_html=True)


def render_metric_cards(data: dict, delta: dict):
    """Render sensor metric cards."""
    cols = st.columns(len(SENSOR_NAMES))
    for i, name in enumerate(SENSOR_NAMES):
        with cols[i]:
            value = data.get(name, 0)
            pct = delta.get(name, 0)
            if pct > 5:
                delta_class, delta_text = "delta-positive", f"+{pct:.1f}%"
            elif pct < -5:
                delta_class, delta_text = "delta-negative", f"{pct:.1f}%"
            else:
                delta_class, delta_text = "delta-neutral", f"{pct:+.1f}%"
            
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{name}</div>
                <div class="metric-delta {delta_class}">{delta_text} from baseline</div>
            </div>
            ''', unsafe_allow_html=True)


def render_chart():
    """Render real-time sensor chart."""
    chart_data = []
    for name in SENSOR_NAMES:
        history = list(st.session_state.data_history[name])
        for i, val in enumerate(history):
            chart_data.append({'index': i, 'value': val, 'sensor': name})
    
    if not chart_data:
        st.info("Waiting for sensor data...")
        return
    
    df = pd.DataFrame(chart_data)
    color_scale = alt.Scale(domain=SENSOR_NAMES, range=['#0071E3', '#FF6B35', '#34C759'])
    
    chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X('index:Q', axis=None),
        y=alt.Y('value:Q', scale=alt.Scale(domain=[Y_AXIS_MIN, Y_AXIS_MAX]), title='Sensor Reading'),
        color=alt.Color('sensor:N', scale=color_scale, legend=alt.Legend(orient='top', title=None, labelFontSize=12))
    ).properties(height=280).configure_view(strokeWidth=0).configure_axis(labelFontSize=11, titleFontSize=12, gridColor='#E5E5E7')
    
    st.altair_chart(chart, use_container_width=True)


def main():
    """Main application."""
    st.markdown(APPLE_CSS, unsafe_allow_html=True)
    init_session_state()
    st.session_state.model = load_model()
    
    render_header()
    render_connection_controls()
    render_calibration_progress()
    st.markdown("---")
    
    status_container = st.empty()
    metrics_container = st.empty()
    
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Real-time Sensor Readings</div>', unsafe_allow_html=True)
    chart_container = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.connected:
        data = read_sensors()
        if data:
            for name in SENSOR_NAMES:
                st.session_state.data_history[name].append(data[name])
            if st.session_state.calibrating:
                update_baseline(data)
            delta = get_delta(data)
            prediction, confidence = (None, None) if st.session_state.calibrating else make_prediction(data)
            
            with status_container:
                render_status_card(prediction, confidence)
            with metrics_container:
                render_metric_cards(data, delta)
            with chart_container:
                render_chart()
        
        time.sleep(UPDATE_INTERVAL)
        st.rerun()
    else:
        with status_container:
            render_status_card(None, None)
        with chart_container:
            st.caption("Connect to sensor to view real-time data")


if __name__ == "__main__":
    main()
