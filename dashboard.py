# dashboard.py
# integrated_gsr_ppg_app with enhanced UI and emotional state classification

import os
import time
import csv
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import streamlit as st
import logging
import joblib
from serial import Serial, SerialException
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
from serial.tools import list_ports
from queue import Queue
from typing import Optional, List, Dict

# -----------------------------
# Config / Globals
# -----------------------------
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Sampling rates (adjust if your device differs)
gsr_sampling_rate = 1000
ppg_sampling_rate = 1000

DATA_STORAGE_DIR = 'uploaded_data_gsr_ppg'
PREDICTIONS_FILE = os.path.join(DATA_STORAGE_DIR, 'emotional_predictions.csv')
EMOTIONAL_STATE_FILE = os.path.join(DATA_STORAGE_DIR, 'emotional_state.txt')
os.makedirs(DATA_STORAGE_DIR, exist_ok=True)

data_queue = Queue()
sdnn_values: List[float] = []
sdnn_timestamps: List[str] = []
gsr_values: List[float] = []
gsr_timestamps: List[str] = []

# In-memory predictions cache
predictions_buffer: List[Dict] = []

# Add custom CSS for UI enhancements
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }

    /* Tab styling */
    div[data-baseweb="tab-list"] {
        gap: 10px;
        padding: 0 15px;
    }

    div[data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 8px;
        background: #f0f2f6;
        transition: all 0.3s ease;
        margin: 0 5px !important;
    }

    div[data-baseweb="tab"]:hover {
        background: #e0e5ec;
    }

    div[aria-selected="true"] {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%) !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(106,17,203,0.2);
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #6a11cb;
        margin-bottom: 20px;
    }

    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(106,17,203,0.3);
    }

    /* Headers */
    .section-header {
        padding-bottom: 10px;
        margin-bottom: 20px;
        border-bottom: 2px solid #e0e7ff;
        color: #2c3e50;
    }

    /* Value cards */
    .value-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }

    .value-card h3 {
        margin: 5px 0;
        font-size: 1.8rem;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .value-card p {
        margin: 0;
        color: #6c757d;
        font-size: 0.9rem;
    }

    /* File uploader */
    .stFileUploader > div > div {
        border: 2px dashed #6a11cb;
        border-radius: 8px;
        padding: 20px;
    }

    /* Sliders */
    .stSlider .thumb {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%) !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    /* Emotion timeline */
    .timeline-point {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Model Loading
# -----------------------------
@st.cache_resource
def load_emotion_model():
    """Load ML model from common locations."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_rf_model.pkl"),
        os.path.join(os.getcwd(), "emotion_rf_model.pkl"),
        os.path.join(DATA_STORAGE_DIR, "emotion_rf_model.pkl"),
    ]
    for model_path in candidates:
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except Exception as e:
                st.warning(f"Failed to load model at {model_path}: {e}")
    st.warning("Emotion model not found. Place 'emotion_rf_model.pkl' next to this script, "
               "in the working directory, or inside 'uploaded_data_gsr_ppg/'.")
    return None


emotion_model = load_emotion_model()


# -----------------------------
# Utilities: Saving / Logging
# -----------------------------
def ensure_predictions_csv():
    """Ensure predictions CSV exists with header."""
    if not os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "WindowStart", "WindowEnd", "Emotion", "SDNN_ms", "GSR_Mean", "Session"])


def log_prediction(window_start: pd.Timestamp,
                   window_end: pd.Timestamp,
                   emotion: str,
                   sdnn_ms: Optional[float],
                   gsr_mean: Optional[float],
                   session: str):
    """Append a single prediction to buffer and CSV."""
    ensure_predictions_csv()
    ts_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "Timestamp": ts_now,
        "WindowStart": window_start.strftime("%Y-%m-%d %H:%M:%S"),
        "WindowEnd": window_end.strftime("%Y-%m-%d %H:%M:%S"),
        "Emotion": str(emotion),
        "SDNN_ms": float(sdnn_ms) if sdnn_ms is not None else np.nan,
        "GSR_Mean": float(gsr_mean) if gsr_mean is not None else np.nan,
        "Session": session,
    }
    predictions_buffer.append(row)
    # Append to CSV
    with open(PREDICTIONS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(list(row.values()))
    # Also write to human-readable txt log
    save_emotional_state(emotion, sdnn_ms, gsr_mean, output_file=EMOTIONAL_STATE_FILE)


def save_emotional_state(emotional_state, sdnn=None, gsr_mean=None, output_file=EMOTIONAL_STATE_FILE):
    dir_name = os.path.dirname(output_file)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, 'a', encoding='utf-8') as file:
        file.write(f"=== Emotional Analysis Entry ===\n")
        file.write(f"Timestamp: {timestamp_str}\n")
        if sdnn is not None and np.isfinite(sdnn):
            file.write(f"HRV_SDNN: {sdnn:.2f} ms\n")
        if gsr_mean is not None and np.isfinite(gsr_mean):
            file.write(f"GSR_Mean: {gsr_mean:.4f} μS\n")
        file.write(f"State Inference: {emotional_state}\n\n")
    st.success(f"Saved emotional state → `{output_file}`", icon="✅")


# -----------------------------
# ML Inference
# -----------------------------
def infer_emotional_state_ml(sdnn: Optional[float], gsr_mean: Optional[float]) -> str:
    if emotion_model is None:
        return "Model not loaded"
    if sdnn is None or gsr_mean is None or not np.isfinite(sdnn) or not np.isfinite(gsr_mean):
        return "Insufficient data"
    input_data = pd.DataFrame([[sdnn, gsr_mean]], columns=["sdnn", "gsr_mean"])
    try:
        return str(emotion_model.predict(input_data)[0])
    except Exception as e:
        st.error(f"Emotion model prediction failed: {e}")
        return "Prediction error"


# -----------------------------
# Emotional State Classification from Data
# -----------------------------
def classify_emotional_state(row):
    """
    Classify emotional state based on Dr Fatema.csv data patterns
    Positive: Low GSR values (below 1900)
    Neutral: GSR values between 1900-1925
    Stressed: GSR values above 1925
    """
    gsr = row['GSR_Value']

    if gsr < 1900:
        return "Positive"
    elif 1900 <= gsr <= 1925:
        return "Neutral"
    else:
        return "Stressed"


# -----------------------------
# Signal Processing & Visualization
# -----------------------------
def plot_signals(df: pd.DataFrame, ppg_cleaned: np.ndarray, ppg_peaks: dict, window: tuple = None):
    """Plot raw and processed signals with detected peaks."""
    if df.empty:
        return

    # If window is provided, slice the data
    if window:
        start_idx, end_idx = window
        df = df.iloc[start_idx:end_idx]
        ppg_cleaned = ppg_cleaned[start_idx:end_idx]
        if 'PPG_Peaks' in ppg_peaks:
            mask = (ppg_peaks['PPG_Peaks'] >= start_idx) & (ppg_peaks['PPG_Peaks'] < end_idx)
            ppg_peaks['PPG_Peaks'] = ppg_peaks['PPG_Peaks'][mask] - start_idx

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot PPG signals
    time_axis = np.arange(len(df)) / ppg_sampling_rate
    ax1.plot(time_axis, df['PPG_A13_CAL'], label='Raw PPG', alpha=0.7, color='#6a11cb')
    ax1.plot(time_axis, ppg_cleaned, label='Cleaned PPG', color='#2575fc')

    # Mark peaks if available
    if 'PPG_Peaks' in ppg_peaks and len(ppg_peaks['PPG_Peaks']) > 0:
        peak_times = ppg_peaks['PPG_Peaks'] / ppg_sampling_rate
        ax1.scatter(peak_times, ppg_cleaned[ppg_peaks['PPG_Peaks']],
                    color='#ff5252', label='Detected Peaks', s=50)

    ax1.set_title('PPG Signal Processing', fontsize=16)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot GSR signal
    ax2.plot(time_axis, df['GSR_Skin_Conductance_CAL'], label='GSR', color='#00c853')
    ax2.set_title('GSR Signal', fontsize=16)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Conductance', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig)


def plot_hrv_rr_intervals(intervals: np.ndarray):
    """Plot RR intervals for HRV analysis."""
    if len(intervals) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(intervals, marker='o', linestyle='-', color='#6a11cb')
    ax.set_title('RR Interval Dynamics (Heart Rate Variability)', fontsize=16)
    ax.set_xlabel('Beat Number', fontsize=12)
    ax.set_ylabel('RR Interval (ms)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)


def plot_emotion_timeline(emotions: list, timestamps: list, selected_time: Optional[float] = None):
    """Plot emotional state changes over time using Plotly with time slider."""
    if not emotions:
        return

    # Map emotions to colors
    emotion_colors = {
        "Positive": "#4CAF50",
        "Neutral": "#9E9E9E",
        "Stressed": "#F44336",
        "unknown": "#9E9E9E"
    }

    # Create a DataFrame for Plotly
    df_emotion = pd.DataFrame({
        "Timestamp": timestamps,
        "Emotion": emotions,
        "Color": [emotion_colors.get(e, "#9E9E9E") for e in emotions]
    })

    # Create a timeline plot
    fig = px.scatter(
        df_emotion,
        x="Timestamp",
        y="Emotion",
        color="Emotion",
        color_discrete_map=emotion_colors,
        hover_data={"Timestamp": "|%B %d, %Y %H:%M:%S"},
        title="Emotional State Timeline",
        height=400
    )

    # Add vertical line for selected time
    if selected_time is not None:
        fig.add_vline(
            x=selected_time,
            line_dash="dash",
            line_color="red",
            annotation_text="Selected Time",
            annotation_position="top left"
        )

    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Emotional State",
        legend_title="Emotions",
        hovermode="closest",
        plot_bgcolor='rgba(240,242,246,0.5)'
    )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Emotional State Visualization
# -----------------------------
def visualize_emotional_state(df):
    """Visualize emotional state over time with GSR values"""
    if df.empty or 'Emotion' not in df.columns:
        return

    # Create a copy for visualization
    vis_df = df.copy()

    # Map emotions to colors
    emotion_colors = {
        "Positive": "#4CAF50",
        "Neutral": "#9E9E9E",
        "Stressed": "#F44336"
    }

    # Create a timeline plot
    fig = go.Figure()

    # Add GSR trace
    fig.add_trace(go.Scatter(
        x=vis_df.index,
        y=vis_df['GSR_Value'],
        name='GSR Value',
        mode='lines',
        line=dict(color='#2196F3', width=2),
        yaxis='y1'
    ))

    # Add emotional state markers
    for emotion, color in emotion_colors.items():
        emotion_data = vis_df[vis_df['Emotion'] == emotion]
        if not emotion_data.empty:
            fig.add_trace(go.Scatter(
                x=emotion_data.index,
                y=emotion_data['GSR_Value'],
                name=emotion,
                mode='markers',
                marker=dict(color=color, size=8, symbol='diamond'),
                yaxis='y1'
            ))

    # Set layout
    fig.update_layout(
        title='Emotional State Timeline',
        xaxis_title='Time',
        yaxis_title='GSR Value',
        legend_title="Emotional State",
        hovermode="x unified",
        plot_bgcolor='rgba(240,242,246,0.5)',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Create a distribution of emotional states
    emotion_counts = vis_df['Emotion'].value_counts().reset_index()
    emotion_counts.columns = ['Emotion', 'Count']

    # Create pie chart
    fig = px.pie(
        emotion_counts,
        names='Emotion',
        values='Count',
        color='Emotion',
        color_discrete_map=emotion_colors,
        hole=0.4,
        title='Emotional State Distribution'
    )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Core Processing Functions
# -----------------------------
def compute_sdnn_from_ppg(ppg_segment: np.ndarray, sr: int) -> Optional[float]:
    """Return SDNN in ms from a PPG segment."""
    if ppg_segment.size < max(5, int(0.3 * sr)):
        return None
    ppg_clean = nk.ppg_clean(ppg_segment.astype(float), sampling_rate=sr)
    try:
        peaks = nk.ppg_findpeaks(ppg_clean, sampling_rate=sr)
        ppg_idx = peaks.get('PPG_Peaks', [])
        if not isinstance(ppg_idx, np.ndarray) or ppg_idx.size <= 1:
            return None
        intervals_ms = np.diff(ppg_idx) * (1000.0 / sr)
        return float(np.std(intervals_ms))
    except Exception:
        return None


def process_data(df: pd.DataFrame, window_sec: int, session_name: str):
    """Process data with variable window size and generate predictions."""
    if df.empty:
        st.warning("No data available for processing.")
        return

    # Setup progress visualization
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_placeholder = st.empty()

    # Preprocess signals
    scaler_gsr = StandardScaler()
    scaler_ppg = StandardScaler()
    df['GSR_Skin_Conductance_CAL'] = scaler_gsr.fit_transform(df[['GSR_Skin_Conductance_CAL']])
    df['PPG_A13_CAL'] = scaler_ppg.fit_transform(df[['PPG_A13_CAL']])

    # Process entire PPG signal
    ppg_signal = df['PPG_A13_CAL'].values
    ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=ppg_sampling_rate)
    ppg_peaks = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=ppg_sampling_rate)

    # Show signal processing visualization
    st.subheader("Signal Processing Visualization")
    plot_signals(df, ppg_cleaned, ppg_peaks, window=(0, min(5000, len(df))))

    # Calculate HRV metrics from entire signal
    st.subheader("Heart Rate Variability Analysis")
    if 'PPG_Peaks' in ppg_peaks and len(ppg_peaks['PPG_Peaks']) > 1:
        intervals = np.diff(ppg_peaks['PPG_Peaks']) * (1000 / ppg_sampling_rate)
        sdnn = np.std(intervals)
        rmssd = np.sqrt(np.mean(np.square(np.diff(intervals))))

        col1, col2 = st.columns(2)
        col1.markdown(f"""
        <div class="value-card">
            <h3>{sdnn:.2f}</h3>
            <p>SDNN (Overall HRV) ms</p>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="value-card">
            <h3>{rmssd:.2f}</h3>
            <p>RMSSD (Parasympathetic Activity) ms</p>
        </div>
        """, unsafe_allow_html=True)

        plot_hrv_rr_intervals(intervals)
    else:
        st.warning("Not enough PPG peaks to compute HRV metrics.")

    # Process in sliding windows
    st.subheader("Windowed Emotional State Analysis")
    window_size = window_sec * ppg_sampling_rate
    step_size = window_size // 2  # 50% overlap

    emotions = []
    emotion_timestamps = []
    all_sdnn = []
    all_gsr = []

    if window_size > len(df):
        st.warning(f"Selected window size ({window_sec}s) is larger than data duration. Using full data.")
        window_size = len(df)
        step_size = len(df)

    max_windows = max(1, (len(df) - window_size) // step_size + 1)

    for i, start in enumerate(range(0, len(df) - window_size + 1, step_size)):
        # Update progress
        progress = min((i + 1) / max_windows, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing window {i + 1}/{max_windows} - {progress * 100:.1f}%")

        # Update visualization
        if i % 3 == 0:  # Update every 3 windows for performance
            with chart_placeholder.container():
                # Create a simple visualization
                fig, ax = plt.subplots(figsize=(10, 3))
                window_data = df.iloc[start:start + window_size]
                ax.plot(window_data['PPG_A13_CAL'], label='PPG', color='#6a11cb')
                ax.plot(window_data['GSR_Skin_Conductance_CAL'], label='GSR', color='#00c853')
                ax.set_title(f"Processing Window {i + 1}")
                ax.legend()
                st.pyplot(fig)

        end = start + window_size
        window_data = df.iloc[start:end]

        # Get PPG segment and compute SDNN
        ppg_segment = window_data['PPG_A13_CAL'].values
        sdnn = compute_sdnn_from_ppg(ppg_segment, ppg_sampling_rate)

        # Get GSR mean for window
        gsr_mean = np.mean(window_data['GSR_Skin_Conductance_CAL'].values)

        # Predict emotion
        emotion = infer_emotional_state_ml(sdnn, gsr_mean)

        # Store for visualization
        timestamp = window_data.index[int((start + end) / 2)]
        emotions.append(emotion)
        emotion_timestamps.append(timestamp)
        if sdnn is not None:
            all_sdnn.append(sdnn)
        if gsr_mean is not None:
            all_gsr.append(gsr_mean)

        # Log prediction
        log_prediction(
            window_start=window_data.index[0],
            window_end=window_data.index[-1],
            emotion=emotion,
            sdnn_ms=sdnn,
            gsr_mean=gsr_mean,
            session=session_name
        )

    # Clear progress visualization
    progress_bar.empty()
    status_text.empty()
    chart_placeholder.empty()

    # Show emotion timeline with time slider
    if emotions:
        # Calculate relative time positions for slider
        min_time = 0
        max_time = len(df) / ppg_sampling_rate
        time_step = 1.0  # seconds

        # Create slider for time selection
        st.subheader("Emotion Timeline Explorer")
        selected_time = st.slider(
            "Select time in recording (seconds):",
            min_value=min_time,
            max_value=max_time,
            value=min_time,
            step=time_step
        )

        # Show timeline with selected time
        plot_emotion_timeline(emotions, emotion_timestamps, selected_time)

        # Find emotion at selected time
        selected_emotion = "unknown"
        closest_index = 0
        min_diff = float('inf')

        # Find the closest timestamp to the selected time
        for i, ts in enumerate(emotion_timestamps):
            ts_seconds = (ts - emotion_timestamps[0]).total_seconds()
            time_diff = abs(ts_seconds - selected_time)
            if time_diff < min_diff:
                min_diff = time_diff
                closest_index = i
                selected_emotion = emotions[i]

        # Display selected emotion
        emotion_colors = {
            "Positive": ("#4CAF50", "😊"),
            "Neutral": ("#9E9E9E", "😐"),
            "Stressed": ("#F44336", "😰"),
            "unknown": ("#9E9E9E", "❓")
        }
        color, emoji = emotion_colors.get(selected_emotion, ("#9E9E9E", "❓"))

        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {color}; text-align: center">
            <h3 style="color: {color}; margin-top:0">{emoji} {selected_emotion}</h3>
            <p>Emotional State at {selected_time:.1f} seconds</p>
        </div>
        """, unsafe_allow_html=True)

        # Show metrics at selected time
        if all_sdnn and all_gsr:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="value-card">
                    <h3>{all_sdnn[closest_index]:.2f}</h3>
                    <p>HRV (SDNN) ms</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="value-card">
                    <h3>{all_gsr[closest_index]:.4f}</h3>
                    <p>GSR Mean</p>
                </div>
                """, unsafe_allow_html=True)

    # Show metrics over time
    if all_sdnn and all_gsr:
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Plot SDNN
        color = '#6a11cb'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('SDNN (ms)', color=color)
        ax1.plot(emotion_timestamps, all_sdnn, color=color, marker='o', label='HRV (SDNN)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Create second y-axis for GSR
        ax2 = ax1.twinx()
        color = '#00c853'
        ax2.set_ylabel('GSR Conductance', color=color)
        ax2.plot(emotion_timestamps, all_gsr, color=color, marker='x', label='GSR Mean')
        ax2.tick_params(axis='y', labelcolor=color)

        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title('Physiological Metrics Over Time')
        st.pyplot(fig)


# -----------------------------
# Data Preparation
# -----------------------------
def prepare_data(df: pd.DataFrame, session_name: str):
    """Prepare and validate data for processing."""
    if df.empty:
        st.error("No data available for analysis")
        return None

    # Map alternate column names
    col_map = {
        'Timestamp': 'Timestamp_Unix_CAL',
        'GSR_Value': 'GSR_Skin_Conductance_CAL',
        'PPG_Value': 'PPG_A13_CAL'
    }
    df = df.copy()

    # Rename if needed
    for orig, new in col_map.items():
        if orig in df.columns and new not in df.columns:
            df.rename(columns={orig: new}, inplace=True)

    # Validate required columns
    required_cols = ['Timestamp_Unix_CAL', 'GSR_Skin_Conductance_CAL', 'PPG_A13_CAL']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return None

    # Process timestamps
    df['Timestamp_Unix_CAL'] = pd.to_numeric(df['Timestamp_Unix_CAL'], errors='coerce')
    df = df.dropna(subset=['Timestamp_Unix_CAL'])
    if df.empty:
        st.error("No valid data after timestamp processing")
        return None

    # Convert ms unix → datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp_Unix_CAL'], unit='ms', errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    if df.empty:
        st.error("No valid timestamps after conversion")
        return None

    df.set_index('Timestamp', drop=True, inplace=True)

    # Show data sample
    st.subheader('📄 Data Sample')
    st.write(df.head())

    # Show recording duration
    duration = df.index[-1] - df.index[0]
    st.info(f"Recording Duration: {duration.total_seconds():.1f} seconds")

    return df


# -----------------------------
# Emotional State Analysis from Data
# -----------------------------
def analyze_emotional_state(df):
    """Analyze emotional state from provided data"""
    if df.empty:
        st.warning("No data available for emotional state analysis")
        return

    # Classify emotional states
    df['Emotion'] = df.apply(classify_emotional_state, axis=1)

    # Show emotional state distribution
    st.subheader("😊 Emotional State Distribution")

    # Create emotion counts
    emotion_counts = df['Emotion'].value_counts().reset_index()
    emotion_counts.columns = ['Emotion', 'Count']

    # Create pie chart
    emotion_colors = {
        "Positive": "#4CAF50",
        "Neutral": "#9E9E9E",
        "Stressed": "#F44336"
    }

    fig = px.pie(
        emotion_counts,
        names='Emotion',
        values='Count',
        color='Emotion',
        color_discrete_map=emotion_colors,
        hole=0.4,
        title='Emotional State Distribution',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show timeline visualization
    st.subheader("📈 Emotional State Timeline")
    visualize_emotional_state(df)

    # Show sample data with emotions
    st.subheader("🔍 Emotional State Samples")
    st.dataframe(df[['GSR_Value', 'Emotion']].head(20))

    # Create summary statistics
    st.subheader("📊 Emotional State Statistics")
    stats = df.groupby('Emotion')['GSR_Value'].agg(['min', 'max', 'mean', 'std'])
    st.dataframe(stats.style.background_gradient(cmap='Blues'))


# -----------------------------
# Shimmer Data Collection
# -----------------------------
def list_available_ports():
    """List and return all available COM ports."""
    ports = list_ports.comports()
    return [port.device for port in ports]


def handler(pkt: DataPacket, csv_writer, data_queue) -> None:
    """Callback to handle incoming data packets from Shimmer."""
    try:
        timestamp = pkt.timestamp_unix

        def safe_get(channel_type):
            try:
                return pkt[channel_type]
            except KeyError:
                return None

        cur_value_adc = safe_get(EChannelType.INTERNAL_ADC_13)
        cur_value_accel_x = safe_get(EChannelType.ACCEL_LSM303DLHC_X)
        cur_value_accel_y = safe_get(EChannelType.ACCEL_LSM303DLHC_Y)
        cur_value_accel_z = safe_get(EChannelType.ACCEL_LSM303DLHC_Z)
        cur_value_gsr = safe_get(EChannelType.GSR_RAW)
        cur_value_ppg = safe_get(EChannelType.INTERNAL_ADC_13)
        cur_value_gyro_x = safe_get(EChannelType.GYRO_MPU9150_X)
        cur_value_gyro_y = safe_get(EChannelType.GYRO_MPU9150_Y)
        cur_value_gyro_z = safe_get(EChannelType.GYRO_MPU9150_Z)

        # Write to CSV
        csv_writer.writerow([
            timestamp, cur_value_adc,
            cur_value_accel_x, cur_value_accel_y, cur_value_accel_z,
            cur_value_gsr, cur_value_ppg,
            cur_value_gyro_x, cur_value_gyro_y, cur_value_gyro_z
        ])

        # Queue for on-screen sample table
        data_queue.put((timestamp, cur_value_adc, cur_value_accel_x, cur_value_accel_y, cur_value_accel_z,
                        cur_value_gsr, cur_value_ppg, cur_value_gyro_x, cur_value_gyro_y, cur_value_gyro_z))

    except Exception as e:
        print(f"Unexpected error in handler: {e}")


def run_streaming(username, selected_port, duration_seconds):
    """Run the Shimmer streaming session and save to CSV."""
    if not username or not username.strip():
        st.warning("Invalid username")
        return None

    session_csv = os.path.join(DATA_STORAGE_DIR, f"{username}.csv")
    if os.path.exists(session_csv):
        try:
            os.remove(session_csv)
        except Exception as e:
            st.error(f"Could not remove old file: {str(e)}")
            return None

    try:
        with open(session_csv, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "Timestamp", "ADC_Value", "Accel_X", "Accel_Y", "Accel_Z",
                "GSR_Value", "PPG_Value", "Gyro_X", "Gyro_Y", "Gyro_Z"
            ])

            print(f"Connecting to {selected_port}...")
            serial_conn = Serial(selected_port, DEFAULT_BAUDRATE)
            shim_dev = ShimmerBluetooth(serial_conn)

            shim_dev.initialize()
            dev_name = shim_dev.get_device_name()
            print(f"Connected to Shimmer device: {dev_name}")

            shim_dev.add_stream_callback(lambda pkt: handler(pkt, csv_writer, data_queue))

            print("Starting data streaming...")
            shim_dev.start_streaming()
            time.sleep(duration_seconds)
            shim_dev.stop_streaming()
            print("Stopped data streaming.")

            shim_dev.shutdown()
            print("Shimmer device connection closed.")
            print("Data collection complete!")

    except SerialException as e:
        print(f"Serial Error: {e}")
        return None
    except ValueError as e:
        print(f"Invalid COM port: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    return session_csv


# -----------------------------
# Emotion Dashboard
# -----------------------------
def show_emotion_dashboard():
    st.header("🧭 Emotion Dashboard")

    try:
        # Ensure predictions file exists
        if not os.path.exists(PREDICTIONS_FILE):
            st.info("No prediction data available yet.")
            return

        dfp = pd.read_csv(PREDICTIONS_FILE, parse_dates=['WindowStart', 'WindowEnd'])
    except Exception as e:
        st.error(f"Could not read predictions file: {e}")
        return

    if dfp.empty:
        st.info("No prediction data available yet.")
        return

    # Filters
    sessions = ["(All)"] + sorted(dfp['Session'].unique().tolist())
    selected_session = st.selectbox("Session filter", sessions, index=0)
    if selected_session != "(All)":
        dfp = dfp[dfp['Session'] == selected_session]

    # Emotion state cards
    if not dfp.empty:
        latest = dfp.iloc[-1]

        emotion_colors = {
            "Positive": ("#4CAF50", "😊"),
            "Neutral": ("#9E9E9E", "😐"),
            "Stressed": ("#F44336", "😰"),
            "unknown": ("#9E9E9E", "❓")
        }

        color, emoji = emotion_colors.get(latest['Emotion'], ("#9E9E9E", "❓"))

        st.subheader("🧠 Current Emotional State")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {color}">
                <h3 style="color: {color}; margin-top:0">{emoji} {latest['Emotion']}</h3>
                <p>Current State</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #2196F3; margin-top:0">{latest['SDNN_ms']:.1f} ms</h3>
                <p>HRV (SDNN)</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #4CAF50; margin-top:0">{latest['GSR_Mean']:.2f} μS</h3>
                <p>GSR Mean</p>
            </div>
            """, unsafe_allow_html=True)

    # Timeline visualization
    st.subheader("📈 Emotion Timeline")
    if not dfp.empty:
        # Create emotion timeline with Plotly
        emotion_colors = {
            "Positive": "#4CAF50",
            "Neutral": "#9E9E9E",
            "Stressed": "#F44336",
            "unknown": "#9E9E9E"
        }

        # Create a continuous timeline
        fig = px.timeline(
            dfp,
            x_start="WindowStart",
            x_end="WindowEnd",
            y="Emotion",
            color="Emotion",
            color_discrete_map=emotion_colors,
            title="Emotional State Timeline",
            height=400
        )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Emotional State",
            legend_title="Emotions",
            plot_bgcolor='rgba(240,242,246,0.5)'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to display for selected session")

    # Metrics visualization
    st.subheader("📊 Physiological Metrics")
    if not dfp.empty and 'SDNN_ms' in dfp.columns and 'GSR_Mean' in dfp.columns:
        fig = go.Figure()

        # Add SDNN trace
        fig.add_trace(go.Scatter(
            x=dfp['WindowStart'],
            y=dfp['SDNN_ms'],
            name='HRV (SDNN)',
            mode='lines+markers',
            line=dict(color='#6a11cb', width=3),
            marker=dict(size=8, symbol='circle')
        ))

        # Add GSR trace
        fig.add_trace(go.Scatter(
            x=dfp['WindowStart'],
            y=dfp['GSR_Mean'],
            name='GSR Mean',
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='#00c853', width=3),
            marker=dict(size=8, symbol='x')
        ))

        # Set layout
        fig.update_layout(
            title='Physiological Metrics Over Time',
            xaxis_title='Time',
            yaxis=dict(
                title='SDNN (ms)',
                titlefont=dict(color='#6a11cb'),
                tickfont=dict(color='#6a11cb')
            ),
            yaxis2=dict(
                title='GSR Conductance',
                titlefont=dict(color='#00c853'),
                tickfont=dict(color='#00c853'),
                overlaying='y',
                side='right'
            ),
            legend=dict(
                x=0,
                y=1.2,
                orientation='h'
            ),
            plot_bgcolor='rgba(240,242,246,0.5)'
        )

        st.plotly_chart(fig, use_container_width=True)

    # Emotion distribution
    st.subheader("😊 Emotion Distribution")
    if not dfp.empty:
        emotion_counts = dfp['Emotion'].value_counts().reset_index()
        emotion_counts.columns = ['Emotion', 'Count']

        emotion_colors = {
            "Positive": "#4CAF50",
            "Neutral": "#9E9E9E",
            "Stressed": "#F44336",
            "unknown": "#9E9E9E"
        }

        fig = px.pie(
            emotion_counts,
            names='Emotion',
            values='Count',
            color='Emotion',
            color_discrete_map=emotion_colors,
            hole=0.4,
            height=400
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#ffffff', width=2))
        )

        fig.update_layout(
            title='Emotion Distribution',
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    # Recent predictions
    st.subheader("🔎 Recent Predictions")
    st.dataframe(dfp.tail(10))


# -----------------------------
# Emotional State Analysis Tab
# -----------------------------
def show_emotional_state_analysis():
    st.header("😌 Emotional State Analysis")

    uploaded_file = st.file_uploader("Upload your physiological data (CSV format)", type="csv")

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)

            # Show data preview
            st.subheader("📄 Data Preview")
            st.write(df.head())

            # Check for required columns
            required_cols = ['Timestamp', 'GSR_Value', 'PPG_Value']
            if not all(col in df.columns for col in required_cols):
                st.warning("CSV must contain 'Timestamp', 'GSR_Value', and 'PPG_Value' columns")
                return

            # Process the data
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df = df.dropna(subset=['Timestamp'])
            df.set_index('Timestamp', inplace=True)

            # Analyze emotional state
            analyze_emotional_state(df)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a CSV file to analyze emotional states")

        # Show sample data structure
        st.subheader("📋 Expected Data Format")
        sample_data = pd.DataFrame({
            "Timestamp": ["2023-01-01 14:20:00", "2023-01-01 14:20:01", "2023-01-01 14:20:02"],
            "GSR_Value": [1890, 1915, 1930],
            "PPG_Value": [1200, 1250, 1300],
            "Accel_X": [0.5, 0.6, 0.4],
            "Accel_Y": [-0.2, -0.3, -0.1],
            "Accel_Z": [0.8, 0.7, 0.9]
        })
        st.dataframe(sample_data)

        st.markdown("""
        **Column descriptions:**
        - `Timestamp`: Date and time of measurement
        - `GSR_Value`: Galvanic Skin Response value
        - `PPG_Value`: Photoplethysmography value
        - `Accel_X`, `Accel_Y`, `Accel_Z`: Accelerometer values (optional)
        """)


# -----------------------------
# Streamlit App
# -----------------------------
def gsr_ppg_app():
    st.title("🧠 GSR & PPG Emotion Recognition Dashboard")

    # Initialize session state
    if 'last_collected_file' not in st.session_state:
        st.session_state.last_collected_file = None
    if 'window_size' not in st.session_state:
        st.session_state.window_size = 5  # Default 5-second windows

    tabs = st.tabs(["Live Data", "Upload Data", "Emotion Dashboard", "Emotional State Analysis"])

    # ----------------- Live Tab -----------------
    with tabs[0]:
        st.header("📡 Real-time Data Collection")

        col1, col2 = st.columns([3, 2])
        with col1:
            user_name = st.text_input("Session name:", key="user_name_live")
            available_ports = list_available_ports()
            port_name = st.selectbox("COM port:", available_ports, key="port_select_live")
            stream_duration = st.slider("Duration (seconds):", 5, 300, 30, key="duration_live")

        with col2:
            st.session_state.window_size = st.slider("Analysis window (seconds):", 1, 60, 5, key="window_live")
            st.info(f"Using {st.session_state.window_size}-second windows for analysis")

        if st.button("Start Data Collection", key="stream_btn_live", use_container_width=True,
                     help="Begin real-time data acquisition from Shimmer device"):
            if not user_name or not user_name.strip():
                st.warning("Please enter a valid session name")
            elif not port_name:
                st.warning("No COM ports available")
            else:
                with st.spinner(f"Collecting data for {stream_duration} seconds..."):
                    csv_file_path = run_streaming(user_name.strip(), port_name, stream_duration)

                if csv_file_path:
                    st.success("Data collection completed!")
                    st.session_state.last_collected_file = csv_file_path
                    st.write(f"Data saved to: `{csv_file_path}`")

                    # Display sample data
                    collected_samples = []
                    while not data_queue.empty():
                        collected_samples.append(data_queue.get())

                    if collected_samples:
                        # Animated real-time display
                        placeholder = st.empty()
                        st.subheader("📊 Live Data Stream")

                        # Create animated display
                        for i in range(min(20, len(collected_samples))):
                            with placeholder.container():
                                # Show last 5 samples in a table
                                df_live = pd.DataFrame(collected_samples[max(0, i - 4):i + 1], columns=[
                                    "Timestamp", "ADC", "Accel_X", "Accel_Y", "Accel_Z",
                                    "GSR", "PPG", "Gyro_X", "Gyro_Y", "Gyro_Z"
                                ])

                                # Apply styling
                                def gsr_color(val):
                                    intensity = min(100, abs(val) * 2)
                                    return f"background: linear-gradient(90deg, #4CAF50 {intensity}%, transparent {intensity}%);"

                                def ppg_color(val):
                                    intensity = min(100, abs(val) * 2)
                                    return f"background: linear-gradient(90deg, #2196F3 {intensity}%, transparent {intensity}%);"

                                styled_df = df_live.style \
                                    .applymap(lambda x: gsr_color(x) if isinstance(x, (int, float)) else '',
                                              subset=['GSR']) \
                                    .applymap(lambda x: ppg_color(x) if isinstance(x, (int, float)) else '',
                                              subset=['PPG'])

                                st.dataframe(styled_df)

                                # Simple real-time graph
                                if i > 5:
                                    fig, ax = plt.subplots(figsize=(10, 3))
                                    gsr_vals = [s[5] for s in collected_samples[:i + 1]]
                                    ppg_vals = [s[6] for s in collected_samples[:i + 1]]
                                    ax.plot(gsr_vals, label='GSR', color='#4CAF50')
                                    ax.plot(ppg_vals, label='PPG', color='#2196F3')
                                    ax.set_title("Live Signals")
                                    ax.legend(loc='upper right')
                                    ax.grid(True, linestyle='--', alpha=0.7)
                                    st.pyplot(fig)

                            time.sleep(0.2)

                        # 3D Sensor Visualization
                        if len(collected_samples) > 10:
                            st.subheader("🔄 Sensor Orientation")

                            # Create accelerometer visualization
                            accel_data = [(s[2], s[3], s[4]) for s in collected_samples[-20:]]

                            fig = go.Figure(
                                go.Scatter3d(
                                    x=[a[0] for a in accel_data],
                                    y=[a[1] for a in accel_data],
                                    z=[a[2] for a in accel_data],
                                    mode='markers+lines',
                                    marker=dict(
                                        size=5,
                                        color=np.arange(len(accel_data)),
                                        colorscale='Viridis'
                                    ),
                                    line=dict(
                                        color='#6a11cb',
                                        width=2
                                    )
                                )
                            )

                            fig.update_layout(
                                scene=dict(
                                    xaxis_title='X',
                                    yaxis_title='Y',
                                    zaxis_title='Z',
                                    aspectmode='cube'
                                ),
                                height=400,
                                title="Accelerometer Movement"
                            )

                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Data collection failed. Check device connection.")

        if st.session_state.last_collected_file:
            st.markdown("---")
            st.subheader("Analyze Collected Data")
            if st.button("Process and Analyze", key="analyze_collected_live", use_container_width=True):
                try:
                    path = st.session_state.last_collected_file
                    if not os.path.exists(path):
                        st.error(f"File not found: {path}")
                    else:
                        df = pd.read_csv(path)
                        prepared_df = prepare_data(df, session_name=os.path.basename(path))
                        if prepared_df is not None:
                            process_data(
                                prepared_df,
                                st.session_state.window_size,
                                session_name=os.path.basename(path)
                            )
                except Exception as e:
                    st.error(f"Error analyzing data: {str(e)}")

    # ----------------- Upload Tab -----------------
    with tabs[1]:
        st.header("📤 Upload Data for Analysis")

        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("Select CSV file", type="csv", key="file_uploader_demo")
        with col2:
            session_name = st.text_input("Session name:", value="uploaded_session", key="demo_session")
            st.session_state.window_size = st.slider("Analysis window (seconds):", 1, 60, 5, key="window_upload")

        if uploaded_file is not None:
            try:
                # Save uploaded file
                save_path = os.path.join(DATA_STORAGE_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.last_collected_file = save_path

                # Process immediately
                df = pd.read_csv(save_path)
                prepared_df = prepare_data(df, session_name=session_name)
                if prepared_df is not None:
                    process_data(
                        prepared_df,
                        st.session_state.window_size,
                        session_name=session_name
                    )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

        st.header("📂 Analyze Stored Data")
        try:
            stored_files = [f for f in os.listdir(DATA_STORAGE_DIR) if f.lower().endswith('.csv')]
        except Exception as e:
            st.error(f"Error accessing storage: {str(e)}")
            stored_files = []

        if stored_files:
            col1, col2 = st.columns([3, 2])
            with col1:
                selected_file = st.selectbox("Select a file", stored_files, key="stored_files_demo")
            with col2:
                stored_session = st.text_input("Session name:", value=selected_file.split('.')[0], key="stored_session")
                window_size = st.slider("Window size (seconds):", 1, 60, 5, key="window_stored")

            if st.button("Analyze Selected File", key="analyze_stored_demo", use_container_width=True):
                file_path = os.path.join(DATA_STORAGE_DIR, selected_file)
                if not os.path.exists(file_path):
                    st.error(f"File not found: {file_path}")
                else:
                    try:
                        df = pd.read_csv(file_path)
                        prepared_df = prepare_data(df, session_name=stored_session)
                        if prepared_df is not None:
                            process_data(
                                prepared_df,
                                window_size,
                                session_name=stored_session
                            )
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
        else:
            st.info("No stored files available.")

    # ----------------- Dashboard Tab -----------------
    with tabs[2]:
        show_emotion_dashboard()

    # ----------------- Emotional State Analysis Tab -----------------
    with tabs[3]:
        show_emotional_state_analysis()


if __name__ == "__main__":
    gsr_ppg_app()
