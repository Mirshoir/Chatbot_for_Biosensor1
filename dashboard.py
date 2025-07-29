# dashboard.py
# integrated_gsr_ppg_app with ML-based emotional inference and data collection

import os
import time
import csv
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import streamlit as st
import logging
import joblib
from serial import Serial, SerialException
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
from serial.tools import list_ports
from queue import Queue

# Suppress matplotlib warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Sampling rates
gsr_sampling_rate = 1000
ppg_sampling_rate = 1000

DATA_STORAGE_DIR = 'uploaded_data_gsr_ppg'
EMOTIONAL_STATE_FILE = 'emotional_state.txt'

# Ensure data storage path exists
os.makedirs(DATA_STORAGE_DIR, exist_ok=True)
data_queue = Queue()
sdnn_values = []
sdnn_timestamps = []
gsr_values = []
gsr_timestamps = []


# Load ML emotion model
@st.cache_resource
def load_emotion_model():
    model_path = os.path.join(os.path.dirname(__file__), "emotion_rf_model.pkl")
    return joblib.load(model_path)


emotion_model = load_emotion_model()


# Save emotional state with timestamp
def save_emotional_state(emotional_state, sdnn=None, gsr_mean=None, output_file=EMOTIONAL_STATE_FILE):
    dir_path = os.path.dirname(output_file)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, 'a', encoding='utf-8') as file:
        file.write(f"=== Emotional Analysis Entry ===\n")
        file.write(f"Timestamp: {timestamp_str}\n")
        if sdnn is not None:
            file.write(f"HRV_SDNN: {sdnn:.2f} ms\n")
        if gsr_mean is not None:
            file.write(f"GSR_Mean: {gsr_mean:.4f} μS\n")
        file.write(f"State Inference: {emotional_state}\n\n")
    st.success(f"Emotional state and metrics saved in `{output_file}`.")


# ML prediction
def infer_emotional_state_ml(sdnn, gsr_mean):
    if sdnn is None or gsr_mean is None:
        return "Unknown"
    input_data = pd.DataFrame([[sdnn, gsr_mean]], columns=["sdnn", "gsr_mean"])
    return emotion_model.predict(input_data)[0]


# Plot SDNN trend
def show_sdnn_fluctuations():
    if not sdnn_values:
        return
    st.subheader("📈 HRV Fluctuation Dashboard (SDNN)")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(sdnn_timestamps, sdnn_values, marker='o', linestyle='-', color='blue')
    ax.set_ylabel("SDNN (ms)")
    ax.set_xlabel("Time")
    ax.set_title("SDNN (HRV) Over Time")
    ax.grid(True)
    st.pyplot(fig)


# Plot GSR trend
def show_gsr_fluctuations():
    if not gsr_values:
        return
    st.subheader("🧪 GSR Conductance Fluctuation Dashboard")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(gsr_timestamps, gsr_values, marker='o', linestyle='-', color='green')
    ax.set_ylabel("GSR Conductance (normalized)")
    ax.set_xlabel("Time")
    ax.set_title("GSR Fluctuation Over Time")
    ax.grid(True)
    st.pyplot(fig)


# Core analysis function
def analyze_gsr_ppg_data(gsr_ppg_data):
    scaler = StandardScaler()
    gsr_ppg_data['GSR_Skin_Conductance_CAL'] = scaler.fit_transform(gsr_ppg_data[['GSR_Skin_Conductance_CAL']])
    gsr_ppg_data['PPG_A13_CAL'] = scaler.fit_transform(gsr_ppg_data[['PPG_A13_CAL']])

    st.subheader('📄 Data Sample')
    st.write(gsr_ppg_data.head())

    # HRV/SDNN
    st.header('PPG HRV Analysis')
    ppg_signal = gsr_ppg_data['PPG_A13_CAL'].values
    ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=ppg_sampling_rate)
    ppg_peaks = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=ppg_sampling_rate)
    ppg_peak_indices = ppg_peaks['PPG_Peaks']

    if len(ppg_peak_indices) > 1:
        intervals = np.diff(ppg_peak_indices) * (1000 / ppg_sampling_rate)
        sdnn = np.std(intervals)
        st.write(f"🔹 SDNN: {sdnn:.2f} ms")
        sdnn_values.append(sdnn)
        sdnn_timestamps.append(datetime.datetime.now().strftime('%H:%M:%S'))
        show_sdnn_fluctuations()
    else:
        st.warning("Not enough PPG peaks to compute SDNN.")
        sdnn = None

    # GSR
    st.header("GSR Analysis")
    gsr_current = np.mean(gsr_ppg_data['GSR_Skin_Conductance_CAL'].values)
    gsr_values.append(gsr_current)
    gsr_timestamps.append(datetime.datetime.now().strftime('%H:%M:%S'))
    show_gsr_fluctuations()

    # Emotion Inference
    st.header("🧠 Predicted Emotional State")
    if sdnn is not None:
        emotion = infer_emotional_state_ml(sdnn, gsr_current)
        st.markdown(f"### 👉 **{emotion.upper()}**")
        save_emotional_state(emotion, sdnn, gsr_current)
    else:
        st.warning("Cannot predict emotional state without HRV data")


# ------------------------------------------
# Shimmer Data Collection Functions
# ------------------------------------------

def list_available_ports():
    """List and return all available COM ports."""
    ports = list_ports.comports()
    return [port.device for port in ports]


def handler(pkt: DataPacket, csv_writer, data_queue) -> None:
    """
    Callback function to handle incoming data packets from the Shimmer device.
    """
    try:
        timestamp = pkt.timestamp_unix

        # Safely extract channel data
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

        # Write data to the CSV file
        csv_writer.writerow([
            timestamp, cur_value_adc,
            cur_value_accel_x, cur_value_accel_y, cur_value_accel_z,
            cur_value_gsr, cur_value_ppg,
            cur_value_gyro_x, cur_value_gyro_y, cur_value_gyro_z
        ])

        # Put data into the queue for the main thread to read
        data_queue.put((timestamp, cur_value_adc, cur_value_accel_x, cur_value_accel_y, cur_value_accel_z,
                        cur_value_gsr, cur_value_ppg, cur_value_gyro_x, cur_value_gyro_y, cur_value_gyro_z))

    except Exception as e:
        print(f"Unexpected error in handler: {e}")


def run_streaming(username, selected_port, duration_seconds):
    """Run the Shimmer data streaming session and save to a CSV named <username>.csv."""
    csv_file_path = os.path.join(DATA_STORAGE_DIR, f"{username}.csv")
    # If the file exists, remove it before starting a new session
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row
        csv_writer.writerow([
            "Timestamp", "ADC_Value", "Accel_X", "Accel_Y", "Accel_Z",
            "GSR_Value", "PPG_Value", "Gyro_X", "Gyro_Y", "Gyro_Z"
        ])

        try:
            print(f"Connecting to {selected_port}...")
            serial_conn = Serial(selected_port, DEFAULT_BAUDRATE)
            shim_dev = ShimmerBluetooth(serial_conn)

            # Initialize Shimmer device
            shim_dev.initialize()
            dev_name = shim_dev.get_device_name()
            print(f"Connected to Shimmer device: {dev_name}")

            # Add callback for incoming data
            shim_dev.add_stream_callback(lambda pkt: handler(pkt, csv_writer, data_queue))

            # Start streaming
            print("Starting data streaming...")
            shim_dev.start_streaming()
            time.sleep(duration_seconds)
            shim_dev.stop_streaming()
            print("Stopped data streaming.")

            # Shut down the device connection
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
    return csv_file_path


def process_and_analyze(df):
    col_map = {'Timestamp': 'Timestamp_Unix_CAL',
               'GSR_Value': 'GSR_Skin_Conductance_CAL',
               'PPG_Value': 'PPG_A13_CAL'}
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Rename columns only if needed
    for orig, new in col_map.items():
        if orig in df.columns and new not in df.columns:
            df.rename(columns={orig: new}, inplace=True)

    required_cols = ['Timestamp_Unix_CAL', 'GSR_Skin_Conductance_CAL', 'PPG_A13_CAL']
    
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    # Handle timestamp conversion safely
    df['Timestamp_Unix_CAL'] = pd.to_numeric(df['Timestamp_Unix_CAL'], errors='coerce')
    
    # Remove rows with invalid timestamps
    df = df.dropna(subset=['Timestamp_Unix_CAL'])
    
    if df.empty:
        st.error("No valid data after timestamp processing")
        return

    # Create datetime index
    df['Timestamp'] = pd.to_datetime(df['Timestamp_Unix_CAL'], unit='ms', errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    
    if df.empty:
        st.error("No valid timestamps after conversion")
        return
        
    df = df.set_index('Timestamp', drop=True)
    analyze_gsr_ppg_data(df)


# Main dashboard app
def gsr_ppg_app():
    st.title("GSR and PPG Real-Time Emotion Dashboard")

    # Initialize session state for last collected file
    if 'last_collected_file' not in st.session_state:
        st.session_state.last_collected_file = None

    # ------------------------------------------
    # Data Acquisition from Shimmer Device
    # ------------------------------------------
    st.header("📡 Real-time Data Collection from Shimmer Device")
    user_name = st.text_input("Enter your name for data labeling:", key="user_name")
    available_ports = list_available_ports()
    port_name = st.selectbox("Select the COM port for the Shimmer device:", available_ports, key="port_select")
    stream_duration = st.number_input("Enter the streaming duration in seconds:", min_value=1, value=5, key="duration")

    if st.button("Start Streaming", key="stream_btn"):
        if not user_name.strip():
            st.warning("Please enter a valid name before starting.")
        elif not port_name:
            st.warning("No COM ports available or selected.")
        else:
            with st.spinner(f"Collecting data from Shimmer device for {stream_duration} seconds..."):
                csv_file_path = run_streaming(user_name.strip(), port_name, stream_duration)

            if csv_file_path:
                st.success("Data collection completed!")
                st.session_state.last_collected_file = csv_file_path
                st.write(f"Data saved as `{os.path.basename(csv_file_path)}` in `{DATA_STORAGE_DIR}`.")

                # Display sample data
                collected_samples = []
                while not data_queue.empty():
                    collected_samples.append(data_queue.get())

                if collected_samples:
                    st.subheader("Sample Collected Data")
                    st.write(pd.DataFrame(collected_samples[:5], columns=[
                        "Timestamp", "ADC_Value", "Accel_X", "Accel_Y", "Accel_Z",
                        "GSR_Value", "PPG_Value", "Gyro_X", "Gyro_Y", "Gyro_Z"
                    ]))
            else:
                st.error("Data collection failed. Please check the device connection.")

    # Analyze last collected file
    if st.session_state.last_collected_file:
        if st.button("Analyze Collected Data", key="analyze_collected"):
            try:
                df = pd.read_csv(st.session_state.last_collected_file)
                process_and_analyze(df)
            except Exception as e:
                st.error(f"Error analyzing collected data: {str(e)}")

    # ------------------------------------------
    # Upload Data for Analysis
    # ------------------------------------------
    st.header("📤 Upload Data for Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")

    if uploaded_file is not None:
        try:
            # Save uploaded file to storage directory
            save_path = os.path.join(DATA_STORAGE_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the saved file
            st.session_state.last_collected_file = save_path
            df = pd.read_csv(save_path)
            st.success(f"File saved successfully: {uploaded_file.name}")
            process_and_analyze(df)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    # ------------------------------------------
    # Re-analyze Stored Data
    # ------------------------------------------
    st.header("📂 Re-analyze Stored Data")
    stored_files = [f for f in os.listdir(DATA_STORAGE_DIR) if f.endswith('.csv')]

    if stored_files:
        file_options = ["-- Select a file --"] + stored_files
        selected_file = st.selectbox("Select a file", file_options, key="stored_files")

        if st.button("Analyze Selected File", key="analyze_stored"):
            if selected_file == "-- Select a file --":
                st.error("Please select a valid CSV file from the list.")
            else:
                file_path = os.path.join(DATA_STORAGE_DIR, selected_file)
                if not os.path.isfile(file_path):
                    st.error(f"File not found: {file_path}")
                else:
                    try:
                        df = pd.read_csv(file_path)
                        process_and_analyze(df)
                    except Exception as e:
                        st.error(f"Error processing stored file: {str(e)}")
    else:
        st.info("No stored files available.")


if __name__ == "__main__":
    gsr_ppg_app()
