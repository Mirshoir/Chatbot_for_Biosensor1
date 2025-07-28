import joblib
import streamlit as st
import os
import time
import csv
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import logging
from serial import Serial, SerialException
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
from serial.tools import list_ports
from queue import Queue
import threading

# Suppress warnings from matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# ------------------------------------------
# Global Variables and Setup
# ------------------------------------------

# Define sampling rate for ECG (known value)
ECG_SAMPLING_RATE = 500  # in Hz

# Directory to store uploaded or collected ECG data
DATA_STORAGE_DIR = 'uploaded_data_ecg'

# Create the directory if it doesn't exist
if not os.path.exists(DATA_STORAGE_DIR):
    os.makedirs(DATA_STORAGE_DIR)

# Log file path
ANALYSIS_LOG_FILE = os.path.join(DATA_STORAGE_DIR, 'ecgAnalysisSteps.txt')

# Configure logging to file
logging.basicConfig(
    filename=os.path.join(DATA_STORAGE_DIR, 'app.log'),
    level=logging.ERROR,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


# ------------------------------------------
# Functions
# ------------------------------------------

def save_model(model, filepath):
    """
    Save a model to a file using joblib.
    """
    try:
        joblib.dump(model, filepath)
        st.success(f"Model saved to {filepath}")
    except Exception as e:
        # Log the exception instead of displaying it
        logging.error(f"Error saving model: {e}")


def load_model(filepath):
    """
    Load a model from a file using joblib.
    """
    try:
        return joblib.load(filepath)
    except Exception as e:
        # Log the exception instead of displaying it
        logging.error(f"Error loading model: {e}")
        return None


def adc_to_voltage(adc_value, resolution=24, v_ref=3.3):
    """
    Convert raw ADC counts to voltage.
    """
    max_adc = 2 ** resolution - 1
    return (adc_value / max_adc) * v_ref


def save_analysis_steps(stress_level, file_path):
    """
    Save the estimated stress level to a text file.
    """
    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(file_path, 'a') as log_file:
            log_file.write(f"[{timestamp_str}] Estimated Stress Level: {stress_level}\n")
    except Exception as e:
        # Log the exception instead of displaying it
        logging.error(f"Error saving analysis steps: {e}")


def infer_sampling_rate(timestamps):
    """
    (Optional) Infer the sampling rate from the timestamps' column (in seconds).
    If timestamps are constant or invalid, return None.
    """
    if len(timestamps) < 2:
        return None
    time_differences = np.diff(timestamps)
    # Filter out zero or negative intervals
    time_differences = time_differences[time_differences > 0]
    if len(time_differences) == 0:
        return None
    median_interval = np.median(time_differences)
    if median_interval <= 0:
        return None
    # Calculate sampling rate in Hz
    sampling_rate = int(round(1.0 / median_interval))
    # Validate sampling rate within plausible range
    if sampling_rate < 100 or sampling_rate > 10000:
        return None
    return sampling_rate


def analyze_ecg_data(ecg_data, sampling_rate):
    """
    Perform ECG data analysis and HRV computation.
    """
    # Initialize session state for analysis steps logging
    if 'analysis_log_initialized' not in st.session_state:
        st.session_state['analysis_log_initialized'] = False

    # Default stress level
    stress_level = "Normal Stress"

    try:
        # Check and derive 'ECG_CAL' if not present
        if 'ECG_CAL' not in ecg_data.columns:
            st.warning("'ECG_CAL' column is missing. Deriving 'ECG_CAL' from 'ECG_CH1'.")
            ecg_data['ECG_CAL'] = ecg_data['ECG_CH1']

        # Convert ADC counts to voltage
        ecg_data['ECG_CAL_Voltage'] = ecg_data['ECG_CAL'].apply(adc_to_voltage)

        # Display ECG_CAL_Voltage statistics
        ecg_cal_min = ecg_data['ECG_CAL_Voltage'].min()
        ecg_cal_max = ecg_data['ECG_CAL_Voltage'].max()
        ecg_cal_mean = ecg_data['ECG_CAL_Voltage'].mean()
        ecg_cal_std = ecg_data['ECG_CAL_Voltage'].std()

        st.write(f"**ECG_CAL_Voltage Statistics:**")
        st.write(f"- **Min**: {ecg_cal_min:.3f} V")
        st.write(f"- **Max**: {ecg_cal_max:.3f} V")
        st.write(f"- **Mean**: {ecg_cal_mean:.3f} V")
        st.write(f"- **Standard Deviation**: {ecg_cal_std:.3f} V")

        # Proceed only if standard deviation is above a threshold (e.g., 0.01 V)
        if ecg_cal_std < 0.01:
            st.warning("ECG_CAL_Voltage has very low variability. Setting stress level to **Normal Stress**.")
            save_analysis_steps(stress_level, ANALYSIS_LOG_FILE)
            if not st.session_state['analysis_log_initialized']:
                st.session_state['analysis_log_initialized'] = True
            st.subheader('Estimated Stress Level')
            st.write(f"Based on HRV analysis, the estimated stress level is: **{stress_level}**")
            return

        # Standardize the ECG_CAL_Voltage data
        scaler = StandardScaler()
        ecg_data['ECG_CAL_Standardized'] = scaler.fit_transform(ecg_data[['ECG_CAL_Voltage']])

        st.subheader('Data Sample')
        st.write(ecg_data[['Timestamp', 'ECG_CH1', 'ECG_CH2', 'ECG_CAL_Voltage', 'ECG_CAL_Standardized']].head())

        st.subheader('ECG Signal (Standardized Voltage)')
        st.line_chart(ecg_data['ECG_CAL_Standardized'])

        st.header('ECG Signal Processing and Heart Rate Variability (HRV) Analysis')

        ecg_signal = ecg_data['ECG_CAL_Standardized'].values

        # Ensure sufficient data length for HRV computation
        MIN_DURATION = 30  # Minimum 30 seconds of data
        min_samples = MIN_DURATION * sampling_rate

        if len(ecg_signal) < min_samples:
            st.warning(
                f"Insufficient data length ({len(ecg_signal)} samples). Setting stress level to **Normal Stress**.")
            save_analysis_steps(stress_level, ANALYSIS_LOG_FILE)
            if not st.session_state['analysis_log_initialized']:
                st.session_state['analysis_log_initialized'] = True
            st.subheader('Estimated Stress Level')
            st.write(f"Based on HRV analysis, the estimated stress level is: **{stress_level}**")
            return

        # Clean the ECG signal
        try:
            ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
        except Exception as e:
            st.warning("Error during ECG cleaning. Setting stress level to **Normal Stress**.")
            save_analysis_steps(stress_level, ANALYSIS_LOG_FILE)
            if not st.session_state['analysis_log_initialized']:
                st.session_state['analysis_log_initialized'] = True
            st.subheader('Estimated Stress Level')
            st.write(f"Based on HRV analysis, the estimated stress level is: **{stress_level}**")
            return

        # Detect R-peaks
        try:
            ecg_peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        except Exception as e:
            st.warning("Error during ECG peak detection. Setting stress level to **Normal Stress**.")
            save_analysis_steps(stress_level, ANALYSIS_LOG_FILE)
            if not st.session_state['analysis_log_initialized']:
                st.session_state['analysis_log_initialized'] = True
            st.subheader('Estimated Stress Level')
            st.write(f"Based on HRV analysis, the estimated stress level is: **{stress_level}**")
            return

        r_peak_indices = ecg_peaks['ECG_R_Peaks']
        num_peaks = len(r_peak_indices)
        st.write(f"Number of R-peaks detected: {num_peaks}")

        # Define a minimum number of R-peaks required for HRV analysis
        MIN_RPEAKS = 10  # Adjust based on requirements
        if num_peaks < MIN_RPEAKS:
            st.warning(
                f"Number of R-peaks detected ({num_peaks}) is below the minimum required ({MIN_RPEAKS}) for reliable HRV analysis.\n"
                f"Setting stress level to **Normal Stress**."
            )
            save_analysis_steps(stress_level, ANALYSIS_LOG_FILE)
            if not st.session_state['analysis_log_initialized']:
                st.session_state['analysis_log_initialized'] = True
            st.subheader('Estimated Stress Level')
            st.write(f"Based on HRV analysis, the estimated stress level is: **{stress_level}**")
            return

        if num_peaks > 0:
            # Plot ECG signal with R-peaks
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                time_ecg = np.arange(len(ecg_cleaned)) / sampling_rate
                ax.plot(time_ecg, ecg_cleaned, label='ECG Signal')
                ax.scatter(time_ecg[r_peak_indices], ecg_cleaned[r_peak_indices], color='red', label='R-peaks')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude (Standardized)')
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.warning("Error while plotting ECG signal. Proceeding with HRV analysis.")

            # Compute HRV metrics
            try:
                ecg_hrv = nk.hrv_time(ecg_peaks, sampling_rate=sampling_rate)
                if ecg_hrv.empty:
                    st.warning("HRV DataFrame is empty. Setting stress level to **Normal Stress**.")
                    save_analysis_steps(stress_level, ANALYSIS_LOG_FILE)
                    if not st.session_state['analysis_log_initialized']:
                        st.session_state['analysis_log_initialized'] = True
                    st.subheader('Estimated Stress Level')
                    st.write(f"Based on HRV analysis, the estimated stress level is: **{stress_level}**")
                    return

                st.subheader('ECG HRV Features')
                st.write(ecg_hrv)

                # Check for 'HRV_SDNN'
                if 'HRV_SDNN' in ecg_hrv.columns and not ecg_hrv['HRV_SDNN'].dropna().empty:
                    hrv_sdnn_series = ecg_hrv['HRV_SDNN'].dropna()
                    if not hrv_sdnn_series.empty:
                        hrv_sdnn_value = hrv_sdnn_series.iloc[0]
                        st.write(f"**HRV_SDNN**: {hrv_sdnn_value:.2f} ms")

                        # Plot HRV_SDNN
                        try:
                            hrv_fig, hrv_ax = plt.subplots(figsize=(6, 4))
                            hrv_ax.bar(['HRV_SDNN'], [hrv_sdnn_value], color='orange')
                            hrv_ax.set_ylabel('SDNN (ms)')
                            hrv_ax.set_title('HRV_SDNN')
                            st.pyplot(hrv_fig)
                        except Exception as e:
                            st.warning("Error while plotting HRV_SDNN.")

                        # Define population-based thresholds (hypothetical values)
                        if hrv_sdnn_value < 50:
                            stress_level = 'High Stress'
                        elif hrv_sdnn_value > 100:
                            stress_level = 'Low Stress'
                        else:
                            stress_level = 'Moderate Stress'
                    else:
                        st.warning("All 'HRV_SDNN' values are NaN. Setting stress level to **Normal Stress**.")
                else:
                    st.warning(
                        "No 'HRV_SDNN' feature found or all values are NaN. Setting stress level to **Normal Stress**.")

                # Assign default stress level if not set by HRV_SDNN
                if 'stress_level' not in locals():
                    stress_level = "Normal Stress"

                # Save analysis steps only once
                if not st.session_state['analysis_log_initialized']:
                    save_analysis_steps(stress_level, ANALYSIS_LOG_FILE)
                    st.session_state['analysis_log_initialized'] = True

                # Display the stress level
                st.subheader('Estimated Stress Level')
                st.write(f"Based on HRV analysis, the estimated stress level is: **{stress_level}**")
            except Exception as e:
                # In case of any unforeseen errors during HRV computation
                st.warning(
                    "An unexpected error occurred during HRV computation. Setting stress level to **Normal Stress by default**.")
                save_analysis_steps(stress_level, ANALYSIS_LOG_FILE)
                if not st.session_state['analysis_log_initialized']:
                    st.session_state['analysis_log_initialized'] = True
                st.subheader('Estimated Stress Level')
                st.write(f"Based on HRV analysis, the estimated stress level is: **{stress_level}**")
        else:
            st.warning("No R-peaks detected in the ECG signal. Setting stress level to **Normal Stress**.")
            save_analysis_steps(stress_level, ANALYSIS_LOG_FILE)
            if not st.session_state['analysis_log_initialized']:
                st.session_state['analysis_log_initialized'] = True
            st.subheader('Estimated Stress Level')
            st.write(f"Based on HRV analysis, the estimated stress level is: **{stress_level}**")
    except:
        st.write(f"Based on HRV analysis, the estimated stress level is: **{stress_level}**")


def list_available_ports():
    """List and return all available COM ports."""
    ports = list_ports.comports()
    return [port.device for port in ports]


def ecg_handler(pkt: DataPacket, csv_writer):
    try:
        # Extract ECG channels from the data packet
        ecg_ch1 = pkt[EChannelType.EXG_ADS1292R_1_CH1_24BIT]
        ecg_ch2 = pkt[EChannelType.EXG_ADS1292R_1_CH2_24BIT]
        timestamp = pkt.timestamp_unix  # In seconds with fractional part

        # Derive ECG_CAL from ECG_CH1 and ECG_CH2
        ecg_cal = ecg_ch1  # Using ECG_CH1 as ECG_CAL for simplicity

        # Debugging: Print timestamp and ECG values to ensure they are varying
        print(f"Timestamp: {timestamp}, ECG_CH1: {ecg_ch1}, ECG_CH2: {ecg_ch2}, ECG_CAL: {ecg_cal}")

        # Write ECG data to CSV
        csv_writer.writerow([timestamp, ecg_ch1, ecg_ch2, ecg_cal])

    except KeyError as e:
        # If the device isn't streaming these particular channels, you may need to configure the sensors
        print(f"ECG channel not found in packet: {e}")


def connect_shimmer(selected_port):
    """
    Connect to a Shimmer device on the specified COM port.
    Returns:
        shim_dev: ShimmerBluetooth object if successful, else None
    """
    try:
        print(f"Trying to connect to Shimmer device on port: {selected_port}")
        serial_conn = Serial(selected_port, DEFAULT_BAUDRATE, timeout=None)  # timeout=None as per previous fixes
        shim_dev = ShimmerBluetooth(serial_conn)
        shim_dev.initialize()
        dev_name = shim_dev.get_device_name()
        print(f"Connected to Shimmer device: {dev_name} on {selected_port}")
        return shim_dev
    except SerialException as e:
        # Log the exception instead of displaying it
        logging.error(f"Serial Error on port {selected_port}: {e}")
    except Exception as e:
        # Log the exception instead of displaying it
        logging.error(f"Could not connect on port {selected_port}: {e}")
    return None


def collect_ecg_data(shim_dev, duration, csv_file_path, stop_event):
    """
    Collect ECG data from the Shimmer device for a specified duration and save to CSV.
    """
    try:
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Timestamp", "ECG_CH1", "ECG_CH2", "ECG_CAL"])  # Updated header to include ECG_CAL
            print(f"Saving ECG data to {csv_file_path}...")

            # Add callback for incoming data
            shim_dev.add_stream_callback(lambda pkt: ecg_handler(pkt, csv_writer))

            # Start streaming
            print("Starting ECG data streaming...")
            shim_dev.start_streaming()

            # Collect data for the specified duration
            start_time = time.time()
            while time.time() - start_time < duration:
                if stop_event.is_set():
                    print("Stop event set. Terminating data collection.")
                    break
                time.sleep(1)  # Sleep to prevent blocking

            # Stop streaming
            shim_dev.stop_streaming()
            print("Stopped data streaming.")

            # Shutdown device connection
            shim_dev.shutdown()
            print("Shimmer device connection closed.")
    except Exception as e:
        # Log the exception instead of displaying it
        logging.error(f"An error occurred during data collection: {e}")


def start_data_collection(selected_port, duration, csv_file_path):
    """
    Start ECG data collection in a separate thread to avoid blocking Streamlit.
    Returns:
        collection_thread: threading.Thread object
        stop_event: threading.Event object
    """
    shim_dev = connect_shimmer(selected_port)
    if shim_dev is None:
        return None, None

    # Create a stop event to allow stopping the thread if needed
    stop_event = threading.Event()

    # Start the data collection in a separate thread
    collection_thread = threading.Thread(target=collect_ecg_data, args=(shim_dev, duration, csv_file_path, stop_event))
    collection_thread.start()

    return collection_thread, stop_event


def analyze_uploaded_data(ecg_data):
    """
    Preprocess and analyze the uploaded ECG data.
    """
    # Preprocess and analyze the data
    analyze_ecg_data(ecg_data, ECG_SAMPLING_RATE)


def ecg_app():
    """
    Streamlit app for uploading and analyzing ECG data.
    """
    st.title("Integrated ECG Data Collection and Analysis for Stress Level Estimation")
    st.write("""
    This application allows you to **collect ECG data** from a Shimmer device and **analyze** it to estimate stress levels based on Heart Rate Variability (HRV).
    """)

    # ------------------------------------------
    # ECG Data Collection Section
    # ------------------------------------------
    st.header("ECG Data Collection from Shimmer Device")
    user_name = st.text_input("Enter your name for data labeling:", "")
    available_ports = list_available_ports()
    if available_ports:
        port_name = st.selectbox("Select the COM port for the Shimmer device:", available_ports)
    else:
        port_name = ""
        st.warning("No COM ports detected. Please connect your Shimmer device.")

    stream_duration = st.number_input(
        "Enter the streaming duration in seconds:",
        min_value=10,
        max_value=3600,
        value=300,  # Increased default to 300 seconds (5 minutes)
        step=10
    )

    # Initialize session state inside the app
    if 'is_collecting' not in st.session_state:
        st.session_state['is_collecting'] = False

    if st.button("Start ECG Data Collection"):
        if user_name.strip() == "":
            st.warning("Please enter a valid name before starting.")
        elif not port_name:
            st.warning("No COM ports available or selected.")
        else:
            if st.session_state['is_collecting']:
                st.warning("Data collection is already in progress. Please wait until it's completed.")
            else:
                # Prepare CSV file path
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file_path = os.path.join(DATA_STORAGE_DIR,
                                             f"ecg_data_{user_name}_{timestamp_str}.csv")

                # Start data collection in a separate thread
                collection_thread, stop_event = start_data_collection(port_name, stream_duration, csv_file_path)
                if collection_thread:
                    st.session_state['is_collecting'] = True
                    st.success(f"Started data collection for {stream_duration} seconds. Please wait...")

                    # Display a progress bar
                    progress_bar = st.progress(0)
                    for i in range(stream_duration):
                        if not collection_thread.is_alive():
                            st.warning("Data collection thread has terminated unexpectedly.")
                            break
                        progress = (i + 1) / stream_duration
                        progress_bar.progress(progress)
                        time.sleep(1)  # Update every second
                    progress_bar.empty()

                    st.session_state['is_collecting'] = False
                    st.success(f"Data collection completed and saved to `{csv_file_path}`.")

                    # Optionally, display some sample data
                    try:
                        ecg_data = pd.read_csv(csv_file_path)
                        st.subheader("Sample Data")
                        st.write(ecg_data.head())
                        analyze_uploaded_data(ecg_data)
                    except Exception as e:
                        # Log the exception instead of displaying it
                        logging.error(f"Error reading the saved data: {e}")
    st.markdown("---")

    # ------------------------------------------
    # Upload or Use Existing Data Section
    # ------------------------------------------
    st.header("Upload or Use Existing Data for Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    def process_and_analyze(dataframe):
        # Preprocess and analyze the data
        analyze_uploaded_data(dataframe)

    # If a file is uploaded, process it
    if uploaded_file:
        try:
            ecg_data = pd.read_csv(uploaded_file)
            # Check if 'ECG_CAL' exists; if not, derive it
            if 'ECG_CAL' not in ecg_data.columns:
                st.warning("'ECG_CAL' column is missing. Deriving 'ECG_CAL' from 'ECG_CH1'.")
                ecg_data['ECG_CAL'] = ecg_data['ECG_CH1']
            process_and_analyze(ecg_data)
        except Exception as e:
            # Log the exception instead of displaying it
            logging.error(f"Error processing file: {e}")
            st.warning("There was an issue processing the uploaded file. Please ensure it is correctly formatted.")
    else:
        st.info("Upload a file to start analysis or select from previously saved data below.")

    st.markdown("---")

    # ------------------------------------------
    # Re-analyze Stored Data Section
    # ------------------------------------------
    st.header("Re-analyze Stored Data")
    stored_files = [f for f in os.listdir(DATA_STORAGE_DIR) if f.endswith('.csv') and f != 'ecgAnalysisSteps.txt']

    if stored_files:
        selected_file = st.selectbox("Select a file", stored_files)
        if st.button('Analyze Selected File'):
            try:
                file_path = os.path.join(DATA_STORAGE_DIR, selected_file)
                ecg_data = pd.read_csv(file_path)
                # Check if 'ECG_CAL' exists; if not, derive it
                if 'ECG_CAL' not in ecg_data.columns:
                    st.warning("'ECG_CAL' column is missing. Deriving 'ECG_CAL' from 'ECG_CH1'.")
                    ecg_data['ECG_CAL'] = ecg_data['ECG_CH1']
                process_and_analyze(ecg_data)
            except Exception as e:
                # Log the exception instead of displaying it
                logging.error(f"Error processing stored file: {e}")
                st.warning("There was an issue processing the selected file. Please ensure it is correctly formatted.")
    else:
        st.write("No stored data files available. Please collect ECG data first.")


if __name__ == "__main__":
    ecg_app()
