import os
import time
import csv
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import streamlit as st
import logging
from logging.handlers import RotatingFileHandler  # Import RotatingFileHandler
from serial import Serial, SerialException
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
from serial.tools import list_ports
from queue import Queue
import pyshimmer

# Suppress warnings from matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# ------------------------------------------
# Global Variables and Setup
# ------------------------------------------

# Define sampling rate for EMG (adjust as needed)
emg_sampling_rate = 1000  # in Hz

# Directory to store uploaded or collected EMG data
DATA_STORAGE_DIR = 'uploaded_data_emg'

# Create the directory if it doesn't exist
if not os.path.exists(DATA_STORAGE_DIR):
    os.makedirs(DATA_STORAGE_DIR)

# A global queue for inter-thread communication (Shimmer data)
data_queue = Queue()

# ------------------------------------------
# Logger Configuration
# ------------------------------------------

# Create a logger for EMG analysis
emg_logger = logging.getLogger('EMGAnalysisLogger')
emg_logger.setLevel(logging.INFO)  # Set the logging level to INFO

# Path to the log file
log_file_path = os.path.join(DATA_STORAGE_DIR,
                             'C:\\Users\\dtfygu876\\prompt_codes\\csvChunking\\Chatbot_for_Biosensor\\emgAnalysisSteps.txt')

# Create a rotating file handler that appends and rotates the log file
rotating_handler = RotatingFileHandler(
    log_file_path,
    mode='a',  # Append mode
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5,  # Keep up to 5 backup files
    encoding='utf-8',
    delay=0
)
rotating_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
rotating_handler.setFormatter(formatter)

# Add the rotating file handler to the logger
emg_logger.addHandler(rotating_handler)


# ------------------------------------------
# Functions
# ------------------------------------------

def list_available_ports():
    """List and return all available COM ports."""
    ports = list_ports.comports()
    return [port.device for port in ports]


def handler(pkt: DataPacket, csv_writer, data_queue) -> None:
    """
    Callback function to handle incoming data packets from the Shimmer device.

    Args:
        pkt (DataPacket): The data packet received from the Shimmer device.
        csv_writer (csv.writer): CSV writer object to save the data.
        data_queue (Queue): Queue to store incoming data for real-time processing.
    """
    try:
        timestamp = pkt.timestamp_unix

        # Extract EMG channel data using dictionary-like access
        emg_ch1_1 = pkt[EChannelType.EXG_ADS1292R_1_CH1_24BIT]
        emg_ch1_2 = pkt[EChannelType.EXG_ADS1292R_1_CH2_24BIT]
        emg_ch2_1 = pkt[EChannelType.EXG_ADS1292R_2_CH1_24BIT]
        emg_ch2_2 = pkt[EChannelType.EXG_ADS1292R_2_CH2_24BIT]

        # Write data to the CSV file
        csv_writer.writerow([timestamp, emg_ch1_1, emg_ch1_2, emg_ch2_1, emg_ch2_2])

        # Put data into the queue for the main thread to read (if needed for real-time processing)
        data_queue.put((timestamp, emg_ch1_1, emg_ch1_2, emg_ch2_1, emg_ch2_2))

    except KeyError as e:
        # Handle missing channels gracefully
        print(f"Channel not found in data packet: {e}")
    except Exception as e:
        print(f"Unexpected error in handler: {e}")


def run_streaming(username, selected_port, duration_seconds):
    """
    Run the Shimmer data streaming session and save to a CSV named <username>_<timestamp>.csv.

    Args:
        username (str): Username for labeling the data file.
        selected_port (str): COM port to connect to the Shimmer device.
        duration_seconds (int): Duration for data streaming in seconds.

    Returns:
        str or None: Path to the saved CSV file, or None if an error occurred.
    """
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_path = os.path.join(DATA_STORAGE_DIR, f"{username}_{timestamp_str}.csv")

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row
        csv_writer.writerow(["Timestamp", "EMG_CH1_1", "EMG_CH1_2", "EMG_CH2_1", "EMG_CH2_2"])

        try:
            st.write(f"Connecting to {selected_port}...")
            emg_logger.info(f"Connecting to Shimmer device on port {selected_port}.")
            # Set timeout=None to avoid the warning and allow proper read cancellation
            serial_conn = Serial(selected_port, DEFAULT_BAUDRATE, timeout=None)
            shim_dev = ShimmerBluetooth(serial_conn)

            # Initialize Shimmer device
            shim_dev.initialize()
            dev_name = shim_dev.get_device_name()
            st.write(f"Connected to Shimmer device: {dev_name}")
            emg_logger.info(f"Connected to Shimmer device: {dev_name}")

            # Add callback for incoming data
            shim_dev.add_stream_callback(lambda pkt: handler(pkt, csv_writer, data_queue))

            # Start streaming
            st.write("Starting data streaming...")
            emg_logger.info("Starting data streaming.")
            shim_dev.start_streaming()

            # Display a progress bar
            progress_bar = st.progress(0)
            start_time = time.time()
            while time.time() - start_time < duration_seconds:
                elapsed = time.time() - start_time
                progress = min(elapsed / duration_seconds, 1.0)
                progress_bar.progress(progress)
                time.sleep(0.1)  # Update every 100ms
            shim_dev.stop_streaming()
            progress_bar.empty()
            st.write("Stopped data streaming.")
            emg_logger.info("Stopped data streaming.")

            # Shut down the device connection
            shim_dev.shutdown()
            st.write("Shimmer device connection closed.")
            emg_logger.info("Shimmer device connection closed.")
            st.success("Data collection complete!")
            emg_logger.info("Data collection complete.")

        except SerialException as e:
            st.error(f"Serial Error: {e}")
            emg_logger.error(f"Serial Error: {e}")
            return None
        except ValueError as e:
            st.error(f"Invalid COM port: {e}")
            emg_logger.error(f"Invalid COM port: {e}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            emg_logger.error(f"An unexpected error occurred: {e}")
            return None

    return csv_file_path


def analyze_emg_data(emg_data):
    """
    Perform analysis on EMG data, including signal cleaning, envelope extraction,
    muscle activation estimation, and visualization.

    Args:
        emg_data (pd.DataFrame): DataFrame containing EMG data.
    """
    st.header('EMG Data Analysis')

    # Path to the log file
    log_file_path = os.path.join(DATA_STORAGE_DIR, 'C:\\Users\\dtfygu876\\prompt_codes\\csvChunking\\Chatbot_for_Biosensor\\emgAnalysisSteps.txt')

    # Check for the correct timestamp column
    if 'Timestamp' not in emg_data.columns:
        st.error("No 'Timestamp' column found in the data.")
        return

    # Generate time axis based on sampling rate and number of samples
    num_samples = len(emg_data)
    times = np.arange(num_samples) / emg_sampling_rate  # in seconds

    # Initialize a dictionary to store results
    results = {}

    # Process each EMG channel
    emg_channels = ['EMG_CH1', 'EMG_CH2']
    for ch in emg_channels:
        st.subheader(f'Processing {ch}')

        # Combine the two measurements for the channel (assuming averaging is appropriate)
        ch_cols = [col for col in emg_data.columns if ch in col]
        if len(ch_cols) == 2:
            emg_signal = emg_data[ch_cols].mean(axis=1).values
        elif len(ch_cols) == 1:
            emg_signal = emg_data[ch_cols[0]].values
        else:
            st.warning(f"No data found for {ch}")
            continue

        # Standardize the EMG signal
        emg_signal = (emg_signal - np.mean(emg_signal)) / np.std(emg_signal)

        # Clean the EMG signal
        emg_cleaned = nk.emg_clean(emg_signal, sampling_rate=emg_sampling_rate)

        # Compute the envelope of the EMG signal
        emg_amplitude = nk.emg_amplitude(emg_cleaned)

        # Plot the EMG signal and envelope
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(times, emg_cleaned, label='EMG Signal', color='blue', alpha=0.5)
        ax.plot(times, emg_amplitude, label='EMG Envelope', color='red')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{ch} Signal and Envelope')
        ax.legend()
        st.pyplot(fig)

        # Estimate Muscle Activation Level
        activation_threshold = np.mean(emg_amplitude) + np.std(emg_amplitude)
        high_activation = emg_amplitude > activation_threshold

        # Visualize muscle activation over time
        fig2, ax2 = plt.subplots(figsize=(12, 2))
        ax2.plot(times, high_activation.astype(int), label='High Activation', color='green')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Activation')
        ax2.set_title(f'{ch} Muscle Activation Over Time')
        ax2.legend()
        st.pyplot(fig2)

        # Calculate percentage of time under high muscle activation
        high_activation_percentage = np.mean(high_activation) * 100
        results[ch] = high_activation_percentage

        # Display percentage of time under high muscle activation
        st.write(f"**{ch}** - Percentage of time under high muscle activation: {high_activation_percentage:.2f}%")

    # Explanation of Muscle Activation Levels
    explanation = """
    - The **EMG Envelope** represents the overall muscle activity level by taking the absolute value or using the Hilbert transform of the EMG signal.
    - The **activation threshold** is calculated as the mean plus one standard deviation of the envelope values. This threshold helps identify periods of significant muscle activity.
    - **High Activation** indicates that the muscle is exerting force above the normal resting level, suggesting periods of physical effort or tension.
    """

    st.subheader('Explanation of Muscle Activation Levels')
    st.write(explanation)

    # Save results to file in the specified format
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"EMG Analysis Results\n")
        log_file.write(f"--------------------\n")
        for ch, percentage in results.items():
            log_file.write(f"{ch} - Percentage of time under high muscle activation: {percentage:.2f}%\n")
        log_file.write(f"\nExplanation of Muscle Activation Levels:\n{explanation}\n")

    st.success(f"Analysis results have been saved to `{log_file_path}`.")


def save_emg_data(emg_data, username):
    """
    Save the EMG data to a CSV file with the username and timestamp.

    Args:
        emg_data (pd.DataFrame): DataFrame containing EMG data.
        username (str): Username for labeling the data file.

    Returns:
        str: Path to the saved CSV file.
    """
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"emg_data_{username}_{timestamp_str}.csv"
    file_path = os.path.join(DATA_STORAGE_DIR, filename)
    emg_data.to_csv(file_path, index=False)
    return file_path


def emg_app():
    """
    Streamlit app for uploading, collecting, and analyzing EMG data.
    """
    st.title("EMG Data Acquisition and Analysis")

    # ------------------------------------------
    # Data Acquisition from Shimmer Device
    # ------------------------------------------
    st.header("Data Acquisition from Shimmer Device")
    user_name = st.text_input("Enter your name for data labeling:", "")
    available_ports = list_available_ports()
    port_name = st.selectbox("Select the COM port for the Shimmer device:", available_ports)
    stream_duration = st.number_input("Enter the streaming duration in seconds:", min_value=1, value=60)

    if st.button("Start Streaming"):
        if user_name.strip() == "":
            st.warning("Please enter a valid name before starting.")
        elif not port_name:
            st.warning("No COM ports available or selected.")
        else:
            with st.spinner("Collecting data from Shimmer device..."):
                csv_file_path = run_streaming(user_name.strip(), port_name, stream_duration)
            if csv_file_path:
                st.success("Data collection completed!")
                st.write(f"Data saved as `{os.path.basename(csv_file_path)}` in `{DATA_STORAGE_DIR}`.")
                # Optionally, display some sample data
                try:
                    emg_data = pd.read_csv(csv_file_path)
                    st.subheader("Sample Data")
                    st.write(emg_data.head())
                    analyze_emg_data(emg_data)
                except Exception as e:
                    st.error(f"Error reading the saved data: {e}")

    st.markdown("---")

    # ------------------------------------------
    # Upload or Use Existing Data
    # ------------------------------------------
    st.header("Upload or Use Existing Data for Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file containing your EMG data.", type="csv")

    def process_and_analyze_uploaded_data(dataframe):
        """
        Process and analyze uploaded EMG data.

        Args:
            dataframe (pd.DataFrame): Uploaded EMG data.
        """
        # Data preprocessing steps
        dataframe.dropna(how='all', inplace=True)
        # Convert columns to numeric, coercing errors to NaN
        for col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
        dataframe.dropna(inplace=True)
        dataframe.reset_index(drop=True, inplace=True)

        # Save the processed data
        username = user_name if user_name.strip() != "" else "uploaded"
        saved_file_path = save_emg_data(dataframe, username)
        st.write(f"Data has been saved as `{os.path.basename(saved_file_path)}` in the `{DATA_STORAGE_DIR}` directory.")
        emg_logger.info(f"Uploaded data saved as {os.path.basename(saved_file_path)}.")

        # Analyze the data
        analyze_emg_data(dataframe)

    # If a file is uploaded, process it
    if uploaded_file:
        try:
            emg_data = pd.read_csv(uploaded_file)
            process_and_analyze_uploaded_data(emg_data)
        except Exception as e:
            st.error(f"Error processing file: {e}")
            emg_logger.error(f"Error processing uploaded file: {e}")
    else:
        st.info("Upload a CSV file to start analysis or select from previously saved data below.")

    st.markdown("---")

    # ------------------------------------------
    # Re-analyze Stored Data
    # ------------------------------------------
    st.header("Re-analyze Stored EMG Data")

    # List all CSV files in the data storage directory
    stored_files = [f for f in os.listdir(DATA_STORAGE_DIR) if f.endswith('.csv')]

    if len(stored_files) == 0:
        st.write('No stored EMG data files found.')
    else:
        selected_file = st.selectbox('Select an EMG data file to re-analyze:', stored_files)
        if st.button('Load and Analyze Selected File'):
            file_path = os.path.join(DATA_STORAGE_DIR, selected_file)
            try:
                emg_data = pd.read_csv(file_path)
                emg_logger.info(f"Re-analyzing stored data from {selected_file}.")
                analyze_emg_data(emg_data)
            except Exception as e:
                st.error(f"An error occurred while loading and analyzing the file: {e}")
                emg_logger.error(f"Error re-analyzing file {selected_file}: {e}")


if __name__ == "__main__":
    emg_app()
