# Fix SQLite version issue - MUST BE AT VERY TOP
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules['pysqlite3']  # Use get instead of pop
except ImportError:
    pass  # Fall back to built-in sqlite3 if pysqlite3 isn't available

import streamlit as st
import os
import time

st.set_page_config(
    page_title="Biosignal Data Analysis App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Diagnostic output
st.sidebar.write(f"Current directory: {os.getcwd()}")
st.sidebar.write(f"Files in directory: {', '.join(os.listdir())}")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
main_selection = st.sidebar.radio("Go to", ["Home", "BioSignal Analysis", "Offline Assistant"])

# --- Navigation logic ---
if main_selection == "Home":
    st.title("Welcome to the Cognitive Load & Biosignal App")
    st.write("""
    This app allows you to analyze GSR and PPG biosignal data for stress and physiological trends.
    You can also use the multimodal AI assistant that combines image and emotional data to offer real-time suggestions.
    Use the menu on the left to get started.
    """)

elif main_selection == "BioSignal Analysis":
    try:
        from dashboard import gsr_ppg_app
    except ImportError as e:
        st.error(f"Failed to import dashboard module: {e}")
        st.stop()
    
    sub_selection = st.sidebar.selectbox("Select Analysis Type", ["GSR/PPG Analysis"])
    if sub_selection == "GSR/PPG Analysis":
        gsr_ppg_app()

elif main_selection == "Offline Assistant":
    # Initialize chatbot in session state
    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False
    
    try:
        from chatBot import run_chatbot
        
        if not st.session_state.chatbot_initialized:
            with st.spinner("Initializing chatbot components..."):
                # Initialize chatbot
                run_chatbot()
                st.session_state.chatbot_initialized = True
                time.sleep(1)  # Allow UI to update
                
        if st.session_state.chatbot_initialized:
            run_chatbot()
            
    except ImportError as e:
        st.error(f"Failed to import chatbot module: {e}")
        st.error("Please ensure chatBot.py exists and has the correct dependencies")
        st.code("pip install gradio_client httpx")
        
    except Exception as e:
        st.error(f"Error in chatbot: {e}")
        if "gradio_client" in str(e):
            st.error("Please ensure gradio_client is installed in requirements.txt")
