import streamlit as st
import os
import time
import threading
import queue
import json
from datetime import datetime, timedelta
from PIL import Image
import chromadb
import pandas as pd
import logging
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import schedule

# Configure logging
logging.basicConfig(filename='app_debug.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StreamlitApp")

# === Constants and file paths ===
DATA_STORAGE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(DATA_STORAGE_DIR, "chroma_data")
DASHBOARD_RESULTS_PATH = r"C:\Users\dtfygu876\prompt_codes\csvChunking\Chatbot_for_Biosensor"
EMOTIONAL_STATE_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "emotional_state.txt")
IMAGE_ANALYSIS_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "imageAnalysis.txt")
CAPTURED_IMAGES_DIR = os.path.join(DASHBOARD_RESULTS_PATH, "captured_images")
FEEDBACK_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "teacher_feedback.csv")
CHART_DATA_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "chart_data.json")
ADVISOR_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "advisor_recommendations.json")

os.makedirs(DASHBOARD_RESULTS_PATH, exist_ok=True)
os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)

# === Custom CSS for Enhanced UI ===
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #f0f2f6;
        padding-top: 1rem;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    
    /* Section headers */
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
        margin: 15px 0 20px 0;
        font-size: 1.4rem;
    }
    
    /* Metric cards */
    .metric-card {
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e7f4 100%);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    
    /* Status indicators */
    .status-high { color: #27ae60; font-weight: bold; }
    .status-medium { color: #f39c12; font-weight: bold; }
    .status-low { color: #e74c3c; font-weight: bold; }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Primary button */
    .primary-button>button {
        background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
        color: white;
        border: none;
    }
    
    /* Secondary button */
    .secondary-button>button {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        border: none;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        margin-bottom: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 20px;
        border-radius: 8px 8px 0 0 !important;
        background-color: #e0e0e0;
        margin-right: 5px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #3498db;
        border-radius: 4px;
    }
    
    /* Camera container */
    .camera-container {
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
    }
    
    /* Custom metric styling */
    .custom-metric {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Status badges */
    .status-badge {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# === Clients ===
@st.cache_resource(show_spinner=False)
def get_chroma_client():
    logger.info("Creating ChromaDB client")
    return chromadb.PersistentClient(path=CHROMA_PATH)

@st.cache_resource(show_spinner=False)
def get_deepmind_agent():
    logger.info("Creating DeepMind agent")
    return DeepMindAgent()  # Assume this class exists

if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = get_chroma_client()

collection = st.session_state.chroma_client.get_or_create_collection(name="chat_history")
deep_agent = get_deepmind_agent()

# === Utility Functions ===
def save_chat_to_chroma(user_message: str, bot_response: str):
    base_id = f"chat_{datetime.utcnow().isoformat()}"
    metadata = {"timestamp": datetime.utcnow().isoformat(), "source": "avicenna-chatbot"}
    collection.add(
        documents=[user_message, bot_response],
        metadatas=[metadata, metadata],
        ids=[base_id + "_user", base_id + "_bot"],
    )
    logger.info(f"Saved chat to Chroma: {user_message[:50]}...")

def get_emotional_state():
    try:
        if os.path.exists(EMOTIONAL_STATE_FILE):
            with open(EMOTIONAL_STATE_FILE, "r") as f:
                lines = f.readlines()
                if not lines:
                    return "No data"
                latest_block = []
                for line in reversed(lines):
                    if line.strip() == "=== Emotional Analysis Entry ===":
                        break
                    latest_block.insert(0, line.strip())
                return "<br>".join(latest_block)
        return "Unknown"
    except Exception as e:
        logger.error(f"Emotional state error: {str(e)}")
        return f"Error reading emotional state: {e}"

def analyze_and_save_image(image_bytes: bytes):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_filename = f"captured_{timestamp}.jpg"
    img_path = os.path.join(CAPTURED_IMAGES_DIR, img_filename)

    try:
        with open(img_path, "wb") as f:
            f.write(image_bytes)

        with st.spinner("🧠 Analyzing image with VLM..."):
            # Simulate analysis
            analysis_result = {
                "behavior": "Student is focused and engaged, maintaining good posture",
                "environment": "Well-lit classroom with minimal distractions",
                "distractions": ["Open book on side desk"],
                "emotional_cues": "Positive facial expression, attentive gaze",
                "engagement_score": 82
            }
            
            # Save structured results
            log_entry = f"{timestamp}|{img_path}|{json.dumps(analysis_result)}\n"
            with open(IMAGE_ANALYSIS_FILE, "a", encoding="utf-8") as f:
                f.write(log_entry)

        return analysis_result, img_path, None
    except Exception as e:
        error_msg = f"Image processing error: {str(e)}"
        logger.error(error_msg)
        with open(IMAGE_ANALYSIS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{timestamp}|{img_path}|{error_msg}\n")
        return None, None, error_msg

def display_vlm_analysis(analysis_result):
    """Display VLM analysis in a structured format"""
    if not analysis_result:
        st.warning("No analysis result available")
        return
    
    with st.expander("🔍 VISION ANALYSIS DETAILS", expanded=True):
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        # Student Behavior Card
        with col1:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("🧍 STUDENT BEHAVIOR")
                behavior = analysis_result.get("behavior", "No behavior analysis")
                
                # Behavior status with icon
                if "leaning back" in behavior.lower():
                    st.markdown('<p class="status-low">⚠️ Disengaged Posture Detected</p>', unsafe_allow_html=True)
                elif "looking away" in behavior.lower():
                    st.markdown('<p class="status-medium">⚠️ Attention Wandering</p>', unsafe_allow_html=True)
                elif "fidgeting" in behavior.lower():
                    st.markdown('<p class="status-medium">⚠️ Restlessness Detected</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="status-high">✅ Good Engagement</p>', unsafe_allow_html=True)
                
                st.info(behavior)
                st.markdown('</div>', unsafe_allow_html=True)

        # Environment Card
        with col1:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("💡 ENVIRONMENT")
                environment = analysis_result.get("environment", "No environment analysis")
                
                # Environment status
                if "dark" in environment.lower():
                    st.markdown('<p class="status-medium">⚠️ Low Lighting</p>', unsafe_allow_html=True)
                elif "clutter" in environment.lower():
                    st.markdown('<p class="status-medium">⚠️ Cluttered Space</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="status-high">✅ Optimal Conditions</p>', unsafe_allow_html=True)
                
                st.info(environment)
                st.markdown('</div>', unsafe_allow_html=True)

        # Distractions Card
        with col2:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("🚫 DISTRACTIONS")
                distractions = analysis_result.get("distractions", [])
                
                if distractions:
                    st.markdown(f'<p class="status-low">⚠️ {len(distractions)} Distractions Detected</p>', unsafe_allow_html=True)
                    for i, item in enumerate(distractions, 1):
                        st.markdown(f"{i}. {item}")
                else:
                    st.markdown('<p class="status-high">✅ No Distractions</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Engagement Card
        with col2:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📊 ENGAGEMENT METRICS")
                emotional_cues = analysis_result.get("emotional_cues", "No emotional cues detected")
                
                if "engagement_score" in analysis_result:
                    engagement = analysis_result["engagement_score"]
                    st.metric("Engagement Score", f"{engagement}/100")
                    
                    # Color-coded progress bar
                    if engagement < 40:
                        progress_color = "#e74c3c"
                    elif engagement < 70:
                        progress_color = "#f39c12"
                    else:
                        progress_color = "#27ae60"
                    
                    st.markdown(
                        f'<div style="background: #eee; border-radius: 5px; margin: 10px 0;">'
                        f'<div style="background: {progress_color}; width: {engagement}%; '
                        f'height: 20px; border-radius: 5px;"></div></div>',
                        unsafe_allow_html=True
                    )
                    
                    if engagement < 40:
                        st.error("🔴 Low engagement - intervention needed")
                    elif engagement < 70:
                        st.warning("🟡 Medium engagement - monitor closely")
                    else:
                        st.success("🟢 High engagement - good focus")
                st.markdown('</div>', unsafe_allow_html=True)

# === Dashboard Functions ===
def show_cognitive_dashboard():
    # Simulate cognitive load data
    cognitive_data = [
        {"timestamp": "2023-10-01 09:00", "load_level": "Medium", "value": 55},
        {"timestamp": "2023-10-01 09:15", "load_level": "High", "value": 75},
        {"timestamp": "2023-10-01 09:30", "load_level": "Medium", "value": 60},
        {"timestamp": "2023-10-01 09:45", "load_level": "Low", "value": 35},
        {"timestamp": "2023-10-01 10:00", "load_level": "Medium", "value": 65},
        {"timestamp": "2023-10-01 10:15", "load_level": "High", "value": 80},
        {"timestamp": "2023-10-01 10:30", "load_level": "Medium", "value": 55},
    ]
    
    # Create DataFrame
    df = pd.DataFrame(cognitive_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Map load levels to numeric values
    level_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df['load_value'] = df['load_level'].map(level_map).fillna(1.5)

    # Create figure
    fig = px.line(
        df,
        x='timestamp',
        y='load_value',
        markers=True,
        labels={'load_value': 'Cognitive Load Level', 'timestamp': 'Time'},
        title='Cognitive Load Over Time'
    )

    # Customize y-axis
    fig.update_yaxes(
        tickvals=list(level_map.values()),
        ticktext=list(level_map.keys()),
        range=[0.5, 3.5]
    )

    # Add custom hover template
    fig.update_traces(
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M:%S}</b><br>Level: %{customdata[0]}"
    )
    
    # Add load value to custom data
    fig.data[0].customdata = df[['value']].values

    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation and recommendations
    with st.expander("📝 EXPLANATION & RECOMMENDATIONS", expanded=True):
        cols = st.columns([1, 1])
        with cols[0]:
            st.subheader("Analysis")
            st.info("Cognitive load peaked during complex problem-solving activities around 10:15 AM. Students showed signs of overload during advanced algebra concepts.")
        
        with cols[1]:
            st.subheader("Recommendations")
            st.info("Break complex problems into smaller steps. Provide visual aids for abstract concepts. Schedule short breaks between challenging activities.")

def show_engagement_trends():
    # Simulate engagement data
    engagement_data = [
        {"timestamp": "2023-10-01 09:00", "engagement": 65},
        {"timestamp": "2023-10-01 09:15", "engagement": 72},
        {"timestamp": "2023-10-01 09:30", "engagement": 68},
        {"timestamp": "2023-10-01 09:45", "engagement": 82},
        {"timestamp": "2023-10-01 10:00", "engagement": 75},
        {"timestamp": "2023-10-01 10:15", "engagement": 58},
        {"timestamp": "2023-10-01 10:30", "engagement": 70},
    ]
    
    df = pd.DataFrame(engagement_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Create rolling average
    df['rolling_avg'] = df['engagement'].rolling(window=3, min_periods=1).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['engagement'], 
        mode='markers+lines',
        name='Raw Score',
        marker=dict(color='#3498db')
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['rolling_avg'], 
        mode='lines',
        name='3-Point Avg',
        line=dict(color='#e74c3c', width=3)
    ))
    
    fig.update_layout(
        title="Engagement Score Over Time",
        xaxis_title="Time",
        yaxis_title="Engagement Score",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Engagement statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Engagement", "72/100", "3% ↑")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Session High", "82/100", None)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Session Low", "58/100", None)
        st.markdown('</div>', unsafe_allow_html=True)

# === Periodic Image Capture ===
def periodic_capture_ui():
    """UI for periodic capture with unique key"""
    if 'capture_running' not in st.session_state:
        st.session_state.capture_running = False
        st.session_state.last_capture_time = None
        
    col1, col2 = st.columns([1, 2])
    with col1:
        # Use unique key here
        if st.button("▶️ Start Auto Capture" if not st.session_state.capture_running else "⏹️ Stop Auto Capture", 
                     key="periodic_capture_toggle_btn",
                     use_container_width=True):
            st.session_state.capture_running = not st.session_state.capture_running
            
    if st.session_state.capture_running:
        current_time = datetime.now()
        if (st.session_state.last_capture_time is None or 
                (current_time - st.session_state.last_capture_time).seconds >= 10):
            
            # Simulate camera capture
            if 'camera_image' in st.session_state and st.session_state.camera_image is not None:
                image_bytes = st.session_state.camera_image.getvalue()
                analysis_result, img_path, error = analyze_and_save_image(image_bytes)
                if error:
                    st.error(f"Auto capture failed: {error}")
                else:
                    st.session_state.last_analysis = analysis_result
                    st.session_state.last_image_path = img_path
                    st.session_state.last_analysis_time = datetime.now()
                    st.session_state.last_capture_time = current_time
                    st.toast("🔄 Auto-capture completed!", icon="📸")
            
        with col2:
            if st.session_state.last_capture_time:
                elapsed = (datetime.now() - st.session_state.last_capture_time).seconds
                seconds_left = max(0, 10 - elapsed)
                st.info(f"⏱️ Next capture in: {seconds_left} seconds")
            else:
                st.info("⏱️ Preparing first capture...")

# === Teacher Tools ===
def teacher_tools():
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🧑‍🏫 TEACHER TOOLS</div>', unsafe_allow_html=True)
        
        # Cognitive Load Assessment
        st.subheader("📝 Cognitive Load Assessment")
        col1, col2 = st.columns(2)
        with col1:
            state = st.radio("Cognitive Load State",
                             ["Overloaded", "Medium", "Focus"],
                             index=1,
                             key="cognitive_state_radio")
        with col2:
            level = st.slider("Cognitive Load Level", 0, 100, 50,
                              key="cognitive_level_slider")

        if st.button("💾 Save Assessment", key="save_assessment_btn", use_container_width=True):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"""
=== Emotional Analysis Entry ===
Timestamp: {timestamp}
State Inference: {state} - Level {level}
"""
            with open(EMOTIONAL_STATE_FILE, "a", encoding="utf-8") as f:
                f.write(entry)
            st.success("Assessment saved successfully!")
            st.toast("Assessment saved!", icon="✅")

        # Teaching Advisor
        st.subheader("💡 Teaching Advisor")
        task_options = ["Lecture", "Quiz", "Group Discussion", "Self-study", "Practical Work"]
        current_task = st.selectbox("Select current task type:", task_options, key="teacher_task_select")
        st.session_state.current_task = current_task

        if st.button("🛠️ Get Teaching Strategies", key="teacher_advice_btn", use_container_width=True):
            if 'current_task' in st.session_state:
                with st.spinner("Generating teaching strategies..."):
                    # Simulated advice
                    advice = """
                    ### Teaching Strategies for Group Discussion:
                    
                    1. **Think-Pair-Share**: Have students think individually, discuss in pairs, then share with the group
                    2. **Jigsaw Technique**: Assign each group a specific topic, then regroup to share knowledge
                    3. **Discussion Roles**: Assign roles like facilitator, note-taker, and timekeeper
                    4. **Prompt Cards**: Provide discussion prompts to guide conversation
                    5. **Timed Rounds**: Use short timed discussion rounds to maintain focus
                    """
                    st.session_state.teacher_advice = advice
                    st.toast("Strategies generated!", icon="💡")
            else:
                st.warning("Please select a task type first")

        if st.session_state.get("teacher_advice"):
            st.info(st.session_state.teacher_advice)
            
        st.markdown('</div>', unsafe_allow_html=True)

# === Teacher Feedback ===
def teacher_feedback_ui():
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📄 TEACHER FEEDBACK</div>', unsafe_allow_html=True)
        
        st.markdown("**Was this app useful to you?**")
        col1, col2 = st.columns(2)
        with col1:
            useful = st.button("👍 Yes", key="feedback_yes_btn", use_container_width=True)
        with col2:
            not_useful = st.button("👎 No", key="feedback_no_btn", use_container_width=True)

        feedback_flag = useful or not_useful
        feedback_value = useful and not not_useful
        suggestion = st.text_area("Any suggestion or comment?", key="suggestion_input")

        if feedback_flag:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            feedback_data = pd.DataFrame([{
                "timestamp": timestamp,
                "useful": "Yes" if feedback_value else "No",
                "suggestion": suggestion.strip() or "None"
            }])

            if os.path.exists(FEEDBACK_FILE):
                df_existing = pd.read_csv(FEEDBACK_FILE)
                df_combined = pd.concat([df_existing, feedback_data], ignore_index=True)
            else:
                df_combined = feedback_data

            df_combined.to_csv(FEEDBACK_FILE, index=False)
            st.success("✅ Thank you for your feedback!")
            logger.info(f"Feedback received: useful={feedback_value}")
            
        st.markdown('</div>', unsafe_allow_html=True)

# === Session Report ===
def generate_session_report():
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📝 SESSION SUMMARY REPORT</div>', unsafe_allow_html=True)
        
        # Simulate report data
        report = """
        ## 📊 Session Performance Summary
        **Date:** October 1, 2023  
        **Duration:** 1 hour 30 minutes  
        **Subject:** Algebra - Quadratic Equations  
        
        ### Key Metrics
        | Metric | Value | Trend |
        |--------|-------|-------|
        | Average Engagement | 72% | ↑ 3% from previous session |
        | Cognitive Load | Medium | Optimal |
        | Distractions Detected | 2 | ↓ 1 from previous session |
        | Peak Engagement | 82% | During problem-solving activities |
        
        ### 📈 Performance Analysis
        Students showed strong engagement during collaborative problem-solving activities but experienced cognitive overload during complex equation solving around 10:15 AM. The visual learning aids helped reduce cognitive load for most students.
        
        ### 💡 Recommendations
        1. Break complex problems into smaller, manageable steps
        2. Incorporate more visual representations of equations
        3. Schedule short movement breaks between challenging sections
        4. Provide differentiated problems for varying skill levels
        
        ### 🎯 Action Plan for Next Session
        - Prepare visual aids for factorization methods
        - Create tiered problem sets (basic, intermediate, advanced)
        - Schedule 5-minute movement break at 10:00 AM
        """
        
        st.markdown(report)
        
        # Download button
        st.download_button(
            label="📥 Download Full Report",
            data=report,
            file_name=f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            key="report_download_btn",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# === Main Chatbot UI ===
def run_chatbot():
    # Enhanced title with status indicator
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
        <h1 style="margin: 0; color: #2c3e50;">🧠 Avicenna - Cognitive Load & Classroom Insight System</h1>
        <span style="background: linear-gradient(135deg, #ff5f6d, #ffc371); 
                    color: white; padding: 5px 15px; border-radius: 20px; 
                    font-size: 16px; font-weight: bold;">
            LIVE CLASSROOM FEED
        </span>
    </div>
    <p style="color: #7f8c8d; font-size: 1.1rem; margin-bottom: 30px;">
        Real-time cognitive load monitoring and teaching optimization for modern classrooms
    </p>
    """, unsafe_allow_html=True)
    
    # Initialize session state keys
    init_keys = [
        "last_analysis", "last_image_path", "last_analysis_time",
        "deepmind_response", "deepmind_error", "last_captured_image",
        "chat_history", "deepmind_processing", "processing_thread",
        "result_queue", "current_user_input", "processing_start_time",
        "current_task", "teacher_advice", "advisor_response",
        "camera_image", "capture_running", "last_capture_time"
    ]

    for key in init_keys:
        if key not in st.session_state:
            if key == "chat_history":
                st.session_state[key] = []
            elif key == "capture_running":
                st.session_state[key] = False
            else:
                st.session_state[key] = None

    # Main layout using tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📸 Real-Time Analysis", "📝 Reports & Tools"])
    
    with tab1:  # Dashboard Tab
        st.markdown('<div class="section-header">CLASSROOM PERFORMANCE OVERVIEW</div>', unsafe_allow_html=True)
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Current Engagement", "72%", "3% ↑")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Cognitive Load", "Medium", "Optimal")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Distractions", "2", "1 ↓")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Session Duration", "42 min", None)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Cognitive Load History Graph
        st.markdown('<div class="section-header">COGNITIVE LOAD HISTORY</div>', unsafe_allow_html=True)
        show_cognitive_dashboard()
        
        # Engagement Trends
        st.markdown('<div class="section-header">ENGAGEMENT TRENDS</div>', unsafe_allow_html=True)
        show_engagement_trends()
        
    with tab2:  # Real-Time Analysis Tab
        st.markdown('<div class="section-header">CLASSROOM MONITORING</div>', unsafe_allow_html=True)
        
        # Camera section with auto-capture
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown('<div class="section-header">CLASSROOM CAPTURE</div>', unsafe_allow_html=True)
            
            # Auto-capture controls
            periodic_capture_ui()
            
            # Camera input
            st.markdown('<div class="camera-container">', unsafe_allow_html=True)
            st.session_state.camera_image = st.camera_input(
                "Capture classroom image for VLM analysis", 
                key="classroom_camera",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analysis trigger button
            if st.button("🔍 Analyze Current Image", 
                         key="analyze_image_btn",
                         use_container_width=True,
                         disabled=st.session_state.camera_image is None):
                if st.session_state.camera_image is not None:
                    image_bytes = st.session_state.camera_image.getvalue()
                    analysis_result, img_path, error = analyze_and_save_image(image_bytes)
                    if error:
                        st.error(f"Image processing failed: {error}")
                    else:
                        st.session_state.last_analysis = analysis_result
                        st.session_state.last_image_path = img_path
                        st.session_state.last_analysis_time = datetime.now()
                        st.session_state.last_capture_time = datetime.now()
                        st.toast("Analysis completed!", icon="✅")
        
        with col2:
            st.markdown('<div class="section-header">LIVE ANALYSIS</div>', unsafe_allow_html=True)
            
            # Display current analysis
            if st.session_state.last_analysis:
                display_vlm_analysis(st.session_state.last_analysis)
            else:
                st.info("Capture and analyze an image to see results here")
                
            # Quick status indicators
            st.markdown('<div class="section-header">QUICK STATUS</div>', unsafe_allow_html=True)
            status_col1, status_col2 = st.columns(2)
            with status_col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**😊 Emotional State**")
                st.markdown('<p class="status-medium">Neutral</p>', unsafe_allow_html=True)
                st.progress(65)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with status_col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**💡 Environment**")
                st.markdown('<p class="status-high">Optimal</p>', unsafe_allow_html=True)
                st.progress(85)
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:  # Reports & Tools Tab
        st.markdown('<div class="section-header">TEACHING RESOURCES</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Teacher Tools
            teacher_tools()
            
            # Teacher Feedback
            teacher_feedback_ui()
            
        with col2:
            # Session Report
            generate_session_report()
            
            # Advisor Recommendations
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">💡 ADVISOR RECOMMENDATIONS</div>', unsafe_allow_html=True)
                
                if st.button("🧠 Generate Recommendations", 
                             key="advisor_btn",
                             use_container_width=True):
                    with st.spinner("Analyzing cognitive load..."):
                        # Simulated recommendations
                        recommendations = """
                        ### Cognitive Load Optimization Strategies:
                        
                        1. **Scaffold Complex Concepts**: Break quadratic equations into smaller steps
                        2. **Visual Aids**: Use graphing tools to show equation solutions
                        3. **Peer Teaching**: Have students explain concepts to each other
                        4. **Strategic Breaks**: Schedule 2-minute breaks every 25 minutes
                        5. **Differentiated Problems**: Provide tiered worksheets for varied skill levels
                        """
                        st.session_state.advisor_response = recommendations
                        st.toast("Recommendations ready!", icon="💡")
                
                if st.session_state.get("advisor_response"):
                    st.info(st.session_state.advisor_response)
                else:
                    st.info("Generate personalized recommendations based on current classroom data")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Student Interaction
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">💬 STUDENT SUPPORT</div>', unsafe_allow_html=True)
                
                user_input = st.text_area("Describe student situation or ask a question:", height=120, key="user_input")
                
                if st.button("🧠 Generate Cognitive Support Plan", 
                             key="deepmind_button",
                             use_container_width=True):
                    if not user_input:
                        st.warning("Please enter your message.")
                    else:
                        # Simulate response
                        st.session_state.deepmind_processing = True
                        time.sleep(2)  # Simulate processing time
                        st.session_state.deepmind_response = f"""
                        ### Cognitive Support Plan for Student:
                        
                        Based on your description:  
                        > "{user_input[:100]}..."
                        
                        **Recommended Interventions:**
                        1. **Chunk Information**: Break material into smaller segments
                        2. **Multisensory Approach**: Combine visual, auditory, and kinesthetic elements
                        3. **Scaffolded Practice**: Provide guided examples before independent work
                        4. **Self-Regulation Strategies**: Teach self-monitoring techniques
                        5. **Frequent Check-ins**: Provide immediate feedback every 10 minutes
                        
                        **Differentiation Strategy:**  
                        Create tiered assignments with varying complexity levels
                        """
                        st.session_state.deepmind_processing = False
                        st.toast("Support plan generated!", icon="✅")
                
                if st.session_state.get("deepmind_response"):
                    st.subheader("💡 Cognitive Support Plan")
                    st.markdown(st.session_state.deepmind_response)
                st.markdown('</div>', unsafe_allow_html=True)

    # Processing status
    if st.session_state.get("deepmind_processing"):
        with st.spinner("Generating cognitive support plan... Please wait."):
            time.sleep(2)

if __name__ == "__main__":
    run_chatbot()
