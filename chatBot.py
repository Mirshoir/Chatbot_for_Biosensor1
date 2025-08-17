import streamlit as st
import os
import time
import threading
import queue
import json
from datetime import datetime, timedelta
from PIL import Image
import chromadb
from image_processor import analyze_image_with_gradio
from deepMind import DeepMindAgent
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
EMOTIONAL_STATE_FILE = emotional_state.txt
IMAGE_ANALYSIS_FILE = imageAnalysis.txt
CAPTURED_IMAGES_DIR = os.path.join(DASHBOARD_RESULTS_PATH, "captured_images")
FEEDBACK_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "teacher_feedback.csv")
CHART_DATA_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "chart_data.json")
ADVISOR_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "advisor_recommendations.json")

os.makedirs(DASHBOARD_RESULTS_PATH, exist_ok=True)
os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)

# === Custom CSS for Enhanced UI ===
st.markdown("""
<style>
    /* Global Styles */
    :root {
        --primary: #4361ee;
        --secondary: #3f37c9;
        --success: #4cc9f0;
        --warning: #f72585;
        --info: #4895ef;
        --light: #f8f9fa;
        --dark: #212529;
        --bg: #f5f7ff;
        --card-bg: #ffffff;
        --border: #e0e7ff;
    }

    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    body {
        background-color: var(--bg);
        color: var(--dark);
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(67, 97, 238, 0.3);
    }

    /* Cards */
    .card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border-left: 4px solid var(--primary);
    }

    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border-left: 4px solid var(--info);
    }

    .metric-card h3 {
        margin: 5px 0;
        font-size: 1.8rem;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Tabs */
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
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(67, 97, 238, 0.2);
    }

    /* Expanders */
    .stExpander {
        background: var(--card-bg);
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid var(--border);
    }

    .stExpander .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--primary);
    }

    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }

    /* Custom headers */
    .section-header {
        padding-bottom: 10px;
        margin-bottom: 20px;
        border-bottom: 2px solid var(--border);
        color: var(--primary);
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Camera container */
    .camera-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
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
    return DeepMindAgent()


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
            analysis_result = analyze_image_with_gradio(img_path)

            if isinstance(analysis_result, dict) and "error" in analysis_result:
                raise RuntimeError(analysis_result["error"])
            elif isinstance(analysis_result, str) and "error" in analysis_result.lower():
                raise RuntimeError(analysis_result)

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
    if not analysis_result or ("error" in analysis_result if isinstance(analysis_result, dict) else False):
        if isinstance(analysis_result, dict):
            error = analysis_result.get("error", "Unknown error")
        else:
            error = str(analysis_result)
        st.error(f"Analysis error: {error}")
        return

    # Handle both structured dict and legacy string output
    if isinstance(analysis_result, str):
        # Legacy format - display as is
        st.info(analysis_result)
        return

    with st.expander("🔍 VLM Analysis Details", expanded=True):
        cols = st.columns(2)

        with cols[0]:
            st.subheader("🧍 Student Behavior")
            behavior = analysis_result.get("behavior", "No behavior analysis")
            st.markdown(f"<div class='card'>{behavior}</div>", unsafe_allow_html=True)

            # Behavior recommendations
            if "leaning back" in behavior.lower():
                st.warning("⚠️ Student posture suggests disengagement - try active learning techniques")
            if "looking away" in behavior.lower():
                st.warning("⚠️ Student attention wandering - try proximity or questioning")
            if "fidgeting" in behavior.lower():
                st.warning("⚠️ Student appears restless - consider movement break")

            st.subheader("💡 Environment")
            environment = analysis_result.get("environment", "No environment analysis")
            st.markdown(f"<div class='card'>{environment}</div>", unsafe_allow_html=True)

            # Environment recommendations
            if "dark" in environment.lower():
                st.warning("⚠️ Low lighting may cause eye strain - adjust lighting")
            if "clutter" in environment.lower():
                st.warning("⚠️ Cluttered environment may reduce focus - suggest cleanup")

        with cols[1]:
            st.subheader("🚫 Distractions")
            distractions = analysis_result.get("distractions", [])
            if distractions:
                st.warning(f"⚠️ {len(distractions)} distractions detected:")
                for i, item in enumerate(distractions, 1):
                    st.markdown(f"<div class='card'>{i}. {item}</div>", unsafe_allow_html=True)
            else:
                st.success("✅ No distractions detected")

            st.subheader("😔 Emotional Cues")
            emotional_cues = analysis_result.get("emotional_cues", "No emotional cues detected")
            st.markdown(f"<div class='card'>{emotional_cues}</div>", unsafe_allow_html=True)

            # Engagement score if available
            if "engagement_score" in analysis_result:
                engagement = analysis_result["engagement_score"]
                st.metric("Engagement Score", f"{engagement}/100")
                st.progress(engagement / 100)

                if engagement < 40:
                    st.error("🔴 Low engagement - intervention needed")
                elif engagement < 70:
                    st.warning("🟡 Medium engagement - monitor closely")
                else:
                    st.success("🟢 High engagement - good focus")


# === Dashboard Functions ===
def show_cognitive_dashboard():
    st.subheader("📊 Cognitive Load History Graph")

    try:
        if not os.path.exists(CHART_DATA_FILE):
            st.warning("No chart data available yet")
            return

        with open(CHART_DATA_FILE, "r") as f:
            chart_data = json.load(f)

        # Extract data
        load_history = chart_data.get("cognitive_load_history", [])

        if not load_history:
            st.info("Dashboard data is being collected. Check back soon.")
            return

        # Create DataFrame
        df = pd.DataFrame(load_history)
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
            hover_data=['load_level', 'explanation'],
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
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M:%S}</b><br>Level: %{customdata[0]}<br>%{customdata[1]}"
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add explanation and recommendations
        if st.session_state.get("cognitive_analysis"):
            analysis = st.session_state.cognitive_analysis
            with st.expander("📝 Explanation & Recommendations", expanded=True):
                cols = st.columns([1, 1])
                with cols[0]:
                    st.subheader("Explanation")
                    explanation = analysis.get("explanation", "No explanation available")
                    st.markdown(f"<div class='card'>{explanation}</div>", unsafe_allow_html=True)

                with cols[1]:
                    st.subheader("Recommendations")
                    if st.session_state.get("advisor_response"):
                        st.markdown(f"<div class='card'>{st.session_state.advisor_response}</div>",
                                    unsafe_allow_html=True)
                    else:
                        st.warning("Run Cognitive Load Analysis to get recommendations")

    except Exception as e:
        st.error(f"Dashboard error: {str(e)}")
        logger.error(f"Dashboard error: {str(e)}")


def show_engagement_trends():
    st.subheader("📈 Engagement Trends")

    try:
        if not os.path.exists(IMAGE_ANALYSIS_FILE):
            st.warning("No engagement data available yet")
            return

        engagement_data = []
        with open(IMAGE_ANALYSIS_FILE, "r") as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    try:
                        timestamp = datetime.strptime(parts[0], "%Y%m%d_%H%M%S")
                        analysis = json.loads(parts[2])
                        if "engagement_score" in analysis:
                            engagement_data.append({
                                "timestamp": timestamp,
                                "engagement": analysis["engagement_score"]
                            })
                    except:
                        continue

        if not engagement_data:
            st.info("No engagement scores found in analysis data")
            return

        df = pd.DataFrame(engagement_data)
        df = df.sort_values('timestamp')

        # Create rolling average
        df['rolling_avg'] = df['engagement'].rolling(window=3, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['engagement'],
            mode='markers+lines',
            name='Raw Score',
            marker=dict(color='#4361ee')
        ))
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['rolling_avg'],
            mode='lines',
            name='3-Point Avg',
            line=dict(color='#f72585', width=3)
        ))

        fig.update_layout(
            title="Engagement Score Over Time",
            xaxis_title="Time",
            yaxis_title="Engagement Score",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Engagement statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{df['engagement'].iloc[-1]}</h3>
                <p>Current Engagement</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{df['engagement'].max()}</h3>
                <p>Session High</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{df['engagement'].min()}</h3>
                <p>Session Low</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Engagement dashboard error: {str(e)}")
        logger.error(f"Engagement dashboard error: {str(e)}")


# === Periodic Image Capture ===
def periodic_capture_ui():
    """UI for periodic capture with unique key"""
    if 'capture_running' not in st.session_state:
        st.session_state.capture_running = False
        st.session_state.last_capture_time = None

    col1, col2 = st.columns([1, 3])
    with col1:
        # Use unique key here
        if st.button("▶️ Start Auto Capture" if not st.session_state.capture_running else "⏹️ Stop Auto Capture",
                     key="unique_auto_capture_btn"):  # Fixed unique key
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
                st.markdown(f"<div class='card'>⏱️ Next capture in: {seconds_left} seconds</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div class='card'>⏱️ Preparing first capture...</div>", unsafe_allow_html=True)


# === Teacher Tools ===
def teacher_tools():
    with st.expander("🧑‍🏫 Teacher Tools", expanded=True):
        # Teaching Advisor
        st.subheader("💡 Teaching Advisor")

        # Display current task from app.py sidebar
        if 'current_task' in st.session_state:
            st.markdown(f"<div class='card'>**Current Task Type:** {st.session_state.current_task}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='card'>**Current Task Type:** Not specified</div>", unsafe_allow_html=True)

        # Map task types to DeepMind agent format
        task_map = {
            "Lecture": "Lecture",
            "Group Work": "Group Discussion",
            "Assessment": "Quiz",
            "Discussion": "Group Discussion",
            "Practical": "Practical Work"
        }
        mapped_task = task_map.get(st.session_state.get("current_task", "Lecture"), "Lecture")

        if st.button("🛠️ Get Teaching Strategies", key="teacher_advice_btn", use_container_width=True):
            with st.spinner("Generating teaching strategies..."):
                advice = deep_agent.get_teacher_advice(mapped_task)
                st.session_state.teacher_advice = advice

        if st.session_state.get("teacher_advice"):
            st.markdown(f"<div class='card'>{st.session_state.teacher_advice}</div>", unsafe_allow_html=True)

        # Cognitive Load Advisor
        st.subheader("🧠 Cognitive Load Advisor")
        st.markdown("<div class='card'>Get personalized suggestions based on current cognitive load</div>",
                    unsafe_allow_html=True)
        if st.button("📊 Generate Advisor Recommendations", key="advisor_btn", use_container_width=True):
            with st.spinner("Analyzing cognitive load..."):
                # Get cognitive load analysis
                analysis = deep_agent.get_cognitive_load_analysis()

                if "error" in analysis:
                    st.error(f"Analysis failed: {analysis['error']}")
                else:
                    # Get advisor recommendations
                    advisor_response = deep_agent.get_cognitive_load_advisor(analysis)
                    st.session_state.advisor_response = advisor_response

                    # Save recommendation
                    recommendation = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "state": analysis.get("cognitive_load_level", "Unknown"),
                        "level": analysis.get("cognitive_load_value", 0),
                        "recommendation": advisor_response
                    }

                    # Load existing recommendations
                    if os.path.exists(ADVISOR_FILE):
                        with open(ADVISOR_FILE, "r") as f:
                            recommendations = json.load(f)
                    else:
                        recommendations = []

                    recommendations.append(recommendation)

                    # Save back to file
                    with open(ADVISOR_FILE, "w") as f:
                        json.dump(recommendations, f, indent=2)

        if st.session_state.get("advisor_response"):
            st.markdown(f"<div class='card'>{st.session_state.advisor_response}</div>", unsafe_allow_html=True)


# === Teacher Feedback ===
def teacher_feedback_ui():
    with st.expander("📄 Teacher Feedback", expanded=False):
        st.subheader("🧑‍🏫 Teacher Feedback")
        st.markdown("<div class='card'>**Was this app useful to you?**</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            useful = st.button("👍 Yes", key="feedback_yes", use_container_width=True)
        with col2:
            not_useful = st.button("👎 No", key="feedback_no", use_container_width=True)

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
                df_existing = pd.read_csv(FEEDBACK_FILE)  # Fixed typo here
                df_combined = pd.concat([df_existing, feedback_data], ignore_index=True)
            else:
                df_combined = feedback_data

            df_combined.to_csv(FEEDBACK_FILE, index=False)
            st.success("✅ Thank you for your feedback!")
            logger.info(f"Feedback received: useful={feedback_value}")

        if os.path.exists(FEEDBACK_FILE):
            st.markdown("---")
            st.markdown("📋 **Recent Feedback**")
            try:
                df = pd.read_csv(FEEDBACK_FILE)
                st.dataframe(df.tail(5))
            except Exception as e:
                st.error(f"Error reading feedback: {e}")


# === Session Report ===
def generate_session_report():
    st.subheader("📝 Session Summary Report")

    try:
        report_data = {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "engagement": "N/A",
            "cognitive_load": "N/A",
            "distractions": 0,
            "key_observations": []
        }

        # Get engagement data
        if os.path.exists(IMAGE_ANALYSIS_FILE):
            with open(IMAGE_ANALYSIS_FILE, "r") as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 3:
                        try:
                            analysis = json.loads(parts[2])
                            if "engagement_score" in analysis:
                                report_data["engagement"] = analysis["engagement_score"]
                            if "distractions" in analysis:
                                report_data["distractions"] = len(analysis["distractions"])
                        except:
                            continue

        # Get cognitive load data
        if st.session_state.get("cognitive_analysis"):
            analysis = st.session_state.cognitive_analysis
            report_data["cognitive_load"] = analysis.get("cognitive_load_level", "N/A")
            report_data["key_observations"].append(analysis.get("explanation", ""))

        # Generate report
        report = f"""
# Session Summary Report
**Date:** {datetime.now().strftime("%Y-%m-%d")}

## Key Metrics
- **Average Engagement:** {report_data['engagement']}/100
- **Cognitive Load Level:** {report_data['cognitive_load']}
- **Distractions Detected:** {report_data['distractions']}

## Observations
{st.session_state.get("advisor_response", "No observations recorded")}

## Recommendations
{st.session_state.get("teacher_advice", "No recommendations available")}
        """

        st.markdown(report)

        # Download button
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

    except Exception as e:
        st.error(f"Report generation error: {str(e)}")


# === Main Chatbot UI ===
def run_chatbot():
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

    # Set default task if not set from app.py
    if 'current_task' not in st.session_state or not st.session_state.current_task:
        st.session_state.current_task = "Lecture"

    # Main layout columns
    col1, col2 = st.columns([3, 2])

    with col1:
        # Driver Data Streams
        st.subheader("Driver Data Streams")
        st.markdown("<div class='card'>**Data stream: live**</div>", unsafe_allow_html=True)

        # DSR Section
        with st.expander("DSR", expanded=True):
            cols = st.columns(3)
            with cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>2 mm</h3>
                    <p>PPG</p>
                </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                emotional_state = get_emotional_state().split("<br>")[
                    0] if "<br>" in get_emotional_state() else get_emotional_state()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{emotional_state[:10]}</h3>
                    <p>Emotional State</p>
                </div>
                """, unsafe_allow_html=True)
            with cols[2]:
                if st.session_state.get("last_analysis"):
                    vlm_summary = st.session_state.last_analysis.get("behavior", "No analysis")[:10] + "..."
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{vlm_summary}</h3>
                        <p>VLM</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>No data</h3>
                        <p>VLM</p>
                    </div>
                    """, unsafe_allow_html=True)

        # Demo indicator
        st.markdown("---")
        st.markdown("---")

        # Image Capture and Analysis
        st.subheader("Cognitive Load")

        # Periodic capture UI
        periodic_capture_ui()

        # Camera input with unique key
        st.markdown("<div class='camera-container'>", unsafe_allow_html=True)
        st.session_state.camera_image = st.camera_input(
            "Capture classroom image for VLM analysis",
            key="classroom_camera"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.camera_image is not None:
            if (st.session_state.last_captured_image != st.session_state.camera_image.getvalue() or
                    st.session_state.last_image_path is None):

                st.session_state.last_captured_image = st.session_state.camera_image.getvalue()
                image_bytes = st.session_state.camera_image.getvalue()
                analysis_result, img_path, error = analyze_and_save_image(image_bytes)
                if error:
                    st.error(f"Image processing failed: {error}")
                else:
                    st.session_state.last_analysis = analysis_result
                    st.session_state.last_image_path = img_path
                    st.session_state.last_analysis_time = datetime.now()
                    st.success("Image captured and analyzed successfully!")

        # Display image and analysis
        if st.session_state.last_analysis:
            try:
                # Check if file exists before opening
                if os.path.exists(st.session_state.last_image_path):
                    img = Image.open(st.session_state.last_image_path)
                    st.image(img, caption=f"📸 {st.session_state.last_analysis_time.strftime('%Y-%m-%d %H:%M:%S')}",
                             use_container_width=True)
                else:
                    st.warning("Image file not found")
            except Exception as e:
                st.warning(f"Could not display image: {str(e)}")

            # Display VLM analysis
            display_vlm_analysis(st.session_state.last_analysis)
        else:
            st.info("No image analysis available. Capture an image to begin.")

        # Cognitive Load Dashboard
        show_cognitive_dashboard()

        # Engagement Trends Dashboard
        show_engagement_trends()

    with col2:
        # Teacher Tools Section
        teacher_tools()

        # Advisor Recommendations
        st.subheader("Recommendations")
        st.markdown("<div class='card'>**Advisor Agent**</div>", unsafe_allow_html=True)
        if st.session_state.get("advisor_response"):
            st.markdown(f"<div class='card'>{st.session_state.advisor_response}</div>", unsafe_allow_html=True)
        else:
            st.warning("Generate recommendations using Teacher Tools")

        # Teacher Feedback
        teacher_feedback_ui()

        # Reports section
        st.subheader("Reports")
        st.markdown("<div class='card'>**Session summary**</div>", unsafe_allow_html=True)
        generate_session_report()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Cognitive Load Analysis Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header with title and logo
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <h1 style="margin: 0; color: #4361ee;">Cognitive Load Analysis Dashboard</h1>
        <div style="margin-left: auto; font-size: 2rem;">🧠</div>
    </div>
    """, unsafe_allow_html=True)

    run_chatbot()
