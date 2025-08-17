# Set page config must be the first Streamlit command
import streamlit as st

st.set_page_config(
    page_title="Biosignal Data Analysis App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",  # Collapse sidebar since we're using top nav
)

# Fix SQLite version issue - must come right after set_page_config
try:
    __import__('pysqlite3')
    import sys

    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass  # Fall back to built-in sqlite3 if pysqlite3 isn't available

import streamlit as st
import os
import time

# Custom CSS for enhanced UI with top navigation
st.markdown("""
<style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }

    /* Top Navigation Bar */
    .top-nav {
        display: flex;
        justify-content: center;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        padding: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        position: sticky;
        top: 0;
        z-index: 100;
        border-radius: 0 0 15px 15px;
        margin-bottom: 2rem;
    }

    .nav-item {
        padding: 0.5rem 1.5rem;
        margin: 0 0.5rem;
        border-radius: 30px;
        color: white !important;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        position: relative;
        overflow: hidden;
        display: inline-block;
        text-align: center;
        border: none;
        background: transparent;
    }

    .nav-item:hover {
        background: rgba(255,255,255,0.15);
        transform: translateY(-2px);
    }

    .nav-item.active {
        background: rgba(255,255,255,0.25);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        font-weight: 600;
    }

    .nav-item.active:after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 25%;
        width: 50%;
        height: 3px;
        background: #ffd166;
        border-radius: 3px;
    }

    /* Main content styling */
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 12px 24px rgba(0,0,0,0.05);
        margin: 0 auto;
        max-width: 1400px;
    }

    /* Card styling */
    .card {
        background: white;
        border-radius: 18px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: none;
        position: relative;
        overflow: hidden;
    }

    .card:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(to bottom, #6a11cb, #2575fc);
    }

    .card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
    }

    .card-title {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 15px;
        font-size: 1.4rem;
    }

    .card-title span {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
    }

    /* Custom header */
    .header {
        display: flex;
        align-items: center;
        gap: 20px;
        padding-bottom: 20px;
        margin-bottom: 30px;
        border-bottom: 2px solid #e0e7ff;
    }

    .header-icon {
        font-size: 3rem;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Feature grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 30px;
        margin-top: 2.5rem;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        padding: 25px;
        margin-top: 50px;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid #e0e7ff;
        background: rgba(106,17,203,0.03);
        border-radius: 15px;
    }

    /* Interactive elements */
    .mode-selector {
        background: white;
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        margin-bottom: 30px;
    }

    .stRadio [role="radiogroup"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e0e7ff;
    }

    .stSelectbox select {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        border-radius: 12px !important;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.05) !important;
        border: 1px solid #e0e7ff !important;
        padding: 12px !important;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main {
            padding: 1.5rem;
        }
        .header {
            flex-direction: column;
            text-align: center;
        }
        .top-nav {
            flex-wrap: wrap;
            padding: 0.5rem;
        }
        .nav-item {
            padding: 0.5rem 1rem;
            margin: 0.3rem;
            font-size: 0.9rem;
        }
    }

    /* Glowing animation for active elements */
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(106,17,203,0.5); }
        50% { box-shadow: 0 0 20px rgba(106,17,203,0.8); }
        100% { box-shadow: 0 0 5px rgba(106,17,203,0.5); }
    }

    .glowing {
        animation: glow 2s infinite;
    }

    /* Hide Streamlit footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'current_task' not in st.session_state:
    st.session_state.current_task = "Lecture"
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = "Live"  # Default to Live mode

# Navigation page names and icons
nav_items = {
    "Home": "🏠",
    "BioSignal Analysis": "📊",
    "Offline Assistant": "🤖"
}


# --- Top Navigation Bar ---
def create_nav_bar():
    """Create the top navigation bar using Streamlit buttons"""
    with st.container():
        st.markdown('<div class="top-nav">', unsafe_allow_html=True)

        # Create columns for navigation items
        cols = st.columns(len(nav_items))
        for i, (item, icon) in enumerate(nav_items.items()):
            with cols[i]:
                # Add custom styling to make it look like our nav item
                if st.button(
                        f"{icon} {item}",
                        key=f"nav_{item}",
                        use_container_width=True,
                ):
                    st.session_state.current_page = item

                # Add active class styling
                if st.session_state.current_page == item:
                    st.markdown(
                        f"""
                        <style>
                            div[data-testid="stButton"] > button[kind="secondary"][data-testid="baseButton-secondary"][id="nav_{item}"] {{
                                background: rgba(255,255,255,0.25) !important;
                                box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
                                font-weight: 600 !important;
                                color: white !important;
                            }}
                            div[data-testid="stButton"] > button[kind="secondary"][data-testid="baseButton-secondary"][id="nav_{item}"]:after {{
                                content: '';
                                position: absolute;
                                bottom: 0;
                                left: 25%;
                                width: 50%;
                                height: 3px;
                                background: #ffd166;
                                border-radius: 3px;
                            }}
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

        st.markdown('</div>', unsafe_allow_html=True)


# Create the navigation bar
create_nav_bar()

# --- Navigation logic ---
if st.session_state.current_page == "Home":
    # Header
    st.markdown("""
    <div class="header">
        <div class="header-icon">🧠</div>
        <div>
            <h1 style="margin:0;color:#2c3e50">Cognitive Load & Biosignal Analysis</h1>
            <p style="color:#6c757d;margin-top:5px">Real-time physiological insights for educators</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main content container
    st.markdown('<div class="main">', unsafe_allow_html=True)

    # Introduction card
    st.markdown("""
    <div class="card">
        <p style="font-size:1.1rem">This application combines biosignal analysis with AI-powered insights to help educators optimize teaching effectiveness by monitoring cognitive load in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Features header
    st.markdown('<h3 style="color:#2c3e50;border-left:5px solid #6a11cb;padding-left:15px">Key Features</h3>',
                unsafe_allow_html=True)

    # Feature grid
    st.markdown("""
    <div class="feature-grid">
        <div class="card">
            <div class="card-title"><span>📊</span> Live BioSignal Analysis</div>
            <p>Real-time visualization of GSR and PPG data with stress level indicators during teaching activities.</p>
        </div>
        <div class="card">
            <div class="card-title"><span>🤖</span> AI Teaching Assistant</div>
            <p>Multimodal AI assistant that combines image analysis and emotional data to provide real-time suggestions.</p>
        </div>
        <div class="card">
            <div class="card-title"><span>📝</span> Activity Insights</div>
            <p>Compare cognitive load across different teaching methods and student engagement levels.</p>
        </div>
        <div class="card">
            <div class="card-title"><span>📱</span> Responsive Design</div>
            <p>Fully responsive interface that works on desktops, tablets, and mobile devices.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Getting Started header
    st.markdown('<h3 style="color:#2c3e50;border-left:5px solid #2575fc;padding-left:15px">Getting Started</h3>',
                unsafe_allow_html=True)

    # Getting Started card
    st.markdown("""
    <div class="card">
        <ol style="font-size:1.05rem">
            <li>Select <strong style="color:#6a11cb">BioSignal Analysis</strong> in the top navigation to begin real-time monitoring</li>
            <li>Choose your teaching activity type (Lecture, Group Work, etc.)</li>
            <li>View physiological metrics and cognitive load estimates</li>
            <li>Use the Offline Assistant for post-session analysis and recommendations</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>Developed with ❤️ for educators | v2.1</p>
    </div>
    """, unsafe_allow_html=True)

    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "BioSignal Analysis":
    st.markdown("""
    <div class="header">
        <div class="header-icon">📊</div>
        <div>
            <h1 style="margin:0;color:#2c3e50">BioSignal Analysis Dashboard</h1>
            <p style="color:#6c757d;margin-top:5px">Real-time physiological monitoring</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main">', unsafe_allow_html=True)

    # Teaching Activity Type Selector
    st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
    st.subheader("Teaching Activity Type")
    task_type = st.selectbox(
        "Select teaching activity type:",
        ["Lecture", "Group Work", "Assessment", "Discussion", "Practical"],
        key="task_type_select"
    )
    st.session_state.current_task = task_type

    # Data Collection Mode
    st.markdown("---")
    st.subheader("Data Collection Mode")
    analysis_mode = st.radio(
        "Select data collection mode:",
        ["Live", "Demo"],
        index=0 if st.session_state.analysis_mode == "Live" else 1,
        horizontal=True,
        key="analysis_mode_selector"
    )
    st.session_state.analysis_mode = analysis_mode
    st.markdown('</div>', unsafe_allow_html=True)

    # Run the dashboard app
    try:
        from dashboard import gsr_ppg_app

        gsr_ppg_app()
    except ImportError as e:
        st.error(f"Failed to import dashboard module: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error in dashboard: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "Offline Assistant":
    # Initialize chatbot in session state
    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False

    st.markdown("""
    <div class="header">
        <div class="header-icon">🤖</div>
        <div>
            <h1 style="margin:0;color:#2c3e50">AI Teaching Assistant</h1>
            <p style="color:#6c757d;margin-top:5px">Post-session analysis and recommendations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main">', unsafe_allow_html=True)

    # Teaching Activity Type Selector
    st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
    st.subheader("Teaching Activity Type")
    task_type = st.selectbox(
        "Select teaching activity type:",
        ["Lecture", "Group Work", "Assessment", "Discussion", "Practical"],
        key="task_type_select_chat"
    )
    st.session_state.current_task = task_type
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        from chatBot import run_chatbot

        if not st.session_state.chatbot_initialized:
            with st.spinner("Initializing AI assistant components..."):
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

    st.markdown('</div>', unsafe_allow_html=True)
