import streamlit as st
import pandas as pd
from datetime import datetime
import os

FEEDBACK_FILE = "teacher_feedback.csv"

def save_feedback(useful, suggestion):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feedback_data = {
        "timestamp": timestamp,
        "useful": "Yes" if useful else "No",
        "suggestion": suggestion.strip() if suggestion else "None"
    }

    df_new = pd.DataFrame([feedback_data])

    if os.path.exists(FEEDBACK_FILE):
        df_existing = pd.read_csv(FEEDBACK_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(FEEDBACK_FILE, index=False)

def teacher_feedback_ui():
    st.subheader("🧑‍🏫 Teacher Feedback")

    st.markdown("**Was this app useful to you?**")
    col1, col2 = st.columns([1, 1])
    with col1:
        useful = st.button("👍 Yes", key="feedback_yes")
    with col2:
        not_useful = st.button("👎 No", key="feedback_no")

    feedback_flag = useful or not_useful
    feedback_value = useful and not not_useful  # True only if 'Yes' clicked

    suggestion = st.text_area("Any suggestion or comment?", key="suggestion_input")

    if feedback_flag:
        save_feedback(useful=feedback_value, suggestion=suggestion)
        st.success("✅ Thank you for your feedback!")

    if os.path.exists(FEEDBACK_FILE):
        st.markdown("---")
        st.markdown("📋 **Previous Feedback Summary**")
        try:
            df = pd.read_csv(FEEDBACK_FILE)
            st.dataframe(df.tail(5))
        except Exception as e:
            st.error(f"Error reading feedback file: {e}")
