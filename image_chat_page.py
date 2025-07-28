# === image_chat_page.py ===
import streamlit as st
from PIL import Image
import tempfile
import os
from mistralai.client import MistralClient
from image_processor import analyze_image_with_gradio

def run_image_chat_page():
    st.title("🖼️ Student/Worker Condition Detector")
    st.markdown(
        "Upload an image or take a photo to detect the emotional or physical condition "
        "of a student or worker, especially signs of stress or overwhelm."
    )

    uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])
    camera_image = st.camera_input("Or take a photo")
    image_to_use = camera_image if camera_image is not None else uploaded_image

    if image_to_use:
        st.image(image_to_use, use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_to_use.getbuffer())
            tmp_path = tmp.name

        if st.button("🔍 Analyze Condition"):
            with st.spinner("Analyzing image condition using LLaVA..."):
                llava_description = analyze_image_with_gradio(
                    tmp_path,
                    prompt=(
                        "Analyze the condition or emotional state of the person in this image. "
                        "Focus on visible signs of stress, overwhelm, tiredness, or burnout commonly experienced by students or office workers."
                    )
                )

            st.subheader("🧠 LLaVA Condition Analysis")

            if llava_description.lower().startswith("error"):
                st.error(llava_description)
                return
            else:
                st.success(llava_description)

            # Get Mistral API key
            MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

            if not MISTRAL_API_KEY:
                st.error("MISTRAL_API_KEY not set in environment variables.")
                return

            mistral_client = MistralClient(api_key=MISTRAL_API_KEY)

            with st.spinner("Generating concise suggestions using Mistral..."):
                try:
                    result = mistral_client.chat(
                        model="mistral-large-latest",
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    f"The following is an analysis of a person's condition: '{llava_description}'. "
                                    "Please generate 2–3 short, clear, and helpful suggestions to improve their emotional or physical state. "
                                    "Keep it simple and effective—no long paragraphs."
                                )
                            }
                        ],
                        temperature=0.5,
                        max_tokens=300,
                    )
                    suggestion = result.choices[0].message.content
                    st.subheader("💡 Supportive Suggestions")
                    st.write(suggestion)
                except Exception as e:
                    st.error(f"Error from Mistral API: {e}")
