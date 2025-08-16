import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import google.generativeai as genai
import json
from PIL import Image

# Configure Gemini API (from Streamlit secrets)
genai.configure(api_key=st.secrets["GEMINI_API"])

# Streamlit UI settings
st.set_page_config(page_title="🔍 Vision-Based AI Agent (YOLO + Gemini)", layout="wide")
st.title("🔍 Vision-Based AI Agent (YOLO + Gemini)")

# YOLO model selection
model_choice = st.selectbox("Choose YOLO model:", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
model = YOLO(model_choice)

# File uploader
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Load image with PIL and convert to numpy
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)

        # YOLO Inference
        results = model(img_array)
        annotated_img = results[0].plot()

        # Gemini Vision API
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        response = model_gemini.generate_content(
            ["Describe this image in detail", img]
        )
        description = response.text.strip() if response and response.text else "No description generated."

        # Layout: side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("YOLO Detection")
            st.image(annotated_img, caption="YOLO Object Detection", use_container_width=True)

        with col2:
            st.subheader("✨ Gemini Description")
            st.write(description)

            # JSON download
            description_json = {
                "file_name": uploaded_file.name,
                "model": model_choice,
                "description": description
            }

            st.download_button(
                label="📥 Download Description (JSON)",
                data=json.dumps(description_json, indent=4),
                file_name="image_description.json",
                mime="application/json"
            )

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
