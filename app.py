import os
import torch
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API"))

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Streamlit UI
st.title("🚀 YOLO + Gemini AI App")
st.write("Upload an image, detect objects with YOLO, and get insights from Gemini AI.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file)

    # Show uploaded image
    st.image(img, caption="📸 Uploaded Image", use_container_width=True)

    # Run YOLO detection
    st.subheader("🔍 YOLO Detection Results")
    results = model(img)
    results_img = results[0].plot()  # Annotated image
    st.image(results_img, caption="✅ Detected Objects", use_container_width=True)

    # Extract detected labels
    labels = results[0].names
    detected_objects = [labels[int(cls)] for cls in results[0].boxes.cls]

    st.write("**Objects Detected:**", ", ".join(detected_objects) if detected_objects else "None")

    # Ask Gemini for insights
    if detected_objects:
        st.subheader("🤖 Gemini AI Insights")
        prompt = f"I detected these objects: {', '.join(detected_objects)}. Describe them in detail."
        try:
            response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
            st.write(response.text)
        except Exception as e:
            st.error(f"Gemini API Error: {e}")
