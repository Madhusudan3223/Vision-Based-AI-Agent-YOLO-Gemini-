import os
import streamlit as st
from PIL import Image
import cv2
import torch
from ultralytics import YOLO
import google.generativeai as genai

# ----------------- Setup Gemini -----------------
genai.configure(api_key=os.getenv("GEMINI_API"))  # Make sure GEMINI_API is set in secrets

# Load Gemini model
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------- Load YOLO model -----------------
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")  # lightweight model

yolo_model = load_yolo()

# ----------------- Streamlit App -----------------
st.title("🔍 Vision-Based AI Agent (YOLO + Gemini)")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Run YOLO
    st.subheader("YOLO Detection")
    results = yolo_model(image)

    # Annotate image
    annotated_frame = results[0].plot()
    st.image(annotated_frame, caption="📦 YOLO Detection Result", use_column_width=True)

    # Extract detected objects
    detected_objects = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        detected_objects.append(yolo_model.names[cls_id])

    # Show detected classes
    st.write("Detected Objects:", detected_objects if detected_objects else "None")

    # Gemini description
    st.subheader("✨ Gemini Description")
    prompt = f"Describe this image. The objects detected are: {detected_objects}."
    response = gemini_model.generate_content([prompt, image])
    st.write(response.text)
