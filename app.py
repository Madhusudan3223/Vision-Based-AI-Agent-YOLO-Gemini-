import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai
import os

# Load Gemini API key from secret
genai.configure(api_key=st.secrets["GEMINI_API"])

# Load YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8n.pt")

st.title("🖼️ AI Object Detector + Gemini Describer")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO detection
    results = model(image)
    labels = results[0].names
    detected_classes = [labels[int(cls)] for cls in results[0].boxes.cls]

    st.subheader("Detected Objects")
    st.write(detected_classes)

    # Generate description using Gemini
    if detected_classes:
        prompt = f"Describe an image containing the following objects: {', '.join(detected_classes)}"
        model_gemini = genai.GenerativeModel("gemini-pro")
        response = model_gemini.generate_content(prompt)

        st.subheader("Gemini Description")
        st.write(response.text)
