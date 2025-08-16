import streamlit as st
import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO
import google.generativeai as genai

# =============================
# Load Gemini API Key
# =============================
try:
    with open("GEMINI_API", "r") as f:
        api_key = f.read().strip()
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.error(f"⚠️ Failed to load Gemini API Key: {e}")
    model_gemini = None

# =============================
# Load YOLO model
# =============================
yolo_model = YOLO("yolov8n.pt")  # lightweight YOLOv8 model

# =============================
# Streamlit App
# =============================
st.title("🚀 YOLO + Gemini AI Scene Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Run YOLO detection
    results = yolo_model(img)
    detected_objects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = yolo_model.names[cls_id]
            detected_objects.append(label)

            # Draw bounding boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show YOLO annotated image
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="YOLO Detection", use_column_width=True)

    # =============================
    # Gemini Insights (Text Only)
    # =============================
    if model_gemini:
        detected_text = ", ".join(detected_objects) if detected_objects else "nothing detected"
        prompt = f"""
        I used YOLO to detect these objects: {detected_text}.
        Based on this, please describe the scene in detail and what might be happening.
        """

        try:
            response = model_gemini.generate_content(prompt)
            st.subheader("🧠 Gemini AI Scene Analysis")
            st.write(response.text)
        except Exception as e:
            st.error(f"⚠️ Gemini API call failed: {e}")
