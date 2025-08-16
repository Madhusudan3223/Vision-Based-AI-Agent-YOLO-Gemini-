import streamlit as st
import cv2
from ultralytics import YOLO
import google.generativeai as genai
import os

# ----------------------------
# Load Gemini API Key
# ----------------------------
try:
    with open("GEMINI_API", "r") as f:
        api_key = f.read().strip()
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.error(f"⚠️ Failed to load Gemini API Key: {e}")
    model_gemini = None

# ----------------------------
# Load YOLO Model
# ----------------------------
try:
    yolo_model = YOLO("yolov8n.pt")  # using nano version for speed
except Exception as e:
    st.error(f"⚠️ Failed to load YOLO model: {e}")
    yolo_model = None

# ----------------------------
# Gemini Insights Function
# ----------------------------
def get_gemini_insights(detected_objects):
    if not model_gemini:
        return "Gemini model not available."
    try:
        prompt = f"The following objects were detected: {detected_objects}. Describe the scene in detail."
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🚀 YOLO + Gemini AI Scene Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image
    img_path = "uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(img_path, caption="Uploaded Image", use_container_width=True)

    # Run YOLO detection
    if yolo_model:
        results = yolo_model(img_path)
        detected_objects = []

        for r in results:
            for c in r.boxes.cls:
                detected_objects.append(yolo_model.names[int(c)])

        detected_objects = list(set(detected_objects))  # unique objects
        st.subheader("🟢 Objects Detected")
        st.write(detected_objects if detected_objects else "No objects found.")

        # Gemini analysis
        if detected_objects:
            st.subheader("✨ Gemini AI Scene Description")
            description = get_gemini_insights(detected_objects)
            st.write(description)
    else:
        st.error("YOLO model not loaded properly.")
