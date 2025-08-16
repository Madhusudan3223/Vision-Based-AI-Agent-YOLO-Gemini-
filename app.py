import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import google.generativeai as genai

# -------------------------------
# Load Gemini API Key from Streamlit secrets
# -------------------------------
try:
    api_key = st.secrets["GEMINI_API"]
    genai.configure(api_key=api_key)
except Exception as e:
    st.error("⚠️ Failed to load Gemini API Key. Please check your Streamlit secrets.")
    st.stop()

# -------------------------------
# App UI
# -------------------------------
st.title("🚀 YOLO + Gemini AI Scene Analyzer")
st.write("Upload an image, detect objects with YOLO, and get insights from Gemini AI.")

# Upload image
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        img_path = tmp.name

    # Display uploaded image
    st.subheader("📸 Uploaded Image")
    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)

    # -------------------------------
    # Run YOLO object detection
    # -------------------------------
    st.subheader("🔍 YOLO Detection Results")
    model = YOLO("yolov8n.pt")  # small model for speed
    results = model(img_path)

    detected_objects = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detected_objects.append(model.names[cls_id])

    st.write("✅ Detected Objects")
    st.write(", ".join(detected_objects) if detected_objects else "No objects detected")

    # -------------------------------
    # Ask Gemini AI for insights
    # -------------------------------
    st.subheader("🤖 Gemini AI Insights")

    prompt = f"""
    I detected these objects in the image: {', '.join(detected_objects)}.
    Please describe what this scene might represent in a natural way.
    """

    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        response = model_gemini.generate_content(prompt)
        st.write(response.text)
    except Exception as e:
        st.error(f"Gemini API Error: {str(e)}")
