import os
import streamlit as st
from ultralytics import YOLO
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API")

if not GEMINI_API_KEY:
    st.error("⚠️ Gemini API key not found. Please add it in your .env file as GEMINI_API=your_key_here")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------
# Load YOLO model
# ---------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo_model()

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("🚀 YOLO + Gemini AI Scene Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run YOLO detection
    st.subheader("🔎 YOLO Detection")
    results = yolo_model.predict(image)

    if results and len(results[0].boxes) > 0:
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="Detected Objects", use_container_width=True)

        # Extract detected labels
        labels = results[0].names
        detected_objects = [labels[int(box.cls)] for box in results[0].boxes]
        st.write("✅ Objects Detected:", detected_objects)

        # ---------------------------
        # Gemini AI Scene Analysis
        # ---------------------------
        st.subheader("🧠 Gemini Scene Understanding")
        prompt = f"""
        You are an AI assistant. Analyze this image scene.
        Objects detected: {detected_objects}.
        Give a short description of the scene and possible context.
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        st.write("### 📖 Scene Description:")
        st.write(response.text if response else "No description generated.")
    else:
        st.warning("⚠️ No objects detected in the image.")
