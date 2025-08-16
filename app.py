import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try getting Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.warning("⚠️ Gemini API key not found! Please set `GEMINI_API_KEY` in .env or Streamlit secrets.")
else:
    genai.configure(api_key=api_key)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Streamlit UI
st.title("🚀 YOLO + Gemini AI Scene Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # YOLO detection
    results = model(image)
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="YOLO Detection", use_container_width=True)

    # Object list
    objects = results[0].boxes.cls.cpu().numpy()
    labels = [model.names[int(cls)] for cls in objects]
    st.write("🔍 Objects Detected:", ", ".join(labels))

    # Gemini analysis
    if api_key:
        prompt = f"Describe the scene in detail. The objects detected are: {', '.join(labels)}"
        response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
        st.subheader("🤖 Gemini AI Insights")
        st.write(response.text)
