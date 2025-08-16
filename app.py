import streamlit as st
import os
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai

# 🔑 Load Gemini API key (from Streamlit secret)
api_key = st.secrets["GEMINI_API"]
genai.configure(api_key=api_key)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("🚀 YOLO + Gemini AI App")
st.write("Upload an image, detect objects with YOLO, and get insights from Gemini AI.")

# Upload image
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Run YOLO
    results = model(image)

    # Get annotated image
    annotated_img = results[0].plot()  # returns a numpy array
    st.image(annotated_img, caption="🔍 YOLO Detection Results", use_column_width=True)

    # Extract detected objects
    names = results[0].names
    detected_classes = results[0].boxes.cls.tolist()
    detected_objects = [names[int(cls)] for cls in detected_classes]

    st.write("✅ Detected Objects")
    st.write(f"Objects Detected: {', '.join(detected_objects)}")

    # Use Gemini for insights
    if detected_objects:
        prompt = f"I detected these objects in the image: {', '.join(detected_objects)}. Can you describe the scene?"
        try:
            model_gemini = genai.GenerativeModel("gemini-1.5-flash")
            response = model_gemini.generate_content(prompt)
            st.subheader("🤖 Gemini AI Insights")
            st.write(response.text)
        except Exception as e:
            st.error(f"Gemini API Error: {e}")
