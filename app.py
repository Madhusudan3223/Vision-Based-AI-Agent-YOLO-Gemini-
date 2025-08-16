import streamlit as st
import torch
import os
from PIL import Image
import google.generativeai as genai

# Load YOLO model (make sure you exported YOLOv8 from Colab to same folder)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # small model for speed

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API"))

st.title("🔎 Vision-Based AI Agent")
st.write("Upload an image → Detect objects with YOLO → Generate description with Gemini")

# Upload button
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO detection
    results = model(image)
    detected_objects = results.pandas().xyxy[0]["name"].tolist()
    st.write("### 🟢 Detected Objects:", detected_objects)

    if detected_objects:
        # Generate description using Gemini
        model_gemini = genai.GenerativeModel("gemini-pro")
        prompt = f"The image contains the following objects: {', '.join(detected_objects)}. Write a detailed description."
        response = model_gemini.generate_content(prompt)

        st.write("### 📝 Gemini Description:")
        st.write(response.text)
    else:
        st.warning("No objects detected.")
