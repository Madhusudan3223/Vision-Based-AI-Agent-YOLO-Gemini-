import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai

# 🔑 Load Gemini API key from secret file
with open("GEMINI_API", "r") as f:
    api_key = f.read().strip()

genai.configure(api_key=api_key)

# ✅ Use correct model name (not gemini-pro)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# 🎯 Load YOLO model
yolo_model = YOLO("yolov8n.pt")

st.title("🚀 YOLO + Gemini AI App")
st.write("Upload an image, detect objects with YOLO, and get insights from Gemini AI.")

# 📂 Upload image
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    # Run YOLO detection
    results = yolo_model(image)
    detected_objects = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            detected_objects.append(label)

    st.subheader("🔍 YOLO Detection Results")
    results[0].show()  # show detection in logs (optional)
    st.write("✅ Detected Objects:", detected_objects)

    # 🤖 Ask Gemini for insights
    if detected_objects:
        prompt = f"The image contains: {', '.join(detected_objects)}. Provide a short, interesting description."
        try:
            response = gemini_model.generate_content(prompt)
            st.subheader("🤖 Gemini AI Insights")
            st.write(response.text)
        except Exception as e:
            st.error(f"Gemini API Error: {e}")
