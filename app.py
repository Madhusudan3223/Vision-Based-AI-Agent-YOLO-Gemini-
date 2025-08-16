import streamlit as st
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai
import os

# 🚀 Title
st.title("YOLO + Gemini AI App")
st.write("Upload an image, detect objects with YOLO, and get insights from Gemini AI.")

# 🔑 Load Gemini API Key from Streamlit secrets
genai.configure(api_key=st.secrets["GEMINI_API"])

# 📦 Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 📂 Upload an image
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    # 🔍 YOLO detection
    results = model(image)
    result_image = results[0].plot()  # annotated image
    st.image(result_image, caption="🔍 YOLO Detection Results", use_container_width=True)

    # ✅ Extract detected objects
    detected_objects = results[0].names
    object_counts = {}
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]
        object_counts[label] = object_counts.get(label, 0) + 1

    st.subheader("✅ Detected Objects")
    st.write(", ".join([f"{obj} ({count})" for obj, count in object_counts.items()]))

    # 🤖 Ask Gemini AI for insights
    st.subheader("🤖 Gemini AI Insights")

    prompt = f"""
    I detected the following objects in an image: {object_counts}.
    Please describe the scene in natural language and explain what might be happening.
    """

    model_gemini = genai.GenerativeModel("gemini-pro")
    response = model_gemini.generate_content(prompt)

    if response and response.text:
        st.write(response.text)
    else:
        st.write("⚠️ No insights received from Gemini.")
