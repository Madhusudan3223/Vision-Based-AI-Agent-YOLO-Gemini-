import streamlit as st
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image

# ✅ Read Gemini API key from Streamlit secrets
api_key = st.secrets["GEMINI_API"]

# Configure Gemini
genai.configure(api_key=api_key)

# Load YOLO model
model = YOLO("yolov8n.pt")

st.title("🚀 YOLO + Gemini AI App")
st.write("Upload an image, detect objects with YOLO, and get insights from Gemini AI.")

uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Uploaded Image", use_container_width=True)

    # Run YOLO detection
    results = model(img)
    st.subheader("🔍 YOLO Detection Results")
    results.show()
    st.write("✅ Detected Objects")
    detected_objects = results[0].names
    detected_labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
    st.write("Objects Detected:", ", ".join(detected_labels))

    # Ask Gemini for insights
    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"The following objects were detected: {', '.join(detected_labels)}. Provide a short description of the scene."
        response = model_gemini.generate_content(prompt)
        st.subheader("🤖 Gemini AI Insights")
        st.write(response.text)
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
