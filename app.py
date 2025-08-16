import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image

# Configure Gemini with API key from Streamlit secrets
genai.configure(api_key=st.secrets["GEMINI_API"])

# Load YOLO model
model = YOLO("yolov8n.pt")  # small, fast model

# Streamlit UI
st.title("🔍 Vision-Based AI Agent (YOLO + Gemini)")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Read image
    image = Image.open(tfile.name)
    img_array = np.array(image)

    # YOLO detection
    results = model(img_array)
    detected_classes = []
    annotated_img = img_array.copy()

    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            detected_classes.append(cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, cls, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Show YOLO result
    st.image(annotated_img, caption="YOLO Detection", use_container_width=True)

    # Ask Gemini for description
    if detected_classes:
        prompt = f"Describe an image containing the following objects: {', '.join(detected_classes)}"
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        response = model_gemini.generate_content(prompt)

        st.subheader("✨ Gemini Description")
        st.write(response.text)
    else:
        st.warning("No objects detected.")
