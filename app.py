import streamlit as st
import google.generativeai as genai

# Gemini API Key from Streamlit secrets
API_KEY = st.secrets.get("GEMINI_API")
if not API_KEY:
    st.error("GEMINI_API key not found in Streamlit secrets!")
    st.stop()

genai.configure(api_key=API_KEY)

def get_gemini_description(object_name):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Describe in one sentence what a {object_name} is.")
    return response.text.strip()

st.title("AI Object Describer")
object_name = st.text_input("Enter an object to describe:")

if st.button("Generate Description"):
    if object_name.strip():
        with st.spinner("Generating description..."):
            description = get_gemini_description(object_name)
            st.success(description)
    else:
        st.warning("Please enter an object name.")
