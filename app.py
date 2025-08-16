import streamlit as st
import google.generativeai as genai

# Gemini API key (add your key in Streamlit secrets)
API_KEY = st.secrets["GEMINI_API"]

# Configure Gemini API
genai.configure(api_key=API_KEY)

def get_gemini_description(object_name):
    """Generate a one-sentence description using Gemini API."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Describe in one sentence what a {object_name} is.")
    return response.text.strip()

# Streamlit UI
st.title("Gemini API: Object Description Generator")
st.write("Enter an object name, and get a creative one-line description!")

object_name = st.text_input("Enter object name:", value="car")

if st.button("Generate Description"):
    if object_name:
        with st.spinner("Generating description..."):
            description = get_gemini_description(object_name)
        st.success("Description Generated!")
        st.write(f"**{object_name.capitalize()}:** {description}")
    else:
        st.warning("Please enter an object name.")
