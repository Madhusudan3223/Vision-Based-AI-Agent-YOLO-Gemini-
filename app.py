import streamlit as st
from google.generativeai import TextGenerationClient

# Initialize Google Generative AI client
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
client = TextGenerationClient(api_key=API_KEY)

st.title("AI Object Describer")
st.write("Enter an object to describe:")

# Input from user
object_name = st.text_input("Object Name")

if st.button("Describe"):
    if object_name:
        prompt = f"Describe the object '{object_name}' in detail."
        response = client.generate_text(model="text-bison-001", prompt=prompt, max_output_tokens=200)
        st.success(response.text)
    else:
        st.error("Please enter an object name!")
