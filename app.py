import streamlit as st
import google.generativeai as genai

# Configure API key
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your API key
genai.configure(api_key=API_KEY)

st.title("AI Object Describer")
st.write("Enter an object to describe:")

# Input from user
object_name = st.text_input("Object Name")

if st.button("Describe"):
    if object_name:
        response = genai.generate_text(
            model="text-bison-001",
            prompt=f"Describe the object '{object_name}' in detail.",
            max_output_tokens=200
        )
        st.success(response.result)
    else:
        st.error("Please enter an object name!")
