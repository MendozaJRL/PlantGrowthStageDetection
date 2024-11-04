import streamlit as st

# Title of the app
st.title("Simple Streamlit App")

# Text input for the user's name
name = st.text_input("Enter your name:")

# Number input for the user's age
age = st.number_input("Enter your age:", min_value=0)

# Button to submit
if st.button("Submit"):
    if name and age:
        st.success(f"Hello, {name}! You are {age} years old.")
    else:
        st.error("Please enter both your name and age.")
