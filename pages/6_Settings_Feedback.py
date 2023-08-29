import streamlit as st

st.set_page_config(page_title="AnalytiXpress", page_icon="./assets/logo.png")

# User Settings section
st.markdown("# Settings")
st.markdown("User Settings")
st.sidebar.header("Settings and Feedback")

# Feedback Form Section
st.markdown("# Feedback Form")
feedbackName = st.text_input("Enter your name here: ")
feedbackContact = st.text_input("Enter your contact here: ")
feedbackTitle = st.text_input("Enter your feedback title here: ")
feedbackDesc = st.text_input("Enter your feedback description here: ")

if st.button("Submit Form"):
    st.write("Feedback Form Submitted!")
    st.write("Name: " + feedbackName)
    st.write("Contact: " + feedbackContact)
    st.write("Title: " + feedbackTitle)
    st.write("Detail: " + feedbackDesc)
