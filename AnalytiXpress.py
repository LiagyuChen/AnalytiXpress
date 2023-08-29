import streamlit as st

st.set_page_config(
    page_title="AnalytiXpress",
    page_icon="assets/logo.png",
)

st.write("# Welcome to AnalytiXpress! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    AnalytiXpress is a No-Code Data Science Platform built by StreamLit!
    
    **👈 Select a page from the sidebar** to start the journey of AnalytiXpress!
    
    ## AnalytiXpress Features
    
    ### Data Loading and Editing:
    1. Load data from Excel, CSV, and Databases.
    2. Edit Dataframe.
    3. Data Cleaning and Data Transformation.
    
    ### Data Visualization:
    1. Scientific Research style visualization charts.
    2. Business Intelligence style visualization charts.
    3. Custom chart settings.
    
    ### Dashboard Creation:
    1. Creat dashboard through drag & drop.
    2. Custom dashboard settings.
    
    ### Machine Learning:
    1. A simple UI to apply ML algorithms without coding.
    2. Optimization through parameter adjusting.
    
    ### Large Language Model:
    1. To edit dataset through natural languages.
    2. To create charts through natural languages.
    3. To summarize and extract data through dataset Q & A. 
    
    ## About the Project
    - Author: Jack Chen
    - Date: September, 2023
    - [Link to the Github Repository](https://github.com/streamlit/demo-uber-nyc-pickups)
    - This project will be update to a new version in the future.
"""
)