# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:46:34 2023

@author: katri
"""

import streamlit as st
import streamlit.components.v1 as components
# import numpy as np
import pandas as pd
# import plotly.express as px          # Plotly
# import plotly.graph_objects as go
from PIL import Image

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

# Adding path to Images folder
path = "Images/"

# Page title
st.write("# Data-Driven Decision Support System for International Job Applicants")
st.caption("A Master Project submitted to the Faculty of California Polytechnic State University, San Luis Obispo")
st.caption("In Partial Fulfillment of the Requirements for the Degree of Master of Science in Engineering Management")

# Centered layout
st.write("")
col1, col2, col3 = st.columns([1, 2, 1])

# Display the image in the middle column
# Adding image with caption
with col2:
    image = Image.open(path+'h1b-stress-visual.jpg')
    st.image(image,caption='Figure 1: Stress Caused From H-1B Visa System')

# Problem Statement area with header and body
st.header('Problem Statement')
body1 = st.empty()
body1.write("International students graduating from U.S. educational institutions encounter difficulties in identifying employers who are open to sponsoring H-1B visas.")

# Background area with header and body
st.header('Background')
body2 = st.empty()
body2.write('The **absence of a reliable system** providing insights and recommendations on potential visa sponsors impedes their ability to target suitable employers. This limitation significantly **reduces their chances of securing employment in their respective fields** and leveraging their education and skills gained in the United States.')

#new line and research objectives content
st.write("")
st.header('Research Objectives')
st.write("1. Develop and evaluate mathematical models to optimize the matching process between international candidates and potential employers for H-1B visa sponsorship.​")
st.write("2. Assess the effectiveness and usability of the H-1B visa sponsor recommendation system through user testing, feedback analysis, and performance metrics evaluation.")
st.write("3. Implement accessibility features and design enhancements on the recommendation system website to ensure it is user-friendly, culturally sensitive, and accommodating to the diverse needs of international applicants, regardless of language proficiency or technological familiarity.")

#st.subheader('Descriptive')
#st.write("We will create a novel analytical framework to address immigration issues and provide essential information for aspiring permanent residents in the US. These analytical frameworks will have a decision support system that people can leverage to make better decisions, with a data analytics page displaying trends and insights from immigration data, helping users understand the system and manage expectations.")
# st.subheader('Predictive')
# st.write("The predictive analytics page will allow users to input data and receive personalized estimates on their immigration timeline based on a variety of factors. Through this approach, the DSS will empower applicants and promote a transparent and efficient immigration process.")

#new line and team intro
st.write("")
st.header('The Team')
st.write('**Team Members:** Katrina Apiado, Ryan Keany')
st.write('**Cal Poly Computer Science Research Fellows:** Jordan Anthony Costa, Dr. Joydeep Mukherjee')
st.write('**Advisor:** Dr. Puneet Agarwal')
