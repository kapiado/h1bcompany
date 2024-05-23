import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Adding path to Images folder
path = "Images/"


# st.set_page_config(
#     page_title="Background",
#     page_icon="👋",
# )
#st.markdown("# Overview")
st.set_page_config(page_title="Data Cleaning")
st.write("# Data Cleaning")

body1 = st.empty()
body1.write("The flowchart below illustrates the process of obtaining and manipulating the data used in this project.")

# image = Image.open(path+'h1b_data_flowchart.png')
# st.image(image)

image = Image.open(path+'h1b-company-flowchart.png')
st.image(image)