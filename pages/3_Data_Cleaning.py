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

# Centered layout
st.write("")
col1, col2, col3 = st.columns([1, 2, 1])

# Display the image in the middle column
# Adding image with caption
with col2:
    image = Image.open(path+'h1b_data_flowchart.png')
    st.image(image)