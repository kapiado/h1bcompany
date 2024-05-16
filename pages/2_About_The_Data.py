import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# st.set_page_config(
#     page_title="Background",
#     page_icon="👋",
# )
#st.markdown("# Overview")
st.set_page_config(page_title="About the Data")
st.write("# About the Data")

st.header("H-1B Visa Data")

# Adding path to Images folder
path = "Images/"

# Create two columns with st.columns()
col1, col2 = st.columns(2)

# Add elements to the first column
with col1:
    image = Image.open(path+'USCIS_logo.png')
    st.image(image,caption='Figure 3: USCIS Logo')

# Add elements to the second column
with col2:
    image2 = Image.open(path+'Seal_of_the_United_States_Department_of_Labor.png')
    st.image(image2,caption='Figure 4: Seal of the United States Department of Labor',width=400)


body1 = st.empty()
body1.write("The H-1B visa data was provided by the United States Department of Labor in conjunction with the United States Citizenship and Immigration Services (USCIS).")
#The timeframe of the data, so the data was compiled from those years specifically.")

body2 = st.empty()
body2.write("Datasets can be found at this link under 'LCA Programs (H-1B, H-1B1, E-3)': https://www.dol.gov/agencies/eta/foreign-labor/performance")

# Add a horizontal line using HTML
st.write("<hr>", unsafe_allow_html=True)

st.header("Company Data")

# Create two columns with st.columns()
col1, col2 = st.columns(2)

# Add elements to the first column
with col1:
    image3 = Image.open(path+'ScrapeStorm-logo.png')
    st.image(image3,caption='Figure 5: ScrapeStorm Logo')

# Add elements to the second column
with col2:
    image4 = Image.open(path+'Glassdoor_Logo.png')
    st.image(image4,caption='Figure 6: Glassdoor Logo')

body2 = st.empty()
body2.write("With the assistance of research fellows from the Cal Poly Computer Science Department, we were able to obtain company data through web scraping. Using a paid subscription for ScrapeStorm, company data was obtained by web scraping Glassdoor links for each company.")

