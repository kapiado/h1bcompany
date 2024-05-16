import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Adding path to Images folder
path = "Images/"

st.set_page_config(page_title="Background")
st.write("# Background")

st.header("How do you answer this job application question?")

# Centered layout
st.write("")
col1, col2, col3 = st.columns([1, 2, 1])

# Display the image in the middle column
# Adding image with caption
with col2:
    image1 = Image.open(path+'will-you-now-require-sponsorship.jpg')
    st.image(image1,caption='Figure 2: Question')

option = st.radio(
    label = "Will you now or in the future require sponsorship to work in the U.S.?",
    options = ("Yes","No"),
    index = None
)

body2 = st.empty()
body2.write("The following may be true in your case:")
if option == "Yes":
    #st.write("Currently do not possess the necessary work authorization (such as a U.S. work visa) to legally work in the U.S.")
    st.markdown("""
                - Currently do not possess the necessary work authorization (such as a U.S. work visa) to legally work in the U.S.
                
                **AND**

                - Need the company's assistance in obtaining the required visa or work permit
                """)

if option == "No":
    st.markdown("""
                - U.S. citizen
                - A permanent resident (Green Card holder)

                **OR**

                - Possess type of work authorization that does not require employer sponsorship (certain types of visas that allow unrestricted work rights)
                """)
    
# Add a horizontal line using HTML
st.write("<hr>", unsafe_allow_html=True)

st.header("What is H-1B Visa?")
st.subheader("About the H-1B Visa Program")

# Create two columns with st.columns()
col1, col2 = st.columns(2)

# Add elements to the first column
with col1:
    image2 = Image.open(path+'visa_h1b.jpg')
    st.image(image2,caption='Figure 3: H-1B Visa')

# Add elements to the second column
with col2:
    st.markdown("""
                - H-1B visa program is a **non-immigrant visa category** in the United States
                - Allow U.S. employers to **hire foreign workers in specialty occupations** that require theoretical or technical expertise
                - Enables companies to **temporarily** employ foreign professionals in occupations that typically **require a higher education degree**)
                """)

# Add a horizontal line using HTML
st.write("<hr>", unsafe_allow_html=True)

st.header("H-1B Visa Eligibility")

# Create two columns with st.columns()
col1, col2, col3 = st.columns(3)

# Add elements to the first column
with col1:
    image3 = Image.open(path+'work_icon.png')
    st.image(image3)
    body1 = st.empty()
    body1.write("**Job Offer**")
    body2 = st.empty()
    body2.write('Hold a job offer from a U.S. employer for a specialty occupation')

# Add elements to the second column
with col2:
    image4 = Image.open(path+'contract_icon.png')
    st.image(image4)
    body3 = st.empty()
    body3.write("**Academic/Work Credentials**")
    body4 = st.empty()
    body4.write('Possess the necessary academic credentials or work experience required for the position')

# Add elements to the third column
with col3:
    image5 = Image.open(path+'work_icon.png')
    st.image(image5)
    body5 = st.empty()
    body5.write("**Prevailing Wage**")
    body6 = st.empty()
    body6.write('The job offered must pay at least the prevailing wage in the location where the work will be performed')
