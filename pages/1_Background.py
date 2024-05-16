import streamlit as st
import streamlit.components.v1 as components

# Adding path to Images folder
path = "Images/"

st.set_page_config(page_title="Background")
st.write("# Background")

st.subheader("How do you answer this job application question?")
# body1 = st.empty()
# body1.write("Will you now or in the future require sponsorship to work in the U.S.?")

image = Image.open(path+'will-you-now-require-sponsorship.jpg')
st.image(image,caption='Figure 2: Question')


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