import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Background")
st.write("# Background")

st.header("How do you answer this job application question?")
option = st.radio(
    label = "Will you now or in the future require sponsorship to work in the U.S.?",
    options = ("Yes","No"),
    index = None
)

st.subheader("The following may be true in your case:")
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