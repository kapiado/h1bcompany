# import pandas as pd
# import streamlit as st
# import numpy as np
# import warnings


# warnings.filterwarnings('ignore')

# # Inject CSS to expand page width and make the table fit on the screen
# st.markdown(
#     """
#     <style>
#     .reportview-container .main .block-container{
#         max-width: 100%;
#         padding: 1rem;
#     }
#     .reportview-container .main{
#         color: black;
#         background-color: white;
#     }
#     .dataframe-container {
#         overflow: auto;
#         white-space: nowrap;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.title('H-1B Visa Company Recommendation Form') 
# st.write("This app recommends companies that are likely to sponsor an H-1B visa based on your user input. Use the form below to get started!")

# # Load data
# df_original = pd.read_csv('/mount/src/h1bcompany/pages/merged_data_6_12_24.csv')

# # Create a copy of the data
# df_cleaned = df_original.copy()

# # Select relevant columns
# columns_list = [
#     'EMPLOYER_NAME_CLEAN', 'EMPLOYER_NAME', 'SOC_TITLE', 'FULL_WORKSITE_STATE', 'PREVAILING_WAGE_ANNUAL', 
#     'SECTOR_CODE', 'SUBSECTOR_CODE', 'SUBSECTOR_NAME', 'EMPLOYEE_COUNT_CATEGORY', 'COMPANY_AGE_CATEGORY', 'COMPANY_LINK', 'SPONSORED_2012.0', 
#     'SPONSORED_2013.0', 'SPONSORED_2014.0', 'SPONSORED_2015.0', 'SPONSORED_2016.0', 'SPONSORED_2017.0', 
#     'SPONSORED_2018.0', 'SPONSORED_2019.0', 'SPONSORED_2020.0', 'SPONSORED_2021.0', 'SPONSORED_2022.0', 
#     'SPONSORED_2023.0', 'SPONSORED_2024.0'
# ]
# df_cleaned = df_cleaned[columns_list]

# # Sponsorship year weights
# sponsorship_weights = {
#     'SPONSORED_2012.0': 0.0294, 'SPONSORED_2013.0': 0.0294, 'SPONSORED_2014.0': 0.0294, 
#     'SPONSORED_2015.0': 0.0294, 'SPONSORED_2016.0': 0.0588, 'SPONSORED_2017.0': 0.0588, 
#     'SPONSORED_2018.0': 0.0588, 'SPONSORED_2019.0': 0.0882, 'SPONSORED_2020.0': 0.0882, 
#     'SPONSORED_2021.0': 0.0882, 'SPONSORED_2022.0': 0.1471, 'SPONSORED_2023.0': 0.1471, 
#     'SPONSORED_2024.0': 0.1471
# }

# # Combine sponsored visas into one column
# df_cleaned['SPONSORED'] = sum(df_cleaned[col] * weight for col, weight in sponsorship_weights.items())
# df_cleaned['SPONSORED'] = df_cleaned['SPONSORED'].round()

# # Drop the sponsorship columns except the combined one
# columns_to_drop = list(sponsorship_weights.keys())
# columns_to_drop.remove('SPONSORED_2023.0')
# df_cleaned.drop(columns=columns_to_drop, inplace=True)

# # Load SOC Titles from Google Sheet
# sheet_id = "1oLjpm4KLNj-tUN_Pnbrk_ihU7bNylJwG"
# sheet_name = "Final"
# url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

# # Read the CSV file into a DataFrame
# soc_titles_df = pd.read_csv(url, dtype=str)

# # Fill NaN values with empty strings
# soc_titles_df["OCCUPATION"] = soc_titles_df["OCCUPATION"].fillna("")

# # Convert the OCCUPATION column to uppercase
# soc_titles_df["OCCUPATION"] = soc_titles_df["OCCUPATION"].str.strip().str.upper()

# # Drop duplicate rows based on the OCCUPATION column
# soc_titles_df = soc_titles_df.drop_duplicates(subset=["OCCUPATION"])
# # soc_titles_df = pd.read_csv(url, dtype=str).fillna("").drop_duplicates(subset=["OCCUPATION"])
# # soc_titles_df["OCCUPATION"] = soc_titles_df["OCCUPATION"].str.strip()

# # Create the form
# with st.form(key='my_form'):
#     st.subheader("Selections")

#     # text_search = st.text_input("Search for SOC Title", help="Type here to retrieve results in dropdown menu below.")
#     # filtered_df = soc_titles_df[soc_titles_df["OCCUPATION"].str.contains(text_search, case=False, na=False)]
#     # titleInfo = st.multiselect("Select SOC Title(s)", options=soc_titles_df["OCCUPATION"].tolist())
#     titleInfo = st.multiselect("Select SOC Title(s)", options=sorted(soc_titles_df["OCCUPATION"].tolist()))

#     # Industry codes
#     codeOptions = ['11 - Agriculture, Forestry, Fishing and Hunting', '22 - Utilities', 
#                    '31 - Manufacturing (Food, Beverage, Tobacco, Apparel, Leather, Textiles)', 
#                    '32 - Manufacturing (Paper, Printing, Petroleum, Coal, Chemicals, Plastics, Rubber, Nonmetallic)', 
#                    '33 - Manufacturing (Primary Metals, Fabricated Metal, Machinery, Computer and Electronic Products, Electrical Equipment and Appliances, Transportations Equipment, Furniture, Miscellaneous Manufacturing)', 
#                    '42 - Wholesale Trade', '44 - Retail Trade (Automotive Sales and Services, Home Furnishing and Improvement, Food and Beverage, Health and Personal Care, Clothing and Accessories, Gasoline Stations)', 
#                    '45 - Retail Trade (Sporting Goods, Hobbies, Books, Department Stores, General Merchandise Stores, Florists, Office Supplies, Pet Supplies, Art Dealers, Various Specialty Stores)', 
#                    '48 - Transportation and Warehousing (Air, Rail, Water, Truck, Transit, Pipeline, Scenic and Sightseeing Services, Transportation Support Activities)', 
#                    '51 - Information', '52 - Finance and Insurance', '53 - Real Estate and Rental and Leasing', 
#                    '54 - Professional, Scientific, and Technical Services', '55 - Management of Companies and Enterprises', 
#                    '56 - Administrative and Support and Waste Management and Remediation Services', '61 - Educational Services', 
#                    '62 - Health Care and Social Assistance', '71 - Arts, Entertainment, and Recreation', '72 - Accommodation and Food Services', 
#                    '81 - Other Services (except Public Administration)']
    
#     codeInfo = st.multiselect('Select industry/industries', codeOptions, help="Select the most appropriate Industry Code as found here https://www.census.gov/naics/?58967?yearbck=2022")

#     # Extract selected sector codes
#     selected_sector_codes = [int(code.split(' ')[0]) for code in codeInfo]
#     st.session_state.selected_sector_codes = selected_sector_codes

#     # Initialize subsector options
#     if 'subsector_options' not in st.session_state:
#         st.session_state.subsector_options = []

#     # Define a function to update subsector options
#     def update_subsector_options(selected_sector_codes):
#         if selected_sector_codes:
#             print("Selected sector codes:", selected_sector_codes)  # Debugging statement
            
#             # Filter the DataFrame based on selected sector codes
#             filtered_df = df_cleaned[df_cleaned['SECTOR_CODE'].isin(selected_sector_codes)]
            
#             # Check if any rows are filtered
#             if filtered_df.empty:
#                 print("No rows found for selected sector codes.")  # Debugging statement
#                 return
            
#             # Sort the filtered DataFrame by subsector code
#             sorted_df = filtered_df.sort_values(by='SUBSECTOR_CODE')
            
#             # Create a new column combining subsector code and name
#             sorted_df['SUBSECTOR_CODE_NAME'] = sorted_df['SUBSECTOR_CODE'].astype(str) + ' - ' + sorted_df['SUBSECTOR_NAME']
            
#             # Get unique subsector code names as a list
#             subsector_options = sorted_df['SUBSECTOR_CODE_NAME'].unique().tolist()

#             # Update session state with new subsector options
#             st.session_state.subsector_options = subsector_options
#             print("Updated subsector options:", st.session_state.subsector_options)  # Debugging statement
#         else:
#             # Clear subsector options in session state
#             st.session_state.subsector_options = []
#             print("No selected sector codes. Cleared subsector options.")  # Debugging statement

#     # Use session state for subsector options in multiselect widget
#     subsectorInfo = st.multiselect('Select Subsector Code(s)', st.session_state.subsector_options, help="Select the appropriate Subsector Code based on your selected Sector Code(s).")

#     # Call the function to update subsector options
#     update_subsector_options(st.session_state.selected_sector_codes)
#     # state_abbreviations = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "DISTRICT OF COLUMBIA", "FL", "FM", 
#     #                        "GA", "GU", "GUAM", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MH", 
#     #                        "MI", "MN", "MO", "MP", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", 
#     #                        "OK", "OR", "PA", "PR", "PUERTO RICO", "PW", "RI", "SC", "SD", "TN", "TX", "UT", "VA", 
#     #                        "VI", "VIRGIN ISLANDS", "VT", "WA", "WI", "WV", "WY"]

#     state_full_names = sorted(df_cleaned["FULL_WORKSITE_STATE"].unique().tolist())

#     stateInfo = st.multiselect('Select U.S. Work State/Territory(s)', state_full_names, help="Select the state/territory where you would like to work.")

#     employeenum_categories = ['<50', '51-200', '201-500', '501-1000', '1001-5000', '5001-10000', '10000+']
#     employeenumInfo = st.multiselect('Select Company Size(s)', employeenum_categories)

#     companyage_categories = ['Newly Established (0-2 years)','Early Stage (2-5 years)','Growth Stage (5-10 years)',
#                              'Mature Stage (10-20 years)','Established Stage (20-50 years)', 'Legacy Stage (50+ years)']
#     companyageInfo = st.multiselect('Select Company Age(s)', companyage_categories)

#     st.subheader("Weights of Importance")
#     body1 = st.empty()

#     text = """
#     Please indicate your level of importance for each of the above inputs.<br>
#     <b>How important is each input in finding a job?</b><br>
#     <i>Please refer to this scale:</i><br>
#     1 = Not important at all<br>2 = Less important<br>3 = Neutral<br>4 = Important<br>5 = Most important
#     """
#     body1.markdown(text, unsafe_allow_html=True)

#     titleWeight = st.slider('How important is **your role** (SOC Title) when looking for a job?', 1, 5, 3)
#     codeWeight = st.slider('How important is **the industry** when looking for a job?', 1, 5, 3)
#     stateWeight = st.slider('How important is **the state you work in** when looking for a job?', 1, 5, 3)
#     employeenumWeight = st.slider('How important is **company size** when looking for a job?', 1, 5, 3)
#     companyageWeight = st.slider('How important is **company age** when looking for a job?', 1, 5, 3)
    
    

#     submit_button = st.form_submit_button(label='Submit')

import pandas as pd
import streamlit as st
import numpy as np
import base64

# Function to load data (cached)
@st.cache
def load_data():
    df_original = pd.read_csv('/mount/src/h1bcompany/pages/merged_data_6_12_24.csv')

    # Select relevant columns
    columns_list = [
        'EMPLOYER_NAME_CLEAN', 'EMPLOYER_NAME', 'SOC_TITLE', 'FULL_WORKSITE_STATE', 'PREVAILING_WAGE_ANNUAL', 
        'SECTOR_CODE', 'SUBSECTOR_CODE', 'SUBSECTOR_NAME', 'EMPLOYEE_COUNT_CATEGORY', 'COMPANY_AGE_CATEGORY', 
        'COMPANY_LINK', 'SPONSORED_2012.0', 'SPONSORED_2013.0', 'SPONSORED_2014.0', 'SPONSORED_2015.0', 
        'SPONSORED_2016.0', 'SPONSORED_2017.0', 'SPONSORED_2018.0', 'SPONSORED_2019.0', 'SPONSORED_2020.0', 
        'SPONSORED_2021.0', 'SPONSORED_2022.0', 'SPONSORED_2023.0', 'SPONSORED_2024.0'
    ]
    df_cleaned = df_original[columns_list]

    # Combine sponsored visas into one column
    sponsorship_weights = {
        'SPONSORED_2012.0': 0.0294, 'SPONSORED_2013.0': 0.0294, 'SPONSORED_2014.0': 0.0294, 
        'SPONSORED_2015.0': 0.0294, 'SPONSORED_2016.0': 0.0588, 'SPONSORED_2017.0': 0.0588, 
        'SPONSORED_2018.0': 0.0588, 'SPONSORED_2019.0': 0.0882, 'SPONSORED_2020.0': 0.0882, 
        'SPONSORED_2021.0': 0.0882, 'SPONSORED_2022.0': 0.1471, 'SPONSORED_2023.0': 0.1471, 
        'SPONSORED_2024.0': 0.1471
    }
    df_cleaned['SPONSORED'] = sum(df_cleaned[col] * weight for col, weight in sponsorship_weights.items())
    df_cleaned['SPONSORED'] = df_cleaned['SPONSORED'].round()

    # Drop the sponsorship columns except the combined one
    columns_to_drop = list(sponsorship_weights.keys())
    columns_to_drop.remove('SPONSORED_2023.0')
    df_cleaned.drop(columns=columns_to_drop, inplace=True)

    return df_cleaned

# Function to load SOC Titles from Google Sheet (cached)
@st.cache
def load_soc_titles():
    sheet_id = "1oLjpm4KLNj-tUN_Pnbrk_ihU7bNylJwG"
    sheet_name = "Final"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    # Read the CSV file into a DataFrame
    soc_titles_df = pd.read_csv(url, dtype=str)

    # Fill NaN values with empty strings
    soc_titles_df["OCCUPATION"] = soc_titles_df["OCCUPATION"].fillna("")

    # Convert the OCCUPATION column to uppercase
    soc_titles_df["OCCUPATION"] = soc_titles_df["OCCUPATION"].str.strip().str.upper()

    # Drop duplicate rows based on the OCCUPATION column
    soc_titles_df = soc_titles_df.drop_duplicates(subset=["OCCUPATION"])

    return soc_titles_df

# Main Streamlit app code
def main():
    st.title('H-1B Visa Company Recommendation Form') 
    st.write("This app recommends companies that are likely to sponsor an H-1B visa based on your user input. Use the form below to get started!")

    # Load data (cached)
    df_cleaned = load_data()

    # Load SOC Titles from Google Sheet (cached)
    soc_titles_df = load_soc_titles()

    # Create the form
    with st.form(key='my_form'):
        st.subheader("Selections")

        titleInfo = st.multiselect("Select SOC Title(s)", options=sorted(soc_titles_df["OCCUPATION"].tolist()))
        codeOptions = ['11 - Agriculture, Forestry, Fishing and Hunting', '22 - Utilities', 
                       '31 - Manufacturing (Food, Beverage, Tobacco, Apparel, Leather, Textiles)', 
                       '32 - Manufacturing (Paper, Printing, Petroleum, Coal, Chemicals, Plastics, Rubber, Nonmetallic)', 
                       '33 - Manufacturing (Primary Metals, Fabricated Metal, Machinery, Computer and Electronic Products, Electrical Equipment and Appliances, Transportations Equipment, Furniture, Miscellaneous Manufacturing)', 
                       '42 - Wholesale Trade', '44 - Retail Trade (Automotive Sales and Services, Home Furnishing and Improvement, Food and Beverage, Health and Personal Care, Clothing and Accessories, Gasoline Stations)', 
                       '45 - Retail Trade (Sporting Goods, Hobbies, Books, Department Stores, General Merchandise Stores, Florists, Office Supplies, Pet Supplies, Art Dealers, Various Specialty Stores)', 
                       '48 - Transportation and Warehousing (Air, Rail, Water, Truck, Transit, Pipeline, Scenic and Sightseeing Services, Transportation Support Activities)', 
                       '51 - Information', '52 - Finance and Insurance', '53 - Real Estate and Rental and Leasing', 
                       '54 - Professional, Scientific, and Technical Services', '55 - Management of Companies and Enterprises', 
                       '56 - Administrative and Support and Waste Management and Remediation Services', '61 - Educational Services', 
                       '62 - Health Care and Social Assistance', '71 - Arts, Entertainment, and Recreation', '72 - Accommodation and Food Services', 
                       '81 - Other Services (except Public Administration)']
        codeInfo = st.multiselect('Select industry/industries', codeOptions, help="Select the most appropriate Industry Code as found here https://www.census.gov/naics/?58967?yearbck=2022")
        #  Extract selected sector codes
        selected_sector_codes = [int(code.split(' ')[0]) for code in codeInfo]
        
        # Update subsector options based on selected sector codes
            # Initialize session state for selected sector codes and subsector options
        if 'selected_sector_codes' not in st.session_state:
            st.session_state.selected_sector_codes = []
        
        if 'subsector_options' not in st.session_state:
            st.session_state.subsector_options = []

        # Function to update subsector options based on selected sector codes
        def update_subsector_options(selected_sector_codes):
            if selected_sector_codes:
                # Filter the DataFrame based on selected sector codes
                filtered_df = df_cleaned[df_cleaned['SECTOR_CODE'].isin(selected_sector_codes)]

                # Sort the filtered DataFrame by subsector code
                sorted_df = filtered_df.sort_values(by='SUBSECTOR_CODE')

                # Create a new column combining subsector code and name
                sorted_df['SUBSECTOR_CODE_NAME'] = sorted_df['SUBSECTOR_CODE'].astype(str) + ' - ' + sorted_df['SUBSECTOR_NAME']

                # Get unique subsector code names as a list
                subsector_options = sorted_df['SUBSECTOR_CODE_NAME'].unique().tolist()

                # Update session state with new subsector options
                st.session_state.subsector_options = subsector_options
                st.session_state.selected_sector_codes = selected_sector_codes
            else:
                # Clear subsector options in session state
                st.session_state.subsector_options = []
                st.session_state.selected_sector_codes = []
            update_subsector_options(codeInfo)

        # Use session state for subsector options in multiselect widget
        subsectorInfo = st.multiselect('Select Subsector Code(s)', st.session_state.subsector_options, help="Select the appropriate Subsector Code based on your selected Sector Code(s).")

        state_full_names = sorted(df_cleaned["FULL_WORKSITE_STATE"].unique().tolist())
        stateInfo = st.multiselect('Select U.S. Work State/Territory(s)', state_full_names, help="Select the state/territory where you would like to work.")

        employeenum_categories = ['<50', '51-200', '201-500', '501-1000', '1001-5000', '5001-10000', '10000+']
        employeenumInfo = st.multiselect('Select Company Size(s)', employeenum_categories)

        companyage_categories = ['Newly Established (0-2 years)','Early Stage (2-5 years)','Growth Stage (5-10 years)',
                                 'Mature Stage (10-20 years)','Established Stage (20-50 years)', 'Legacy Stage (50+ years)']
        companyageInfo = st.multiselect('Select Company Age(s)', companyage_categories)

        st.subheader("Weights of Importance")
        body1 = st.empty()

        text = """
        Please indicate your level of importance for each of the above inputs.<br>
        <b>How important is each input in finding a job?</b><br>
        <i>Please refer to this scale:</i><br>
        1 = Not important at all<br>2 = Less important<br>3 = Neutral<br>4 = Important<br>5 = Most important
        """
        body1.markdown(text, unsafe_allow_html=True)

        titleWeight = st.slider('How important is **your role** (SOC Title) when looking for a job?', 1, 5, 3)
        codeWeight = st.slider('How important is **the industry** when looking for a job?', 1, 5, 3)
        stateWeight = st.slider('How important is **the state you work in** when looking for a job?', 1, 5, 3)
        employeenumWeight = st.slider('How important is **company size** when looking for a job?', 1, 5, 3)
        companyageWeight = st.slider('How important is **company age** when looking for a job?', 1, 5, 3)

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        # Filter the dataframe based on user input
        def apply_filters(df):
            filters = {
                'SOC_TITLE': titleInfo,
                'SECTOR_CODE': selected_sector_codes,
                'FULL_WORKSITE_STATE': stateInfo,
                'EMPLOYEE_COUNT_CATEGORY': employeenumInfo,
                'COMPANY_AGE_CATEGORY': companyageInfo
            }

            for col, values in filters.items():
                if values:
                    df = df[df[col].isin(values)]

            return df

        filtered_df = apply_filters(df_cleaned)

        if filtered_df.empty:
            st.warning("No companies found matching your criteria. Please adjust your filters and try again.")
        else:
            def topsis(df, weights):
                df = df.copy()
                numeric_cols = df.select_dtypes(include=[np.number]).columns.intersection(weights.keys())
                
                # Normalize numeric columns
                for col in numeric_cols:
                    df[col] = df[col] / np.sqrt((df[col] ** 2).sum())

                # Apply weights to numeric columns
                weighted_scores = df[numeric_cols].multiply(weights, axis=1)
                df['topsis_score'] = weighted_scores.sum(axis=1)
                return df.sort_values(by='topsis_score', ascending=False)

            weights = {
                'EMPLOYEE_COUNT_CATEGORY': employeenumWeight,
                'COMPANY_AGE_CATEGORY': companyageWeight,
                'SPONSORED': 5  # Sponsor weight is fixed
            }

            # Normalize weights
            total_weight = sum(weights.values())
            normalized_weights = {k: v / total_weight for k, v in weights.items()}

            # Check and align weights with numeric columns
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            valid_weights = {k: v for k, v in normalized_weights.items() if k in numeric_cols}

            if len(valid_weights) == 0:
                st.error("No valid numeric columns found for TOPSIS calculation.")
            else:
                result_df = topsis(filtered_df, valid_weights)

                # Add job count column
                result_df['JOB_COUNT'] = result_df.groupby('EMPLOYER_NAME_CLEAN')['EMPLOYER_NAME_CLEAN'].transform('count')

                # Condense worksite state
                grouped_ws = result_df.groupby('EMPLOYER_NAME_CLEAN')['FULL_WORKSITE_STATE'].agg(list).reset_index()
                grouped_ws['FULL_WORKSITE_STATE_LIST'] = grouped_ws['FULL_WORKSITE_STATE'].apply(lambda x: list(set(x)))  # Remove duplicates
                grouped_ws['OTHER_WORKSITE_STATE'] = grouped_ws['FULL_WORKSITE_STATE'].apply(lambda x: x[1:] if len(x) > 1 else [])
                result_df = result_df.merge(grouped_ws[['EMPLOYER_NAME_CLEAN', 'FULL_WORKSITE_STATE_LIST', 'OTHER_WORKSITE_STATE']], on='EMPLOYER_NAME_CLEAN', how='left')
                result_df.rename(columns={'FULL_WORKSITE_STATE_LIST': 'FULL_WORKSITE_STATE'}, inplace=True)

                # Condense SOC title
                grouped_soc = result_df.groupby('EMPLOYER_NAME_CLEAN')['SOC_TITLE'].agg(list).reset_index()
                grouped_soc['SOC_TITLE_LIST'] = grouped_soc['SOC_TITLE'].apply(lambda x: list(set(x)))  # Remove duplicates
                grouped_soc['OTHER_SOC_TITLES'] = grouped_soc['SOC_TITLE'].apply(lambda x: x[1:] if len(x) > 1 else [])
                result_df = result_df.merge(grouped_soc[['EMPLOYER_NAME_CLEAN', 'SOC_TITLE_LIST', 'OTHER_SOC_TITLES']], on='EMPLOYER_NAME_CLEAN', how='left')
                result_df.rename(columns={'SOC_TITLE_LIST': 'SOC_TITLE'}, inplace=True)

                # Drop any possible duplicate columns before final selection
                result_df = result_df.loc[:,~result_df.columns.duplicated()]

                # Display only unique outputs
                result_df.drop_duplicates(subset=['EMPLOYER_NAME_CLEAN'], keep='first', inplace=True)
                result_df = result_df[['EMPLOYER_NAME', 'SOC_TITLE', 'FULL_WORKSITE_STATE', 'PREVAILING_WAGE_ANNUAL', 
                                    'EMPLOYEE_COUNT_CATEGORY', 'COMPANY_AGE_CATEGORY', 'COMPANY_LINK', 
                                    'JOB_COUNT', 'OTHER_WORKSITE_STATE', 'OTHER_SOC_TITLES']]

                # Display top recommendations
                st.write("#### Top 10 Recommendations")
                st.dataframe(result_df.head(10), hide_index=True)

                # Download link for full results
                csv = result_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="h1b_recommendations.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

    # Add a horizontal line using HTML
    st.write("<hr>", unsafe_allow_html=True)

    st.write("### About the App")
    st.write("This app recommends companies likely to sponsor H-1B visas based on user input. Use the form on the left to select SOC Titles, Industries, U.S. States/Territories, Company Sizes, and Company Ages, along with indicating the importance of each factor. Click 'Submit' to view the top 10 recommendations.")

    st.write("#### Disclaimer")
    st.write("This tool provides recommendations based on aggregated data and user inputs. Actual results may vary. Always verify with official sources and consult legal experts when making decisions regarding H-1B visa applications.")

if __name__ == '__main__':
    main()
    # if submit_button:
    # # Filter the dataframe based on user input
    #     def apply_filters(df):
    #         filters = {
    #             'SOC_TITLE': titleInfo,
    #             'SECTOR_CODE': selected_sector_codes,
    #             'FULL_WORKSITE_STATE': stateInfo,
    #             'EMPLOYEE_COUNT_CATEGORY': employeenumInfo,
    #             'COMPANY_AGE_CATEGORY': companyageInfo
    #         }

    #         for col, values in filters.items():
    #             if values:
    #                 df = df[df[col].isin(values)]

    #         return df

    #     filtered_df = apply_filters(df_cleaned)

    #     if filtered_df.empty:
    #         st.warning("No companies found matching your criteria. Please adjust your filters and try again.")
    #     else:
    #         def topsis(df, weights):
    #             df = df.copy()
    #             numeric_cols = df.select_dtypes(include=[np.number]).columns.intersection(weights.keys())
                
    #             # Normalize numeric columns
    #             for col in numeric_cols:
    #                 df[col] = df[col] / np.sqrt((df[col] ** 2).sum())

    #             # Apply weights to numeric columns
    #             weighted_scores = df[numeric_cols].multiply(weights, axis=1)
    #             df['topsis_score'] = weighted_scores.sum(axis=1)
    #             return df.sort_values(by='topsis_score', ascending=False)

    #         weights = {
    #             'EMPLOYEE_COUNT_CATEGORY': employeenumWeight,
    #             'COMPANY_AGE_CATEGORY': companyageWeight,
    #             'SPONSORED': 5  # Sponsor weight is fixed
    #         }

    #         # Normalize weights
    #         total_weight = sum(weights.values())
    #         normalized_weights = {k: v / total_weight for k, v in weights.items()}

    #         # Check and align weights with numeric columns
    #         numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    #         valid_weights = {k: v for k, v in normalized_weights.items() if k in numeric_cols}

    #         if len(valid_weights) == 0:
    #             st.error("No valid numeric columns found for TOPSIS calculation.")
    #         else:
    #             result_df = topsis(filtered_df, valid_weights)

    #             # Add job count column
    #             result_df['JOB_COUNT'] = result_df.groupby('EMPLOYER_NAME_CLEAN')['EMPLOYER_NAME_CLEAN'].transform('count')

    #             # # Condense worksite state and SOC title
    #             # grouped_ws = result_df.groupby('EMPLOYER_NAME_CLEAN')['FULL_WORKSITE_STATE'].agg(list).reset_index()
    #             # grouped_ws['WORKSITE_STATE'] = grouped_ws['FULL_WORKSITE_STATE'].apply(lambda x: list(set(x)))  # Remove duplicates
    #             # grouped_ws['OTHER_WORKSITE_STATE'] = grouped_ws['FULL_WORKSITE_STATE'].apply(lambda x: x[1:] if len(x) > 1 else [])
    #             # result_df = result_df.merge(grouped_ws, on='EMPLOYER_NAME_CLEAN', how='left')
    #             # result_df.rename(columns={'FULL_WORKSITE_STATE_x': 'WORKSITE_STATE'}, inplace=True)
    #             # result_df.drop(columns=['FULL_WORKSITE_STATE_y'], inplace=True)

    #             # grouped_soc = result_df.groupby('EMPLOYER_NAME_CLEAN')['SOC_TITLE'].agg(list).reset_index()
    #             # grouped_soc['SOC_TITLE_LIST'] = grouped_soc['SOC_TITLE'].apply(lambda x: list(set(x)))  # Remove duplicates
    #             # grouped_soc['OTHER_SOC_TITLES'] = grouped_soc['SOC_TITLE_LIST'].apply(lambda x: x[1:] if len(x) > 1 else [])
    #             # result_df = result_df.merge(grouped_soc, on='EMPLOYER_NAME_CLEAN', how='left')
    #             # result_df.rename(columns={'SOC_TITLE_x': 'SOC_TITLE'}, inplace=True)
    #             # result_df.drop(columns=['SOC_TITLE_y'], inplace=True)

    #             # # Display only unique outputs
    #             # result_df.drop_duplicates(subset=['EMPLOYER_NAME_CLEAN'], keep='first', inplace=True)
    #             # result_df = result_df[['EMPLOYER_NAME', 'SOC_TITLE', 'FULL_WORKSITE_STATE', 'PREVAILING_WAGE_ANNUAL', 
    #             #                     'EMPLOYEE_COUNT_CATEGORY', 'COMPANY_AGE_CATEGORY', 'COMPANY_LINK', 'SPONSORED', 
    #             #                     'JOB_COUNT', 'OTHER_WORKSITE_STATE', 'OTHER_SOC_TITLES']]

    #             # Condense worksite state
    #             grouped_ws = result_df.groupby('EMPLOYER_NAME_CLEAN')['FULL_WORKSITE_STATE'].agg(list).reset_index()
    #             grouped_ws['FULL_WORKSITE_STATE_LIST'] = grouped_ws['FULL_WORKSITE_STATE'].apply(lambda x: list(set(x)))  # Remove duplicates
    #             grouped_ws['OTHER_WORKSITE_STATE'] = grouped_ws['FULL_WORKSITE_STATE_LIST'].apply(lambda x: x[1:] if len(x) > 1 else [])
    #             result_df = result_df.merge(grouped_ws[['EMPLOYER_NAME_CLEAN', 'FULL_WORKSITE_STATE_LIST', 'OTHER_WORKSITE_STATE']], on='EMPLOYER_NAME_CLEAN', how='left')
    #             result_df.rename(columns={'FULL_WORKSITE_STATE_LIST': 'FULL_WORKSITE_STATE'}, inplace=True)

    #             # Condense SOC title
    #             grouped_soc = result_df.groupby('EMPLOYER_NAME_CLEAN')['SOC_TITLE'].agg(list).reset_index()
    #             grouped_soc['SOC_TITLE_LIST'] = grouped_soc['SOC_TITLE'].apply(lambda x: list(set(x)))  # Remove duplicates
    #             grouped_soc['OTHER_SOC_TITLES'] = grouped_soc['SOC_TITLE_LIST'].apply(lambda x: x[1:] if len(x) > 1 else [])
    #             result_df = result_df.merge(grouped_soc[['EMPLOYER_NAME_CLEAN', 'SOC_TITLE_LIST', 'OTHER_SOC_TITLES']], on='EMPLOYER_NAME_CLEAN', how='left')
    #             result_df.rename(columns={'SOC_TITLE_LIST': 'SOC_TITLE'}, inplace=True)

    #             # Drop any possible duplicate columns before final selection
    #             result_df = result_df.loc[:,~result_df.columns.duplicated()]

    #             # Display only unique outputs
    #             result_df.drop_duplicates(subset=['EMPLOYER_NAME_CLEAN'], keep='first', inplace=True)
    #             result_df = result_df[['EMPLOYER_NAME', 'SOC_TITLE', 'FULL_WORKSITE_STATE', 'PREVAILING_WAGE_ANNUAL', 
    #                                 'EMPLOYEE_COUNT_CATEGORY', 'COMPANY_AGE_CATEGORY', 'COMPANY_LINK', 
    #                                 'JOB_COUNT', 'OTHER_WORKSITE_STATE', 'OTHER_SOC_TITLES']]

    #             # Display top recommendations
    #             st.write("#### Top 10 Recommendations")
    #             st.dataframe(result_df.head(10), hide_index=True)