
# Import libraries
import pandas as pd             # Pandas
import streamlit as st          # Streamlit
import numpy as np              # NumPy
# import matplotlib.pyplot as plt # Matplotlib
# import seaborn as sns           # Seaborn

# Module to save and load Python objects to and from files
# import pickle 
# import joblib

# Package to implement Random Forest Model
# import sklearn
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier


import warnings
warnings.filterwarnings('ignore')

st.title('H-1B Visa Company Recommendation Form') 

st.write("This app recommends companies that are likely to sponsor an H-1B visa based on your user input. Use the form below to get started!")
         #uses *x* inputs to determine companies that would be a good fit for you to apply for. Use the form below to get started!")
import os
print("Current working directory:", os.getcwd())
# df_original = pd.read_csv('merged_data_5_16_24.csv')

# # create copy of compete dataframe to be manipulated
# df_cleaned = df_original.copy()
# df_cleaned.info()

# # Select only the relevant columns
# columns_list = [
#     'EMPLOYER_NAME_CLEAN',
#     'EMPLOYER_NAME',
#     'SOC_TITLE',
#     'WORKSITE_STATE',
#     'PREVAILING_WAGE_ANNUAL',
#     'SECTOR_CODE',
#     'EMPLOYEE_COUNT_CATEGORY',
#     'COMPANY_AGE',
#     'COMPANY_LINK',
#     'SPONSORED_2012.0',
#     'SPONSORED_2013.0',
#     'SPONSORED_2014.0',
#     'SPONSORED_2015.0',
#     'SPONSORED_2016.0',
#     'SPONSORED_2017.0',
#     'SPONSORED_2018.0',
#     'SPONSORED_2019.0',
#     'SPONSORED_2020.0',
#     'SPONSORED_2021.0',
#     'SPONSORED_2022.0',
#     'SPONSORED_2023.0',
#     'SPONSORED_2024.0'
# ]
# df_cleaned = df_cleaned.loc[:, columns_list]

# # sponsorship year weights
# sponsorship_weights = {
#     'SPONSORED_2012.0': 0.0294,
#     'SPONSORED_2013.0': 0.0294,
#     'SPONSORED_2014.0': 0.0294,
#     'SPONSORED_2015.0': 0.0294,
#     'SPONSORED_2016.0': 0.0588,
#     'SPONSORED_2017.0': 0.0588,
#     'SPONSORED_2018.0': 0.0588,
#     'SPONSORED_2019.0': 0.0882,
#     'SPONSORED_2020.0': 0.0882,
#     'SPONSORED_2021.0': 0.0882,
#     'SPONSORED_2022.0': 0.1471,
#     'SPONSORED_2023.0': 0.1471,
#     'SPONSORED_2024.0': 0.1471
# }

# # combine the number of sponsored visas per year into one columns
# df_cleaned['SPONSORED'] = 0.0

# for col, weight in sponsorship_weights.items():
#     df_cleaned['SPONSORED'] += df_cleaned[col] * weight
    
# df_cleaned['SPONSORED'] = df_cleaned['SPONSORED'].round()

# # drop all the sponsored counts columns besides the combined value
# columns_to_drop = list(sponsorship_weights.keys())
# columns_to_drop.remove('SPONSORED_2023.0')
# df_cleaned.drop(columns=columns_to_drop, inplace=True)
#df2.drop(['WAITING_TIMERANGE'], axis=1, inplace=True)


# # Loading model and mapping pickle files
# rf_pickle = open("pages/11_30_23_rf_model_final.pkl", 'rb') 
# rf_model = pickle.load(rf_pickle) 
# rf_pickle.close() 

# SOC TITLE
# WORKSITE STATE
# SECTOR CODE
# EMPLOYEE COUNT CATEGORY
# COMPANY AGE CATEGORY

# Connect to the Google Sheet of SOC TITLEs
sheet_id = "1oLjpm4KLNj-tUN_Pnbrk_ihU7bNylJwG"
sheet_name = "Final"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
df = pd.read_csv(url, dtype=str).fillna("")

# Remove leading and trailing whitespaces in all columns
df["OCCUPATION"] = df["OCCUPATION"].str.strip()

df = df.drop_duplicates(subset=["OCCUPATION"])
st.write(df)

# # Show the dataframe (we'll delete this later)
# st.write(df)
#MAX_SELECTIONS = 3
    # Some code
with st.form(key='my_form'):
    st.subheader("Selections")
    # Use a text_input to get the keywords to filter the dataframe
    text_search = st.text_input("Search for SOC Title", help = "Type here to retrieve results in dropdown menu below.")
    # Filter the DataFrame based on the search input
    filtered_df = df[df["OCCUPATION"].str.contains(text_search, case=False, na=False)]

    # Display search results
    titleInfo = st.multiselect("Select SOC Title(s)", options=filtered_df["OCCUPATION"].tolist())

    # all codes
    codeOptions = ['11 - Agriculture, Forestry, Fishing and Hunting', 
                   '21 - Mining, Quarrying, and Oil and Gas Extraction', 
                   '22 - Utilities', 
                   '23 - Construction', 
                   '31 - Manufacturing (Food, Beverage, Tobacco, Apparel, Leather, Textiles)', 
                   '32 - Manufacturing (Paper, Printing, Petroleum, Coal, Chemicals, Plastics, Rubber, Nonmetallic)', 
                   '33 - Manufacturing (Primary Metals, Fabricated Metal, Machinery, Computer and Electronic Products, Electrical Equipment and Appliances, Transportations Equipment, Furniture, Miscellaneous Manufacturing)', 
                   '42 - Wholesale Trade',
                   '44 - Retail Trade (Automotive Sales and Services, Home Furnishing and Improvement, Food and Beverage, Health and Personal Care, Clothing and Accessories, Gasoline Stations)',
                   '45 - Retail Trade (Sporting Goods, Hobbies, Books, Department Stores, General Merchandise Stores, Florists, Office Supplies, Pet Supplies, Art Dealers, Various Specialty Stores)', 
                   '48 - Transportation and Warehousing (Air, Rail, Water, Truck, Transit, Pipeline, Scenic and Sightseeing Services, Transportation Support Activities)', 
                   '49 - Transportation and Warehousing (Federal Government-Operated Postal Services, Couriers, Messengers, Warehousing Storage-Related Services)',
                   '51 - Information',
                   '52 - Finance and Insurance',
                   '53 - Real Estate and Rental and Leasing',
                   '54 - Professional, Scientific, and Technical Services',
                   '55 - Management of Companies and Enterprises',
                   '56 - Administrative and Support and Waste Management and Remediation Services',
                   '61 - Educational Services',
                   '62 - Health Care and Social Assistance',
                   '71 - Arts, Entertainment, and Recreation',
                   '72 - Accommodation and Food Services',
                   '81 - Other Services (except Public Administration)',
                   '92 - Public Administration']
    
    # codes only in dataset
    codeOptions2 = ['11 - Agriculture, Forestry, Fishing and Hunting', 
                   '22 - Utilities', 
                   '31 - Manufacturing (Food, Beverage, Tobacco, Apparel, Leather, Textiles)', 
                   '32 - Manufacturing (Paper, Printing, Petroleum, Coal, Chemicals, Plastics, Rubber, Nonmetallic)', 
                   '33 - Manufacturing (Primary Metals, Fabricated Metal, Machinery, Computer and Electronic Products, Electrical Equipment and Appliances, Transportations Equipment, Furniture, Miscellaneous Manufacturing)', 
                   '42 - Wholesale Trade',
                   '44 - Retail Trade (Automotive Sales and Services, Home Furnishing and Improvement, Food and Beverage, Health and Personal Care, Clothing and Accessories, Gasoline Stations)',
                   '45 - Retail Trade (Sporting Goods, Hobbies, Books, Department Stores, General Merchandise Stores, Florists, Office Supplies, Pet Supplies, Art Dealers, Various Specialty Stores)', 
                   '48 - Transportation and Warehousing (Air, Rail, Water, Truck, Transit, Pipeline, Scenic and Sightseeing Services, Transportation Support Activities)', 
                   '51 - Information',
                   '52 - Finance and Insurance',
                   '53 - Real Estate and Rental and Leasing',
                   '54 - Professional, Scientific, and Technical Services',
                   '55 - Management of Companies and Enterprises',
                   '56 - Administrative and Support and Waste Management and Remediation Services',
                   '61 - Educational Services',
                   '62 - Health Care and Social Assistance',
                   '71 - Arts, Entertainment, and Recreation',
                   '72 - Accommodation and Food Services',
                   '81 - Other Services (except Public Administration)',
                   ]
    
    
# To get the selected value from the select box
    # codeInfo = st.selectbox('Select industry (up to 3)', codeOptions2, help="Select most appropriate Industry Code as found here https://www.census.gov/naics/?58967?yearbck=2022")
    codeInfo = st.multiselect('Select industry/industries', codeOptions2, help="Select most appropriate Industry Code as found here https://www.census.gov/naics/?58967?yearbck=2022")

    # have them choose the weights
    # 1 - 5

    # take the number assignemt, 2/5
    # add up all numbers and then divide number they chose over the total amount = 1
    # input "X" = Xi/SUM(all X) for all Xi
    # each value over the sum of all the values
    # x = 3
    # y = 5
    # z = 2
    # sum = 10
    # xnorm = 3/10
    # ynorm = 5/10
    # znorm = 2/10

    # ranking states
    # make a list

    # key to dictionary is column and value is a list
    # needs to be 

#     state_abbreviations = [
#     "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
#     "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
#     "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
#     "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
#     "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
# ]

    state_abbreviations = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "DISTRICT OF COLUMBIA",
    "FL", "FM", "GA", "GU", "GUAM", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA",
    "MA", "MD", "ME", "MH", "MI", "MN", "MO", "MP", "MS", "MT", "NC", "ND", "NE",
    "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "PR", "PUERTO RICO", "PW",
    "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VI", "VIRGIN ISLANDS", "VT", "WA",
    "WI", "WV", "WY"
]

    # ranking, enabling them to choose multiple options
    # Need to abbreviate these states
    # stateInfo = st.selectbox('U.S. Work State', state_abbreviations, 
    #                          help = "Select the U.S. state of primary worksite")
    
    stateInfo = st.multiselect('Select U.S. Work State(s)', state_abbreviations, help="Select most appropriate Industry Code as found here https://www.census.gov/naics/?58967?yearbck=2022")


    employeenum_categories = ['<50', '51-200', '201-500', '501-1000', 
                              '1001-5000','5001-10000', '10000+']
    #employeenumInfo = st.number_input('Number of Employees at Company', min_value = 0)
    # employeenumInfo = st.selectbox('Number of Employees at Company', employeenum_categories)

    employeenumInfo = st.multiselect('Select Company Size(s)', employeenum_categories)

    companyage_categories = ['Newly Established (0-2 years)','Early Stage (2-5 years)',
                                   'Growth Stage (5-10 years)','Mature Stage (10-20 years)',
                                   'Established Stage (20-50 years)', 'Legacy Stage (50+ years)']

    # companyageInfo = st.select_slider('Age Category', companyage_categories)
    companyageInfo = st.multiselect('Select Company Age(s)', companyage_categories)

    ### Ranking to apply weights to categories
    # Define the label for the slider
    st.subheader("Weights of Importance")
#     label = "Select importance level"

# # Define the custom labels for the slider endpoints
#     custom_labels = {1: "Not important at all", 5: "Most important"}

#     # Create the select slider
#     titleWeight = st.slider(
#         label,
#         min_value=1,
#         max_value=5,
#         value=(1, 5),  # Initial value
#         step=1,
#         format="%d",  # Format as integers
#         format_func=lambda x: custom_labels.get(x)
#     )

#     # Display the selected importance level
#     st.write("Selected importance level:", titleWeight)


    submit = st.form_submit_button('Submit',args=(1,
                    [titleInfo,codeInfo, stateInfo, employeenumInfo, companyageInfo]))

## Code for MAX_SELECTIONS, selections constraint
# if submit:
#     if len(codeInfo) > MAX_SELECTIONS:
#         st.error(f"Please select up to {MAX_SELECTIONS} industries only.")
#     if len(stateInfo) > MAX_SELECTIONS:
#         st.error(f"Please select up to {MAX_SELECTIONS} states only.")
#     if len(employeenumInfo) > MAX_SELECTIONS:
#         st.error(f"Please select up to {MAX_SELECTIONS} company size categories only.")
#     if len(companyageInfo) > MAX_SELECTIONS:
#         st.error(f"Please select up to {MAX_SELECTIONS} company age categories only.")
#     if len(codeInfo) <= MAX_SELECTIONS and len(stateInfo) <= MAX_SELECTIONS and len(employeenumInfo) <= MAX_SELECTIONS and len(companyageInfo) <= MAX_SELECTIONS:
#         # Sort the selected states and cities
#         selected_codes_sorted = sorted(codeInfo)
#         selected_states_sorted = sorted(stateInfo)
#         selected_empnum_sorted = sorted(employeenumInfo)
#         selected_compage_sorted = sorted(companyageInfo)
        
# user preferences
user_preferences = {
    'SOC_TITLE': titleInfo,
    'WORKSITE_STATE': stateInfo,
    'SECTOR_CODE': codeInfo,
    'EMPLOYEE_COUNT_CATEGORY': employeenumInfo,
    'COMPANY_AGE_CATEGORY': companyageInfo
} 

# column weights
weights = {
    'SPONSORED': .3,
    'SOC_TITLE': .2,
    'WORKSITE_STATE': .2,
    'PREVAILING_WAGE_ANNUAL': .1,
    'SECTOR_CODE': .1,
    'EMPLOYEE_COUNT_CATEGORY': .05,
    'COMPANY_AGE_CATEGORY': .05
}
# df3 = df2.copy()
# df3.loc[len(df3)] = [codeInfo, wagelevelInfo, wageamountInfo, stateInfo, countryInfo, employeenumInfo,  admiclassInfo,  jobeducationInfo, expInfo, expmonthsInfo, layoffInfo, educationInfo]
# # Create dummies for encode_df
# cat_var = ['NAICS_CODE', 'PW_LEVEL','WORK_STATE','COUNTRY_OF_CITIZENSHIP','CLASS_OF_ADMISSION','JOB_EDUCATION','EXPERIENCE','LAYOFF_IN_PAST_SIX_MONTHS','WORKER_EDUCATION']
# df3 = pd.get_dummies(df3, columns = cat_var)
# # Extract encoded user data
# user_encoded_df = df3.tail(1)

# df4 = pd.DataFrame(columns = ['NAICS_CODE', 'PW_LEVEL', 'PW_AMOUNT', 'WORK_STATE',
#          'COUNTRY_OF_CITIZENSHIP', 'EMPLOYER_NUM_EMPLOYEES',
#         'CLASS_OF_ADMISSION', 'JOB_EDUCATION', 'EXPERIENCE',
#          'EXPERIENCE_MONTHS', 'LAYOFF_IN_PAST_SIX_MONTHS', 'WORKER_EDUCATION'])
# df4.loc[-1]=[codeInfo, wagelevelInfo, wageamountInfo, stateInfo, countryInfo, employeenumInfo,  admiclassInfo,  jobeducationInfo, expInfo, expmonthsInfo, layoffInfo, educationInfo]

# st.subheader("Your Input")
# df4
# #df5 = pd.DataFrame(data = [codeInfo, wagelevelInfo, wageamountInfo, stateInfo, countryInfo, employeenumInfo,  admiclassInfo,  jobeducationInfo, expInfo, expmonthsInfo, layoffInfo, educationInfo])
# #pd.concat([df4, df5])
# st.subheader("Predicting Waiting Time")


#     # Using RF to predict() with encoded user data
# new_prediction_rf = rf_model.predict(user_encoded_df)
# new_prediction_prob_rf = rf_model.predict_proba(user_encoded_df).max()
# # Show the predicted cost range on the app
# st.write("Random Forest Prediction: {}".format(*new_prediction_rf))
# st.write("Prediction Probability: {:.0%}".format(new_prediction_prob_rf))

# # Showing additional items
# st.subheader("Prediction Performance")
# st.image("pages/NewFeatureImportance.svg")
# # tab1 = st.tabs(["Feature Importance"])
# # with tab1:
    
# # ONLY INCLUDE FEATURE IMPORTANCE
