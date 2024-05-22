
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
# import os
# current_dir = os.getcwd()
# st.write("Current working directory:", current_dir)
#df_original = pd.read_csv('/mount/src/h1bcompany/pages/merged_data_5_16_24.csv')
df_original = pd.read_csv('/mount/src/h1bcompany/pages/merged_data_5_16_24.csv')

# create copy of compete dataframe to be manipulated
df_cleaned = df_original.copy()
df_cleaned.info()

# Select only the relevant columns
columns_list = [
    'EMPLOYER_NAME_CLEAN',
    'EMPLOYER_NAME',
    'SOC_TITLE',
    'WORKSITE_STATE',
    'PREVAILING_WAGE_ANNUAL',
    'SECTOR_CODE',
    'EMPLOYEE_COUNT_CATEGORY',
    'COMPANY_AGE_CATEGORY',
    'COMPANY_LINK',
    'SPONSORED_2012.0',
    'SPONSORED_2013.0',
    'SPONSORED_2014.0',
    'SPONSORED_2015.0',
    'SPONSORED_2016.0',
    'SPONSORED_2017.0',
    'SPONSORED_2018.0',
    'SPONSORED_2019.0',
    'SPONSORED_2020.0',
    'SPONSORED_2021.0',
    'SPONSORED_2022.0',
    'SPONSORED_2023.0',
    'SPONSORED_2024.0'
]
df_cleaned = df_cleaned.loc[:, columns_list]

# sponsorship year weights
sponsorship_weights = {
    'SPONSORED_2012.0': 0.0294,
    'SPONSORED_2013.0': 0.0294,
    'SPONSORED_2014.0': 0.0294,
    'SPONSORED_2015.0': 0.0294,
    'SPONSORED_2016.0': 0.0588,
    'SPONSORED_2017.0': 0.0588,
    'SPONSORED_2018.0': 0.0588,
    'SPONSORED_2019.0': 0.0882,
    'SPONSORED_2020.0': 0.0882,
    'SPONSORED_2021.0': 0.0882,
    'SPONSORED_2022.0': 0.1471,
    'SPONSORED_2023.0': 0.1471,
    'SPONSORED_2024.0': 0.1471
}

# combine the number of sponsored visas per year into one columns
df_cleaned['SPONSORED'] = 0.0

for col, weight in sponsorship_weights.items():
    df_cleaned['SPONSORED'] += df_cleaned[col] * weight
    
df_cleaned['SPONSORED'] = df_cleaned['SPONSORED'].round()

# drop all the sponsored counts columns besides the combined value
columns_to_drop = list(sponsorship_weights.keys())
columns_to_drop.remove('SPONSORED_2023.0')
df_cleaned.drop(columns=columns_to_drop, inplace=True)
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

# # Show the dataframe (we'll delete this later)
# st.write(df)
#MAX_SELECTIONS = 3
    # Some code
with st.form(key='my_form'):
    st.subheader("Selections")
    st.write(df)
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

    st.subheader("Weights of Importance")
    body1 = st.empty()

    # Define the list of inputs
    inputs_list = ["Title", "Code", "State", "Employee Number", "Company Age"]

    # Define the importance scale
    importance_scale = "1 = Not important at all<br>2 = Less important<br>3 = Neutral<br>4 = Important<br>5 = Most important"

    # Define the text to display
    text = f"""
    Please indicate your level of importance for each of the above inputs.<br>
    <b>How important is each input in finding a job?</b><br>
    <i>Please refer to this scale:</i><br>
    {importance_scale}
    """

    # Display the text with a bulleted list
    body1.markdown(text, unsafe_allow_html=True)

    titleWeight = st.slider(
        'How important is **your role** (SOC Title) when looking for a job?',
        min_value=1,
        max_value=5,
        value=3,  # Default value
        step=1,
        format="%d"  # Format the slider to show as integer
    )

    codeWeight = st.slider(
        'How important is **the industry** when looking for a job?',
        min_value=1,
        max_value=5,
        value=3,  # Default value
        step=1,
        format="%d"  # Format the slider to show as integer
    )

    stateWeight = st.slider(
        'How important is **the state you work in** when looking for a job?',
        min_value=1,
        max_value=5,
        value=3,  # Default value
        step=1,
        format="%d"  # Format the slider to show as integer
    )

    employeenumWeight = st.slider(
        'How important is **the company size** (number of employees at the company) when looking for a job?',
        min_value=1,
        max_value=5,
        value=3,  # Default value
        step=1,
        format="%d"  # Format the slider to show as integer
    )

    companyageWeight = st.slider(
        'How important is **the company age** (how long company has been active) when looking for a job?',
        min_value=1,
        max_value=5,
        value=3,  # Default value
        step=1,
        format="%d"  # Format the slider to show as integer
    )

    wageWeight = st.slider(
        'How important is **the prevailing wage/salary** when looking for a job?',
        min_value=1,
        max_value=5,
        value=3,  # Default value
        step=1,
        format="%d"  # Format the slider to show as integer
    )

    sponsoredyearWeight = st.slider(
        'Lastly, this form includes data from 2013-2024. How important is **the sponsored year** (when the company submitted H-1B applications) when looking for a job?',
        min_value=1,
        max_value=5,
        value=3,  # Default value
        step=1,
        format="%d"  # Format the slider to show as integer
    )
    submit = st.form_submit_button('Submit',args=(1,
                    [titleInfo,codeInfo, stateInfo, employeenumInfo, companyageInfo,
                     titleWeight,codeWeight,stateWeight,employeenumWeight,companyageWeight,
                     wageWeight,sponsoredyearWeight]))

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

total_sum = titleWeight + codeWeight + stateWeight + employeenumWeight + companyageWeight + sponsoredyearWeight
# column weights
weights = {
    'SPONSORED': sponsoredyearWeight / total_sum,
    'SOC_TITLE': titleWeight / total_sum,
    'WORKSITE_STATE': stateWeight / total_sum,
    'PREVAILING_WAGE_ANNUAL': wageWeight / total_sum,
    'SECTOR_CODE': codeWeight / total_sum,
    'EMPLOYEE_COUNT_CATEGORY': employeenumWeight / total_sum,
    'COMPANY_AGE_CATEGORY': companyageWeight / total_sum
}

### Filter the dataset

# create copy to be filtered
df = df_cleaned.copy()

# filter by the user preferences
mask = pd.Series([True] * len(df), index=df.index)  # Initialize a mask of True values
for column, values in user_preferences.items():
    mask &= df[column].isin(values)

# Apply the combined mask to the DataFrame
df1 = df[mask]
df1 = df1.copy()

# Remove all columns that will not contribute to the rankings
df1.drop(columns=['EMPLOYER_NAME', 'EMPLOYER_NAME_CLEAN', 'SPONSORED_2023.0', 'COMPANY_LINK'], inplace=True)

# Apply preferences to the columns, put assigned values into new columns
# as a safeguard against errors I had anything not included in the preferences be automatically set to 0
for column, preferences in user_preferences.items():
    preference_mapping = {name: len(preferences) - rank for rank, name in enumerate(preferences)}
    df1[column] = df1[column].map(preference_mapping).fillna(0).astype(int)

### Apply TOPSIS
# calculate denominator value for normalization
numeric_cols = df1.select_dtypes(include=[np.number])
numeric_cols.fillna(0, inplace=True)

col_1_squared = numeric_cols['SOC_TITLE'].apply(np.square)
col_1_sum_squared = np.sqrt(col_1_squared.sum())

col_2_squared = numeric_cols['WORKSITE_STATE'].apply(np.square)
col_2_sum_squared = np.sqrt(col_2_squared.sum())

col_3_squared = numeric_cols['PREVAILING_WAGE_ANNUAL'].apply(np.square)
col_3_sum_squared = np.sqrt(col_3_squared.sum())

col_4_squared = numeric_cols['SECTOR_CODE'].apply(np.square)
col_4_sum_squared = np.sqrt(col_4_squared.sum())

col_5_squared = numeric_cols['EMPLOYEE_COUNT_CATEGORY'].apply(np.square)
col_5_sum_squared = np.sqrt(col_5_squared.sum())

col_6_squared = numeric_cols['COMPANY_AGE_CATEGORY'].apply(np.square)
col_6_sum_squared = np.sqrt(col_6_squared.sum())

col_7_squared = numeric_cols['SPONSORED'].apply(np.square)
col_7_sum_squared = np.sqrt(col_7_squared.sum())

# normalize the columns
df1['SOC_TITLE'] /= col_1_sum_squared
df1['WORKSITE_STATE'] /= col_2_sum_squared
df1['PREVAILING_WAGE_ANNUAL'] /= col_3_sum_squared
df1['SECTOR_CODE'] /= col_4_sum_squared
df1['EMPLOYEE_COUNT_CATEGORY'] /= col_5_sum_squared
df1['COMPANY_AGE_CATEGORY'] /= col_6_sum_squared
df1['SPONSORED'] /= col_7_sum_squared

# apply the weights
weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['weight']) #turning the weights into a df
df1 = df1.multiply(weights_df['weight'], axis=1)

# find positive and negative ideal solutions
ideals = df1.agg(['max','min'])

# calculate the separation measures for positive S
pos_diff_sq = (df1 - ideals.iloc[0])**2
df1['Si+'] = np.sqrt(pos_diff_sq.sum(axis=1))

# calculate the separation measures for negative S
neg_diff_sq = (df1 - ideals.iloc[1])**2
df1['Si-'] = np.sqrt(neg_diff_sq.sum(axis=1))

# calculate relative closeness
df1['S_i'] = df1['Si-'] / (df1['Si+'] + df1['Si-'])

# sort index by highest summed value
sorted_indexes = df1.sort_values(by='S_i', ascending=False).index

# Obtain the best companies for the user
df = df.reindex(sorted_indexes)

# add a column with the count of jobs by the company
df['JOB_COUNT'] = df.groupby('EMPLOYER_NAME_CLEAN')['EMPLOYER_NAME_CLEAN'].transform('count')

### Condense worksite state and SOC title

# group by company to get all worksite states
grouped_ws = df.groupby('EMPLOYER_NAME_CLEAN')['WORKSITE_STATE'].agg(list).reset_index()
# grouped_ws['OTHER_WORKSITE_STATE'] = grouped_ws.apply(lambda row: 
#                                                 list(set(row['WORKSITE_STATE']) - set([row['WORKSITE_STATE'][0]])), 
#                                                 axis=1)
grouped_ws['OTHER_WORKSITE_STATE'] = ''
for index, row in grouped_ws.iterrows():
    row_values = row['WORKSITE_STATE']
    row_values.remove(row_values[0])  # Remove the first value
    grouped_ws.at[index, 'OTHER_WORKSITE_STATE'] = row_values
result = pd.merge(df, grouped_ws, on='EMPLOYER_NAME_CLEAN', how='left')

# fix formating 
result.rename(columns={'WORKSITE_STATE_x': 'WORKSITE_STATE'}, inplace=True)
result.drop(columns=['WORKSITE_STATE_y'], inplace=True)




# group by company to get all SOC titles
grouped_soc = result.groupby('EMPLOYER_NAME_CLEAN')['SOC_TITLE'].agg(list).reset_index()
# grouped_soc['OTHER_SOC_TITLES'] = grouped_soc.apply(lambda row: 
#                                                     list(set(row['SOC_TITLE']) - set([row['SOC_TITLE'][0]])), 
#                                                     axis=1)
for index, row in grouped_soc.iterrows():
    row_values = row['SOC_TITLE']
    row_values.remove(row_values[0])  # Remove the first value
    grouped_soc.at[index, 'OTHER_SOC_TITLES'] = row_values
result = pd.merge(result, grouped_soc, on='EMPLOYER_NAME_CLEAN', how='left')

# fix formating 
result.rename(columns={'SOC_TITLE_x': 'SOC_TITLE'}, inplace=True)
result.drop(columns=['SOC_TITLE_y'], inplace=True)

# only display unique outputs
result.drop_duplicates(subset=['EMPLOYER_NAME_CLEAN'], keep='first', inplace=True)
result.drop(columns = 'SPONSORED', inplace = True)
st.write(result)