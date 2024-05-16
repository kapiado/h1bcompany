
# Import libraries
import pandas as pd             # Pandas
import streamlit as st          # Streamlit
#import matplotlib.pyplot as plt # Matplotlib
import seaborn as sns           # Seaborn

# Module to save and load Python objects to and from files
import pickle 
import joblib

# Package to implement Random Forest Model
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


import warnings
warnings.filterwarnings('ignore')

st.title('H-1B Visa Company Recommendation Form') 

st.write("This app recommends companies that are likely to sponsor an H-1B visa based on your user input. Use the form below to get started!")
         #uses *x* inputs to determine companies that would be a good fit for you to apply for. Use the form below to get started!")

# df2 = pd.read_csv("pages/11_30_23_Pred_Data_Final1.csv")

# df2.drop(['WAITING_TIMERANGE'], axis=1, inplace=True)


# # Loading model and mapping pickle files
# rf_pickle = open("pages/11_30_23_rf_model_final.pkl", 'rb') 
# rf_model = pickle.load(rf_pickle) 
# rf_pickle.close() 


    # Some code
with st.form(key='my_form'):
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
    

# To get the selected value from the select box
    codeInfo = st.selectbox('NAICS Code', codeOptions, help="Select most appropriate Industry Code as found here https://www.census.gov/naics/?58967?yearbck=2022")

    # Initialize variables for NAICS Codes
# Assign numerical values based on specific strings in the NAICS Code
    education_options = [
"High School", "Associate's", "Bachelor's", "Doctorate", 
"Master's", "None", "Other"
]
    # was job education -- how to address this since this is the education required by the job
    educationInfo = st.selectbox('Highest Completed Education Level', options=education_options) 
    
    wage_levels = ["Level I", "Level II", "Level III", "Level IV"]
    wagelevelInfo = st.selectbox('Prevailing Wage Level', wage_levels, help = "Select most appropriate prevailing wage level")

    wageamountInfo = st.number_input('Prevailing Wage Amount', min_value = 0)
    admiclasses = [
"A-3", "A1/A2", "B-1", "B-2", "C-1", "C-3", "CW-1", "D-1", "D-2", 
"E-1", "E-2", "E-3", "EWI", "F-1", "F-2", "G-1", "G-4", "G-5", 
"H-1A", "H-1B", "H-1B1", "H-2A", "H-2B", "H-3", "H-4", "I", 
"J-1", "J-2", "K-4", "L-1", "L-2", "M-1", "M-2", "N", "Not in USA", 
"O-1", "O-2", "O-3", "P-1", "P-3", "P-4", "Parolee", "Q", 
"R-1", "R-2", "T-1", "T-2", "TD", "TN", "TPS", "U-1", "V-2", 
"VWB", "VWT"
]

    admiclassInfo = st.selectbox('Class of Admission', admiclasses)
    
    countryInfo = st.selectbox('Country of Citizenship', options=[
"ARGENTINA", "AUSTRALIA", "BANGLADESH", "BELARUS", "BRAZIL", 
"BULGARIA", "CANADA", "CHILE", "CHINA", "COLOMBIA", "ECUADOR", 
"EGYPT", "FRANCE", "GERMANY", "GREECE", "HONG KONG", "INDIA", 
"INDONESIA", "IRAN", "IRELAND", "ISRAEL", "ITALY", "JAPAN", 
"JORDAN", "LEBANON", "MALAYSIA", "MEXICO", "NEPAL", "NETHERLANDS", 
"NIGERIA", "PAKISTAN", "PERU", "PHILIPPINES", "POLAND", "ROMANIA", 
"RUSSIA", "SINGAPORE", "SOUTH AFRICA", "SOUTH KOREA", "SPAIN", 
"SRI LANKA", "SWEDEN", "SYRIA", "TAIWAN", "THAILAND", "TURKEY", 
"UKRAINE", "UNITED KINGDOM", "VENEZUELA", "VIETNAM"
])

    stateInfo = st.selectbox('U.S. Work State',
                                                  [
    'ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO', 'CONNECTICUT', 'DELAWARE',
    'DISTRICT OF COLUMBIA', 'FLORIDA', 'GEORGIA', 'GUAM', 'HAWAII', 'IDAHO', 'ILLINOIS', 'INDIANA',
    'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MAINE', 'MARSHALL ISLANDS', 'MARYLAND', 'MASSACHUSETTS',
    'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI', 'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE',
    'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA', 'NORTH DAKOTA', 'NORTHERN MARIANA ISLANDS',
    'OHIO', 'OKLAHOMA', 'OREGON', 'PENNSYLVANIA', 'PUERTO RICO', 'RHODE ISLAND', 'SOUTH CAROLINA',
    'SOUTH DAKOTA', 'TENNESSEE', 'TEXAS', 'UTAH', 'VERMONT', 'VIRGIN ISLANDS', 'VIRGINIA', 'WASHINGTON',
    'WEST VIRGINIA', 'WISCONSIN', 'WYOMING'
], 
                             help = "Select the U.S. state of primary worksite")

    employeenumInfo = st.number_input('Number of Employees at Company', min_value = 0)


    jobeducation_options = [
"High School", "Associate's", "Bachelor's", "Doctorate", 
"Master's", "None", "Other"] 
    # was job education -- how to address this since this is the education required by the job
    jobeducationInfo = st.selectbox('Education Level Required by Job', options=jobeducation_options) 

    expInfo = st.radio('Do you have job/industry experience?', options=["Yes","No"])
    
    expmonthsInfo = st.number_input('Months of Experience', min_value = 0, help = "Input how many months of job experience you have")
    
    layoffInfo =  st.radio('Have you been affected from layoff(s) in the past six months?', options =["Yes","No"])


    submit = st.form_submit_button('Submit',args=(1,
                    [codeInfo, wagelevelInfo, wageamountInfo, stateInfo, countryInfo, employeenumInfo,  admiclassInfo,  jobeducationInfo, expInfo, expmonthsInfo, layoffInfo, educationInfo]))

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
