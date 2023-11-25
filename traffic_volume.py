# App to predict the traffic volume
# Using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle
from calendar import month_abbr, day_name
import warnings
warnings.filterwarnings('ignore')

st.title('Predicting Traffic Volume: A Machine Learning App') 

# Display the image
st.image('traffic_image.gif', width = 650)

st.write("This machine learning app uses multiple inputs to predict traffic volume.\n Use the following form to get started!") 

# Reading the pickle files that we created before 
# Decision Tree
dt_pickle = open('dt_traffic.pickle', 'rb') 
dt_model = pickle.load(dt_pickle) 
dt_pickle.close()

# Random Forest
rf_pickle = open('rf_traffic.pickle', 'rb') 
rf_model = pickle.load(rf_pickle) 
rf_pickle.close()

# Adaboost
ada_pickle = open('ada_traffic.pickle', 'rb') 
ada_model = pickle.load(ada_pickle) 
ada_pickle.close()

# XGboost
xg_pickle = open('xg_traffic.pickle', 'rb') 
xg_model = pickle.load(xg_pickle) 
xg_pickle.close()

# Loading default dataset
traffic_df = pd.read_csv('Traffic_Volume.csv')

# Convert date_time column to datetime object
traffic_df['date_time'] = traffic_df['date_time'].apply(pd.to_datetime)

# Extract month, weekday, and hour
traffic_df['month'] = traffic_df['date_time'].dt.month
traffic_df['weekday'] = traffic_df['date_time'].dt.day_name()
traffic_df['hour'] = traffic_df['date_time'].dt.hour

# Dropping weather description and date_time column
traffic_df.drop(columns=['weather_description', 'date_time'], inplace = True)

# Read in Model Metric Comparison
metric_df = pd.read_csv('model_comparison.csv', index_col = 0)
# function to color rows (best & worst models)
def highlight(row):
    if row['ML Model'] == 'XG Boost':
        return ['background-color: lime'] * len(row)
    elif row['ML Model'] == 'Ada Boost':
        return ['background-color: orange'] * len(row)
    else:
        return ['background-color:'] * len(row)
# color rows
styled_df = metric_df.style.apply(highlight, axis=1)
# hide index
styled_df = styled_df.hide(axis="index")

with st.form('user_inputs'): 
    # For numerical variables, using number_input
    # NOTE: Make sure that variable names are same as that of training dataset
    # temp	rain_1h	snow_1h	clouds_all
    temp = st.number_input('Average temperature in Kelvin') 
    rain_1h = st.number_input('Amount of rain that occured in the hour (in mm)') 
    snow_1h = st.number_input('Amount of snow that occured in the hour (in mm)') 
    clouds_all = st.number_input('Percentage of cloud cover', min_value = 0)
    clouds_all = int(clouds_all)    # variable is an int in the original dataset

    # For categorical variables, using selectbox 
    # holiday weather_main	month	weekday	hour
    holiday = st.selectbox('Is today a designated holiday? If so, which one?', options = traffic_df['holiday'].value_counts().index.to_list())
    weather_main = st.selectbox('What is the weather currently?', options = traffic_df['weather_main'].value_counts().index.to_list())
    month = st.selectbox('Which month is it?', options = [month_abbr[i] for i in range(1, 13)])
    # convert month input to number
    for k, v in enumerate(month_abbr):
        if v == month:
            month = k
            break
    weekday = st.selectbox('Which day of the week is it?', options = [day_name[i] for i in range(0, 7)])
    hour = st.selectbox('What hour of day is it?', options = list(range(0, 24)))

    ml_model = st.selectbox('Which Machine Learning model do you want to use for prediction?', 
                            options = ['Decision Tree', 'Random Forest', 'Ada Boost', 'XG Boost'])
    
    # display model comparison
    st.write("The ML models exhibited the following cross-validated predictive performance on test data.")
    st.dataframe(styled_df)

    # Submit
    st.form_submit_button() 

# df = pd.DataFrame(list(zip([holiday], [temp], [rain_1h], [snow_1h], [clouds_all], [weather_main], [month], [weekday], [hour])),
#                   columns = ['holiday', 'temp','rain_1h', 'snow_1h','clouds_all','weather_main', 'month', 'weekday', 'hour'])
# st.write(df.dtypes)
# # convert clouds to int

# Create df to encode the input
encode_df = traffic_df.copy()
encode_df = encode_df.drop(columns = ['traffic_volume'])

# Combine the list of user data as a row to default_df
encode_df.loc[len(encode_df)] = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, month, weekday, hour]
# Create dummies for encode_df
cat_var = ['holiday', 'weather_main', 'month', 'weekday', 'hour']
encode_dummy_df = pd.get_dummies(encode_df, columns = cat_var)
# Extract encoded user data
user_encoded_df = encode_dummy_df.tail(1)

st.subheader("Predicting Traffic with Inputs and Model Selection")

if ml_model == 'Decision Tree':
    # Using DT to predict() with encoded user data
    new_prediction_dt = dt_model.predict(user_encoded_df)

    # Show the predicted cost range on the app
    st.write("Decision Tree Traffic Prediction: {}".format(round(*new_prediction_dt)))

    # Showing additional items
    st.subheader("Plot of Feature Importance")
    st.image('dt_feature_imp.svg')

elif ml_model == 'Random Forest':
    # Using RF to predict() with encoded user data
    new_prediction_rf = rf_model.predict(user_encoded_df)

    # Show the predicted cost range on the app
    st.write("Random Forest Traffic Prediction: {}".format(round(*new_prediction_rf)))

    # Showing additional items
    st.subheader("Plot of Feature Importance")
    st.image('rf_feature_imp.svg')

elif ml_model == 'Ada Boost':
    # Using AdaBoost to predict() with encoded user data
    new_prediction_ada = ada_model.predict(user_encoded_df)

    # Show the predicted cost range on the app
    st.write("Ada Boost Traffic Prediction: {}".format(round(*new_prediction_ada)))

    # Showing additional items
    st.subheader("Plot of Feature Importance")
    st.image('ada_feature_imp.svg')

else:  # if XG Boost
    # Using XGBoost to predict() with encoded user data
    new_prediction_xg = xg_model.predict(user_encoded_df)

    # Show the predicted cost range on the app
    st.write("XG Boost Traffic Prediction: {}".format(round(*new_prediction_xg)))

    # Showing additional items
    st.subheader("Plot of Feature Importance")
    st.image('xg_feature_imp.svg')

