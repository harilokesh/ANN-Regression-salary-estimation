import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder 
import pandas as pd
import pickle

# load the trained model
model = tf.keras.models.load_model('reg_model.h5')

#load Encoders
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('One_hot_encoder_geo.pkl','rb') as file:
    One_hot_encoder_geo = pickle.load(file)

with open('scaler_reg.pkl','rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Salary Estimation')

# User input:
geography = st.selectbox('Geography', One_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age',18,100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited',[0,1])
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',0,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is active member',[0,1])


Input_data = pd.DataFrame({
    'CreditScore':credit_score,
    'Geography':geography,
    'Gender':label_encoder_gender.transform([gender]),
    'Age':age,
    'Tenure':tenure,
    'Balance':balance,
    'NumOfProducts':num_of_products,
    'HasCrCard':has_cr_card,
    'IsActiveMember':is_active_member,
    'Exited':exited
})

#Encoding Geography col
geo_encode = One_hot_encoder_geo.transform([Input_data['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encode, columns=One_hot_encoder_geo.get_feature_names_out())

#Concatenating the geo data
input_df = pd.concat([Input_data.drop(columns='Geography'), geo_encoded_df], axis = 1)

# Scaling the data
input_sc = scaler.transform(input_df)


# prediction
prediction = model.predict(input_sc)[0][0]

#Churn Probability
st.write('Estimated salary:', prediction)