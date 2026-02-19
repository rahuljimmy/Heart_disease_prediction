import streamlit as st
import pickle
import numpy as np
import pandas as pd

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Heart Disease Predictor 🫀")

age = st.slider('Age',20,80,40)

sex = st.selectbox('Gender',df['Sex'].unique())

chest_pain_type = st.selectbox('Chest Pain Type',df['ChestPainType'].unique())

resting_bp = st.slider('Resting Blood Pressure',80,200,140)

cholesterol = st.slider('Cholesterol',80,600,240)

fasting_bs = st.selectbox('Fasting Blood Sugar',df['FastingBS'].unique())

resting_ecg = st.selectbox('Resting ECG',df['RestingECG'].unique())

max_hr = st.slider('Maximum Heart Rate',60,210,150)

exercise_angina = st.selectbox('Exercise Angina',df['ExerciseAngina'].unique())

oldpeak = st.slider('Oldpeak',0.0,4.0,1.0)

st_slope = st.selectbox('ST Slope',df['ST_Slope'].unique())

if st.button('Predict Heart Disease'):

    query = pd.DataFrame([[age,sex,chest_pain_type,resting_bp,cholesterol,fasting_bs,
                           resting_ecg,max_hr,exercise_angina,oldpeak,st_slope]],
                columns=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
                          'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope'])
    
    prediction = pipe.predict(query)[0]

    if prediction == 1:
        st.error('🚨 The persion is likely to have heart disease')
    else:
        st.success('✅ The person is unlikely to have heart disease')
    
