import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

def create_model():
    
    data = pd.read_csv('heart1.csv')
    X = data.drop('output', axis=1)
    y = data['output']
    model = GaussianNB()
    model.fit(X, y)
    return model

def main():
    st.title('Heart Disease Prediction System')
    st.write('Enter the required information to predict heart disease risk')
    
    # Create input fields for all features
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', min_value=20, max_value=100, value=40)
        sex = st.selectbox('Sex', ['Male', 'Female'])
        cp = st.selectbox('Chest Pain Type', 
                         ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        trtbps = st.number_input('Resting Blood Pressure (mm Hg)', 
                                min_value=90, max_value=200, value=120)
        chol = st.number_input('Cholesterol (mg/dl)', 
                              min_value=100, max_value=600, value=200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        
    with col2:
        restecg = st.selectbox('Resting ECG Results', 
                              ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
        thalachh = st.number_input('Maximum Heart Rate', 
                                  min_value=60, max_value=220, value=150)
        exng = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        oldpeak = st.number_input('ST Depression Induced by Exercise', 
                                 min_value=0.0, max_value=6.0, value=0.0, step=0.1)
        slp = st.selectbox('Slope of Peak Exercise ST Segment', 
                          ['Upsloping', 'Flat', 'Downsloping'])
        caa = st.number_input('Number of Major Vessels Colored by Fluoroscopy', 
                             min_value=0, max_value=4, value=0)
        thall = st.selectbox('Thalassemia', 
                            ['Normal', 'Fixed Defect', 'Reversible Defect'])

    # Convert categorical inputs to numerical values
    sex = 1 if sex == 'Male' else 0
    cp = {'Typical Angina': 0, 'Atypical Angina': 1, 
          'Non-anginal Pain': 2, 'Asymptomatic': 3}[cp]
    fbs = 1 if fbs == 'Yes' else 0
    restecg = {'Normal': 0, 'ST-T Wave Abnormality': 1, 
               'Left Ventricular Hypertrophy': 2}[restecg]
    exng = 1 if exng == 'Yes' else 0
    slp = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}[slp]
    thall = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}[thall]

    # Create a button for prediction
    if st.button('Predict Heart Disease Risk'):
        # Prepare input data
        input_data = pd.DataFrame([[age, sex, cp, trtbps, chol, fbs, restecg,
                                  thalachh, exng, oldpeak, slp, caa, thall]],
                                columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg',
                                       'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall'])
        
        # Make prediction
        model = create_model()
        prediction = model.predict(input_data)[0]
        
        # Display result
        st.write('---')
        if prediction == 1:
            st.error('⚠️ High Risk: The model predicts a high likelihood of heart disease.')
            st.write('''
            Please consult with a healthcare professional for a thorough evaluation.
            This prediction is based on the provided information but should not be
            considered as a medical diagnosis.
            ''')
        else:
            st.success('✅ Low Risk: The model predicts a low likelihood of heart disease.')
            st.write('''
            While the prediction shows low risk, regular check-ups are still important
            for maintaining heart health. This is not a medical diagnosis.
            ''')

if __name__ == '__main__':
    main()