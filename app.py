import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

scaler = joblib.load("scaler.pkl")

st.title("Churn prediction App")

st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction.")

st.divider()

age = st.number_input("Enter age",min_value = 10, max_value = 100, value = 10)

tenure = st.number_input("Enter tenure", min_value= 0, max_value= 130, value= 10)

monthlycharge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150)

gender = st.selectbox("Enter the Gender",["Male","Female"])

st.divider()

predictbutton = st.button("Predict!")

st.divider()

if predictbutton:

    gender_selection = 1 if gender == 'Female' else 0

    X = [age,gender_selection,tenure,monthlycharge]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.balloons()

    st.write(f"predicted: {predicted}")

else : 
    st.write("please enter the values and use predict button")