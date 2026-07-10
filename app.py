import streamlit as st
import pandas as pd
import numpy as np
import joblib
model = joblib.load("cardio_logreg_model.pkl")
scaler = joblib.load("cardio_scaler.pkl")
feature_order = joblib.load("cardio_feature_order.pkl")
st.set_page_config(page_title="Cardiovascular Disease Risk Predictor", page_icon="❤️")
st.title("❤️ Cardiovascular Disease Risk Predictor")
st.write(
    "This app uses a **Logistic Regression** model trained on the cardio_train dataset "
    "to estimate the risk of cardiovascular disease based on your health data."
)
st.header("Enter your details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)
    weight = st.number_input("Weight (kg)", min_value=20, max_value=250, value=70)
    ap_hi = st.number_input("Systolic blood pressure (ap_hi)", min_value=80, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic blood pressure (ap_lo)", min_value=40, max_value=150, value=80)
with col2:
    cholesterol = st.selectbox(
        "Cholesterol level", options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x]
    )
    gluc = st.selectbox(
        "Glucose level", options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x]
    )
    smoke = st.radio("Do you smoke?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    alco = st.radio("Do you consume alcohol?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    active = st.radio("Are you physically active?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
if st.button("Predict"):
    if ap_hi <= ap_lo:
        st.error("Systolic (ap_hi) must be greater than diastolic (ap_lo). Please correct your inputs.")
    else:
        bmi = weight / ((height / 100) ** 2)

        row = {
            "age": age,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "smoke": smoke,
            "alco": alco,
            "active": active,
            "cholesterol_2": 1 if cholesterol == 2 else 0,
            "cholesterol_3": 1 if cholesterol == 3 else 0,
            "gluc_2": 1 if gluc == 2 else 0,
            "gluc_3": 1 if gluc == 3 else 0,
            "BMI": bmi,
        }
        input_df = pd.DataFrame([row])[feature_order]
        continuous_cols = ["age", "ap_hi", "ap_lo", "BMI"]
        input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("Result")
        st.write(f"**BMI:** {bmi:.2f}")

        if pred == 1:
            st.error(f"⚠️ High risk of cardiovascular disease (probability: {prob:.1%})")
        else:
            st.success(f"✅ Low risk of cardiovascular disease (probability: {prob:.1%})")

        st.caption(
            "This is a statistical estimate based on a logistic regression model "
            "and is not a medical diagnosis. Please consult a doctor for medical advice."
        )