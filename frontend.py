import streamlit as st
import requests

API_BASE_URL = "https://health-ml-api-n6ie.onrender.com"

st.set_page_config(page_title="Health ML Predictor", layout="centered")

st.title("🩺 Health ML Prediction System")
st.write("Predict diabetes progression and breast cancer classification using ML.")

option = st.selectbox(
    "Choose Prediction Type",
    ["Diabetes Progression", "Breast Cancer"]
)

if option == "Diabetes Progression":
    st.subheader("Diabetes Prediction")

    features = []
    for i in range(10):
        val = st.number_input(f"Feature {i+1}", value=0.0)
        features.append(val)

    if st.button("Predict Diabetes"):
        payload = {"features": features}
        response = requests.post(f"{API_BASE_URL}/predict/diabetes", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction Score: {result['prediction']:.2f}")
        else:
            st.error("Error calling API")

elif option == "Breast Cancer":
    st.subheader("Breast Cancer Prediction")

    features = []
    for i in range(30):
        val = st.number_input(f"Feature {i+1}", value=0.0)
        features.append(val)

    if st.button("Predict Cancer"):
        payload = {"features": features}
        response = requests.post(f"{API_BASE_URL}/predict/cancer", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['class'].upper()}")
        else:
            st.error("Error calling API")
