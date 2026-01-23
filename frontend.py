import streamlit as st
import requests

API_BASE_URL = "https://health-ml-api-n6ie.onrender.com"  # Change this

st.set_page_config(
    page_title="Health ML Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Health ML Prediction System")
st.markdown(
    """
    This application uses machine learning models to:
    - Predict **diabetes disease progression**
    - Classify **breast cancer tumors**

    ⚠️ *This is a demo system and not a medical diagnosis tool.*
    """
)

st.divider()

option = st.radio(
    "Select Prediction Type",
    ["Diabetes Progression", "Breast Cancer"],
    horizontal=True
)

st.divider()

if option == "Diabetes Progression":
    st.subheader("📈 Diabetes Progression Prediction")

    st.info("Enter 10 numeric clinical features")

    features = []
    cols = st.columns(2)

    for i in range(10):
        with cols[i % 2]:
            val = st.number_input(f"Feature {i+1}", value=0.0)
            features.append(val)

    if st.button("🔍 Predict Diabetes"):
        with st.spinner("Calling ML model..."):
            payload = {"features": features}
            response = requests.post(f"{API_BASE_URL}/predict/diabetes", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Score: **{result['prediction']:.2f}**")
        else:
            st.error("Error calling API")

elif option == "Breast Cancer":
    st.subheader("🧬 Breast Cancer Classification")

    st.info("Enter 30 numeric tumor features")

    features = []
    cols = st.columns(3)

    for i in range(30):
        with cols[i % 3]:
            val = st.number_input(f"Feature {i+1}", value=0.0)
            features.append(val)

    if st.button("🔍 Predict Cancer"):
        with st.spinner("Calling ML model..."):
            payload = {"features": features}
            response = requests.post(f"{API_BASE_URL}/predict/cancer", json=payload)

        if response.status_code == 200:
            result = response.json()
            if result["class"] == "malignant":
                st.error("⚠️ Prediction: **MALIGNANT**")
            else:
                st.success("✅ Prediction: **BENIGN**")
        else:
            st.error("Error calling API")

st.divider()

st.caption("Built with FastAPI, Docker, Streamlit, and Scikit-learn")
