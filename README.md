## 🩺 Production-Grade Health ML System

A full-stack, production-ready machine learning system that demonstrates how to take ML models from training to real-world deployment with APIs, Docker, CI/CD, monitoring, and a live web frontend.

### 🔗 Live Links

* **API (Swagger Docs):** *[HealthML_API](https://health-ml-api-n6ie.onrender.com)*
* **Web App (Streamlit):** *[HealthML_StreamlitUI](https://health-ml-api-o7tuaccsdlh77g2qva2ebs.streamlit.app/)*

### 🚀 What This Project Shows

* End-to-end ML lifecycle: training → serving → deployment
* Multi-model inference (regression + classification)
* Real MLOps practices (Docker, CI/CD, logging, monitoring)
* Full-stack integration with a web UI

### 🧠 Models

* **Diabetes Progression Prediction** (Regression)
* **Breast Cancer Classification** (Malignant vs Benign)

### 🏗 Architecture

```
User
 ↓
Streamlit Frontend
 ↓
FastAPI Backend
 ↓
Dockerized ML Models
```

### 🧰 Tech Stack

* Python, Scikit-learn, NumPy
* FastAPI, Pydantic
* Docker
* Render (API deployment)
* Streamlit Cloud (Frontend)

### 📦 API Endpoints

* GET `/health`
* POST `/predict/diabetes`
* POST `/predict/cancer`
* GET `/version`

### ⚠️ Disclaimer

This project is for educational purposes only and not intended for medical diagnosis.
