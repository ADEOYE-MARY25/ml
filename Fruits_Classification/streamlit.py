import streamlit as st
import numpy as np
import pickle

# ===============================
# Load Model & Scaler
# ===============================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ===============================
# App UI
# ===============================
st.set_page_config(page_title="Fruit Classification App", layout="centered")

st.title("🍎 Fruit Classification App")
st.write("Enter fruit features below to predict the fruit class.")

# ===============================
# Input Features
# (EDIT THESE BASED ON YOUR NOTEBOOK)
# ===============================
feature_1 = st.number_input("Feature 1 (e.g Weight)", min_value=0.0)
feature_2 = st.number_input("Feature 2 (e.g Height)", min_value=0.0)
feature_3 = st.number_input("Feature 3 (e.g Width)", min_value=0.0)
feature_4 = st.number_input("Feature 4 (e.g Color Intensity)", min_value=0.0)

# Collect inputs
input_data = np.array([[feature_1, feature_2, feature_3, feature_4]])

# ===============================
# Prediction
# ===============================
if st.button("Predict"):
    try:
        # Scale input
        scaled_input = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        st.success(f"✅ Predicted Class: **{prediction[0]}**")

        st.write("### Prediction Probabilities")
        st.write(prediction_proba)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
