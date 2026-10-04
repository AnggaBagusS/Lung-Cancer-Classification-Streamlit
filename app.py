import streamlit as st
import pandas as pd
import joblib
import os

# Konfigurasi Halaman
st.set_page_config(
    page_title="Lung Cancer Risk Prediction",
    page_icon="🫁",
    layout="centered"
)

# Load model
MODEL_PATH = "model/logistic_regression_cancer.pkl"

@st.cache_resource
def load_prediction_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_prediction_model()

# Tampilan UI
st.title("🫁 Prediksi Risiko Kanker Paru-Paru")
st.write("Masukkan data klinis dan faktor risiko pasien di bawah ini:")

if model is None:
    st.error(f"File model tidak ditemukan di `{MODEL_PATH}`. Pastikan file model ikut ter-upload.")
else:
    with st.form("prediction_form"):
        st.subheader("Data Pasien & Faktor Lingkungan")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age (Usia Pasien)", min_value=1, max_value=120, value=35)
            air_pollution = st.slider("Air Pollution (1 - 9)", 1, 9, 2)
            passive_smoker = st.slider("Passive Smoker (1 - 9)", 1, 9, 2)
            occupational_hazards = st.slider("Occupational Hazards (1 - 9)", 1, 9, 2)
            dust_allergy = st.slider("Dust Allergy (1 - 9)", 1, 9, 2)
            chronic_lung_disease = st.slider("Chronic Lung Disease (1 - 9)", 1, 9, 2)

        with col2:
            shortness_breath = st.slider("Shortness of Breath (1 - 9)", 1, 9, 2)
            coughing_blood = st.slider("Coughing of Blood (1 - 9)", 1, 9, 2)
            dry_cough = st.slider("Dry Cough (1 - 9)", 1, 9, 2)
            snoring = st.slider("Snoring (1 - 9)", 1, 9, 2)
            swallowing_diff = st.slider("Swallowing Difficulty (1 - 9)", 1, 9, 2)

        submit_btn = st.form_submit_button("🔍 Prediksi Sekarang", use_container_width=True)

    if submit_btn:
        input_data = {
            "Age": age,
            "Coughing of Blood": coughing_blood,
            "Dust Allergy": dust_allergy,
            "Passive Smoker": passive_smoker,
            "OccuPational Hazards": occupational_hazards,
            "Air Pollution": air_pollution,
            "chronic Lung Disease": chronic_lung_disease,
            "Shortness of Breath": shortness_breath,
            "Dry Cough": dry_cough,
            "Snoring": snoring,
            "Swallowing Difficulty": swallowing_diff
        }

        # Jalankan Prediksi
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]

        # Tampilkan Hasil
        st.write("---")
        st.subheader("Hasil Prediksi:")
        if prediction == 1:
            st.success("🟢 **Low Risk (Risiko Rendah)**")
        elif prediction == 2:
            st.warning("🟡 **Medium Risk (Risiko Sedang)**")
        else:
            st.error("🔴 **High Risk (Risiko Tinggi)**")
