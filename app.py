import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import os

# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Lung Cancer Risk System - UAS DSP",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. LOAD MODEL MACHINE LEARNING
# ============================================================
MODEL_PATH = "model/logistic_regression_cancer.pkl"

@st.cache_resource
def load_prediction_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    return None

model = load_prediction_model()

# ============================================================
# 3. SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🫁 Navigation")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Home", "📊 Dashboard (Looker Studio)", "🔍 Prediksi Risiko"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Data Science Project**\n\n"
    "Sistem Prediksi Risiko Kanker Paru-Paru berbasis Machine Learning & Visualisasi Data."
)

# ============================================================
# 4. HALAMAN: HOME
# ============================================================
if menu == "🏠 Home":
    st.markdown("<h4 style='color: #888;'>OUR STRATEGIES THAT DRIVE</h4>", unsafe_allow_html=True)
    st.title("Impact and Growth 🚀")
    
    st.write(
        """
        Sistem prediksi risiko kanker paru-paru berbasis Machine Learning menggunakan model **Logistic Regression** 
        untuk membantu tenaga medis dan pengguna dalam pengambilan keputusan klinis secara lebih cepat, terukur, dan akurat.
        """
    )
    
    st.markdown("---")
    
    # Metrik Card
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Status Model", value="100% Loaded" if model else "Not Loaded")
    with col2:
        st.metric(label="Data Records", value="1,000+")
    with col3:
        st.metric(label="Features Analyzed", value="11 Fitur")
        
    st.markdown("---")
    st.subheader("Fitur Aplikasi")
    st.markdown(
        """
        - **Dashboard Analitik**: Menyajikan visualisasi interaktif langsung dari Google Looker Studio mengenai distribusi data dan faktor risiko.
        - **Prediksi Risiko**: Menggunakan 11 parameter kesehatan dan lingkungan untuk menentukan klasifikasi risiko kanker paru-paru (*Low*, *Medium*, *High*).
        """
    )

# ============================================================
# 5. HALAMAN: DASHBOARD (GOOGLE LOOKER STUDIO)
# ============================================================
elif menu == "📊 Dashboard (Looker Studio)":
    st.title("📊 Dashboard Analitik")
    st.write(
        "Visualisasi data hasil analisis dan performa model prediksi risiko kanker paru-paru "
        "yang terintegrasi langsung dari **Google Looker Studio**."
    )
    
    # URL Google Looker Studio dari template Flask Anda
    LOOKER_URL = "https://lookerstudio.google.com/embed/reporting/ecf0e212-64d5-4a3d-a4cc-a4b237bcc5d4/page/1RxiF"
    
    with st.container():
        components.iframe(
            src=LOOKER_URL,
            height=750,
            scrolling=True
        )

# ============================================================
# 6. HALAMAN: PREDIKSI RISIKO
# ============================================================
elif menu == "🔍 Prediksi Risiko":
    st.title("🔍 Prediksi Risiko Kanker Paru-Paru")
    st.write("Masukkan nilai parameter klinis dan lingkungan pasien di bawah, lalu klik **Prediksi Sekarang**.")
    
    if model is None:
        st.error(f"⚠️ Model `{MODEL_PATH}` tidak ditemukan. Pastikan file model sudah ada di repository.")
    else:
        with st.form("form_prediksi"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age (Usia)", min_value=1, max_value=120, value=35)
                air_pollution = st.slider("Air Pollution (1 - 9)", 1, 9, 2)
                passive_smoker = st.slider("Passive Smoker (1 - 9)", 1, 9, 2)
                occupational_hazards = st.slider("OccuPational Hazards (1 - 9)", 1, 9, 2)
                dust_allergy = st.slider("Dust Allergy (1 - 9)", 1, 9, 2)
                chronic_lung_disease = st.slider("chronic Lung Disease (1 - 9)", 1, 9, 2)
                
            with col2:
                shortness_breath = st.slider("Shortness of Breath (1 - 9)", 1, 9, 2)
                coughing_blood = st.slider("Coughing of Blood (1 - 9)", 1, 9, 2)
                dry_cough = st.slider("Dry Cough (1 - 9)", 1, 9, 2)
                snoring = st.slider("Snoring (1 - 9)", 1, 9, 2)
                swallowing_diff = st.slider("Swallowing Difficulty (1 - 9)", 1, 9, 2)
            
            st.markdown("<br>", unsafe_allow_html=True)
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
            
            df = pd.DataFrame([input_data])
            pred = model.predict(df)[0]
            
            st.markdown("---")
            st.subheader("Hasil Diagnosis Prediksi:")
            
            if pred == 1:
                st.success("### 🟢 Low Risk (Tingkat Risiko Rendah)")
                st.caption("Hasil analisis menunjukkan tingkat risiko rendah terhadap kanker paru-paru.")
            elif pred == 2:
                st.warning("### 🟡 Medium Risk (Tingkat Risiko Sedang)")
                st.caption("Hasil analisis menunjukkan tingkat risiko sedang. Disarankan pemeriksaan berkala.")
            else:
                st.error("### 🔴 High Risk (Tingkat Risiko Tinggi)")
                st.caption("Hasil analisis menunjukkan tingkat risiko tinggi. Disarankan untuk segera berkonsultasi dengan dokter spesialis.")
