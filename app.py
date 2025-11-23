import numpy as np
import streamlit as st
import joblib
import tensorflow as tf

# CONFIG
st.set_page_config(
    page_title="Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="centered"
)

# 12 fitur final
FEATURES = [
    'HighBP',
    'HighChol',
    'CholCheck',
    'BMI',
    'Smoker',
    'Stroke',
    'HeartDiseaseorAttack',
    'PhysActivity',
    'Fruits',
    'Veggies',
    'GenHlth',
    'Age'
]

# LOAD MODEL & SCALER
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler_diabetes.pkl")
    model = tf.keras.models.load_model("diabetes_dnn_binary.h5")  
    return scaler, model

scaler, model = load_artifacts()

# UI
st.title("🩺 Prediksi Risiko Diabetes (BRFSS + DNN)")
st.write("Masukkan data kesehatan responden untuk memprediksi risiko diabetes (0 = tidak, 1 = ya).")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    HighBP = st.selectbox(
        "Tekanan darah tinggi (HighBP)",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )

    HighChol = st.selectbox(
        "Kolesterol tinggi (HighChol)",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )

    CholCheck = st.selectbox(
        "Cek kolesterol dalam 5 tahun terakhir (CholCheck)",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )

    BMI = st.number_input(
        "BMI (Body Mass Index)",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

    Smoker = st.selectbox(
        "Pernah merokok ≥ 100 batang seumur hidup? (Smoker)",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )

    Stroke = st.selectbox(
        "Pernah stroke? (Stroke)",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )

with col2:
    HeartDiseaseorAttack = st.selectbox(
        "Penyakit jantung / serangan jantung (HeartDiseaseorAttack)",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )

    PhysActivity = st.selectbox(
        "Aktivitas fisik 30 hari terakhir di luar pekerjaan (PhysActivity)",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )

    Fruits = st.selectbox(
        "Konsumsi buah ≥1x/hari (Fruits)",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )

    Veggies = st.selectbox(
        "Konsumsi sayur ≥1x/hari (Veggies)",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )

    GenHlth = st.selectbox(
        "Penilaian kesehatan umum (GenHlth)",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{x} (1=sangat baik, 5=sangat buruk)"
    )

    Age = st.selectbox(
        "Kategori umur (Age, BRFSS kode 1–13)",
        options=list(range(1, 14)),
        format_func=lambda x: f"{x}"
    )

st.markdown("---")

if st.button("Prediksi Risiko Diabetes"):
    x_input = np.array([[
        HighBP,
        HighChol,
        CholCheck,
        BMI,
        Smoker,
        Stroke,
        HeartDiseaseorAttack,
        PhysActivity,
        Fruits,
        Veggies,
        GenHlth,
        Age
    ]], dtype=float)

    # Scaling
    x_scaled = scaler.transform(x_input)

    # Prediksi
    prob = float(model.predict(x_scaled)[0][0])
    label = 1 if prob >= 0.5 else 0

    st.subheader("Hasil Prediksi:")

    if label == 1:
        st.error(f"Model memprediksi: **BERISIKO DIABETES (1)**\n\nProbabilitas: **{prob:.3f}**")
    else:
        st.success(f"Model memprediksi: **TIDAK DIABETES (0)**\n\nProbabilitas: **{prob:.3f}**")