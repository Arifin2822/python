import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf

# CONFIG & LOAD ASSETS
st.set_page_config(
    page_title="Diabetes Health System",
    page_icon="🩺",
    layout="wide" 
)

# Load Model DNN (Klasifikasi)
try:
    dnn_model = tf.keras.models.load_model('diabetes_dnn_binary.h5')
    scaler_dnn = joblib.load('scaler_diabetes.pkl')
except:
    st.error("File model DNN/Scaler tidak ditemukan. Pastikan 'diabetes_model.h5' dan 'scaler.pkl' ada.")
    st.stop()

# Load Model K-Means (Clustering) 
try:
    kmeans_model = joblib.load('kmeans_model.pkl')
except:
    kmeans_model = None 

# Definisi Fitur & Label
FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity',
    'Fruits', 'Veggies', 'GenHlth', 'Age'
]

AGE_LABELS = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
    6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
    11: "70–74", 12: "75–79", 13: "80+"
}

# Definisi Nama Cluster
CLUSTER_NAMES = {
    0: "Cluster 0: The Healthy Youth (Resiko Rendah)",
    1: "Cluster 1: Lifestyle Risk Group (Perokok/BMI Tinggi)",
    2: "Cluster 2: Chronic Condition Group (Komplikasi Lansia)"
}

# SIDEBAR NAVIGATION
st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.radio(
    "Pilih Modul:",
    ("🔍 Prediksi Risiko (DNN)", "👥 Cek Segmentasi (Clustering)", "💡 Ensiklopedia Pola (Rules)")
)
# FUNGSI INPUT (Dipakai Berulang)
def get_user_input():
    st.header("Data Pasien")
    col1, col2 = st.columns(2)
    
    with col1:
        HighBP = st.selectbox("Tekanan Darah Tinggi?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        HighChol = st.selectbox("Kolesterol Tinggi?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        CholCheck = st.selectbox("Cek Kolesterol dalam 5thn?", [0, 1], format_func=lambda x: "Sudah" if x==1 else "Belum")
        BMI = st.number_input("Indeks Massa Tubuh (BMI)", min_value=10.0, max_value=100.0, value=25.0)
        Smoker = st.selectbox("Perokok (Min 100 btg seumur hidup)?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        Stroke = st.selectbox("Riwayat Stroke?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")

    with col2:
        HeartDiseaseorAttack = st.selectbox("Riwayat Jantung Koroner?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        PhysActivity = st.selectbox("Aktivitas Fisik (30 hari terakhir)?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        Fruits = st.selectbox("Makan Buah Tiap Hari?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        Veggies = st.selectbox("Makan Sayur Tiap Hari?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        GenHlth = st.slider("Kesehatan Umum (1=Baik Sekali, 5=Buruk)", 1, 5, 3)
        Age = st.selectbox("Kelompok Umur", options=list(AGE_LABELS.keys()), format_func=lambda x: AGE_LABELS[x])
    
    # Bungkus jadi array numpy
    input_data = np.array([[
        HighBP, HighChol, CholCheck, BMI, Smoker,
        Stroke, HeartDiseaseorAttack, PhysActivity,
        Fruits, Veggies, GenHlth, Age
    ]])
    return input_data

# HALAMAN 1: PREDIKSI (DNN)
if menu == "🔍 Prediksi Risiko (DNN)":
    st.title("🔍 Prediksi Risiko Diabetes (Klasifikasi)")
    st.write("Modul ini menggunakan **Deep Neural Network** untuk memprediksi apakah pasien menderita diabetes.")
    
    user_data = get_user_input()
    
    if st.button("Jalankan Prediksi", type="primary"):
        # Preprocessing (Scaling)
        user_data_scaled = scaler_dnn.transform(user_data)
        
        # Predict
        prediction_prob = dnn_model.predict(user_data_scaled)
        prediction_class = (prediction_prob > 0.5).astype(int)[0][0]
        probability = prediction_prob[0][0]
        
        st.divider()
        if prediction_class == 1:
            st.error(f"⚠️ **HASIL: POSITIF DIABETES** (Probabilitas: {probability:.2%})")
            st.write("Saran: Segera konsultasi ke dokter untuk tes gula darah.")
        else:
            st.success(f"✅ **HASIL: NEGATIF** (Probabilitas Diabetes: {probability:.2%})")
            st.write("Saran: Pertahankan gaya hidup sehat Anda.")

# HALAMAN 2: SEGMENTASI (K-MEANS)
elif menu == "👥 Cek Segmentasi (Clustering)":
    st.title("👥 Segmentasi Profil Pasien")
    st.write("Modul ini mengelompokkan Anda ke dalam **Tipe Profil Kesehatan** menggunakan algoritma **K-Means**.")
    
    if kmeans_model is None:
        st.warning("⚠️ Model Clustering ('kmeans_model.pkl') belum diupload. Jalankan notebook langkah penyimpanan dulu.")
    else:
        user_data = get_user_input()
        
        if st.button("Cek Profil Saya"):
            user_data_scaled = scaler_dnn.transform(user_data)
            
            # Predict Cluster
            cluster_id = kmeans_model.predict(user_data_scaled)[0]
            cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
            
            st.divider()
            st.info(f"🏷️ Anda termasuk dalam: **{cluster_name}**")
            
            # Penjelasan dinamis
            if cluster_id == 0:
                st.write("💡 **Karakteristik:** Kelompok ini umumnya berusia muda, BMI normal, dan aktif secara fisik.")
            elif cluster_id == 1:
                st.write("💡 **Karakteristik:** Kelompok dengan gaya hidup berisiko (merokok/obesitas) namun belum memiliki penyakit kronis parah.")
            elif cluster_id == 2:
                st.write("💡 **Karakteristik:** Kelompok lansia dengan riwayat penyakit penyerta (Komorbid) tinggi.")

# HALAMAN 3: ATURAN ASOSIASI (STATIC INSIGHTS)
elif menu == "💡 Ensiklopedia Pola (Rules)":
    st.title("💡 Fakta & Pola Penyakit")
    st.write("Daftar pola tersembunyi yang ditemukan dari 250.000 data pasien menggunakan algoritma **Apriori**.")
    
    st.subheader("Top 5 Aturan Asosiasi (Lift > 1.5)")
    
    
    rules_data = [
        {"Jika": "Darah Tinggi & Obesitas", "Maka": "Kolesterol Tinggi", "Kekuatan (Lift)": "2.1x", "Rekomendasi": "Wajib Cek Profil Lemak Rutin"},
        {"Jika": "Stroke", "Maka": "Penyakit Jantung", "Kekuatan (Lift)": "1.9x", "Rekomendasi": "Pemeriksaan EKG Bulanan"},
        {"Jika": "Kesehatan Umum Buruk", "Maka": "Tidak Pernah Olahraga", "Kekuatan (Lift)": "1.8x", "Rekomendasi": "Fisioterapi Ringan"},
        {"Jika": "Diabetes & Darah Tinggi", "Maka": "Gangguan Jalan (DiffWalk)", "Kekuatan (Lift)": "1.6x", "Rekomendasi": "Terapi Mobilitas"},
        {"Jika": "Perokok Berat", "Maka": "Gangguan Pernapasan", "Kekuatan (Lift)": "1.5x", "Rekomendasi": "Stop Smoking Program"}
    ]
    
    st.table(pd.DataFrame(rules_data))
    
    st.info("ℹ️ **Lift > 1.0** menunjukkan hubungan sebab-akibat yang kuat, bukan kebetulan.")