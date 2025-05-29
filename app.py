import streamlit as st
import pandas as pd
import numpy as np
import joblib # pycaret saves the model as a .pkl file, joblib is commonly used for this
from pycaret.regression import load_model, predict_model # Use PyCaret's load/predict functions

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Bike Sharing Demand Prediction", layout="wide")

st.title("Prediksi Permintaan Sewa Sepeda")
st.markdown("""
Aplikasi ini memprediksi jumlah total sewa sepeda (casual + registered)
berdasarkan parameter waktu dan cuaca menggunakan model regresi.
""")

# --- Memuat Model ---
# Asumsikan file model 'XGBoost_BikeSharing_Final_Model_PyCaret.pkl' ada di direktori yang sama
# dengan script app.py saat deploy.
@st.cache_resource # Cache the model loading
def load_trained_model():
    try:
        # PyCaret load_model handles the entire pipeline
        model = load_model('XGBoost_BikeSharing_Final_Model_PyCaret')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_trained_model()

if model:
    st.success("Model berhasil dimuat.")

    # --- Bagian Prediksi Interaktif (Contoh menggunakan input manual) ---
    st.header("Buat Prediksi Baru")
    st.write("Masukkan detail waktu dan cuaca untuk mendapatkan prediksi jumlah sewa.")

    # Input fields (contoh, bisa disesuaikan dengan fitur yang paling penting)
    col1, col2, col3 = st.columns(3)

    with col1:
        input_datetime = st.date_input("Tanggal")
        input_hour = st.slider("Jam (0-23)", 0, 23, 10)
        input_season = st.selectbox("Musim", {1: 'Musim Dingin', 2: 'Musim Semi', 3: 'Musim Panas', 4: 'Musim Gugur'})
        input_holiday = st.selectbox("Hari Libur?", {0: 'Tidak', 1: 'Ya'})

    with col2:
        input_workingday = st.selectbox("Hari Kerja?", {0: 'Tidak (Akhir Pekan/Libur)', 1: 'Ya'})
        input_weather = st.selectbox("Cuaca", {1: 'Cerah/Sedikit Awan', 2: 'Kabut/Berawan', 3: 'Hujan/Salju Ringan', 4: 'Hujan Lebat/Kabut Tebal'})
        input_temp = st.slider("Suhu (°C)", -10.0, 40.0, 20.0)
        input_atemp = st.slider("Suhu yang Dirasakan (°C)", -16.0, 50.0, 24.0)


    with col3:
        input_humidity = st.slider("Kelembapan (%)", 0, 100, 60)
        input_windspeed = st.slider("Kecepatan Angin", 0.0, 60.0, 15.0)

    # Buat DataFrame dari input pengguna (penting agar formatnya sama dengan data training)
    # Pastikan nama kolomnya sesuai dengan yang diharapkan oleh pipeline PyCaret
    input_data = pd.DataFrame([{
        'datetime': pd.to_datetime(f"{input_datetime} {input_hour}:00:00"), # Combine date and hour
        'season': input_season,
        'holiday': input_holiday,
        'workingday': input_workingday,
        'weather': input_weather,
        'temp': input_temp,
        'atemp': input_atemp,
        'humidity': input_humidity,
        'windspeed': input_windspeed
        # NOTE: Anda TIDAK perlu membuat fitur siklikal atau 'day', 'year_cat' dll.
        # di sini secara manual. Pipeline PyCaret (dari `load_model`) akan melakukannya
        # saat `predict_model` dipanggil, selama kolom input awal sesuai.
    }])

    st.write("\nData Input:")
    st.dataframe(input_data)

    if st.button("Prediksi Jumlah Sewa"):
        try:
            # Gunakan predict_model dari PyCaret. Ini akan menjalankan pipeline preprocessing
            # yang tersimpan dalam model sebelum membuat prediksi.
            predictions = predict_model(model, data=input_data)

            # Nama kolom prediksi default di PyCaret adalah 'prediction_label'
            predicted_count = predictions['prediction_label'].iloc[0]

            st.subheader("Hasil Prediksi")
            # Pastikan prediksi diubah kembali jika target di-transformasi
            # PyCaret predict_model sudah menangani inverse transform secara otomatis
            st.metric("Jumlah Total Sewa Sepeda Diprediksi:", f"{predicted_count:.2f}")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membuat prediksi: {e}")
            st.write("Pastikan format data input sesuai dengan yang dilatih model.")
            st.write("Detail error:", e)

    # --- Opsional: Menampilkan data atau visualisasi dari EDA ---
    # Ini bisa diimplementasikan jika Anda menyimpan hasil EDA atau data yang relevan
    # Misalnya, plot distribusi fitur, plot korelasi, dll.
    # st.header("Analisis Data (Opsional)")
    # Anda bisa memuat df_train lagi di sini dan menampilkan plot
