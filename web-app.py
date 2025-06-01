import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler


# List nama fitur yang digunakan untuk melatih model
KOLOM_STROKE = ["Age", "Gender", "SES", "Hypertension", "Heart_Disease", "BMI", "Avg_Glucose", "Diabetes", "Smoking_Status"]
KOLOM_JANTUNG = ["GenHlth", "HighBP", "Age", "Diabetes", "HighChol", "Smoker", "Sex", "AnyHealthcare"]
KOLOM_DIABETES = ["Age", "HighChol", "BMI", "GenHlth", "DiffWalk", "HighBP"]

# List nama penyakit yang diprediksi model
PENYAKIT = ["Stroke", "Jantung", "Diabetes"]


# ==================================
# USER INTERFACE [UI] ISIAN FORMULIR
# ==================================

# [UI] Judul formulir
st.title("🩺RiMed AI")

# [UI] Isian formulir untuk menerima input data dari user
gender = st.selectbox("Jenis Kelamin", ["-", "Perempuan", "Laki-laki"])
age = st.number_input("Usia", min_value=0)
body_weight = st.number_input("Berat Badan (kg)", min_value=0)
body_height = st.number_input("Tinggi Badan (cm)", min_value=0)
avg_glucose = st.number_input("Kadar gula rata-rata (mg/dL)", min_value=0)
high_chol = st.selectbox("Apakah kadar kolestrol Anda tinggi?", ["-", "Ya", "Tidak"])
high_bp = st.selectbox("Apakah Anda memiliki riwayat tekanan darah tinggi?", ["-", "Ya", "Tidak"])
# heart_disease = st.selectbox("Apakah Anda memiliki riwayat penyakit jantung?", ["-", "Ya", "Tidak"])
# diabetes = st.selectbox("Apakah Anda memiliki riwayat diabetes?", ["-", "Ya", "Tidak"])
difficulty_walk = st.selectbox("Apakah Anda mengalami kesulitan dalam berjalan atau saat menapaki tangga?", ["-", "Ya", "Tidak"])
smoking_status = st.selectbox("Apa status kebiasaan merokok Anda saat ini?", ["-", "Tidak Pernah Merokok", "Pernah Merokok", "Masih Merokok"])
general_health = st.selectbox("Jika diskalakan, berapa skala kesehatan Anda menurut Anda? (skala 1 - 5)", ["-", "1: Sangat Sehat", "2: Sehat", "3: Biasa", "4: Kurang Sehat", "5: Sangat Tidak Sehat"])
any_healthcare = st.selectbox("Apakah Anda memiliki akses terhadap layanan kesehatan (misalnya: rumah sakit, puskesmas, klinik, atau dokter)?", ["-", "Ya", "Tidak"])
ses = st.selectbox("Bagaimana Anda menilai kondisi sosial ekonomi Anda saat ini?", ["-", "Menengah ke Bawah", "Menengah", "Menengah ke Atas"])
submit = st.button("Prediksi")


# ===============================================
# PREPROCESS & ENCODE INPUT DATA KATEGORIKAL USER
# ===============================================

smoker = "Ya" if smoking_status == "Masih Merokok" else "Tidak"
heart_disease = "Tidak"
diabetes = "Tidak"

# Mapping label ke list nilai yang akan di-encode
encoding_maps = {
    "gender": ["Perempuan", "Laki-laki"],
    "high_chol": ["Tidak", "Ya"],
    "general_health": [
        "1: Sangat Sehat",
        "2: Sehat",
        "3: Biasa",
        "4: Kurang Sehat",
        "5: Sangat Tidak Sehat"
    ],
    "difficulty_walk": ["Tidak", "Ya"],
    "high_bp": ["Tidak", "Ya"],
    "heart_disease": ["Tidak", "Ya"],
    "diabetes": ["Tidak", "Ya"],
    "smoking_status": ["Tidak Pernah Merokok", "Pernah Merokok", "Masih Merokok"],
    "smoker": ["Tidak", "Ya"],
    "any_healthcare": ["Tidak", "Ya"],
    "ses": ["Menengah ke Bawah", "Menengah", "Menengah ke Atas"],
}

selectbox_variables = [
    ("gender", gender, "Jenis Kelamin"),
    ("high_chol", high_chol, "Kadar kolestrol"),
    ("general_health", general_health, "Skala kesehatan"),
    ("difficulty_walk", difficulty_walk, "Skala kesulitan berjalan"),
    ("high_bp", high_bp, "Riwayat tekanan darah tinggi"),
    ("heart_disease", heart_disease, "Riwayat penyakit jantung"), 
    ("diabetes", diabetes, "Riwayat diabetes"), 
    ("smoking_status", smoking_status, "Riwayat merokok"), 
    ("smoker", smoker, "Status merokok saat ini"), 
    ("any_healthcare", any_healthcare, "Akses layanan kesehatan"),
    ("ses", ses, "Status sosial ekonomi"),
]

encoded_values = {}

for var_name, variable, desc in selectbox_variables:
    options = encoding_maps.get(var_name, [])
    encoded = [i for i, v in enumerate(options) if variable == v]
    encoded_values[var_name] = encoded[0] if encoded else None

high_chol_encoded = encoded_values["high_chol"]
general_health_encoded = encoded_values["general_health"]
difficulty_walk_encoded = encoded_values["difficulty_walk"]
high_bp_encoded = encoded_values["high_bp"]
gender_encoded = encoded_values["gender"]
heart_disease_encoded = encoded_values["heart_disease"]
diabetes_encoded = encoded_values["diabetes"]
smoking_status_encoded = encoded_values["smoking_status"]
smoker_encoded = encoded_values["smoker"]
any_healthcare_encoded = encoded_values["any_healthcare"]
ses_encoded = encoded_values["ses"]


try:
    bmi = body_weight / (body_height ** 2)
except (ZeroDivisionError, TypeError, ValueError):
    bmi = None

# Simpan data input dari user ke dalam DataFrame
df_stroke = pd.DataFrame([[age, gender_encoded, ses_encoded, high_bp_encoded, heart_disease_encoded, bmi, avg_glucose, diabetes_encoded, smoking_status_encoded]], columns=KOLOM_STROKE)
df_jantung = pd.DataFrame([[general_health_encoded, high_bp_encoded, age, diabetes_encoded, high_chol_encoded, smoker_encoded, gender_encoded, any_healthcare_encoded]], columns=KOLOM_JANTUNG)
df_diabetes = pd.DataFrame([[age, high_chol_encoded, bmi, general_health_encoded, difficulty_walk_encoded, high_bp_encoded]], columns=KOLOM_DIABETES)


# ==========================================================
# RUN PREDIKSI MODEL ML & USER INTERFACE [UI] HASIL PREDIKSI
# ==========================================================

if submit:
    # Cek user sudah mengisi seluruh formulir atau belum
    errors = []
    numeric_variables = [
        (age, "Usia"),
        (body_weight, "Berat badan"),
        (body_height, "Tinggi badan"),
        (avg_glucose, "Kadar gula darah rata-rata"),
    ]

    for var_name, variable, desc in selectbox_variables:
        if variable == "-":
            errors.append(f"**{desc}** belum dipilih. Silahkan pilih terlebih dahulu.")

    for variable, desc in numeric_variables:
        if age == 0:
            errors.append(f"**{desc}** belum diisi. Silahkan isi terlebih dahulu.")

    if age > 120:
        errors.append("**Usia** melewati batas wajar. Mohon periksa kembali.")
    
    if body_weight > 300:
        errors.append("**Berat badan** melewati batas wajar. Mohon periksa kembali.")
    
    if body_weight > 300:
        errors.append("**Tinggi badan** melewati batas wajar. Mohon periksa kembali.")

    if errors:
        for err in errors:
            st.error(err)
    else:
        # Kalau formulir sudah terisi semua, lanjut ke prediksi
        # loading ...
        with st.spinner("Sedang memproses prediksi..."):
            # Load 3 model ML
            model_stroke = joblib.load('export-model/stroke_rf_bayes_model_smote.pkl')
            model_jantung = joblib.load('export-model/lr_jantung_smoteenn.pkl')
            model_diabetes = joblib.load('export-model/Deteksi_diabetes_NN2.pkl')

            # Prediksi
            prediksi_stroke = model_stroke.predict(df_stroke)
            persentase_risiko_stroke = model_stroke.predict_proba(df_stroke)[0][1] * 100
            prediksi_jantung = model_jantung.predict(df_jantung)
            persentase_risiko_jantung = model_jantung.predict_proba(df_jantung)[0][1] * 100
            prediksi_diabetes = model_diabetes.predict(df_diabetes)
            persentase_risiko_diabetes = model_diabetes.predict(df_diabetes)[0][0] * 100 

            # Konversi hasil prediksi jadi label risiko
            label_stroke = "tinggi" if persentase_risiko_stroke >= 50 else "rendah"
            label_jantung = "tinggi" if persentase_risiko_jantung >= 50 else "rendah"
            label_diabetes = "tinggi" if persentase_risiko_diabetes >= 50 else "rendah"

        # Buat hasil prediksi semua penyakit
        hasil_prediksi = pd.DataFrame({
            "penyakit": PENYAKIT,
            "risiko": [label_stroke, label_jantung, label_diabetes]
        })

        # Filter berdasarkan nilai risiko
        tinggi = hasil_prediksi[hasil_prediksi["risiko"] == "tinggi"]["penyakit"].tolist()
        rendah = hasil_prediksi[hasil_prediksi["risiko"] == "rendah"]["penyakit"].tolist()

        def format_daftar_penyakit(nama_list):
            if len(nama_list) == 1:
                return nama_list[0]
            elif len(nama_list) == 2:
                return f"{nama_list[0]} dan {nama_list[1]}"
            else:
                return ", ".join(nama_list[:-1]) + f", dan {nama_list[-1]}"

        # [UI] Tampilkan kalimat hasil prediksi
        if tinggi:
            st.markdown(
                f"<h3 style='color: darkred; font-weight: bold;'>⚠️Risiko {format_daftar_penyakit(tinggi)} tinggi!</h3>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style='
                    background-color: #FADBD8;
                    padding: 15px;
                    border-left: 5px solid #C0392B;
                    border-radius: 8px;
                    color: #641E16;
                    font-size: 16px;
                '>
                    <h5><strong>Hasil Prediksi Risiko:</strong></h5>
                    ➤ <strong>Stroke: {persentase_risiko_stroke:.0f}%</strong><br>
                    ➤ <strong>Jantung: {persentase_risiko_jantung:.0f}%</strong><br>
                    ➤ <strong>Diabetes: {persentase_risiko_diabetes:.0f}%</strong><br><br>
                    <strong>Segera konsultasikan ke dokter</strong> untuk pemeriksaan lebih lanjut.<br>
                    Disarankan untuk mulai menerapkan pola hidup sehat secara konsisten.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h3 style='color: #1E8449; font-weight: bold;'>Risiko {format_daftar_penyakit(rendah)} rendah</h3>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style='
                    background-color: #D5F5E3;
                    padding: 15px;
                    border-left: 5px solid #27AE60;
                    border-radius: 8px;
                    color: #145A32;
                    font-size: 16px;
                '>
                    <h5><strong>Hasil Prediksi Risiko:</strong></h5>
                    ➤ <strong>Stroke: {persentase_risiko_stroke:.0f}%</strong><br>
                    ➤ <strong>Jantung: {persentase_risiko_jantung:.0f}%</strong><br>
                    ➤ <strong>Diabetes: {persentase_risiko_diabetes:.0f}%</strong><br><br>
                    Tetap jaga pola hidup sehat. <br>
                    Lanjutkan kebiasaan baik seperti <strong>olahraga rutin</strong> dan <strong>makan bergizi</strong>.
                </div>
                """,
                unsafe_allow_html=True
            )
