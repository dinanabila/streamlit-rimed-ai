import streamlit as st
import pandas as pd

# List nama fitur yang digunakan untuk melatih model
KOLOM_STROKE = ["Age", "Gender"]
KOLOM_JANTUNG = ["Age", "Gender"]
KOLOM_DIABETES = ["Age", "Gender"]
# List nama penyakit yang diprediksi model
PENYAKIT = ["Stroke", "Jantung", "Diabetes"]

# ===================
# USER INTERFACE [UI]
# ===================

# [UI] Judul formulir
st.title("🩺RiMed AI")

# [UI] Isian formulir untuk menerima input data dari user
gender = st.selectbox("Jenis Kelamin", ["Pilih Jenis Kelamin", "Perempuan", "Laki-laki"])
age = st.number_input("Usia", min_value=0)

# [UI] Tombol Prediksi
submit = st.button("Prediksi")

# Simpan data input dari user ke dalam DataFrame
data_stroke = pd.DataFrame([[age, gender]], columns=KOLOM_STROKE)
data_jantung = pd.DataFrame([[age, gender]], columns=KOLOM_JANTUNG)
data_diabetes = pd.DataFrame([[age, gender]], columns=KOLOM_DIABETES)

if submit:
    # Cek user sudah mengisi seluruh formulir atau belum
    errors = []

    if gender == "Pilih Jenis Kelamin":
        errors.append("Jenis kelamin belum dipilih. Silahkan pilih terlebih dahulu.")
    if age == 0:
        errors.append("Usia belum diisi. Silahkan isi terlebih dahulu.")
    if age > 120:
        errors.append("Usia melewati batas wajar. Mohon periksa kembali.")

    if errors:
        for err in errors:
            st.error(err)
    else:
        # Kalau formulir sudah terisi semua, lanjut ke prediksi

        # st.success("Data valid. Memproses prediksi...") # yang ini ntar dulu deh, opsional, cocok buat kalau modelnya mayan lama nge-run
    
        hasil_prediksi = pd.DataFrame({
            "penyakit": PENYAKIT,
            "risiko": ["rendah", "tinggi", "tinggi"]  # sementara ini dulu aja buat demo, nanti bikin list benerannya pas model udah fix
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
                f"<h3 style='color: darkred; font-weight: bold;'>Risiko {format_daftar_penyakit(tinggi)} tinggi!</h3>",
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <div style='
                    background-color: #FADBD8;
                    padding: 15px;
                    border-left: 5px solid #C0392B;
                    border-radius: 8px;
                    color: #641E16;
                    font-size: 16px;
                '>
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
                """
                <div style='
                    background-color: #D5F5E3;
                    padding: 15px;
                    border-left: 5px solid #27AE60;
                    border-radius: 8px;
                    color: #145A32;
                    font-size: 16px;
                '>
                    Tetap jaga pola hidup sehat. <br>
                    Lanjutkan kebiasaan baik seperti <strong>olahraga rutin</strong> dan <strong>makan bergizi</strong>.
                </div>
                """,
                unsafe_allow_html=True
            )
