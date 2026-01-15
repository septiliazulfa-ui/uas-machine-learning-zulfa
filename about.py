import streamlit as st
import pandas as pd

def show_about():
    st.title("📘 About Dataset")

    # =========================
    # PENGAMAN
    # =========================
    if "df" not in st.session_state:
        st.warning("Silakan upload dataset terlebih dahulu.")
        return

    df = st.session_state["df"]
    dataset_type = st.session_state.get("dataset_type", "Tidak diketahui")

    # =========================
    # DESKRIPSI DATASET
    # =========================
    st.subheader("📂 Informasi Umum Dataset")

    if dataset_type == "Lingkungan":
        st.write("""
        Dataset **Lingkungan (Occupancy Detection)** digunakan untuk
        memprediksi apakah suatu ruangan **terisi atau tidak**
        berdasarkan kondisi lingkungan seperti suhu, kelembaban,
        cahaya, dan kadar CO₂.
        """)
    elif dataset_type == "Kesehatan":
        st.write("""
        Dataset **Kesehatan (Maternal Health Risk)** digunakan untuk
        mengklasifikasikan tingkat risiko kesehatan ibu hamil
        berdasarkan indikator medis seperti tekanan darah,
        kadar gula darah, suhu tubuh, dan denyut jantung.
        """)
    else:
        st.write("Dataset yang digunakan diunggah oleh pengguna.")

    # =========================
    # RINGKASAN DATASET
    # =========================
    st.subheader("📊 Ringkasan Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Data", df.shape[0])
    col2.metric("Jumlah Fitur", df.shape[1])
    col3.metric("Jenis Dataset", dataset_type)

    # =========================
    # INFORMASI KOLOM
    # =========================
    st.subheader("🧱 Struktur Dataset")

    column_info = pd.DataFrame({
        "Nama Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str)
    })

    st.dataframe(column_info, use_container_width=True)

    # =========================
    # CONTOH DATA
    # =========================
    st.subheader("🔍 Contoh Data")
    st.dataframe(df.head(), use_container_width=True)

    # =========================
    # CATATAN KHUSUS
    # =========================
    st.info(
        "Catatan:\n"
        "- Dataset ini bersifat **dinamis**, menyesuaikan file yang di-upload.\n"
        "- Identifikasi fitur numerik dan kategorik dilakukan otomatis.\n"
        "- Target klasifikasi akan digunakan pada menu **Machine Learning**."
    )
