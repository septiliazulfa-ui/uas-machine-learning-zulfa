import streamlit as st
import pandas as pd

def sidebar_upload():
    st.sidebar.title("📂 Dataset")

    dataset_type = st.sidebar.selectbox(
        "Jenis Dataset",
        ["Lingkungan", "Kesehatan"]
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset (CSV)",
        type=["csv"]
    )

    if uploaded_file is not None:
        # ===== FIX DELIMITER CSV =====
        try:
            df = pd.read_csv(uploaded_file, sep=";")
        except:
            df = pd.read_csv(uploaded_file)

        # ===== SIMPAN KE SESSION STATE =====
        st.session_state["df"] = df
        st.session_state["dataset_type"] = dataset_type

        st.sidebar.success("Dataset berhasil diupload!")

    # ===== INFO DATASET =====
    if "df" in st.session_state:
        st.sidebar.write("📊 Shape:", st.session_state["df"].shape)
        st.sidebar.write("📌 Kolom:", list(st.session_state["df"].columns))
