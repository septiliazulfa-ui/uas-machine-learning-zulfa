import streamlit as st
import pandas as pd
import numpy as np

def prediction_page():
    st.title("🔮 Prediction App")

    # =========================
    # PENGAMAN
    # =========================
    if "best_model" not in st.session_state:
        st.warning("Silakan jalankan menu Machine Learning terlebih dahulu.")
        return

    if "df" not in st.session_state:
        st.warning("Silakan upload dataset terlebih dahulu.")
        return

    model = st.session_state["best_model"]
    df = st.session_state["df"]
    dataset_type = st.session_state.get("dataset_type")

    scaler = st.session_state.get("scaler")
    feature_columns = st.session_state.get("feature_columns")
    label_encoder = st.session_state.get("label_encoder")

    st.write(
        "Menu ini melakukan prediksi secara otomatis menggunakan "
        "data terakhir yang tersedia pada dataset."
    )

    st.markdown("---")

    # =========================
    # INPUT OTOMATIS
    # =========================
    st.subheader("📥 Data Input Otomatis")

    input_df = df[feature_columns].iloc[-1:].copy()

    st.write(
        "Data berikut diambil dari **baris terakhir dataset** "
        "dan digunakan sebagai input prediksi."
    )
    st.dataframe(input_df)

    # =========================
    # PREPROCESSING
    # =========================
    if scaler is not None:
        input_processed = scaler.transform(input_df)
    else:
        input_processed = input_df.values

    # =========================
    # PREDIKSI
    # =========================
    if st.button("🔍 Jalankan Prediksi"):
        prediction = model.predict(input_processed)[0]

        # Decode label jika ada
        if label_encoder is not None:
            prediction_label = label_encoder.inverse_transform([prediction])[0]
        else:
            prediction_label = prediction

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        if dataset_type == "Lingkungan":
            if prediction_label == 1:
                st.success("🏢 Prediksi Status Ruangan: **TERISI (Occupied)**")
            else:
                st.info("🏢 Prediksi Status Ruangan: **KOSONG (Not Occupied)**")

        elif dataset_type == "Kesehatan":
            st.success(f"🏥 Prediksi Tingkat Risiko: **{prediction_label.upper()}**")

        st.write(
            "Prediksi ini dilakukan menggunakan **model terbaik** "
            "yang dipilih berdasarkan evaluasi performa pada menu Machine Learning."
        )

    st.markdown("---")
    st.info(
        "Catatan: Prediksi ini **bukan forecasting waktu**. "
        "Sistem menggunakan data terakhir yang tersedia pada dataset "
        "sebagai representasi kondisi saat ini."
    )
