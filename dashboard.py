import streamlit as st
import plotly.express as px
import pandas as pd

def dashboard_page():
    # ===== PENGAMAN =====
    if "df" not in st.session_state:
        st.warning("Silakan upload dataset terlebih dahulu.")
        return

    df = st.session_state["df"]
    dataset_type = st.session_state.get("dataset_type")

    st.title("📊 Dashboards")

    # ===== NARASI SETELAH JUDUL =====
    st.markdown(
        """
        Dashboard ini menyajikan hasil *Exploratory Data Analysis (EDA)* dan visualisasi data 
        untuk memahami karakteristik dataset serta pola hubungan antar variabel. 
        Analisis ini bertujuan untuk memberikan gambaran awal mengenai distribusi data, 
        korelasi fitur numerik, serta faktor-faktor yang berpotensi memengaruhi tingkat risiko.
        """
    )

    # =========================================================
    # 🔍 EDA – Exploratory Data Analysis (UMUM)
    # =========================================================
    st.subheader("🔍 Exploratory Data Analysis (EDA)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Data", df.shape[0])
    col2.metric("Jumlah Fitur", df.shape[1])
    col3.metric("Missing Value", df.isnull().sum().sum())

    st.write("### 📌 Tipe Data")
    st.dataframe(df.dtypes.astype(str), use_container_width=True)

    st.write("### 📊 Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            title="Heatmap Korelasi Fitur Numerik"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            **Interpretasi Heatmap Korelasi:**

            Heatmap korelasi menunjukkan hubungan antar fitur numerik dalam dataset. 
            Terlihat bahwa *SystolicBP* dan *DiastolicBP* memiliki korelasi positif yang kuat, 
            yang menandakan bahwa peningkatan tekanan darah sistolik cenderung diikuti oleh 
            peningkatan tekanan darah diastolik.

            Fitur *Blood Sugar (BS)* menunjukkan korelasi sedang terhadap tekanan darah, 
            sehingga berpotensi menjadi indikator penting dalam penentuan risiko kesehatan.
            Sementara itu, *BodyTemp* dan *HeartRate* memiliki korelasi yang relatif rendah 
            terhadap fitur lainnya.
            """
        )

    st.markdown("---")

    # =========================================================
    # 🏥 DATASET KESEHATAN – MATERNAL HEALTH
    # =========================================================
    if dataset_type == "Kesehatan":

        st.subheader("🏥 Dashboard Kesehatan – Maternal Health Risk")

        st.markdown(
            """
            Dashboard ini bertujuan untuk menganalisis kondisi kesehatan ibu hamil 
            berdasarkan indikator medis seperti tekanan darah, kadar gula darah, 
            suhu tubuh, dan detak jantung. Visualisasi digunakan untuk mengidentifikasi 
            distribusi tingkat risiko serta karakteristik pasien dengan risiko rendah, 
            sedang, dan tinggi.
            """
        )

        total_data = df.shape[0]
        high_risk = (df["RiskLevel"] == "high risk").sum()
        risk_rate = (high_risk / total_data) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pasien", total_data)
        col2.metric("High Risk", high_risk)
        col3.metric("Persentase High Risk", f"{risk_rate:.2f}%")

        st.dataframe(df.head(), use_container_width=True)

        col1, col2 = st.columns(2)

        # ===== PIE CHART =====
        with col1:
            fig = px.pie(
                df,
                names="RiskLevel",
                title="Distribusi Risk Level"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
                **Interpretasi Distribusi Risk Level:**

                Distribusi Risk Level menunjukkan bahwa mayoritas pasien berada pada kategori 
                risiko rendah dan sedang. Namun, terdapat proporsi yang cukup signifikan pada 
                kategori risiko tinggi. Hal ini menandakan pentingnya deteksi dini terhadap 
                ibu hamil yang berisiko tinggi guna mencegah terjadinya komplikasi kehamilan.
                """
            )

        with col2:
            fig = px.pie(
                df,
                values="HeartRate",
                names="RiskLevel",
                title="Heart Rate berdasarkan Risk Level"
            )
            st.plotly_chart(fig, use_container_width=True)

        # ===== HISTOGRAM =====
        fig = px.histogram(
            df,
            x="Age",
            nbins=30,
            title="Distribusi Umur Pasien"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===== SCATTER =====
        fig = px.scatter(
            df,
            x="SystolicBP",
            y="DiastolicBP",
            color="RiskLevel",
            title="Tekanan Darah Sistolik vs Diastolik"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            **Interpretasi Tekanan Darah Sistolik vs Diastolik:**

            Grafik scatter menunjukkan bahwa pasien dengan kategori risiko tinggi 
            cenderung memiliki nilai tekanan darah sistolik dan diastolik yang lebih tinggi 
            dibandingkan pasien dengan risiko rendah dan sedang. Pola ini mengindikasikan 
            bahwa tekanan darah merupakan faktor penting dalam penentuan risiko kesehatan maternal.
            """
        )

        # ===== BOX PLOT =====
        fig = px.box(
            df,
            x="RiskLevel",
            y="BS",
            title="Distribusi Blood Sugar berdasarkan Risk Level"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            **Interpretasi Blood Sugar berdasarkan Risk Level:**

            Pasien dengan kategori risiko tinggi memiliki median dan rentang kadar gula darah 
            yang lebih besar dibandingkan kategori risiko lainnya. Hal ini menunjukkan adanya 
            hubungan antara peningkatan kadar gula darah dengan meningkatnya risiko kesehatan 
            pada ibu hamil.
            """
        )

        # ===== LINE CHART =====
        age_risk = df.groupby(
            ["Age", "RiskLevel"]
        ).size().reset_index(name="count")

        fig = px.line(
            age_risk,
            x="Age",
            y="count",
            color="RiskLevel",
            title="Distribusi Risiko berdasarkan Umur"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            **Interpretasi Risiko berdasarkan Umur:**

            Distribusi risiko berdasarkan umur menunjukkan bahwa kategori risiko tinggi 
            cenderung meningkat pada kelompok usia tertentu. Hal ini mengindikasikan bahwa 
            faktor usia berperan dalam peningkatan risiko kesehatan maternal, terutama pada 
            usia kehamilan yang lebih matang.
            """
        )
