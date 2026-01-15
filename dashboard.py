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

    # Heatmap Korelasi (numerik saja)
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            title="Heatmap Korelasi Fitur Numerik"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # =========================================================
    # 🌱 DATASET LINGKUNGAN – OCCUPANCY DETECTION
    # =========================================================
    if dataset_type == "Lingkungan":

        st.subheader("🌱 Dashboard Lingkungan – Occupancy Detection")

        total_data = df.shape[0]
        occupied = df["Occupancy"].sum()
        occupancy_rate = (occupied / total_data) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Data", total_data)
        col2.metric("Ruang Terisi", occupied)
        col3.metric("Persentase Terisi", f"{occupancy_rate:.2f}%")

        st.dataframe(df.head(), use_container_width=True)

        # ===== PIE CHART =====
        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                df,
                names="Occupancy",
                title="Status Occupancy (0 = Kosong, 1 = Terisi)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(
                df,
                values="Light",
                names="Occupancy",
                title="Distribusi Cahaya terhadap Occupancy"
            )
            st.plotly_chart(fig, use_container_width=True)

        # ===== HISTOGRAM =====
        fig = px.histogram(
            df,
            x="Temperature",
            nbins=30,
            title="Distribusi Temperature"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===== SCATTER =====
        fig = px.scatter(
            df,
            x="CO2",
            y="Humidity",
            color="Occupancy",
            title="CO2 vs Humidity"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===== BOX PLOT =====
        fig = px.box(
            df,
            x="Occupancy",
            y="Temperature",
            title="Distribusi Temperature berdasarkan Occupancy"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===== LINE CHART =====
        df_time = df.copy()
        df_time["date"] = pd.to_datetime(df_time["date"])

        occupancy_time = (
            df_time.groupby(df_time["date"].dt.hour)["Occupancy"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            occupancy_time,
            x="date",
            y="Occupancy",
            title="Rata-rata Occupancy berdasarkan Jam"
        )
        st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # 🏥 DATASET KESEHATAN – MATERNAL HEALTH
    # =========================================================
    elif dataset_type == "Kesehatan":

        st.subheader("🏥 Dashboard Kesehatan – Maternal Health Risk")

        total_data = df.shape[0]
        high_risk = (df["RiskLevel"] == "high risk").sum()
        risk_rate = (high_risk / total_data) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pasien", total_data)
        col2.metric("High Risk", high_risk)
        col3.metric("Persentase High Risk", f"{risk_rate:.2f}%")

        st.dataframe(df.head(), use_container_width=True)

        # ===== PIE CHART =====
        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                df,
                names="RiskLevel",
                title="Distribusi Risk Level"
            )
            st.plotly_chart(fig, use_container_width=True)

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

        # ===== BOX PLOT =====
        fig = px.box(
            df,
            x="RiskLevel",
            y="BS",
            title="Distribusi Blood Sugar berdasarkan Risk Level"
        )
        st.plotly_chart(fig, use_container_width=True)

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
