import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# =========================
# UTIL
# =========================
def safe_read_csv(path):
    return pd.read_csv(path, sep=None, engine="python")


def force_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =========================
# TRAIN: MATERNAL
# =========================
@st.cache_resource
def train_maternal():
    df = safe_read_csv("Maternal Health Risk Data Set.csv")

    features = [
        "Age", "SystolicBP", "DiastolicBP",
        "BS", "BodyTemp", "HeartRate"
    ]
    target = "RiskLevel"

    df = force_numeric(df, features)
    df = df.dropna(subset=features + [target])

    X = df[features]
    y_raw = df[target].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model.fit(X_train, y_train)

    return model, features, le


# =========================
# TRAIN: OCCUPANCY
# =========================
@st.cache_resource
def train_occupancy():
    df = safe_read_csv("datatraining occupancy.csv")

    features = [
        "Temperature", "Humidity",
        "Light", "CO2", "HumidityRatio"
    ]
    target = "Occupancy"

    df = force_numeric(df, features + [target])
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target].astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model.fit(X_train, y_train)

    return model, features


# =========================
# UI
# =========================
def prediction_page():
    st.title("🔮 Prediction App")

    pilihan = st.selectbox(
        "Pilih Jenis Prediksi",
        [
            "Kesehatan (Maternal Health Risk)",
            "Lingkungan (Occupancy Detection)"
        ]
    )

    st.markdown("---")

    if pilihan.startswith("Kesehatan"):
        maternal_app()
    else:
        occupancy_app()


# =========================
# APP: MATERNAL
# =========================
def maternal_app():
    st.subheader("🩺 Prediksi Risiko Kehamilan")

    model, features, le = train_maternal()

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 10, 80, 25)
        sys = st.number_input("SystolicBP", 70, 200, 120)

    with col2:
        dia = st.number_input("DiastolicBP", 40, 140, 80)
        bs = st.number_input("BS", 5.0, 30.0, 7.0)

    with col3:
        temp = st.number_input("BodyTemp", 90.0, 110.0, 98.0)
        hr = st.number_input("HeartRate", 40, 140, 80)

    input_df = pd.DataFrame(
        [[age, sys, dia, bs, temp, hr]],
        columns=features
    )

    if st.button("🔍 Prediksi Risiko"):
        probs = model.predict_proba(input_df)[0]
        idx = np.argmax(probs)
        label = le.inverse_transform([idx])[0]

        st.success(f"**Prediksi:** {label.upper()}")
        st.metric("Confidence", f"{probs[idx]:.2%}")


# =========================
# APP: OCCUPANCY
# =========================
def occupancy_app():
    st.subheader("🏢 Prediksi Status Ruangan")

    model, features = train_occupancy()

    col1, col2, col3 = st.columns(3)

    with col1:
        temp = st.number_input("Temperature", -10.0, 60.0, 23.0)
        hum = st.number_input("Humidity", 0.0, 100.0, 27.0)

    with col2:
        light = st.number_input("Light", 0.0, 2000.0, 426.0)
        co2 = st.number_input("CO2", 0.0, 5000.0, 720.0)

    with col3:
        hratio = st.number_input("HumidityRatio", 0.0, 1.0, 0.0048)

    input_df = pd.DataFrame(
        [[temp, hum, light, co2, hratio]],
        columns=features
    )

    if st.button("🔍 Prediksi Ruangan"):
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        status = "TERISI" if pred == 1 else "KOSONG"
        st.success(f"**Status Ruangan:** {status}")
        st.metric("Probabilitas Terisi", f"{prob:.2%}")
