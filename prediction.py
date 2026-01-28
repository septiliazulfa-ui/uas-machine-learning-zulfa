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
def _safe_read_csv(path):
    return pd.read_csv(path, sep=None, engine="python")


def _coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# =========================
# TRAIN MODEL (CACHE)
# =========================
@st.cache_resource
def _train_maternal_model():
    df = _safe_read_csv("Maternal Health Risk Data Set.csv")

    features = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
    target = "RiskLevel"

    df = _coerce_numeric(df, features)
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
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    return {"model": model, "features": features, "le": le}


@st.cache_resource
def _train_occupancy_model():
    df = _safe_read_csv("datatraining occupancy.csv")

    features = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    target = "Occupancy"

    df = _coerce_numeric(df, features + [target])

    if "date" in df.columns:
        df["Hour"] = pd.to_datetime(df["date"], errors="coerce").dt.hour
        features.append("Hour")

    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target].astype(int)

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=2
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    return {"model": model, "features": features}


# =========================
# UI HELPERS
# =========================
def _risk_badge(p):
    if p < 0.30:
        return "🟢 Risiko Rendah"
    elif p < 0.60:
        return "🟡 Risiko Sedang"
    return "🔴 Risiko Tinggi"


# =========================
# MAIN PAGE
# =========================
def prediction_page():
    st.title("🔮 Prediction App")
    st.write("Aplikasi prediksi Machine Learning berbasis input numerik")

    pilihan = st.selectbox(
        "Pilih Jenis Prediksi Machine Learning",
        ["Kesehatan", "Lingkungan"],
        index=0
    )

    st.markdown("---")

    if pilihan == "Kesehatan":
        _app_prediksi_maternal()
    else:
        _app_prediksi_occupancy()


# =========================
# APP: KESEHATAN
# =========================
def _app_prediksi_maternal():
    st.subheader("🩺 Prediksi Risiko Kehamilan")
    st.caption("Input numerik → klasifikasi Low / Mid / High Risk")

    bundle = _train_maternal_model()
    model = bundle["model"]
    features = bundle["features"]
    le = bundle["le"]

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 10, 80, 25)
        sbp = st.number_input("SystolicBP", 70.0, 200.0, 120.0)

    with col2:
        dbp = st.number_input("DiastolicBP", 40.0, 140.0, 80.0)
        bs = st.number_input("Blood Sugar (BS)", 5.0, 30.0, 7.0)

    with col3:
        temp = st.number_input("BodyTemp", 90.0, 110.0, 98.0)
        hr = st.number_input("HeartRate", 40, 140, 80)

    input_df = pd.DataFrame([[age, sbp, dbp, bs, temp, hr]], columns=features)

    if st.button("🔍 Prediksi Risiko"):
        probs = model.predict_proba(input_df)[0]
        idx = np.argmax(probs)
        label = le.inverse_transform([idx])[0]

        st.markdown("---")
        colA, colB = st.columns(2)
        colA.metric("Kategori Risiko", label.upper())
        colB.metric("Confidence", f"{probs[idx]:.2%}")


# =========================
# APP: LINGKUNGAN
# =========================
def _app_prediksi_occupancy():
    st.subheader("🏢 Prediksi Status Ruangan")
    st.caption("Input sensor → Occupied / Not Occupied")

    bundle = _train_occupancy_model()
    model = bundle["model"]
    features = bundle["features"]

    col1, col2, col3 = st.columns(3)

    with col1:
        temp = st.number_input("Temperature", -10.0, 60.0, 23.0)
        hum = st.number_input("Humidity", 0.0, 100.0, 27.0)

    with col2:
        light = st.number_input("Light", 0.0, 2000.0, 426.0)
        co2 = st.number_input("CO2", 0.0, 5000.0, 720.0)

    with col3:
        hratio = st.number_input("HumidityRatio", 0.0, 1.0, 0.0048)
        hour = st.number_input("Hour", 0, 23, 17) if "Hour" in features else None

    row = []
    for f in features:
        row.append(locals()[f.lower()] if f.lower() in locals() else hour)

    input_df = pd.DataFrame([row], columns=features)

    if st.button("🔍 Prediksi Status"):
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.markdown("---")
        colA, colB = st.columns(2)
        colA.metric("Probabilitas Terisi", f"{prob:.2%}")
        colB.metric("Prediksi", "TERISI" if pred == 1 else "KOSONG")

        st.markdown(f"**Kategori Risiko:** {_risk_badge(prob)}")
