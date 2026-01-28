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
def safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]  # rapihin header
    return df


def normalize_col(s: str) -> str:
    return str(s).strip().lower().replace(" ", "")


def resolve_columns(df: pd.DataFrame, wanted: list[str]) -> dict:
    lookup = {normalize_col(c): c for c in df.columns}
    mapping = {}
    missing = []
    for w in wanted:
        key = normalize_col(w)
        if key in lookup:
            mapping[w] = lookup[key]
        else:
            missing.append(w)

    if missing:
        raise KeyError(
            f"Kolom tidak ditemukan: {missing}\n"
            f"Kolom terbaca di file: {list(df.columns)}"
        )
    return mapping


def force_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =========================
# TRAINERS (CACHE)
# =========================
@st.cache_resource
def train_maternal_from_repo():
    df = safe_read_csv("Maternal Health Risk Data Set.csv")

    wanted_features = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
    wanted_target = "RiskLevel"

    mapping = resolve_columns(df, wanted_features + [wanted_target])

    feature_cols = [mapping[c] for c in wanted_features]
    target_col = mapping[wanted_target]

    df = force_numeric(df, feature_cols)
    df[target_col] = df[target_col].astype(str).str.strip()
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    X = df[feature_cols].copy()
    y_raw = df[target_col].copy()

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

    return model, wanted_features, feature_cols, le


@st.cache_resource
def train_occupancy_from_repo():
    df = safe_read_csv("datatraining occupancy.csv")

    wanted_features = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    wanted_target = "Occupancy"

    mapping = resolve_columns(df, wanted_features + [wanted_target])

    feature_cols = [mapping[c] for c in wanted_features]
    target_col = mapping[wanted_target]

    df = force_numeric(df, feature_cols + [target_col])
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)

    return model, wanted_features, feature_cols


# =========================
# PAGE
# =========================
def prediction_page():
    st.write("**Aplikasi Prediksi Machine Learning**")

    pilihan = st.selectbox(
        "Pilih Jenis Prediksi Machine Learning",
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
    st.title("🩺 Prediksi Risiko Kehamilan (Maternal Health Risk)")
    st.caption("Input indikator → prediksi tingkat risiko (low / mid / high)")

    try:
        model, ui_features, real_features, le = train_maternal_from_repo()
    except Exception as e:
        st.error("Dataset Kesehatan tidak terbaca / kolom tidak cocok.")
        st.code(str(e))
        return

    st.subheader("📋 Input Indikator Pasien")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 10, 80, 25)
        systolic = st.number_input("SystolicBP", 70, 200, 120)
    with c2:
        diastolic = st.number_input("DiastolicBP", 40, 140, 80)
        bs = st.number_input("BS", 5.0, 30.0, 7.0)
    with c3:
        bodytemp = st.number_input("BodyTemp", 90.0, 110.0, 98.0)
        heartrate = st.number_input("HeartRate", 40, 140, 80)

    input_df = pd.DataFrame([[age, systolic, diastolic, bs, bodytemp, heartrate]], columns=real_features)

    if st.button("🔍 Prediksi Risiko Kehamilan"):
        probs = model.predict_proba(input_df)[0]
        idx = int(np.argmax(probs))
        label = le.inverse_transform([idx])[0]

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        colA, colB = st.columns(2)
        colA.metric("Prediksi Risiko", label.upper())
        colB.metric("Confidence", f"{probs[idx]:.2%}")

        st.subheader("🧾 Probabilitas Setiap Kelas")
        prob_df = pd.DataFrame({
            "Kelas": le.inverse_transform(np.arange(len(probs))),
            "Probabilitas": probs
        }).sort_values("Probabilitas", ascending=False)
        st.table(prob_df)


# =========================
# APP: OCCUPANCY
# =========================
def occupancy_app():
    st.title("🏢 Prediksi Status Ruangan (Occupancy Detection)")
    st.caption("Input indikator sensor → prediksi ruangan TERISI / KOSONG")

    try:
        model, ui_features, real_features = train_occupancy_from_repo()
    except Exception as e:
        st.error("Dataset Lingkungan tidak terbaca / kolom tidak cocok.")
        st.code(str(e))
        return

    st.subheader("📋 Input Indikator Sensor")

    c1, c2, c3 = st.columns(3)
    with c1:
        temp = st.number_input("Temperature", -10.0, 60.0, 23.0, step=0.1)
        hum = st.number_input("Humidity", 0.0, 100.0, 27.0, step=0.1)
    with c2:
        light = st.number_input("Light", 0.0, 2000.0, 426.0, step=1.0)
        co2 = st.number_input("CO2", 0.0, 5000.0, 720.0, step=1.0)
    with c3:
        hratio = st.number_input("HumidityRatio", 0.0, 1.0, 0.0048, step=0.0001)

    input_df = pd.DataFrame([[temp, hum, light, co2, hratio]], columns=real_features)

    if st.button("🔍 Prediksi Status Ruangan"):
        prob_occ = float(model.predict_proba(input_df)[0][1])
        pred = int(model.predict(input_df)[0])

        status = "TERISI (Occupied)" if pred == 1 else "KOSONG (Not Occupied)"

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        colA, colB = st.columns(2)
        colA.metric("Probabilitas Terisi", f"{prob_occ:.2%}")
        colB.metric("Prediksi Status", status)
