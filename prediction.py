import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# =========================
# UTIL: LOAD + CLEAN
# =========================
def _safe_read_csv(path: str) -> pd.DataFrame:
    # encoding="utf-8-sig" -> buang BOM (contoh: \ufeffAge)
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def _coerce_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _require_columns(df: pd.DataFrame, required_cols: list[str], title: str) -> bool:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.markdown(f"### ❌ {title} tidak terbaca / kolom tidak cocok")
        st.markdown("**Detail:**")
        st.code(
            f"Kolom tidak ditemukan: {missing}\n"
            f"Kolom terbaca di file: {list(df.columns)}"
        )
        return False
    return True


# =========================
# TRAIN MODEL (CACHE)
# =========================
@st.cache_resource
def _train_maternal_model():
    """
    Kesehatan: Maternal Health Risk
    Features: Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate
    Target: RiskLevel
    """
    df = _safe_read_csv("Maternal Health Risk Data Set.csv")

    features = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
    target = "RiskLevel"

    if not _require_columns(df, features + [target], "Dataset Kesehatan"):
        return None

    df = _coerce_numeric(df, features)
    df[target] = df[target].astype(str).str.strip().str.lower()
    df = df.dropna(subset=features + [target]).reset_index(drop=True)

    X = df[features].copy()
    y_raw = df[target].copy()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # ✅ FIX: jangan pakai multi_class (biar aman di versi sklearn berbeda)
    # lbfgs aman untuk multinomial multiclass, default juga oke.
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=5000
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    return {"model": model, "features": features, "label_encoder": le}


@st.cache_resource
def _train_occupancy_model():
    """
    Lingkungan: Occupancy Detection
    Features: Temperature, Humidity, Light, CO2, HumidityRatio (+Hour opsional)
    Target: Occupancy
    """
    df = _safe_read_csv("datatraining occupancy.csv")

    base_features = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    target = "Occupancy"

    if not _require_columns(df, base_features + [target], "Dataset Lingkungan"):
        return None

    df = _coerce_numeric(df, base_features + [target])

    features = base_features.copy()
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        df["Hour"] = dt.dt.hour
        features.append("Hour")

    df = df.dropna(subset=features + [target]).reset_index(drop=True)

    X = df[features].copy()
    y = df[target].astype(int).copy()

    model = Pipeline(steps=[
        ("clf", RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            min_samples_leaf=2
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    return {"model": model, "features": features}


# =========================
# LABEL HELPERS
# =========================
def _maternal_badge(label: str) -> str:
    l = label.lower().strip()
    if "low" in l:
        return "🟢 Low Risk"
    if "mid" in l or "medium" in l:
        return "🟡 Mid Risk"
    if "high" in l:
        return "🔴 High Risk"
    return f"🔎 {label}"


# =========================
# MAIN PAGE
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

    if pilihan == "Kesehatan (Maternal Health Risk)":
        _app_prediksi_maternal()
    else:
        _app_prediksi_occupancy()


# =========================
# APP: MATERNAL
# =========================
def _app_prediksi_maternal():
    st.title("🩺 Prediksi Risiko Kehamilan (Maternal Health Risk)")
    st.caption("Input indikator → prediksi tingkat risiko (low / mid / high)")

    bundle = _train_maternal_model()
    if bundle is None:
        return

    model = bundle["model"]
    features = bundle["features"]
    le = bundle["label_encoder"]

    st.subheader("📋 Input Indikator Pasien")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (Tahun)", min_value=10, max_value=80, value=25, step=1)
        systolic = st.number_input("SystolicBP", min_value=70.0, max_value=200.0, value=120.0, step=1.0)

    with col2:
        diastolic = st.number_input("DiastolicBP", min_value=40.0, max_value=140.0, value=80.0, step=1.0)
        bs = st.number_input("BS (Blood Sugar)", min_value=5.0, max_value=30.0, value=7.0, step=0.1)

    with col3:
        bodytemp = st.number_input("BodyTemp", min_value=90.0, max_value=110.0, value=98.0, step=0.1)
        heartrate = st.number_input("HeartRate", min_value=40, max_value=140, value=80, step=1)

    input_df = pd.DataFrame([[age, systolic, diastolic, bs, bodytemp, heartrate]], columns=features)

    if st.button("🔍 Prediksi Risiko Kehamilan"):
        probs = model.predict_proba(input_df)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = le.inverse_transform([pred_idx])[0]

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        colA, colB = st.columns(2)
        colA.metric("Prediksi Risiko", _maternal_badge(pred_label))
        colB.metric("Confidence (kelas terpilih)", f"{probs[pred_idx]:.2%}")

        st.subheader("🧾 Probabilitas Setiap Kelas")
        prob_df = pd.DataFrame({
            "Kelas": le.inverse_transform(np.arange(len(probs))),
            "Probabilitas": probs
        }).sort_values("Probabilitas", ascending=False).reset_index(drop=True)
        st.table(prob_df)

        st.subheader("💡 Catatan")
        st.markdown("""
- Hasil prediksi bersifat **edukatif** dan bergantung pada pola data latih.
- Untuk keputusan klinis, tetap diperlukan **konsultasi tenaga kesehatan**.
        """)


# =========================
# APP: OCCUPANCY
# =========================
def _app_prediksi_occupancy():
    st.title("🏢 Prediksi Status Ruangan (Occupancy Detection)")
    st.caption("Input indikator sensor → prediksi ruangan TERISI / KOSONG")

    bundle = _train_occupancy_model()
    if bundle is None:
        return

    model = bundle["model"]
    features = bundle["features"]

    st.subheader("📋 Input Indikator Sensor")
    col1, col2, col3 = st.columns(3)

    with col1:
        temp = st.number_input("Temperature", min_value=-10.0, max_value=60.0, value=23.0, step=0.1)
        humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=27.0, step=0.1)

    with col2:
        light = st.number_input("Light", min_value=0.0, max_value=2000.0, value=426.0, step=1.0)
        co2 = st.number_input("CO2", min_value=0.0, max_value=5000.0, value=720.0, step=1.0)

    with col3:
        hratio = st.number_input("HumidityRatio", min_value=0.0, max_value=1.0, value=0.0048, step=0.0001)
        hour = None
        if "Hour" in features:
            hour = st.number_input("Hour (opsional)", min_value=0, max_value=23, value=17, step=1)

    row = []
    for f in features:
        if f == "Temperature":
            row.append(temp)
        elif f == "Humidity":
            row.append(humidity)
        elif f == "Light":
            row.append(light)
        elif f == "CO2":
            row.append(co2)
        elif f == "HumidityRatio":
            row.append(hratio)
        elif f == "Hour":
            row.append(hour if hour is not None else 12)

    input_df = pd.DataFrame([row], columns=features)

    if st.button("🔍 Prediksi Status Ruangan"):
        prob_occupied = float(model.predict_proba(input_df)[0][1])
        pred = int(model.predict(input_df)[0])
        status = "TERISI (Occupied)" if pred == 1 else "KOSONG (Not Occupied)"

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        colA, colB = st.columns(2)
        colA.metric("Probabilitas Terisi", f"{prob_occupied:.2%}")
        colB.metric("Prediksi Status", status)

        st.subheader("💡 Catatan")
        st.markdown("""
- Prediksi bersifat **klasifikasi**, bukan forecasting waktu.
- Nilai sensor yang ekstrem dapat memengaruhi hasil prediksi.
        """)
