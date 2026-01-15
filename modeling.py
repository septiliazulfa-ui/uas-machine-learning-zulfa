import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def modeling_page():

    # =========================
    # PENGAMAN
    # =========================
    if "df" not in st.session_state:
        st.warning("Silakan upload dataset terlebih dahulu.")
        return

    df = st.session_state["df"]
    dataset_type = st.session_state.get("dataset_type")

    st.title("🤖 Machine Learning")
    st.write(
        "Halaman ini digunakan untuk melatih dan membandingkan "
        "beberapa algoritma klasifikasi menggunakan dataset yang di-upload."
    )

    # =========================
    # TENTUKAN TARGET
    # =========================
    if dataset_type == "Lingkungan":
        target_col = "Occupancy"
    elif dataset_type == "Kesehatan":
        target_col = "RiskLevel"
    else:
        st.error("Jenis dataset tidak dikenali.")
        return

    st.write(f"**Target Klasifikasi:** `{target_col}`")

    # =========================
    # PREPROCESSING AMAN
    # =========================
    df_model = df.copy()

    # DROP kolom waktu (tidak dipakai ML)
    if "date" in df_model.columns:
        df_model = df_model.drop(columns=["date"])

    # Encode target jika kategorikal
    if df_model[target_col].dtype == "object":
        le = LabelEncoder()
        df_model[target_col] = le.fit_transform(df_model[target_col])
        st.session_state["label_encoder"] = le

    # Pisahkan X dan y
    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    # =========================
    # KONVERSI NUMERIK PAKSA
    # =========================
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # HAPUS DATA RUSAK
    before = len(X)
    X["target"] = y
    X = X.dropna()
    y = X["target"]
    X = X.drop(columns=["target"])
    after = len(X)

    st.info(f"🧹 Data dibersihkan: {before - after} baris dibuang")

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================
    # SCALING
    # =========================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    st.session_state["scaler"] = scaler
    st.session_state["feature_columns"] = X.columns.tolist()

    # =========================
    # MODEL
    # =========================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    results = []
    best_model = None
    best_f1 = 0
    best_model_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append({
            "Algoritma": name,
            "Accuracy": round(acc * 100, 2),
            "Precision": round(prec * 100, 2),
            "Recall": round(rec * 100, 2),
            "F1-Score": round(f1 * 100, 2),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results)

    # =========================
    # OUTPUT
    # =========================
    st.subheader("📊 Hasil Evaluasi Model")
    st.dataframe(results_df, use_container_width=True)

    st.success(f"🏆 Model Terbaik: **{best_model_name}** (F1-Score tertinggi)")

    st.session_state["best_model"] = best_model

    st.info(
        "Model terbaik telah disimpan dan akan digunakan "
        "pada menu **Prediction App**."
    )
