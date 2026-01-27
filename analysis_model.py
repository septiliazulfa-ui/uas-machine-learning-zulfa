import streamlit as st
import pandas as pd
import numpy as np


def analysis_model_page():

    # =========================
    # PENGAMAN DATASET
    # =========================
    if "df" not in st.session_state:
        st.warning("Silakan upload dataset terlebih dahulu.")
        return

    df = st.session_state["df"]
    dataset_type = st.session_state.get("dataset_type")
    target_col = "RiskLevel" if dataset_type == "Kesehatan" else "Occupancy"

    st.title("🧠 Analisis Model Klasifikasi (Proses & Perhitungan Detail)")
    st.markdown("""
    Halaman ini menampilkan **alur proses dan perhitungan matematis algoritma klasifikasi**.
    Fokus utama adalah **bagaimana model bekerja langkah demi langkah**, bukan hanya hasil akhir.

    ⚠️ Prediksi final tersedia pada menu **Prediction App**.
    """)

    st.markdown("---")

    algo = st.selectbox(
        "Pilih Algoritma",
        ["Random Forest"]
    )

    # =========================================================
    # RANDOM FOREST
    # =========================================================
    if algo == "Random Forest":

        st.subheader("📌 Random Forest — Proses Lengkap & Bertahap")

        st.markdown("""
        **Random Forest** adalah algoritma *ensemble learning* yang membangun
        banyak **decision tree** dari data yang berbeda-beda,
        lalu menggabungkan hasilnya menggunakan **voting mayoritas**.
        """)

        st.markdown("---")

        # =========================
        # PARAMETER
        # =========================
        st.markdown("### 🔧 Pengaturan Parameter")

        n_trees = st.slider("Jumlah Decision Tree (T)", 3, 20, 5)
        sample_ratio = st.slider("Persentase Data Bootstrap (%)", 50, 100, 100)

        st.markdown("""
        **Keterangan Parameter:**
        - **T** : jumlah decision tree dalam Random Forest
        - **Bootstrap (%)** : proporsi data yang diambil untuk setiap tree
        """)

        st.markdown("---")

        # ==================================================
        # ① BOOTSTRAP SAMPLING
        # ==================================================
        st.markdown("## ① Bootstrap Sampling")

        st.markdown("""
        Tahap ini bertujuan untuk membuat **dataset yang berbeda**
        bagi setiap decision tree menggunakan teknik **sampling dengan pengembalian (with replacement)**.
        """)

        st.latex(r"D^* = \{x_i^* \mid x_i^* \sim D\}")

        st.markdown("""
        **Keterangan simbol:**
        - \(D\) : dataset asli
        - \(D^*\) : dataset hasil bootstrap
        - \(x_i^*\) : data terpilih secara acak
        """)

        n = int(len(df) * sample_ratio / 100)

        st.markdown("### 🔹 Menentukan jumlah data bootstrap")
        st.latex(rf"n = {sample_ratio}\% \times {len(df)} = {n}")

        np.random.seed(42)
        bootstrap_indices = np.random.choice(df.index, size=n, replace=True)

        st.markdown("### 🔹 Contoh proses pengambilan data")
        st.dataframe(pd.DataFrame({
            "Langkah ke-": range(1, 11),
            "Index Terpilih": bootstrap_indices[:10],
            "With Replacement": ["Ya"] * 10
        }))

        st.info("""
        Karena **with replacement**, satu data bisa muncul lebih dari satu kali
        atau tidak muncul sama sekali dalam satu tree.
        """)

        st.markdown("---")

        # ==================================================
        # ② ENTROPY DATASET AWAL
        # ==================================================
        st.markdown("## ② Pembentukan Decision Tree (Entropy Awal)")

        st.markdown("""
        Sebelum melakukan split, decision tree menghitung **entropy awal**
        untuk mengukur **ketidakpastian distribusi kelas target**.
        """)

        st.latex(r"Entropy(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)")

        st.markdown("""
        **Keterangan simbol:**
        - \(S\) : seluruh dataset training
        - \(c\) : jumlah kelas
        - \(n_i\) : jumlah data pada kelas ke-\(i\)
        - \(N\) : total seluruh data
        - \(p_i = \frac{n_i}{N}\)
        """)

        # Hitung simbol satu per satu
        class_counts = df[target_col].value_counts()
        N = class_counts.sum()

        rows = []
        subs = []
        entropy_S = 0

        for cls, ni in class_counts.items():
            pi = ni / N
            rows.append([cls, ni, f"{ni}/{N}", round(pi, 4)])
            subs.append(rf"\frac{{{ni}}}{{{N}}}\log_2({round(pi,4)})")
            entropy_S += -pi * np.log2(pi)

        st.markdown("### 🔹 Menghitung nilai setiap simbol")
        st.dataframe(pd.DataFrame(
            rows,
            columns=["Kelas", "nᵢ", "Substitusi", "pᵢ"]
        ))

        st.markdown("### 🔹 Substitusi ke rumus entropy")
        st.latex(r"Entropy(S) = -(" + " + ".join(subs) + ")")
        st.latex(rf"Entropy(S) = {round(entropy_S,4)}")

        st.success("""
        Nilai entropy ini menjadi **acuan awal**
        sebelum proses split dilakukan.
        """)

        st.markdown("---")

        # ==================================================
        # ③ SPLIT DATA
        # ==================================================
        st.markdown("## ③ Proses Split Data")

        st.markdown("""
        Pada tahap ini, decision tree mencari **fitur dan threshold**
        yang menghasilkan **Information Gain terbesar**.
        """)

        numeric_features = df.select_dtypes(include="number").columns.drop(target_col)
        split_feature = st.selectbox("Pilih fitur", numeric_features)

        min_val, max_val = float(df[split_feature].min()), float(df[split_feature].max())
        threshold = st.slider("Threshold", min_val, max_val, (min_val + max_val) / 2)

        st.latex(rf"{split_feature} \le {round(threshold,2)}")

        left = df[df[split_feature] <= threshold]
        right = df[df[split_feature] > threshold]

        def entropy_subset(sub):
            probs = sub[target_col].value_counts(normalize=True)
            return -(probs * np.log2(probs)).sum()

        entropy_left = entropy_subset(left)
        entropy_right = entropy_subset(right)

        weighted_entropy = (
            len(left)/len(df) * entropy_left +
            len(right)/len(df) * entropy_right
        )

        IG = entropy_S - weighted_entropy

        st.latex(rf"Entropy_{{split}} = {round(weighted_entropy,4)}")
        st.latex(rf"IG = {round(entropy_S,4)} - {round(weighted_entropy,4)} = {round(IG,4)}")

        st.markdown("""
        **Interpretasi:**  
        Semakin besar **Information Gain**, semakin baik split tersebut.
        """)

        st.markdown("---")

        # ==================================================
        # ④ PREDIKSI SETIAP TREE
        # ==================================================
        st.markdown("## ④ Prediksi Tiap Decision Tree")

        st.latex(r"\hat{y}_1, \hat{y}_2, ..., \hat{y}_T")

        fake_preds = np.random.choice(df[target_col].unique(), size=n_trees)

        st.dataframe(pd.DataFrame({
            "Tree ke-": range(1, n_trees + 1),
            "Prediksi": fake_preds
        }))

        st.markdown("---")

        # ==================================================
        # ⑤ VOTING MAYORITAS
        # ==================================================
        st.markdown("## ⑤ Voting Mayoritas")

        st.latex(r"\hat{y} = \arg\max_c \sum_{t=1}^{T} I(\hat{y}_t = c)")

        vote = pd.Series(fake_preds).value_counts()
        st.dataframe(vote.to_frame("Jumlah Suara"))

        st.success("""
        Kelas dengan suara terbanyak menjadi **prediksi akhir Random Forest**.
        """)

        st.info("Menu ini fokus pada **alur & perhitungan**, bukan hasil akhir.")
