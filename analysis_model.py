import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm


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
    Fokus utama adalah **bagaimana setiap algoritma bekerja**, bukan sekadar hasil akhir.
    
    ⚠️ Prediksi final tersedia pada menu **Prediction App**.
    """)

    st.markdown("---")

    algo = st.selectbox(
        "Pilih Algoritma",
        [
            "K-Nearest Neighbor (KNN)",
            "Logistic Regression",
            "Naive Bayes",
            "Decision Tree",
            "Random Forest"
        ]
    )

    st.markdown("---")

    # =========================================================
    # RANDOM FOREST
    # =========================================================
    if algo == "Random Forest":

        st.subheader("📌 Random Forest — Proses Lengkap & Bertahap")

        st.markdown("""
        Random Forest merupakan algoritma **ensemble learning** yang membangun
        banyak decision tree dan menggabungkan hasilnya melalui **voting mayoritas**.
        """)

        st.markdown("---")

        # =========================
        # PARAMETER USER
        # =========================
        st.markdown("### 🔧 Pengaturan Parameter")

        n_trees = st.slider("Jumlah Decision Tree (T)", 3, 20, 5)
        sample_ratio = st.slider("Persentase Data Bootstrap (%)", 50, 100, 100)

        st.markdown("---")

        # ==================================================
        # 1. BOOTSTRAP SAMPLING
        # ==================================================
        st.markdown("## ① Bootstrap Sampling")

        st.latex(r"D^* = \{x_i^* \mid x_i^* \sim D\}")

        n = int(len(df) * sample_ratio / 100)
        st.latex(rf"n = {sample_ratio}\% \times {len(df)} = {n}")

        np.random.seed(42)
        bootstrap_indices = np.random.choice(df.index, size=n, replace=True)

        step_table = pd.DataFrame({
            "Langkah ke-": range(1, 11),
            "Index Terpilih": bootstrap_indices[:10],
            "With Replacement": ["Ya"] * 10
        })

        st.dataframe(step_table)

        st.markdown("---")

        # ==================================================
        # 2. PEMBENTUKAN DECISION TREE (ENTROPY — FULL PROSES)
        # ==================================================
        st.markdown("## ② Pembentukan Decision Tree")

        st.markdown("""
        Setelah dataset bootstrap terbentuk, decision tree dibangun dengan
        memilih split terbaik berdasarkan **entropy**.
        """)

        # 🔹 RUMUS ENTROPY
        st.latex(r"Entropy(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)")

        st.markdown("""
        **Keterangan simbol:**
        - \(S\) : dataset
        - \(c\) : jumlah kelas
        - \(p_i\) : probabilitas kelas ke-\(i\)
        """)

        # =========================
        # LANGKAH 1: JUMLAH DATA PER KELAS
        # =========================
        st.markdown("### 🔹 Langkah 1: Jumlah Data per Kelas")

        class_counts = df[target_col].value_counts().sort_index()
        N = class_counts.sum()

        df_count = pd.DataFrame({
            "Kelas": class_counts.index,
            "Jumlah Data (nᵢ)": class_counts.values
        })

        st.dataframe(df_count, use_container_width=True)
        st.latex(rf"N = {N}")

        # =========================
        # LANGKAH 2: RUMUS PROBABILITAS
        # =========================
        st.markdown("### 🔹 Langkah 2: Rumus Probabilitas")

        st.latex(r"p_i = \frac{n_i}{N}")

        # =========================
        # LANGKAH 3: SUBSTITUSI & p_i (REVISI)
        # =========================
        st.markdown("### 🔹 Langkah 3: Substitusi Angka & Perhitungan $p_i$")

        # Rumus probabilitas ditampilkan TERPISAH
        st.latex(r"p_i = \frac{n_i}{N}")

        prob_rows = []
        for cls, ni in class_counts.items():
            pi = ni / N
            prob_rows.append([
                cls,
                ni,
                f"{ni}/{N}",
                round(pi, 4)
            ])

        df_prob = pd.DataFrame(
            prob_rows,
            columns=[
                "Kelas",
                "Jumlah Data (nᵢ)",
                "Substitusi Angka",
                "Nilai pᵢ"
            ]
        )

        st.dataframe(df_prob, use_container_width=True)

        # =========================
        # LANGKAH 4: SUBSTITUSI ENTROPY
        # =========================
        st.markdown("### 🔹 Langkah 4: Substitusi ke Rumus Entropy")

        entropy_value = 0
        subs = []

        for pi in df_prob["Nilai pᵢ"]:
            subs.append(f"{pi} \\log_2({pi})")
            entropy_value += -pi * np.log2(pi)

        st.latex(
            r"Entropy(S) = -(" + " + ".join(subs) + ")"
        )

        st.latex(
            r"Entropy(S) = " + str(round(entropy_value, 4))
        )

        st.markdown("""
        **Interpretasi:**  
        Nilai entropy menunjukkan tingkat ketidakpastian distribusi kelas
        sebelum dilakukan split pada decision tree.
        """)

        st.markdown("---")

        # ==================================================
        # 3. PREDIKSI SETIAP DECISION TREE
        # ==================================================
        st.markdown("## ③ Prediksi Tiap Decision Tree")

        st.latex(r"\hat{y}_1, \hat{y}_2, \dots, \hat{y}_T")

        fake_preds = np.random.choice(df[target_col].unique(), size=n_trees)

        st.dataframe(pd.DataFrame({
            "Tree ke-": range(1, n_trees + 1),
            "Prediksi": fake_preds
        }))

        st.markdown("---")

        # ==================================================
        # 4. VOTING MAYORITAS
        # ==================================================
        st.markdown("## ④ Voting Mayoritas")

        st.latex(r"\hat{y} = \arg\max_c \sum_{t=1}^{T} I(\hat{y}_t = c)")

        vote = pd.Series(fake_preds).value_counts()
        st.dataframe(vote.to_frame("Jumlah Suara"))

        st.info("Fokus menu ini adalah **proses matematis**, bukan hasil akhir.")

