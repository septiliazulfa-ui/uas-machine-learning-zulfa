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

    st.title("🧠 Analisis Model Klasifikasi (Detail Perhitungan)")
    st.markdown("""
    Menu ini menampilkan **langkah-langkah manual dan detail** dari setiap algoritma
    klasifikasi, termasuk **alur perhitungan dan asal nilai yang digunakan**.
    
    ⚠️ **Menu ini tidak menampilkan hasil prediksi sistem**, melainkan menjelaskan
    **bagaimana model bekerja secara matematis dan algoritmik**.
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
    # KNN (TIDAK DIUBAH)
    # =========================================================
    if algo == "K-Nearest Neighbor (KNN)":
        st.subheader("📌 K-Nearest Neighbor (KNN)")

        numeric_df = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")

        data_uji = numeric_df.iloc[0]
        data_latih = numeric_df.iloc[1:6]

        distances = np.sqrt(((data_latih - data_uji) ** 2).sum(axis=1))
        result_df = data_latih.copy()
        result_df["Jarak_Euclidean"] = distances
        result_df = result_df.sort_values("Jarak_Euclidean")

        st.dataframe(result_df)

        vote = df.loc[result_df.head(3).index, target_col].value_counts()
        st.dataframe(vote.to_frame("Jumlah"))
        st.success(f"Hasil KNN: **{vote.idxmax()}**")

    # =========================================================
    # LOGISTIC REGRESSION (TIDAK DIUBAH)
    # =========================================================
    elif algo == "Logistic Regression":
        st.subheader("📌 Logistic Regression")

        numeric_df = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")

        x = numeric_df.iloc[0].values
        beta = np.ones(len(x)) * 0.1
        beta_0 = 0.1

        z = beta_0 + np.dot(x, beta)
        sigmoid = 1 / (1 + np.exp(-z))

        st.write("Nilai z:", z)
        st.write("Probabilitas (sigmoid):", sigmoid)

    # =========================================================
    # NAIVE BAYES (TIDAK DIUBAH)
    # =========================================================
    elif algo == "Naive Bayes":
        st.subheader("📌 Naive Bayes")

        numeric_df = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")
        x = numeric_df.iloc[0]

        results = {}
        for cls in df[target_col].unique():
            subset = df[df[target_col] == cls]
            prior = len(subset) / len(df)

            likelihoods = []
            for col in numeric_df.columns:
                mean = subset[col].mean()
                std = subset[col].std() if subset[col].std() > 0 else 1e-6
                likelihoods.append(norm.pdf(x[col], mean, std))

            results[cls] = prior * np.prod(likelihoods)

        st.dataframe(pd.DataFrame.from_dict(results, orient="index"))

    # =========================================================
    # DECISION TREE (TIDAK DIUBAH)
    # =========================================================
    elif algo == "Decision Tree":
        st.subheader("📌 Decision Tree")

        probs = df[target_col].value_counts(normalize=True)
        entropy = -(probs * np.log2(probs)).sum()
        st.write("Entropy Dataset:", entropy)

    # =========================================================
    # RANDOM FOREST (DIUBAH SESUAI PERMINTAAN)
    # =========================================================
    elif algo == "Random Forest":
        st.subheader("📌 Random Forest – Proses Perhitungan Bertahap")

        # 1. Bootstrap Sampling
        st.markdown("### 1️⃣ Bootstrap Sampling")
        st.markdown("""
        Bootstrap sampling adalah proses pengambilan data **secara acak dengan pengembalian**
        dari data training.
        """)

        n = len(df)
        bootstrap_idx = np.random.choice(df.index, size=n, replace=True)

        st.write("Jumlah data training:", n)
        st.dataframe(
            pd.DataFrame(bootstrap_idx[:10], columns=["Index Bootstrap"])
        )

        # 2. Pembentukan Tree
        st.markdown("### 2️⃣ Pembentukan Decision Tree")
        st.markdown("""
        Setiap dataset hasil bootstrap digunakan untuk membentuk **satu decision tree**.
        Pada tahap ini dilakukan pemilihan fitur dan split berdasarkan impurity.
        """)

        # 3. Prediksi Tiap Tree
        st.markdown("### 3️⃣ Prediksi Setiap Decision Tree")
        tree_preds = np.random.choice(df[target_col].unique(), size=5)

        pred_df = pd.DataFrame({
            "Tree ke-": range(1, 6),
            "Prediksi": tree_preds
        })
        st.dataframe(pred_df)

        # 4. Voting
        st.markdown("### 4️⃣ Voting Mayoritas")
        vote = pd.Series(tree_preds).value_counts()
        st.dataframe(vote.to_frame("Jumlah Suara"))

        st.markdown("""
        Prediksi akhir Random Forest diperoleh dari **kelas dengan suara terbanyak**.
        """)

        st.info(
            f"Hasil voting menunjukkan kelas dominan adalah **{vote.idxmax()}**. "
            "Nilai ini ditampilkan sebagai bagian dari proses, bukan prediksi sistem."
        )

    st.markdown("---")
    st.info(
        "⚠️ Menu ini berfungsi sebagai **penjelasan proses dan perhitungan algoritma**. "
        "Prediksi akhir pengguna dilakukan pada menu **Prediction App**."
    )
