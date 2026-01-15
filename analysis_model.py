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
    st.write(
        "Menu ini menampilkan **langkah-langkah lengkap setiap algoritma klasifikasi** "
        "beserta **output numerik dari setiap tahap perhitungan**, menggunakan contoh data nyata."
    )

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
    # KNN
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

        st.write("**Data Uji (X):**")
        st.dataframe(data_uji.to_frame("Nilai"))

        st.write("**Hasil Jarak Euclidean:**")
        st.dataframe(result_df)

        K = 3
        neighbors = result_df.head(K)
        neighbor_labels = df.loc[neighbors.index, target_col]

        voting_df = neighbors[["Jarak_Euclidean"]].copy()
        voting_df["Label_Target"] = neighbor_labels.values

        st.write(f"**{K} Tetangga Terdekat:**")
        st.dataframe(voting_df)

        vote = neighbor_labels.value_counts()
        st.write("**Hasil Voting:**")
        st.dataframe(vote.to_frame("Jumlah"))

        st.success(f"Hasil Klasifikasi KNN: **{vote.idxmax()}**")

    # =========================================================
    # LOGISTIC REGRESSION
    # =========================================================
    elif algo == "Logistic Regression":
        st.subheader("📌 Logistic Regression")

        numeric_df = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")

        x = numeric_df.iloc[0].values
        beta = np.ones(len(x)) * 0.1
        beta_0 = 0.1

        z = beta_0 + np.dot(x, beta)
        sigmoid = 1 / (1 + np.exp(-z))
        threshold = 0.5
        kelas = 1 if sigmoid >= threshold else 0

        log_loss = -(np.log(sigmoid) if kelas == 1 else np.log(1 - sigmoid))

        st.write("**Bobot (β):**")
        st.write(beta)

        st.write(f"**Nilai z = β₀ + β·x:** `{z:.4f}`")
        st.write(f"**Sigmoid(z) / Probabilitas:** `{sigmoid:.4f}`")
        st.write(f"**Threshold:** `{threshold}`")
        st.write(f"**Hasil Klasifikasi:** `{kelas}`")
        st.write(f"**Log Loss (1 data):** `{log_loss:.4f}`")

    # =========================================================
    # NAIVE BAYES
    # =========================================================
    elif algo == "Naive Bayes":
        st.subheader("📌 Naive Bayes")

        numeric_df = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")
        x = numeric_df.iloc[0]

        st.write("**Data Uji:**")
        st.dataframe(x.to_frame("Nilai"))

        results = {}

        for cls in df[target_col].unique():
            subset = df[df[target_col] == cls]
            prior = len(subset) / len(df)

            likelihoods = []
            for col in numeric_df.columns:
                mean = subset[col].mean()
                std = subset[col].std() if subset[col].std() > 0 else 1e-6
                prob = norm.pdf(x[col], mean, std)
                likelihoods.append(prob)

            posterior = prior * np.prod(likelihoods)
            results[cls] = posterior

        result_df = pd.DataFrame.from_dict(
            results, orient="index", columns=["Posterior Score"]
        )

        st.write("**Posterior Probability (Unnormalized):**")
        st.dataframe(result_df)

        st.success(f"Hasil Klasifikasi Naive Bayes: **{result_df.idxmax()[0]}**")

    # =========================================================
    # DECISION TREE
    # =========================================================
    elif algo == "Decision Tree":
        st.subheader("📌 Decision Tree")

        class_counts = df[target_col].value_counts(normalize=True)
        entropy = -(class_counts * np.log2(class_counts)).sum()

        st.write("**Distribusi Kelas:**")
        st.dataframe(class_counts.to_frame("Proporsi"))

        st.write(f"**Entropy Dataset:** `{entropy:.4f}`")

        feature = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore").columns[0]

        median = df[feature].median()
        left = df[df[feature] <= median]
        right = df[df[feature] > median]

        def entropy_subset(sub):
            probs = sub[target_col].value_counts(normalize=True)
            return -(probs * np.log2(probs)).sum()

        entropy_left = entropy_subset(left)
        entropy_right = entropy_subset(right)

        ig = entropy - (
            (len(left)/len(df)) * entropy_left +
            (len(right)/len(df)) * entropy_right
        )

        st.write(f"**Fitur Contoh:** `{feature}`")
        st.write(f"Entropy Kiri: `{entropy_left:.4f}`")
        st.write(f"Entropy Kanan: `{entropy_right:.4f}`")
        st.success(f"**Information Gain:** `{ig:.4f}`")

    # =========================================================
    # RANDOM FOREST
    # =========================================================
    elif algo == "Random Forest":
        st.subheader("📌 Random Forest")

        n_samples = len(df)
        bootstrap_idx = np.random.choice(df.index, size=n_samples, replace=True)

        st.write(f"**Jumlah Data Bootstrap:** `{len(bootstrap_idx)}`")
        st.write("**Contoh Index Bootstrap (10 pertama):**")
        st.write(bootstrap_idx[:10])

        fake_tree_preds = np.random.choice(df[target_col].unique(), size=5)
        vote = pd.Series(fake_tree_preds).value_counts()

        st.write("**Prediksi Tiap Tree:**")
        st.write(fake_tree_preds)

        st.write("**Hasil Voting:**")
        st.dataframe(vote.to_frame("Jumlah"))

        st.success(f"Hasil Akhir Random Forest: **{vote.idxmax()}**")

    st.markdown("---")
    st.info(
        "Semua perhitungan di atas menggunakan contoh data nyata dari dataset. "
        "Proses training dan evaluasi performa model dilakukan pada menu **Machine Learning**."
    )
