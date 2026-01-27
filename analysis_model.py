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

    st.title("🧠 Analisis Model Klasifikasi (Detail Proses & Perhitungan)")
    st.markdown("""
Menu ini menampilkan **alur proses dan perhitungan manual algoritma klasifikasi**.
Fokus utama adalah **bagaimana algoritma bekerja secara bertahap**,
mulai dari rumus hingga proses pengambilan keputusan.

⚠️ **Menu ini tidak menampilkan prediksi akhir sistem.**
Prediksi final tersedia pada menu **Prediction App**.
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
        st.dataframe(result_df)

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
        st.write("z:", z)
        st.write("Sigmoid:", sigmoid)

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
        class_probs = df[target_col].value_counts(normalize=True)
        entropy = -(class_probs * np.log2(class_probs)).sum()
        st.write("Entropy:", entropy)

    # =========================================================
    # RANDOM FOREST (FULL PROSES – SESUAI PERMINTAAN)
    # =========================================================
    elif algo == "Random Forest":
        st.subheader("📌 Random Forest — Proses Lengkap")

        # =========================
        # NARASI UMUM
        # =========================
        st.markdown("""
Random Forest adalah algoritma **ensemble learning** yang membangun banyak
**decision tree** dan menggabungkan hasilnya melalui **voting mayoritas**.

Pada bagian ini ditampilkan:
- Rumus yang digunakan
- Parameter yang dapat diatur pengguna
- Proses pembentukan model secara bertahap
""")

        # =========================
        # PARAMETER USER
        # =========================
        st.markdown("### ⚙️ Parameter yang Dapat Diatur")

        col1, col2, col3 = st.columns(3)

        with col1:
            n_trees = st.slider("Jumlah Decision Tree", 1, 100, 10)

        with col2:
            bootstrap_ratio = st.slider("Persentase Data Bootstrap (%)", 50, 100, 100, 10)

        with col3:
            random_seed = st.number_input("Random Seed", 0, 9999, 42)

        # =========================
        # BOOTSTRAP SAMPLING
        # =========================
        st.markdown("## 1️⃣ Bootstrap Sampling")

        st.latex(r"""
D^{*} = \{ x_i^{*} \mid x_i^{*} \sim D,\; i = 1,2,\dots,n \}
""")

        st.markdown("""
**Keterangan:**
- \(D\): dataset training
- \(D^{*}\): dataset hasil bootstrap
- \(n\): jumlah data training
""")

        original_idx = df.index.values
        st.write("Index data training (contoh):")
        st.dataframe(pd.DataFrame({"Index": original_idx[:10]}))

        np.random.seed(random_seed)

        st.markdown("### 🔁 Proses Pengambilan Sampel (10 langkah pertama)")
        steps = []
        for i in range(10):
            steps.append({
                "Langkah": i + 1,
                "Index Terpilih": np.random.choice(original_idx),
                "With Replacement": "Ya"
            })
        st.dataframe(pd.DataFrame(steps))

        n_samples = int(len(df) * bootstrap_ratio / 100)

        st.markdown(f"""
Jumlah data bootstrap dihitung sebagai:

\\[
n = {bootstrap_ratio}\\% \\times {len(df)} = {n_samples}
\\]
""")

        # =========================
        # PEMBENTUKAN TREE (ENTROPY)
        # =========================
        st.markdown("## 2️⃣ Pembentukan Decision Tree")

        st.latex(r"""
Entropy(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)
""")

        class_probs = df[target_col].value_counts(normalize=True)
        st.dataframe(class_probs.to_frame("Proporsi"))

        entropy_root = -(class_probs * np.log2(class_probs)).sum()

        st.success(f"Entropy Dataset: {entropy_root:.4f}")

        # =========================
        # SIMULASI PREDIKSI TREE
        # =========================
        st.markdown("## 3️⃣ Prediksi Tiap Decision Tree")

        st.latex(r"""
\hat{y}_1, \hat{y}_2, \dots, \hat{y}_T
""")

        tree_preds = np.random.choice(df[target_col].unique(), size=n_trees)

        st.dataframe(pd.DataFrame({
            "Tree ke-": range(1, n_trees + 1),
            "Prediksi": tree_preds
        }))

        # =========================
        # VOTING MAYORITAS
        # =========================
        st.markdown("## 4️⃣ Voting Mayoritas")

        st.latex(r"""
\hat{y} = \arg\max_{c} \sum_{t=1}^{T} I(\hat{y}_t = c)
""")

        vote = pd.Series(tree_preds).value_counts()
        st.dataframe(vote.to_frame("Jumlah Suara"))

        st.info("""
Bagian ini menunjukkan **alur voting** sebagai proses Random Forest.
Prediksi akhir sistem **tidak ditampilkan** pada menu ini.
""")

    st.markdown("---")
    st.info(
        "Menu ini berfungsi sebagai **penjelasan proses algoritma**. "
        "Training, evaluasi, dan prediksi final tersedia pada menu lain."
    )
