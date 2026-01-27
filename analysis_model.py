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
    # RANDOM FOREST (FULL PROSES + NARASI + RUMUS)
    # =========================================================
    if algo == "Random Forest":

        st.subheader("📌 Random Forest — Proses Lengkap & Bertahap")

        st.markdown("""
        Random Forest merupakan algoritma **ensemble learning** yang membangun
        banyak decision tree dan menggabungkan hasilnya melalui **voting mayoritas**.
        
        Pada bagian ini ditampilkan **proses matematis lengkap**, mulai dari
        bootstrap sampling hingga voting, **tanpa menampilkan hasil prediksi akhir**.
        """)

        st.markdown("---")

        # =========================
        # PARAMETER USER
        # =========================
        st.markdown("### 🔧 Pengaturan Parameter")

        n_trees = st.slider("Jumlah Decision Tree (T)", 3, 20, 5)
        sample_ratio = st.slider("Persentase Data Bootstrap (%)", 50, 100, 100)

        st.markdown("""
        Parameter di atas mengatur:
        - Jumlah decision tree yang dibangun
        - Proporsi data training yang diambil untuk setiap bootstrap
        """)

        st.markdown("---")

        # ==================================================
        # 1. BOOTSTRAP SAMPLING
        # ==================================================
        st.markdown("## ① Bootstrap Sampling")

        st.markdown("""
        Tahap bootstrap sampling bertujuan untuk membentuk dataset training baru
        dengan cara mengambil data **secara acak dengan pengembalian (with replacement)**.
        Dataset ini akan digunakan untuk membangun masing-masing decision tree.
        """)

        st.latex(r"D^* = \{x_i^* \mid x_i^* \sim D,\ i = 1,2,\dots,n\}")

        st.markdown("""
        **Keterangan simbol:**
        - \(D\) : dataset training asli  
        - \(D^*\) : dataset hasil bootstrap  
        - \(x_i^*\) : sampel hasil pengambilan acak  
        - \(n\) : jumlah data bootstrap  
        """)

        n = int(len(df) * sample_ratio / 100)

        st.markdown("**Perhitungan jumlah data bootstrap:**")
        st.latex(rf"n = {sample_ratio}\% \times {len(df)} = {n}")

        np.random.seed(42)
        bootstrap_indices = np.random.choice(df.index, size=n, replace=True)

        st.markdown("**Contoh 10 langkah pertama proses bootstrap:**")

        step_table = pd.DataFrame({
            "Langkah ke-": range(1, 11),
            "Index Terpilih": bootstrap_indices[:10],
            "With Replacement": ["Ya"] * 10
        })

        st.dataframe(step_table)

        st.markdown("""
        **Interpretasi:**  
        Karena sampling dilakukan dengan pengembalian, satu data dapat terpilih
        lebih dari satu kali dalam dataset bootstrap.
        """)

        st.markdown("---")

        # ==================================================
        # 2. PEMBENTUKAN DECISION TREE
        # ==================================================
        st.markdown("## ② Pembentukan Decision Tree")

        st.markdown("""
        Setelah dataset bootstrap terbentuk, langkah selanjutnya adalah membangun
        **decision tree**. Pemilihan split pada tree dilakukan dengan mengukur
        tingkat ketidakpastian data menggunakan **entropy**.
        """)

        st.latex(r"Entropy(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)")

        class_prop = df[target_col].value_counts(normalize=True)

        st.markdown("**Proporsi kelas pada dataset:**")
        st.dataframe(class_prop.to_frame("pᵢ"))

        subs = " + ".join(
            [f"{p:.4f} \\log_2({p:.4f})" for p in class_prop.values]
        )

        entropy_val = -(class_prop * np.log2(class_prop)).sum()

        st.markdown("**Substitusi nilai ke dalam rumus entropy:**")
        st.latex(rf"Entropy(S) = -({subs}) = {entropy_val:.4f}")

        st.markdown("""
        **Interpretasi:**  
        Nilai entropy menunjukkan tingkat ketidakpastian data sebelum dilakukan split.
        Semakin besar nilai entropy, semakin tidak homogen distribusi kelasnya.
        """)

        st.markdown("---")

        # ==================================================
        # 3. PREDIKSI SETIAP DECISION TREE
        # ==================================================
        st.markdown("## ③ Prediksi Tiap Decision Tree")

        st.markdown("""
        Setiap decision tree yang terbentuk menghasilkan **satu prediksi kelas**
        berdasarkan struktur pohonnya masing-masing.
        """)

        st.latex(r"\hat{y}_1, \hat{y}_2, \dots, \hat{y}_T")

        fake_preds = np.random.choice(df[target_col].unique(), size=n_trees)

        pred_table = pd.DataFrame({
            "Tree ke-": range(1, n_trees + 1),
            "Prediksi": fake_preds
        })

        st.dataframe(pred_table)

        st.markdown("""
        **Interpretasi:**  
        Karena setiap tree dibangun dari dataset yang berbeda, hasil prediksi
        antar tree dapat bervariasi.
        """)

        st.markdown("---")

        # ==================================================
        # 4. VOTING MAYORITAS
        # ==================================================
        st.markdown("## ④ Voting Mayoritas")

        st.markdown("""
        Tahap terakhir adalah menggabungkan seluruh prediksi decision tree
        menggunakan metode **voting mayoritas**.
        """)

        st.latex(r"\hat{y} = \arg\max_c \sum_{t=1}^{T} I(\hat{y}_t = c)")

        st.markdown("""
        **Keterangan simbol:**
        - \(T\) : jumlah decision tree  
        - \(\hat{y}_t\) : prediksi dari tree ke-\(t\)  
        - \(I(\cdot)\) : fungsi indikator  
        """)

        vote = pd.Series(fake_preds).value_counts()

        st.markdown("**Perhitungan jumlah suara untuk setiap kelas:**")
        st.dataframe(vote.to_frame("Jumlah Suara"))

        st.markdown("""
        **Interpretasi:**  
        Kelas dengan jumlah suara terbanyak akan dipilih sebagai hasil voting.
        Pada menu ini, hasil akhir **tidak ditampilkan** karena fokus pada proses.
        """)

        st.info(
            "Menu ini menampilkan **alur matematis Random Forest secara lengkap**. "
            "Prediksi akhir dapat dilihat pada menu **Prediction App**."
        )
