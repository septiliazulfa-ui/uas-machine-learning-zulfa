import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm

def analysis_model_page():
    st.title("🧠 Analisis Model Klasifikasi (Definisi, Narasi, Rumus & Keterangan)")

    st.markdown("""
    Halaman ini berisi **penjelasan teori** untuk beberapa algoritma klasifikasi:
    **KNN, Logistic Regression, Naive Bayes, Decision Tree, dan Random Forest**.

    ✅ **Tidak wajib upload dataset** untuk membuka menu ini.  
    Kamu cukup memilih algoritma, lalu akan tampil **definisi, narasi, rumus, dan keterangan simbol (dalam LaTeX)**.
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
    # KNN
    # =========================================================
    if algo == "K-Nearest Neighbor (KNN)":
        st.subheader("📌 K-Nearest Neighbor (KNN)")

        st.markdown("## Definisi")
        st.markdown("""
        **K-Nearest Neighbor (KNN)** adalah algoritma klasifikasi berbasis jarak (distance-based).
        KNN menentukan kelas sebuah data baru berdasarkan **mayoritas kelas** dari **K tetangga terdekat**
        pada data latih.
        """)

        st.markdown("## Narasi Cara Kerja")
        st.markdown("""
        1. Tentukan nilai **K** (jumlah tetangga).
        2. Hitung jarak data uji terhadap semua data latih.
        3. Urutkan jarak, pilih **K** data dengan jarak terkecil.
        4. Prediksi kelas = **voting mayoritas** dari kelas tetangga terdekat.
        """)

        st.markdown("## Rumus Jarak (Euclidean Distance)")
        st.latex(r"d(x,x_i)=\sqrt{\sum_{j=1}^{p}(x_j-x_{ij})^2}")

        st.markdown("### Keterangan Simbol")
        st.latex(r"""
        \begin{aligned}
        x &:\ \text{data uji} \\
        x_i &:\ \text{data latih ke-}i \\
        p &:\ \text{jumlah fitur} \\
        x_j &:\ \text{fitur ke-}j \text{ pada data uji} \\
        x_{ij} &:\ \text{fitur ke-}j \text{ pada data latih ke-}i
        \end{aligned}
        """)

        st.markdown("## Rumus Prediksi (Majority Voting)")
        st.latex(r"\hat{y}=\text{mode}\{y_{(1)},y_{(2)},\dots,y_{(K)}\}")

        st.markdown("### Keterangan Simbol")
        st.latex(r"""
        \begin{aligned}
        \hat{y} &:\ \text{kelas prediksi} \\
        y_{(k)} &:\ \text{label dari tetangga terdekat ke-}k \\
        K &:\ \text{jumlah tetangga terdekat}
        \end{aligned}
        """)

        st.info("Catatan: KNN sensitif terhadap skala fitur, sehingga normalisasi/standarisasi biasanya diperlukan.")

    # =========================================================
    # LOGISTIC REGRESSION
    # =========================================================
    elif algo == "Logistic Regression":
        st.subheader("📌 Logistic Regression")

        st.markdown("## Definisi")
        st.markdown("""
        **Logistic Regression** adalah algoritma klasifikasi yang memodelkan probabilitas suatu kelas
        menggunakan fungsi **sigmoid**. Pada klasifikasi biner, model menghasilkan probabilitas \(P(y=1|x)\).
        """)

        st.markdown("## Narasi Cara Kerja")
        st.markdown("""
        1. Hitung kombinasi linear fitur: \(z=w^Tx+b\)  
        2. Ubah menjadi probabilitas dengan sigmoid: \(\sigma(z)\)  
        3. Bandingkan probabilitas dengan threshold (umumnya 0.5) untuk menentukan kelas
        """)

        st.markdown("## Rumus Model Linear")
        st.latex(r"z=w^Tx+b")

        st.markdown("### Keterangan Simbol")
        st.latex(r"""
        \begin{aligned}
        x &:\ \text{vektor fitur} \\
        w &:\ \text{vektor bobot/koefisien} \\
        b &:\ \text{bias} \\
        z &:\ \text{skor linear}
        \end{aligned}
        """)

        st.markdown("## Rumus Sigmoid")
        st.latex(r"\sigma(z)=\frac{1}{1+e^{-z}}")

        st.markdown("### Keterangan Simbol")
        st.latex(r"""
        \begin{aligned}
        \sigma(z) &:\ \text{probabilitas kelas positif} \\
        e &:\ \text{bilangan eksponensial} \\
        z &:\ \text{skor linear}
        \end{aligned}
        """)

        st.markdown("## Aturan Keputusan (Binary)")
        st.latex(r"""
        \hat{y}=
        \begin{cases}
        1 & \text{jika } \sigma(z)\ge 0.5\\
        0 & \text{jika } \sigma(z)<0.5
        \end{cases}
        """)

        st.markdown("## Loss Function (Binary Cross Entropy)")
        st.latex(r"""
        \mathcal{L}=-\frac{1}{n}\sum_{i=1}^{n}
        \left[y_i\log(\hat{p}_i)+(1-y_i)\log(1-\hat{p}_i)\right]
        """)

        st.markdown("### Keterangan Simbol")
        st.latex(r"""
        \begin{aligned}
        n &:\ \text{jumlah data} \\
        y_i &:\ \text{label aktual data ke-}i \\
        \hat{p}_i &:\ \text{probabilitas prediksi data ke-}i \\
        \mathcal{L} &:\ \text{nilai loss}
        \end{aligned}
        """)

    # =========================================================
    # NAIVE BAYES
    # =========================================================
    elif algo == "Naive Bayes":
        st.subheader("📌 Naive Bayes")

        st.markdown("## Definisi")
        st.markdown("""
        **Naive Bayes** adalah algoritma klasifikasi probabilistik berbasis **Teorema Bayes**,
        dengan asumsi “naive” bahwa fitur-fitur bersifat **independen bersyarat** terhadap kelas.
        """)

        st.markdown("## Narasi Cara Kerja")
        st.markdown("""
        1. Hitung prior kelas \(P(C)\)  
        2. Hitung likelihood fitur \(P(x_j|C)\)  
        3. Hitung posterior \(P(C|x)\)  
        4. Pilih kelas dengan posterior terbesar
        """)

        st.markdown("## Rumus Teorema Bayes")
        st.latex(r"P(C|x)=\frac{P(x|C)\,P(C)}{P(x)}")

        st.markdown("### Keterangan Simbol")
        st.latex(r"""
        \begin{aligned}
        C &:\ \text{kelas} \\
        x &:\ \text{data fitur} \\
        P(C) &:\ \text{prior kelas} \\
        P(x|C) &:\ \text{likelihood} \\
        P(C|x) &:\ \text{posterior} \\
        P(x) &:\ \text{evidence (konstanta normalisasi)}
        \end{aligned}
        """)

        st.markdown("## Asumsi Independensi (Naive)")
        st.latex(r"P(x|C)=\prod_{j=1}^{p}P(x_j|C)")

        st.markdown("### Keterangan Simbol")
        st.latex(r"""
        \begin{aligned}
        p &:\ \text{jumlah fitur} \\
        x_j &:\ \text{fitur ke-}j \\
        \prod &:\ \text{operasi perkalian berulang}
        \end{aligned}
        """)

        st.markdown("## Gaussian Naive Bayes (untuk fitur numerik)")
        st.latex(r"""
        P(x_j|C)=\frac{1}{\sqrt{2\pi\sigma_C^2}}
        \exp\left(-\frac{(x_j-\mu_C)^2}{2\sigma_C^2}\right)
        """)

        st.markdown("### Keterangan Simbol")
        st.latex(r"""
        \begin{aligned}
        \mu_C &:\ \text{rata-rata fitur pada kelas }C \\
        \sigma_C^2 &:\ \text{variansi fitur pada kelas }C \\
        \pi &:\ \text{konstanta pi} \\
        \exp(\cdot) &:\ \text{fungsi eksponensial}
        \end{aligned}
        """)

    # =========================================================
    # DECISION TREE
    # =========================================================
    elif algo == "Decision Tree":
        st.subheader("📌 Decision Tree")

        st.markdown("## Definisi")
        st.markdown("""
        **Decision Tree** adalah algoritma klasifikasi yang memisahkan data ke dalam beberapa subset
        berdasarkan aturan split pada fitur tertentu, sehingga membentuk struktur pohon keputusan.
        """)

        st.markdown("## Narasi Cara Kerja")
        st.markdown("""
        1. Hitung impurity awal (Entropy atau Gini).  
        2. Uji berbagai kemungkinan split pada fitur.  
        3. Pilih split terbaik yang paling menurunkan impurity (IG terbesar atau Gini terkecil).  
        4. Ulangi proses sampai memenuhi kondisi berhenti.
        """)

        st.markdown("## Rumus Entropy")
        st.latex(r"Entropy(S)=-\sum_{i=1}^{c}p_i\log_2(p_i)")

        st.markdown("### Keterangan Simbol")
        st.latex(r"""
        \begin{aligned}
        S &:\ \text{data pada node} \\
        c &:\ \text{jumlah kelas} \\
        p_i &:\ \text{proporsi kelas ke-}i \\
        \log_2(\cdot) &:\ \text{logaritma basis 2}
        \end{aligned}
        """)

        st.markdown("## Rumus Information Gain")
        st.latex(r"IG=Entropy(S)-\sum_{j=1}^{k}\frac{|S_j|}{|S|}Entropy(S_j)")

        st.markdown("### Keterangan Simbol")
        st.latex(r"""
        \begin{aligned}
        IG &:\ \text{Information Gain} \\
        S &:\ \text{dataset induk} \\
        S_j &:\ \text{subset hasil split ke-}j \\
        |S| &:\ \text{jumlah data pada }S \\
        |S_j| &:\ \text{jumlah data pada }S_j \\
        k &:\ \text{jumlah subset}
        \end{aligned}
        """)

        st.markdown("## Rumus Gini Index (Alternatif)")
        st.latex(r"Gini(S)=1-\sum_{i=1}^{c}p_i^2")

        st.markdown("### Keterangan Simbol")
        st.latex(r"""
        \begin{aligned}
        Gini(S) &:\ \text{tingkat impurity berdasarkan Gini} \\
        p_i &:\ \text{proporsi kelas ke-}i \\
        c &:\ \text{jumlah kelas}
        \end{aligned}
        """)

    # =========================================================
    # RANDOM FOREST
    # =========================================================
    elif algo == "Random Forest":
        st.subheader("🌲 Random Forest Classification")

        st.markdown("## Definisi Random Forest Classification")
        st.markdown("""
        **Random Forest Classification** adalah metode klasifikasi berbasis **ensemble learning** yang membangun
        sejumlah **decision tree** menggunakan teknik **bootstrap sampling** dan **pemilihan fitur secara acak**,
        kemudian menentukan kelas akhir berdasarkan **majority voting** dari seluruh decision tree.

        Metode ini bertujuan untuk meningkatkan akurasi klasifikasi serta mengurangi masalah **overfitting**
        yang sering terjadi pada decision tree tunggal.
        """)

        st.markdown("---")
        st.markdown("## Tahapan Metode Random Forest Classification")

        st.markdown("### 1. Persiapan Dataset")
        st.markdown("""
        **Narasi:**  
        Tahap awal dalam Random Forest Classification adalah menyiapkan dataset yang akan digunakan untuk proses pelatihan model.
        Dataset terdiri dari sekumpulan data latih yang memiliki fitur dan label kelas.
        """)
        st.markdown("**Rumus Dataset:**")
        st.latex(r"D=\{(x_1,y_1),(x_2,y_2),\dots,(x_n,y_n)\}")

        st.markdown("**Keterangan simbol:**")
        st.latex(r"""
        \begin{aligned}
        D &:\ \text{dataset training} \\
        x_i &:\ \text{vektor fitur data ke-}i \\
        y_i &:\ \text{label kelas data ke-}i \\
        n &:\ \text{jumlah data}
        \end{aligned}
        """)

        st.markdown("---")

        st.markdown("### 2. Bootstrap Sampling")
        st.markdown("""
        **Narasi:**  
        Random Forest membentuk beberapa dataset baru dengan teknik **bootstrap sampling**,
        yaitu pengambilan data secara acak dengan pengembalian (with replacement) dari dataset asli.
        """)
        st.markdown("**Rumus Konseptual:**")
        st.latex(r"D_b \sim D")

        st.markdown("**Keterangan simbol:**")
        st.latex(r"""
        \begin{aligned}
        D &:\ \text{dataset asli} \\
        D_b &:\ \text{dataset hasil bootstrap} \\
        OOB &:\ \text{data yang tidak terpilih (Out-of-Bag)}
        \end{aligned}
        """)

        st.markdown("---")

        st.markdown("### 3. Pemilihan Fitur Secara Acak")
        st.markdown("""
        **Narasi:**  
        Pada setiap node decision tree, Random Forest hanya menggunakan sebagian fitur yang dipilih secara acak
        untuk menentukan pemisahan terbaik, dengan tujuan mengurangi korelasi antar tree.
        """)
        st.markdown("**Rumus Jumlah Fitur (umum untuk klasifikasi):**")
        st.latex(r"m=\sqrt{p}")

        st.markdown("**Keterangan simbol:**")
        st.latex(r"""
        \begin{aligned}
        p &:\ \text{jumlah total fitur} \\
        m &:\ \text{jumlah fitur acak pada setiap split}
        \end{aligned}
        """)

        st.markdown("---")

        st.markdown("### 4. Pembangunan Decision Tree")
        st.markdown("Setiap dataset bootstrap digunakan untuk membangun satu decision tree.")

        st.markdown("#### 4.1 Kriteria Pemisahan Node")

        st.markdown("**a. Gini Index**")
        st.latex(r"Gini(D)=1-\sum_{i=1}^{c}p_i^2")

        st.markdown("**b. Entropy**")
        st.latex(r"Entropy(D)=-\sum_{i=1}^{c}p_i\log_2(p_i)")

        st.markdown("**c. Information Gain**")
        st.latex(r"IG = Entropy(D) - \sum_{j=1}^{k}\frac{|D_j|}{|D|}Entropy(D_j)")

        st.markdown("**Keterangan simbol:**")
        st.latex(r"""
        \begin{aligned}
        D &:\ \text{dataset pada node} \\
        c &:\ \text{jumlah kelas} \\
        p_i &:\ \text{proporsi kelas ke-}i \\
        D_j &:\ \text{subset hasil split ke-}j \\
        |D| &:\ \text{jumlah data pada }D \\
        |D_j| &:\ \text{jumlah data pada }D_j \\
        k &:\ \text{jumlah subset hasil split}
        \end{aligned}
        """)

        st.markdown("---")

        st.markdown("### 5. Pembentukan Random Forest")
        st.markdown("""
        **Narasi:**  
        Proses bootstrap sampling dan pembangunan decision tree diulang hingga terbentuk sekumpulan pohon keputusan
        yang disebut Random Forest.
        """)
        st.markdown("**Rumus Ensemble:**")
        st.latex(r"RF=\{Tree_1,Tree_2,\dots,Tree_T\}")

        st.markdown("**Keterangan simbol:**")
        st.latex(r"""
        \begin{aligned}
        RF &:\ \text{Random Forest} \\
        T &:\ \text{jumlah decision tree (n\_estimators)} \\
        Tree_t &:\ \text{decision tree ke-}t
        \end{aligned}
        """)

        st.markdown("---")

        st.markdown("### 6. Proses Prediksi (Classification)")
        st.markdown("""
        **Narasi:**  
        Setiap decision tree menghasilkan prediksi kelas terhadap data uji.
        Kelas akhir ditentukan menggunakan metode **majority voting**.
        """)

        st.markdown("**Rumus Majority Voting:**")
        st.latex(r"\hat{y}=\text{mode}\{h_1(x),h_2(x),\dots,h_T(x)\}")

        st.markdown("**Keterangan simbol:**")
        st.latex(r"""
        \begin{aligned}
        \hat{y} &:\ \text{kelas prediksi akhir} \\
        h_t(x) &:\ \text{prediksi dari tree ke-}t \\
        x &:\ \text{data uji} \\
        T &:\ \text{jumlah tree}
        \end{aligned}
        """)

        st.markdown("---")

        st.markdown("### 7. Evaluasi Model")

        st.markdown("#### 7.1 Confusion Matrix")
        st.markdown("""
        | Aktual / Prediksi | Positif | Negatif |
        |---|---:|---:|
        | Positif | TP | FN |
        | Negatif | FP | TN |
        """)

        st.markdown("#### 7.2 Akurasi")
        st.latex(r"Accuracy=\frac{TP+TN}{TP+TN+FP+FN}")

        st.markdown("#### 7.3 Precision")
        st.latex(r"Precision=\frac{TP}{TP+FP}")

        st.markdown("#### 7.4 Recall")
        st.latex(r"Recall=\frac{TP}{TP+FN}")

        st.markdown("#### 7.5 F1-Score")
        st.latex(r"F1=2\times\frac{Precision\times Recall}{Precision+Recall}")

        st.markdown("**Keterangan simbol:**")
        st.latex(r"""
        \begin{aligned}
        TP &:\ \text{True Positive} \\
        TN &:\ \text{True Negative} \\
        FP &:\ \text{False Positive} \\
        FN &:\ \text{False Negative}
        \end{aligned}
        """)

        st.markdown("---")

        st.markdown("### 8. Feature Importance")
        st.markdown("""
        **Narasi:**  
        Random Forest Classification dapat mengukur tingkat kepentingan setiap fitur berdasarkan kontribusinya
        dalam mengurangi impurity selama proses pembentukan tree.
        """)

        st.markdown("**Rumus Umum:**")
        st.latex(r"FI_j=\sum_{t=1}^{T}\sum_{n\in t}\Delta Gini_{n,j}")

        st.markdown("**Keterangan simbol:**")
        st.latex(r"""
        \begin{aligned}
        FI_j &:\ \text{nilai kepentingan fitur ke-}j \\
        \Delta Gini_{n,j} &:\ \text{penurunan Gini oleh fitur }j\text{ pada node }n \\
        T &:\ \text{jumlah decision tree}
        \end{aligned}
        """)

        st.markdown("---")

        st.markdown("## Kesimpulan")
        st.markdown("""
        Random Forest Classification merupakan metode klasifikasi yang efektif karena menggabungkan banyak decision tree,
        menggunakan data dan fitur acak, serta menentukan hasil akhir melalui mekanisme **majority voting** sehingga
        menghasilkan performa yang lebih stabil dan akurat.
        """)
