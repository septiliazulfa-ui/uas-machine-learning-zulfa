import streamlit as st


def analysis_model_page():
    st.title("🧭 Langkah-Langkah Algoritma Klasifikasi (Detail)")

    st.markdown("""
    Halaman ini menjelaskan **alur langkah-langkah (step-by-step)** dari beberapa algoritma klasifikasi:
    **K-Nearest Neighbor (KNN), Logistic Regression, Naive Bayes, Decision Tree, dan Random Forest**.

    Penjelasan difokuskan pada **proses kerja algoritma**, bukan pada hasil prediksi maupun perhitungan matematis.
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

        st.markdown("""
        **Gambaran Umum:**  
        K-Nearest Neighbor (KNN) adalah algoritma klasifikasi yang menentukan kelas data baru
        berdasarkan kedekatan jarak dengan data lain di sekitarnya.
        """)

        st.markdown("### Input")
        st.markdown("""
        1. Data latih (fitur dan label kelas)  
        2. Data uji (fitur)  
        3. Nilai parameter **K** (jumlah tetangga)
        """)

        st.markdown("### Langkah-Langkah Proses")
        st.markdown("""
        **Langkah 1:** Tentukan nilai **K** yang akan digunakan.  

        **Langkah 2:** Lakukan normalisasi atau standarisasi fitur agar skala data seimbang.  

        **Langkah 3:** Ambil satu data uji yang akan diprediksi kelasnya.  

        **Langkah 4:** Hitung jarak antara data uji dengan seluruh data latih.  

        **Langkah 5:** Urutkan data latih berdasarkan jarak dari yang paling dekat.  

        **Langkah 6:** Pilih **K data terdekat** sebagai tetangga.  

        **Langkah 7:** Tentukan kelas prediksi berdasarkan kelas yang paling sering muncul di antara K tetangga.
        """)

        st.markdown("### Output")
        st.markdown("""
        1. Kelas prediksi data uji  
        2. (Opsional) Proporsi suara dari masing-masing kelas
        """)

        st.markdown("""
        **Catatan:**  
        KNN tidak membangun model di awal sehingga proses prediksi dapat menjadi lambat
        jika jumlah data latih sangat besar.
        """)

    # =========================================================
    # LOGISTIC REGRESSION
    # =========================================================
    elif algo == "Logistic Regression":
        st.subheader("📌 Logistic Regression")

        st.markdown("""
        **Gambaran Umum:**  
        Logistic Regression adalah algoritma klasifikasi yang memprediksi probabilitas
        suatu data termasuk ke dalam kelas tertentu.
        """)

        st.markdown("### Input")
        st.markdown("""
        1. Data latih (fitur dan label)  
        2. Data uji  
        3. Parameter pelatihan (solver, iterasi, regularisasi)
        """)

        st.markdown("### Langkah-Langkah Proses Training")
        st.markdown("""
        **Langkah 1:** Pisahkan fitur dan label dari data latih.  

        **Langkah 2:** Lakukan standarisasi fitur agar proses pelatihan stabil.  

        **Langkah 3:** Inisialisasi bobot model.  

        **Langkah 4:** Hitung skor prediksi untuk setiap data latih.  

        **Langkah 5:** Ubah skor menjadi nilai probabilitas kelas.  

        **Langkah 6:** Hitung selisih antara probabilitas prediksi dan label aktual.  

        **Langkah 7:** Perbarui bobot model untuk meminimalkan kesalahan.  

        **Langkah 8:** Ulangi proses hingga model konvergen atau iterasi maksimum tercapai.
        """)

        st.markdown("### Langkah-Langkah Proses Prediksi")
        st.markdown("""
        **Langkah 9:** Masukkan data uji ke dalam model terlatih.  

        **Langkah 10:** Hasilkan probabilitas kelas.  

        **Langkah 11:** Tentukan kelas akhir berdasarkan nilai ambang (threshold).
        """)

        st.markdown("### Output")
        st.markdown("""
        1. Probabilitas kelas  
        2. Kelas prediksi  
        3. Bobot fitur yang dapat dianalisis
        """)

    # =========================================================
    # NAIVE BAYES
    # =========================================================
    elif algo == "Naive Bayes":
        st.subheader("📌 Naive Bayes")

        st.markdown("""
        **Gambaran Umum:**  
        Naive Bayes adalah algoritma klasifikasi berbasis probabilitas
        yang menggunakan prinsip Teorema Bayes dengan asumsi independensi antar fitur.
        """)

        st.markdown("### Input")
        st.markdown("""
        1. Data latih  
        2. Data uji  
        3. Jenis Naive Bayes (Gaussian, Multinomial, atau Bernoulli)
        """)

        st.markdown("### Langkah-Langkah Proses Training")
        st.markdown("""
        **Langkah 1:** Identifikasi seluruh kelas pada label target.  

        **Langkah 2:** Hitung probabilitas awal (prior) untuk setiap kelas.  

        **Langkah 3:** Hitung parameter probabilitas fitur untuk masing-masing kelas.  

        **Langkah 4:** Simpan seluruh parameter sebagai model.
        """)

        st.markdown("### Langkah-Langkah Proses Prediksi")
        st.markdown("""
        **Langkah 5:** Ambil satu data uji.  

        **Langkah 6:** Hitung probabilitas data uji terhadap setiap kelas.  

        **Langkah 7:** Bandingkan probabilitas antar kelas.  

        **Langkah 8:** Pilih kelas dengan probabilitas tertinggi sebagai hasil prediksi.
        """)

        st.markdown("### Output")
        st.markdown("""
        1. Skor probabilitas tiap kelas  
        2. Kelas prediksi akhir
        """)

    # =========================================================
    # DECISION TREE
    # =========================================================
    elif algo == "Decision Tree":
        st.subheader("📌 Decision Tree")

        st.markdown("""
        **Gambaran Umum:**  
        Decision Tree membentuk aturan keputusan dalam bentuk pohon
        dengan memecah data menjadi subset yang semakin homogen.
        """)

        st.markdown("### Input")
        st.markdown("""
        1. Data latih  
        2. Parameter pohon (kedalaman maksimum, minimum data per node)
        """)

        st.markdown("### Langkah-Langkah Proses Training")
        st.markdown("""
        **Langkah 1:** Letakkan seluruh data latih pada node akar.  

        **Langkah 2:** Hitung tingkat ketidakmurnian data pada node tersebut.  

        **Langkah 3:** Evaluasi setiap fitur sebagai kandidat pemisah data.  

        **Langkah 4:** Pilih fitur dan kondisi split terbaik.  

        **Langkah 5:** Bagi data menjadi node-node anak.  

        **Langkah 6:** Ulangi proses hingga kondisi berhenti terpenuhi.  

        **Langkah 7:** Tentukan kelas pada setiap node daun.
        """)

        st.markdown("### Langkah-Langkah Proses Prediksi")
        st.markdown("""
        **Langkah 8:** Data uji mengikuti jalur pohon dari akar ke daun.  

        **Langkah 9:** Kelas pada node daun menjadi hasil prediksi.
        """)

        st.markdown("### Output")
        st.markdown("""
        1. Struktur pohon keputusan  
        2. Kelas prediksi
        """)

    # =========================================================
    # RANDOM FOREST
    # =========================================================
    elif algo == "Random Forest":
        st.subheader("🌲 Random Forest Classification")

        st.markdown("""
        **Gambaran Umum:**  
        Random Forest adalah metode ensemble yang menggabungkan banyak decision tree
        untuk menghasilkan prediksi yang lebih stabil dan akurat.
        """)

        st.markdown("### Input")
        st.markdown("""
        1. Data latih  
        2. Jumlah pohon keputusan (n_estimators)  
        3. Parameter tree dan jumlah fitur acak
        """)

        st.markdown("### Langkah-Langkah Proses Training")
        st.markdown("""
        **Langkah 1:** Siapkan dataset training.  

        **Langkah 2:** Tentukan jumlah decision tree yang akan dibangun.  

        **Langkah 3:** Untuk setiap decision tree:  

        **Langkah 3.1:** Lakukan bootstrap sampling dari dataset training.  

        **Langkah 3.2:** Bangun decision tree dari sampel bootstrap.  

        **Langkah 3.3:** Pada setiap node, pilih subset fitur secara acak untuk menentukan split terbaik.  

        **Langkah 3.4:** Lanjutkan hingga pohon selesai dibangun.  

        **Langkah 4:** Kumpulkan seluruh decision tree menjadi satu model Random Forest.
        """)

        st.markdown("### Langkah-Langkah Proses Prediksi")
        st.markdown("""
        **Langkah 5:** Masukkan data uji ke setiap decision tree.  

        **Langkah 6:** Setiap decision tree menghasilkan prediksi kelas.  

        **Langkah 7:** Gabungkan seluruh prediksi menggunakan voting mayoritas.
        """)

        st.markdown("### Output")
        st.markdown("""
        1. Kelas prediksi akhir  
        2. (Opsional) Probabilitas kelas berdasarkan voting  
        3. (Opsional) Informasi kepentingan fitur
        """)

        st.markdown("""
        **Kesimpulan:**  
        Random Forest mampu meningkatkan performa klasifikasi dengan
        menggabungkan banyak pohon keputusan serta mengurangi overfitting.
        """)
