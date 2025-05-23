# Laporan Proyek Analisis Prediksi - Silvia Zahro

## Domain Proyek

Stroke adalah salah satu penyakit kardiovaskular yang menjadi penyebab utama kematian dan kecacatan di seluruh dunia. Menurut **WHO (2021)**, stroke terjadi ketika pasokan darah ke bagian otak terhenti atau berkurang secara drastis, menyebabkan kerusakan jaringan otak. Faktor risiko stroke meliputi hipertensi, penyakit jantung, diabetes, obesitas, kebiasaan merokok, dan usia lanjut. Deteksi dini risiko stroke sangat penting untuk mengambil tindakan pencegahan guna mengurangi angka kematian dan dampak jangka panjang.

Penelitian-penelitian terkini, seperti yang dilakukan oleh **Saposnik et al. (2016)**, memanfaatkan machine learning untuk membangun model prediksi risiko stroke dengan menggunakan data medis pasien, termasuk variabel demografi dan riwayat kesehatan. Model prediksi ini membantu tenaga medis dalam mengidentifikasi pasien berisiko tinggi sehingga dapat dilakukan intervensi lebih cepat dan tepat.

Selain itu, **Chen et al. (2019)** mengaplikasikan algoritma Random Forest dan XGBoost untuk prediksi stroke berdasarkan data klinis dan gaya hidup pasien, dengan hasil akurasi yang menjanjikan. Studi tersebut menekankan pentingnya variabel seperti tekanan darah, kadar glukosa, BMI, dan kebiasaan merokok sebagai faktor utama dalam meningkatkan akurasi model.

Berdasarkan temuan-temuan ini, penerapan machine learning dalam prediksi stroke menjadi alat bantu yang potensial untuk meningkatkan pencegahan dan perawatan pasien secara lebih personal dan efisien.

## Business Understanding

### Problem Statements
1. Bagaimana cara memprediksi kemungkinan seorang mengidap stroke berdasarkan fakto-faktor kesehatan yang dialami?
2. Seberapa akurat model dapat memprediksi dibandingkan dengan metode konvensional dalam mendeteksi stroke?

### Goals
1. Mengembangkan sebuah model machine learning untuk memprediksi kemungkinan seseorang mengidap penyakit stroke berdasarkan faktor-faktor kesehatan yang dialami.
2. Mengevaluasi kinerja model Machine Learning dengan berbagai metrik evaluasi seperti akurasi, precision, recall, F1-score, dan confusion matrix, guna memastikan model memiliki performa yang optimal dalam mendeteksi stroke.

### Solution Statement
1. Menggunakan beberapa algoritma Machine Learning seperti Random Forest, XGBoost, dan LightGBM untuk membandingkan performa model dalam mendeteksi stroke.
2. Melakukan analisis hasil model berdasarkan metrik evaluasi untuk memilih model terbaik yang mampu memberikan prediksi paling akurat.

## Data Understanding
**`Stroke prediction dataset`** merupakan kumpulan data medis dan demografi dari pasien, beserta status stroke mereka (positif atau negatif). Data tersebut mencakup fitur-fitur seperti usia, jenis kelamin, indeks massa tubuh (IMT/BMI), hipertensi, penyakit jantung, riwayat merokok, status menikah, tipe pekerjaan, tipe tempat tinggal, dan kadar glukosa darah. Dataset berjumlah 12 kolom, 5110 baris dan dataset bersih tidak ada missing values, dataset ini diambil dari platform **[Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)**.

### Variabel-variabel pada dataset prediksi penyakit stroke adalah sebagai berikut:
- **`id`** : Nomor identifikasi unik untuk setiap pasien dalam dataset.
- **`gender`** : Jenis kelamin pasien, yang dapat memengaruhi risiko stroke. Kategori dalam data adalah laki-laki dan perempuan.
- **`age`** : Usia pasien dalam tahun, faktor penting karena risiko stroke meningkat seiring bertambahnya usia.
- **`hypertension`** : Status hipertensi pasien, dimana 0 berarti tidak memiliki hipertensi dan 1 berarti memiliki hipertensi. Hipertensi adalah faktor risiko utama stroke.
- **`heart_disease`** : Status penyakit jantung pasien, dengan nilai 0 untuk tidak dan 1 untuk ya. Penyakit jantung juga meningkatkan risiko stroke.
- **`ever_married`** : Status pernikahan pasien, berupa "Ya" atau "Tidak". Faktor sosial ini dapat memengaruhi gaya hidup dan risiko kesehatan.
- **`work_type`** : Tipe pekerjaan pasien, seperti pemerintah, swasta, wiraswasta, tidak bekerja, atau anak-anak. Kegiatan kerja dapat memengaruhi kesehatan dan stres.
- **`Residence_type`** : Jenis tempat tinggal pasien, berupa "Urban" (kota) atau "Rural" (desa), yang dapat berdampak pada akses layanan kesehatan dan pola hidup.
- **`avg_glucose_level`** : Rata-rata kadar glukosa darah pasien, yang berhubungan dengan risiko penyakit metabolik dan stroke.
- **`bmi`** : Indeks Massa Tubuh pasien, ukuran lemak tubuh berdasarkan berat dan tinggi badan. BMI yang tinggi meningkatkan risiko stroke.
- **`smoking_status`** : Status merokok pasien, yang mencakup kategori seperti "never smoked" (tidak pernah merokok), "formerly smoked" (pernah merokok), "smokes" (merokok saat ini), dan "unknown" (tidak diketahui). Merokok adalah faktor risiko utama stroke.
- **`stroke`** : Variabel target, dengan nilai 1 jika pasien pernah mengalami stroke dan 0 jika tidak.
