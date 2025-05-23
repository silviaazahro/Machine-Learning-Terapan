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

### Visualisasi Distribusi Data Numerik
Visualisasi distribusi variabel menunjukkan bahwa **age** memiliki sebaran yang relatif normal dengan sedikit skew ke kanan, **bmi** menunjukkan distribusi yang cenderung right-skewed dengan puncak tajam di sekitar nilai 29–30, sementara **avg_glucose_level** memperlihatkan distribusi yang multimodal dengan beberapa puncak, termasuk lonjakan signifikan pada nilai tinggi.

![Numerik](Predictive Analysis/Distribusi Numerik.png)

### Visualisasi Distribusi Data Kategori
Visualisasi distribusi data kategori menunjukkan bahwa jumlah individu **perempuan** lebih banyak dibandingkan laki-laki. Mayoritas sampel **tidak memiliki hipertensi maupun penyakit jantung**, terlihat dari dominasi label 0 pada kedua kategori tersebut. Sebagian besar responden juga tercatat **pernah menikah**, dan **bekerja di sektor swasta (Private)** merupakan kategori pekerjaan yang paling umum. Selain itu, distribusi tempat tinggal menunjukkan jumlah yang hampir seimbang antara individu yang tinggal di **daerah urban dan rural**.

![Kategori](Predictive Analysis/Distribusi Kategorik.png)

### Visualisasi Rata Rata Diabetes vs Fitur
Visualisasi menunjukkan bahwa **laki-laki memiliki rata-rata kasus stroke sedikit lebih tinggi dibandingkan perempuan**. Individu yang memiliki **hipertensi atau penyakit jantung** menunjukkan kemungkinan stroke yang jauh lebih tinggi dibandingkan yang tidak memiliki kondisi tersebut.

Selain itu, **individu yang pernah menikah (ever_married = Yes)** memiliki rata-rata stroke yang lebih tinggi dibandingkan yang belum pernah menikah. Dalam hal pekerjaan, **pekerja mandiri (self-employed)** memiliki persentase stroke tertinggi, diikuti oleh pekerja swasta (private) dan pegawai negeri (govt_job). Anak-anak dan individu yang tidak pernah bekerja memiliki kasus stroke yang sangat rendah atau tidak ada sama sekali.

Distribusi berdasarkan tempat tinggal menunjukkan bahwa rata-rata kasus stroke sedikit lebih tinggi pada individu yang tinggal di daerah **urban** dibandingkan **rural**. Sementara itu, dalam kategori status merokok, **mantan perokok (formerly smoked)** memiliki rata-rata stroke tertinggi, disusul oleh perokok aktif (smokes), tidak pernah merokok (never smoked), dan yang tidak diketahui (unknown).

![Mean](Predictive Analysis/MeanStrokeVSFiturLainnya.png)

### Visualisasi KDE
Visualisasi menunjukkan hubungan antara berbagai fitur dalam dataset, seperti **id, age, hypertension, heart\_disease, avg\_glucose\_level, bmi**, dan **stroke**. Scatter plot memperlihatkan distribusi titik data di antara pasangan variabel, sedangkan plot KDE di diagonal menggambarkan distribusi probabilitas dari masing-masing variabel. Dari grafik ini, beberapa fitur seperti **age, avg\_glucose\_level, dan bmi** memiliki distribusi yang lebih bervariasi, sedangkan variabel biner seperti **hypertension dan heart\_disease** memiliki titik data yang lebih terpisah tanpa pola yang jelas dalam scatter plot.

![Mean](Predictive Analysis/KDE.png)

### Visualisasi Correlation Matrix
Visualisasi menunjukkan hubungan korelasi antar fitur numerik dalam dataset. Dari matriks korelasi ini, kita dapat mengamati bahwa fitur **age** memiliki korelasi yang cukup tinggi dengan **bmi** (0.28), menunjukkan adanya kecenderungan peningkatan indeks massa tubuh seiring bertambahnya usia. Selain itu, **avg\_glucose\_level** juga menunjukkan korelasi yang relatif lebih tinggi dengan **age** (0.20) dan **bmi** (0.15). Namun, korelasi antara fitur-fitur lain seperti **hypertension** dan **heart\_disease** dengan **stroke** tampak lebih rendah. Secara keseluruhan, matriks ini memberikan gambaran mengenai kekuatan dan arah hubungan linear antar variabel numerik dalam dataset.

![Mean](Predictive Analysis/Matrik Korelasi.png)

## Data Preparation

