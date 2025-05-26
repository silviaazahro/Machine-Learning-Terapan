# Laporan Proyek Analisis Prediksi - Silvia Zahro

## Domain Proyek

Stroke adalah salah satu penyakit kardiovaskular yang menjadi penyebab utama kematian dan kecacatan di seluruh dunia. Menurut **WHO (2021)**, stroke terjadi ketika pasokan darah ke bagian otak terhenti atau berkurang secara drastis, menyebabkan kerusakan jaringan otak. Faktor risiko stroke meliputi hipertensi, penyakit jantung, diabetes, obesitas, kebiasaan merokok, dan usia lanjut. Deteksi dini risiko stroke sangat penting untuk mengambil tindakan pencegahan guna mengurangi angka kematian dan dampak jangka panjang.

Penelitian-penelitian terkini, seperti yang dilakukan oleh **Saposnik et al. (2016)**, memanfaatkan machine learning untuk membangun model prediksi risiko stroke dengan menggunakan data medis pasien, termasuk variabel demografi dan riwayat kesehatan. Model prediksi ini membantu tenaga medis dalam mengidentifikasi pasien berisiko tinggi sehingga dapat dilakukan intervensi lebih cepat dan tepat.

Selain itu, **Chen et al. (2019)** mengaplikasikan algoritma Random Forest dan XGBoost untuk prediksi stroke berdasarkan data klinis dan gaya hidup pasien, dengan hasil akurasi yang menjanjikan. Studi tersebut menekankan pentingnya variabel seperti tekanan darah, kadar glukosa, BMI, dan kebiasaan merokok sebagai faktor utama dalam meningkatkan akurasi model.

Berdasarkan temuan-temuan ini, penerapan machine learning dalam prediksi stroke menjadi alat bantu yang potensial untuk meningkatkan pencegahan dan perawatan pasien secara lebih personal dan efisien.

## Business Understanding

### Problem Statements
1. Bagaimana cara memprediksi kemungkinan seorang mengidap stroke berdasarkan faktor-faktor kesehatan yang dialami?
2. Seberapa akurat model dapat memprediksi dibandingkan dengan metode konvensional dalam mendeteksi stroke?

### Goals
1. Mengembangkan sebuah model machine learning untuk memprediksi kemungkinan seseorang mengidap penyakit stroke berdasarkan faktor-faktor kesehatan yang dialami.
2. Mengevaluasi kinerja model Machine Learning dengan berbagai metrik evaluasi seperti akurasi, precision, recall, F1-score, dan confusion matrix, guna memastikan model memiliki performa yang optimal dalam mendeteksi stroke.

### Solution Statement
1. Menggunakan beberapa algoritma Machine Learning seperti Random Forest, XGBoost, dan LightGBM untuk membandingkan performa model dalam mendeteksi stroke.
2. Melakukan analisis hasil model berdasarkan metrik evaluasi untuk memilih model terbaik yang mampu memberikan prediksi paling akurat.

## Data Understanding
**`Stroke prediction dataset`** merupakan kumpulan data medis dan demografi dari pasien, beserta status stroke mereka (positif atau negatif). Data tersebut mencakup fitur-fitur seperti usia, jenis kelamin, indeks massa tubuh (IMT/BMI), hipertensi, penyakit jantung, riwayat merokok, status menikah, tipe pekerjaan, tipe tempat tinggal, dan kadar glukosa darah. Dataset berjumlah 12 kolom, 5110 baris dan dataset kotor yang masih terdapat 201 missing values pada kolom BMI, dataset ini diambil dari platform **[Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)**.

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

![Distribusi Numerik](https://github.com/silviaazahro/Machine-Learning-Terapan/raw/main/Predictive%20Analysis/Distribusi%20Numerik.png)

### Visualisasi Distribusi Data Kategori
Visualisasi distribusi data kategori menunjukkan bahwa jumlah individu **perempuan** lebih banyak dibandingkan laki-laki. Mayoritas sampel **tidak memiliki hipertensi maupun penyakit jantung**, terlihat dari dominasi label 0 pada kedua kategori tersebut. Sebagian besar responden juga tercatat **pernah menikah**, dan **bekerja di sektor swasta (Private)** merupakan kategori pekerjaan yang paling umum. Selain itu, distribusi tempat tinggal menunjukkan jumlah yang hampir seimbang antara individu yang tinggal di **daerah urban dan rural**.

![Distribusi Kategorik](https://github.com/silviaazahro/Machine-Learning-Terapan/raw/main/Predictive%20Analysis/Distribusi%20Kategorik.png)

### Visualisasi Rata Rata Stroke vs Fitur
Visualisasi menunjukkan bahwa **laki-laki memiliki rata-rata kasus stroke sedikit lebih tinggi dibandingkan perempuan**. Individu yang memiliki **hipertensi atau penyakit jantung** menunjukkan kemungkinan stroke yang jauh lebih tinggi dibandingkan yang tidak memiliki kondisi tersebut.

Selain itu, **individu yang pernah menikah (ever_married = Yes)** memiliki rata-rata stroke yang lebih tinggi dibandingkan yang belum pernah menikah. Dalam hal pekerjaan, **pekerja mandiri (self-employed)** memiliki persentase stroke tertinggi, diikuti oleh pekerja swasta (private) dan pegawai negeri (govt_job). Anak-anak dan individu yang tidak pernah bekerja memiliki kasus stroke yang sangat rendah atau tidak ada sama sekali.

Distribusi berdasarkan tempat tinggal menunjukkan bahwa rata-rata kasus stroke sedikit lebih tinggi pada individu yang tinggal di daerah **urban** dibandingkan **rural**. Sementara itu, dalam kategori status merokok, **mantan perokok (formerly smoked)** memiliki rata-rata stroke tertinggi, disusul oleh perokok aktif (smokes), tidak pernah merokok (never smoked), dan yang tidak diketahui (unknown).

![Mean Stroke vs Fitur Lainnya](https://github.com/silviaazahro/Machine-Learning-Terapan/raw/main/Predictive%20Analysis/MeanStrokeVSFiturLainnya.png)

### Visualisasi KDE
Visualisasi menunjukkan hubungan antara berbagai fitur dalam dataset, seperti **id, age, hypertension, heart\_disease, avg\_glucose\_level, bmi**, dan **stroke**. Scatter plot memperlihatkan distribusi titik data di antara pasangan variabel, sedangkan plot KDE di diagonal menggambarkan distribusi probabilitas dari masing-masing variabel. Dari grafik ini, beberapa fitur seperti **age, avg\_glucose\_level, dan bmi** memiliki distribusi yang lebih bervariasi, sedangkan variabel biner seperti **hypertension dan heart\_disease** memiliki titik data yang lebih terpisah tanpa pola yang jelas dalam scatter plot.

![KDE](https://github.com/silviaazahro/Machine-Learning-Terapan/raw/main/Predictive%20Analysis/KDE.png)

### Visualisasi Correlation Matrix
Visualisasi menunjukkan hubungan korelasi antar fitur numerik dalam dataset. Dari matriks korelasi ini, kita dapat mengamati bahwa fitur **age** memiliki korelasi yang cukup tinggi dengan **bmi** (0.28), menunjukkan adanya kecenderungan peningkatan indeks massa tubuh seiring bertambahnya usia. Selain itu, **avg\_glucose\_level** juga menunjukkan korelasi yang relatif lebih tinggi dengan **age** (0.20) dan **bmi** (0.15). Namun, korelasi antara fitur-fitur lain seperti **hypertension** dan **heart\_disease** dengan **stroke** tampak lebih rendah. Secara keseluruhan, matriks ini memberikan gambaran mengenai kekuatan dan arah hubungan linear antar variabel numerik dalam dataset.

![Matrik Korelasi](https://github.com/silviaazahro/Machine-Learning-Terapan/raw/main/Predictive%20Analysis/Matrik%20Korelasi.png)

## Data Preparation

1.  **Teknik Data Preparation yang Diterapkan:**
    * **Penanganan Missing Value:**
        * Teknik: Imputasi dengan nilai mean dan median.
        * Kode Snippet:
            ```python
            df['bmi'].fillna(df['bmi'].mean(), inplace=True)
            df['bmi'].fillna(df['bmi'].median(), inplace=True)
            ```
        * Proses: Missing value pada fitur `bmi` diisi dengan nilai median dari fitur tersebut.
        * Alasan: Fitur `bmi` memiliki missing value yang cukup signifikan. Imputasi dengan median dipilih karena mean dan median robust terhadap outlier, yang mungkin ada dalam distribusi `bmi`. Hal ini mencegah outlier mendistorsi representasi tipikal dari data.
   * **`Handling Outlier`** : Outlier dicari pada kolom numerik dengan metode IQR, setelah outlier terdeteksi akan diterapkan clipping pada nilai yang berada di luar batas yang telah ditentukan. Outlier dapat memberikan pengaruh yang berlebihan pada proses pelatihan model. Dengan clipping, nilai ekstrim dihilangkan atau dibatasi, sehingga model dapat belajar dari data yang lebih representatif.
    ```python
   fitur_numerik = ['age', 'bmi', 'avg_glucose_level']
   nilai_outlier = {}
   for kolom in fitur_numerik:
       kuartil_1 = np.percentile(df[kolom].dropna(), 25)
       kuartil_3 = np.percentile(df[kolom].dropna(), 75)
       rentang_iqr = kuartil_3 - kuartil_1
       batas_bawah = kuartil_1 - 1.5 * rentang_iqr
       batas_atas = kuartil_3 + 1.5 * rentang_iqr
       outlier = df[kolom][(df[kolom] < batas_bawah) | (df[kolom] > batas_atas)]
       nilai_outlier[kolom] = outlier
       df[kolom] = np.clip(df[kolom], batas_bawah, batas_atas)
     ```
    * **Encoding Variabel Kategorikal:**
        * Teknik: One-Hot Encoding.
        * Kode Snippet:
            ```python
            fitur_kategorikal = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            hasil_encoding = ohe.fit_transform(df[fitur_kategorikal])
            df_encoded = pd.DataFrame(hasil_encoding, columns=ohe.get_feature_names_out(fitur_kategorikal), index=df.index)
            df.drop(columns=fitur_kategorikal, inplace=True)
            df = pd.concat([df, df_encoded], axis=1)
            ```
      **Proses:** Variabel kategorikal diubah menjadi variabel dummy/indikator biner menggunakan `OneHotEncoder` dari `sklearn.preprocessing`. `OneHotEncoder` cocok digunakan dalam pipeline dan mendukung transformasi yang konsisten antara data pelatihan dan pengujian. Parameter `drop='first'` dapat digunakan untuk menghindari multikolinearitas.
      **Alasan:** Model machine learning umumnya memerlukan input numerik. One-Hot Encoding mengubah kategori menjadi format yang dapat diproses model, tanpa memberikan asumsi ordinalitas yang tidak tepat. Dibandingkan `pd.get_dummies`, `OneHotEncoder` lebih fleksibel, terutama saat mengaplikasikan transformasi yang sama ke data baru, karena encoder dapat disimpan dan digunakan ulang menggunakan metode `fit` dan `transform`.
    * **Scaling Fitur Numerik:**
        * Teknik: StandardScaler.
        * Kode Snippet:
            ```python
            fitur_numerik = ['age', 'bmi', 'avg_glucose_level']
            standarisasi = StandardScaler()
            df[fitur_numerik] = standarisasi.fit_transform(df[fitur_numerik])
            ```
        * Proses: Fitur numerik diubah skalanya sehingga memiliki mean 0 dan standar deviasi 1.
        * Alasan: Scaling menyamakan rentang nilai fitur, yang penting untuk algoritma yang sensitif terhadap skala data (misalnya, algoritma berbasis jarak). Ini mencegah fitur dengan rentang besar mendominasi fitur dengan rentang kecil.
    * **Pembagian Data:**
        * Teknik: Train-Test Split.
        * Kode Snippet:
            ```python
            fitur = df.drop(columns=["stroke"])
            target = df["stroke"]
            X_train, X_test, y_train, y_test = train_test_split(fitur, target, test_size=0.3, random_state=123)
            X_train, X_val, y_train, y_val = train_test_split(fitur, target, test_size=0.3, random_state=123)
            ```
        * Proses: Data dibagi menjadi set pelatihan (70%) dan pengujian (30%). Set pelatihan selanjutnya dibagi menjadi set pelatihan (70%) dan validasi (30%).
        * Alasan: Pembagian ini memungkinkan pelatihan model pada sebagian data dan evaluasi pada data yang belum dilihat sebelumnya (data uji) untuk mengukur generalisasi. Set validasi digunakan untuk validasi model.

## Modelling
Pada studi kali ini, model yang digunakan adalah **Random Forest**, **XGBoost**, dan **LightGBM** untuk memprediksi kemungkinan seseorang mengidap Stroke berdasarkan fitur-fitur yang ada. Alasan pemilihan ketiga model tersebut adalah:
- ### Penjelasan Model dan Alasan Pemilihan Parameter
**Random Forest:**  
Model ini merupakan metode ensemble yang menggabungkan banyak decision tree melalui teknik bagging (bootstrap aggregating). Setiap pohon dilatih pada subset data yang berbeda secara acak dan hasil akhirnya diambil melalui voting (klasifikasi) atau rata-rata (regresi).  
- **Kelebihan:** Mampu menangani data non-linear, robust terhadap overfitting karena rata-rata antar pohon, dan bekerja baik tanpa banyak tuning.
- **Kekurangan:** Proses pelatihan lebih lambat dibanding model boosting, serta interpretabilitas lebih rendah karena kompleksitas ensemble.

**XGBoost (Extreme Gradient Boosting):**  
Merupakan model boosting yang membangun pohon secara berurutan, di mana setiap pohon baru berusaha memperbaiki kesalahan dari pohon sebelumnya. XGBoost menggunakan teknik regularisasi (L1 dan L2) untuk menghindari overfitting dan mengoptimalkan performa.
- **Mekanisme:** Berdasarkan prinsip gradient boosting, di mana loss function diminimalkan secara iteratif melalui pendekatan gradien.
- **eval_metric='logloss':**  
  Digunakan dalam masalah klasifikasi biner untuk mengevaluasi seberapa baik model memprediksi probabilitas. Log-loss menghukum prediksi yang jauh dari label aktual, sehingga cocok untuk mengukur kualitas probabilistik prediksi.
- **n_estimators=100:**  
  Menentukan jumlah pohon (iterasi boosting). Nilai 100 dipilih sebagai default yang cukup umum, memberikan keseimbangan antara akurasi dan waktu pelatihan. Namun, nilainya bisa dioptimasi lebih lanjut melalui tuning hyperparameter (misalnya dengan cross-validation).

**LightGBM (Light Gradient Boosting Machine):**  
Model boosting yang dirancang untuk efisiensi dan kecepatan. LightGBM menggunakan teknik histogram-based dan leaf-wise tree growth, yang membuatnya lebih cepat dan efisien dalam memori dibanding XGBoost.
- **Kelebihan:** Sangat cepat dalam pelatihan, cocok untuk dataset besar, mendukung paralelisasi.
- **Kekurangan:** Lebih sensitif terhadap outlier dan bisa lebih mudah overfitting jika tidak dikontrol dengan parameter regularisasi seperti `max_depth` atau `min_child_samples`.

### Kesimpulan:
Pemilihan model dan konfigurasi seperti `n_estimators=100` atau `eval_metric='logloss'` harus disesuaikan dengan karakteristik data dan tujuan model. Tanpa memahami prinsip kerja dasar masing-masing algoritma, sulit untuk menilai apakah parameter tersebut sudah optimal. Oleh karena itu, pemahaman menyeluruh terhadap mekanisme model sangat penting dalam proses pemodelan machine learning.

### Tahapan yang dilakukan pada proses pemodelan adalah sebagai berikut:
1. **`Load Model`**:

   - **Random Forest** diload dengan parameter `n_estimators=100` dan `random_state=123`:
     ```python
     rf_model = RandomForestClassifier(n_estimators=100, random_state=123)
     ```
   - **XGBoost** diload dengan parameter `use_label_encoder=False`, `eval_metric='logloss'`, dan `random_state=123`:
     ```python
     xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=123)
     ```
   - **LightGBM** diload dengan parameter `random_state=123`:
     ```python
     lgbm_model = LGBMClassifier(random_state=123)
     ```
2. **`Pelatihan Model`**: 

   - **Random Forest** dilatih dengan data latih yaitu `X_train dan y_train`:
     ```python
     rf_model.fit(X_train, y_train)
     ```
   - **XGBoost** dilatih dengan data latih yaitu `X_train dan y_train`:
     ```python
     xgb_model.fit(X_train, y_train)
     ```
   - **LightGBM** dilatih dengan data latih yaitu `X_train dan y_train`:
     ```python
     lgbm_model.fit(X_train, y_train)
     ```
3. **Evaluasi Model**: 
   Hasil pelatihan dari ketiga model dibandingkan untuk menentukan model terbaik berdasarkan metrik evaluasi.

Berdasarkan hasil evaluasi (lihat bagian Evaluation), **XGBoost Classifier** dipilih sebagai model terbaik. Meskipun LightGBM menawarkan akurasi yang sebanding dengan kecepatan dan efisiensi yang lebih baik, XGBoost Classifier diputuskan sebagai model terbaik untuk prediksi stroke. Pertimbangan utama adalah bahwa dalam kasus stroke, biaya False Negative (gagal mendeteksi stroke) jauh lebih tinggi daripada biaya False Positive. Oleh karena itu, XGBoost, yang menunjukkan recall yang lebih baik (kemampuan lebih baik untuk mengidentifikasi kasus stroke yang sebenarnya), dipilih meskipun mungkin membutuhkan lebih banyak sumber daya komputasi.

## Evaluation
**Evaluasi model** dilakukan menggunakan beberapa metrik utama yang sesuai dengan konteks klasifikasi biner, yaitu **Accuracy**, **Precision**, **Recall**, **F1-Score**, dan **Confusion Matrix**. Metrik ini dipilih karena dataset yang digunakan melibatkan prediksi suatu kondisi (kemungkinan stroke) di mana keseimbangan antara deteksi positif dan negatif sangat penting.

Metrik Evaluasi yang Digunakan
1. **`Accuracy Score`** :
- **Accuracy**: Persentase prediksi yang benar dari seluruh prediksi.
   Formula:  `Akurasi = (Jumlah prediksi benar) / (Total jumlah prediksi)`
        ```python
        test_acc = accuracy_score(y_test, y_test_pred)
        ```
    
2. **`Classification Report`** :
    - **Precision**: Proporsi prediksi positif yang benar.
      Formula = `Precision = (Jumlah TP) / (Jumlah TP + Jumlah FP)`  
    - **Recall (Sensitivity)**: Proporsi kasus positif yang berhasil dideteksi.
      Formula = `Recall = (Jumlah TP) / (Jumlah TP + Jumlah FN)`
    - **F1-Score**: Rata-rata harmonik antara Precision dan Recall, yang memberikan gambaran keseimbangan antara keduanya.    
      Formula = `F1-score = 2 * (Precision * Recall) / (Precision + Recall)`
        ```python
        print("\n--- Classification Report (Test) ---\n", classification_report(y_test, y_test_pred))
        ```

3. **`Confusion Matrix`** : 

    |                | Predicted Negatif (0) |  Predicted Positif (1) |
    |----------------|---------------|--------------------|
    | Actual Negatif (0)  | True Negative (TN)	        | False Positive (FP)              |
    | Actual Positif (1)        | False Negative (FN)	        | True Positive (TP)              |

     ```python
    test_cm = confusion_matrix(y_test, y_test_pred)
    ```

Berikut adalah ringkasan hasil evaluasi berdasarkan prediksi pada data :
1. Accuracy dan Classification Report :

    | Model          | Accuracy |  Precision |  Recall |  F1-Score |
    |----------------|---------------|--------------------|-----------------|-------------------|
    | Random Forest  | 0.9550        | 0.33             | 0.01              | 0.03             |
    | XGBoost        | 0.9472        | 0.28             | 0.12              | 0.16             |
    | LightGBM       | 0.9511        | 0.27             | 0.06              | 0.10             |

    Analisis Hasil
    - Accuracy dari ketiga model sangat tinggi (sekitar 95%), yang menunjukkan bahwa model mampu memprediksi dengan sangat baik pada data uji.

    - Precision untuk kelas 1 (stroke) relatif rendah di semua model. Ini berarti bahwa ketika model memprediksi seseorang memiliki stroke (kelas 1), prediksi tersebut sering kali salah.
        * Random Forest memiliki precision 0.33, yang terendah.
        * XGBoost memiliki precision 0.28.
        * LightGBM memiliki precision 0.27.

    - Recall untuk kelas 1 juga sangat rendah, yang menunjukkan bahwa model-model ini kurang efektif dalam mengidentifikasi semua pasien yang sebenarnya memiliki stroke.
        * Random Forest memiliki recall yang sangat buruk, hanya 0.01.
        * XGBoost memiliki recall 0.12, yang terbaik di antara ketiganya, tetapi masih rendah.
        * LightGBM memiliki recall 0.06.

    - F1-score, yang merupakan rata-rata harmonik dari precision dan recall, juga rendah. Ini mencerminkan trade-off antara precision dan recall yang buruk dalam memprediksi stroke.
        * Random Forest memiliki F1-score terendah, 0.03.
        * XGBoost memiliki F1-score 0.16.
        * LightGBM memiliki F1-score 0.10.

    Berdasarkan hasil evaluasi, model **`Random Forest`** menunjukkan akurasi tertinggi **(0.9550)**, tetapi perbedaannya dengan **XGBoost** dan **LightGBM** kecil. Mengingat keseimbangan antara Precision dan Recall, XGBoost dipilih sebagai solusi final karena memiliki kombinasi Precision dan Recall yang lebih baik dibandingkan model lain.

2. Confusion Matrix :

    | Model         |    Actual           | Predicted Negatif (0) |  Predicted Positif (1)  |
    | -----         |----------------     |---------------        |--------------------     |
    |Random Forest  | Actual Negatif (0)  | 1463             	  | 2                     |
    |Random Forest  | Actual Positif (1)  | 67               	  | 1                    |
    |XGBoost        | Actual Negatif (0)  | 1444             	  | 21                      |
    |XGBoost        | Actual Positif (1)  | 60               	  | 8                    |
    |LightGBM       | Actual Negatif (0)  | 1454            	  | 11                      |
    |LightGBM       | Actual Positif (1)  | 64               	  | 4                    |

    Berdasarkan confusion matrix, ketiga model memiliki performa yang sangat baik dalam mengklasifikasikan kasus negatif, yaitu pasien yang tidak mengalami stroke, dengan jumlah True Negative yang tinggi dan False Positive yang sangat rendah. Ini menunjukkan bahwa model jarang salah memprediksi seseorang tidak mengalami stroke. Namun, terdapat perbedaan dalam menangani kasus positif, yaitu pasien yang mengalami stroke. Random Forest memiliki jumlah False Negative tertinggi (67), yang berarti model ini paling banyak melewatkan kasus stroke dibandingkan model lain. Hanya 1 kasus stroke yang terdeteksi dengan benar. LightGBM memiliki 64 False Negative, sedikit lebih baik dari Random Forest, tetapi masih mengindikasikan banyak kasus stroke yang tidak terdeteksi. XGBoost memiliki jumlah False Negative terendah (60), menunjukkan bahwa model ini paling sedikit melewatkan kasus stroke. Selain itu, Random Forest memiliki False Positive terendah (2), yang berarti model ini paling jarang salah memprediksi seseorang mengalami stroke padahal tidak. LightGBM memiliki 11 False Positive, sedikit lebih banyak dari Random Forest. XGBoost memiliki 21 False Positive, yang terbanyak di antara ketiga model, menunjukkan kecenderungan untuk sedikit lebih sering salah memprediksi seseorang mengalami stroke. Meskipun ketiga model sangat baik dalam mengidentifikasi pasien yang tidak mengalami stroke, terdapat perbedaan signifikan dalam kemampuan mereka mendeteksi kasus stroke. XGBoost menunjukkan keseimbangan terbaik dalam hal ini, dengan jumlah False Negative terendah (paling sedikit melewatkan kasus stroke) meskipun memiliki jumlah False Positive tertinggi (sedikit lebih banyak salah memprediksi stroke). Namun, perlu diingat bahwa dalam konteks medis, mengurangi False Negative (gagal mendeteksi stroke) sering kali lebih penting daripada mengurangi False Positive, karena gagal mendeteksi stroke dapat menunda penanganan yang kritis.
