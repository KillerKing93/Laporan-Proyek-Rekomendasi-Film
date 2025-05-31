# **Laporan Proyek Akhir Machine Learning: Sistem Rekomendasi Film**

- **Nama:** Alif Nurhidayat
- **Email:** alifnurhidayatwork@gmail.com | mc189d5y0351@student.devacademy.id
- **ID Dicoding:** Alif Nurhidayat | MC189D5Y0351

## **Pendahuluan (Project Overview)**

### **Latar Belakang Proyek**

Di era digital saat ini, jumlah konten yang tersedia bagi pengguna, seperti film, musik, berita, dan produk, sangatlah melimpah. Hal ini seringkali membuat pengguna kesulitan untuk menemukan item yang benar-benar mereka sukai atau butuhkan. Sistem rekomendasi hadir sebagai solusi untuk membantu pengguna menavigasi lautan informasi ini dengan menyajikan pilihan-pilihan yang paling relevan secara personal.

Dalam konteks industri perfilman dan platform streaming, sistem rekomendasi memainkan peran krusial. Dengan merekomendasikan film yang tepat, platform dapat meningkatkan kepuasan pengguna, memperpanjang waktu interaksi pengguna dengan platform (_engagement_), dan pada akhirnya meningkatkan retensi pengguna serta potensi pendapatan. Rekomendasi yang akurat membantu pengguna menemukan film baru yang mungkin mereka lewatkan, serta menemukan kembali film lama yang sesuai dengan selera mereka.

### **Mengapa Proyek Ini Penting Untuk Diselesaikan**

Proyek ini penting untuk diselesaikan karena memberikan pemahaman praktis tentang bagaimana membangun sistem rekomendasi dari awal hingga akhir, mencakup pemrosesan data, implementasi algoritma, hingga evaluasi. Keterampilan ini sangat relevan di berbagai industri yang mengandalkan personalisasi untuk meningkatkan pengalaman pengguna. Dengan menguasai teknik-teknik dalam sistem rekomendasi, kita dapat berkontribusi dalam menciptakan pengalaman pengguna yang lebih baik dan lebih personal di berbagai aplikasi digital. Selain itu, proyek ini juga menjadi sarana untuk mengaplikasikan konsep-konsep _machine learning_ yang telah dipelajari ke dalam sebuah kasus nyata yang memiliki dampak signifikan.

### **Referensi**

1.  **Dataset:** Dataset yang digunakan adalah "MovieLens Small Dataset" yang bersumber dari Kaggle ([https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset)). Dataset ini merupakan versi kecil dari dataset MovieLens yang lebih besar, dikumpulkan dan dikelola oleh GroupLens Research dari University of Minnesota. Dataset ini populer digunakan untuk tujuan edukasi dan riset awal sistem rekomendasi karena ukurannya yang relatif mudah dikelola namun tetap kaya akan informasi rating dan metadata film.
2.  **Teknik Sistem Rekomendasi:**
    - **Content-Based Filtering:** Pendekatan ini dijelaskan secara mendalam dalam berbagai literatur, salah satunya adalah Aggarwal, C. C. (2016). _Recommender Systems: The Textbook_. Springer. Bab ini membahas bagaimana atribut item dapat digunakan untuk membangun profil dan menghitung kemiripan.
    - **Collaborative Filtering:** Buku yang sama juga membahas berbagai teknik _collaborative filtering_, termasuk pendekatan berbasis tetangga (_neighborhood-based_) dan faktorisasi matriks (_matrix factorization_) seperti SVD.
    - **TF-IDF:** Manning, C. D., Raghavan, P., & Schütze, H. (2008). _Introduction to Information Retrieval_. Cambridge University Press. Buku ini memberikan penjelasan detail mengenai Term Frequency-Inverse Document Frequency (TF-IDF) sebagai teknik pembobotan fitur teks.
    - **Cosine Similarity:** Teknik ini umum digunakan untuk mengukur kemiripan antar vektor dalam ruang berdimensi tinggi, seperti yang dijelaskan dalam banyak sumber standar _machine learning_ dan _information retrieval_.
    - **Singular Value Decomposition (SVD):** Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix Factorization Techniques for Recommender Systems." _Computer_, 42(8), 30-37. Artikel ini merupakan salah satu referensi kunci yang mempopulerkan penggunaan SVD dalam konteks sistem rekomendasi, terutama setelah keberhasilannya dalam Netflix Prize.
    - **Library Surprise:** Dokumentasi resmi library Surprise ([https://surprise.readthedocs.io/](https://surprise.readthedocs.io/)) menyediakan panduan lengkap mengenai penggunaan berbagai algoritma _collaborative filtering_, termasuk SVD.

## **1. Business Understanding**

### **Problem Statements**

Berdasarkan latar belakang di atas, permasalahan yang ingin diselesaikan dalam proyek ini adalah:

1.  Bagaimana cara membangun sistem rekomendasi film yang dapat memberikan rekomendasi berdasarkan kemiripan konten film (dalam kasus ini, genre film)? (_Content-Based Filtering_)
2.  Bagaimana cara membangun sistem rekomendasi film yang dapat memberikan rekomendasi berdasarkan pola preferensi pengguna lain yang memiliki selera serupa, menggunakan data rating historis? (_Collaborative Filtering_)
3.  Fitur genre film, ketika direpresentasikan menggunakan TF-IDF, seberapa efektif dapat digunakan untuk menentukan kemiripan antar film dalam pendekatan _content-based_?
4.  Bagaimana performa model _Collaborative Filtering_ berbasis SVD dalam memprediksi rating film, diukur menggunakan metrik RMSE dan MAE?

### **Goals**

Tujuan utama dari proyek ini adalah:

1.  Mengembangkan model _Content-Based Filtering_ yang mampu merekomendasikan 10 film teratas berdasarkan kemiripan genre dengan film yang diberikan sebagai input.
2.  Mengembangkan model _Collaborative Filtering_ menggunakan algoritma Singular Value Decomposition (SVD) yang mampu memprediksi rating film dan menghasilkan 10 rekomendasi film teratas untuk pengguna tertentu.
3.  Mengevaluasi model _Collaborative Filtering_ (SVD) menggunakan metrik Root Mean Squared Error (RMSE) dan Mean Absolute Error (MAE) untuk mengukur akurasi prediksi ratingnya.
4.  Menganalisis secara kualitatif hasil rekomendasi dari kedua pendekatan dan membahas kelebihan serta kekurangan masing-masing.

### **Solution Approach**

Untuk mencapai tujuan-tujuan tersebut, dua pendekatan solusi sistem rekomendasi akan diimplementasikan dan dievaluasi:

1.  **Solusi A: Content-Based Filtering**

    - **Teknik:** Pendekatan ini akan merekomendasikan film berdasarkan kemiripan atribut konten, yaitu genre film.
    - **Langkah-langkah:**
      1.  Pra-pemrosesan data genre: Mengubah format genre agar dapat diproses oleh TF-IDF Vectorizer (misalnya, mengganti pemisah '|' dengan spasi).
      2.  Menerapkan TF-IDF Vectorizer pada data genre untuk mengubah teks genre menjadi representasi vektor numerik.
      3.  Menghitung matriks kemiripan antar film menggunakan metrik _cosine similarity_ berdasarkan vektor TF-IDF genre.
      4.  Membuat fungsi yang menerima judul film sebagai input dan menghasilkan daftar top-N (N=10) film lain yang paling mirip berdasarkan skor _cosine similarity_.
    - **Evaluasi Solusi A:** Evaluasi akan bersifat kualitatif dengan melihat relevansi film yang direkomendasikan berdasarkan kesamaan genre dengan film input.

2.  **Solusi B: Collaborative Filtering menggunakan Singular Value Decomposition (SVD)**
    - **Teknik:** Pendekatan ini akan merekomendasikan film berdasarkan pola rating dari pengguna di masa lalu, dengan asumsi bahwa pengguna yang memiliki preferensi serupa di masa lalu akan memiliki preferensi serupa di masa depan.
    - **Langkah-langkah:**
      1.  Mempersiapkan data rating untuk digunakan dengan library `Surprise`. Ini melibatkan pendefinisian `Reader` untuk skala rating dan memuat data dari DataFrame pandas.
      2.  Membagi dataset rating menjadi data latih dan data uji.
      3.  Melatih model SVD pada data latih. Parameter SVD yang digunakan (sesuai notebook): `n_factors=100`, `n_epochs=20`, `lr_all=0.005`, `reg_all=0.02`, `random_state=42`.
      4.  Membuat fungsi yang menerima ID pengguna sebagai input, memprediksi rating untuk film yang belum dirating oleh pengguna tersebut, dan menghasilkan daftar top-N (N=10) film dengan prediksi rating tertinggi.
    - **Evaluasi Solusi B:** Performa model SVD akan dievaluasi secara kuantitatif menggunakan metrik RMSE dan MAE pada data uji. Nilai RMSE dan MAE yang lebih rendah menunjukkan performa prediksi rating yang lebih baik.

Pemilihan pendekatan terbaik akan didasarkan pada analisis kualitatif untuk _Content-Based Filtering_ dan hasil metrik evaluasi kuantitatif untuk _Collaborative Filtering_. Kedua pendekatan ini memberikan jenis rekomendasi yang berbeda dan dapat saling melengkapi.

## **2. Data Understanding**

### **Sumber Data**

Dataset yang digunakan adalah "MovieLens Small Dataset" yang bersumber dari platform kompetisi Kaggle.

- **Tautan:** [https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset)

Dataset ini terdiri dari beberapa file, namun untuk proyek ini, fokus utama adalah pada dua file berikut:

- `movies.csv`: Berisi informasi metadata film.
- `ratings.csv`: Berisi informasi rating yang diberikan oleh pengguna terhadap film.

### **Jumlah Data dan Informasi Awal**

Setelah memuat dataset:

- **movies_df (dari movies.csv):**
  - Jumlah sampel (film): 9742
  - Jumlah kolom: 3 (`movieId`, `title`, `genres`)
- **ratings_df (dari ratings.csv):**
  - Jumlah sampel (rating): 100836
  - Jumlah kolom: 4 (`userId`, `movieId`, `rating`, `timestamp`)

Informasi tambahan dari inspeksi data awal:

- Jumlah film unik: 9742
- Jumlah pengguna unik: 610
- Jumlah total rating: 100836
- Rentang nilai rating: 0.5 hingga 5.0 (dalam skala interval 0.5)

### **Uraian Variabel/Fitur pada Data**

Berikut adalah penjelasan untuk setiap variabel/fitur yang relevan dari kedua file CSV:

**1. `movies.csv`**

- `movieId`:
  - Deskripsi: ID unik untuk setiap film.
  - Tipe Data: Numerik (Integer).
  - Sifat: Identifier, Kategorikal diskrit.
- `title`:
  - Deskripsi: Judul film, biasanya disertai tahun rilis dalam tanda kurung.
  - Tipe Data: Teks (String).
  - Sifat: Kategorikal.
- `genres`:
  - Deskripsi: Genre atau beberapa genre film, dipisahkan oleh karakter pipa (`|`). Contoh: "Adventure|Animation|Children|Comedy|Fantasy".
  - Tipe Data: Teks (String).
  - Sifat: Kategorikal (multivalue). Ini akan menjadi fitur utama untuk _Content-Based Filtering_.

**2. `ratings.csv`**

- `userId`:
  - Deskripsi: ID unik untuk setiap pengguna.
  - Tipe Data: Numerik (Integer).
  - Sifat: Identifier, Kategorikal diskrit.
- `movieId`:
  - Deskripsi: ID unik untuk setiap film yang dirating. Ini merupakan _foreign key_ yang merujuk ke `movieId` pada `movies.csv`.
  - Tipe Data: Numerik (Integer).
  - Sifat: Identifier, Kategorikal diskrit.
- `rating`:
  - Deskripsi: Rating yang diberikan oleh pengguna untuk film tersebut. Skala rating adalah dari 0.5 hingga 5.0, dengan interval 0.5.
  - Tipe Data: Numerik (Float).
  - Sifat: Numerik kontinu (ordinal). Ini adalah variabel target implisit untuk _Collaborative Filtering_ (prediksi rating).
- `timestamp`:
  - Deskripsi: Waktu saat rating diberikan, dalam format detik sejak epoch (Unix timestamp).
  - Tipe Data: Numerik (Integer).
  - Sifat: Numerik. Fitur ini tidak digunakan secara langsung dalam model rekomendasi pada proyek ini, namun bisa relevan untuk analisis temporal atau model yang lebih canggih.

### **Penanganan Missing Values dan Duplikat**

- **Missing Values:**
  - `movies_df`: Tidak ditemukan nilai yang hilang pada kolom `movieId`, `title`, maupun `genres`.
  - `ratings_df`: Tidak ditemukan nilai yang hilang pada kolom `userId`, `movieId`, `rating`, maupun `timestamp`.
  - Kesimpulan: Dataset relatif bersih dari nilai yang hilang pada kolom-kolom yang akan digunakan.
- **Duplicated Data:**
  - `movies_df`: Tidak ditemukan baris data yang duplikat.
  - `ratings_df`: Tidak ditemukan baris data yang duplikat.
  - Kesimpulan: Tidak ada data duplikat yang perlu ditangani.

### **Analisis Data Eksploratif (EDA) & Visualisasi**

Analisis data eksploratif dilakukan untuk mendapatkan pemahaman yang lebih mendalam mengenai karakteristik data.

- **Distribusi Nilai Rating Film:**

  - Visualisasi: _Countplot_ untuk kolom `rating` pada `ratings_df`.
  - [Gambar Distribusi Nilai Rating Film]
  - Insight:
    - Rating paling sering diberikan adalah 4.0, diikuti oleh 3.0 dan 5.0.
    - Rating 3.5 dan 4.5 juga cukup umum.
    - Rating rendah (0.5, 1.0, 1.5) lebih jarang diberikan.
    - Distribusi menunjukkan bahwa pengguna cenderung memberikan rating yang positif atau netral-positif.

- **Top 15 Genre Film Paling Umum:**

  - Proses: Kolom `genres` pada `movies_df` dipisahkan berdasarkan karakter `|`, kemudian frekuensi kemunculan setiap genre dihitung.
  - Visualisasi: _Bar chart_ menampilkan 15 genre dengan jumlah film terbanyak.
  - [Gambar Top 15 Genre Film Paling Umum]
  - Insight:
    - Genre Drama adalah yang paling umum, diikuti oleh Comedy, Thriller, dan Action.
    - Genre seperti Romance, Adventure, dan Sci-Fi juga memiliki representasi yang signifikan.
    - Ini memberikan gambaran mengenai komposisi genre dalam dataset, yang penting untuk _Content-Based Filtering_.

- **Top 15 Pengguna dengan Jumlah Rating Terbanyak:**

  - Proses: Menghitung jumlah rating yang diberikan oleh setiap `userId` pada `ratings_df`.
  - Visualisasi: _Bar chart_ menampilkan 15 pengguna paling aktif.
  - [Gambar Top 15 Pengguna dengan Jumlah Rating Terbanyak]
  - Insight:
    - Terdapat variasi signifikan dalam aktivitas pengguna. Pengguna dengan `userId` 414 adalah yang paling aktif, memberikan lebih dari 2500 rating.
    - Memahami distribusi aktivitas pengguna penting untuk _Collaborative Filtering_, karena pengguna dengan sedikit rating (cold start user) lebih sulit untuk dimodelkan.

- **Top 15 Film dengan Jumlah Rating Terbanyak:**
  - Proses: Menghitung jumlah rating yang diterima oleh setiap `movieId` pada `ratings_df`, kemudian digabungkan dengan `movies_df` untuk mendapatkan judul film.
  - Visualisasi: _Bar chart_ menampilkan 15 film yang paling banyak dirating.
  - [Gambar Top 15 Film dengan Jumlah Rating Terbanyak]
  - Insight:
    - Film seperti "Forrest Gump (1994)", "Shawshank Redemption, The (1994)", dan "Pulp Fiction (1994)" adalah yang paling populer (paling banyak dirating).
    - Ini menunjukkan bahwa beberapa film memiliki popularitas yang jauh lebih tinggi dibandingkan yang lain, yang dapat mempengaruhi hasil rekomendasi.

**Insight Tambahan dari EDA:**

- Dataset ini memiliki tingkat _sparsity_ yang perlu dipertimbangkan untuk _Collaborative Filtering_, meskipun untuk "small dataset" ini masih relatif padat dibandingkan dataset rating yang sangat besar.
- Kombinasi genre yang beragam menunjukkan potensi untuk _Content-Based Filtering_ yang menarik.

## **3. Data Preparation**

Tahapan persiapan data dilakukan untuk memastikan data siap digunakan oleh algoritma _machine learning_. Proses ini berbeda untuk kedua pendekatan rekomendasi.

**Untuk Content-Based Filtering:**

1.  **Pembuatan Salinan DataFrame Film:**
    - Proses: `movies_df_cb = movies_df.copy()` dibuat untuk menjaga DataFrame asli tidak berubah.
    - Alasan: Praktik yang baik untuk menghindari modifikasi data asli secara tidak sengaja.
2.  **Pemrosesan Kolom Genre:**
    - Proses: Karakter pipa (`|`) dalam kolom `genres` diganti dengan spasi. Contoh: "Adventure|Animation|Children" menjadi "Adventure Animation Children".
      ```python
      movies_df_cb['genres_processed'] = movies_df_cb['genres'].str.replace('|', ' ')
      ```
    - Alasan: Ini dilakukan agar setiap genre dapat dianggap sebagai "kata" atau token terpisah oleh `TfidfVectorizer`. Jika pemisah tidak diganti, "Adventure|Animation" akan dianggap sebagai satu token unik, bukan dua token "Adventure" dan "Animation".
3.  **TF-IDF Vectorization pada Genre:**
    - Proses: `TfidfVectorizer` dari `sklearn.feature_extraction.text` diinisialisasi dan diterapkan pada kolom `genres_processed`.
      ```python
      tfidf = TfidfVectorizer()
      tfidf_matrix = tfidf.fit_transform(movies_df_cb['genres_processed'])
      ```
      Ini menghasilkan matriks TF-IDF (`tfidf_matrix`) di mana setiap baris merepresentasikan satu film dan setiap kolom merepresentasikan satu genre unik yang telah dibobotkan dengan skor TF-IDF.
    - Alasan: TF-IDF (Term Frequency-Inverse Document Frequency) mengubah data teks genre menjadi representasi numerik (vektor). Skor TF-IDF memberikan bobot yang lebih tinggi pada genre yang sering muncul dalam satu film tetapi jarang muncul di seluruh koleksi film, sehingga menyoroti genre yang lebih distingtif untuk sebuah film. Ini penting untuk menghitung kemiripan konten.
4.  **Pembuatan Indeks Judul Film:**
    - Proses: Sebuah Pandas Series (`indices_cb`) dibuat yang memetakan judul film ke indeks barisnya dalam `movies_df_cb`. Duplikat judul (jika ada) dihilangkan dengan mempertahankan yang pertama.
      ```python
      indices_cb = pd.Series(movies_df_cb.index, index=movies_df_cb['title']).drop_duplicates()
      ```
    - Alasan: Ini memudahkan pencarian film berdasarkan judulnya untuk mendapatkan vektor TF-IDF yang sesuai saat membuat rekomendasi.

**Untuk Collaborative Filtering (menggunakan library `Surprise`):**

1.  **Persiapan Data Rating untuk `Surprise`:**
    - Proses:
      1.  Sebuah `Reader` dari `surprise` diinisialisasi dengan `rating_scale=(0.5, 5.0)` untuk memberitahu library rentang nilai rating yang digunakan.
      2.  Dataset dimuat dari DataFrame `ratings_df` (hanya kolom `userId`, `movieId`, `rating`) menggunakan `Dataset.load_from_df()`.
          ```python
          reader = Reader(rating_scale=(0.5, 5.0))
          data_surprise = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
          ```
    - Alasan: Library `Surprise` memiliki format data internalnya sendiri. Langkah ini mengkonversi data rating dari format Pandas DataFrame ke format yang dapat diproses oleh algoritma dalam `Surprise`.
2.  **Pembagian Data Latih dan Data Uji:**
    - Proses: Dataset `data_surprise` dibagi menjadi data latih (`trainset_surprise`) dan data uji (`testset_surprise`) dengan rasio 80:20, menggunakan `surprise_train_test_split` dengan `random_state=42` untuk reproduktifitas.
      ```python
      trainset_surprise, testset_surprise = surprise_train_test_split(data_surprise, test_size=0.2, random_state=42)
      ```
    - Alasan: Data latih digunakan untuk melatih model SVD, sedangkan data uji digunakan untuk mengevaluasi seberapa baik model tersebut memprediksi rating pada data yang belum pernah dilihatnya. Ini penting untuk mengukur kemampuan generalisasi model.

Tidak ada _feature scaling_ atau _encoding_ eksplisit yang diperlukan untuk data rating dalam konteks penggunaan library `Surprise` dengan SVD, karena algoritma SVD bekerja langsung dengan ID pengguna dan ID item serta nilai rating. Library `Surprise` menangani pemetaan ID internal secara otomatis.

## **4. Modeling and Result**

Dua pendekatan model sistem rekomendasi dikembangkan dan diimplementasikan: _Content-Based Filtering_ dan _Collaborative Filtering_.

### **Model 1: Content-Based Filtering**

Pendekatan ini merekomendasikan item (film) berdasarkan kemiripan atribut atau kontennya. Dalam proyek ini, fitur genre film digunakan sebagai dasar untuk menentukan kemiripan.

- **Algoritma dan Teknik yang Digunakan:**

  1.  **TF-IDF (Term Frequency-Inverse Document Frequency):** Seperti dijelaskan pada tahap Data Preparation, genre setiap film diubah menjadi vektor numerik menggunakan TF-IDF. Matriks TF-IDF (`tfidf_matrix`) yang dihasilkan memiliki dimensi (jumlah film x jumlah genre unik). Setiap entri dalam matriks ini merepresentasikan bobot TF-IDF dari suatu genre untuk suatu film.
  2.  **Cosine Similarity:** Setelah mendapatkan representasi vektor TF-IDF untuk setiap film, kemiripan antar film dihitung menggunakan metrik _cosine similarity_. _Cosine similarity_ mengukur kosinus sudut antara dua vektor, yang memberikan ukuran seberapa mirip orientasi kedua vektor tersebut. Skor kemiripan berkisar antara 0 (tidak mirip) hingga 1 (sangat mirip/identik).
      ```python
      from sklearn.metrics.pairwise import cosine_similarity
      cosine_sim_genre = cosine_similarity(tfidf_matrix, tfidf_matrix)
      ```
      Hasilnya adalah matriks `cosine_sim_genre` berukuran (jumlah film x jumlah film), di mana setiap entri (i, j) adalah skor kemiripan antara film i dan film j.

- **Fungsi Rekomendasi `get_recommendations_content_based`:**
  Sebuah fungsi Python dibuat untuk menghasilkan rekomendasi:

  ```python
  def get_recommendations_content_based(title, cosine_sim=cosine_sim_genre, df=movies_df_cb, indices=indices_cb, top_n=10):
      # ... (implementasi fungsi seperti di notebook) ...
      return recommendations
  ```

  Fungsi ini bekerja sebagai berikut:

  1.  Menerima `title` film sebagai input.
  2.  Mencari indeks film tersebut dalam DataFrame menggunakan `indices_cb`.
  3.  Mengambil baris skor kemiripan film tersebut dengan semua film lain dari matriks `cosine_sim_genre`.
  4.  Mengurutkan film berdasarkan skor kemiripan secara menurun.
  5.  Mengambil `top_n` (default 10) film dengan skor kemiripan tertinggi (tidak termasuk film input itu sendiri).
  6.  Mengembalikan DataFrame berisi judul, genre, dan skor kemiripan dari film-film yang direkomendasikan.

- **Hasil Top-N Recommendation (Contoh):**
  Berikut adalah contoh output rekomendasi untuk beberapa film (sesuai notebook):

  - **Rekomendasi untuk 'Toy Story (1995)':**

    ```
                                                  title                                            genres  similarity_score
    1706                                        Antz (1998)  Adventure|Animation|Children|Comedy|Fantasy               1.0
    2355                                 Toy Story 2 (1999)  Adventure|Animation|Children|Comedy|Fantasy               1.0
    2809     Adventures of Rocky and Bullwinkle, The (2000)  Adventure|Animation|Children|Comedy|Fantasy               1.0
    3000                   Emperor's New Groove, The (2000)  Adventure|Animation|Children|Comedy|Fantasy               1.0
    3568                              Monsters, Inc. (2001)  Adventure|Animation|Children|Comedy|Fantasy               1.0
    ...                                                 ...                                           ...               ...
    ```

    Film-film yang direkomendasikan memiliki genre yang identik atau sangat mirip (Adventure, Animation, Children, Comedy, Fantasy) dengan "Toy Story". Skor kemiripan 1.0 menunjukkan kesamaan genre yang sempurna.

  - **Rekomendasi untuk 'Jumanji (1995)':**

    ```
                                                  title                      genres  similarity_score
    53                   Indian in the Cupboard, The (1995)  Adventure|Children|Fantasy               1.0
    109                   NeverEnding Story III, The (1994)  Adventure|Children|Fantasy               1.0
    ...                                                 ...                             ...               ...
    ```

    Rekomendasi untuk "Jumanji" juga menghasilkan film-film dengan genre yang sama (Adventure, Children, Fantasy).

  - **Rekomendasi untuk 'Heat (1995)':**
    ```
                                  title                 genres  similarity_score
    22                     Assassins (1995)  Action|Crime|Thriller               1.0
    138   Die Hard: With a Vengeance (1995)  Action|Crime|Thriller               1.0
    ...                                   ...                    ...               ...
    ```
    Film-film yang direkomendasikan untuk "Heat" adalah film dengan genre Action, Crime, dan Thriller, yang sesuai.

- **Kelebihan Content-Based Filtering:**

  - Tidak memerlukan data dari pengguna lain (mengatasi masalah _cold start_ untuk item baru jika item tersebut memiliki deskripsi atribut yang cukup).
  - Dapat merekomendasikan item yang spesifik dan _niche_ yang mungkin tidak populer tetapi sesuai dengan preferensi konten pengguna.
  - Rekomendasi bersifat transparan dan mudah dijelaskan (misalnya, "direkomendasikan karena genrenya sama").

- **Kekurangan Content-Based Filtering:**
  - Terbatas pada fitur item yang ada. Jika fitur (misalnya, genre) kurang deskriptif atau tidak lengkap, kualitas rekomendasi bisa menurun.
  - Cenderung menghasilkan rekomendasi yang terlalu mirip dengan apa yang sudah disukai pengguna (_overspecialization_ atau _filter bubble_), sehingga kurang mampu mengeksplorasi item baru yang berbeda namun mungkin disukai.
  - Membutuhkan _domain knowledge_ untuk membuat profil fitur item yang baik dan memilih fitur yang relevan.

### **Model 2: Collaborative Filtering (menggunakan SVD)**

Pendekatan ini merekomendasikan item berdasarkan kemiripan preferensi antar pengguna atau antar item, yang dideduksi dari data rating historis.

- **Algoritma dan Teknik yang Digunakan:**

  1.  **Singular Value Decomposition (SVD):** SVD adalah teknik faktorisasi matriks yang populer untuk _collaborative filtering_. Ide dasarnya adalah untuk menguraikan matriks interaksi pengguna-item (matriks rating) menjadi produk dari tiga matriks yang lebih kecil. Dalam konteks sistem rekomendasi, SVD digunakan untuk menemukan faktor-faktor laten (tersembunyi) yang merepresentasikan preferensi pengguna dan karakteristik item. Model SVD dari library `Surprise` digunakan.
      ```python
      from surprise import SVD
      svd_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
      svd_model.fit(trainset_surprise)
      ```
      Parameter yang digunakan:
      - `n_factors=100`: Jumlah faktor laten.
      - `n_epochs=20`: Jumlah iterasi proses training.
      - `lr_all=0.005`: Laju pembelajaran.
      - `reg_all=0.02`: Term regulariasi.
      - `random_state=42`: Untuk reproduktifitas.

- **Fungsi Rekomendasi `get_top_n_recommendations_svd`:**
  Sebuah fungsi Python dibuat untuk menghasilkan rekomendasi bagi pengguna tertentu:

  ```python
  def get_top_n_recommendations_svd(user_id, svd_model, movies_df, ratings_df, n=10):
      # ... (implementasi fungsi seperti di notebook) ...
      return recommendation_details
  ```

  Fungsi ini bekerja sebagai berikut:

  1.  Menerima `user_id` sebagai input.
  2.  Mengidentifikasi semua film yang ada dalam dataset.
  3.  Mengidentifikasi film-film yang sudah pernah dirating oleh `user_id` tersebut.
  4.  Membuat daftar film yang belum dirating oleh `user_id`.
  5.  Menggunakan model `svd_model` yang sudah dilatih untuk memprediksi rating yang mungkin diberikan oleh `user_id` untuk setiap film yang belum dirating.
  6.  Mengurutkan film-film yang belum dirating berdasarkan estimasi rating tertinggi.
  7.  Mengambil `top_n` (default 10) film dengan prediksi rating tertinggi.
  8.  Menggabungkan hasil dengan metadata film (judul, genre) dan mengembalikan DataFrame rekomendasi.

- **Hasil Top-N Recommendation (Contoh):**
  Berikut adalah contoh output rekomendasi untuk beberapa pengguna (sesuai notebook):

  - **Rekomendasi Film untuk User ID 1 (SVD):**

    ```
         movieId                                              title                                          genres  estimated_rating
    474      541                                Blade Runner (1982)                        Action|Sci-Fi|Thriller               5.0
    596      741         Ghost in the Shell (KÃ´kaku kidÃ´tai) (1995)                              Animation|Sci-Fi               5.0
    ...      ...                                                ...                                             ...               ...
    ```

    Model SVD merekomendasikan film-film klasik dan _cult_ dengan prediksi rating tinggi (5.0) untuk pengguna ini, seperti "Blade Runner" dan "Ghost in the Shell".

  - **Rekomendasi Film untuk User ID 50 (SVD):**
    ```
          movieId                                              title                                genres  estimated_rating
    2020     2692                   Run Lola Run (Lola rennt) (1998)                        Action|Crime             3.598
    585       720  Wallace & Gromit: The Best of Aardman Animatio...          Adventure|Animation|Comedy             3.573
    ...       ...                                                ...                                   ...               ...
    ```
    Untuk pengguna 50, model merekomendasikan film seperti "Run Lola Run" dan "Wallace & Gromit" dengan estimasi rating yang lebih bervariasi.

- **Kelebihan Collaborative Filtering (SVD):**

  - Tidak memerlukan informasi fitur item secara eksplisit, hanya berdasarkan interaksi pengguna-item (rating).
  - Mampu menemukan item baru yang menarik bagi pengguna secara tak terduga (_serendipity_) berdasarkan preferensi pengguna lain yang memiliki selera serupa.
  - Umumnya memberikan hasil yang baik dan akurat jika data rating cukup banyak dan tidak terlalu _sparse_.

- **Kekurangan Collaborative Filtering (SVD):**
  - Mengalami masalah _cold start_: sulit memberikan rekomendasi untuk pengguna baru (yang belum memberikan rating) atau item baru (yang belum menerima rating).
  - _Data sparsity_: Jika matriks interaksi pengguna-item sangat jarang (banyak pengguna hanya merating sedikit item, atau banyak item hanya dirating oleh sedikit pengguna), performa model bisa menurun karena kurangnya data untuk menemukan pola.
  - Kurang transparan dalam menjelaskan mengapa suatu item direkomendasikan dibandingkan _Content-Based Filtering_. Rekomendasi didasarkan pada faktor laten yang mungkin tidak mudah diinterpretasikan.

## **5. Evaluasi**

Evaluasi sistem rekomendasi dilakukan secara berbeda untuk kedua pendekatan.

**Untuk Content-Based Filtering:**
Evaluasi untuk model _Content-Based Filtering_ bersifat **kualitatif**. Ini dilakukan dengan mengamati relevansi film-film yang direkomendasikan berdasarkan genre.

- **Hasil:** Seperti yang ditunjukkan pada contoh di bagian "Modeling and Result", rekomendasi yang dihasilkan oleh pendekatan _content-based_ (menggunakan kemiripan genre) secara umum sangat relevan. Film-film yang direkomendasikan memiliki genre yang sama atau sangat mirip dengan film input. Misalnya, untuk "Toy Story (1995)" (Adventure|Animation|Children|Comedy|Fantasy), film-film seperti "Antz (1998)" dan "Toy Story 2 (1999)" yang juga memiliki genre serupa direkomendasikan dengan skor kemiripan tinggi.
- **Kesesuaian Metrik:** Evaluasi kualitatif ini sesuai karena tujuan utama _Content-Based Filtering_ di sini adalah menemukan item yang "mirip" berdasarkan atribut konten yang telah ditentukan (genre).

**Untuk Collaborative Filtering (SVD):**
Evaluasi untuk model _Collaborative Filtering_ berbasis SVD dilakukan secara **kuantitatif** menggunakan metrik yang mengukur akurasi prediksi rating pada data uji.

- **Metrik Evaluasi yang Digunakan:**

  1.  **RMSE (Root Mean Squared Error):**
      - Formula: RMSE = $\sqrt{\frac{1}{N} \sum (\text{actual\_rating} - \text{predicted\_rating})^2}$
      - Cara Kerja: RMSE mengukur rata-rata magnitudo dari error prediksi rating. Karena perbedaan rating dikuadratkan sebelum di-rata-rata dan diakarkan, RMSE memberikan bobot yang lebih besar pada error prediksi yang besar. Nilai RMSE yang lebih rendah menunjukkan performa model yang lebih baik dalam memprediksi rating secara akurat. Unitnya sama dengan unit rating.
      - Kesesuaian: Umum digunakan dalam sistem rekomendasi untuk mengevaluasi akurasi prediksi rating.
  2.  **MAE (Mean Absolute Error):**
      - Formula: MAE = $\frac{1}{N} \sum |\text{actual\_rating} - \text{predicted\_rating}|$
      - Cara Kerja: MAE mengukur rata-rata magnitudo absolut dari error prediksi rating. MAE memberikan bobot yang sama untuk semua error, besar maupun kecil. Seperti RMSE, nilai MAE yang lebih rendah lebih baik, dan unitnya juga sama dengan unit rating.
      - Kesesuaian: MAE lebih mudah diinterpretasikan secara langsung sebagai rata-rata seberapa jauh prediksi rating menyimpang dari rating aktual.

- **Hasil Proyek Berdasarkan Metrik Evaluasi (SVD pada `testset_surprise`):**
  (Sesuai output dari `accuracy.rmse(predictions_svd)` dan `accuracy.mae(predictions_svd)` di notebook)

  - **RMSE: 0.8807**
  - **MAE: 0.6766**

- **Interpretasi Hasil Metrik:**
  - RMSE sebesar 0.8807 menunjukkan bahwa, secara rata-rata, prediksi rating yang dihasilkan oleh model SVD memiliki simpangan sekitar 0.88 dari nilai rating aktual pada skala 0.5-5.0.
  - MAE sebesar 0.6766 berarti bahwa, secara rata-rata, prediksi rating menyimpang sebesar 0.6766 poin dari rating aktual.
  - Nilai-nilai ini menunjukkan bahwa model SVD memiliki kemampuan prediksi rating yang cukup baik pada dataset ini. Untuk konteks dataset MovieLens Small, nilai RMSE di bawah 0.9 sering dianggap sebagai hasil yang baik.

## **6. Kesimpulan**

Proyek ini berhasil mengimplementasikan dan mengevaluasi dua jenis sistem rekomendasi film: _Content-Based Filtering_ berdasarkan kemiripan genre dan _Collaborative Filtering_ menggunakan algoritma Singular Value Decomposition (SVD) pada data rating pengguna.

- **Content-Based Filtering** menunjukkan kemampuannya dalam merekomendasikan film-film yang memiliki genre serupa dengan film yang dijadikan acuan. Pendekatan ini efektif untuk pengguna yang ingin mengeksplorasi film dengan tema atau nuansa yang sudah mereka kenal dan sukai. Evaluasi kualitatif menunjukkan relevansi yang tinggi dari rekomendasi yang dihasilkan.

- **Collaborative Filtering (SVD)** berhasil dilatih untuk memprediksi rating yang mungkin diberikan pengguna terhadap film. Model SVD yang dikembangkan mencapai performa yang baik pada data uji, dengan **RMSE sebesar 0.8807** dan **MAE sebesar 0.6766**. Hasil ini menunjukkan bahwa model mampu menangkap pola preferensi pengguna dari data rating historis dan memberikan prediksi rating yang cukup akurat. Rekomendasi top-N yang dihasilkan berdasarkan prediksi rating ini dapat membantu pengguna menemukan film baru yang mungkin mereka sukai berdasarkan selera pengguna lain yang serupa.

Kedua pendekatan memiliki kelebihan dan kekurangannya masing-masing. _Content-Based Filtering_ unggul dalam transparansi dan penanganan item baru (selama memiliki fitur yang deskriptif), namun rentan terhadap _overspecialization_. _Collaborative Filtering_ mampu menghasilkan rekomendasi yang lebih beragam dan tak terduga (_serendipity_), namun menghadapi tantangan _cold start_ dan _data sparsity_.

Sebagai pengembangan lebih lanjut, beberapa area yang dapat dieksplorasi meliputi:

- Menggabungkan kedua pendekatan menjadi sistem rekomendasi _hybrid_ untuk memanfaatkan kelebihan masing-masing dan mengatasi kekurangannya.
- Mengeksplorasi fitur konten lain selain genre untuk _Content-Based Filtering_ (misalnya, sinopsis, aktor, sutradara).
- Mencoba algoritma _Collaborative Filtering_ lain (misalnya, berbasis KNN, atau teknik _matrix factorization_ yang lebih canggih).
- Melakukan _hyperparameter tuning_ yang lebih ekstensif untuk model SVD atau model lainnya.
- Mengimplementasikan metrik evaluasi yang berfokus pada ranking untuk top-N recommendation (misalnya, Precision@k, Recall@k, MAP, NDCG).
