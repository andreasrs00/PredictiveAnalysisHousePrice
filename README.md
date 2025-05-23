# Laporan Proyek Machine Learning - Andreas Robson Simanjuntak

## Domain Proyek

Pasar properti di Melbourne, Australia, telah mengalami pertumbuhan signifikan dalam beberapa tahun terakhir, dipengaruhi oleh faktor-faktor seperti urbanisasi, pertumbuhan ekonomi, dan preferensi gaya hidup masyarakat. Faktor-faktor seperti lokasi geografis, jumlah kamar, luas tanah, dan fasilitas sekitar (seperti akses ke transportasi umum, sekolah, dan pusat perbelanjaan) memiliki pengaruh besar terhadap harga properti.

Dataset Melbourne Housing Snapshot menyediakan informasi detail mengenai properti yang dijual di Melbourne, termasuk alamat, jenis properti, metode penjualan, jumlah kamar, harga, agen real estat, tanggal penjualan, dan jarak dari pusat bisnis kota (CBD). Data ini memungkinkan analisis mendalam terhadap faktor-faktor yang memengaruhi harga rumah di wilayah tersebut.

## Pentingnya Proyek

Prediksi harga rumah yang akurat memiliki manfaat luas bagi berbagai pemangku kepentingan:
- Pembeli dan Penjual: Membantu dalam pengambilan keputusan yang lebih baik dengan memahami nilai pasar properti berdasarkan lokasi dan fasilitas.
- Agen Real Estat: Menyediakan estimasi harga yang lebih akurat untuk klien, meningkatkan kepercayaan dan efisiensi dalam transaksi.
- Pengembang Properti: Membantu dalam perencanaan dan pengembangan proyek dengan memahami tren harga berdasarkan lokasi dan fasilitas yang tersedia.
- Pemerintah dan Perencana Kota: Memberikan wawasan untuk perencanaan infrastruktur dan kebijakan perumahan yang lebih efektif.

**Referensi:**

[1] ao, G., Bao, Z., Cao, J., Qin, A. K., Sellis, T., & Wu, Z. (2019). Location-Centered House Price Prediction: A Multi-Task Learning Approach. arXiv preprint arXiv:1901.01774.

[2] Hasan, M. H., Jahan, M. A., Ali, M. E., Li, Y. F., & Sellis, T. (2024). A Multi-Modal Deep Learning Based Approach for House Price Prediction. arXiv preprint arXiv:2409.05335.

## Business Understanding
### **Problem Statements:** <br>
Penentuan harga rumah di kota metropolitan seperti Melbourne sering kali menjadi tantangan karena melibatkan banyak variabel yang saling terkait, seperti lokasi, jumlah kamar, luas bangunan, dan fasilitas sekitar. Tanpa pendekatan berbasis data, estimasi harga rumah rentan terhadap bias dan ketidaktepatan. Para pembeli, penjual, dan agen properti membutuhkan sistem yang mampu memprediksi harga rumah secara cepat, objektif, dan akurat untuk mendukung pengambilan keputusan yang optimal.

### **Goals:** <br>
- Memprediksi harga rumah di Melbourne berdasarkan lokasi dan fasilitas.
- Membandingkan akurasi antara model Linear Regression dan Deep Neural Network (DNN).
- Memberikan wawasan berbasis data untuk mendukung keputusan pembelian dan penjualan properti.



### **Solution Statements:**
- Menggunakan dataset Melbourne Housing Snapshot dari Kaggle.
- Membangun model prediksi menggunakan Linear Regression sebagai baseline dan DNN untuk menangani hubungan non-linear.
- Mengevaluasi kinerja model dengan MAE, RMSE, R², dan MAPE.
- Menyediakan visualisasi prediksi vs aktual untuk membantu interpretasi hasil.

## Data Understanding
### **Dataset**<br>
Dataset yang digunakan dalam proyek ini adalah Melbourne Housing Snapshot dari Kaggle. Dataset ini merekam data penjualan rumah di wilayah Melbourne dan sekitarnya, mencakup berbagai aspek penting properti, baik dari sisi fisik maupun lokasional.

Data ini menyediakan gambaran komprehensif tentang karakteristik rumah yang beredar di pasar, dan sangat sesuai digunakan untuk membangun model prediktif harga rumah berdasarkan faktor-faktor yang tersedia.

markdown_content = """
# 📊 Fitur-Fitur Dataset Melbourne Housing Snapshot

Berikut adalah penjelasan lengkap mengenai semua fitur (kolom) yang terdapat dalam dataset *Melbourne Housing Snapshot* dari Kaggle.

---

## 🏘️ Informasi Properti

1. **Suburb**  
   Nama wilayah atau lingkungan tempat properti berada.

2. **Address**  
   Alamat lengkap dari properti.

3. **Rooms**  
   Total jumlah ruangan di dalam properti (termasuk kamar tidur, ruang tamu, dll).

4. **Type**  
   Jenis properti, terdiri dari:
   - `h` = House (rumah tinggal)
   - `u` = Unit / apartemen kecil
   - `t` = Townhouse

5. **Price**  
   Harga jual properti dalam satuan dolar Australia (AUD).

6. **Method**  
   Metode penjualan properti, seperti:
   - `S` = Sold
   - `SP` = Sold Prior (terjual sebelum lelang)
   - `PI` = Passed In (tidak terjual saat lelang)
   - `VB` = Vendor Bid (tawaran penjual)
   - `SA` = Sold After Auction
   - `PN` = Sold Prior (tanpa harga yang diungkap)
   - `SN` = Sold Not disclosed
   - `NB` = No Bid
   - `W` = Withdrawn
   - `SS` = Sold after auction (tidak diketahui harganya)
   - `N/A` = Informasi harga tidak tersedia

7. **SellerG**  
   Nama agen atau perusahaan real estat yang menjual properti.

8. **Date**  
   Tanggal terjadinya transaksi penjualan.

---

## 📍 Lokasi & Wilayah

9. **Distance**  
   Jarak properti dari pusat kota Melbourne (CBD) dalam kilometer.

10. **Postcode**  
    Kode pos lokasi properti.

11. **Bedroom2**  
    Jumlah kamar tidur di properti (dari sumber tambahan).

12. **Bathroom**  
    Jumlah kamar mandi di properti.

13. **Car**  
    Jumlah tempat parkir mobil yang tersedia.

14. **Landsize**  
    Luas tanah properti dalam meter persegi.

15. **BuildingArea**  
    Luas bangunan properti dalam meter persegi.

16. **YearBuilt**  
    Tahun properti dibangun.

17. **CouncilArea**  
    Nama dewan kota atau wilayah administratif yang mengelola area properti.

18. **Lattitude**  
    Garis lintang lokasi properti (koordinat GPS).

19. **Longtitude**  
    Garis bujur lokasi properti (koordinat GPS).

20. **Regionname**  
    Nama wilayah umum tempat properti berada, seperti "Northern Metropolitan", "Western Victoria", dll.

21. **Propertycount**  
    Jumlah total properti yang terdaftar di daerah suburb tersebut.

# 📊 Tahapan Awal Eksplorasi Data

1. **Import Libraries**  
   Memuat pustaka Python seperti pandas, numpy, matplotlib, seaborn, sklearn.

2. **Load Data**  
   Membaca dataset ke dalam DataFrame menggunakan `pd.read_csv()`.

3. **Data Cleaning & Type Conversion**  
   Mengubah tipe data dan membersihkan nilai tidak valid (mis. konversi tanggal, angka).

4. **Handle Missing Values**  
   Menangani data yang hilang dengan mengisi atau menghapus.

5. **Handle Duplicate Data**  
   Menghapus baris duplikat untuk menjaga kualitas data.

6. **Sort Data & Set Index**  
   Mengurutkan data berdasarkan kolom tertentu dan mereset index.

7. **Exploratory Data Analysis (EDA)**  
   Melakukan analisis awal: distribusi, outlier, korelasi.

8. **Summary**  
   Menampilkan statistik deskriptif dan tipe data.


### **Visualisasi yang di gunakan**:<br>
- Histogram: Untuk menampilkan perbandingan antara actual price dengan prediction price
- Boxplot: Untuk melihat outlier pada fitur numerik.
- Correlation Matrix: Untuk melihat korelasi antar variabel numerik 

## ** <br>
Dalam Data Preparation ini akan dilakukan proses cleaning dan transforming data.
- Menghapus data yang null dan duplikat
- Melakukan standartscaler dalam data spliting
- Split data ke train dan test dengan rasio 80:20  


## **Modeling**<br>
###  Linear Regression

#### Kelebihan
- **Sederhana & Cepat**: Mudah dipahami dan diimplementasikan.
- **Interpretable**: Koefisien regresi menunjukkan pengaruh fitur terhadap target.
- **Efisien**: Komputasi ringan dan cocok untuk dataset kecil-menengah.

#### Kekurangan
- **Tidak Menangani Relasi Non-Linear**: Kurang efektif untuk data dengan hubungan kompleks.
- **Sensitif terhadap Outlier**: Outlier dapat memengaruhi hasil secara signifikan.
- **Model Terlalu Sederhana**: Kurang akurat untuk masalah yang kompleks.

---

###  Deep Neural Network (DNN)

#### Kelebihan
- **Menangani Pola Non-Linear**: Mampu mempelajari hubungan kompleks dalam data.
- **Fleksibel**: Bisa disesuaikan dengan berbagai bentuk dan ukuran data.
- **Akurat untuk Data Besar**: Performa meningkat seiring jumlah data dan pelatihan.

#### Kekurangan
- **Kurang Transparan**: Sulit untuk ditafsirkan (“black-box”).
- **Butuh Banyak Data & Komputasi**: Kurang optimal untuk dataset kecil.
- **Rawan Overfitting**: Memerlukan validasi dan regularisasi yang baik.


## Evaluation
# 📊 Evaluasi Model Prediksi Harga Rumah

## 📐 Rumus Metrik Evaluasi

- **MAE (Mean Absolute Error)**  
  `MAE = (1/n) * Σ |y_i - ŷ_i|`  
  → Rata-rata kesalahan absolut antara nilai aktual dan prediksi.

---

- **MSE (Mean Squared Error)**  
  `MSE = (1/n) * Σ (y_i - ŷ_i)²`  
  → Rata-rata kuadrat kesalahan prediksi.

---

- **RMSE (Root Mean Squared Error)**  
  `RMSE = √MSE`  
  → Mengembalikan satuan kesalahan ke skala aslinya.

---

- **R² (R-squared)**  
  `R² = 1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)`  
  → Persentase variansi target yang dijelaskan oleh model.

---

- **MAPE (Mean Absolute Percentage Error)**  
  `MAPE = (100% / n) * Σ |(y_i - ŷ_i) / y_i|`  
  → Rata-rata kesalahan dalam bentuk persentase terhadap nilai aktual.

---

## 📊 Hasil Evaluasi

### 🔹 Linear Regression
- **MAE**: 318,367.21  
- **MSE**: 238,329,050,760.70  
- **RMSE**: 488,189.56  
- **R²**: 0.4566  
- **MAPE**: 32.53%

### 🔹 Deep Learning
- **MAE**: 310,192.72  
- **MSE**: 228,036,649,457.14  
- **RMSE**: 477,531.83  
- **R²**: 0.4800  
- **MAPE**: 30.48%

---

## ✅ Kesimpulan

- **Deep Learning** menunjukkan performa **lebih baik** daripada Linear Regression.
- Semua metrik error (MAE, MSE, RMSE, MAPE) lebih rendah.
- Nilai R² lebih tinggi (48% vs 46%).
- Peningkatan tergolong **moderat**, masih bisa ditingkatkan lewat tuning model atau fitur tambahan.


# 📈 Kesimpulan Visualisasi Prediksi Harga Rumah

Grafik menunjukkan perbandingan antara harga rumah **aktual** (garis biru) dengan hasil prediksi dari dua model:
- **Linear Regression** (garis putus-putus hijau)
- **Deep Neural Network (DNN)** (garis putus-putus oranye)

---

## ✅ Kesimpulan

1. **Kedua model berhasil mengikuti pola umum** harga rumah, terutama untuk harga yang berada dalam rentang normal (sekitar 0–2 juta AUD).

2. **Model Deep Neural Network (DNN)** menunjukkan **fleksibilitas yang lebih tinggi** dalam menyesuaikan prediksi terhadap fluktuasi harga, terutama di bagian yang kompleks.

3. **Harga ekstrem (outlier)** di atas 3 juta AUD **tidak berhasil diprediksi dengan akurat oleh kedua model**, menandakan perlunya penanganan atau fitur tambahan untuk kasus seperti ini.

4. **Linear Regression** menghasilkan prediksi yang cenderung **lebih datar** atau "rata-rata", sesuai sifat dasarnya yang hanya memodelkan hubungan linier antar fitur.

5. **DNN memberikan prediksi yang lebih dekat ke nilai aktual**, menunjukkan potensi keunggulannya dalam menangani hubungan non-linier dan data kompleks.

---

![Grafik](images/grafik.png)#
