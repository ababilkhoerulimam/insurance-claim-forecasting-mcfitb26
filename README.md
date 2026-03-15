# AXA Financial Indonesia — Prediksi Klaim Asuransi

Submission untuk kompetisi **MCF ITB 2026**. Notebook ini menyajikan pipeline lengkap untuk memprediksi frekuensi klaim, severitas klaim, dan total klaim bulanan portofolio AXA Financial Indonesia yang terdiri dari 4.096 pemegang polis, dengan horizon forecast 17 bulan (Agustus 2025 – Desember 2026).



## Gambaran Masalah

| Parameter | Detail |
|---|---|
| Data historis | 19 bulan (Januari 2024 – Juli 2025), 4.627 klaim PAID |
| Horizon forecast | 17 bulan (Agustus 2025 – Desember 2026) |
| Metrik evaluasi | MAPE (Mean Absolute Percentage Error) |
| Tantangan utama | Rasio historis vs forecast hampir 1:1, risiko overfitting tinggi |



## Pendekatan

Digunakan ensemble tiga model komplementer yang digabungkan dengan **inverse MAPE weighting** dari walk-forward cross-validation.

| Model | Peran |
|---|---|
| Prophet | Menangkap tren jangka panjang dan seasonality kuartalan |
| LightGBM | Menangkap pola non-linear dan interaksi fitur |
| Seasonal Naive | Baseline statistik berbasis rata-rata historis per bulan |

Untuk mencegah error compounding di horizon panjang, diterapkan **progressive weighting** yang secara bertahap meningkatkan bobot Seasonal Naive. Pada 2026 (zona naive), bobot Seasonal Naive berkisar antara 50–80%.



## Struktur Notebook

| Cell | Deskripsi |
|---|---|
| 0 | Instalasi library dan import |
| 1 | Load data, preprocessing, dan winsorisasi |
| 2 | Exploratory Data Analysis (EDA) |
| 3 | Agregasi bulanan, proyeksi exposure, feature engineering |
| 4 | Walk-forward cross-validation dan perhitungan bobot ensemble |
| 4A | Sensitivity analysis: Prophet hyperparameter & Naive blending |
| 5 | Training model final pada seluruh data |
| 6 | Forecast 17 bulan dengan progressive weighting |
| 6A | Block bootstrap prediction intervals (Total Claim, Frequency, Severity) |
| 6B | Backtesting empirical coverage prediction intervals |
| 7 | Pembuatan file submission Kaggle |
| 8 | Analisis dampak feature engineering |
| 9A | Analisis faktor berpengaruh level individual klaim |
| 9B | Analisis feature importance level portofolio |
| 10 | Ekspor model |
| 11 | Rekomendasi bisnis |



## Data

### Sumber

- **Data_Polis.csv** — 4.096 pemegang polis aktif dengan atribut demografis (gender, tanggal lahir, tanggal efektif polis) dan informasi produk (Plan Code: M-001, M-002, M-003).
- **Data_Klaim.csv** — 5.781 transaksi klaim dengan diagnosis ICD, tipe layanan (Inpatient/Outpatient), metode pembayaran (Cashless/Reimburse), tanggal masuk-keluar RS, dan nominal klaim yang disetujui.

### Preprocessing

1. Parsing dan standarisasi format tanggal dari YYYYMMDD (data polis) dan format campuran (data klaim).
2. Filter hanya klaim berstatus PAID; tersisa 4.627 records.
3. Deduplikasi berdasarkan Claim ID.
4. Winsorisasi 1%–99% untuk membatasi pengaruh outlier ekstrem pada level individual klaim.

**Mengapa winsorisasi, bukan StandardScaler?** Distribusi nominal klaim sangat right-skewed. Winsorisasi dipilih karena mempertahankan skala Rupiah asli yang penting untuk pelaporan aktuaria, sedangkan LightGBM sebagai tree-based model bersifat invariant terhadap monotonic transformation sehingga StandardScaler tidak memberikan manfaat tambahan.

Batas winsorisasi:
- Lower bound: Rp 187.242
- Upper bound: Rp 633.686.130

### Feature Engineering

| Fitur | Deskripsi |
|---|---|
| Umur | Dihitung per 1 Juli 2025 sebagai tanggal referensi |
| Lama_Polis | Tahun sejak tanggal efektif polis |
| Kelompok_Umur | 5 kategori: <18, 18-30, 31-45, 46-60, >60 |
| Length_of_Stay | Selisih tanggal masuk dan keluar RS |
| Claim_Ratio | Nominal klaim disetujui dibagi biaya RS aktual |



## Hasil

### Performa Model

| Metrik | Nilai |
|---|---|
| MAPE internal (walk-forward) | 4,79% |
| MAPE public leaderboard | 5,8% (5 bulan pertama) |
| 80% Prediction Interval (Total Claim) | Rp 132,34 – 149,11 miliar |
| Empirical coverage (backtesting 9 window) | 88,9% untuk Total Claim |

### Prediksi Ringkasan 2026

| Metrik | Prediksi |
|---|---|
| Total Frekuensi | ~2.827 klaim (+0,8% vs 2025 annualized) |
| Rata-rata Severitas | ~Rp 49,4 juta per klaim (-3,0% vs 2025 annualized) |
| Total Klaim (point estimate) | Rp 143,00 miliar |
| Total Klaim (P50 bootstrap) | Rp 143,49 miliar |

### Cadangan Teknis per Kuartal 2026

| Kuartal | Point Estimate | P90 Konservatif |
|---|---|---|
| Q1 2026 (Jan–Mar) | Rp 37,13 M | Rp 38,70 M |
| Q2 2026 (Apr–Jun) | Rp 33,32 M | Rp 34,74 M |
| Q3 2026 (Jul–Sep) | Rp 36,01 M | Rp 37,54 M |
| Q4 2026 (Okt–Des) | Rp 36,55 M | Rp 38,13 M |
| **Total 2026** | **Rp 143,00 M** | **Rp 149,11 M** |



## Temuan Utama (EDA)

**Distribusi klaim** sangat right-skewed; mayoritas klaim di bawah Rp 50 juta dengan puncak distribusi di bawah Rp 20 juta.

**Faktor risiko dominan:**
- Kelompok umur >60 tahun mendominasi dengan lebih dari 2.200 klaim dan median tertinggi.
- N18.0 (gagal ginjal stadium akhir) adalah diagnosis dengan volume klaim terbesar (~300 klaim), diikuti C50 (kanker payudara) dan H26 (katarak).
- Length of Stay adalah prediktor numerik terkuat (Spearman rho = 0,503, p < 0,001).
- Klaim Inpatient memiliki nilai median 9,94x lebih tinggi dibanding Outpatient.
- Metode Cashless secara konsisten menghasilkan median klaim lebih tinggi (~Rp 20 juta) dibanding Reimburse (<Rp 10 juta).

**Tren temporal:** Puncak historis terjadi di Januari 2024 (302 klaim, total Rp 18,95 miliar). Setelahnya, frekuensi stabil di 208–278 klaim/bulan. Pola musiman ini sangat berulang dan terkonfirmasi di forecast 2026.



## Model yang Diekspor

Semua model tersimpan di folder `models/`:

| File | Deskripsi |
|---|---|
| `lgbm_freq.pkl` | LightGBM untuk prediksi frekuensi |
| `lgbm_sev.pkl` | LightGBM untuk prediksi severitas |
| `lgbm_total.pkl` | LightGBM untuk prediksi total klaim |
| `prophet_freq.pkl` | Prophet untuk prediksi frekuensi |
| `prophet_sev.pkl` | Prophet untuk prediksi severitas |
| `prophet_total.pkl` | Prophet untuk prediksi total klaim |
| `seasonal_naive.json` | Lookup table Seasonal Naive per bulan dan target |
| `ensemble_meta.json` | Bobot ensemble dan daftar fitur |



## Instalasi

```bash
pip install prophet lightgbm
```

Library lain yang digunakan: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `statsmodels`.



## Cara Menjalankan

Jalankan notebook secara berurutan dari Cell 0 hingga Cell 11. Pastikan file `Data_Polis.csv` dan `Data_Klaim.csv` berada di direktori yang sama dengan notebook.

```
.
├── notebook.ipynb
├── Data_Polis.csv
├── Data_Klaim.csv
└── models/           # dibuat otomatis saat Cell 10 dijalankan
```



## Catatan Metodologis

- Prediction interval Total Claim tervalidasi secara empiris dengan coverage 88,9% pada 9 window backtesting, melampaui target nominal 80%.
- Prediction interval Frequency dan Severity bersifat indikatif (empirical coverage 66,7%, undercoverage terdokumentasi).
- Prediksi 2026 berada di zona naive dengan bobot Seasonal Naive 50–80%, mencerminkan keterbatasan ekstrapolasi model ML pada horizon panjang dengan data historis 19 bulan.
