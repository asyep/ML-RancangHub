# RAB AI Engine

Proyek AI untuk memprediksi Rencana Anggaran Biaya (RAB) konstruksi menggunakan Machine Learning.

## 📋 Deskripsi Proyek

RAB AI Engine adalah sistem kecerdasan buatan yang dirancang untuk memprediksi kebutuhan material dan volume pekerjaan konstruksi berdasarkan data historis proyek. Sistem ini memisahkan pengembangan logic AI dengan pengembangan website, sehingga memudahkan fokus pada pelatihan model terlebih dahulu sebelum integrasi ke sistem yang lebih besar.

### Fitur Utama
- Prediksi volume pondasi, dinding, dan material lainnya
- Berbasis data historis proyek (Masjid, Sekolah, Rumah, dll)
- Menggunakan algoritma Ensemble Learning (Random Forest)
- Input: Tipe bangunan, Luas area, Lokasi
- Output: Estimasi volume pekerjaan dan kebutuhan material

## 📁 Struktur Folder

```
RAB_AI_Engine/
│
├── data/                           # Dataset untuk training
│   └── data_proyek_historis.csv    # Data RAB proyek lama (Excel/CSV)
│
├── models/                         # Model AI yang sudah dilatih
│   └── (file .pkl akan tersimpan di sini)
│
├── notebooks/                      # Jupyter Notebooks untuk eksplorasi
│   └── 1_eksplorasi_data.ipynb     # Analisis dan visualisasi data
│
├── src/                            # Source code utama
│   ├── train_model.py              # Script untuk melatih model AI
│   └── predict.py                  # Script untuk prediksi/testing
│
├── requirements.txt                # Daftar library Python yang dibutuhkan
└── README.md                       # Dokumentasi proyek (file ini)
```

## 🔧 Penjelasan File

### 1. `data/data_proyek_historis.csv`
**Fungsi:** Dataset utama untuk melatih AI

**Format:** CSV (Comma Separated Values)
- Lebih ringan dan cepat dibaca dibanding Excel
- Berisi data historis proyek konstruksi

**Kolom yang diperlukan:**
- **Input:** Tipe Bangunan, Luas (m²), Lokasi
- **Output:** Volume Pondasi, Volume Dinding, dll.

### 2. `notebooks/1_eksplorasi_data.ipynb`
**Fungsi:** Workspace untuk eksperimen dan analisis data

**Kegunaan:**
- Eksplorasi data secara interaktif
- Visualisasi grafik dan statistik
- Deteksi data kosong atau anomali
- Testing algoritma sebelum implementasi final

**Cara menggunakan:** Install extension "Jupyter" di VS Code

### 3. `src/train_model.py`
**Fungsi:** Script utama untuk melatih model AI

**Proses:**
1. Membaca data dari folder `data/`
2. Menjalankan algoritma Ensemble Learning (Random Forest)
3. Menyimpan model terlatih ke folder `models/`

### 4. `src/predict.py`
**Fungsi:** Script untuk simulasi prediksi

**Kegunaan:**
- Testing model yang sudah dilatih
- Contoh: Input "Masjid 200m² di Bandung" → Output estimasi volume
- Memanggil file model dari folder `models/`

### 5. `requirements.txt`
**Fungsi:** Daftar library Python yang dibutuhkan

**Isi:**
```
pandas          # Manipulasi data
numpy           # Komputasi numerik
scikit-learn    # Machine Learning
joblib          # Menyimpan/load model
matplotlib      # Visualisasi data
```

## 🚀 Cara Memulai

### 1. Setup Environment

```bash
# Buka folder proyek di VS Code
cd RAB_AI_Engine

# Install semua library yang dibutuhkan
pip install -r requirements.txt
```

### 2. Persiapan Data

Karena mungkin belum ada data asli yang rapi, gunakan dummy dataset terlebih dahulu untuk testing.

**Opsi:**
- Generate dummy data menggunakan script Python
- Import data historis yang sudah ada (format Excel/CSV)
- Convert file Excel ke CSV jika diperlukan

### 3. Eksplorasi Data

```bash
# Buka Jupyter Notebook di VS Code
# File: notebooks/1_eksplorasi_data.ipynb
```

Lakukan:
- Analisis distribusi data
- Cek missing values
- Visualisasi korelasi antar variabel

### 4. Training Model

```bash
# Jalankan script training
python src/train_model.py
```

Hasil: File model (`.pkl`) akan tersimpan di folder `models/`

### 5. Testing Prediksi

```bash
# Jalankan script prediksi
python src/predict.py
```

Contoh output:
```
Input: Masjid, 200m², Bandung
Prediksi Volume Pondasi: 45.3 m³
Prediksi Volume Dinding: 128.7 m³
```

## 💡 Best Practice

### Pemisahan Pengembangan
- **Logic AI (Model Training):** Fokus pada akurasi model
- **Pengembangan Website:** Backend/Frontend terpisah
- **Keuntungan:** Memastikan "otak AI" pintar dulu sebelum integrasi

### Workflow Development
1. ✅ Siapkan dataset yang bersih
2. ✅ Eksplorasi di Jupyter Notebook
3. ✅ Training model dengan algoritma terbaik
4. ✅ Testing dan validasi akurasi
5. ✅ Export model untuk integrasi ke website

## 📊 Dataset Requirements

Format data yang ideal:

| Tipe Bangunan | Luas (m²) | Lokasi  | Volume Pondasi | Volume Dinding | ... |
|---------------|-----------|---------|----------------|----------------|-----|
| Masjid        | 200       | Bandung | 45.3           | 128.7          | ... |
| Sekolah       | 500       | Jakarta | 112.5          | 320.4          | ... |
| Rumah         | 120       | Bogor   | 28.9           | 85.2           | ... |

## 🛠️ Tech Stack

- **Python 3.8+**
- **pandas:** Data manipulation
- **scikit-learn:** Machine Learning
- **numpy:** Numerical computing
- **matplotlib:** Data visualization
- **joblib:** Model persistence

## 📝 Catatan

- File `.pkl` di folder `models/` adalah model yang sudah dilatih (jangan dihapus)
- Pastikan data CSV memiliki format yang konsisten
- Untuk produksi, pertimbangkan validasi data input yang lebih ketat

## 🔜 Next Steps

1. Generate atau import dataset historis
2. Eksplorasi data untuk memahami pola
3. Training model dengan berbagai algoritma
4. Evaluasi performa model (accuracy, RMSE, dll)
5. Siapkan API endpoint untuk integrasi website

---

**Status:** Development Phase - Model Training
**Last Updated:** 2025-11-27