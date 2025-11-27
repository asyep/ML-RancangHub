# ML-RancangHub
🤖 RAB AI Engine Project Documentation

💡 Gambaran Umum Proyek

Proyek ini bertujuan untuk mengembangkan sebuah Machine Learning Engine yang mampu memprediksi Volume Satuan Pekerjaan (misalnya Volume Pondasi, Volume Dinding) berdasarkan karakteristik input proyek (misalnya Tipe Proyek, Luas Bangunan, Lokasi).

Proyek AI ini dikembangkan secara terpisah dari Web Development (Backend/Frontend) sebagai Best Practice untuk memastikan fokus utama pada akurasi dan performa model (otak AI).

🚀 Struktur Direktori

Berikut adalah struktur folder dan file yang wajib disiapkan:

RAB_AI_Engine/
│
├── data/                       <-- Dataset historis untuk training
│   └── data_proyek_historis.csv
│
├── models/                     <-- Model AI yang sudah dilatih (.pkl)
│
├── notebooks/                  <-- Area eksperimen (Jupyter Notebook)
│   └── 1_eksplorasi_data.ipynb
│
├── src/                        <-- Source Code inti (Python Scripts)
│   ├── train_model.py          <-- Script melatih & menyimpan model
│   └── predict.py              <-- Script simulasi prediksi
│
├── requirements.txt            <-- Daftar dependensi library Python
└── README.md                   <-- Dokumentasi proyek ini (File ini)


🛠️ Persiapan Awal (Setup)

Ikuti langkah-langkah berikut untuk menyiapkan lingkungan pengembangan (environment):

Buka Proyek: Buka folder RAB_AI_Engine di VS Code.

Instal Python: Pastikan Python sudah terinstal di sistem Anda.

Instal Dependensi: Buka Terminal di VS Code (Ctrl + \``) dan jalankan perintah berikut untuk menginstal semua *library* yang terdaftar di requirements.txt`:

pip install -r requirements.txt


Dependensi yang Digunakan (requirements.txt)

Library

Fungsi Utama

pandas

Manipulasi dan data handling data CSV.

numpy

Operasi numerik untuk Machine Learning.

scikit-learn

Implementasi algoritma ML seperti Random Forest dan preprocessing data.

joblib

Untuk serialisasi (menyimpan dan memuat) model AI.

matplotlib

Visualisasi data (opsional, untuk eksplorasi di Notebook).

📂 Alur Kerja Pengembangan

1. Data Source (data/)

Data historis dalam format CSV adalah input utama untuk melatih AI.

Kolom Input (Features): Tipe Proyek, Luas, Lokasi.

Kolom Target (Labels): Volume Pondasi, Volume Dinding, dll.

2. Eksplorasi & Preprocessing (notebooks/)

Gunakan 1_eksplorasi_data.ipynb untuk:

Membaca data dan membersihkan (handling missing values).

Melihat distribusi data (visualisasi).

Mencoba teknik Encoding untuk fitur kategorikal.

3. Pelatihan Model (src/train_model.py)

Script ini akan menjalankan algoritma Ensemble Learning (misal Random Forest) pada data yang sudah di-preprocessing.

Setelah pelatihan selesai, model yang sudah "pintar" akan disimpan ke models/model_final.pkl menggunakan library joblib.

4. Pengujian Simulasi (src/predict.py)

Script ini berfungsi sebagai simulasi integrasi awal.

Ia akan memuat model (model_final.pkl) dan menerima dummy input (misal: Tipe: Masjid, Luas: 200m2, Lokasi: Bandung), lalu mengembalikan output prediksi volume.

➡️ Langkah Selanjutnya

Langkah pertama yang harus dilakukan adalah membuat Dummy Dataset untuk mengisi data/data_proyek_historis.csv agar Anda memiliki "bahan bakar" untuk memulai coding di Notebook.