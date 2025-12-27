# ML-RancangHub

Project ini adalah sistem estimasi RAB cerdas menggunakan Machine Learning.

## Persiapan

1.  Pastikan Python sudah terinstall.
2.  Install library yang dibutuhkan:

    ```bash
    pip install -r requirements.txt
    ```

## Cara Menjalankan

### 1. Training Model (Wajib dijalankan pertama kali)

Sebelum menggunakan aplikasi, Anda perlu melatih model AI terlebih dahulu agar file model (`models/rab_model.json`) terbentuk.

Jalankan perintah:

```bash
python developer_train.py
```

Tunggu hingga proses training selesai dan muncul pesan "Model Tersimpan".

### 2. Menjalankan Aplikasi User

Setelah model siap, Anda bisa menjalankan aplikasi utama:

```bash
python user_app.py
```

Ikuti petunjuk di layar untuk melakukan estimasi RAB.
