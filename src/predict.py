import pandas as pd
import joblib
import os

# --- 1. KONFIGURASI PATH ---
MODEL_PATH = 'models/rab_ensemble_model.pkl'
COLUMNS_PATH = 'models/model_columns.pkl'

def get_ai_prediction(tipe, lokasi, luas_bangunan, luas_tanah, lantai):
    # Cek apakah model ada
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        return "Error: Model belum dilatih. Jalankan train_model.py dulu."

    # --- 2. LOAD MODEL & KOLOM ---
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)

    # --- 3. SIAPKAN DATA INPUT ---
    # Buat dictionary sesuai input user
    input_data = {
        'Tipe_Proyek': [tipe],
        'Lokasi': [lokasi],
        'Luas_Bangunan': [luas_bangunan],
        'Luas_Tanah': [luas_tanah],
        'Jumlah_Lantai': [lantai]
    }
    
    # Ubah ke DataFrame
    query_df = pd.DataFrame(input_data)

    # --- 4. PREPROCESSING (SAMA SEPERTI SAAT TRAINING) ---
    # Lakukan One-Hot Encoding pada input user
    query_df = pd.get_dummies(query_df)

    # KUNCI PENTING: Menyamakan kolom input dengan kolom saat training
    # Saat training, ada kolom 'Lokasi_Jakarta', 'Lokasi_Bandung', dll.
    # Jika user input 'Surabaya', kolom 'Lokasi_Jakarta' tidak akan terbentuk otomatis.
    # Maka kita harus "memaksa" struktur kolomnya sama persis dengan model_columns.
    query_df = query_df.reindex(columns=model_columns, fill_value=0)

    # --- 5. PREDIKSI ---
    prediction = model.predict(query_df)

    # Hasil prediksi berupa array, kita ambil index 0
    hasil = prediction[0]
    
    # Return dalam bentuk dictionary yang rapi
    return {
        'Vol_Pondasi': hasil[0],
        'Vol_Dinding': hasil[1],
        'Vol_Keramik': hasil[2],
        'Vol_Atap': hasil[3],
        'Vol_Beton': hasil[4]
    }

if __name__ == "__main__":
    print("\n--- 🤖 SIMULASI AI RAB CONTRUCTION ---")
    print("Silakan input data proyek baru:")
    
    # Input Interaktif di Terminal
    # (Pastikan mengetik Tipe Proyek & Lokasi persis sesuai pilihan di generate_dummy.py
    #  Pilihan Tipe: Masjid, Sekolah, Rumah Tinggal, Gudang
    #  Pilihan Lokasi: Jakarta, Bandung, Surabaya, Medan, Makassar)
    
    in_tipe = input("Tipe Proyek (Masjid/Sekolah/Rumah Tinggal/Gudang): ")
    in_lokasi = input("Lokasi (Jakarta/Bandung/Surabaya/Medan/Makassar): ")
    in_luas_bangunan = float(input("Luas Bangunan (m2): "))
    in_luas_tanah = float(input("Luas Tanah (m2): "))
    in_lantai = int(input("Jumlah Lantai: "))

    print("\n⏳ Sedang mengkalkulasi prediksi kebutuhan material...")
    
    # Panggil Fungsi AI
    hasil_prediksi = get_ai_prediction(in_tipe, in_lokasi, in_luas_bangunan, in_luas_tanah, in_lantai)

    print("\n" + "="*40)
    print(f"HASIL PREDIKSI AI UNTUK: {in_tipe} ({in_luas_bangunan} m2)")
    print("="*40)
    print(f"🏗️  Volume Pondasi      : {hasil_prediksi['Vol_Pondasi']:.2f} m3")
    print(f"🧱  Volume Dinding      : {hasil_prediksi['Vol_Dinding']:.2f} m2")
    print(f"⬜  Volume Keramik      : {hasil_prediksi['Vol_Keramik']:.2f} m2")
    print(f"🏠  Volume Atap         : {hasil_prediksi['Vol_Atap']:.2f} m2")
    print(f"🏛️  Volume Beton        : {hasil_prediksi['Vol_Beton']:.2f} m3")
    print("="*40)
    print("Note: Output ini nanti dikalikan dengan Harga Satuan di Database Website.")