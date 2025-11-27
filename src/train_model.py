import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --- 1. KONFIGURASI ---
DATA_PATH = 'data/data_proyek_historis.csv'
MODEL_PATH = 'models/rab_ensemble_model.pkl'
COLUMNS_PATH = 'models/model_columns.pkl' # Penting untuk menyimpan urutan kolom

def train():
    print("🚀 Memulai proses Training Model AI...")

    # --- 2. LOAD DATASET ---
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Data berhasil dimuat: {df.shape[0]} baris data.")
    except FileNotFoundError:
        print("❌ Error: File data tidak ditemukan. Jalankan generate_dummy.py dulu!")
        return

    # --- 3. DATA PREPROCESSING ---
    # Pisahkan antara Fitur (Input) dan Target (Output yang mau diprediksi)
    
    # X = Input (Apa yang user masukkan nanti)
    X = df[['Tipe_Proyek', 'Lokasi', 'Luas_Bangunan', 'Luas_Tanah', 'Jumlah_Lantai']]
    
    # y = Output (Apa yang AI harus tebak)
    y = df[['Target_Vol_Pondasi', 'Target_Vol_Dinding', 'Target_Vol_Keramik', 'Target_Vol_Atap', 'Target_Vol_Beton']]

    # One-Hot Encoding
    # Mengubah data teks (Masjid, Jakarta) menjadi angka (0 dan 1) agar bisa dihitung mesin.
    X = pd.get_dummies(X, columns=['Tipe_Proyek', 'Lokasi'])
    
    # Simpan nama-nama kolom hasil encoding ini. 
    # Penting agar nanti saat prediksi satu data, urutan kolomnya sama persis.
    model_columns = list(X.columns)
    joblib.dump(model_columns, COLUMNS_PATH)

    # Bagi data: 80% untuk Latihan (Train), 20% untuk Ujian (Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. TRAINING (ENSEMBLE LEARNING) ---
    print("🤖 Sedang melatih Random Forest (ini mungkin memakan waktu beberapa detik)...")
    
    # Kita pakai Random Forest Regressor
    # n_estimators=100 artinya kita menggunakan 100 pohon keputusan (Ensemble)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Proses belajar dimulai di sini
    model.fit(X_train, y_train)

    # --- 5. EVALUASI ---
    # Cek seberapa pintar modelnya dengan data ujian (Test)
    predictions = model.predict(X_test)
    
    print("\n📊 Evaluasi Akurasi (Mean Absolute Error):")
    print("(Semakin kecil angkanya, semakin akurat)")
    
    # Menghitung rata-rata kesalahan untuk setiap target
    mae = mean_absolute_error(y_test, predictions, multioutput='raw_values')
    
    target_names = ['Pondasi', 'Dinding', 'Keramik', 'Atap', 'Beton']
    for name, error in zip(target_names, mae):
        print(f"   - Error Rata-rata Vol {name}: {error:.2f}")

    # --- 6. SIMPAN MODEL ---
    # Simpan otak AI ke folder models
    joblib.dump(model, MODEL_PATH)
    print(f"\n💾 Model berhasil disimpan di: {MODEL_PATH}")
    print("✅ Siap digunakan untuk prediksi!")

if __name__ == "__main__":
    train()