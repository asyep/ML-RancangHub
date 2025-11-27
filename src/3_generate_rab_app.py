import pandas as pd
import joblib
import os
from datetime import datetime

# --- SETTING ---
MODEL_PATH = 'src/rab_ai_model.pkl'
AHSP_PATH = 'dataset/AHSP.csv' 
OUTPUT_FOLDER = 'output'

def load_ahsp_prices():
    """
    Load data harga AHSP.
    Disini kita pakai dummy data dulu sesuai skenario awal.
    """
    database_harga = {
        'Vol_Pondasi_m3': {'Uraian': 'Pekerjaan Pondasi Batu Kali', 'Harga': 850000},
        'Vol_Dinding_m2': {'Uraian': 'Pasangan Dinding Bata Merah', 'Harga': 125000},
        'Vol_Lantai_m2': {'Uraian': 'Pemasangan Keramik 40x40', 'Harga': 175000},
        'Vol_Atap_m2': {'Uraian': 'Rangka Atap Baja Ringan + Genteng', 'Harga': 350000}
    }
    return database_harga

def generate_excel(project_name, location, luas, rab_data):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # Buat DataFrame
    df_rab = pd.DataFrame(rab_data)
    
    # Hitung Grand Total
    grand_total = df_rab['Total Harga'].sum()
    
    # --- PERBAIKAN DISINI ---
    # Menambahkan baris Total. Jumlah elemen list harus sama dengan jumlah kolom (6 kolom)
    # Kolom: [No, Uraian, Vol, Satuan, Harga Satuan, Total Harga]
    df_rab.loc['Total'] = ['', '', '', '', 'GRAND TOTAL', grand_total]
    
    # Nama File
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{OUTPUT_FOLDER}/RAB_{project_name}_{timestamp}.xlsx"
    
    # Export ke Excel
    try:
        df_rab.to_excel(filename, index=False)
        print(f"\n[SUKSES] File Excel berhasil dibuat: {filename}")
        print(f"Total Anggaran: Rp {grand_total:,.2f}")
        print("Silakan cek folder 'output' untuk melihat hasilnya.")
    except Exception as e:
        print(f"\n[ERROR] Gagal menyimpan Excel: {e}")
        print("Pastikan file Excel tidak sedang dibuka.")

def main():
    # 1. Load Model AI
    if not os.path.exists(MODEL_PATH):
        print("Model belum ada! Jalankan '2_train_ai.py' dulu.")
        return
    
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    prices = load_ahsp_prices()

    print("=== AI GENERATOR RAB KONSTRUKSI ===")
    
    # 2. Input Admin (Skenario User)
    print("Masukkan Data Proyek Baru:")
    input_tipe = input("Tipe Proyek (Masjid/Rumah Tinggal/Kantor Desa): ")
    input_luas = input("Luas Tanah (m2): ")
    input_lokasi = input("Lokasi Proyek: ")
    input_nama = input("Nama Proyek (Label): ")

    # Validasi input angka
    try:
        input_luas = float(input_luas)
    except ValueError:
        print("Error: Luas tanah harus berupa angka!")
        return

    # 3. AI Melakukan Prediksi Volume
    input_data = pd.DataFrame({
        'Tipe_Proyek': [input_tipe], 
        'Luas_Tanah_m2': [input_luas]
    })
    
    print("\nAI sedang menghitung volume pekerjaan...")
    
    try:
        predicted_volumes = model.predict(input_data)[0] # Hasil prediksi array
    except Exception as e:
        print(f"Error pada saat prediksi AI: {e}")
        print("Pastikan Tipe Proyek sesuai dengan data training (Masjid/Rumah Tinggal/Kantor Desa)")
        return
    
    # Nama kolom target sesuai urutan training
    col_names = ['Vol_Pondasi_m3', 'Vol_Dinding_m2', 'Vol_Lantai_m2', 'Vol_Atap_m2']
    
    # 4. Susun RAB (Gabungkan Volume AI + Harga AHSP)
    rab_list = []
    
    for i, col in enumerate(col_names):
        vol = predicted_volumes[i]
        item_info = prices.get(col)
        
        if item_info:
            total_harga = vol * item_info['Harga']
            
            rab_list.append({
                'No': i + 1,
                'Uraian Pekerjaan': item_info['Uraian'],
                'Volume (AI)': round(vol, 2),
                'Satuan': col.split('_')[-1], # Ambil m2 atau m3 dari nama kolom
                'Harga Satuan (AHSP)': item_info['Harga'],
                'Total Harga': total_harga
            })

    # 5. Output Excel
    generate_excel(input_nama, input_lokasi, input_luas, rab_list)

if __name__ == "__main__":
    main()