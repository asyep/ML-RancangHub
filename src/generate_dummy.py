import pandas as pd
import numpy as np
import random
import os

# Konfigurasi: Jumlah data yang ingin digenerate
JUMLAH_DATA = 500

def generate_dummy_data(n_samples):
    data = []
    
    locations = ['Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Makassar']
    # Faktor kemahalan/kesulitan berdasarkan lokasi (hanya simulasi)
    loc_factor = {'Jakarta': 1.1, 'Bandung': 1.0, 'Surabaya': 1.05, 'Medan': 1.05, 'Makassar': 1.1}
    
    project_types = ['Masjid', 'Sekolah', 'Rumah Tinggal', 'Gudang']
    
    print(f"Sedang meng-generate {n_samples} data proyek fiktif...")

    for _ in range(n_samples):
        # --- 1. GENERATE INPUT (FITUR) ---
        p_type = random.choice(project_types)
        location = random.choice(locations)
        
        # Luas bangunan acak tapi logis sesuai tipe
        if p_type == 'Masjid':
            area = random.randint(100, 1000)
            floors = random.choices([1, 2], weights=[0.7, 0.3])[0]
        elif p_type == 'Sekolah':
            area = random.randint(200, 1500)
            floors = random.randint(1, 3)
        elif p_type == 'Rumah Tinggal':
            area = random.randint(36, 300)
            floors = random.randint(1, 2)
        else: # Gudang
            area = random.randint(100, 2000)
            floors = 1
            
        land_area = int(area * random.uniform(1.2, 2.0)) # Tanah pasti lebih besar dari bangunan

        # --- 2. LOGIKA GENERATE OUTPUT (TARGET VOLUME) ---
        # Kita pakai rumus pendekatan kasar + Random Noise agar terlihat nyata
        
        # Variabel pengganggu (Noise) +/- 10% agar data tidak terlalu sempurna
        noise = lambda: random.uniform(0.9, 1.1) 
        
        # a. Volume Pondasi (m3)
        # Logika: Berbanding lurus dengan luas lantai dasar
        vol_pondasi = (area / floors) * 0.25 * noise() 
        if p_type == 'Gudang': vol_pondasi *= 1.5 # Gudang pondasi lebih kuat
        
        # b. Volume Dinding Bata (m2)
        # Logika: Keliling * Tinggi. Masjid/Gudang lebih sedikit sekat dibanding Rumah/Sekolah.
        wall_ratio = 1.0
        if p_type == 'Masjid': wall_ratio = 1.2 # Sedikit sekat
        elif p_type == 'Gudang': wall_ratio = 0.8 # Sangat sedikit sekat
        elif p_type == 'Rumah Tinggal': wall_ratio = 2.5 # Banyak kamar
        elif p_type == 'Sekolah': wall_ratio = 2.0 # Banyak kelas
        
        tinggi_dinding = 3.5 * floors
        vol_dinding = (np.sqrt(area/floors) * 4) * tinggi_dinding * wall_ratio * noise()
        
        # c. Volume Keramik (m2)
        # Logika: Hampir sama dengan luas bangunan, dikurangi sedikit teras/toilet beda keramik
        vol_keramik = area * 0.95 * noise()
        
        # d. Volume Atap (m2)
        # Logika: Luas lantai dasar * faktor kemiringan atap
        luas_dasar = area / floors
        if floors > 1:
            vol_atap = luas_dasar * 1.4 * noise() # Atap limasan/pelana
        else:
            vol_atap = luas_dasar * 1.3 * noise()

        # e. Volume Beton Bertulang (m3)
        # Logika: Sekitar 0.15 - 0.25 m3 per m2 bangunan (rule of thumb)
        vol_beton = area * random.uniform(0.15, 0.25) * noise()

        # Append ke list
        data.append({
            'Tipe_Proyek': p_type,
            'Lokasi': location,
            'Luas_Bangunan': area,
            'Luas_Tanah': land_area,
            'Jumlah_Lantai': floors,
            # Target Variables (Yang mau diprediksi AI)
            'Target_Vol_Pondasi': round(vol_pondasi, 2),
            'Target_Vol_Dinding': round(vol_dinding, 2),
            'Target_Vol_Keramik': round(vol_keramik, 2),
            'Target_Vol_Atap': round(vol_atap, 2),
            'Target_Vol_Beton': round(vol_beton, 2)
        })

    return pd.DataFrame(data)

# --- EKSEKUSI ---
if __name__ == "__main__":
    # Pastikan folder 'data' ada
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Folder 'data' berhasil dibuat.")
    
    # Generate data
    df = generate_dummy_data(JUMLAH_DATA)
    
    # Simpan ke CSV
    file_path = 'data/data_proyek_historis.csv'
    df.to_csv(file_path, index=False)
    
    print("="*50)
    print(f"SUKSES! Data dummy berhasil disimpan di: {file_path}")
    print("Contoh 5 data pertama:")
    print(df.head())
    print("="*50)