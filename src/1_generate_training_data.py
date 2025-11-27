import pandas as pd
import numpy as np
import random

# Kita buat simulasi data 500 proyek masa lalu
# Agar AI belajar hubungan antara: Tipe Bangunan + Luas -> Volume Material
data = []
tipe_bangunan = ['Masjid', 'Rumah Tinggal', 'Kantor Desa']

print("Sedang membuat data training simulasi...")

for i in range(500):
    tipe = random.choice(tipe_bangunan)
    luas_tanah = random.randint(50, 500) # Luas acak antara 50m2 - 500m2
    
    # RUMUS LOGIKA DASAR (Ini pola yang akan dipelajari AI)
    # Misal: Volume Pondasi masjid lebih tebal dari rumah
    if tipe == 'Masjid':
        vol_pondasi = luas_tanah * 0.25  # Masjid pondasinya banyak
        vol_dinding = luas_tanah * 1.5   # Dinding tinggi
        vol_lantai  = luas_tanah * 1.0   # Keramik full
        vol_atap    = luas_tanah * 1.3   # Atap limas/kubah
    elif tipe == 'Rumah Tinggal':
        vol_pondasi = luas_tanah * 0.20
        vol_dinding = luas_tanah * 2.0   # Banyak sekat kamar
        vol_lantai  = luas_tanah * 0.9
        vol_atap    = luas_tanah * 1.2
    else: # Kantor Desa
        vol_pondasi = luas_tanah * 0.22
        vol_dinding = luas_tanah * 1.2
        vol_lantai  = luas_tanah * 1.0
        vol_atap    = luas_tanah * 1.1

    # Tambahkan sedikit "noise" (variasi acak) supaya AI belajar data realistis
    # Karena di dunia nyata, itungan tidak selalu pas matematika
    noise = np.random.uniform(0.9, 1.1) 
    
    data.append({
        'Tipe_Proyek': tipe,
        'Luas_Tanah_m2': luas_tanah,
        'Vol_Pondasi_m3': round(vol_pondasi * noise, 2),
        'Vol_Dinding_m2': round(vol_dinding * noise, 2),
        'Vol_Lantai_m2': round(vol_lantai * noise, 2),
        'Vol_Atap_m2': round(vol_atap * noise, 2)
    })

df = pd.DataFrame(data)
df.to_csv('dataset/historical_projects.csv', index=False)
print("Selesai! File 'dataset/historical_projects.csv' berhasil dibuat.")
print(df.head())