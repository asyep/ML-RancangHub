import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BUILDING_TYPES, QUALITY_LEVELS

class SmartAppLogic:
    def __init__(self):
        self.model_path = 'models/rab_model.json'
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model belum dilatih.")
            
        self.model = joblib.load(self.model_path)
        self.mlb = joblib.load('models/label_encoder.pkl')
        self.df_master = pd.read_csv('data/raw/master_ahsp.csv')
        self.df_detail = pd.read_csv('data/processed/rincian_komponen.csv')
        self.master_dict = self.df_master.set_index('kode').to_dict('index')

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def predict_and_interact(self, tipe, luas, lantai, kualitas):
        self.clear_screen()
        print(f"🤖 AI Menganalisis: {tipe} | Luas: {luas}m2 | Lantai: {lantai} | Kualitas: {kualitas}")
        
        # Prediksi
        inp = pd.DataFrame({'tipe': [tipe], 'luas': [luas], 'lantai': [lantai], 'kualitas': [kualitas]})
        pred = self.model.predict(inp)
        codes = self.mlb.inverse_transform(pred)[0]
        
        if len(codes) == 0:
            print("⚠️ Tidak ada rekomendasi item.")
            input("Enter untuk kembali..."); return

        # LOOP UTAMA: Tampilkan List -> Minta Input Kode Detail
        while True:
            self.clear_screen()
            print(f"✅ HASIL REKOMENDASI AI ({len(codes)} Item)")
            print("="*100)
            print(f"{'KODE':<10} {'URAIAN PEKERJAAN':<50} {'SAT':<6} {'HARGA SAT (F)'}")
            print("="*100)
            
            sorted_codes = sorted(list(codes))
            for kode in sorted_codes:
                m = self.master_dict.get(kode)
                if m: print(f"{kode:<10} {m['uraian'][:48]:<50} {m['satuan']:<6} {m['harga_satuan']:,.0f}")
            print("="*100)
            
            print("\n[OPSI]")
            print("1. Ketik KODE ANALISA (contoh: IV.001) untuk lihat detail A-F")
            print("2. Ketik 'exit' untuk kembali ke Dashboard")
            
            pilih = input(">> Pilihan Anda: ").strip()
            
            if pilih.lower() == 'exit':
                break

            
            if pilih in codes:
                self.show_detail(pilih)
                input("\nTekan Enter untuk kembali ke list...")
            else:
                print("❌ Kode tidak valid atau tidak ada di list rekomendasi.")
                input("Enter...")

    def show_detail(self, kode):
        m = self.master_dict.get(kode)
        print(f"\n=== RINCIAN ANALISA: {kode} ===")
        print(f"Uraian: {m['uraian']}")
        print("-" * 80)
        print(f"{'URAIAN':<40} {'KOEF':<10} {'HARGA':<15} {'JUMLAH'}")
        print("-" * 80)
        
        comps = self.df_detail[self.df_detail['kode'] == kode]
        total_d = 0
        
        for grp in ['A. Tenaga', 'B. Bahan', 'C. Alat']:
            sub = comps[comps['kategori'] == grp]
            if not sub.empty:
                print(f"[{grp}]")
                for _, row in sub.iterrows():
                    print(f"   - {row['item']:<35} {row['koef']:<10} {row['harga']:<15,.0f} {row['subtotal']:,.0f}")
                    total_d += row['subtotal']
        
        profit = int(total_d * 0.1)
        total = total_d + profit
        print("-" * 80)
        print(f"D. JUMLAH (A+B+C)           : Rp {total_d:,.0f}")
        print(f"E. OVERHEAD & PROFIT (10%)  : Rp {profit:,.0f}")
        print(f"F. HARGA SATUAN (D+E)       : Rp {total:,.0f}")
        print("=" * 80)