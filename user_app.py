import os
import sys
from config.settings import BUILDING_TYPES, QUALITY_LEVELS
from src.utils import SmartAppLogic

def main():
    logic = SmartAppLogic()
    
    while True:
        logic.clear_screen()
        print("===================================")
        print("   SMART RAB AI - DASHBOARD")
        print("===================================")
        print("1. Mulai Estimasi RAB")
        print("2. Keluar")
        
        pilih = input("\nPilih Menu: ")
        
        if pilih == '1':
            # STEP 1: PILIH TIPE
            logic.clear_screen()
            print("--- PILIH TIPE BANGUNAN ---")
            for i, t in enumerate(BUILDING_TYPES):
                print(f"{i+1}. {t}")
            try:
                idx = int(input("\nNomor Tipe: ")) - 1
                tipe = BUILDING_TYPES[idx]
            except:
                continue

            # STEP 2: ISI DETAIL
            logic.clear_screen()
            print(f"--- DETAIL BANGUNAN: {tipe} ---")
            try:
                luas = float(input("Luas Bangunan (m2): "))
                lantai = int(input("Jumlah Lantai: "))
            except:
                print("Input salah.")
                input("Enter...")
                continue
                
            print("\nPilih Kualitas:")
            for i, q in enumerate(QUALITY_LEVELS):
                print(f"{i+1}. {q}")
            try:
                idx_q = int(input("Nomor: ")) - 1
                kual = QUALITY_LEVELS[idx_q]
            except:
                kual = "Standar"

            # STEP 3: PREDIKSI & INTERAKSI
            logic.predict_and_interact(tipe, luas, lantai, kual)
            
        elif pilih == '2':
            print("Keluar...")
            break

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Pastikan sudah Generate Data dan Training Model.")