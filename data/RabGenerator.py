import pandas as pd
import numpy as np
from typing import Dict, List, Any
import random

class RABDataGenerator:
    def __init__(self, ahsp_data: pd.DataFrame, seed: int = 42):
        """
        Generator data sintetis untuk training model RAB
        
        Parameters:
        - ahsp_data: DataFrame yang berisi master data AHSP
        - seed: random seed untuk reproducibility
        """
        self.ahsp_data = ahsp_data
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        
        # Konfigurasi proyek
        self.tipe_proyek_list = ['rumah_tinggal', 'ruko', 'kantor', 'gedung_sekolah']
        self.spek_list = ['standar', 'menengah', 'premium']
        self.lokasi_list = ['jawa_barat', 'jawa_tengah', 'jawa_timur', 'luar_jawa']
        
        # Rules untuk volume berdasarkan karakteristik proyek
        self.volume_rules = self._setup_volume_rules()
    
    def _setup_volume_rules(self) -> Dict[str, Dict[str, Any]]:
        """Setup aturan volume berdasarkan tipe dan spek proyek"""
        return {
            'persiapan': {
                'base_multiplier': 1.0,
                'dependencies': ['luas', 'spek'],
                'formula': self._formula_persiapan
            },
            'pondasi': {
                'base_multiplier': 1.0,
                'dependencies': ['luas', 'lantai', 'tipe'],
                'formula': self._formula_pondasi
            },
            'struktur': {
                'base_multiplier': 1.0,
                'dependencies': ['luas', 'lantai'],
                'formula': self._formula_struktur
            },
            'dinding': {
                'base_multiplier': 1.0,
                'dependencies': ['luas', 'lantai'],
                'formula': self._formula_dinding
            },
            'atap': {
                'base_multiplier': 1.0,
                'dependencies': ['luas', 'spek'],
                'formula': self._formula_atap
            },
            'finishing': {
                'base_multiplier': 1.0,
                'dependencies': ['luas', 'lantai', 'spek'],
                'formula': self._formula_finishing
            },
            'mep': {
                'base_multiplier': 1.0,
                'dependencies': ['luas', 'spek'],
                'formula': self._formula_mep
            }
        }
    
    # =========================================================================
    # FORMULA FUNCTIONS - FIXED VERSION
    # =========================================================================
    
    def _formula_persiapan(self, luas: float, spek: str) -> float:
        """Formula untuk pekerjaan persiapan"""
        multiplier = 1.0 if spek == 'standar' else 1.2 if spek == 'menengah' else 1.5
        return luas * multiplier
    
    def _formula_pondasi(self, luas: float, lantai: int, tipe: str) -> float:
        """Formula untuk pekerjaan pondasi"""
        tipe_multiplier = 1.2 if tipe == 'ruko' else 1.0
        return luas * lantai * tipe_multiplier
    
    def _formula_struktur(self, luas: float, lantai: int) -> float:
        """Formula untuk pekerjaan struktur"""
        return luas * lantai * 1.1
    
    def _formula_dinding(self, luas: float, lantai: int) -> float:
        """Formula untuk pekerjaan dinding"""
        return (luas * 0.8) * lantai  # asumsi keliling bangunan
    
    def _formula_atap(self, luas: float, spek: str) -> float:
        """Formula untuk pekerjaan atap"""
        multiplier = 1.0 if spek == 'standar' else 1.3 if spek == 'menengah' else 1.6
        return luas * multiplier
    
    def _formula_finishing(self, luas: float, lantai: int, spek: str) -> float:
        """Formula untuk pekerjaan finishing"""
        multiplier = 1.0 if spek == 'standar' else 1.5 if spek == 'menengah' else 2.5
        return luas * lantai * multiplier
    
    def _formula_mep(self, luas: float, spek: str) -> float:
        """Formula untuk pekerjaan MEP"""
        multiplier = 1.0 if spek == 'standar' else 1.8 if spek == 'menengah' else 3.0
        return luas * multiplier
    
    def _get_dependency_value(self, project: Dict, dep: str) -> Any:
        """Mendapatkan nilai dependency dari project data"""
        if dep in project:
            return project[dep]
        elif dep == 'tipe':
            return project['tipe_proyek']
        elif dep == 'spek':
            return project['spek']
        else:
            raise ValueError(f"Dependency '{dep}' tidak dikenali")
    
    def generate_project_features(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate fitur proyek (X features)"""
        projects = []
        
        for i in range(n_samples):
            project = {
                'id_proyek': f'PROJ_{i:04d}',
                'tipe_proyek': random.choice(self.tipe_proyek_list),
                'luas': self.rng.integers(36, 500),  # m2
                'lantai': self.rng.integers(1, 4),
                'spek': random.choice(self.spek_list),
                'lokasi': random.choice(self.lokasi_list),
                'tahun': self.rng.integers(2020, 2024)
            }
            projects.append(project)
        
        return pd.DataFrame(projects)
    
    def calculate_item_volume(self, project: Dict, kelompok: str, item_row: pd.Series) -> float:
        """Hitung volume item berdasarkan aturan dan karakteristik proyek"""
        if kelompok not in self.volume_rules:
            # Fallback untuk kelompok yang tidak terdefinisi
            return round(self.rng.uniform(0.1, 10.0), 2)
        
        rule = self.volume_rules[kelompok]
        
        # Siapkan argumen untuk formula
        args = []
        for dep in rule['dependencies']:
            arg_value = self._get_dependency_value(project, dep)
            args.append(arg_value)
        
        try:
            # Panggil formula dengan args
            base_volume = rule['formula'](*args)
            
            # Add random variation (±20%)
            variation = self.rng.uniform(0.8, 1.2)
            final_volume = max(0.1, base_volume * variation * rule['base_multiplier'])
            
            return round(final_volume, 2)
            
        except Exception as e:
            print(f"Error calculating volume for {kelompok}: {e}")
            print(f"Args: {args}")
            return round(self.rng.uniform(0.1, 10.0), 2)  # fallback
    
    def generate_target_volumes(self, projects_df: pd.DataFrame) -> pd.DataFrame:
        """Generate volume untuk setiap item AHSP (y targets)"""
        volume_data = []
        
        print("Generating volume data...")
        
        # Dapatkan semua kode item unik dari AHSP
        all_items = self.ahsp_data['kode_item'].unique()
        
        for idx, project in projects_df.iterrows():
            if idx % 100 == 0:
                print(f"Processing project {idx}/{len(projects_df)}")
                
            project_volumes = {'id_proyek': project['id_proyek']}
            project_dict = project.to_dict()
            
            # Hitung volume untuk setiap item berdasarkan kelompoknya
            for _, item in self.ahsp_data.iterrows():
                kode_item = item['kode_item']
                kelompok = item['kelompok']
                
                volume = self.calculate_item_volume(project_dict, kelompok, item)
                project_volumes[kode_item] = volume
            
            volume_data.append(project_volumes)
        
        return pd.DataFrame(volume_data)
    
    def generate_complete_dataset(self, n_samples: int = 1000) -> tuple:
        """Generate dataset lengkap (X features + y targets)"""
        print("Generating project features...")
        projects_df = self.generate_project_features(n_samples)
        
        print("Generating target volumes...")
        volumes_df = self.generate_target_volumes(projects_df)
        
        # Merge features dengan volumes
        complete_df = pd.merge(projects_df, volumes_df, on='id_proyek')
        
        return projects_df, volumes_df, complete_df

    def calculate_total_costs(self, volumes_df: pd.DataFrame) -> pd.DataFrame:
        """Hitung total biaya berdasarkan volume dan harga satuan"""
        print("Calculating total costs...")
        
        cost_data = []
        
        for _, row in volumes_df.iterrows():
            project_costs = {'id_proyek': row['id_proyek']}
            total_biaya = 0
            
            for _, item in self.ahsp_data.iterrows():
                kode_item = item['kode_item']
                harga_satuan = item['harga_satuan']
                volume = row[kode_item]
                
                biaya_item = volume * harga_satuan
                project_costs[kode_item + '_biaya'] = biaya_item
                total_biaya += biaya_item
            
            project_costs['total_biaya'] = total_biaya
            cost_data.append(project_costs)
        
        return pd.DataFrame(cost_data)

# =============================================================================
# CONTOH PENGGUNAAN - FIXED VERSION
# =============================================================================

def create_sample_ahsp_data() -> pd.DataFrame:
    """Buat sample data AHSP untuk testing"""
    sample_ahsp = []
    
    # Sample items per kelompok
    items_template = [
        # Persiapan
        {'kode_item': 'P001', 'kelompok': 'persiapan', 'nama_pekerjaan': 'Pembersihan Lapangan', 'satuan': 'm2', 'harga_satuan': 15000},
        {'kode_item': 'P002', 'kelompok': 'persiapan', 'nama_pekerjaan': 'Pemasangan Bouwplank', 'satuan': 'm1', 'harga_satuan': 25000},
        
        # Pondasi
        {'kode_item': 'F001', 'kelompok': 'pondasi', 'nama_pekerjaan': 'Galian Tanah Pondasi', 'satuan': 'm3', 'harga_satuan': 75000},
        {'kode_item': 'F002', 'kelompok': 'pondasi', 'nama_pekerjaan': 'Urugan Pasir Bawah Pondasi', 'satuan': 'm3', 'harga_satuan': 185000},
        {'kode_item': 'F003', 'kelompok': 'pondasi', 'nama_pekerjaan': 'Beton Pondasi', 'satuan': 'm3', 'harga_satuan': 850000},
        
        # Struktur
        {'kode_item': 'S001', 'kelompok': 'struktur', 'nama_pekerjaan': 'Beton Sloof', 'satuan': 'm3', 'harga_satuan': 950000},
        {'kode_item': 'S002', 'kelompok': 'struktur', 'nama_pekerjaan': 'Beton Kolom', 'satuan': 'm3', 'harga_satuan': 980000},
        {'kode_item': 'S003', 'kelompok': 'struktur', 'nama_pekerjaan': 'Beton Balok', 'satuan': 'm3', 'harga_satuan': 970000},
        
        # Dinding
        {'kode_item': 'D001', 'kelompok': 'dinding', 'nama_pekerjaan': 'Pasangan Dinding Bata', 'satuan': 'm2', 'harga_satuan': 125000},
        {'kode_item': 'D002', 'kelompok': 'dinding', 'nama_pekerjaan': 'Plesteran Dinding', 'satuan': 'm2', 'harga_satuan': 45000},
        
        # Atap
        {'kode_item': 'A001', 'kelompok': 'atap', 'nama_pekerjaan': 'Rangka Atap Baja Ringan', 'satuan': 'm2', 'harga_satuan': 145000},
        {'kode_item': 'A002', 'kelompok': 'atap', 'nama_pekerjaan': 'Penutup Atap Genteng', 'satuan': 'm2', 'harga_satuan': 95000},
        
        # Finishing
        {'kode_item': 'FN001', 'kelompok': 'finishing', 'nama_pekerjaan': 'Cat Dinding', 'satuan': 'm2', 'harga_satuan': 35000},
        {'kode_item': 'FN002', 'kelompok': 'finishing', 'nama_pekerjaan': 'Lantai Keramik', 'satuan': 'm2', 'harga_satuan': 125000},
        {'kode_item': 'FN003', 'kelompok': 'finishing', 'nama_pekerjaan': 'Pintu Kayu', 'satuan': 'buah', 'harga_satuan': 450000},
        
        # MEP
        {'kode_item': 'M001', 'kelompok': 'mep', 'nama_pekerjaan': 'Instalasi Listrik', 'satuan': 'm2', 'harga_satuan': 85000},
        {'kode_item': 'M002', 'kelompok': 'mep', 'nama_pekerjaan': 'Instalasi Air Bersih', 'satuan': 'm2', 'harga_satuan': 65000},
        {'kode_item': 'M003', 'kelompok': 'mep', 'nama_pekerjaan': 'Instalasi Sanitasi', 'satuan': 'm2', 'harga_satuan': 75000},
    ]
    
    return pd.DataFrame(items_template)

def main():
    """Contoh penggunaan generator yang sudah diperbaiki"""
    print("🚀 Membuat Generator Data Sintetis ML RAB (Fixed Version)...")
    
    try:
        # 1. Load atau buat sample data AHSP
        ahsp_data = create_sample_ahsp_data()
        print(f"✅ AHSP Data: {len(ahsp_data)} items")
        print("Kelompok yang tersedia:", ahsp_data['kelompok'].unique())
        
        # 2. Inisialisasi generator
        generator = RABDataGenerator(ahsp_data, seed=42)
        
        # 3. Generate dataset kecil dulu untuk testing (10 proyek)
        print("\n🔧 Generating sample dataset...")
        projects_df, volumes_df, complete_df = generator.generate_complete_dataset(n_samples=10)
        
        # 4. Hitung total biaya
        costs_df = generator.calculate_total_costs(volumes_df)
        
        # 5. Simpan hasil
        import os
        os.makedirs('data', exist_ok=True)
        
        projects_df.to_csv('data/projects_features.csv', index=False)
        volumes_df.to_csv('data/projects_volumes.csv', index=False)
        complete_df.to_csv('data/complete_dataset.csv', index=False)
        costs_df.to_csv('data/projects_costs.csv', index=False)
        ahsp_data.to_csv('data/ahsp_master.csv', index=False)
        
        print(f"\n✅ Generated {len(projects_df)} projects")
        print(f"✅ Features: {len(projects_df.columns)} columns")
        print(f"✅ Target volumes: {len(volumes_df.columns)} items")
        print(f"✅ Complete dataset shape: {complete_df.shape}")
        print(f"✅ Total costs calculated: {len(costs_df)} projects")
        
        # 6. Preview hasil
        print("\n📊 Preview Projects Features:")
        print(projects_df.head(3).to_string())
        
        print("\n📊 Preview Volumes (first 5 items):")
        volume_cols = [col for col in volumes_df.columns if col != 'id_proyek'][:5]
        print(volumes_df[['id_proyek'] + volume_cols].head(3).to_string())
        
        print("\n📊 Preview Costs:")
        cost_cols = [col for col in costs_df.columns if 'biaya' in col][:3] + ['total_biaya']
        print(costs_df[['id_proyek'] + cost_cols].head(3).to_string())
        
        return generator, projects_df, volumes_df, complete_df, costs_df, ahsp_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

def generate_large_dataset():
    """Generate dataset besar untuk training"""
    print("\n🎯 Generating large dataset for training...")
    
    ahsp_data = create_sample_ahsp_data()
    generator = RABDataGenerator(ahsp_data, seed=42)
    
    # Generate 1000 samples untuk training
    projects_large, volumes_large, complete_large = generator.generate_complete_dataset(n_samples=1000)
    costs_large = generator.calculate_total_costs(volumes_large)
    
    # Simpan dataset besar
    projects_large.to_csv('data/projects_features_large.csv', index=False)
    volumes_large.to_csv('data/projects_volumes_large.csv', index=False)
    complete_large.to_csv('data/complete_dataset_large.csv', index=False)
    costs_large.to_csv('data/projects_costs_large.csv', index=False)
    
    print(f"✅ Large dataset generated: {len(projects_large)} projects")
    return projects_large, volumes_large, complete_large, costs_large

if __name__ == "__main__":
    # Test dengan dataset kecil dulu
    result = main()
    
    if result[0] is not None:
        # Jika berhasil, generate dataset besar
        generate_large_dataset()