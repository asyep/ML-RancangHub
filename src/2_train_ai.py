import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor

print("Memulai Training AI...")

# 1. Load Data
df = pd.read_csv('dataset/historical_projects.csv')

# 2. Pisahkan Input (X) dan Output (y)
# Input: Tipe Proyek & Luas
X = df[['Tipe_Proyek', 'Luas_Tanah_m2']]

# Output: Volume pekerjaan (Target prediksi)
y = df[['Vol_Pondasi_m3', 'Vol_Dinding_m2', 'Vol_Lantai_m2', 'Vol_Atap_m2']]

# 3. Buat Pipeline AI
# Karena 'Tipe_Proyek' adalah teks, harus diubah jadi angka pakai OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Tipe_Proyek'])
    ], remainder='passthrough')

# Kita pakai Random Forest (Ensemble)
# MultiOutputRegressor artinya 1 AI bisa prediksi banyak volume sekaligus
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))
])

# 4. Latih Model
model.fit(X, y)
print("Training Selesai.")

# 5. Simpan Model
joblib.dump(model, 'src/rab_ai_model.pkl')
print("Model AI berhasil disimpan sebagai 'src/rab_ai_model.pkl'")