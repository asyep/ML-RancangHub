import pandas as pd
import joblib
import os
import sys
import time
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelTrainer:
    def __init__(self):
        self.data_path = 'data/raw/training_data.csv'
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self): 
        print("\n" + "█"*60)
        print("🚀 REAL AI TRAINING SESSION (XGBoost + Hyperparameter Tuning)")
        print("█"*60)
        
        if not os.path.exists(self.data_path):
            print("❌ Data tidak ditemukan. Jalankan menu 1 dulu."); return

        # Load & Encode
        df = pd.read_csv(self.data_path)
        X = df[['tipe', 'luas', 'lantai', 'kualitas']]
        y = df['ahsp_labels'].apply(lambda x: str(x).split(','))
        
        mlb = MultiLabelBinarizer()
        y_encoded = mlb.fit_transform(y)
        print(f"📊 Dataset: {len(df)} Sampel | Labels: {len(mlb.classes_)} AHSP Items")

        # Pipeline
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['tipe', 'kualitas']),
                          ('num', StandardScaler(), ['luas', 'lantai'])])

        xgb = XGBClassifier(n_jobs=-1, eval_metric='logloss', tree_method='hist')
        pipeline = Pipeline([('preprocessor', preprocessor), ('clf', MultiOutputClassifier(xgb))])

        # Tuning Parameter 
        param_dist = {
            'clf__estimator__n_estimators': [100, 150],
            'clf__estimator__max_depth': [3, 5],
            'clf__estimator__learning_rate': [0.1, 0.2]
        }

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        print("\n🧠 STARTING GRID SEARCH (Mencari Settingan Terbaik)...")
        search = RandomizedSearchCV(pipeline, param_dist, n_iter=3, cv=2, verbose=3, n_jobs=-1)
        
        start = time.time()
        search.fit(X_train, y_train)
        
        print(f"\n✅ Training Selesai dalam {time.time()-start:.2f} detik.")
        print(f"   Best Params: {search.best_params_}")

        # Save
        joblib.dump(search.best_estimator_, f'{self.model_dir}/rab_model.json')
        joblib.dump(mlb, f'{self.model_dir}/label_encoder.pkl')
        print("💾 Model Tersimpan.")