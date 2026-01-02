import os
import json
import joblib
import numpy as np
import pandas as pd

from ..db import read_sql_df
from .data_extractor import load_valid_ahsp_items  # CHANGED: pakai load_valid_ahsp_items
from .feature_builder import add_derived_features
from ..config import MODEL_DIR

def load_latest_model():
    latest_path = os.path.join(MODEL_DIR, "latest.json")
    if not os.path.exists(latest_path):
        raise RuntimeError("Belum ada model V2 yang dilatih.")
    
    with open(latest_path, "r", encoding="utf-8") as f:
        latest = json.load(f)
    
    version = latest["model_version"]
    pre = joblib.load(os.path.join(MODEL_DIR, f"preprocess_{version}.pkl"))
    rf = joblib.load(os.path.join(MODEL_DIR, f"rf_{version}.pkl"))
    xgb = joblib.load(os.path.join(MODEL_DIR, f"xgb_{version}.pkl"))
    
    meta_path = os.path.join(MODEL_DIR, f"meta_{version}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    w_rf = meta["weights"]["w_rf"]
    w_xgb = meta["weights"]["w_xgb"]
    
    return version, pre, rf, xgb, w_rf, w_xgb, meta

def get_province_id(regency_id: str) -> str:
    sql = """
    SELECT province_id
    FROM regencies
    WHERE id = %s
    LIMIT 1;
    """
    df = read_sql_df(sql, params=(regency_id,))
    if df.empty:
        raise ValueError("regency_id tidak ditemukan di DB")
    return df.loc[0, "province_id"]

def predict_for_new_project(
    regency_id: str,
    price_period_id: str,
    project_name: str,
    description: str,
    volume_threshold: float = 0.0,
):
    version, pre, rf, xgb, w_rf, w_xgb, meta = load_latest_model()
    
    # 1) Bangun fitur proyek baru (1 baris)
    province_id = get_province_id(regency_id)
    
    # default percentage bisa kamu sesuaikan atau ambil dari config
    default_overhead = 12.0
    default_profit = 8.0
    default_smkk = 2.5
    default_ppn = 11.0
    
    df_proj_new = pd.DataFrame(
        [{
            "project_id": "NEW",
            "project_code": None,
            "project_name": project_name,
            "description": description,
            "regency_id": regency_id,
            "province_id": province_id,
            "price_period_id": price_period_id,
            "overhead_percentage": default_overhead,
            "profit_percentage": default_profit,
            "smkk_percentage": default_smkk,
            "ppn_percentage": default_ppn,
            "ahsp_items_count": np.nan,  # tidak dipakai kalau tidak dimasukkan ke fitur
            "total_amount": np.nan,  # tidak dipakai
        }]
    )
    
    # 2) Ambil HANYA AHSP yang VALID (punya resource coefficients)
    # CHANGED: Pakai load_valid_ahsp_items() bukan load_ahsp_items()
    df_ahsp = load_valid_ahsp_items()
    
    print(f"[Predictor] Loaded {len(df_ahsp)} valid AHSP items for prediction")
    
    # 3) Bentuk kandidat (project_new, semua ahsp VALID)
    df_proj_new["key"] = 1
    df_ahsp_small = df_ahsp[[
        "ahsp_item_id", "ahsp_code", "ahsp_name", "ahsp_unit", "ahsp_group_id", "ahsp_group_name"
    ]].copy()
    df_ahsp_small["key"] = 1
    df_candidates = df_proj_new.merge(df_ahsp_small, on="key").drop(columns=["key"])
    
    # 4) Set volume dummy supaya pipeline feature_builder tetap bekerja
    df_candidates["volume"] = 0.0
    
    # 5) Fitur turunan
    df_candidates = add_derived_features(df_candidates)
    
    # Susun X_raw sesuai meta features
    feature_meta = meta["features"]
    num_cols = feature_meta["feature_cols_numeric"]
    cat_cols = feature_meta["feature_cols_categ"]
    flag_cols = feature_meta["feature_cols_flags"]
    
    X_raw = df_candidates[num_cols + cat_cols + flag_cols].copy()
    
    # 6) Transform & predict
    X = pre.transform(X_raw)
    y_rf = rf.predict(X)
    y_xgb = xgb.predict(X)
    y_final = w_rf * y_rf + w_xgb * y_xgb
    
    # 7) Susun output
    df_out = df_candidates[[
        "ahsp_item_id", "ahsp_code", "ahsp_name", "ahsp_unit"
    ]].copy()
    df_out["volume_pred"] = y_final
    
    if volume_threshold is not None:
        df_out = df_out[df_out["volume_pred"] > volume_threshold]
    
    df_out = df_out.sort_values("volume_pred", ascending=False)
    
    print(f"[Predictor] Returning {len(df_out)} predicted items (threshold: {volume_threshold})")
    
    return version, df_out
