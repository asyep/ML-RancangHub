import os
import json
import joblib
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from .feature_builder import (
    build_raw_dataset,
    train_valid_split_by_project,
    prepare_X_y,
    build_preprocessor,
)
from ..config import MODEL_DIR

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return float('inf')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

def train_v2_from_db():
    df_raw = build_raw_dataset()
    if df_raw.empty:
        raise RuntimeError("Dataset kosong, tidak bisa training.")

    df_train, df_valid = train_valid_split_by_project(df_raw)

    X_train_raw, y_train, _, meta = prepare_X_y(df_train)
    X_valid_raw, y_valid, _, _ = prepare_X_y(df_valid)

    preprocessor = build_preprocessor(meta)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_valid = preprocessor.transform(X_valid_raw)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_valid)

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    y_xgb = xgb.predict(X_valid)

    mae_rf = mean_absolute_error(y_valid, y_rf)
    mae_xgb = mean_absolute_error(y_valid, y_xgb)
    mape_rf = mape(y_valid, y_rf)
    mape_xgb = mape(y_valid, y_xgb)

    # bobot ensemble
    eps = 1e-6
    w_rf_raw = 1.0 / (mape_rf + eps)
    w_xgb_raw = 1.0 / (mape_xgb + eps)
    w_sum = w_rf_raw + w_xgb_raw
    w_rf = w_rf_raw / w_sum
    w_xgb = w_xgb_raw / w_sum

    y_ens = w_rf * y_rf + w_xgb * y_xgb
    mae_ens = mean_absolute_error(y_valid, y_ens)
    mape_ens = mape(y_valid, y_ens)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    model_version = f"v2-{timestamp}"

    joblib.dump(preprocessor, os.path.join(MODEL_DIR, f"preprocess_{model_version}.pkl"))
    joblib.dump(rf, os.path.join(MODEL_DIR, f"rf_{model_version}.pkl"))
    joblib.dump(xgb, os.path.join(MODEL_DIR, f"xgb_{model_version}.pkl"))

    meta_dict = {
        "model_version": model_version,
        "created_at_utc": timestamp,
        "metrics": {
            "mae_rf": float(mae_rf),
            "mape_rf": float(mape_rf),
            "mae_xgb": float(mae_xgb),
            "mape_xgb": float(mape_xgb),
            "mae_ensemble": float(mae_ens),
            "mape_ensemble": float(mape_ens),
        },
        "weights": {
            "w_rf": float(w_rf),
            "w_xgb": float(w_xgb),
        },
        "features": meta,
    }

    meta_path = os.path.join(MODEL_DIR, f"meta_{model_version}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    # simpan pointer latest
    latest_path = os.path.join(MODEL_DIR, "latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump({"model_version": model_version}, f)

    return meta_dict
