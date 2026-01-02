import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from .data_extractor import load_projects, load_ahsp_items, load_project_items

def build_raw_dataset() -> pd.DataFrame:
    df_projects = load_projects()
    df_ahsp = load_ahsp_items()  # Ini sudah ada flag has_coefficients
    df_items = load_project_items()  # Ini sudah di-filter di query
    
    print(f"[FeatureBuilder] Loaded {len(df_projects)} projects")
    print(f"[FeatureBuilder] Loaded {len(df_ahsp)} AHSP items")
    print(f"[FeatureBuilder] Loaded {len(df_items)} project items (filtered)")
    
    # Filter AHSP: hanya yang punya coefficients
    df_ahsp_valid = df_ahsp[df_ahsp['has_coefficients'] == 1].copy()
    print(f"[FeatureBuilder] Using {len(df_ahsp_valid)} valid AHSP items for training")
    
    # candidate (project, ahsp VALID)
    projects = df_projects[['project_id']].drop_duplicates().copy()
    ahsps = df_ahsp_valid[['ahsp_item_id']].drop_duplicates().copy()
    
    projects['key'] = 1
    ahsps['key'] = 1
    df_candidates = projects.merge(ahsps, on='key').drop(columns=['key'])
    
    # label volume
    df_items_short = df_items[['project_id', 'ahsp_item_id', 'volume']]
    df_dataset = df_candidates.merge(
        df_items_short,
        on=['project_id', 'ahsp_item_id'],
        how='left'
    )
    
    df_dataset['volume'] = df_dataset['volume'].fillna(0.0)
    
    # join fitur proyek & ahsp
    df_dataset = df_dataset.merge(df_projects, on='project_id', how='left')
    df_dataset = df_dataset.merge(
        df_ahsp_valid[['ahsp_item_id', 'ahsp_code', 'ahsp_name', 'ahsp_unit', 'ahsp_group_id', 'ahsp_group_name']], 
        on='ahsp_item_id', 
        how='left'
    )
    
    print(f"[FeatureBuilder] Final dataset: {len(df_dataset)} rows")
    
    return df_dataset

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # flag project_is_rskgm
    df['project_is_rskgm'] = df['project_name'].str.contains(
        'RSKGM', case=False, na=False
    ).astype(int)
    
    # project_type_category sederhana
    name = df['project_name'].str.lower().fillna('')
    cond_power = name.str.contains('power')
    cond_hospital = name.str.contains('rs ')
    cond_office = name.str.contains('kantor')
    
    df['project_type_category'] = np.select(
        [cond_power, cond_hospital, cond_office],
        ['power', 'hospital', 'office'],
        default='other'
    )
    
    # flag kategori dari ahsp_group_name
    group = df['ahsp_group_name'].str.lower().fillna('')
    df['ahsp_is_k3'] = group.str.contains('k3').astype(int)
    df['ahsp_is_structural'] = group.str.contains('struktur|beton').astype(int)
    df['ahsp_is_mep'] = group.str.contains('mep|mekanikal|elektrikal').astype(int)
    df['ahsp_is_panel'] = group.str.contains('panel').astype(int)
    df['ahsp_is_cable'] = group.str.contains('kabel').astype(int)
    
    return df

def train_valid_split_by_project(df: pd.DataFrame, test_size=0.2, random_state=42):
    unique_projects = df['project_id'].dropna().unique()
    train_proj, valid_proj = train_test_split(
        unique_projects,
        test_size=test_size,
        random_state=random_state
    )
    
    df_train = df[df['project_id'].isin(train_proj)].copy()
    df_valid = df[df['project_id'].isin(valid_proj)].copy()
    
    return df_train, df_valid

def prepare_X_y(df: pd.DataFrame):
    df = add_derived_features(df)
    
    keys = df[['project_id', 'ahsp_item_id', 'ahsp_code', 'ahsp_name', 'ahsp_unit']].copy()
    y = df['volume'].astype(float).values
    
    feature_cols_numeric = [
        'overhead_percentage', 'profit_percentage',
        'smkk_percentage', 'ppn_percentage',
        'ahsp_items_count',
    ]
    
    feature_cols_categ = [
        'regency_id', 'province_id', 'price_period_id',
        'project_type_category',
        'ahsp_unit', 'ahsp_group_name',
    ]
    
    feature_cols_flags = [
        'project_is_rskgm',
        'ahsp_is_k3', 'ahsp_is_structural',
        'ahsp_is_mep', 'ahsp_is_panel', 'ahsp_is_cable',
    ]
    
    X = df[feature_cols_numeric + feature_cols_categ + feature_cols_flags].copy()
    
    meta = {
        'feature_cols_numeric': feature_cols_numeric,
        'feature_cols_categ': feature_cols_categ,
        'feature_cols_flags': feature_cols_flags,
    }
    
    return X, y, keys, meta

def build_preprocessor(meta):
    numeric_features = meta['feature_cols_numeric']
    categorical_features = meta['feature_cols_categ']
    flags_features = meta['feature_cols_flags']
    
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('flags', 'passthrough', flags_features),
        ]
    )
    
    return preprocessor
