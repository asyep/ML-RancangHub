# app/ml_v2/data_extractor.py
import pandas as pd
from ..db import read_sql_df

# Minimum coefficient threshold untuk dianggap valid
MIN_COEFFICIENT_VALUE = 0.001  # Coefficient < 0.001 dianggap terlalu kecil

def load_projects() -> pd.DataFrame:
    sql = """
    SELECT
        p.id AS project_id,
        p.project_code,
        p.project_name,
        p.description,
        p.regency_id,
        r.province_id,
        p.price_period_id,
        p.overhead_percentage,
        p.profit_percentage,
        p.smkk_percentage,
        p.ppn_percentage,
        p.ahsp_items_count,
        p.total_amount
    FROM projects p
    JOIN regencies r ON r.id = p.regency_id
    WHERE p.status = 'completed'
      AND p.ahsp_items_count > 0;
    """
    return read_sql_df(sql)

def load_ahsp_items() -> pd.DataFrame:
    """
    Load AHSP items dengan validasi coefficient value.
    Coefficient valid jika >= MIN_COEFFICIENT_VALUE (0.001)
    """
    sql = f"""
    SELECT
        ai.id AS ahsp_item_id,
        ai.code AS ahsp_code,
        ai.name AS ahsp_name,
        ai.unit AS ahsp_unit,
        ag.id AS ahsp_group_id,
        ag.name AS ahsp_group_name,
        -- Flag: apakah item ini punya resource coefficients VALID
        CASE 
            WHEN EXISTS (
                SELECT 1 
                FROM ahsp_resource_coefficients arc
                WHERE arc.ahsp_item_id = ai.id
                  AND arc.coefficient >= {MIN_COEFFICIENT_VALUE}  -- CHANGED: >= threshold
                  AND arc.coefficient IS NOT NULL
            ) THEN 1 
            ELSE 0 
        END AS has_coefficients
    FROM ahsp_items ai
    JOIN ahsp_groups ag ON ag.id = ai.group_id;
    """
    return read_sql_df(sql)

def load_valid_ahsp_items() -> pd.DataFrame:
    """
    Load HANYA AHSP items yang punya resource coefficients VALID.
    Coefficient valid = >= MIN_COEFFICIENT_VALUE (0.001)
    """
    sql = f"""
    SELECT
        ai.id AS ahsp_item_id,
        ai.code AS ahsp_code,
        ai.name AS ahsp_name,
        ai.unit AS ahsp_unit,
        ag.id AS ahsp_group_id,
        ag.name AS ahsp_group_name
    FROM ahsp_items ai
    JOIN ahsp_groups ag ON ag.id = ai.group_id
    WHERE EXISTS (
        SELECT 1 
        FROM ahsp_resource_coefficients arc
        WHERE arc.ahsp_item_id = ai.id
          AND arc.coefficient >= {MIN_COEFFICIENT_VALUE}  -- CHANGED: >= threshold
          AND arc.coefficient IS NOT NULL
    );
    """
    return read_sql_df(sql)

def load_project_items() -> pd.DataFrame:
    """
    Load project items yang AHSP-nya punya valid coefficients.
    """
    sql = f"""
    SELECT
        pi.project_id,
        pi.ahsp_item_id,
        pi.volume,
        pi.unit_price,
        pi.subtotal
    FROM project_items pi
    WHERE EXISTS (
        SELECT 1 
        FROM ahsp_resource_coefficients arc
        WHERE arc.ahsp_item_id = pi.ahsp_item_id
          AND arc.coefficient >= {MIN_COEFFICIENT_VALUE}  -- CHANGED: >= threshold
          AND arc.coefficient IS NOT NULL
    );
    """
    return read_sql_df(sql)
