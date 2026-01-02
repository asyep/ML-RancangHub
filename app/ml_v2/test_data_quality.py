# app/ml_v2/test_data_quality.py
"""
Test script untuk validasi data quality dengan coefficient threshold.
Jalankan: python -m app.ml_v2.test_data_quality
"""

import sys
from .data_extractor import (
    load_ahsp_items, 
    load_valid_ahsp_items, 
    load_project_items,
    MIN_COEFFICIENT_VALUE  # Import threshold
)
from ..db import read_sql_df

def test_data_quality():
    print("="*60)
    print("DATA QUALITY TEST")
    print(f"MIN_COEFFICIENT_VALUE = {MIN_COEFFICIENT_VALUE}")
    print("="*60)
    
    # Test 1: Load all AHSP items
    print("\n[Test 1] Load all AHSP items...")
    df_ahsp_all = load_ahsp_items()
    total_ahsp = len(df_ahsp_all)
    valid_ahsp = df_ahsp_all['has_coefficients'].sum()
    invalid_ahsp = total_ahsp - valid_ahsp
    
    print(f"  Total AHSP items: {total_ahsp}")
    print(f"  Valid (has coefficients >= {MIN_COEFFICIENT_VALUE}): {valid_ahsp} ({valid_ahsp/total_ahsp*100:.2f}%)")
    print(f"  Invalid (no/too small coefficients): {invalid_ahsp} ({invalid_ahsp/total_ahsp*100:.2f}%)")
    
    # Test 2: Load valid AHSP items
    print("\n[Test 2] Load valid AHSP items...")
    df_ahsp_valid = load_valid_ahsp_items()
    print(f"  Loaded {len(df_ahsp_valid)} valid AHSP items")
    
    if len(df_ahsp_valid) == valid_ahsp:
        print("  ✓ load_valid_ahsp_items() works correctly")
    else:
        print(f"  ✗ Mismatch! Expected {valid_ahsp}, got {len(df_ahsp_valid)}")
        return False
    
    # Test 3: Verify coefficient values
    print(f"\n[Test 3] Verify coefficient values (>= {MIN_COEFFICIENT_VALUE})...")
    valid_ahsp_ids = tuple(df_ahsp_valid['ahsp_item_id'].values)
    
    if len(valid_ahsp_ids) == 0:
        print("  ✗ No valid AHSP items found!")
        return False
    
    # Check coefficients
    sql_coeff = f"""
    SELECT 
        ahsp_item_id,
        resource_id,
        coefficient,
        CASE 
            WHEN coefficient IS NULL THEN 'NULL'
            WHEN coefficient < {MIN_COEFFICIENT_VALUE} THEN 'TOO_SMALL'
            ELSE 'VALID'
        END as coeff_status
    FROM ahsp_resource_coefficients
    WHERE ahsp_item_id IN ({','.join(['%s']*len(valid_ahsp_ids))})
    """
    df_coeffs = read_sql_df(sql_coeff, params=valid_ahsp_ids)
    
    total_coeffs = len(df_coeffs)
    valid_coeffs = len(df_coeffs[df_coeffs['coeff_status'] == 'VALID'])
    null_coeffs = len(df_coeffs[df_coeffs['coeff_status'] == 'NULL'])
    small_coeffs = len(df_coeffs[df_coeffs['coeff_status'] == 'TOO_SMALL'])
    
    print(f"  Total coefficients: {total_coeffs}")
    print(f"  Valid (>= {MIN_COEFFICIENT_VALUE}): {valid_coeffs} ({valid_coeffs/total_coeffs*100:.2f}%)")
    print(f"  NULL: {null_coeffs}")
    print(f"  Too small (< {MIN_COEFFICIENT_VALUE}): {small_coeffs}")
    
    if null_coeffs > 0 or small_coeffs > 0:
        print(f"  ✗ Found invalid coefficients! This should not happen.")
        print("\n  Sample invalid coefficients:")
        invalid_samples = df_coeffs[df_coeffs['coeff_status'] != 'VALID'].head(10)
        print(invalid_samples.to_string(index=False))
        return False
    else:
        print(f"  ✓ All coefficients are valid (>= {MIN_COEFFICIENT_VALUE})")
    
    # Test 4: Check problematic resources
    print("\n[Test 4] Check problematic resources...")
    sql_resources = f"""
    SELECT 
        r.id,
        r.code,
        r.name,
        COUNT(arc.id) as total_coeffs,
        COUNT(CASE WHEN arc.coefficient >= {MIN_COEFFICIENT_VALUE} THEN 1 END) as valid_coeffs,
        MIN(arc.coefficient) as min_coeff,
        MAX(arc.coefficient) as max_coeff,
        AVG(arc.coefficient) as avg_coeff
    FROM resources r
    LEFT JOIN ahsp_resource_coefficients arc ON arc.resource_id = r.id
    WHERE r.code LIKE 'L-%'
    GROUP BY r.id, r.code, r.name
    ORDER BY r.code
    LIMIT 10;
    """
    df_resources = read_sql_df(sql_resources)
    print("  Sample L-series resources:")
    print(df_resources.to_string(index=False))
    
    # Check if L-02 exists and has valid coefficients
    l02_resources = df_resources[df_resources['code'] == 'L-02']
    if not l02_resources.empty:
        l02 = l02_resources.iloc[0]
        print(f"\n  Resource L-02 found:")
        print(f"    ID: {l02['id']}")
        print(f"    Name: {l02['name']}")
        print(f"    Total coefficients: {l02['total_coeffs']}")
        print(f"    Valid coefficients (>= {MIN_COEFFICIENT_VALUE}): {l02['valid_coeffs']}")
        print(f"    Min coeff: {l02['min_coeff']}")
        print(f"    Max coeff: {l02['max_coeff']}")
        print(f"    Avg coeff: {l02['avg_coeff']}")
        
        if l02['valid_coeffs'] == 0:
            print(f"  ⚠ Resource L-02 has NO valid coefficients (all < {MIN_COEFFICIENT_VALUE})")
        else:
            print(f"  ✓ Resource L-02 has {l02['valid_coeffs']} valid coefficients")
    else:
        print("  ℹ Resource L-02 not found in database")
    
    # Test 5: Load project items
    print("\n[Test 5] Load project items...")
    df_project_items = load_project_items()
    print(f"  Loaded {len(df_project_items)} project items (filtered)")
    
    # Test 6: Verify all project items are valid
    print("\n[Test 6] Verify all project items have valid AHSP...")
    valid_ahsp_ids_set = set(df_ahsp_valid['ahsp_item_id'].values)
    project_item_ahsp_ids = set(df_project_items['ahsp_item_id'].values)
    
    invalid_items = project_item_ahsp_ids - valid_ahsp_ids_set
    
    if len(invalid_items) == 0:
        print(f"  ✓ All {len(df_project_items)} project items have valid AHSP")
    else:
        print(f"  ✗ Found {len(invalid_items)} project items with invalid AHSP:")
        print(f"    Invalid AHSP IDs: {list(invalid_items)[:10]}")
        return False
    
    # Test 7: Data loss estimation
    print("\n[Test 7] Estimate data loss...")
    sql_total = "SELECT COUNT(*) as total FROM project_items"
    df_total = read_sql_df(sql_total)
    total_before = df_total.loc[0, 'total']
    total_after = len(df_project_items)
    
    if total_before > 0:
        data_loss = ((total_before - total_after) / total_before) * 100
        print(f"  Project items before filter: {total_before}")
        print(f"  Project items after filter: {total_after}")
        print(f"  Data loss: {data_loss:.2f}%")
        
        if data_loss < 15:  # Increased tolerance karena filter lebih ketat
            print("  ✓ Data loss acceptable (<15%)")
        else:
            print(f"  ⚠ Data loss is {data_loss:.2f}% - consider reviewing MIN_COEFFICIENT_VALUE")
    else:
        print("  ℹ No project items in database yet")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nYou can now safely retrain the model with:")
    print("  python -m app.ml_v2.trainer")
    
    return True

if __name__ == "__main__":
    success = test_data_quality()
    sys.exit(0 if success else 1)
