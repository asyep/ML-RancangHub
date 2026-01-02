# app/ml_v2/test_prediction.py (atau letakkan di root project)
import sys
from app.ml_v2.predictor import predict_for_new_project
from app.ml_v2.data_extractor import load_valid_ahsp_items

def test_prediction():
    print("="*60)
    print("PREDICTION TEST")
    print("="*60)
    
    # ========================================
    # GANTI DENGAN UUID DARI DATABASE KAMU
    # ========================================
    regency_id = "6e0867b3-9420-47fe-a112-7015114a626e"  # Contoh: "550e8400-e29b-41d4-a716-446655440000"
    price_period_id = "2b58f768-844e-4ddc-87e1-288034842bdb"  # Contoh: "660e8400-e29b-41d4-a716-446655440000"
    
    # Test case 1: Project sekolah
    print("\n[Test Case 1] Pembangunan Gedung Sekolah")
    print("-" * 60)
    
    try:
        version, df_results = predict_for_new_project(
            regency_id=regency_id,
            price_period_id=price_period_id,
            project_name="Pembangunan Gedung Sekolah 2 Lantai",
            description="Pembangunan gedung sekolah dengan luas 500m2, 2 lantai, kapasitas 20 ruang kelas",
            volume_threshold=0.1  # Filter prediksi volume < 0.1
        )
        
        print(f"✓ Model version: {version}")
        print(f"✓ Predicted {len(df_results)} items")
        
        # Show top 10
        print("\nTop 10 Predictions:")
        print(df_results.head(10).to_string(index=False))
        
        # Verify all valid
        valid_ids = set(load_valid_ahsp_items()['ahsp_item_id'].values)
        predicted_ids = set(df_results['ahsp_item_id'].values)
        all_valid = predicted_ids.issubset(valid_ids)
        
        print(f"\n✓ All predictions valid: {all_valid}")
        
        if not all_valid:
            invalid = predicted_ids - valid_ids
            print(f"✗ Found {len(invalid)} invalid predictions:")
            print(f"  Invalid IDs: {list(invalid)[:5]}")
            return False
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test case 2: Project rumah sakit
    print("\n" + "="*60)
    print("[Test Case 2] Pembangunan Rumah Sakit")
    print("-" * 60)
    
    try:
        version, df_results = predict_for_new_project(
            regency_id=regency_id,
            price_period_id=price_period_id,
            project_name="Pembangunan RS Umum",
            description="Pembangunan rumah sakit umum 3 lantai dengan luas 2000m2",
            volume_threshold=0.5  # Threshold lebih tinggi
        )
        
        print(f"✓ Model version: {version}")
        print(f"✓ Predicted {len(df_results)} items (threshold: 0.5)")
        
        # Show statistics
        print("\nPrediction Statistics:")
        print(f"  Min volume: {df_results['volume_pred'].min():.4f}")
        print(f"  Max volume: {df_results['volume_pred'].max():.4f}")
        print(f"  Mean volume: {df_results['volume_pred'].mean():.4f}")
        print(f"  Median volume: {df_results['volume_pred'].median():.4f}")
        
        # Group by unit
        print("\nTop 5 Units by Prediction Count:")
        unit_counts = df_results['ahsp_unit'].value_counts().head(5)
        for unit, count in unit_counts.items():
            print(f"  {unit}: {count} items")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("ALL PREDICTION TESTS PASSED!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Integrate with backend API endpoint")
    print("  2. Test from frontend AI Project Wizard")
    print("  3. Monitor prediction quality in production")
    
    return True

if __name__ == "__main__":
    success = test_prediction()
    sys.exit(0 if success else 1)
