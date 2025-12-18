"""
Quick verification of the scraped SHL assessments data.
"""
import pandas as pd
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "shl_assessments.csv"

def verify_data():
    """Verify the data meets our requirements."""
    
    print("=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)
    
    # Check if file exists
    if not CSV_PATH.exists():
        print(f"❌ CSV file not found: {CSV_PATH}")
        return False
    
    print(f"✓ CSV file found: {CSV_PATH}")
    
    # Load data
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False
    
    # Check 1: Minimum 377 tests
    min_required = 377
    if len(df) >= min_required:
        print(f"✓ PASS: Have {len(df)} tests (≥ {min_required} required)")
    else:
        print(f"❌ FAIL: Only {len(df)} tests (< {min_required} required)")
        return False
    
    # Check 2: Required columns exist
    required_columns = ['assessment_name', 'url', 'description', 'skills', 'test_type', 'category']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if not missing_columns:
        print(f"✓ PASS: All required columns present")
    else:
        print(f"❌ FAIL: Missing columns: {missing_columns}")
        return False
    
    # Check 3: No empty assessment names
    empty_names = df['assessment_name'].isna().sum() + (df['assessment_name'] == '').sum()
    if empty_names == 0:
        print(f"✓ PASS: All tests have names")
    else:
        print(f"⚠ WARNING: {empty_names} tests have empty names")
    
    # Check 4: Check for pre-packaged solutions (should be filtered)
    pre_packaged_keywords = ['job solution', 'pre-packaged', 'package', 'suite']
    pre_packaged_count = 0
    for keyword in pre_packaged_keywords:
        pre_packaged_count += df['assessment_name'].str.lower().str.contains(keyword).sum()
    
    if pre_packaged_count == 0:
        print(f"✓ PASS: No pre-packaged job solutions detected")
    else:
        print(f"⚠ WARNING: {pre_packaged_count} possible pre-packaged solutions detected")
    
    # Check 5: Data preview
    print(f"\n📊 DATA PREVIEW:")
    print("-" * 80)
    print(df.head(3).to_string())
    print("-" * 80)
    
    # Check 6: Test type distribution
    print(f"\n📈 TEST TYPE DISTRIBUTION:")
    if 'test_type' in df.columns:
        type_counts = df['test_type'].value_counts()
        for test_type, count in type_counts.items():
            print(f"  {test_type}: {count} tests ({count/len(df)*100:.1f}%)")
    
    # Check 7: Category distribution
    print(f"\n📈 CATEGORY DISTRIBUTION:")
    if 'category' in df.columns:
        cat_counts = df['category'].value_counts()
        for category, count in cat_counts.items():
            print(f"  {category}: {count} tests ({count/len(df)*100:.1f}%)")
    
    # Check 8: Sample of descriptions
    print(f"\n📝 SAMPLE DESCRIPTIONS:")
    sample_descriptions = df['description'].dropna().head(3).tolist()
    for i, desc in enumerate(sample_descriptions, 1):
        preview = desc[:100] + "..." if len(desc) > 100 else desc
        print(f"  {i}. {preview}")
    
    print(f"\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    
    # Final recommendation
    if len(df) >= min_required and not missing_columns:
        print("✅ DATA IS READY FOR NEXT PHASE")
        print("\nNext step: Generate embeddings and build FAISS index")
        return True
    else:
        print("❌ DATA NEEDS IMPROVEMENT")
        print("\nNext step: Improve data quality before proceeding")
        return False

if __name__ == "__main__":
    success = verify_data()
    sys.exit(0 if success else 1)