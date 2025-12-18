"""
Crawls SHL Product Catalog API to extract Individual Test Solutions.
Target: https://www.shl.com/products/product-catalog/
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import os
import sys
from pathlib import Path
from urllib.parse import urljoin

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Configure paths
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_JSON_PATH = DATA_RAW_DIR / "shl_assessments_raw.json"
CSV_PATH = DATA_PROCESSED_DIR / "shl_assessments.csv"

# Configuration
BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/products/product-catalog/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}
ITEMS_PER_PAGE = 12
MAX_ITEMS = 500  # Safety limit

def scrape_catalog() -> list:
    """
    Scrapes all Individual Test Solutions from the paginated catalog.
    Returns list of test dictionaries with basic info.
    """
    all_tests = []
    page_count = 0
    
    print("Starting catalog scrape...")
    
    for start in range(0, MAX_ITEMS, ITEMS_PER_PAGE):
        params = {"start": start}
        
        try:
            response = requests.get(CATALOG_URL, headers=HEADERS, params=params, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page (start={start}): {e}")
            break
        
        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.select("tr[data-entity-id]")
        
        if not rows:
            print(f"No rows found at start={start}. Stopping.")
            break
        
        print(f"Page {page_count + 1}: Found {len(rows)} tests")
        
        for row in rows:
            test_data = extract_basic_test_info(row)
            if test_data:
                all_tests.append(test_data)
        
        page_count += 1
        
        # Check if we have a complete page
        if len(rows) < ITEMS_PER_PAGE:
            print(f"Last page reached (only {len(rows)} items).")
            break
        
        time.sleep(0.5)  # Be polite
    
    print(f"\nTotal tests scraped: {len(all_tests)}")
    return all_tests

def extract_basic_test_info(row) -> dict:
    """
    Extracts basic information from a test row.
    Returns a dictionary with fields we can get from the catalog page.
    """
    try:
        cols = row.find_all("td")
        if len(cols) < 4:
            return None
        
        # Test name (in first column)
        name_elem = cols[0].find("a")
        test_name = name_elem.get_text(strip=True) if name_elem else cols[0].get_text(strip=True)
        
        # Skip if it's a pre-packaged solution (usually contains "Job" or "Solution" in specific ways)
        # We want Individual Test Solutions
        if "job" in test_name.lower() and "solution" in test_name.lower():
            return None
        
        # Test URL (if available)
        test_url = ""
        if name_elem and name_elem.get("href"):
            test_url = urljoin(BASE_URL, name_elem["href"])
        
        # Remote testing
        remote_testing = "Yes" if cols[1].find("span", class_="catalogue__circle -yes") else "No"
        
        # Adaptive IRT
        adaptive_irt = "Yes" if cols[2].find("span", class_="catalogue__circle -yes") else "No"
        
        # Test type (K, P, S, etc.)
        test_type = cols[3].get_text(strip=True)
        
        # Try to extract category from test name or type
        category = infer_category(test_name, test_type)
        
        # Create a basic description based on available info
        description = f"{test_name} - A {category.lower()} assessment"
        if test_type:
            description += f" of type {test_type}"
        
        # Create basic skills from category
        skills = f"{category} skills assessment"
        
        return {
            "assessment_name": test_name,
            "url": test_url,
            "description": description,
            "skills": skills,
            "test_type": test_type,
            "category": category,
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
        }
    
    except Exception as e:
        print(f"Error parsing row: {e}")
        return None

def infer_category(test_name: str, test_type: str) -> str:
    """
    Infers the category from test name and type.
    """
    test_name_lower = test_name.lower()
    test_type = test_type.upper() if test_type else ""
    
    # Map test types to categories
    type_to_category = {
        "K": "Cognitive",  # Knowledge
        "P": "Personality",
        "S": "Skills",  # Simulation
        "B": "Behavioral",
        "V": "Verbal",
        "N": "Numerical",
        "L": "Logical",
        "I": "Interactive",  # Interactive
        "C": "Cognitive",
        "A": "Aptitude",
    }
    
    # First try test type mapping
    if test_type in type_to_category:
        return type_to_category[test_type]
    
    # Fallback: keyword matching in test name
    category_keywords = [
        ("verbal", "Verbal"),
        ("reading", "Verbal"),
        ("comprehension", "Verbal"),
        ("numerical", "Numerical"),
        ("math", "Numerical"),
        ("calculation", "Numerical"),
        ("logical", "Logical"),
        ("reasoning", "Logical"),
        ("abstract", "Logical"),
        ("personality", "Personality"),
        ("behavior", "Personality"),
        ("temperament", "Personality"),
        ("skill", "Skills"),
        ("simulation", "Skills"),
        ("technical", "Skills"),
        ("cognitive", "Cognitive"),
        ("aptitude", "Cognitive"),
        ("knowledge", "Cognitive"),
    ]
    
    for keyword, category in category_keywords:
        if keyword in test_name_lower:
            return category
    
    return "Cognitive"  # Default

def save_data(tests: list):
    """Saves data to raw JSON and processed CSV files."""
    
    # Save raw JSON
    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2, ensure_ascii=False)
    print(f"\nRaw JSON saved to: {RAW_JSON_PATH}")
    
    # Convert to DataFrame and save CSV
    if not tests:
        print("No tests to save!")
        return None
    
    df = pd.DataFrame(tests)
    
    # Reorder columns for better readability
    column_order = [
        "assessment_name", "url", "description", "skills", 
        "test_type", "category", "remote_testing", "adaptive_irt"
    ]
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    # Save to CSV
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"Processed CSV saved to: {CSV_PATH}")
    
    # Print summary
    print(f"\n=== DATA SUMMARY ===")
    print(f"Total tests: {len(df)}")
    
    if 'test_type' in df.columns:
        test_types = df['test_type'].value_counts()
        print(f"\nTest type distribution:")
        for tt, count in test_types.items():
            print(f"  {tt}: {count}")
    
    if 'category' in df.columns:
        categories = df['category'].value_counts()
        print(f"\nCategory distribution:")
        for cat, count in categories.items():
            print(f"  {cat}: {count}")
    
    print(f"\nFirst 3 tests:")
    for i, row in df.head(3).iterrows():
        name_preview = row['assessment_name'][:40] + "..." if len(row['assessment_name']) > 40 else row['assessment_name']
        print(f"{i+1}. {name_preview}")
        if 'test_type' in row:
            print(f"   Type: {row['test_type']}, Category: {row.get('category', 'N/A')}")
    
    return df

def verify_csv_exists():
    """Verify the CSV file was created."""
    if CSV_PATH.exists():
        print(f"\n✓ CSV file exists at: {CSV_PATH}")
        print(f"  File size: {CSV_PATH.stat().st_size} bytes")
        
        # Try to read and display first few rows
        try:
            df = pd.read_csv(CSV_PATH)
            print(f"  CSV shape: {df.shape}")
            print(f"  Columns: {', '.join(df.columns)}")
            print("\nFirst 2 rows:")
            print(df.head(2).to_string())
        except Exception as e:
            print(f"  Error reading CSV: {e}")
    else:
        print(f"\n✗ CSV file NOT FOUND at: {CSV_PATH}")

def main():
    """Main execution function."""
    print("=" * 60)
    print("SHL Assessment Catalog Scraper")
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"CSV will be saved to: {CSV_PATH}")
    print("=" * 60)
    
    # Step 1: Scrape catalog for basic test info
    tests = scrape_catalog()
    
    if not tests:
        print("No tests found. Exiting.")
        return
    
    # Step 2: Save data
    df = save_data(tests)
    
    # Step 3: Verify file was created
    verify_csv_exists()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print(f"1. Check the file at: {CSV_PATH}")
    print("2. Open it in Excel/Google Sheets or with:")
    print(f"   python -c \"import pandas as pd; df = pd.read_csv('{CSV_PATH}'); print(df.head())\"")
    print("3. We should have ≥377 Individual Test Solutions")
    print("=" * 60)

if __name__ == "__main__":
    main()