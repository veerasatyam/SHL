"""
Validate recommendation system against labeled train data
Compare by assessment names instead of URLs (EASIEST FIX)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Fix import - add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Now import YOUR retriever
from recommender.retrieve import SimpleRetriever

def extract_name_from_url(url):
    """
    Extract assessment name from SHL URL
    Example: 
    https://www.shl.com/solutions/products/product-catalog/view/java-8-new/
    -> "java-8-new"
    """
    if pd.isna(url):
        return ""
    
    url = str(url).strip()
    if not url:
        return ""
    
    # Get the last part of URL
    if '/' in url:
        parts = url.rstrip('/').split('/')
        if parts:
            return parts[-1].lower()
    return url.lower()

def load_train_data():
    """Load labeled training data and extract assessment names"""
    possible_paths = [
        "data/external/Gen_AI Dataset.xlsx",
        "../data/external/Gen_AI Dataset.xlsx",
        "Gen_AI Dataset.xlsx"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"📂 Loading train data from: {path}")
            df = pd.read_excel(path, sheet_name="Train-Set")
            
            # Extract assessment names from URLs
            df["assessment_name_from_url"] = df["Assessment_url"].apply(extract_name_from_url)
            
            print(f"   Found {len(df)} labeled pairs")
            print(f"   Unique queries: {df['Query'].nunique()}")
            
            # Show sample of extracted names
            print(f"\n📋 Sample of extracted names:")
            for i, row in df.head(3).iterrows():
                print(f"   URL: {row['Assessment_url'][:60]}...")
                print(f"   Name: {row['assessment_name_from_url']}")
                print()
            
            return df
    
    raise FileNotFoundError("❌ Could not find Gen_AI Dataset.xlsx")

def normalize_name(name):
    """Normalize assessment name for comparison"""
    if not name or pd.isna(name):
        return ""
    
    name = str(name).lower().strip()
    
    # Remove common prefixes/suffixes
    name = name.replace("new", "").replace("assessment", "").replace("test", "")
    name = name.replace("-", " ").replace("_", " ")
    
    # Remove extra spaces
    name = " ".join(name.split())
    
    return name

def evaluate_on_train():
    """Test system on train queries and calculate metrics"""
    print("="*60)
    print("VALIDATING AGAINST TRAIN DATA (COMPARING BY NAMES)")
    print("="*60)
    
    # Load data
    train_df = load_train_data()
    
    # Initialize YOUR retriever
    print("\n🚀 Initializing retriever...")
    retriever = SimpleRetriever()
    print("✅ Retriever ready!")
    
    queries = train_df["Query"].unique()
    
    print(f"\n📊 Found {len(queries)} unique queries in train data")
    
    results = []
    all_details = []  # Store detailed results for debugging
    
    for query_idx, query in enumerate(queries, 1):
        print(f"\n[{query_idx}/{len(queries)}] Processing: {query[:80]}...")
        
        # Get expected assessment names from train data
        expected_rows = train_df[train_df["Query"] == query]
        expected_names = expected_rows["assessment_name_from_url"].tolist()
        expected_urls = expected_rows["Assessment_url"].tolist()
        
        print(f"   Expected: {len(expected_names)} assessments")
        print(f"   Expected names: {expected_names}")
        
        # Get your system's recommendations
        try:
            recommendations = retriever.recommend(query, top_n=10)
            print(f"   Retrieved: {len(recommendations)} recommendations")
            
            # Extract names from your recommendations
            your_names = []
            for rec in recommendations:
                name = rec["assessment_name"]
                your_names.append(normalize_name(name))
                # Debug: print what you found
                if query_idx <= 2:  # Only for first 2 queries
                    print(f"      - Your rec: {name[:40]} -> {normalize_name(name)}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            your_names = []
        
        # Calculate precision@k, recall@k
        k_values = [1, 3, 5, 10]
        metrics = {"query": query}
        
        for k in k_values:
            retrieved_k = your_names[:k] if len(your_names) >= k else your_names
            expected_k = expected_names[:k] if len(expected_names) >= k else expected_names
            
            # Count matches (using normalized names)
            matches = 0
            matched_names = []
            for exp_name in expected_names:
                exp_norm = normalize_name(exp_name)
                for ret_name in retrieved_k:
                    ret_norm = normalize_name(ret_name)
                    
                    # Match if either:
                    # 1. Exact match after normalization
                    # 2. Partial match (one contains the other)
                    if (exp_norm in ret_norm or ret_norm in exp_norm or 
                        exp_norm == ret_norm):
                        matches += 1
                        matched_names.append(exp_name)
                        break
            
            precision = matches / k if k > 0 else 0
            recall = matches / len(expected_names) if expected_names else 0
            
            metrics[f"precision@{k}"] = precision
            metrics[f"recall@{k}"] = recall
            
            if k == 5:  # Show P@5 for quick feedback
                print(f"   Precision@{k}: {precision:.2f} ({matches}/{k})")
                print(f"   Recall@{k}: {recall:.2f} ({matches}/{len(expected_names)})")
                if matches > 0:
                    print(f"   Matched names: {matched_names}")
        
        results.append(metrics)
        
        # Store details for debugging
        all_details.append({
            "query": query,
            "expected_names": expected_names,
            "your_names": your_names[:10],
            "expected_urls": expected_urls,
            "your_recommendations": recommendations[:3] if recommendations else []
        })
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        
        # Create evaluation directory
        eval_dir = Path("data/evaluation")
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        output_path = eval_dir / "train_validation_results.csv"
        results_df.to_csv(output_path, index=False)
        
        # Save detailed results for debugging
        details_df = pd.DataFrame(all_details)
        details_path = eval_dir / "validation_details.json"
        details_df.to_json(details_path, orient="records", indent=2)
        
        # Calculate averages
        avg_precision_5 = results_df["precision@5"].mean()
        avg_recall_5 = results_df["recall@5"].mean()
        
        print(f"\n" + "="*60)
        print("📊 TRAIN SET PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Average Precision@5: {avg_precision_5:.2%}")
        print(f"Average Recall@5: {avg_recall_5:.2%}")
        print(f"Total queries evaluated: {len(results)}")
        
        # Show best and worst queries
        if len(results) > 0:
            print(f"\n🏆 Best performing query:")
            best_idx = results_df["precision@5"].idxmax()
            best = results_df.iloc[best_idx]
            print(f"  Query: {best['query'][:60]}...")
            print(f"  Precision@5: {best['precision@5']:.2%}")
            
            print(f"\n⚠️ Worst performing query:")
            worst_idx = results_df["precision@5"].idxmin()
            worst = results_df.iloc[worst_idx]
            print(f"  Query: {worst['query'][:60]}...")
            print(f"  Precision@5: {worst['precision@5']:.2%}")
        
        # Save summary to file
        summary_path = eval_dir / "validation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Train Data Validation Results\n")
            f.write(f"=============================\n\n")
            f.write(f"Average Precision@5: {avg_precision_5:.2%}\n")
            f.write(f"Average Recall@5: {avg_recall_5:.2%}\n")
            f.write(f"Total queries: {len(results)}\n")
        
        print(f"\n💾 Results saved to:")
        print(f"   - Metrics: {output_path}")
        print(f"   - Details: {details_path}")
        print(f"   - Summary: {summary_path}")
        
        return results_df
    else:
        print("⚠️ No results generated")
        return None

def debug_first_query():
    """Debug the first query to see what's happening"""
    print("\n" + "="*60)
    print("🔍 DEBUGGING FIRST QUERY")
    print("="*60)
    
    train_df = load_train_data()
    retriever = SimpleRetriever()
    
    # Get first query
    first_query = train_df["Query"].iloc[0]
    print(f"Query: {first_query}")
    
    # Expected
    expected = train_df[train_df["Query"] == first_query]
    print(f"\nExpected assessments ({len(expected)}):")
    for _, row in expected.iterrows():
        url_name = extract_name_from_url(row["Assessment_url"])
        print(f"  - {url_name}")
        print(f"    URL: {row['Assessment_url'][:60]}...")
    
    # Get recommendations
    print(f"\nGetting recommendations from your system...")
    recommendations = retriever.recommend(first_query, top_n=10)
    
    print(f"\nYour top 10 recommendations:")
    for i, rec in enumerate(recommendations, 1):
        name_norm = normalize_name(rec["assessment_name"])
        print(f"{i:2}. {rec['assessment_name'][:40]:40} -> {name_norm}")
        print(f"    URL: {rec.get('url', '')[:60]}...")

if __name__ == "__main__":
    # Run validation
    print("="*60)
    print("SHL RECOMMENDER VALIDATION")
    print("="*60)
    
    results = evaluate_on_train()
    
    # Optional: Debug first query
    debug_first_query()
    
    print("\n" + "="*60)
    if results is not None:
        avg_p5 = results["precision@5"].mean()
        if avg_p5 == 0:
            print("⚠️  WARNING: Zero accuracy detected!")
            print("   Your crawled assessments don't match train data.")
            print("   Try: 1. Check if you have the right assessments")
            print("        2. Use better name matching")
        else:
            print("✅ VALIDATION COMPLETE!")
    else:
        print("❌ VALIDATION FAILED!")
    print("="*60)