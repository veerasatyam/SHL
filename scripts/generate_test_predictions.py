"""
Generate predictions for test set queries using YOUR retriever
"""
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from recommender.retrieve import SimpleRetriever

def load_test_queries():
    """Load test queries from Excel"""
    possible_paths = [
        "data/external/Gen_AI Dataset.xlsx",
        "../data/external/Gen_AI Dataset.xlsx",
        "Gen_AI Dataset.xlsx"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"Loading test data from: {path}")
            df = pd.read_excel(path, sheet_name="Test-Set")
            return df["Query"].tolist()
    
    raise FileNotFoundError("Could not find Gen_AI Dataset.xlsx")

def generate_final_predictions():
    """Generate top 10 recommendations for each test query"""
    print("="*60)
    print("GENERATING FINAL PREDICTIONS")
    print("="*60)
    
    # Load test queries
    test_queries = load_test_queries()
    print(f"Found {len(test_queries)} test queries")
    
    # Initialize YOUR retriever
    retriever = SimpleRetriever()
    
    all_predictions = []
    
    for idx, query in enumerate(test_queries, 1):
        print(f"\n[{idx}/{len(test_queries)}] Processing: {query[:80]}...")
        
        # Get recommendations
        try:
            recommendations = retriever.recommend(query, top_n=10)
            
            for rec in recommendations:
                all_predictions.append({
                    "query": query,
                    "assessment_name": rec["assessment_name"],
                    "assessment_url": rec["url"]
                })
            
            print(f"  Generated {len(recommendations)} recommendations")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            # Add empty recommendations if error
            for i in range(10):
                all_predictions.append({
                    "query": query,
                    "assessment_name": f"Error_{i+1}",
                    "assessment_url": ""
                })
    
    # Save to CSV in required format
    predictions_df = pd.DataFrame(all_predictions)
    output_file = "final_submission.csv"
    predictions_df.to_csv(output_file, index=False)
    
    print(f"\n" + "="*60)
    print(f"✅ PREDICTIONS GENERATED!")
    print("="*60)
    print(f"Saved {len(predictions_df)} predictions to: {output_file}")
    print(f"Format: {len(test_queries)} queries × 10 recommendations each")
    
    # Show sample
    print(f"\n📄 Sample of predictions:")
    print(predictions_df.head(5).to_string())

if __name__ == "__main__":
    generate_final_predictions()