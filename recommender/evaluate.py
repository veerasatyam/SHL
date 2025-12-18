"""
Fixed evaluation - matches on test type/category instead of exact names.
"""
import json
from pathlib import Path
import requests
import pandas as pd
import numpy as np

class RealisticEvaluator:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        
    def create_realistic_test_data(self):
        """Create test data based on test types, not exact names."""
        test_data = [
            {
                "query": "Java developer with Spring Boot experience",
                "desired_test_types": ["K", "S"],  # Knowledge and Skills tests
                "desired_categories": ["Cognitive", "Skills"],
                "min_relevant": 2
            },
            {
                "query": "Data analyst with SQL and Python skills",
                "desired_test_types": ["K", "N", "L"],  # Knowledge, Numerical, Logical
                "desired_categories": ["Cognitive", "Numerical"],
                "min_relevant": 2
            },
            {
                "query": "Customer service representative with communication skills",
                "desired_test_types": ["P", "V", "S"],  # Personality, Verbal, Skills
                "desired_categories": ["Personality", "Verbal"],
                "min_relevant": 2
            },
            {
                "query": "Marketing manager with digital marketing expertise",
                "desired_test_types": ["P", "V", "K"],  # Personality, Verbal, Knowledge
                "desired_categories": ["Personality", "Verbal"],
                "min_relevant": 2
            },
            {
                "query": "Software engineer with Python programming",
                "desired_test_types": ["K", "L"],  # Knowledge, Logical
                "desired_categories": ["Cognitive", "Logical"],
                "min_relevant": 2
            }
        ]
        
        # Save test data
        eval_dir = Path(__file__).parent.parent / "data" / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        test_path = eval_dir / "realistic_test_data.json"
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"Created realistic test data with {len(test_data)} queries")
        return test_data
    
    def is_relevant(self, assessment: dict, test_case: dict) -> bool:
        """Check if an assessment is relevant based on test type or category."""
        test_type = assessment.get('test_type', '')
        category = assessment.get('category', '')
        
        # Check test type match
        if test_type in test_case['desired_test_types']:
            return True
        
        # Check category match
        if category in test_case['desired_categories']:
            return True
        
        # Check if name contains relevant keywords
        query_lower = test_case['query'].lower()
        name_lower = assessment.get('assessment_name', '').lower()
        
        # Simple keyword matching
        keywords = {
            'java': ['java', 'programming', 'software', 'developer'],
            'data': ['data', 'analyst', 'analysis', 'numerical'],
            'customer': ['customer', 'service', 'verbal', 'communication'],
            'marketing': ['marketing', 'verbal', 'personality'],
            'python': ['python', 'programming', 'software']
        }
        
        for key, word_list in keywords.items():
            if key in query_lower:
                for word in word_list:
                    if word in name_lower:
                        return True
        
        return False
    
    def evaluate(self, k_values: list = [5, 10]):
        """Evaluate with realistic relevance criteria."""
        print("="*60)
        print("REALISTIC EVALUATION")
        print("="*60)
        
        test_data = self.create_realistic_test_data()
        results = []
        
        for test_case in test_data:
            query = test_case["query"]
            
            print(f"\nQuery: '{query}'")
            print(f"Desired test types: {test_case['desired_test_types']}")
            print(f"Desired categories: {test_case['desired_categories']}")
            
            # Get recommendations
            try:
                response = requests.post(
                    f"{self.api_url}/recommend",
                    json={"query": query, "top_n": max(k_values)},
                    timeout=10
                )
                
                if response.status_code != 200:
                    print(f"  API error: {response.status_code}")
                    continue
                
                data = response.json()
                recommendations = data.get('recommendations', [])
                
                if not recommendations:
                    print("  No recommendations received")
                    continue
                
                print(f"  Received {len(recommendations)} recommendations")
                
                # Check relevance for each recommendation
                relevant_count = 0
                for i, rec in enumerate(recommendations[:10]):
                    is_rel = self.is_relevant(rec, test_case)
                    if is_rel:
                        relevant_count += 1
                        print(f"    {i+1}. ✅ {rec['assessment_name'][:40]}... ({rec.get('test_type', '?')})")
                    else:
                        print(f"    {i+1}. ❌ {rec['assessment_name'][:40]}... ({rec.get('test_type', '?')})")
                
                # Calculate metrics
                for k in k_values:
                    top_k = recommendations[:k]
                    relevant_in_top_k = sum(1 for rec in top_k if self.is_relevant(rec, test_case))
                    
                    recall = relevant_in_top_k / test_case['min_relevant'] if test_case['min_relevant'] > 0 else 0
                    precision = relevant_in_top_k / k if k > 0 else 0
                    
                    results.append({
                        'query': query,
                        'k': k,
                        'recall': recall,
                        'precision': precision,
                        'relevant_found': relevant_in_top_k,
                        'total_recommendations': len(top_k)
                    })
                    
                    print(f"  Recall@{k}: {recall:.2f}, Precision@{k}: {precision:.2f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        # Calculate averages
        if results:
            df = pd.DataFrame(results)
            
            print("\n" + "="*60)
            print("OVERALL RESULTS")
            print("="*60)
            
            for k in k_values:
                k_results = df[df['k'] == k]
                if len(k_results) > 0:
                    avg_recall = k_results['recall'].mean()
                    avg_precision = k_results['precision'].mean()
                    
                    print(f"\nMetrics @ k={k}:")
                    print(f"  Average Recall@{k}:   {avg_recall:.3f} ({avg_recall*100:.1f}%)")
                    print(f"  Average Precision@{k}: {avg_precision:.3f} ({avg_precision*100:.1f}%)")
                    print(f"  Queries evaluated: {len(k_results)}")
            
            # Save results
            eval_dir = Path(__file__).parent.parent / "data" / "evaluation"
            results_path = eval_dir / "realistic_evaluation_results.csv"
            df.to_csv(results_path, index=False)
            print(f"\nResults saved to: {results_path}")
            
            return df
        else:
            print("\nNo results generated")
            return None

def main():
    """Run the evaluation."""
    print("Starting realistic evaluation...")
    
    # Check API
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("API not responding. Start it first: python api/simple_api.py")
            return
    except:
        print("API not running. Start it first: python api/simple_api.py")
        return
    
    # Run evaluation
    evaluator = RealisticEvaluator()
    results = evaluator.evaluate(k_values=[5, 10])
    
    if results is not None:
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        
        # Show summary
        recall_5 = results[results['k'] == 5]['recall'].mean()
        precision_5 = results[results['k'] == 5]['precision'].mean()
        
        print(f"\nFinal Scores:")
        print(f"Recall@5:    {recall_5:.1%}")
        print(f"Precision@5: {precision_5:.1%}")
        
        # Interpretation
        if recall_5 >= 0.8:
            print("\n✅ Excellent recall - system finds most relevant assessments")
        elif recall_5 >= 0.6:
            print("\n⚠️  Good recall - system finds many relevant assessments")
        else:
            print("\n❌ Low recall - system misses many relevant assessments")
            
        if precision_5 >= 0.4:
            print("✅ Good precision - recommendations are relevant")
        elif precision_5 >= 0.2:
            print("⚠️  Moderate precision - some recommendations are relevant")
        else:
            print("❌ Low precision - many irrelevant recommendations")

if __name__ == "__main__":
    main()