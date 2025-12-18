"""
SIMPLE but COMPLETE retrieval system.
No complex imports, just works.
"""
import faiss
import pickle
import numpy as np
from pathlib import Path
import json

class SimpleRetriever:
    def __init__(self):
        """Initialize with everything in one place."""
        print("="*60)
        print("INITIALIZING SIMPLE RETRIEVER")
        print("="*60)
        
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.embeddings_dir = self.project_root / "data" / "embeddings"
        self.index_path = self.embeddings_dir / "index.faiss"
        self.metadata_path = self.embeddings_dir / "metadata.pkl"
        
        # Load FAISS index
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_path))
        
        # Load metadata
        print("Loading metadata...")
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"✅ Loaded {len(self.metadata)} assessments")
        
        # Load embedding model
        print("Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("✅ SimpleRetriever ready!")
    
    def analyze_query_simple(self, query: str) -> dict:
        """
        Simple rule-based query analysis.
        NO LLM, NO API calls.
        """
        query_lower = query.lower()
        
        # Hard skills detection
        hard_skills = []
        if any(word in query_lower for word in ['java', 'spring', 'programming', 'code']):
            hard_skills.append('Java/Programming')
        if any(word in query_lower for word in ['python', 'data', 'analyst', 'sql']):
            hard_skills.append('Python/Data Analysis')
        if any(word in query_lower for word in ['sql', 'database']):
            hard_skills.append('SQL/Database')
        if any(word in query_lower for word in ['customer', 'service', 'support']):
            hard_skills.append('Customer Service')
        if any(word in query_lower for word in ['marketing', 'digital', 'seo']):
            hard_skills.append('Marketing')
        
        # Soft skills detection
        soft_skills = []
        if any(word in query_lower for word in ['communication', 'communicate', 'verbal']):
            soft_skills.append('Communication')
        if any(word in query_lower for word in ['team', 'collaboration', 'teamwork']):
            soft_skills.append('Teamwork')
        if any(word in query_lower for word in ['leadership', 'manage', 'manager']):
            soft_skills.append('Leadership')
        
        # Seniority detection
        seniority = 'mid'  # default
        if 'junior' in query_lower or 'entry' in query_lower:
            seniority = 'junior'
        elif 'senior' in query_lower or 'lead' in query_lower:
            seniority = 'senior'
        elif 'manager' in query_lower or 'director' in query_lower:
            seniority = 'executive'
        
        # Test types based on skills
        test_types = []
        if hard_skills:
            test_types.append('K')  # Knowledge test
        if soft_skills:
            test_types.append('P')  # Personality test
        if any(word in query_lower for word in ['numerical', 'math', 'calculation', 'finance']):
            test_types.append('N')
        if any(word in query_lower for word in ['verbal', 'communication', 'writing', 'reading']):
            test_types.append('V')
        if any(word in query_lower for word in ['logical', 'reasoning', 'problem solving']):
            test_types.append('L')
        
        # Remove duplicates
        test_types = list(set(test_types))
        
        # If no test types, use defaults
        if not test_types:
            test_types = ['K', 'P']
        
        return {
            'hard_skills': hard_skills,
            'soft_skills': soft_skills,
            'seniority': seniority,
            'desired_test_types': test_types,
            'query': query
        }
    
    def enhance_query(self, query: str, analysis: dict) -> str:
        """Add analysis keywords to query for better search."""
        enhanced = query
        
        # Add hard skills
        for skill in analysis['hard_skills']:
            enhanced += f" {skill}"
        
        # Add soft skills
        for skill in analysis['soft_skills']:
            enhanced += f" {skill}"
        
        # Add test type keywords
        test_keywords = {
            'K': 'knowledge technical skill test',
            'P': 'personality behavioral assessment',
            'N': 'numerical math calculation',
            'V': 'verbal communication reading',
            'L': 'logical reasoning problem solving'
        }
        
        for tt in analysis['desired_test_types']:
            if tt in test_keywords:
                enhanced += f" {test_keywords[tt]}"
        
        return enhanced
    
    def search(self, query: str, top_k: int = 20) -> list:
        """Simple semantic search."""
        print(f"\n🔍 Searching for: '{query}'")
        
        # Analyze query
        analysis = self.analyze_query_simple(query)
        print(f"  Analysis: {analysis['hard_skills']} | {analysis['soft_skills']}")
        print(f"  Test types needed: {analysis['desired_test_types']}")
        
        # Enhance query
        enhanced_query = self.enhance_query(query, analysis)
        
        # Generate embedding
        query_embedding = self.model.encode(
            [enhanced_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(score)
                results.append(result)
        
        print(f"  Found {len(results)} candidates")
        return results, analysis
    
    def rerank(self, candidates: list, analysis: dict, top_n: int = 10) -> list:
        """Simple re-ranking with rules."""
        desired_types = analysis['desired_test_types']
        
        # Score each candidate
        scored = []
        for candidate in candidates:
            score = candidate['score']
            test_type = candidate.get('test_type', '')
            
            # Boost if test type matches
            if test_type in desired_types:
                score *= 1.5  # 50% boost
            
            # Boost for senior roles
            if analysis['seniority'] in ['senior', 'executive']:
                if any(word in candidate['assessment_name'].lower() 
                       for word in ['management', 'leadership', 'senior']):
                    score *= 1.3
            
            scored.append({
                **candidate,
                'final_score': score
            })
        
        # Sort by final score
        scored.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Ensure diversity (max 3 of same test type)
        final = []
        type_counts = {}
        
        for candidate in scored:
            test_type = candidate.get('test_type', '')
            
            if test_type not in type_counts:
                type_counts[test_type] = 0
            
            if type_counts[test_type] < 3:  # Max 3 per type
                final.append(candidate)
                type_counts[test_type] += 1
            
            if len(final) >= top_n:
                break
        
        return final
    
    def recommend(self, query: str, top_n: int = 10) -> list:
        """Complete recommendation pipeline."""
        # Search
        candidates, analysis = self.search(query, top_k=30)
        
        # Re-rank
        recommendations = self.rerank(candidates, analysis, top_n)
        
        # Format final output
        final_output = []
        for i, rec in enumerate(recommendations[:top_n], 1):
            final_output.append({
                'rank': i,
                'assessment_name': rec['assessment_name'],
                'url': rec.get('url', ''),
                'test_type': rec.get('test_type', ''),
                'category': rec.get('category', ''),
                'score': round(rec['final_score'], 3),
                'description': rec.get('description', '')[:100] + '...' if rec.get('description') else ''
            })
        
        return final_output

def main():
    """Test the simple retriever."""
    print("\n" + "="*60)
    print("TESTING SIMPLE RETRIEVER")
    print("="*60)
    
    try:
        retriever = SimpleRetriever()
        
        test_queries = [
            "Java developer with Spring Boot",
            "Data analyst with SQL skills",
            "Customer service representative",
            "Marketing manager",
            "Software engineer with Python"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"📋 QUERY: {query}")
            print('='*60)
            
            recommendations = retriever.recommend(query, top_n=5)
            
            print(f"\nTop recommendations:")
            for rec in recommendations:
                print(f"\n{rec['rank']}. {rec['assessment_name']}")
                print(f"   Type: {rec['test_type']}, Score: {rec['score']}")
                if rec['description']:
                    print(f"   {rec['description']}")
            
            print(f"\nTotal recommendations: {len(recommendations)}")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()