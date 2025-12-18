"""
SIMPLE API - No import issues, standalone.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import sys
from pathlib import Path
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup

# ============================================================================
# URL PROCESSOR
# ============================================================================

class URLProcessor:
    """Extract job description text from URLs."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def extract_from_url(self, url: str) -> str:
        """
        Extract job description text from a URL.
        """
        try:
            # Fetch the webpage
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Try to find job description section
            job_keywords = [
                "job description", "requirements", "qualifications",
                "responsibilities", "about the role", "what you'll do"
            ]
            
            # Look for sections with job keywords
            found_sections = []
            for keyword in job_keywords:
                if keyword.lower() in text.lower():
                    idx = text.lower().find(keyword.lower())
                    start = max(0, idx - 500)
                    end = min(len(text), idx + 1500)
                    section = text[start:end]
                    found_sections.append(section)
            
            # Use found sections or full text
            if found_sections:
                text = " ".join(found_sections)
            
            # Limit length
            if len(text) > 5000:
                text = text[:5000]
            
            return text
            
        except Exception as e:
            print(f"Error extracting from URL {url}: {e}")
            return f"Job description from URL: {url}"

# ============================================================================
# SIMPLE RETRIEVER CLASS
# ============================================================================

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
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("✅ SimpleRetriever ready!")
    
    def analyze_query_simple(self, query: str) -> dict:
        """Simple rule-based query analysis."""
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

# ============================================================================
# FASTAPI SETUP
# ============================================================================

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="Get SHL assessment recommendations for job descriptions or URLs",
    version="2.0.0"
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_n: Optional[int] = 10

class Recommendation(BaseModel):
    rank: int
    assessment_name: str
    url: str
    test_type: str
    category: str
    score: float
    description: str

class QueryResponse(BaseModel):
    query: str
    recommendations: List[Recommendation]
    total_found: int
    status: str = "success"

# Global retriever instance
retriever = None

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    global retriever
    try:
        retriever = SimpleRetriever()
        print("\n" + "="*60)
        print("✅ API READY on http://localhost:8000")
        print("="*60)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "SHL Assessment Recommender API",
        "version": "2.0.0",
        "endpoints": {
            "GET /": "This information",
            "GET /health": "Health check",
            "POST /recommend": "Get recommendations for job descriptions or URLs"
        },
        "example_requests": {
            "text_query": {"query": "Java developer with Spring Boot experience", "top_n": 5},
            "url_query": {"query": "https://example.com/job-posting", "top_n": 5}
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    if retriever:
        return {
            "status": "healthy",
            "assessments_loaded": len(retriever.metadata),
            "message": f"System ready with {len(retriever.metadata)} assessments",
            "supports": ["text queries", "URL processing"]
        }
    else:
        return {"status": "unhealthy", "error": "Retriever not initialized"}

@app.post("/recommend", response_model=QueryResponse)
async def recommend(request: QueryRequest):
    """
    Get SHL assessment recommendations for a job description or URL.
    
    Supports:
    1. Text queries: "Java developer with Spring Boot experience"
    2. URLs: "https://example.com/job-posting"
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Service initializing, please try again in a moment")
    
    try:
        print(f"\n📨 Received request: '{request.query[:100]}...' (top_n={request.top_n})")
        
        query_text = request.query
        is_url = False
        
        # Check if it's a URL
        url_processor = URLProcessor()
        if request.query.startswith(("http://", "https://", "www.")):
            is_url = True
            print(f"🔗 Processing URL: {request.query}")
            query_text = url_processor.extract_from_url(request.query)
            print(f"📝 Extracted text: {query_text[:200]}...")
        
        # Get recommendations
        recommendations = retriever.recommend(query_text, request.top_n)
        
        # Convert to response format
        rec_list = []
        for rec in recommendations:
            rec_list.append(Recommendation(**rec))
        
        response = QueryResponse(
            query=request.query,
            recommendations=rec_list,
            total_found=len(rec_list)
        )
        
        print(f"✅ Response sent with {len(rec_list)} recommendations")
        if is_url:
            print(f"   (Processed from URL)")
        
        return response
    
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

# Test endpoints
@app.get("/test/text")
async def test_text_endpoint():
    """Test endpoint with text query."""
    test_query = "Java developer with Spring Boot"
    
    if not retriever:
        return {"error": "Retriever not initialized"}
    
    try:
        recommendations = retriever.recommend(test_query, top_n=3)
        return {
            "test_type": "text_query",
            "test_query": test_query,
            "recommendations": recommendations,
            "message": "Test successful if you see recommendations above"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/test/url")
async def test_url_endpoint():
    """Test endpoint with mock URL."""
    test_url = "https://example.com/java-developer-job"
    
    if not retriever:
        return {"error": "Retriever not initialized"}
    
    try:
        # For demo, use a fixed text when URL is provided
        test_text = "Java developer position requiring Spring Boot, REST APIs, and microservices. Strong communication skills needed."
        recommendations = retriever.recommend(test_text, top_n=3)
        return {
            "test_type": "url_query",
            "test_url": test_url,
            "recommendations": recommendations,
            "message": "URL processing simulated successfully"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("\n🚀 Starting SHL Assessment Recommender API v2.0...")
    print("Supports: Text queries + URL processing")
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )