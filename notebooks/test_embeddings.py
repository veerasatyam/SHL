"""
Test the embeddings and FAISS index.
"""
import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load index and metadata
project_root = Path(__file__).parent.parent
embeddings_dir = project_root / "data" / "embeddings"

index_path = embeddings_dir / "index.faiss"
metadata_path = embeddings_dir / "metadata.pkl"

print("Loading FAISS index...")
index = faiss.read_index(str(index_path))

print("Loading metadata...")
with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)

print(f"\nIndex has {index.ntotal} vectors")
print(f"Metadata has {len(metadata)} entries")

# Test queries
test_queries = [
    ("Java programming skills", 5),
    ("Leadership personality test", 5),
    ("Customer service simulation", 5),
    ("Data analysis numerical test", 5),
]

print("\n" + "="*60)
print("SEMANTIC SEARCH TEST")
print("="*60)

for query, k in test_queries:
    print(f"\n🔍 Query: '{query}' (top {k} results)")
    
    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    
    # Search
    distances, indices = index.search(query_embedding, k)
    
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        if idx < len(metadata):
            meta = metadata[idx]
            print(f"  {i+1}. [{meta.get('test_type', '?')}] {meta['assessment_name']}")
            print(f"     Score: {distance:.3f}, Category: {meta.get('category', 'N/A')}")
            if meta.get('description'):
                desc_preview = meta['description'][:80] + "..." if len(meta['description']) > 80 else meta['description']
                print(f"     Description: {desc_preview}")
            print()

# Test edge cases
print("\n" + "="*60)
print("EDGE CASE TESTS")
print("="*60)

edge_queries = [
    "xyz123 nonsense query",
    "",
    "very very very long query about multiple things including technical skills and personality traits for management positions in large corporations",
]

for query in edge_queries:
    print(f"\nQuery: '{query[:50]}...' if len(query) > 50 else query")
    
    if not query.strip():
        print("  Skipped empty query")
        continue
    
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(query_embedding, 3)
    
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        if idx < len(metadata):
            meta = metadata[idx]
            print(f"  {i+1}. {meta['assessment_name'][:50]}... (score: {distance:.3f})")