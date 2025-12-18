"""
Generate embeddings for SHL assessments and create FAISS index.
"""
import pandas as pd
import numpy as np
import faiss
import pickle
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json
from typing import List, Tuple
import time

class AssessmentEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a sentence transformer model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # Project paths
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.processed_dir = self.data_dir / "processed"
        self.embeddings_dir = self.data_dir / "embeddings"
        
        # Create directories
        self.embeddings_dir.mkdir(exist_ok=True)
        
    def load_data(self, csv_file: str = "shl_assessments.csv") -> pd.DataFrame:
        """
        Load assessment data from CSV.
        
        Args:
            csv_file: Name of CSV file in processed directory
            
        Returns:
            DataFrame with assessment data
        """
        csv_path = self.processed_dir / csv_file
        
        if not csv_path.exists():
            # Try enriched version
            enriched_path = self.processed_dir / "shl_assessments_enriched.csv"
            if enriched_path.exists():
                csv_path = enriched_path
                print(f"Using enriched data: {enriched_path}")
            else:
                raise FileNotFoundError(f"No assessment CSV found in {self.processed_dir}")
        
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Basic data validation
        required_columns = ['assessment_name', 'description']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        print(f"Loaded {len(df)} assessments")
        return df
    
    def create_text_for_embedding(self, row: pd.Series) -> str:
        """
        Combine assessment fields into a single text for embedding.
        
        Strategy: Combine name, description, skills, category, and test_type
        with appropriate weighting through repetition.
        
        Args:
            row: DataFrame row with assessment data
            
        Returns:
            Combined text string
        """
        # Start with the name (most important)
        text_parts = []
        
        # Add name 2x for emphasis
        name = str(row.get('assessment_name', ''))
        if name:
            text_parts.extend([name, name])
        
        # Add description
        description = str(row.get('description', ''))
        if description and description.lower() != 'nan':
            text_parts.append(description)
        
        # Add skills if available
        skills = str(row.get('skills', ''))
        if skills and skills.lower() != 'nan' and len(skills) > 5:
            text_parts.append(f"Skills measured: {skills}")
        
        # Add category and test type
        category = str(row.get('category', ''))
        if category and category.lower() != 'nan':
            text_parts.append(f"Category: {category}")
        
        test_type = str(row.get('test_type', ''))
        if test_type and test_type.lower() != 'nan':
            text_parts.append(f"Test type: {test_type}")
        
        # Combine all parts
        combined_text = " ".join(text_parts)
        
        # Clean up multiple spaces
        combined_text = " ".join(combined_text.split())
        
        return combined_text
    
    def generate_embeddings(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[dict]]:
        """
        Generate embeddings for all assessments.
        
        Args:
            df: DataFrame with assessment data
            
        Returns:
            embeddings: numpy array of shape (n_assessments, embedding_dim)
            metadata: list of dictionaries with assessment metadata
        """
        print("Generating embeddings...")
        
        # Prepare texts and metadata
        texts = []
        metadata = []
        
        for idx, row in df.iterrows():
            # Create combined text
            text = self.create_text_for_embedding(row)
            texts.append(text)
            
            # Store metadata
            meta = {
                'index': idx,
                'assessment_name': row.get('assessment_name', ''),
                'url': row.get('url', ''),
                'description': row.get('description', ''),
                'skills': row.get('skills', ''),
                'test_type': row.get('test_type', ''),
                'category': row.get('category', ''),
                'remote_testing': row.get('remote_testing', ''),
                'adaptive_irt': row.get('adaptive_irt', ''),
            }
            metadata.append(meta)
        
        # Generate embeddings in batches to manage memory
        batch_size = 32
        all_embeddings = []
        
        print(f"Processing {len(texts)} assessments in batches of {batch_size}...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            print(f"  Batch {i//batch_size + 1}/{(len(texts)+batch_size-1)//batch_size}")
            
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        print(f"Generated embeddings: {embeddings.shape}")
        return embeddings, metadata
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create FAISS index for fast similarity search.
        
        Args:
            embeddings: numpy array of embeddings
            
        Returns:
            FAISS index
        """
        print("Creating FAISS index...")
        
        embedding_dim = embeddings.shape[1]
        
        # Create index for cosine similarity (L2 normalized vectors)
        # For normalized vectors, cosine similarity = dot product
        index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        
        # Add embeddings to index
        index.add(embeddings)
        
        print(f"FAISS index created with {index.ntotal} vectors")
        return index
    
    def save_artifacts(self, index: faiss.Index, metadata: List[dict], embeddings: np.ndarray = None):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index: FAISS index
            metadata: List of assessment metadata
            embeddings: Optional raw embeddings for debugging
        """
        print("Saving artifacts...")
        
        # Save FAISS index
        index_path = self.embeddings_dir / "index.faiss"
        faiss.write_index(index, str(index_path))
        print(f"  FAISS index saved to: {index_path}")
        
        # Save metadata
        metadata_path = self.embeddings_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"  Metadata saved to: {metadata_path}")
        
        # Save embedding info
        info = {
            'model_name': self.model_name,
            'embedding_dim': index.d,
            'n_vectors': index.ntotal,
            'index_type': 'IndexFlatIP',
            'normalized': True
        }
        
        info_path = self.embeddings_dir / "embedding_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"  Embedding info saved to: {info_path}")
        
        # Optionally save raw embeddings for debugging
        if embeddings is not None:
            embeddings_path = self.embeddings_dir / "embeddings.npy"
            np.save(embeddings_path, embeddings)
            print(f"  Raw embeddings saved to: {embeddings_path}")
    
    def test_index(self, index: faiss.Index, metadata: List[dict], test_queries: List[str] = None):
        """
        Test the FAISS index with sample queries.
        
        Args:
            index: FAISS index
            metadata: List of assessment metadata
            test_queries: List of test queries
        """
        if test_queries is None:
            test_queries = [
                "Java developer collaboration skills",
                "Numerical reasoning test for finance",
                "Personality assessment for managers",
                "Verbal comprehension test",
                "Technical skills simulation"
            ]
        
        print("\n" + "="*60)
        print("TESTING INDEX WITH SAMPLE QUERIES")
        print("="*60)
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            
            # Search
            k = 3  # Number of results
            distances, indices = index.search(query_embedding, k)
            
            # Display results
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if idx < len(metadata):
                    meta = metadata[idx]
                    print(f"  {i+1}. {meta['assessment_name'][:50]}...")
                    print(f"     Score: {distance:.3f}, Type: {meta.get('test_type', 'N/A')}")
    
    def run(self, csv_file: str = "shl_assessments.csv"):
        """
        Run the complete embedding pipeline.
        
        Args:
            csv_file: Name of CSV file to load
        """
        print("="*60)
        print("ASSESSMENT EMBEDDING PIPELINE")
        print("="*60)
        
        start_time = time.time()
        
        # Step 1: Load data
        df = self.load_data(csv_file)
        
        # Step 2: Generate embeddings
        embeddings, metadata = self.generate_embeddings(df)
        
        # Step 3: Create FAISS index
        index = self.create_faiss_index(embeddings)
        
        # Step 4: Save artifacts
        self.save_artifacts(index, metadata, embeddings)
        
        # Step 5: Test index
        self.test_index(index, metadata)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Pipeline completed in {elapsed:.1f} seconds")
        
        return index, metadata

def main():
    """Main execution function."""
    embedder = AssessmentEmbedder()
    
    # Try to use enriched data if available
    enriched_path = embedder.processed_dir / "shl_assessments_enriched.csv"
    csv_file = "shl_assessments_enriched.csv" if enriched_path.exists() else "shl_assessments.csv"
    
    print(f"Using data file: {csv_file}")
    
    # Run pipeline
    index, metadata = embedder.run(csv_file)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total assessments indexed: {len(metadata)}")
    print(f"Embedding dimension: {index.d}")
    print(f"Index type: FAISS IndexFlatIP")
    print(f"Model: {embedder.model_name}")
    
    # Show where files are saved
    print(f"\nArtifacts saved in: {embedder.embeddings_dir}")
    print("  - index.faiss: FAISS vector index")
    print("  - metadata.pkl: Assessment metadata")
    print("  - embedding_info.json: Configuration info")
    
    # Quick verification
    print(f"\nVerification:")
    print(f"  FAISS index has {index.ntotal} vectors")
    print(f"  Metadata has {len(metadata)} entries")
    
    if index.ntotal == len(metadata):
        print("  ✅ Index and metadata are consistent")
    else:
        print("  ❌ Mismatch between index and metadata!")

if __name__ == "__main__":
    main()