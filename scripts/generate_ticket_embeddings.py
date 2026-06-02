"""
generate_ticket_embeddings.py

Reads the cleaned evaluation dataset and generates dense vector embeddings
using SentenceTransformers (all-MiniLM-L6-v2). Saves the embedding matrix to disk.

Usage:
    python scripts/generate_ticket_embeddings.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main():
    print("--- Step 1: Starting Dense Ticket Embedding Generation ---")
    base_dir = Path(__file__).resolve().parents[1]
    
    csv_path = base_dir / "data" / "processed" / "ticket_extraction_eval.csv"
    output_path = base_dir / "data" / "processed" / "ticket_embeddings.npy"
    
    if not csv_path.exists():
        print(f"Error: Processed dataset not found at {csv_path}.")
        print("Please run the dataset build script first.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path.name} for vectorization.")
    
    print("Initializing SentenceTransformer model: all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding ticket texts into 384-dimensional dense vectors...")
    embeddings = model.encode(
        df["text"].astype(str).tolist(),
        show_progress_bar=True,
        batch_size=32
    )
    
    print(f"Saving embedding matrix to disk at: {output_path}")
    np.save(output_path, embeddings)
    print(f"--- Output Matrix Shape: {embeddings.shape} ---")

if __name__ == "__main__":
    main()
