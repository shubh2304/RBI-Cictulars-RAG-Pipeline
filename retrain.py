import os
import sys

# Add root directory to sys.path to resolve module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.seed import seed_database
from retrieval.dense import IndexManager

def run_retrain():
    print("==================================================")
    print("Starting Pipeline Retraining...")
    print("==================================================")
    
    # 1. Ingest new documents into the database
    print("\n[Step 1/2] Seeding database with new circulars...")
    seed_database()
    
    # 2. Build/Update dense vector index
    print("\n[Step 2/2] Updating FAISS vector index & embeddings...")
    IndexManager.build_and_save_index()
    
    print("\n==================================================")
    print("Retraining completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    run_retrain()
