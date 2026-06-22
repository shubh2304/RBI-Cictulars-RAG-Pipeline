import os
import hashlib
import uuid
from database.connection import get_connection
from database.models import init_db
from ingestion.pdf_extractor import ingest_pdf
from ingestion.parser import parse_document_to_chunks

CIRCULARS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circulars")

def get_file_hash(file_path):
    """Generates a stable document ID based on the file hash."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def seed_database():
    """Ingests all PDFs in the circulars directory and saves them to the DB."""
    # Ensure tables exist
    init_db()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Scan files
    pdf_files = [f for f in os.listdir(CIRCULARS_DIR) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files in {CIRCULARS_DIR}.")
    
    for f in pdf_files:
        pdf_path = os.path.join(CIRCULARS_DIR, f)
        doc_id = get_file_hash(pdf_path)
        
        # Check if already seeded
        cursor.execute("SELECT filename FROM documents WHERE document_id = ?", (doc_id,))
        exists = cursor.fetchone()
        if exists:
            print(f"Document {f} already exists in database. Skipping ingestion.")
            continue
            
        try:
            # Ingest PDF
            ingested = ingest_pdf(pdf_path)
            meta = ingested["metadata"]
            
            # Insert document
            cursor.execute("""
                INSERT INTO documents (document_id, filename, document_name, document_type, ref_number, circular_number, pub_date, source_pdf_path, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                f,
                meta["document_name"],
                meta["document_type"],
                meta["ref_number"],
                meta["circular_number"],
                meta["pub_date"],
                f"circulars/{f}",
                meta["source_url"]
            ))
            
            # Parse chunks
            chunks = parse_document_to_chunks(ingested, doc_id)
            print(f"  Inserting {len(chunks)} chunks...")
            
            # Insert chunks
            for chunk in chunks:
                cursor.execute("""
                    INSERT INTO chunks (chunk_id, document_id, parent_chunk_id, chunk_type, page_number, chapter_title, section_title, subsection_title, chunk_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk["chunk_id"],
                    chunk["document_id"],
                    chunk["parent_chunk_id"],
                    chunk["chunk_type"],
                    chunk["page_number"],
                    chunk["chapter_title"],
                    chunk["section_title"],
                    chunk["subsection_title"],
                    chunk["chunk_text"]
                ))
            
            conn.commit()
            print(f"Successfully seeded {f}!")
            
        except Exception as e:
            conn.rollback()
            print(f"ERROR: Failed to seed {f}: {e}")
            
    conn.close()
    print("\nDatabase seeding completed.")

if __name__ == "__main__":
    seed_database()
