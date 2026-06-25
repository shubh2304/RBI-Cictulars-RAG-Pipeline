import os
import fitz
import sqlite3

conn = sqlite3.connect("rbi_rag.db")
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

def safe_print(text):
    print(text.encode('ascii', errors='replace').decode('ascii'))

circulars_dir = "circulars"
pdf_files = [f for f in os.listdir(circulars_dir) if f.lower().endswith(".pdf")]

print("Checking PDF physical pages vs database index...")
for f in pdf_files:
    pdf_path = os.path.join(circulars_dir, f)
    try:
        doc = fitz.open(pdf_path)
        actual_pages = len(doc)
        doc.close()
    except Exception as e:
        actual_pages = f"Error: {e}"
        
    cursor.execute("SELECT document_id FROM documents WHERE filename = ?", (f,))
    doc_row = cursor.fetchone()
    if doc_row:
        doc_id = doc_row["document_id"]
        cursor.execute("SELECT COUNT(*), MAX(page_number) FROM chunks WHERE document_id = ?", (doc_id,))
        chunk_row = cursor.fetchone()
        chunks_count = chunk_row[0]
        max_page_indexed = chunk_row[1]
    else:
        chunks_count = 0
        max_page_indexed = "Not in DB"
        
    safe_print(f"File: {f}")
    safe_print(f"  Physical Pages: {actual_pages}")
    safe_print(f"  DB Chunks Count: {chunks_count} | Max Indexed Page: {max_page_indexed}")
    print()

conn.close()
