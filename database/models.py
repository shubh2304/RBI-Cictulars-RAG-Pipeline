from database.connection import get_connection

CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    document_name TEXT NOT NULL,
    document_type TEXT NOT NULL,
    ref_number TEXT,
    circular_number TEXT,
    pub_date TEXT,
    source_pdf_path TEXT NOT NULL,
    source_url TEXT
);
"""

CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    parent_chunk_id TEXT,
    chunk_type TEXT NOT NULL, -- 'text', 'table', 'faq_pair'
    page_number INTEGER NOT NULL,
    chapter_title TEXT,
    section_title TEXT,
    subsection_title TEXT,
    chunk_text TEXT NOT NULL,
    vector_index INTEGER, -- references row in FAISS/NumPy index
    FOREIGN KEY(document_id) REFERENCES documents(document_id)
);
"""

CREATE_QUERY_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS query_logs (
    log_id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    response_text TEXT NOT NULL,
    execution_time_ms REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_SEMANTIC_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS semantic_cache (
    cache_id TEXT PRIMARY KEY,
    query_text TEXT UNIQUE NOT NULL,
    response_text TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

def init_db():
    """Initializes the database by creating all necessary tables."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(CREATE_DOCUMENTS_TABLE)
            conn.execute(CREATE_CHUNKS_TABLE)
            conn.execute(CREATE_QUERY_LOGS_TABLE)
            conn.execute(CREATE_SEMANTIC_CACHE_TABLE)
        print("Database tables initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
