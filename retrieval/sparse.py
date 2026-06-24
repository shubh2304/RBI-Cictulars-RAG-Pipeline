import re
from rank_bm25 import BM25Okapi
from database.connection import get_connection

# Simple regex word tokenizer for clean, framework-free lexical matching
def tokenize_text(text):
    if not text:
        return []
    # Lowercase and extract alphanumeric words
    return re.findall(r'\b\w+\b', text.lower())

class BM25Retriever:
    """Keyword search engine using the BM25 algorithm."""

    def __init__(self):
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.load_corpus()

    def load_corpus(self):
        """Loads all text and FAQ chunks from SQLite and builds the BM25 index."""
        conn = get_connection()
        cursor = conn.cursor()
        
        # We index both normal text and FAQ pairs for keyword searches
        cursor.execute("""
            SELECT c.chunk_id, c.document_id, c.parent_chunk_id, c.chunk_type, c.page_number, 
                   c.chapter_title, c.section_title, c.chunk_text, 
                   d.document_name, d.circular_number, d.ref_number, d.filename, d.source_pdf_path
            FROM chunks c
            JOIN documents d ON c.document_id = d.document_id
            WHERE c.chunk_type IN ('text', 'faq_pair')
        """)
        rows = cursor.fetchall()
        conn.close()
        
        self.chunks = [dict(row) for row in rows]
        
        if not self.chunks:
            print("WARNING: BM25 index is empty. Seed the database first.")
            return

        # Tokenize corpus for BM25
        self.tokenized_corpus = [tokenize_text(c["chunk_text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 index built with {len(self.chunks)} passages.")

    def search(self, query, top_k=5):
        """Performs lexical search and returns top-K results with scores."""
        if not self.bm25 or not self.chunks:
            return []
            
        tokenized_query = tokenize_text(query)
        # Compute raw BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Zip chunks with scores and filter out zero-score matches
        scored_results = []
        for i, score in enumerate(scores):
            if score > 0:
                chunk = self.chunks[i].copy()
                chunk["score"] = float(score)
                scored_results.append(chunk)
                
        # Sort by score in descending order
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

if __name__ == "__main__":
    # Test execution
    bm25_retriever = BM25Retriever()
    query = "collateral free agricultural loan limit"
    print(f"\n--- Testing BM25 Retrieval for: '{query}' ---")
    results = bm25_retriever.search(query, top_k=2)
    for r in results:
        print(f"Doc: {r['document_name']} (Page {r['page_number']}) | Score: {r['score']:.4f}")
        print(f"Text snippet: {r['chunk_text'][:150]}...\n")
