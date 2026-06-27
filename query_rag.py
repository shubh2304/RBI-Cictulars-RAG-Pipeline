import sys
import json
import os

# Disable Hugging Face progress bars and warning logs in the terminal
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Force transformers to only log errors, avoiding download prints
from transformers.utils import logging as tf_logging
tf_logging.set_verbosity_error()

from retrieval.fusion import HybridRetriever
from retrieval.reranker import Reranker
from generation.llm_client import LLMClient
from generation.citation_verifier import CitationVerifier

def sanitize_text(text):
    """Sanitizes text for safe terminal printing on Windows CP1252/ASCII consoles."""
    if not text:
        return ""
    # Replace carriage returns and newlines to prevent terminal line overwrites
    text = text.replace('\r', ' ').replace('\n', ' ')
    # Map common unicode characters to clean equivalents
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2010', '-')
    text = text.replace('\u20b9', 'Rs.')
    text = text.replace('\ufffd', '-')
    # Fallback encode/decode to drop any remaining unmappable codes
    return text.encode('ascii', 'replace').decode('ascii')

def query_system(query_text, top_k=5):
    # Check for simple greetings or smalltalk to respond instantly
    greeting_res = LLMClient.check_greetings_and_smalltalk(query_text)
    if greeting_res:
        print("\n" + "#" * 80)
        print("                             GENERATED ANSWER                                   ")
        print("#" * 80)
        print(greeting_res["response"])
        print("=" * 80)
        return
        
    # 1. Initialize Hybrid Search
    print("[1/4] Running Hybrid Retrieval (BM25 + FAISS)...")
    hybrid = HybridRetriever()
    
    # Decompose query if it's compound / multi-query
    sub_queries = LLMClient.decompose_query(query_text)
    if len(sub_queries) > 1:
        print(f"Decomposed into {len(sub_queries)} sub-questions:")
        for idx, sq in enumerate(sub_queries, start=1):
            print(f"  {idx}. {sq}")
            
    merged_chunks = []
    seen_chunk_ids = set()
    
    # 2. Rerank Chunks per sub-query
    print(f"[2/4] Retrieval & Reranking candidates for each sub-question...")
    for sub_q in sub_queries:
        candidates = hybrid.search(sub_q, top_k=5)
        if not candidates:
            continue
        reranked_sub = Reranker.rerank(sub_q, candidates, top_k=top_k)
        for chunk in reranked_sub:
            c_id = chunk["chunk_id"]
            if c_id not in seen_chunk_ids:
                seen_chunk_ids.add(c_id)
                merged_chunks.append(chunk)
            else:
                for existing in merged_chunks:
                    if existing["chunk_id"] == c_id:
                        if chunk.get("rerank_score", 0.0) > existing.get("rerank_score", 0.0):
                            existing["rerank_score"] = chunk["rerank_score"]
                        break
                        
    merged_chunks.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    final_chunks = merged_chunks[:15]
    
    if not final_chunks or final_chunks[0].get("rerank_score", -99.0) < 0.02:
        print("\n" + "#" * 80)
        print("                             GENERATED ANSWER                                   ")
        print("#" * 80)
        print("The query is outside the scope of the ingested RBI regulatory guidelines. I am only trained to answer questions about RBI compliance and circulars.")
        print("=" * 80)
        return
        
    # 3. Generate Answer
    print("[3/4] Generating answer from LLM...")
    llm_output = LLMClient.generate_answer(query_text, final_chunks, sub_queries=sub_queries)
    
    # 4. Verify Citations
    print("[4/4] Verifying citations against source texts...")
    final_response = CitationVerifier.verify_citations(llm_output, final_chunks)
    
    # Render final output
    print("\n" + "#" * 80)
    print("                             GENERATED ANSWER                                   ")
    print("#" * 80)
    
    # Print answer
    answer_text = final_response["response"]
    print(sanitize_text(answer_text))
    
    # Print output length metadata
    words = len(answer_text.split())
    chars = len(answer_text)
    print(f"\n[Response Length: {words} words, {chars} characters]")
    
    print("\n" + "-" * 80)
    print("                                CITATIONS                                       ")
    print("-" * 80)
    
    for cit in final_response["citations"]:
        tag = cit["citation_tag"]
        verified = "VERIFIED" if cit["verified"] else "HALLUCINATION WARNING"
        src = cit["source"]
        statement = sanitize_text(cit["statement"])
        doc = sanitize_text(src["document_name"])
        sec = sanitize_text(src["section_title"])
        pdf_url = cit.get("pdf_url", "")
        
        print(f"{tag} [{verified}]")
        print(f"  Statement   : \"{statement}\"")
        if "matched_sentence" in src:
            matched_sentence = sanitize_text(src["matched_sentence"])
            print(f"  Source Text : \"{matched_sentence}\"")
        print(f"  Source      : {doc}")
        print(f"  Location  : Page {src['page_number']} | Section: {sec}")
        if src["ref_number"] or src["circular_number"]:
            print(f"  Details   : Ref: {src['ref_number']} | Circular: {src['circular_number']}")
        if pdf_url:
            print(f"  PDF Link (Click to Open to specific page/highlight): {pdf_url}")
        print("-" * 50)
        
    if final_response["warnings"]:
        print("\nWarnings:")
        for w in final_response["warnings"]:
            print(f"  - {w}")
            
    print("=" * 80)

if __name__ == "__main__":
    # If a query is provided in the args, run it. Otherwise, prompt the user
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        query_system(user_query)
    else:
        print("Please provide a query as arguments. Example:")
        print("python query_rag.py What is the collateral-free limit for agricultural credit?")
