import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.llm_client import LLMClient

def test_decomposition():
    print("Testing Query Decomposition...")
    # Test short query
    q_short = "What is the KCC limit?"
    decom_short = LLMClient.decompose_query(q_short)
    print(f"Short Query: '{q_short}' -> {decom_short}")
    assert decom_short == [q_short]
    
    # Test question mark split fallback
    q_multi = "What is the limit for agricultural credit? What is the RRB target? Are MSMEs eligible?"
    decom_multi = LLMClient.decompose_query(q_multi)
    print(f"Multi Query: '{q_multi}' -> {decom_multi}")
    # It should split by question mark as fallback if LLM server is not reached
    assert len(decom_multi) == 3
    print("Decomposition test passed!")

def test_chunk_merging():
    print("Testing Chunk Merging Logic...")
    chunks1 = [
        {"chunk_id": "chunk_a", "rerank_score": 0.9, "chunk_text": "text a"},
        {"chunk_id": "chunk_b", "rerank_score": 0.5, "chunk_text": "text b"}
    ]
    chunks2 = [
        {"chunk_id": "chunk_a", "rerank_score": 0.8, "chunk_text": "text a"},
        {"chunk_id": "chunk_c", "rerank_score": 0.7, "chunk_text": "text c"}
    ]
    
    merged = []
    seen = set()
    for chunk_list in [chunks1, chunks2]:
        for chunk in chunk_list:
            c_id = chunk["chunk_id"]
            if c_id not in seen:
                seen.add(c_id)
                merged.append(chunk)
            else:
                for existing in merged:
                    if existing["chunk_id"] == c_id:
                        if chunk.get("rerank_score", 0.0) > existing.get("rerank_score", 0.0):
                            existing["rerank_score"] = chunk["rerank_score"]
                        break
                        
    merged.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    print(f"Merged: {merged}")
    assert len(merged) == 3
    assert merged[0]["chunk_id"] == "chunk_a"
    assert merged[0]["rerank_score"] == 0.9
    assert merged[1]["chunk_id"] == "chunk_c"
    assert merged[2]["chunk_id"] == "chunk_b"
    print("Chunk merging test passed!")

if __name__ == "__main__":
    test_decomposition()
    test_chunk_merging()
