import re
import numpy as np
from retrieval.sparse import tokenize_text
from retrieval.dense import EmbeddingService

class CitationVerifier:
    """Verifies that generated answers trace back to actual, verified source facts."""

    @staticmethod
    def compute_token_jaccard(text1, text2):
        """Computes lexical Jaccard overlap from first principles."""
        words1 = set(tokenize_text(text1))
        words2 = set(tokenize_text(text2))
        if not words1 or not words2:
            return 0.0
        return len(words1.intersection(words2)) / len(words1.union(words2))

    @classmethod
    def compute_semantic_similarity(cls, text1, text2):
        """Computes semantic similarity using the cached BGE embedding model."""
        try:
            # Generate embeddings
            model = EmbeddingService.get_model()
            # Normalize embeddings to use inner product as cosine similarity
            embs = model.encode([text1, text2], normalize_embeddings=True)
            # Dot product
            similarity = float(np.dot(embs[0], embs[1]))
            return similarity
        except Exception as e:
            print(f"Error computing semantic similarity in citation verifier: {e}")
            return 0.0

    @classmethod
    def verify_citations(cls, llm_response, retrieved_chunks, similarity_threshold=0.70):
        """
        Parses LLM JSON output, validates each citation against the source chunks,
        and adds rich metadata to verified citations.
        """
        response_text = llm_response.get("response", "")
        citations = llm_response.get("citations", [])
        
        verified_citations = []
        hallucination_warnings = []
        
        for cit in citations:
            tag = cit.get("citation_tag")
            statement = cit.get("source_statement", "")
            block_idx = cit.get("source_block_index")
            
            # Convert to 0-based index
            idx = block_idx - 1 if block_idx else -1
            
            if idx < 0 or idx >= len(retrieved_chunks):
                # Invalid chunk index cited by LLM
                hallucination_warnings.append(
                    f"Warning: Citation tag {tag} referenced out-of-bounds context block index {block_idx}."
                )
                continue
                
            source_chunk = retrieved_chunks[idx]
            source_text = source_chunk["chunk_text"]
            
            # Compute matching scores
            jaccard_score = cls.compute_token_jaccard(statement, source_text)
            semantic_score = cls.compute_semantic_similarity(statement, source_text)
            
            # The statement is verified if semantic similarity is high OR direct lexical overlap is very high
            is_verified = (semantic_score >= similarity_threshold) or (jaccard_score >= 0.40)
            
            verified_cit = {
                "citation_tag": tag,
                "statement": statement,
                "verified": is_verified,
                "scores": {
                    "semantic_similarity": round(semantic_score, 4),
                    "jaccard_overlap": round(jaccard_score, 4)
                },
                # Rich metadata for citations UI
                "source": {
                    "document_name": source_chunk["document_name"],
                    "page_number": source_chunk["page_number"],
                    "section_title": source_chunk["section_title"],
                    "circular_number": source_chunk["circular_number"],
                    "ref_number": source_chunk["ref_number"]
                }
            }
            
            if not is_verified:
                hallucination_warnings.append(
                    f"Warning: Citation {tag} statement ('{statement[:60]}...') could not be verified in the source context."
                )
                
            verified_citations.append(verified_cit)
            
        return {
            "response": response_text,
            "citations": verified_citations,
            "warnings": hallucination_warnings,
            "hallucination_detected": any(not c["verified"] for c in verified_citations)
        }

if __name__ == "__main__":
    # Test citation verification
    sample_response = {
        "response": "The collateral-free limit is ₹3 Lakhs [1], but for MSMEs it is ₹20 Lakhs [2].",
        "citations": [
            {
                "citation_tag": "[1]",
                "source_statement": "The collateral-free limit is Rs. 3 Lakhs",
                "source_block_index": 1
            },
            {
                "citation_tag": "[2]",
                "source_statement": "The bank will accept collateral security for small loans",
                "source_block_index": 2
            }
        ]
    }
    
    sample_chunks = [
        {
            "document_name": "Credit Flow to Agriculture",
            "page_number": 2,
            "section_title": "1.1",
            "circular_number": "RBI/2024-25/96",
            "ref_number": "FIDD.CO.FSD.BC.No.10/05.05.010/2024-25",
            "chunk_text": "The collateral-free agricultural loan limit has been raised from Rs. 1.6 Lakhs to Rs. 3 Lakhs in agricultural credit."
        },
        {
            "document_name": "Lending to MSMEs",
            "page_number": 4,
            "section_title": "4.1",
            "circular_number": "RBI/2017-18/56",
            "ref_number": "FIDD.MSME & NFS.12/06.02.31/2017-18",
            "chunk_text": "Banks are mandated not to accept collateral security in the case of loans up to Rs. 20 Lakhs to the Micro, Small & Medium Enterprises (MSME) sector."
        }
    ]
    
    print("\n--- Verifying Sample Citations ---")
    verified = CitationVerifier.verify_citations(sample_response, sample_chunks)
    import json
    print(json.dumps(verified, indent=2))
