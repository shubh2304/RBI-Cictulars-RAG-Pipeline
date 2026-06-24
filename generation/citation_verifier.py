import re
import numpy as np
import json
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
        
        # Robust recovery logic for response text if the model returned non-standard JSON keys
        if not response_text:
            non_citations_keys = [k for k in llm_response.keys() if k.lower() not in ["citations", "references"]]
            if non_citations_keys:
                if len(non_citations_keys) == 1 and isinstance(llm_response[non_citations_keys[0]], str):
                    response_text = llm_response[non_citations_keys[0]]
                else:
                    parts = []
                    for k in non_citations_keys:
                        val = llm_response[k]
                        if isinstance(val, list):
                            val_str = ", ".join(map(str, val))
                        elif isinstance(val, dict):
                            val_str = "\n" + json.dumps(val, indent=2)
                        else:
                            val_str = str(val)
                        parts.append(f"**{k}**:\n{val_str}")
                    response_text = "\n\n".join(parts)
        
        citations = llm_response.get("citations", [])
        # Robust recovery logic for citations if the model used the key "references" or other names
        if not citations:
            for alt_key in ["references", "sources", "citation"]:
                if alt_key in llm_response:
                    alt_val = llm_response[alt_key]
                    if isinstance(alt_val, list):
                        citations = []
                        for item in alt_val:
                            if isinstance(item, dict):
                                citations.append(item)
                            elif isinstance(item, str):
                                tag_match = re.search(r'(\[\d+\])', item)
                                tag = tag_match.group(1) if tag_match else "[1]"
                                idx_match = re.search(r'\b(\d+)\b', item)
                                block_idx = int(idx_match.group(1)) if idx_match else 1
                                citations.append({
                                    "citation_tag": tag,
                                    "source_statement": item,
                                    "source_block_index": block_idx
                                })
                    break
        
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
            
        # Enrich response text with detailed inline references
        enriched_response = response_text
        for cit in verified_citations:
            tag = cit["citation_tag"]
            src = cit["source"]
            ref_details = f"{src['document_name']}, Page {src['page_number']}"
            if src["section_title"] and src["section_title"] != "N/A":
                ref_details += f", Sec: {src['section_title']}"
            if src["circular_number"]:
                ref_details += f", Circ: {src['circular_number']}"
                
            if cit["verified"]:
                new_tag = f"{tag[:-1]}: {ref_details}]"
            else:
                new_tag = f"{tag[:-1]}: [UNVERIFIED] {ref_details}]"
                
            enriched_response = enriched_response.replace(tag, new_tag)
            
        return {
            "response": enriched_response,
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
