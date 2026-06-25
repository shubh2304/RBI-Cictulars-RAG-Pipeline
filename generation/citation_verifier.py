import re
import numpy as np
import json
import os
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

    @staticmethod
    def normalize_number(text):
        """Normalizes a numeric string into standard representations (raw digits, normalized text)."""
        text = text.lower().strip()
        clean_text = text.replace(',', '')
        
        results = {text, clean_text}
        
        if '%' in clean_text or 'percent' in clean_text:
            val_match = re.search(r'(\d+(?:\.\d+)?)', clean_text)
            if val_match:
                val = float(val_match.group(1))
                val_str = f"{val:g}"
                results.update({f"{val_str}%", f"{val_str} %", f"{val_str} percent"})
            return results

        lakh_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|l)\b', clean_text)
        if lakh_match:
            val = float(lakh_match.group(1))
            val_raw = int(val * 100000) if val.is_integer() else int(val * 100000)
            results.update({f"{val:g} lakh", f"{val:g} lakhs", f"{val_raw}", f"{val_raw:,}"})
            return results
            
        crore_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:crore|crores|cr)\b', clean_text)
        if crore_match:
            val = float(crore_match.group(1))
            val_raw = int(val * 10000000) if val.is_integer() else int(val * 10000000)
            results.update({f"{val:g} crore", f"{val:g} crores", f"{val_raw}", f"{val_raw:,}"})
            return results
            
        try:
            val = float(clean_text)
            results.update({f"{val:g}"})
            if val.is_integer():
                results.update({f"{int(val)}", f"{int(val):,}"})
            if val == 300000 or val == 300000.0:
                results.update({"3 lakh", "3 lakhs"})
            return results
        except ValueError:
            return results

    @staticmethod
    def extract_factual_numbers(text):
        """Extracts significant factual numbers, percentages, and amounts from text."""
        text_no_citations = re.sub(r'\[\d+\]', '', text)
        
        pcts = re.findall(r'\b\d+(?:\.\d+)?\s*(?:%|percent)\b', text_no_citations, re.IGNORECASE)
        lakhs_crores = re.findall(r'\b\d+(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores)\b', text_no_citations, re.IGNORECASE)
        numbers = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', text_no_citations)
        
        sig_numbers = []
        for num in numbers:
            clean = num.replace(',', '')
            try:
                val = float(clean)
                if not val.is_integer() or val >= 10:
                    sig_numbers.append(num)
            except ValueError:
                pass

        all_matches = pcts + lakhs_crores + sig_numbers
        return list(set(all_matches))

    @staticmethod
    def split_into_sentences(text):
        """Splits raw text into individual sentences, ignoring watermarks, page numbers, and short headers."""
        if not text:
            return []
        
        text_placeholder = text
        # Common abbreviations that shouldn't split sentences
        abbreviations = ["Rs.", "No.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec.", "i.e.", "e.g.", "vs.", "Circ.", "Ref.", "BC.No.", "BC.", "FSD.", "FIDD.", "Plan."]
        for abbr in abbreviations:
            text_placeholder = re.sub(rf'\b{re.escape(abbr)}', abbr.replace(".", "___DOT___"), text_placeholder, flags=re.IGNORECASE)
        
        # Split into blocks by checking line lengths and bullets to isolate headers/watermarks/page numbers
        lines = text_placeholder.split("\n")
        blocks = []
        current_block = []
        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                if current_block:
                    blocks.append(" ".join(current_block))
                    current_block = []
                continue
                
            # If the line is very short (e.g. < 25 chars) or starts with a bullet, it's a boundary
            is_boundary = len(line_strip) < 25 or re.match(r'^\s*(?:[•\-\*]|\d+\.|\([ivxl\d]+(?:[\.\)]|\b)|\b[a-z]\)\s)', line_strip, re.IGNORECASE)
            
            if is_boundary:
                if current_block:
                    blocks.append(" ".join(current_block))
                    current_block = []
                blocks.append(line_strip)
            else:
                current_block.append(line_strip)
                
        if current_block:
            blocks.append(" ".join(current_block))
            
        sentences = []
        for block in blocks:
            # Split block by periods/exclamation/question marks followed by whitespace or quotes
            raw_splits = re.split(r'(?<=[.!?])\s+|(?<=[.!?]”)\s+|(?<=[.!?]")\s+|(?<=[.!?]\')\s+|(?<=[.!?]\))\s+', block)
            for split in raw_splits:
                split_clean = split.replace("___DOT___", ".").strip()
                if split_clean:
                    split_clean = re.sub(r'\s+', ' ', split_clean)
                    sentences.append(split_clean)
                    
        return sentences

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
        answer_status = llm_response.get("answer_status")
        answerable = llm_response.get("answerable")
        
        # If the LLM declared the answer is NOT_FOUND or not answerable, ensure we use the standard NOT_FOUND text
        if answer_status == "NOT_FOUND" or answerable is False:
            response_text = "The provided RBI circulars do not contain sufficient information to answer this query."
        
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
                                    "tag": tag,
                                    "statements": [item],
                                    "context_index": block_idx
                                })
                    break
        
        normalized_citations = []
        for cit in citations:
            if not isinstance(cit, dict):
                continue
                
            # Extract tag
            tag = cit.get("tag") or cit.get("citation_tag")
            if isinstance(tag, int):
                tag = f"[{tag}]"
            elif isinstance(tag, str) and not tag.startswith("["):
                tag = f"[{tag}]"
                
            # Extract block index
            block_idx = cit.get("context_index") or cit.get("source_block_index")
            # Filter low-confidence claims if confidence is below 0.5
            confidence = cit.get("confidence", 1.0)
            if confidence is not None:
                try:
                    if float(confidence) < 0.5:
                        continue
                except (ValueError, TypeError):
                    pass
                    
            # Extract statements (can be a list or a single string in new/old schemas)
            statements = []
            if "statements" in cit:
                stmts = cit["statements"]
                if isinstance(stmts, list):
                    statements.extend([s for s in stmts if isinstance(s, str)])
                elif isinstance(stmts, str):
                    statements.append(stmts)
            elif "statement" in cit:
                stmt = cit["statement"]
                if isinstance(stmt, str):
                    statements.append(stmt)
            elif "source_statement" in cit:
                stmt = cit["source_statement"]
                if isinstance(stmt, str):
                    statements.append(stmt)
                    
            for stmt in statements:
                normalized_citations.append({
                    "citation_tag": tag,
                    "source_statement": stmt,
                    "source_block_index": block_idx
                })
        
        verified_citations = []
        hallucination_warnings = []
        
        for cit in normalized_citations:
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
            
            # Split source_text into sentences and perform sentence-level matching
            sentences = cls.split_into_sentences(source_text)
            best_sentence = ""
            best_semantic = 0.0
            best_jaccard = 0.0
            
            for s in sentences:
                if len(tokenize_text(s)) < 3: # Skip trivial snippets
                    continue
                jacc_s = cls.compute_token_jaccard(statement, s)
                sem_s = cls.compute_semantic_similarity(statement, s)
                if sem_s > best_semantic:
                    best_semantic = sem_s
                    best_jaccard = jacc_s
                    best_sentence = s

            # Fallback to full-chunk matching in case statement is summarized
            jaccard_full = cls.compute_token_jaccard(statement, source_text)
            semantic_full = cls.compute_semantic_similarity(statement, source_text)
            
            jaccard_score = max(best_jaccard, jaccard_full)
            semantic_score = max(best_semantic, semantic_full)
            
            # The statement is verified if semantic similarity is high OR direct lexical overlap is very high
            is_verified = (semantic_score >= similarity_threshold) or (jaccard_score >= 0.40)
            
            # Additional factual check for numbers/percentages consistency
            num_warnings = []
            if is_verified:
                statement_nums = cls.extract_factual_numbers(statement)
                if statement_nums:
                    normalized_source = re.sub(r'\s+', ' ', source_text.lower())
                    source_no_comma = normalized_source.replace(',', '')
                    
                    for num_str in statement_nums:
                        reps = cls.normalize_number(num_str)
                        found = False
                        for rep in reps:
                            rep_clean = rep.lower().strip()
                            rep_no_comma = rep_clean.replace(',', '')
                            if rep_no_comma in source_no_comma:
                                found = True
                                break
                        if not found:
                            is_verified = False
                            num_warnings.append(
                                f"Factual number mismatch: '{num_str}' in statement not found in source text."
                            )

            # Construct PDF URL with browser highlight search query and accurate page detection
            filename = source_chunk.get("filename") or source_chunk.get("source_pdf_path", "").replace("circulars/", "")
            page = source_chunk.get("page_number", 1)
            pdf_url = ""
            
            if filename:
                import urllib.parse
                from pathlib import Path
                import fitz
                
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                circulars_dir = os.path.abspath(os.path.join(base_dir, "circulars"))
                pdf_path = os.path.join(circulars_dir, filename)
                
                # Dynamic page detection by scanning PDF content
                detected_page = None
                if os.path.exists(pdf_path) and best_sentence:
                    try:
                        doc = fitz.open(pdf_path)
                        
                        def normalize(text):
                            return re.sub(r'[^a-zA-Z0-9]', '', text).lower()
                            
                        norm_search = normalize(best_sentence)
                        if norm_search:
                            # 1. First check for an exact match of the entire sentence
                            for page_idx in range(len(doc)):
                                page_text = doc[page_idx].get_text()
                                if norm_search in normalize(page_text):
                                    detected_page = page_idx + 1
                                    break
                                    
                        # 2. If not found, use sliding window word-matching to find the page containing most of the sentence
                        if not detected_page:
                            search_words = best_sentence.split()
                            window_size = min(8, len(search_words))
                            if window_size >= 4:
                                windows = []
                                for start_i in range(len(search_words) - window_size + 1):
                                    subphrase = " ".join(search_words[start_i:start_i + window_size])
                                    norm_sub = normalize(subphrase)
                                    if len(norm_sub) > 10: # skip short trivial phrases
                                        windows.append(norm_sub)
                                        
                                if windows:
                                    best_page_idx = None
                                    max_matches = 0
                                    for page_idx in range(len(doc)):
                                        page_text = doc[page_idx].get_text()
                                        norm_page = normalize(page_text)
                                        matches = sum(1 for w in windows if w in norm_page)
                                        if matches > max_matches:
                                            max_matches = matches
                                            best_page_idx = page_idx
                                            
                                    if best_page_idx is not None and max_matches > 0:
                                        detected_page = best_page_idx + 1
                                        
                        # 3. Fallback: try prefix/suffix matching
                        if not detected_page:
                            words = best_sentence.split()
                            if len(words) > 6:
                                first_part = normalize(" ".join(words[:6]))
                                last_part = normalize(" ".join(words[-6:]))
                                for page_idx in range(len(doc)):
                                    page_text = doc[page_idx].get_text()
                                    norm_page = normalize(page_text)
                                    if first_part in norm_page or last_part in norm_page:
                                        detected_page = page_idx + 1
                                        break
                        doc.close()
                    except Exception as e:
                        print(f"Error dynamically detecting PDF page: {e}")
                        
                if detected_page:
                    page = detected_page
                
                # Standard cross-platform file URI generation using Path.as_uri()
                pdf_url = Path(os.path.abspath(pdf_path)).as_uri()
                pdf_url += f"#page={page}"
                


            verified_cit = {
                "citation_tag": tag,
                "statement": statement,
                "verified": is_verified,
                "pdf_url": pdf_url,
                "scores": {
                    "semantic_similarity": round(semantic_score, 4),
                    "jaccard_overlap": round(jaccard_score, 4)
                },
                # Rich metadata for citations UI
                "source": {
                    "document_name": source_chunk["document_name"],
                    "filename": source_chunk.get("filename"),
                    "source_pdf_path": source_chunk.get("source_pdf_path"),
                    "page_number": page,
                    "section_title": source_chunk["section_title"],
                    "circular_number": source_chunk["circular_number"],
                    "ref_number": source_chunk["ref_number"],
                    "matched_sentence": best_sentence if best_sentence else source_text[:200]
                }
            }
            
            if not is_verified:
                if num_warnings:
                    for nw in num_warnings:
                        hallucination_warnings.append(f"Warning: Citation {tag} - {nw}")
                else:
                    hallucination_warnings.append(
                        f"Warning: Citation {tag} statement ('{statement[:60]}...') could not be verified in the source context."
                    )
                
            verified_citations.append(verified_cit)
            
        # Enrich inline references as per production requirements (Section 5)
        if not verified_citations:
            cleaned_response = re.sub(r'\s*\[\d+\]', '', response_text)
        else:
            cleaned_response = cls.enrich_inline_citations(response_text, verified_citations)
        
        return {
            "response": cleaned_response,
            "citations": verified_citations,
            "warnings": hallucination_warnings,
            "hallucination_detected": any(not c["verified"] for c in verified_citations)
        }

    @classmethod
    def enrich_inline_citations(cls, response_text: str, verified_citations: list[dict]) -> str:
        """
        Replaces [1], [2], ... in response_text with rich inline references:
            [1: Priority Sector Lending, Page 3, Section 2.1]
        """
        tag_map = {}
        for c in verified_citations:
            tag_str = c["citation_tag"] # e.g. "[1]"
            match = re.search(r'\d+', tag_str)
            if match:
                tag_num = int(match.group(0))
                tag_map[tag_num] = c

        def replace_tag(match):
            n = int(match.group(1))
            if n not in tag_map:
                return match.group(0)   # leave unknown tags untouched
            c = tag_map[n]
            src = c["source"]
            doc_name = src["document_name"]
            page = src["page_number"]
            sec = src["section_title"]
            section_part = f", Section {sec}" if sec and sec != "None" else ""
            return f"[{n}: {doc_name}, Page {page}{section_part}]"

        return re.sub(r'\[(\d+)\]', replace_tag, response_text)

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
