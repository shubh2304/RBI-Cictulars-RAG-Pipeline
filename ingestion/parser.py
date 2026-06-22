import re
import uuid
import os

# Regex patterns for structural headers
CHAP_PATTERN = re.compile(r'^\s*(CHAPTER|PART)\s+([IVXLCDM\d]+)\b', re.IGNORECASE)
SEC_PATTERN = re.compile(r'^\s*(\d+(?:\.\d+){1,3})\b')
ANNEX_PATTERN = re.compile(r'^\s*(ANNEXURE|ANNEX|APPENDIX|SCHEDULE)\b[\s\-–I\d\sA-Z]*', re.IGNORECASE)
FAQ_PATTERN = re.compile(r'^\s*(Q\s*\d+\.|Question\s*\d+\.)', re.IGNORECASE)
PAGE_MARKER_PATTERN = re.compile(r'^\[PAGE_NUM:(\d+)\]$')

def parse_document_to_chunks(ingested_data, document_id):
    """
    Parses ingested document data (text and tables) into a list of structured chunk dicts.
    """
    metadata = ingested_data["metadata"]
    pages = ingested_data["pages"]
    doc_type = metadata["document_type"]
    
    # 1. Stitch page text with page markers
    stitched_lines = []
    for page_num in sorted(pages.keys()):
        stitched_lines.append(f"[PAGE_NUM:{page_num}]")
        page_text = pages[page_num]["text"]
        stitched_lines.extend(page_text.split("\n"))

    chunks = []
    
    # State variables for structural tracking
    current_chapter = None
    current_section = None
    current_annex = None
    current_page = 1
    
    current_chunk_lines = []
    current_chunk_type = "text"
    
    def finalize_chunk(lines, chunk_type):
        if not lines:
            return
        text = "\n".join(lines).strip()
        # Remove any page markers from the final chunk text to keep it clean
        text_cleaned = re.sub(r'\[PAGE_NUM:\d+\]\n?', '', text).strip()
        if not text_cleaned:
            return
            
        # Fallback split if chunk is too long (> 600 words)
        word_count = len(text_cleaned.split())
        if word_count > 600 and chunk_type == "text":
            paragraphs = text_cleaned.split("\n\n")
            sub_chunk_lines = []
            sub_word_count = 0
            for para in paragraphs:
                para_strip = para.strip()
                if not para_strip:
                    continue
                para_words = len(para_strip.split())
                if sub_word_count + para_words > 500:
                    # yield current sub-chunk
                    yield_chunk(sub_chunk_lines, chunk_type)
                    sub_chunk_lines = [para_strip]
                    sub_word_count = para_words
                else:
                    sub_chunk_lines.append(para_strip)
                    sub_word_count += para_words
            if sub_chunk_lines:
                yield_chunk(sub_chunk_lines, chunk_type)
        else:
            yield_chunk(lines, chunk_type)

    def yield_chunk(lines, chunk_type):
        text = "\n".join(lines).strip()
        text_cleaned = re.sub(r'\[PAGE_NUM:\d+\]\n?', '', text).strip()
        if not text_cleaned:
            return
            
        chunk_id = str(uuid.uuid4())
        chunks.append({
            "chunk_id": chunk_id,
            "document_id": document_id,
            "parent_chunk_id": None, # Will be set in parent-child linking step
            "chunk_type": chunk_type,
            "page_number": current_page,
            "chapter_title": current_chapter,
            "section_title": current_section if not current_annex else current_annex,
            "subsection_title": None,
            "chunk_text": text_cleaned,
            "vector_index": None
        })

    # Line-by-line state machine parser
    for line in stitched_lines:
        line_strip = line.strip()
        if not line_strip:
            continue
            
        # Check page marker
        page_match = PAGE_MARKER_PATTERN.match(line_strip)
        if page_match:
            current_page = int(page_match.group(1))
            current_chunk_lines.append(line_strip) # Keep to track page within text
            continue
            
        # Check for structural changes
        chap_match = CHAP_PATTERN.match(line_strip)
        sec_match = SEC_PATTERN.match(line_strip)
        annex_match = ANNEX_PATTERN.match(line_strip)
        faq_match = FAQ_PATTERN.match(line_strip)
        
        # If we hit a new structural header, finalize the previous chunk
        if doc_type == "FAQs" and faq_match:
            finalize_chunk(current_chunk_lines, current_chunk_type)
            current_chunk_lines = [line_strip]
            current_chunk_type = "faq_pair"
            current_section = faq_match.group(1).strip()
            current_annex = None
        elif chap_match:
            finalize_chunk(current_chunk_lines, current_chunk_type)
            current_chunk_lines = [line_strip]
            current_chunk_type = "text"
            current_chapter = line_strip
            current_section = None
            current_annex = None
        elif annex_match:
            finalize_chunk(current_chunk_lines, current_chunk_type)
            current_chunk_lines = [line_strip]
            current_chunk_type = "text"
            current_annex = line_strip
            current_chapter = None
            current_section = None
        elif sec_match and not current_annex:
            finalize_chunk(current_chunk_lines, current_chunk_type)
            current_chunk_lines = [line_strip]
            current_chunk_type = "text"
            current_section = line_strip
        else:
            current_chunk_lines.append(line)

    # Finalize remaining text chunk
    finalize_chunk(current_chunk_lines, current_chunk_type)

    # 2. Add extracted tables as specialized chunks
    for page_num, page_data in pages.items():
        for table_md in page_data.get("tables", []):
            chunk_id = str(uuid.uuid4())
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "parent_chunk_id": None,
                "chunk_type": "table",
                "page_number": page_num,
                "chapter_title": None,
                "section_title": "Table Data",
                "subsection_title": None,
                "chunk_text": table_md.strip(),
                "vector_index": None
            })
            
    # 3. Create Parent-Child links
    # For child chunks (e.g. detailed sub-sections like 2.2.1), we link them to parent sections (e.g. 2.2)
    # We do a simple pass looking at section headers
    for chunk in chunks:
        sec = chunk["section_title"]
        if sec and "." in sec:
            parts = sec.split()[0].split(".")
            if len(parts) > 2: # e.g. 2.2.1
                parent_sec_num = ".".join(parts[:-1]) # e.g. 2.2
                # Look for a chunk in the same document with this parent section number
                for p_chunk in chunks:
                    if p_chunk["section_title"] and p_chunk["section_title"].startswith(parent_sec_num + " "):
                        chunk["parent_chunk_id"] = p_chunk["chunk_id"]
                        break

    return chunks

if __name__ == "__main__":
    # Test parser with ingested output
    from ingestion.pdf_extractor import ingest_pdf
    test_file = r"C:\Users\shubh\OneDrive\Desktop\RBI RAG\circulars\04MCKCC03072017.pdf"
    ingested = ingest_pdf(test_file)
    chunks = parse_document_to_chunks(ingested, "test-doc-id")
    print(f"\nTotal parsed chunks: {len(chunks)}")
    print("\n--- Sample Chunk 3 ---")
    print(f"Type: {chunks[2]['chunk_type']}")
    print(f"Page: {chunks[2]['page_number']}")
    print(f"Section: {chunks[2]['section_title']}")
    print(f"Text preview:\n{chunks[2]['chunk_text'][:200]}...")
