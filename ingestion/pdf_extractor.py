import os
import re
import fitz  # PyMuPDF
import pdfplumber
from ingestion.ocr_fallback import perform_ocr_on_pdf

# Regex patterns for metadata extraction
REF_PATTERN = re.compile(r'\b(RBI/(?:[A-Za-z0-9]+/)?\d{4}-\d{2,4}/\d+)\b')

# Matches common circular/direction codes like FIDD.MSME & NFS.BC.No.12/06.02.31/2025-26
CIRC_PATTERN = re.compile(
    r'\b((?:FIDD|RPCD|DOR|DBOD|NFS|CO|FSD|STR|CAP|REC|LBS)[A-Z0-9\s&\.\-/]+(?:BC|REC|Plan|No)[\d\s\.\-/]+)\b',
    re.IGNORECASE
)

DATE_PATTERN = re.compile(
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b',
    re.IGNORECASE
)

DATE_DOT_PATTERN = re.compile(r'\b\d{1,2}\.\d{1,2}\.\d{4}\b')

def extract_pdf_metadata(first_page_text, filename):
    """
    Parses metadata such as ref number, circular number, date, and document type
    from the first page text of an RBI document.
    """
    meta = {
        "document_name": filename.replace(".pdf", "").replace("_", " "),
        "document_type": "Circular",
        "ref_number": None,
        "circular_number": None,
        "pub_date": None,
        "source_url": None
    }
    
    # 1. Extract Ref Number
    ref_match = REF_PATTERN.search(first_page_text)
    if ref_match:
        meta["ref_number"] = ref_match.group(1).strip()
        
    # 2. Extract Circular Number
    circ_match = CIRC_PATTERN.search(first_page_text)
    if circ_match:
        meta["circular_number"] = circ_match.group(1).strip().replace("\n", " ")
        
    # 3. Extract Date
    date_match = DATE_PATTERN.search(first_page_text)
    if date_match:
        meta["pub_date"] = date_match.group(0).strip()
    else:
        date_dot_match = DATE_DOT_PATTERN.search(first_page_text)
        if date_dot_match:
            meta["pub_date"] = date_dot_match.group(0).strip()

    # 4. Classify Document Type
    first_lines = "\n".join([line.strip() for line in first_page_text.split("\n")[:40] if line.strip()])
    first_lines_lower = first_lines.lower()
    
    if "master direction" in first_lines_lower or "master directions" in first_lines_lower:
        meta["document_type"] = "Master Direction"
    elif "master circular" in first_lines_lower or "master circulars" in first_lines_lower:
        meta["document_type"] = "Master Circular"
    elif "frequently asked questions" in first_lines_lower or "faq" in first_lines_lower or "questions & answers" in first_lines_lower:
        meta["document_type"] = "FAQs"
    elif "notification" in first_lines_lower or "notifications" in first_lines_lower:
        meta["document_type"] = "Notification"
    elif "guidelines" in first_lines_lower:
        meta["document_type"] = "Guidelines"
    elif "press release" in first_lines_lower:
        meta["document_type"] = "Press Release"
    else:
        meta["document_type"] = "Circular"

    # 5. Extract Title / Subject using Salutation-based Heuristics
    lines = [l.strip() for l in first_page_text.split("\n")]
    cleaned_lines = [l for l in lines if l]
    
    salutation_idx = -1
    salutation_pattern = re.compile(
        r'^(?:madam|sir|madam\s*/\s*sir|sir\s*/\s*madam|dear\s+sir|dear\s+madam|dear\s+sir\s*/\s*madam|madam\s*/\s*dear\s+sir)[\s,]*$',
        re.IGNORECASE
    )
    
    for idx, line in enumerate(cleaned_lines):
        if salutation_pattern.match(line):
            salutation_idx = idx
            break
            
    title_lines = []
    if salutation_idx != -1 and salutation_idx + 1 < len(cleaned_lines):
        # Inspect lines immediately following salutation
        body_starters = ["please refer", "the reserve bank", "keeping in view", "on a review", "in exercise", "refer to our", "we advise"]
        for j in range(salutation_idx + 1, min(salutation_idx + 4, len(cleaned_lines))):
            line = cleaned_lines[j]
            line_lower = line.lower()
            # Stop if we hit a body paragraph indicator, a section number, or a common body starter
            if (line.startswith("2.") or line.startswith("1.") or 
                any(line_lower.startswith(bs) for bs in body_starters) or
                len(line.split()) > 20):
                break
            title_lines.append(line)

    if title_lines:
        meta["document_name"] = " ".join(title_lines).strip()
    else:
        # Fallback 1: Look for "Subject:" line block
        title_candidates = []
        subject_started = False
        for line in cleaned_lines[:35]:
            if "subject" in line.lower() or "master direction -" in line.lower() or "master circular -" in line.lower():
                subject_started = True
                title_candidates.append(line)
                continue
            if subject_started:
                if "dear sir" in line.lower() or "madam" in line.lower() or line.startswith("2."):
                    break
                title_candidates.append(line)
        if title_candidates:
            subject_title = " ".join(title_candidates)
            subject_title = re.sub(r'^(?:subject|subj)\b[\s\-\:]*', '', subject_title, flags=re.IGNORECASE)
            meta["document_name"] = subject_title.strip()
        else:
            # Fallback 2: scan first 25 lines for keywords
            for line in cleaned_lines[:25]:
                if any(kw in line.lower() for kw in ["master direction -", "master circular -", "lending to", "credit flow", "review of"]):
                    meta["document_name"] = line.strip()
                    break

    # Clean name punctuation and double spaces
    meta["document_name"] = re.sub(r'\s+', ' ', meta["document_name"])
    # Strip any ending punctuation if it is a colon or hyphen
    meta["document_name"] = meta["document_name"].strip(" :-–")
    
    # Construct placeholder URL
    if meta["ref_number"]:
        safe_ref = meta["ref_number"].replace("/", "-")
        meta["source_url"] = f"https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id={safe_ref}"
        
    return meta

def extract_tables_from_page(pdf_path, page_num):
    """
    Extracts tables from a specific page of a PDF using pdfplumber and formats them as Markdown tables.
    """
    tables_md = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num - 1 < len(pdf.pages):
                page = pdf.pages[page_num - 1]
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) == 0:
                        continue
                    
                    cleaned_rows = []
                    for row in table:
                        cleaned_row = [str(val).replace("\n", " ").strip() if val is not None else "" for val in row]
                        cleaned_rows.append(cleaned_row)
                    
                    if not cleaned_rows or len(cleaned_rows[0]) == 0:
                        continue
                        
                    headers = cleaned_rows[0]
                    # Create markdown structure
                    markdown_table = "| " + " | ".join(headers) + " |\n"
                    markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                    for row in cleaned_rows[1:]:
                        markdown_table += "| " + " | ".join(row) + " |\n"
                    tables_md.append(markdown_table)
    except Exception as e:
        print(f"Error extracting tables on page {page_num}: {e}")
        
    return tables_md

def ingest_pdf(pdf_path):
    """
    Ingests a single PDF.
    Returns:
        dict: {
            "metadata": dict,
            "pages": {page_num: {"text": str, "tables": [str]}}
        }
    """
    filename = os.path.basename(pdf_path)
    print(f"Ingesting PDF: {filename}...")
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    pages_data = {}
    total_text_length = 0
    
    # 1. Attempt standard text extraction
    for page_index in range(total_pages):
        page_num = page_index + 1
        page = doc[page_index]
        text = page.get_text()
        total_text_length += len(text.strip())
        pages_data[page_num] = {
            "text": text,
            "tables": []
        }
        
    # 2. Check if scanned PDF (length is 0)
    is_scanned = (total_text_length == 0)
    
    if is_scanned:
        print(f"  {filename} appears to be a scanned document (0 text characters). Triggering OCR fallback...")
        ocr_text_map = perform_ocr_on_pdf(pdf_path)
        for page_num, text in ocr_text_map.items():
            pages_data[page_num] = {
                "text": text,
                "tables": []
            }
    else:
        # 3. Extract tables for readable PDFs using pdfplumber
        print(f"  Extracting tables from {filename}...")
        for page_num in range(1, total_pages + 1):
            tables = extract_tables_from_page(pdf_path, page_num)
            if tables:
                pages_data[page_num]["tables"] = tables
                
    # 4. Extract metadata using first page text
    first_page_text = pages_data[1]["text"] if total_pages > 0 else ""
    metadata = extract_pdf_metadata(first_page_text, filename)
    
    doc.close()
    return {
        "metadata": metadata,
        "pages": pages_data,
        "is_scanned": is_scanned
    }

if __name__ == "__main__":
    # Test execution
    test_file = r"C:\Users\shubh\OneDrive\Desktop\RBI RAG\circulars\04MCKCC03072017.pdf"
    res = ingest_pdf(test_file)
    print("\n--- Extracted Metadata ---")
    print(res["metadata"])
    print("\n--- Page 1 (first 300 chars) ---")
    print(res["pages"][1]["text"][:300])
    if res["pages"][1]["tables"]:
        print("\n--- Page 1 Table 1 ---")
        print(res["pages"][1]["tables"][0])
