import os
import sys

# Optional EasyOCR import with dynamic check
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

def perform_ocr_on_pdf(pdf_path):
    """
    Rasters the PDF pages and runs OCR using EasyOCR.
    Returns a dictionary mapping page numbers (1-indexed) to their extracted text.
    """
    if not EASYOCR_AVAILABLE:
        print("WARNING: easyocr is not installed. Scanned PDFs will not have text extracted.")
        print("To enable OCR, please install easyocr: pip install easyocr torch torchvision")
        return {}

    import fitz  # PyMuPDF
    print(f"Running OCR on {os.path.basename(pdf_path)} using EasyOCR...")
    
    # Initialize easyocr Reader for English. This downloads weights if not already present.
    # We disable GPU if CUDA is not available or if it causes errors
    try:
        reader = easyocr.Reader(['en'], gpu=False) # Safe default, change to gpu=True if CUDA is present
    except Exception as e:
        print(f"Failed to initialize EasyOCR reader: {e}")
        return {}

    doc = fitz.open(pdf_path)
    ocr_results = {}

    for page_index in range(len(doc)):
        page_num = page_index + 1
        print(f"  Processing page {page_num}/{len(doc)}...")
        try:
            page = doc[page_index]
            # Render page to a high-resolution image (150 DPI is a good balance of speed/quality)
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            
            # Run OCR on the rendered image
            results = reader.readtext(img_bytes, detail=0)
            page_text = "\n".join(results)
            ocr_results[page_num] = page_text
        except Exception as e:
            print(f"  Error performing OCR on page {page_num}: {e}")
            ocr_results[page_num] = ""

    print(f"Completed OCR on {os.path.basename(pdf_path)}")
    return ocr_results

if __name__ == "__main__":
    # Test OCR fallback script
    if len(sys.argv) > 1:
        test_pdf = sys.argv[1]
        text_map = perform_ocr_on_pdf(test_pdf)
        for p, t in text_map.items():
            print(f"--- Page {p} (length: {len(t)}) ---")
            print(t[:200])
    else:
        print("Usage: python -m ingestion.ocr_fallback <pdf_path>")
