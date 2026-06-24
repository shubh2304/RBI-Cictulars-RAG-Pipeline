import os
import sys
import io
import numpy as np
from PIL import Image

# Optional PaddleOCR import with dynamic check
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

def perform_ocr_on_pdf(pdf_path):
    """
    Rasters the PDF pages and runs OCR using PaddleOCR.
    Returns a dictionary mapping page numbers (1-indexed) to their extracted text.
    """
    if not PADDLEOCR_AVAILABLE:
        print("WARNING: paddleocr is not installed. Scanned PDFs will not have text extracted.")
        print("To enable OCR, please install paddleocr: pip install paddlepaddle paddleocr")
        return {}

    import fitz  # PyMuPDF
    print(f"Running OCR on {os.path.basename(pdf_path)} using PaddleOCR...")
    
    # Initialize PaddleOCR reader for English.
    try:
        # use_angle_cls detects text direction/rotation, show_log keeps logs clean
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
    except Exception as e:
        print(f"Failed to initialize PaddleOCR reader: {e}")
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
            
            # Convert bytes to PIL Image, then to a numpy array for PaddleOCR
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_array = np.array(image)
            
            # Run OCR on the rendered image array
            result = ocr.ocr(img_array, cls=True)
            
            page_texts = []
            if result:
                for line_group in result:
                    if line_group:
                        for line in line_group:
                            if line and len(line) > 1 and line[1]:
                                text = line[1][0]
                                page_texts.append(text)
                                
            page_text = "\n".join(page_texts)
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
