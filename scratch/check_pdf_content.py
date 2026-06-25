import fitz

doc = fitz.open("circulars/msme_ammendments.pdf")
print(f"Total pages: {len(doc)}")
for i, page in enumerate(doc):
    print(f"\n--- PAGE {i+1} ---")
    text = page.get_text()
    print(text.encode('ascii', errors='replace').decode('ascii')[:600])
doc.close()
