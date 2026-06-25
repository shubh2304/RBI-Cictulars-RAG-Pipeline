import sqlite3

conn = sqlite3.connect("rbi_rag.db")
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

def safe_print(text):
    print(text.encode('ascii', errors='replace').decode('ascii'))

cursor.execute("""
    SELECT filename, document_name, circular_number, ref_number 
    FROM documents 
    WHERE filename IN ('10MC02072019.pdf', '128MD66C4DDCB167C4DC9A5BD913570CB3D47 (1).pdf')
""")
for row in cursor.fetchall():
    safe_print(f"File: {row['filename']}")
    safe_print(f"  Name: {row['document_name']}")
    safe_print(f"  Circular No: {row['circular_number']}")
    safe_print(f"  Ref No: {row['ref_number']}")
    print()

conn.close()
