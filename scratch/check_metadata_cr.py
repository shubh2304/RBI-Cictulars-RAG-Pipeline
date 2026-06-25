import sqlite3

conn = sqlite3.connect("rbi_rag.db")
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("Checking documents table for raw carriage returns/newlines...")
cursor.execute("SELECT document_id, filename, document_name, ref_number, circular_number FROM documents")
for row in cursor.fetchall():
    for field in ["document_name", "ref_number", "circular_number"]:
        val = row[field]
        if val:
            if "\r" in val or "\n" in val:
                print(f"CR/LF found in File: {row['filename']} | Field: {field} | Value: {repr(val)}")

print("\nChecking chunks table section titles...")
cursor.execute("SELECT chunk_id, section_title FROM chunks")
for row in cursor.fetchall():
    val = row["section_title"]
    if val:
        if "\r" in val or "\n" in val:
            print(f"CR/LF found in Chunk: {row['chunk_id']} | Value: {repr(val)}")

conn.close()
print("Check complete.")
