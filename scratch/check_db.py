import sqlite3
import json

conn = sqlite3.connect("rbi_rag.db")
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute("SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT 1")
row = cursor.fetchone()
if row:
    print("TIMESTAMP:", row["timestamp"])
    print("QUERY:", row["query_text"])
    print("RESPONSE JSON:")
    print(row["response_text"])
else:
    print("No logs found.")
conn.close()
