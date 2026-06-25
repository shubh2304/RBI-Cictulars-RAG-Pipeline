import urllib.request
import json
try:
    with urllib.request.urlopen("http://localhost:11434/api/tags") as r:
        print(r.read().decode())
except Exception as e:
    print("Error:", e)
