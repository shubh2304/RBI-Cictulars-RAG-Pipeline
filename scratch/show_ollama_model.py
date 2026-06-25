import urllib.request
import json

req_data = json.dumps({"name": "qwen2.5:7b-instruct"}).encode("utf-8")
req = urllib.request.Request("http://localhost:11434/api/show", data=req_data, headers={"Content-Type": "application/json"}, method="POST")

try:
    with urllib.request.urlopen(req) as r:
        response_text = r.read().decode("utf-8")
        res_json = json.loads(response_text)
        print("SYSTEM:", res_json.get("system"))
        print("\nTEMPLATE:", res_json.get("template"))
        print("\nMODELFILE:")
        print(res_json.get("modelfile"))
except Exception as e:
    print("Error:", e)
