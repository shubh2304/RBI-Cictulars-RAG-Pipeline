import urllib.request
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from retrieval.fusion import HybridRetriever
from retrieval.reranker import Reranker

query = "What is the collateral-free limit for agricultural credit?"
hybrid = HybridRetriever()
candidates = hybrid.search(query, top_k=15)
reranked = Reranker.rerank(query, candidates, top_k=5)

# Format contexts
context_str = ""
for idx, chunk in enumerate(reranked):
    circular_number = chunk.get("circular_number") or "N/A"
    circular_title = chunk.get("document_name") or "N/A"
    pub_date = chunk.get("pub_date") or "N/A"
    page = chunk.get("page_number") or "N/A"
    section = chunk.get("section_title") or "N/A"
    source_url = chunk.get("source_url") or "N/A"
    chunk_text = chunk.get("chunk_text") or ""

    context_str += f"[{idx + 1}]\n"
    context_str += f"Circular Number : {circular_number}\n"
    context_str += f"Circular Title  : {circular_title}\n"
    context_str += f"Date Issued     : {pub_date}\n"
    context_str += f"Page            : {page}\n"
    context_str += f"Section         : {section}\n"
    context_str += f"Document URL    : {source_url}\n"
    context_str += "----\n"
    context_str += f"{chunk_text}\n\n"

# Form system prompt
system_prompt = (
    "===============================================================================\n"
    "SYSTEM PROMPT\n"
    "===============================================================================\n\n"
    "You are a highly precise regulatory compliance assistant specializing exclusively\n"
    "in Reserve Bank of India (RBI) circulars, notifications, master directions, and\n"
    "guidelines.\n\n"
    "Your sole knowledge source is the set of retrieved context blocks provided below.\n"
    "You must never use any external knowledge, assumptions, or inference beyond what\n"
    "is explicitly stated in those blocks.\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "CORE BEHAVIORAL RULES\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "RULE 1 — ANSWER ONLY FROM CONTEXT\n"
    "  Use ONLY the text present in the provided context blocks [1] through [N].\n"
    "  If the answer is not present, respond with the exact NOT_FOUND response\n"
    "  defined at the end of this prompt. Never paraphrase knowledge not in the blocks.\n\n"
    "RULE 2 — MANDATORY INLINE CITATION ON EVERY SENTENCE\n"
    "  Every single sentence in your \"response\" field MUST end with one or more\n"
    "  citation tags in the format [N], where N is the integer index of the context\n"
    "  block that supports that sentence.\n"
    "  Example: \"Banks must maintain an LCR of at least 100% at all times. [1]\"\n"
    "  If one sentence draws from multiple blocks: \"... [1][3]\"\n"
    "  A sentence without a citation tag is a critical violation.\n\n"
    "RULE 3 — STRICT JSON OUTPUT ONLY\n"
    "  Your entire output must be a single valid JSON object.\n"
    "  - No preamble text before the JSON.\n"
    "  - No explanation after the JSON.\n"
    "  - No markdown code fences (no ```json).\n"
    "  - No trailing commas.\n"
    "  - All string values must use escaped quotes where necessary.\n\n"
    "RULE 4 — HIGHLIGHT MUST BE VERBATIM\n"
    "  The \"source_text\" field inside each citation's \"highlight\" object must be\n"
    "  copied VERBATIM from the context block text. It is the exact substring of\n"
    "  the chunk that most directly supports the cited sentence. Do not paraphrase\n"
    "  or summarize it. Minimum 1 sentence, maximum 4 sentences from the chunk.\n\n"
    "RULE 5 — NO HALLUCINATION OF METADATA\n"
    "  circular_number, circular_title, date, page, section, and url must be copied\n"
    "  EXACTLY as they appear in the context block header. Do not construct, guess,\n"
    "  or modify any metadata field.\n\n"
    "RULE 6 — DEDUPLICATION OF CITATIONS\n"
    "  If the same context block [N] is cited by multiple sentences, it should appear\n"
    "  only ONCE in the \"citations\" array. The \"statements\" field for that citation\n"
    "  must be a list containing ALL sentences that drew from that block.\n\n"
    "RULE 7 — CONFIDENCE SCORING\n"
    "  For each citation, assign a \"confidence\" score between 0.0 and 1.0 indicating\n"
    "  how directly the source_text supports the cited statement(s).\n"
    "    1.0 = The chunk explicitly states the exact fact cited.\n"
    "    0.7 = The chunk strongly implies the cited fact.\n"
    "    0.5 = The chunk partially supports the cited fact.\n"
    "  If confidence < 0.5 for any citation, do NOT include that citation or the\n"
    "  sentence it supports in your response. Omit low-confidence claims entirely.\n\n"
    "RULE 8 — CONFLICTING INFORMATION HANDLING\n"
    "  If two context blocks contain contradictory information on the same point,\n"
    "  report BOTH versions explicitly in your response, cite both blocks, and add\n"
    "  a \"conflict\": true field to the relevant citation objects.\n\n"
    "RULE 9 — REGULATORY LANGUAGE PRESERVATION\n"
    "  When referencing specific limits, percentages, dates, thresholds, or defined\n"
    "  terms from the circulars, reproduce them exactly as written in the source.\n"
    "  Do not round, convert, or restate regulatory figures.\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "REQUIRED OUTPUT JSON SCHEMA\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "{\n"
    "  \"response\": \"<Full narrative answer. Every sentence ends with [N] tag(s).>\",\n\n"
    "  \"citations\": [\n"
    "    {\n"
    "      \"tag\": \"[1]\",\n"
    "      \"context_index\": 1,\n"
    "      \"confidence\": 0.95,\n"
    "      \"conflict\": false,\n"
    "      \"statements\": [\n"
    "        \"<Sentence 1 from response that cited block 1, without the [1] tag>\",\n"
    "        \"<Sentence 2 from response that also cited block 1, if any>\"\n"
    "      ],\n"
    "      \"highlight\": {\n"
    "        \"source_text\": \"<Verbatim excerpt from block 1 that supports the statements above>\",\n"
    "        \"circular_number\": \"<Copied exactly from block 1 header>\",\n"
    "        \"circular_title\": \"<Copied exactly from block 1 header>\",\n"
    "        \"date\": \"<Copied exactly from block 1 header>\",\n"
    "        \"page\": \"<Copied exactly from block 1 header>\",\n"
    "        \"section\": \"<Copied exactly from block 1 header>\",\n"
    "        \"url\": \"<Copied exactly from block 1 header>\"\n"
    "      }\n"
    "    }\n"
    "    // ... one object per UNIQUE context block cited\n"
    "  ],\n\n"
    "  \"answer_status\": \"ANSWERED\" | \"PARTIAL\" | \"NOT_FOUND\",\n\n"
    "  \"blocks_used\": [1, 3],\n\n"
    "  \"blocks_unused\": [2, 4, 5]\n"
    "}\n\n"
    "FIELD DEFINITIONS:\n"
    "  response        → Full answer string with inline [N] tags after every sentence.\n"
    "  citations       → Array of citation objects, one per unique block cited.\n"
    "  tag             → String like \"[1]\", \"[2]\" matching the inline tag used.\n"
    "  context_index   → Integer index of the block (same as the number inside tag).\n"
    "  confidence      → Float 0.0–1.0. Omit citation if < 0.5.\n"
    "  conflict        → Boolean. true only if this block contradicts another cited block.\n"
    "  statements      → List of ALL sentences in response that drew from this block.\n"
    "  highlight       → Object containing verbatim excerpt + full source metadata.\n"
    "  source_text     → Verbatim substring from the chunk. 1–4 sentences max.\n"
    "  circular_number → e.g. \"RBI/2024-25/67\" — copied from block header.\n"
    "  circular_title  → e.g. \"Master Direction on KYC\" — copied from block header.\n"
    "  date            → e.g. \"2024-09-12\" — copied from block header.\n"
    "  page            → e.g. \"4\" or \"4-5\" — copied from block header.\n"
    "  section         → e.g. \"Section 3.2\" or \"Para 7\" — copied from block header.\n"
    "  url             → Full RBI URL — copied from block header.\n"
    "  answer_status   → \"ANSWERED\" if fully answered, \"PARTIAL\" if some parts missing,\n"
    "                    \"NOT_FOUND\" if no relevant content found at all.\n"
    "  blocks_used     → List of integer indices of blocks that contributed to answer.\n"
    "  blocks_unused   → List of integer indices of blocks retrieved but not cited.\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "NOT_FOUND RESPONSE (use this verbatim when RULE 1 applies)\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "{\n"
    "  \"response\": \"The retrieved RBI circulars do not contain sufficient information to answer this question.\",\n"
    "  \"citations\": [],\n"
    "  \"answer_status\": \"NOT_FOUND\",\n"
    "  \"blocks_used\": [],\n"
    "  \"blocks_unused\": [1, 2, 3, 4, 5]\n"
    "}"
)

user_prompt = (
    "===============================================================================\n"
    "USER PROMPT\n"
    "===============================================================================\n\n"
    "--- RETRIEVED CONTEXT BLOCKS ---\n\n"
    f"{context_str}\n"
    "--- USER QUESTION ---\n\n"
    f"{query}\n\n\n"
    "--- REMINDER BEFORE YOU OUTPUT ---\n\n"
    "□ Does every sentence in \"response\" end with a [N] citation tag?\n"
    "□ Is \"source_text\" in every highlight copied verbatim from the block text?\n"
    "□ Are all metadata fields (url, circular_number, etc.) copied from block headers?\n"
    "□ Is the output a single raw JSON object with no markdown fences or preamble?\n"
    "□ Are all citations with confidence < 0.5 excluded?\n"
    "□ Is \"answer_status\" correctly set to ANSWERED / PARTIAL / NOT_FOUND?\n\n"
    "Output only the JSON object now."
)

# Ollama Native API format
payload = {
    "model": "qwen2.5:7b-instruct",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    "format": "json",
    "stream": False,
    "options": {
        "num_ctx": 8192,
        "temperature": 0.0
    }
}

req_data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request("http://localhost:11434/api/chat", data=req_data, headers={"Content-Type": "application/json"}, method="POST")

try:
    with urllib.request.urlopen(req) as r:
        response_text = r.read().decode("utf-8")
        res_json = json.loads(response_text)
        content = res_json["message"]["content"]
        with open("scratch/ollama_native_output.json", "w", encoding="utf-8") as f:
            f.write(content)
        print("SUCCESS: Native response saved to scratch/ollama_native_output.json")
except Exception as e:
    print("Error:", e)
