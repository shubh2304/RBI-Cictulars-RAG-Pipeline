import os
import json
import urllib.request
import urllib.error
import re

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "mock")

def clean_json_response(text):
    """Strips any leading/trailing chat explanation text outside the first { and last }."""
    text_strip = text.strip()
    # Remove markdown code block fences if present
    text_strip = re.sub(r'^```(?:json)?', '', text_strip, flags=re.IGNORECASE)
    text_strip = re.sub(r'```$', '', text_strip).strip()
    
    start = text_strip.find('{')
    end = text_strip.rfind('}')
    if start != -1 and end != -1:
        return text_strip[start:end+1]
    return text_strip

class LocalTransformersLLM:
    """Runs a lightweight Qwen 0.5B model locally in Python using Hugging Face transformers."""
    _model = None
    _tokenizer = None

    @classmethod
    def get_model_and_tokenizer(cls):
        if cls._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            model_id = "Qwen/Qwen2.5-0.5B-Instruct"
            print(f"\n[Local Fallback] Loading native Python LLM: {model_id}...")
            print("[Local Fallback] This downloads ~950MB of model weights on its first run and runs entirely on CPU.")
            
            cls._tokenizer = AutoTokenizer.from_pretrained(model_id)
            cls._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32  # CPU friendly
            )
            print("[Local Fallback] Native Python LLM loaded successfully.")
        return cls._model, cls._tokenizer

    @classmethod
    def generate(cls, system_prompt, user_prompt):
        import torch
        model, tokenizer = cls.get_model_and_tokenizer()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt")
        
        print("[Local Fallback] Running text generation on CPU...")
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.1,  # Keep it highly deterministic
                do_sample=False
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

class LLMClient:
    """Interfaces with a local OpenAI-compatible API to generate cited answers."""

    @classmethod
    def generate_answer(cls, query, retrieved_chunks):
        """
        Sends query and formatted contexts to the local LLM.
        Enforces a structured JSON output with citation tags mapping to retrieved chunks.
        """
        # Format the contexts
        context_str = ""
        for idx, chunk in enumerate(retrieved_chunks):
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

        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},  # Force JSON mode
            "temperature": 0.0
        }

        # Attempt REST API call to local server
        req_data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json"
        }
        
        req = urllib.request.Request(LLM_API_URL, data=req_data, headers=headers, method="POST")
        
        try:
            print(f"Connecting to local LLM server at {LLM_API_URL}...")
            with urllib.request.urlopen(req, timeout=45) as response:
                res_data = response.read().decode("utf-8")
                res_json = json.loads(res_data)
                content = res_json["choices"][0]["message"]["content"]
                cleaned_content = clean_json_response(content)
                return json.loads(cleaned_content)
        except (urllib.error.URLError, ConnectionRefusedError) as e:
            print(f"\nWARNING: Could not connect to local LLM server: {e}")
            print("[Local Fallback] Falling back to native in-memory Transformers execution...")
            try:
                raw_response = LocalTransformersLLM.generate(system_prompt, user_prompt)
                cleaned = clean_json_response(raw_response)
                return json.loads(cleaned)
            except Exception as inner_e:
                print(f"[Local Fallback] ERROR running native model: {inner_e}")
                return cls._get_mock_response(query, retrieved_chunks)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return {"response": f"Error generating answer: {e}", "citations": []}

    @staticmethod
    def _get_mock_response(query, chunks):
        """Generates a structured mock response for offline/testing fallback."""
        if not chunks:
            return {
                "response": "The retrieved RBI circulars do not contain sufficient information to answer this question.",
                "citations": [],
                "answer_status": "NOT_FOUND",
                "blocks_used": [],
                "blocks_unused": [1, 2, 3, 4, 5]
            }
            
        chunk = chunks[0]
        circular_number = chunk.get("circular_number") or "N/A"
        circular_title = chunk.get("document_name") or "N/A"
        pub_date = chunk.get("pub_date") or "N/A"
        page = chunk.get("page_number") or "N/A"
        section = chunk.get("section_title") or "N/A"
        source_url = chunk.get("source_url") or "N/A"
        chunk_text = chunk.get("chunk_text") or ""
        
        statement = f"Based on the guidelines, the regulation states that: {chunk_text[:150]}"
        
        return {
            "response": f"{statement} [1]",
            "citations": [
                {
                    "tag": "[1]",
                    "context_index": 1,
                    "confidence": 1.0,
                    "conflict": false,
                    "statements": [
                        statement
                    ],
                    "highlight": {
                        "source_text": chunk_text[:200],
                        "circular_number": circular_number,
                        "circular_title": circular_title,
                        "date": pub_date,
                        "page": str(page),
                        "section": section,
                        "url": source_url
                    }
                }
            ],
            "answer_status": "ANSWERED",
            "blocks_used": [1],
            "blocks_unused": [i + 2 for i in range(len(chunks) - 1)]
        }

if __name__ == "__main__":
    # Quick client test
    mock_chunks = [{
        "chunk_id": "test-id",
        "document_name": "Master Circular - Kisan Credit Card (KCC) Scheme",
        "page_number": 4,
        "section_title": "5.1",
        "ref_number": "RBI/2017-18/4",
        "chunk_text": "The Kisan Credit Card (KCC) scheme aims at providing adequate and timely credit support from the banking system under a single window."
    }]
    ans = LLMClient.generate_answer("What is the aim of KCC?", mock_chunks)
    print("\n--- Generated Answer JSON ---")
    print(json.dumps(ans, indent=2))
