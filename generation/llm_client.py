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

    @staticmethod
    def format_context_blocks(reranked_chunks: list[dict]) -> str:
        """
        Formats the top reranked chunks into numbered context blocks for the LLM prompt.
        Safely gets keys using dictionary lookups to support database naming conventions.
        """
        blocks = []
        for i, chunk in enumerate(reranked_chunks, start=1):
            doc_name = chunk.get("document_name") or "N/A"
            page_val = chunk.get("page") or chunk.get("page_number") or "N/A"
            section_line = chunk.get("section") or chunk.get("section_title") or "None"
            content_val = chunk.get("text") or chunk.get("chunk_text") or ""
            
            block = (
                f"--- CONTEXT BLOCK [{i}] ---\n"
                f"Document: {doc_name}\n"
                f"Page: {page_val}\n"
                f"Section: {section_line}\n"
                f"Content: {content_val.strip()}\n"
                f"--------------------------"
            )
            blocks.append(block)
        return "\n\n".join(blocks)

    @classmethod
    def build_user_message(cls, user_query: str, reranked_chunks: list[dict]) -> str:
        formatted = cls.format_context_blocks(reranked_chunks)
        return (
            f"CONTEXT BLOCKS:\n{formatted}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"COMPLIANCE QUERY:\n{user_query}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Respond ONLY with a valid JSON object matching the schema in your instructions.\n"
            "Do not write anything before or after the JSON."
        )

    @classmethod
    def generate_answer(cls, query, retrieved_chunks):
        """
        Sends query and formatted contexts to the local LLM.
        Enforces a structured JSON output with citation tags mapping to retrieved chunks.
        """
        system_prompt = """You are a precise, citation-strict RBI (Reserve Bank of India) regulatory compliance assistant.

Your role is to answer compliance queries using ONLY the numbered context blocks provided by the user.
You have zero external knowledge. You do not recall any RBI circulars, guidelines, or regulations
from your training data. Every claim in your answer must trace back to a provided context block.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CITATION RULES (STRICT — NO EXCEPTIONS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

R1. Every sentence in the "response" field MUST end with at least one citation tag [N] (where N is the actual context block number, e.g., [1] or [2]).
R2. If a sentence draws from multiple context blocks, cite all of them: [1][3].
R3. If two context blocks say the same thing, cite the one that is more specific.
R4. NEVER write a sentence without a citation tag. Not even transitional sentences. Always replace N with the actual block number.
R5. If you cannot find support for a claim in any context block, DO NOT make that claim.
R6. Numbers, percentages, dates, and monetary limits are high-risk — only state them
    if they appear verbatim in a context block, and always cite the exact block.
R7. Do not rephrase regulatory limits in a way that changes their meaning.
    Use the exact figures as stated in the source.
R8. DO NOT use list numbers, bullet points, circular numbers, or section headings found inside the text of the context blocks (e.g., "7.", "Section 4", or "[7]") as citation tags. Citation tags MUST strictly correspond to the provided Context Block number N (e.g., [1], [2], etc.).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHEN THE QUERY CANNOT BE ANSWERED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If no context block supports the query, return EXACTLY this JSON and nothing else:
{
  "response": "The provided RBI circulars do not contain sufficient information to answer this query.",
  "citations": [],
  "answerable": false
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond ONLY with a valid JSON object. No preamble. No explanation. No markdown fences.
No text before or after the JSON. Start your response with { and end with }.

The JSON must follow this schema exactly:

{
  "response": "<Flowing prose answer. Every sentence ends with citation tags like [1] or [2]. No bullet points.>",
  "answerable": true,
  "citations": [
    {
      "tag": <integer matching the N used inline>,
      "statement": "<Copy the exact sentence from response that uses this tag>",
      "source_block_index": <integer — the context block number N>,
      "document_name": "<Exactly as shown in the context block's Document field>",
      "page": <integer from the context block's Page field>,
      "section": "<Exactly as shown in context block's Section field, or null if None>"
    }
  ]
}

IMPORTANT: Every citation tag used in "response" MUST have a corresponding entry in "citations".
The "citations" array length must equal the total number of unique citation tags used in "response"."""

        user_prompt = cls.build_user_message(query, retrieved_chunks)

        # Determine if we should use Ollama native API or OpenAI completions API
        is_ollama_native = False
        api_url = LLM_API_URL
        if "11434" in api_url and ("/v1/chat/completions" in api_url or api_url == "http://localhost:11434/v1/chat/completions"):
            api_url = api_url.replace("/v1/chat/completions", "/api/chat")
            is_ollama_native = True
        elif "/api/chat" in api_url:
            is_ollama_native = True

        if is_ollama_native:
            # Dynamically restrict tag and source_block_index values to match the number of context blocks sent
            num_chunks = max(1, len(retrieved_chunks))
            prod_json_schema = {
                "type": "object",
                "properties": {
                    "response": { 
                        "type": "string"
                    },
                    "answerable": { 
                        "type": "boolean"
                    },
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tag": { "type": "integer", "minimum": 1, "maximum": num_chunks },
                                "statement": { "type": "string" },
                                "source_block_index": { "type": "integer", "minimum": 1, "maximum": num_chunks },
                                "document_name": { "type": "string" },
                                "page": { "type": "integer" },
                                "section": { "type": ["string", "null"] }
                            },
                            "required": ["tag", "statement", "source_block_index", "document_name", "page", "section"]
                        }
                    }
                },
                "required": ["response", "answerable", "citations"]
            }
            
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "format": prod_json_schema,
                "stream": False,
                "options": {
                    "num_ctx": 8192,
                    "temperature": 0.0
                }
            }
        else:
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
        
        req = urllib.request.Request(api_url, data=req_data, headers=headers, method="POST")
        
        try:
            print(f"Connecting to local LLM server at {api_url}...")
            with urllib.request.urlopen(req, timeout=90) as response:
                res_data = response.read().decode("utf-8")
                res_json = json.loads(res_data)
                if is_ollama_native:
                    content = res_json["message"]["content"]
                else:
                    content = res_json["choices"][0]["message"]["content"]
                cleaned_content = clean_json_response(content)
                return json.loads(cleaned_content)
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError) as e:
            print(f"\nWARNING: Could not connect to or timed out from local LLM server: {e}")
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
            return {
                "response": f"Error generating answer: {e}",
                "citations": [],
                "answerable": False
            }

    @staticmethod
    def _get_mock_response(query, chunks):
        """Generates a structured mock response for offline/testing fallback."""
        if not chunks:
            return {
                "response": "The provided RBI circulars do not contain sufficient information to answer this query.",
                "citations": [],
                "answerable": False
            }
            
        chunk = chunks[0]
        circular_title = chunk.get("document_name") or "N/A"
        page = chunk.get("page") or chunk.get("page_number") or 1
        section = chunk.get("section") or chunk.get("section_title")
        if section == "None" or section == "N/A":
            section = None
        chunk_text = chunk.get("text") or chunk.get("chunk_text") or ""
        
        statement = f"Based on the guidelines, the regulation states that: {chunk_text[:150]}"
        
        try:
            page_int = int(page)
        except (ValueError, TypeError):
            page_int = 1

        return {
            "response": f"{statement} [1]",
            "answerable": True,
            "citations": [
                {
                    "tag": 1,
                    "statement": statement,
                    "source_block_index": 1,
                    "document_name": circular_title,
                    "page": page_int,
                    "section": section
                }
            ]
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
